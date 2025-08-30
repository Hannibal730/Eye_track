#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaze_overlay_12d.py
- MediaPipe Face Mesh(iris 포함)로 12D 시선 피처 추출(uL,vL,uR,vR + 2차 확장)
- 캘리브레이션: 그리드(행x열, 지그재그 순회), 포인트별 체류시간 동안 샘플 수집
- 데이터 저장: data/gaze_samples_*.npz (X,Y,T,pt_index,screen,feature_names,meta)
- 모델: 경량 선형회귀(릿지), .pkl/.npz 로드
- 오버레이: 투명/클릭-스루 빨간 고리
- 프리뷰: 메쉬/홍채중심/축(u/v 벡터 포함) 시각화 토글 가능
- 스무딩: OneEuro + EMA(α) 2중 필터 (UI에서 OneEuro/EMA 파라미터 조정 가능)
- PCA 축 부호 안정화: 좌우(코쪽 anchor), 상하(+y anchor) 기준으로 축 방향 고정

예)
  python gaze_overlay_12d.py --grid 8x12 --per_point 2.0
"""

import os, sys, time, argparse, re, threading, json
import numpy as np
import cv2
import mediapipe as mp
import pickle
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets

# 권장: X11/xcb
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# -------------------- 경로 --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- MediaPipe --------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 홍채(iris) 인덱스 (MediaPipe 표준)
LEFT_IRIS_IDXS  = [474, 475, 476, 477]
RIGHT_IRIS_IDXS = [469, 470, 471, 472]

def _unique_idxs(connections):
    s = set()
    for a, b in connections: s.add(a); s.add(b)
    return sorted(list(s))

# 눈 윤곽(컨투어) 전체 인덱스
LEFT_EYE_IDXS  = _unique_idxs(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDXS = _unique_idxs(mp_face_mesh.FACEMESH_RIGHT_EYE)

# 눈꼬리/코쪽 코너(부호 정렬용 anchor)
# 좌안: 바깥 33, 코쪽 133 / 우안: 코쪽 263, 바깥 362
L_OUTER, L_INNER = 33, 133
R_INNER, R_OUTER = 263, 362  # 순서 주의(코쪽, 바깥)

FEATURE_NAMES = [
    "uL","vL","uR","vR",
    "uL2","vL2","uR2","vR2",
    "uL_vL","uR_vR","uL_uR","vL_vR"
]

# -------------------- 수학/좌표 유틸 --------------------
def _pca_axes(pts: np.ndarray):
    """pts: (N,2) → c, ax1, ax2, w, h  (ax1/ax2: 단위벡터, w/h: 축 폭 추정)"""
    c = pts.mean(axis=0)
    X = pts - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ax1, ax2 = Vt[0], Vt[1]  # 단위벡터
    # RMS 반지름*2 = 폭(안정적 스케일)
    w = 2 * (np.sqrt(np.mean((X @ ax1)**2)) + 1e-6)
    h = 2 * (np.sqrt(np.mean((X @ ax2)**2)) + 1e-6)
    return c, ax1, ax2, w, h

def _lm_xy(landmarks, idx, W, H):
    p = landmarks[idx]
    return np.array([p.x * W, p.y * H], dtype=np.float32)

def _iris_center(landmarks, idxs, W, H):
    pts = np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)

def _align_axes(ax1, ax2, landmarks, side, W, H):
    """
    PCA 축 부호 정렬:
      - 수평축(ax1): 코쪽(anchor) 방향과 같은 부호가 되도록
      - 수직축(ax2): 화면 아래쪽(+y)과 같은 부호가 되도록
    """
    if side == "L":
        anchor = _lm_xy(landmarks, L_INNER, W, H) - _lm_xy(landmarks, L_OUTER, W, H)  # outer→inner(코쪽)
    else:  # "R"
        anchor = _lm_xy(landmarks, R_INNER, W, H) - _lm_xy(landmarks, R_OUTER, W, H)  # outer→inner(코쪽)
    if np.linalg.norm(anchor) > 1e-6:
        anchor = anchor / np.linalg.norm(anchor)
    # 수평축 정렬(코쪽과 같은 방향)
    if float(np.dot(ax1, anchor)) < 0.0:
        ax1 = -ax1
    # 수직축 정렬(+y로 향하도록)
    down = np.array([0.0, 1.0], dtype=np.float32)
    if float(np.dot(ax2, down)) < 0.0:
        ax2 = -ax2
    return ax1, ax2

def _eye_uv_ex(landmarks, eye_idxs, iris_idxs, side, W, H):
    """
    반환:
      (u,v), ic(iris center), c(eye center), ax1, ax2, su, sv, du, dv
      - u = (Δ·û)/sᵤ, v = (Δ·v̂)/sᵥ  (정규화)
      - du = Δ·û, dv = Δ·v̂ (픽셀 단위, 정규화 전)
    """
    eye_pts = np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in eye_idxs], dtype=np.float32)
    c, ax1, ax2, w, h = _pca_axes(eye_pts)
    # 축 부호 안정화
    ax1, ax2 = _align_axes(ax1, ax2, landmarks, side, W, H)

    ic = _iris_center(landmarks, iris_idxs, W, H)
    delta = ic - c
    su = w / 2.0; sv = h / 2.0
    du = float(np.dot(delta, ax1))   # Δ·û  (px)
    dv = float(np.dot(delta, ax2))   # Δ·v̂  (px)
    u = float(du / su)
    v = float(dv / sv)
    return (u, v), ic, c, ax1, ax2, su, sv, du, dv

def _feat_vector(uL, vL, uR, vR):
    return np.array([
        uL, vL, uR, vR,
        uL*uL, vL*vL, uR*uR, vR*vR,
        uL*vL, uR*vR, uL*uR, vL*vR
    ], dtype=np.float32)

# -------------------- 캘리브레이션 --------------------
def make_grid_points(rows:int, cols:int, margin:float=0.10, order:str="serpentine"):
    rows = max(1, rows); cols = max(1, cols)
    margin = max(0.0, min(0.45, margin))
    xs = np.linspace(margin, 1.0 - margin, cols)
    ys = np.linspace(margin, 1.0 - margin, rows)
    pts = []
    for r, y in enumerate(ys):
        row = [(x, y) for x in xs]
        if order == "serpentine" and (r % 2 == 1):
            row = row[::-1]
        pts.extend(row)
    return pts

FIVE_POINTS = [(0.5,0.5), (0.15,0.15), (0.85,0.15), (0.85,0.85), (0.15,0.85)]

class Ridge2D:
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.W = None  # (2,D)
        self.b = None  # (2,)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        N, D = X.shape
        Xb = np.hstack([X, np.ones((N,1), dtype=np.float32)])
        I = np.eye(D+1, dtype=np.float32)
        I[-1, -1] = 0.0  # bias L2 제외
        A = Xb.T @ Xb + self.alpha * I
        Wb = np.linalg.pinv(A) @ (Xb.T @ Y)
        self.W = Wb[:-1, :].T
        self.b = Wb[-1, :].T

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (self.W @ X.T).T + self.b

class Calibrator:
    def __init__(self, screen_w, screen_h, rows=0, cols=0, margin=0.10, per_point_sec=0.9):
        self.sw, self.sh = screen_w, screen_h
        self.rows, self.cols, self.margin = rows, cols, margin
        self.points_norm = make_grid_points(rows, cols, margin, "serpentine") if (rows and cols) else FIVE_POINTS[:]
        self.per_point_sec = per_point_sec
        self.reset()
        self.model = Ridge2D(alpha=10.0)

    def reset(self):
        self.idx = 0
        self.collecting = False
        self.samples_X, self.samples_Y = [], []
        self.samples_T, self.samples_IDX = [], []
        self.start_t = None

    def n_points(self): return len(self.points_norm)
    def current_target_px(self):
        nx, ny = self.points_norm[self.idx]
        return int(nx * self.sw), int(ny * self.sh)

    def begin(self):
        self.reset(); self.collecting = True; self.start_t = time.time()

    def feed(self, feat, t_now=None):
        if not self.collecting: return False
        tx, ty = self.current_target_px()
        self.samples_X.append(np.array(feat, dtype=np.float32))
        self.samples_Y.append(np.array([tx, ty], dtype=np.float32))
        self.samples_IDX.append(int(self.idx))
        self.samples_T.append(float(time.time() if t_now is None else t_now))
        if (time.time() - self.start_t) >= self.per_point_sec:
            self.idx += 1; self.start_t = time.time()
            if self.idx >= len(self.points_norm):
                self.model.fit(np.stack(self.samples_X), np.stack(self.samples_Y))
                self.collecting = False
                return True
        return False

    def has_model(self):  return (self.model.W is not None)

    def predict(self, feat):
        y = self.model.predict(np.array([feat], dtype=np.float32))[0]
        x = int(np.clip(y[0], 0, self.sw - 1))
        yv = int(np.clip(y[1], 0, self.sh - 1))
        return x, yv

    def save_model_pkl(self, path=None):
        if path is None:
            path = os.path.join(DATA_DIR, "calib_gaze.pkl")
        with open(path, "wb") as f:
            pickle.dump({"W": self.model.W, "b": self.model.b, "screen": (self.sw, self.sh)}, f)

    def save_dataset_npz(self, out_dir=DATA_DIR, meta_extra=None):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"gaze_samples_{ts}.npz")
        X = np.stack(self.samples_X).astype(np.float32)
        Y = np.stack(self.samples_Y).astype(np.float32)
        T   = np.array(self.samples_T,  dtype=np.float64)
        IDX = np.array(self.samples_IDX, dtype=np.int32)
        meta = {
            "rows": self.rows, "cols": self.cols,
            "margin": float(self.margin), "per_point_sec": float(self.per_point_sec),
            "n_points": int(len(self.points_norm)), "timestamp": ts
        }
        if meta_extra: meta.update(meta_extra)
        np.savez(path, X=X, Y=Y, T=T, pt_index=IDX,
                 feature_names=np.array(FEATURE_NAMES, dtype=object),
                 screen=np.array([self.sw, self.sh], dtype=np.int32),
                 meta=json.dumps(meta))
        return path

    def load_linear_model(self, path):
        if path.lower().endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            if "W" in d and "b" in d:
                self.model.W = d["W"].astype(np.float32)
                self.model.b = d["b"].astype(np.float32)
                return True
            raise ValueError("npz에 'W','b' 필요")
        elif path.lower().endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model.W = np.asarray(data["W"], dtype=np.float32)
            self.model.b = np.asarray(data["b"], dtype=np.float32)
            return True
        else:
            raise ValueError("지원 포맷: .npz, .pkl")

# -------------------- 스무딩 --------------------
class OneEuro:
    def __init__(self, freq=60.0, mincutoff=0.20, beta=0.003, dcutoff=1.0):
        self.freq=freq; self.mincutoff=mincutoff; self.beta=beta; self.dcutoff=dcutoff
        self.x_prev=None; self.dx_prev=None; self.t_prev=None
    def _alpha(self, cutoff):
        tau = 1.0 / (2*np.pi*cutoff)
        te  = 1.0 / self.freq
        return 1.0 / (1.0 + tau/te)
    def filter(self, x, t=None):
        if t is None: t = time.time()
        if self.t_prev is None:
            self.t_prev = t; self.x_prev = x.copy(); self.dx_prev = np.zeros_like(x)
            return x
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0 / dt
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        cutoff = self.mincutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev=x_hat; self.dx_prev=dx_hat; self.t_prev=t
        return x_hat

# -------------------- 공유 상태 --------------------
class SharedState:
    def __init__(self, sw:int, sh:int):
        self.lock = threading.Lock()
        self.screen_w = sw; self.screen_h = sh
        self.cross = None
        self.calib_target = None
        self.calibrating = False
        self.overlay_enabled = True
        self.fullscreen = False
        self.status = "Gaze Overlay"
        self.substatus = "Use Control Panel"

        # 시각화 옵션
        self.vis_mesh = True
        self.vis_center_box = True
        self.vis_iris = True
        self.vis_gaze_ring = True
        self.vis_calib_target = True
        self.vis_status_text = True

        self.vis_eye_axes = False                # Eye axes (fixed len)
        self.vis_eye_axes_scaled = False         # Eye axes (eye len)
        self.vis_uv_vectors = False              # u/v vectors
        self.vis_uv_vectors_bigger = False       # u/v vectors (bigger)
        self.uv_bigger_gain = 4.0                # u/v bigger gain

        # 스무딩 파라미터 (UI에서 조정)
        self.ema_alpha = 0.8
        self.oe_mincutoff = 0.20
        self.oe_beta = 0.003
        self.oe_dcutoff = 1.0

        self.cmd = {
            "start_calib": False, "load_model": False,
            "toggle_overlay": False, "toggle_fullscreen": False,
            "quit": False
        }
        self._load_path = None

    def set_cmd(self, name):
        with self.lock:
            if name in self.cmd: self.cmd[name] = True
    def pop_cmd(self, name):
        with self.lock:
            v = self.cmd.get(name, False); self.cmd[name] = False; return v
    def set_load_model(self, path:str):
        with self.lock: self._load_path = path; self.cmd["load_model"] = True
    def pop_load_path(self):
        with self.lock: p=self._load_path; self._load_path=None; return p

# -------------------- 오버레이(클릭-스루) --------------------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, app: QtWidgets.QApplication):
        super().__init__(None)
        self.shared = shared; self.app = app
        self.status="Gaze Overlay"; self.substatus="Use Control Panel"
        self.cross=None; self.calib_target=None; self._last_fullscreen=False

        flags = (QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint |
                 QtCore.Qt.Tool | QtCore.Qt.WindowDoesNotAcceptFocus)
        if hasattr(QtCore.Qt, "WindowTransparentForInput"):
            flags |= QtCore.Qt.WindowTransparentForInput
        if hasattr(QtCore.Qt, "X11BypassWindowManagerHint"):
            flags |= QtCore.Qt.X11BypassWindowManagerHint
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.setGeometry(self.app.primaryScreen().geometry())
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(16)
        self.show()

    def tick(self):
        with self.shared.lock:
            self.cross = self.shared.cross
            self.calib_target = self.shared.calib_target
            enabled = self.shared.overlay_enabled
            fs = self.shared.fullscreen
            self.status = self.shared.status
            self.substatus = self.shared.substatus
        if enabled:
            if not self.isVisible(): self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True); self.show()
        else:
            if self.isVisible(): self.hide()
        if fs != self._last_fullscreen:
            st = self.windowState()
            if fs and not (st & QtCore.Qt.WindowFullScreen):
                self.setWindowState(st | QtCore.Qt.WindowFullScreen)
            elif not fs and (st & QtCore.Qt.WindowFullScreen):
                self.setWindowState(st & ~QtCore.Qt.WindowFullScreen)
            self._last_fullscreen = fs
        if enabled: self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # 상태
        pen = QtGui.QPen(QtGui.QColor(200,200,200,230), 2); p.setPen(pen)
        p.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold)); p.drawText(30, 50, self.status)
        pen = QtGui.QPen(QtGui.QColor(160,160,160,200), 1); p.setPen(pen)
        p.setFont(QtGui.QFont("Arial", 12)); p.drawText(30, 80, self.substatus)
        # 타깃
        if self.calib_target is not None:
            tx, ty = self.calib_target
            pen = QtGui.QPen(QtGui.QColor(255,165,0,240), 4); p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush); p.drawEllipse(QtCore.QPointF(tx, ty), 16, 16)
        # 추정 고리
        if self.cross is not None:
            x, y = self.cross
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 230), 4); p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush); p.drawEllipse(QtCore.QPointF(x, y), 14, 14)

# -------------------- 컨트롤 패널 --------------------
class ControlPanel(QtWidgets.QWidget):
    def __init__(self, shared: SharedState):
        super().__init__()
        self.shared = shared
        self.setWindowTitle("Gaze Control")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setFixedWidth(460)

        v = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Ready"); v.addWidget(self.lbl)

        # 버튼
        grid = QtWidgets.QGridLayout(); v.addLayout(grid)
        b_calib = QtWidgets.QPushButton("Start Calibration")
        b_load  = QtWidgets.QPushButton("Load Model (.npz/.pkl)")
        b_ov    = QtWidgets.QPushButton("Toggle Overlay")
        b_fs    = QtWidgets.QPushButton("Fullscreen")
        b_quit  = QtWidgets.QPushButton("Quit")
        grid.addWidget(b_calib, 0,0); grid.addWidget(b_load, 0,1)
        grid.addWidget(b_ov,    1,0); grid.addWidget(b_fs,   1,1)
        grid.addWidget(b_quit,  2,0,1,2)
        b_calib.clicked.connect(lambda: self.shared.set_cmd("start_calib"))
        b_load.clicked.connect(self._choose_and_load_model)
        b_ov.clicked.connect(lambda: self.shared.set_cmd("toggle_overlay"))
        b_fs.clicked.connect(lambda: self.shared.set_cmd("toggle_fullscreen"))
        b_quit.clicked.connect(lambda: self.shared.set_cmd("quit"))

        # 시각화 옵션
        grp = QtWidgets.QGroupBox("Visualization"); v.addWidget(grp)
        gl = QtWidgets.QGridLayout(grp)
        self.cb_mesh        = QtWidgets.QCheckBox("Face Mesh");            self.cb_mesh.setChecked(True)
        self.cb_center_box  = QtWidgets.QCheckBox("Center Box");           self.cb_center_box.setChecked(True)
        self.cb_iris        = QtWidgets.QCheckBox("Iris centers");         self.cb_iris.setChecked(True)
        self.cb_axes        = QtWidgets.QCheckBox("Eye axes (fixed len)"); self.cb_axes.setChecked(False)
        self.cb_axes_s      = QtWidgets.QCheckBox("Eye axes (eye len)");   self.cb_axes_s.setChecked(False)
        self.cb_uvvec       = QtWidgets.QCheckBox("u/v vectors");          self.cb_uvvec.setChecked(False)
        self.cb_uvvec_big   = QtWidgets.QCheckBox("u/v vectors (bigger)"); self.cb_uvvec_big.setChecked(False)
        self.sb_uv_gain     = QtWidgets.QDoubleSpinBox(); self.sb_uv_gain.setRange(0.1, 20.0)
        self.sb_uv_gain.setSingleStep(0.1); self.sb_uv_gain.setDecimals(1); self.sb_uv_gain.setValue(4.0)
        gl.addWidget(self.cb_mesh,       0,0); gl.addWidget(self.cb_center_box, 0,1)
        gl.addWidget(self.cb_iris,       1,0)
        gl.addWidget(self.cb_axes,       2,0); gl.addWidget(self.cb_axes_s,     2,1)
        gl.addWidget(self.cb_uvvec,      3,0); gl.addWidget(self.cb_uvvec_big,  3,1)
        gl.addWidget(QtWidgets.QLabel("u/v bigger gain"), 4,0); gl.addWidget(self.sb_uv_gain, 4,1)

        # 스무딩(OneEuro + EMA)
        grp2 = QtWidgets.QGroupBox("Smoothing"); v.addWidget(grp2)
        g2 = QtWidgets.QGridLayout(grp2)
        # OneEuro
        g2.addWidget(QtWidgets.QLabel("OneEuro mincutoff"), 0,0)
        self.sb_minc = QtWidgets.QDoubleSpinBox(); self.sb_minc.setRange(0.01, 2.0); self.sb_minc.setDecimals(3)
        self.sb_minc.setSingleStep(0.01); self.sb_minc.setValue(0.20)
        g2.addWidget(self.sb_minc, 0,1)

        g2.addWidget(QtWidgets.QLabel("OneEuro beta"), 1,0)
        self.sb_beta = QtWidgets.QDoubleSpinBox(); self.sb_beta.setRange(0.000, 0.100); self.sb_beta.setDecimals(3)
        self.sb_beta.setSingleStep(0.001); self.sb_beta.setValue(0.003)
        g2.addWidget(self.sb_beta, 1,1)

        g2.addWidget(QtWidgets.QLabel("OneEuro dcutoff"), 2,0)
        self.sb_dcut = QtWidgets.QDoubleSpinBox(); self.sb_dcut.setRange(0.10, 5.0); self.sb_dcut.setDecimals(2)
        self.sb_dcut.setSingleStep(0.05); self.sb_dcut.setValue(1.00)
        g2.addWidget(self.sb_dcut, 2,1)

        # EMA
        g2.addWidget(QtWidgets.QLabel("EMA α"), 3,0)
        self.sb_ema = QtWidgets.QDoubleSpinBox()
        self.sb_ema.setRange(0.0, 1.0); self.sb_ema.setSingleStep(0.01); self.sb_ema.setValue(0.8)
        g2.addWidget(self.sb_ema, 3,1)

        # 바인딩
        self.cb_mesh.toggled.connect(lambda v: self._set("vis_mesh", v))
        self.cb_center_box.toggled.connect(lambda v: self._set("vis_center_box", v))
        self.cb_iris.toggled.connect(lambda v: self._set("vis_iris", v))
        self.cb_axes.toggled.connect(lambda v: self._set("vis_eye_axes", v))
        self.cb_axes_s.toggled.connect(lambda v: self._set("vis_eye_axes_scaled", v))
        self.cb_uvvec.toggled.connect(lambda v: self._set("vis_uv_vectors", v))
        self.cb_uvvec_big.toggled.connect(lambda v: self._set("vis_uv_vectors_bigger", v))
        self.sb_uv_gain.valueChanged.connect(lambda val: self._set("uv_bigger_gain", float(val)))

        self.sb_minc.valueChanged.connect(lambda val: self._set("oe_mincutoff", float(val)))
        self.sb_beta.valueChanged.connect(lambda val: self._set("oe_beta", float(val)))
        self.sb_dcut.valueChanged.connect(lambda val: self._set("oe_dcutoff", float(val)))
        self.sb_ema.valueChanged.connect(lambda val: self._set("ema_alpha", float(val)))

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(200)
        self.show()

    def _set(self, name, val):
        with self.shared.lock: setattr(self.shared, name, val)

    def _choose_and_load_model(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model file", "./models", "Model files (*.npz *.pkl);;All files (*)")
        if fname: self.shared.set_load_model(fname)

    def tick(self):
        with self.shared.lock: txt = self.shared.status
        self.lbl.setText(txt)

# -------------------- 워커 --------------------
class GazeWorker(threading.Thread):
    def __init__(self, shared: SharedState, args):
        super().__init__(daemon=True)
        self.shared = shared; self.args = args
        self.stop_flag = threading.Event()
        self.oe = OneEuro(mincutoff=0.20, beta=0.003, dcutoff=1.0)
        self.ema_last = None

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print("Could not open webcam."); return

        # 캘리브 설정
        rows, cols = self.args.rows, self.args.cols
        if self.args.grid:
            m = re.match(r'^\s*(\d+)\s*[,xX]\s*(\d+)\s*$', self.args.grid)
            if m: rows, cols = int(m.group(1)), int(m.group(2))
            else: print("[Warn] --grid 예: 4,8 또는 4x8")
        calib = Calibrator(self.shared.screen_w, self.shared.screen_h,
                           rows=rows, cols=cols, margin=self.args.margin,
                           per_point_sec=self.args.per_point)

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

            while not self.stop_flag.is_set():
                # 옵션 스냅샷
                with self.shared.lock:
                    vis_mesh = self.shared.vis_mesh
                    vis_center_box = self.shared.vis_center_box
                    vis_iris = self.shared.vis_iris
                    vis_axes = self.shared.vis_eye_axes
                    vis_axes_s = self.shared.vis_eye_axes_scaled
                    vis_uvvec = self.shared.vis_uv_vectors
                    vis_uvvec_big = self.shared.vis_uv_vectors_bigger
                    uv_gain = float(self.shared.uv_bigger_gain)

                    ema_a = float(self.shared.ema_alpha)
                    # OneEuro 파라미터 주입
                    self.oe.mincutoff = float(self.shared.oe_mincutoff)
                    self.oe.beta      = float(self.shared.oe_beta)
                    self.oe.dcutoff   = float(self.shared.oe_dcutoff)

                ok, frame = cap.read()
                if not ok: continue
                H, W = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
                res = face_mesh.process(rgb)
                frame_out = frame.copy()

                gaze_feat = None; L = R = None
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    (uL, vL), icL, cL, ax1L, ax2L, suL, svL, duL, dvL = _eye_uv_ex(lms, LEFT_EYE_IDXS, LEFT_IRIS_IDXS, "L", W, H)
                    (uR, vR), icR, cR, ax1R, ax2R, suR, svR, duR, dvR = _eye_uv_ex(lms, RIGHT_EYE_IDXS, RIGHT_IRIS_IDXS, "R", W, H)
                    L = dict(u=uL, v=vL, ic=icL, c=cL, ax1=ax1L, ax2=ax2L, su=suL, sv=svL, du=duL, dv=dvL)
                    R = dict(u=uR, v=vR, ic=icR, c=cR, ax1=ax1R, ax2=ax2R, su=suR, sv=svR, du=duR, dv=dvR)
                    gaze_feat = _feat_vector(uL, vL, uR, vR)

                    if vis_mesh:
                        mp_drawing.draw_landmarks(
                            image=frame_out, landmark_list=res.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=frame_out, landmark_list=res.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=frame_out, landmark_list=res.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                    # 홍채 중심 점
                    if vis_iris:
                        cv2.circle(frame_out, (int(icL[0]), int(icL[1])), 3, (0,255,255), -1)  # L: 노랑
                        cv2.circle(frame_out, (int(icR[0]), int(icR[1])), 3, (255,255,0), -1)  # R: 민트

                    # 눈 축(고정 길이) - 두께 1
                    if vis_axes:
                        for eye in (L, R):
                            c = eye["c"]; a1, a2 = eye["ax1"], eye["ax2"]
                            Llen=25
                            p1a = (int(c[0]-a1[0]*Llen), int(c[1]-a1[1]*Llen))
                            p1b = (int(c[0]+a1[0]*Llen), int(c[1]+a1[1]*Llen))
                            p2a = (int(c[0]-a2[0]*Llen), int(c[1]-a2[1]*Llen))
                            p2b = (int(c[0]+a2[0]*Llen), int(c[1]+a2[1]*Llen))
                            cv2.line(frame_out, p1a, p1b, (0,255,0), 1, cv2.LINE_AA)
                            cv2.line(frame_out, p2a, p2b, (255,0,0), 1, cv2.LINE_AA)

                    # 눈 축(눈 크기 길이 = sᵤ/sᵥ) - 두께 고정 len과 동일(1)
                    if vis_axes_s:
                        for eye in (L, R):
                            c = eye["c"]; a1, a2 = eye["ax1"], eye["ax2"]; su, sv = eye["su"], eye["sv"]
                            p1a = (int(c[0]-a1[0]*su), int(c[1]-a1[1]*su))
                            p1b = (int(c[0]+a1[0]*su), int(c[1]+a1[1]*su))
                            p2a = (int(c[0]-a2[0]*sv), int(c[1]-a2[1]*sv))
                            p2b = (int(c[0]+a2[0]*sv), int(c[1]+a2[1]*sv))
                            cv2.line(frame_out, p1a, p1b, (255,0,255), 1, cv2.LINE_AA)  # sᵤ
                            cv2.line(frame_out, p2a, p2b, (0,255,255), 1, cv2.LINE_AA)  # sᵥ

                    # u/v 벡터(정규화된 픽셀 길이: u*su, v*sv)
                    if vis_uvvec or vis_uvvec_big:
                        for eye in (L, R):
                            c = eye["c"]; a1, a2 = eye["ax1"], eye["ax2"]; su, sv = eye["su"], eye["sv"]
                            u, v = eye["u"], eye["v"]
                            base = (int(c[0]), int(c[1]))
                            # 기본(작은) 벡터
                            if vis_uvvec:
                                vec_u = a1 * (u * su)
                                vec_v = a2 * (v * sv)
                                cv2.arrowedLine(frame_out, base, (base[0]+int(vec_u[0]), base[1]+int(vec_u[1])),
                                                (0,200,0), 2, tipLength=0.3)          # u: 초록
                                cv2.arrowedLine(frame_out, base, (base[0]+int(vec_v[0]), base[1]+int(vec_v[1])),
                                                (200,0,0), 2, tipLength=0.3)          # v: 빨강
                            # 큰(배율) 벡터
                            if vis_uvvec_big and uv_gain > 0.0:
                                big_u = a1 * (u * su * uv_gain)
                                big_v = a2 * (v * sv * uv_gain)
                                cv2.arrowedLine(frame_out, base, (base[0]+int(big_u[0]), base[1]+int(big_u[1])),
                                                (255,0,255), 3, tipLength=0.25)       # 보라
                                cv2.arrowedLine(frame_out, base, (base[0]+int(big_v[0]), base[1]+int(big_v[1])),
                                                (0,255,255), 3, tipLength=0.25)       # 노랑

                # ---- 명령 처리 ----
                if self.shared.pop_cmd("quit"): self.stop_flag.set(); break
                if self.shared.pop_cmd("start_calib"): calib.begin()
                if self.shared.pop_cmd("load_model"):
                    path = self.shared.pop_load_path()
                    if path:
                        try:
                            calib.load_linear_model(path)
                            with self.shared.lock: self.shared.status = "Model loaded"
                            print(f"[Model] Loaded: {path}")
                        except Exception as e:
                            with self.shared.lock: self.shared.status = "Model load failed"
                            print("Model load failed:", e)

                if self.shared.pop_cmd("toggle_overlay"):
                    with self.shared.lock: self.shared.overlay_enabled = not self.shared.overlay_enabled
                if self.shared.pop_cmd("toggle_fullscreen"):
                    with self.shared.lock: self.shared.fullscreen = not self.shared.fullscreen

                # ---- 캘리브/런타임 ----
                if calib.collecting:
                    target_px = calib.current_target_px()
                    finished = False
                    if gaze_feat is not None:
                        finished = calib.feed(gaze_feat, t_now=time.time())
                    with self.shared.lock:
                        self.shared.calibrating = True
                        self.shared.calib_target = target_px
                        self.shared.status = f"Calibration {calib.idx+1}/{calib.n_points()}"
                        self.shared.substatus = "오렌지 점을 응시하세요 (자동 진행)"
                        self.shared.cross = None
                    if finished:
                        calib.save_model_pkl()
                        ds_path = calib.save_dataset_npz(DATA_DIR, {
                            "grid": self.args.grid, "rows": rows, "cols": cols,
                            "margin": float(self.args.margin), "camera_index": int(self.args.camera),
                            "mirror_preview": bool(self.args.mirror_preview)
                        })
                        with self.shared.lock:
                            self.shared.calibrating = False
                            self.shared.calib_target = None
                            self.shared.status = "Calibrated & saved data"
                            self.shared.substatus = f"Saved: {os.path.basename(ds_path)}"
                else:
                    px = None
                    if (gaze_feat is not None) and calib.has_model():
                        pred = np.array(calib.predict(gaze_feat), dtype=np.float32)
                        # 1) OneEuro (파라미터는 위에서 최신화됨)
                        oe = self.oe.filter(pred, t=time.time())
                        # 2) EMA
                        a = ema_a
                        if self.ema_last is None: self.ema_last = oe.copy()
                        else: self.ema_last = a*self.ema_last + (1.0-a)*oe
                        sm = self.ema_last
                        px = (int(sm[0]), int(sm[1]))
                    with self.shared.lock:
                        self.shared.calibrating = False
                        self.shared.calib_target = None
                        self.shared.cross = px
                        self.shared.status = "Gaze Overlay"
                        self.shared.substatus = "Use Control Panel"

                # ---- OpenCV 프리뷰 ----
                if self.args.webcam_window:
                    disp = frame_out
                    if self.args.mirror_preview: disp = cv2.flip(disp, 1)

                    # 중앙 박스
                    if vis_center_box:
                        s=150; h,w=disp.shape[:2]
                        x1,y1=w//2 - s, h//2 - s; x2,y2=w//2 + s, h//2 + s
                        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,0), 2, cv2.LINE_AA)
                        half=s//2
                        cv2.rectangle(disp, (x1, y1+half), (x2, y2-half), (0,0,0), 2, cv2.LINE_AA)

                    cv2.imshow('MediaPipe Face Mesh', disp)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27: self.stop_flag.set(); break
                    elif k == ord('c'): self.shared.set_cmd("start_calib")
                    elif k == ord('o'): self.shared.set_cmd("toggle_overlay")
                    elif k == ord('f'): self.shared.set_cmd("toggle_fullscreen")
                else:
                    time.sleep(0.001)

        cap.release(); cv2.destroyAllWindows()
    def stop(self): self.stop_flag.set()

# -------------------- 인자 --------------------
def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe FaceMesh + PyQt click-through gaze overlay (12D)")
    p.add_argument("--grid", type=str, default="", help="예: '4,8' 또는 '4x8' (없으면 5점)")
    p.add_argument("--rows", type=int, default=0); p.add_argument("--cols", type=int, default=0)
    p.add_argument("--margin", type=float, default=0.03, help="그리드 외곽 여백(0~0.45)")
    p.add_argument("--per_point", type=float, default=2.0, help="점당 응시 시간(초)")
    p.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    p.add_argument("--webcam_window", action="store_true", default=True)
    p.add_argument("--no-webcam_window", dest="webcam_window", action="store_false")
    p.add_argument("--mirror_preview", dest="mirror_preview", action="store_true")
    p.add_argument("--no-mirror_preview", dest="mirror_preview", action="store_false")
    p.set_defaults(mirror_preview=True)
    # 초기 스무딩 파라미터 (UI에서 실시간 변경 가능)
    p.add_argument("--ema_a", type=float, default=0.8, help="EMA alpha (0.7~0.85 권장)")
    p.add_argument("--oe_mincutoff", type=float, default=0.20, help="OneEuro mincutoff")
    p.add_argument("--oe_beta", type=float, default=0.003, help="OneEuro beta")
    p.add_argument("--oe_dcutoff", type=float, default=1.0, help="OneEuro dcutoff")
    return p.parse_args()

# -------------------- 메인 --------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen().geometry()
    sw, sh = screen.width(), screen.height()

    shared = SharedState(sw, sh)
    overlay = OverlayWindow(shared, app)
    panel = ControlPanel(shared)

    args = parse_args()
    # 초기 스무딩 파라미터 동기화
    with shared.lock:
        shared.ema_alpha = float(args.ema_a)
        shared.oe_mincutoff = float(args.oe_mincutoff)
        shared.oe_beta = float(args.oe_beta)
        shared.oe_dcutoff = float(args.oe_dcutoff)

    worker = GazeWorker(shared, args); worker.start()
    ret = app.exec_()
    worker.stop(); worker.join(timeout=1.0)
    sys.exit(ret)

if __name__ == "__main__":
    main()
