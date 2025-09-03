
# -*- coding: utf-8 -*-
"""
gaze_overlay_12d.py (+ Eye-Patch Fusion, Patch ROI boxes only, Quit button)
- MediaPipe Face Mesh(iris 포함)로 12D 시선 피처 추출(uL,vL,uR,vR + 2차 확장)
- (추가) 좌/우 눈 패치(작은 정규화 ROI 이미지)를 벡터화하여 12D와 결합 → Ridge2D 회귀
- (유지) 패치 ROI 사각형 시각화 (썸네일 기능은 삭제)
- 캘리브레이션: 그리드(행x열, 지그재그), 포인트당 체류시간 수집, delay 이후에만 수집
- 오버레이: 캘리브 중 검정 배경, 타깃 고리 → delay 경과 후 채워진 원
- 단축키: OpenCV 창 'q'로 캘리브 중지(또는 패널에서 'Q'); ESC로 전체 종료
- 데이터 저장: data/gaze_samples_*.npz (X,Y,T,pt_index,screen,feature_names,meta)
- 모델: 릿지 선형(.pkl/.npz 로드)
- 스무딩: OneEuro + EMA
- 시각화(토글): FaceMesh/홍채/축/컨투어/u-v 벡터/패치 ROI 박스
- UI: Calibration(Per-point, Delay), Command(시작/중지/로드/오버레이, Quit), Visualization, Smoothing
"""
import os, sys, time, argparse, re, threading, json, inspect
import numpy as np
import cv2
import mediapipe as mp
import pickle
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from collections import deque

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# -------------------- 경로 --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------- MediaPipe --------------------
mp_drawing         = mp.solutions.drawing_utils
mp_drawing_styles  = mp.solutions.drawing_styles
mp_face_mesh       = mp.solutions.face_mesh

LEFT_IRIS_IDXS  = [474, 475, 476, 477]  # 좌안 4점
RIGHT_IRIS_IDXS = [469, 470, 471, 472]  # 우안 4점

def _unique_idxs(connections):
    s = set()
    for a, b in connections: s.add(a); s.add(b)
    return sorted(s)

LEFT_EYE_ALL_IDXS  = _unique_idxs(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_ALL_IDXS = _unique_idxs(mp_face_mesh.FACEMESH_RIGHT_EYE)

# 12D 기하 피처 이름
FEATURE_NAMES_12D = [
    "uL","vL","uR","vR",
    "uL2","vL2","uR2","vR2",
    "uL_vL","uR_vR","uL_uR","vL_vR"
]

# -------------------- 수학/피처 --------------------
def _pca_axes(pts: np.ndarray):
    c = pts.mean(axis=0)
    X = pts - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ax1, ax2 = Vt[0], Vt[1]              # 단위벡터
    su = 2 * (np.sqrt(np.mean((X @ ax1)**2)) + 1e-6) / 2.0  # RMS 기반 스케일
    sv = 2 * (np.sqrt(np.mean((X @ ax2)**2)) + 1e-6) / 2.0
    return c, ax1, ax2, su, sv

def _iris_center(landmarks, idxs, W, H):
    pts = np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)

def _eye_uv_ex(landmarks, eye_idxs, iris_idxs, W, H):
    eye_pts = np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in eye_idxs], dtype=np.float32)
    c, ax1, ax2, su, sv = _pca_axes(eye_pts)
    ic = _iris_center(landmarks, iris_idxs, W, H)
    delta = ic - c
    du = float(np.dot(delta, ax1))   # Δ·û (px)
    dv = float(np.dot(delta, ax2))   # Δ·v̂ (px)
    u = float(du / su)
    v = float(dv / sv)
    return (u, v), ic, c, ax1, ax2, su, sv, du, dv, eye_pts

def _feat_vector_12d(uL, vL, uR, vR):
    return np.array([
        uL, vL, uR, vR,
        uL*uL, vL*vL, uR*uR, vR*vR,
        uL*vL, uR*vR, uL*uR, vL*vR
    ], dtype=np.float32)

def _order_quad_clockwise(pts_xy: np.ndarray) -> np.ndarray:
    c = pts_xy.mean(axis=0)
    ang = np.arctan2(pts_xy[:,1] - c[1], pts_xy[:,0] - c[0])
    order = np.argsort(ang)
    return pts_xy[order]

# -------------------- (PATCH) 눈 패치 유틸 --------------------
def _warp_oriented_patch(bgr, center_xy, ax1, ax2, half_w, half_h, out_w, out_h):
    """
    눈의 PCA 축(ax1, ax2)에 정렬된 직사각형 ROI를 out_h x out_w로 워핑
    center_xy: (cx,cy), ax1/ax2: 단위벡터, half_w/half_h: 픽셀 단위 반폭/반높이
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ax1 = np.asarray(ax1, dtype=np.float32); ax2 = np.asarray(ax2, dtype=np.float32)

    # 소스 평행사변형 3점 (좌상, 우상, 좌하) → 목적 사각형 3점
    src_tl = [cx - ax1[0]*half_w - ax2[0]*half_h, cy - ax1[1]*half_w - ax2[1]*half_h]
    src_tr = [cx + ax1[0]*half_w - ax2[0]*half_h, cy + ax1[1]*half_w - ax2[1]*half_h]
    src_bl = [cx - ax1[0]*half_w + ax2[0]*half_h, cy - ax1[1]*half_w + ax2[1]*half_h]
    src  = np.float32([src_tl, src_tr, src_bl])
    dst  = np.float32([[0,0], [out_w-1,0], [0,out_h-1]])
    M = cv2.getAffineTransform(src, dst)
    patch = cv2.warpAffine(bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return patch, np.array([src_tl, src_tr, src_bl], dtype=np.float32)

def _patch_to_vec(gray_patch, do_clahe=True, norm="z"):
    """
    gray_patch: uint8
    do_clahe: CLAHE 적용 여부
    norm: "z" (z-score per-sample) 또는 None
    반환: float32 1D (0~1 스케일 후 정규화)
    """
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        gray_patch = clahe.apply(gray_patch)
    vec = gray_patch.astype(np.float32) / 255.0
    vec = vec.reshape(-1)
    if norm == "z":
        m = float(vec.mean()); s = float(vec.std() + 1e-6)
        vec = (vec - m) / s
    return vec.astype(np.float32)

def _build_feature_names(use_patches, pw, ph):
    if not use_patches:
        return list(FEATURE_NAMES_12D)
    n = pw*ph
    L = [f"Lpx{i}" for i in range(n)]
    R = [f"Rpx{i}" for i in range(n)]
    return list(FEATURE_NAMES_12D) + L + R

def _draw_oriented_box(img, tri3, color=(0,200,255), thickness=1):
    """Affine에 사용한 3점을 받아 네번째 점을 복원해 사각형 폴리라인을 그림"""
    tl, tr, bl = tri3
    tl = np.asarray(tl); tr = np.asarray(tr); bl = np.asarray(bl)
    br = tr + (bl - tl)
    poly = np.array([tl, tr, br, bl], dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)

# -------------------- 캘리브 --------------------
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
        self.alpha = alpha; self.W = None; self.b = None
    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
        N, D = X.shape
        Xb = np.hstack([X, np.ones((N,1), dtype=np.float32)])
        I = np.eye(D+1, dtype=np.float32); I[-1,-1] = 0.0
        A = Xb.T @ Xb + self.alpha * I
        Wb = np.linalg.pinv(A) @ (Xb.T @ Y)
        self.W = Wb[:-1,:].T; self.b = Wb[-1,:].T
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (self.W @ X.T).T + self.b

class Calibrator:
    def __init__(self, screen_w, screen_h, rows=0, cols=0, margin=0.10, per_point_sec=2.0, delay_sec=0.5,
                 feature_names=None):
        self.sw, self.sh = screen_w, screen_h
        self.rows, self.cols, self.margin = rows, cols, margin
        self.points_norm = make_grid_points(rows, cols, margin, "serpentine") if (rows and cols) else FIVE_POINTS[:]
        self.per_point_sec = per_point_sec
        self.delay_sec = float(delay_sec)
        self.reset()
        self.model = Ridge2D(alpha=10.0)
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES_12D)

    def reset(self):
        self.idx = 0; self.collecting = False
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
        tcur = float(time.time() if t_now is None else t_now)
        if (tcur - self.start_t) >= self.delay_sec:   # delay 경과 시에만 기록
            self.samples_X.append(np.array(feat, dtype=np.float32))
            self.samples_Y.append(np.array([tx, ty], dtype=np.float32))
            self.samples_IDX.append(int(self.idx))
            self.samples_T.append(tcur)

        if (tcur - self.start_t) >= self.per_point_sec:
            self.idx += 1
            self.start_t = time.time()
            if self.idx >= len(self.points_norm):
                self.model.fit(np.stack(self.samples_X), np.stack(self.samples_Y))
                self.collecting = False
                return True
        return False

    def has_model(self):  return (self.model.W is not None)

    def predict(self, feat):
        y = self.model.predict(np.array([feat], dtype=np.float32))[0]
        x = int(np.clip(y[0], 0, self.sw - 1))
        yv= int(np.clip(y[1], 0, self.sh - 1))
        return x, yv

    def save_model_pkl(self, path=None):
        if path is None: path = os.path.join(MODELS_DIR, "calib_gaze.pkl")
        with open(path, "wb") as f:
            pickle.dump({"W": self.model.W, "b": self.model.b, "screen": (self.sw, self.sh),
                         "feature_names": self.feature_names}, f)

    def save_dataset_npz(self, out_dir=DATA_DIR, meta_extra=None):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"gaze_samples_{ts}.npz")
        X = np.stack(self.samples_X).astype(np.float32)
        Y = np.stack(self.samples_Y).astype(np.float32)
        T   = np.array(self.samples_T,  dtype=np.float64)
        IDX = np.array(self.samples_IDX, dtype=np.int32)
        meta = {"rows": self.rows, "cols": self.cols, "margin": float(self.margin),
                "per_point_sec": float(self.per_point_sec), "delay_sec": float(self.delay_sec),
                "n_points": int(len(self.points_norm)), "timestamp": ts}
        if meta_extra: meta.update(meta_extra)
        np.savez(path, X=X, Y=Y, T=T, pt_index=IDX,
                 feature_names=np.array(self.feature_names, dtype=object),
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
            self.t_prev=t; self.x_prev=x.copy(); self.dx_prev=np.zeros_like(x)
            return x
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0/dt
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        cutoff = self.mincutoff + self.beta*float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev=x_hat; self.dx_prev=dx_hat; self.t_prev=t
        return x_hat

# -------------------- 공유 상태 --------------------
class SharedState:
    def __init__(self, sw:int, sh:int):
        self.lock = threading.Lock()
        self.screen_w=sw; self.screen_h=sh
        self.cross=None; self.calib_target=None; self.calibrating=False
        self.overlay_enabled=True
        self.status="Gaze Overlay"; self.substatus="Use Control Panel"

        # Calibration 설정(UI)
        self.calib_rows=8; self.calib_cols=12; self.calib_per_point=2.0; self.calib_margin=0.03
        self.calib_delay_sec = 0.5   # 포인트 이동 후 수집 지연(초)
        self.calib_ready = False     # delay 경과하여 수집 시작 가능한 상태

        # 시각화 옵션
        self.vis_mesh=False
        self.vis_iris=True
        self.vis_iris_quad=True
        self.vis_eye_axes=True
        self.vis_eye_axes_scaled=False
        self.vis_uv_vectors=False
        self.vis_uv_vectors_bigger=True
        self.uv_bigger_gain=25.0
        self.vis_eye_contour_pts   = False
        self.vis_eye_contour_edges = True
        # 패치 시각화 옵션
        self.vis_eye_patch_boxes   = False      # ROI 사각형 (썸네일 옵션 삭제됨)

        # 스무딩
        self.ema_alpha=0.8
        self.oe_mincutoff=0.20; self.oe_beta=0.003; self.oe_dcutoff=1.0

        # 명령
        self.cmd={"start_calib":False,"stop_calib":False,"load_model":False,
                  "toggle_overlay":False,"quit":False}
        self._load_path=None

    def set_cmd(self,name):
        with self.lock:
            if name in self.cmd: self.cmd[name]=True
    def pop_cmd(self,name):
        with self.lock:
            v=self.cmd.get(name,False); self.cmd[name]=False; return v
    def set_load_model(self,path:str):
        with self.lock: self._load_path=path; self.cmd["load_model"]=True
    def pop_load_path(self):
        with self.lock: p=self._load_path; self._load_path=None; return p

# -------------------- 오버레이 --------------------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, app: QtWidgets.QApplication):
        super().__init__(None)
        self.shared=shared; self.app=app
        self.status="Gaze Overlay"; self.substatus="Use Control Panel"
        self.cross=None; self.calib_target=None

        flags = (QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint |
                 QtCore.Qt.Tool | QtCore.Qt.WindowDoesNotAcceptFocus)
        if hasattr(QtCore.Qt,"WindowTransparentForInput"): flags |= QtCore.Qt.WindowTransparentForInput
        if hasattr(QtCore.Qt,"X11BypassWindowManagerHint"): flags |= QtCore.Qt.X11BypassWindowManagerHint
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.setGeometry(self.app.primaryScreen().geometry())
        self.timer=QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(16)
        self.show()

    def tick(self):
        with self.shared.lock:
            self.cross=self.shared.cross
            self.calib_target=self.shared.calib_target
            enabled=self.shared.overlay_enabled
            self.status=self.shared.status; self.substatus=self.shared.substatus

        if enabled:
            if not self.isVisible(): self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True); self.show()
        else:
            if self.isVisible(): self.hide()
        if enabled: self.update()

    def paintEvent(self,_):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(200,200,200,230), 2); p.setPen(pen)
        p.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold)); p.drawText(30,50,self.status)
        pen = QtGui.QPen(QtGui.QColor(160,160,160,200), 1); p.setPen(pen)
        p.setFont(QtGui.QFont("Arial", 12)); p.drawText(30,80,self.substatus)

        with self.shared.lock:
            is_calib = self.shared.calibrating
            is_ready = bool(getattr(self.shared, 'calib_ready', False))
        if is_calib:
            p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 255))  # 검정 배경

        # 캘리브 타깃: 고리 → delay 후 채워진 원
        if self.calib_target is not None:
            tx, ty = self.calib_target
            if is_ready:
                pen = QtGui.QPen(QtGui.QColor(255,165,0,255), 0)
                p.setPen(pen); p.setBrush(QtGui.QColor(255,165,0,255))
                p.drawEllipse(QtCore.QPointF(tx, ty), 16, 16)
            else:
                pen = QtGui.QPen(QtGui.QColor(255,165,0,240), 4)
                p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(QtCore.QPointF(tx, ty), 16, 16)

        # 추정점: 빨간 고리
        with self.shared.lock: cross=self.shared.cross
        if cross is not None:
            x,y=cross; pen=QtGui.QPen(QtGui.QColor(255,0,0,230), 4)
            p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush); p.drawEllipse(QtCore.QPointF(x,y), 14,14)

# -------------------- 컨트롤 패널 --------------------
class ControlPanel(QtWidgets.QWidget):
    def __init__(self, shared: SharedState):
        super().__init__()
        self.shared=shared
        self.setWindowTitle("Control Panel")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setFixedWidth(680)

        v = QtWidgets.QVBoxLayout(self)

        # Calibration grid
        grpC = QtWidgets.QGroupBox("Calibration Grid"); v.addWidget(grpC)
        gc = QtWidgets.QGridLayout(grpC)
        self.sb_rows = QtWidgets.QSpinBox(); self.sb_rows.setRange(1,100); self.sb_rows.setValue(self.shared.calib_rows)
        self.sb_cols = QtWidgets.QSpinBox(); self.sb_cols.setRange(1,100); self.sb_cols.setValue(self.shared.calib_cols)
        gc.addWidget(QtWidgets.QLabel("Rows"),0,0); gc.addWidget(self.sb_rows,0,1)
        gc.addWidget(QtWidgets.QLabel("Columns"),1,0); gc.addWidget(self.sb_cols,1,1)
        # calib sec
        self.sb_per  = QtWidgets.QDoubleSpinBox(); self.sb_per.setRange(0.1,10.0); self.sb_per.setSingleStep(0.1); self.sb_per.setDecimals(2); self.sb_per.setValue(self.shared.calib_per_point)
        self.sb_delay = QtWidgets.QDoubleSpinBox()
        self.sb_delay.setRange(0.0, 10.0); self.sb_delay.setDecimals(2); self.sb_delay.setValue(self.shared.calib_delay_sec)
        gc.addWidget(QtWidgets.QLabel("Per-point (sec)"),2,0); gc.addWidget(self.sb_per,2,1)
        gc.addWidget(QtWidgets.QLabel("Delay (sec)"), 3, 0); gc.addWidget(self.sb_delay, 3, 1)
        # 바인딩
        self.sb_rows.valueChanged.connect(lambda v_: self._set("calib_rows", int(v_)))
        self.sb_cols.valueChanged.connect(lambda v_: self._set("calib_cols", int(v_)))
        self.sb_per .valueChanged.connect(lambda v_: self._set("calib_per_point", float(v_)))
        self.sb_delay.valueChanged.connect(lambda v_: self._set("calib_delay_sec", float(v_)))

        # === Calibration Command ===
        grpCmd = QtWidgets.QGroupBox("Calibration Command")
        glcmd  = QtWidgets.QGridLayout(grpCmd)
        b_calib = QtWidgets.QPushButton("Start Calibration")
        b_stop  = QtWidgets.QPushButton("Stop Calibration")
        b_load  = QtWidgets.QPushButton("Load Model (.npz/.pkl)")
        b_ov    = QtWidgets.QPushButton("Hide/Show Overlay")
        b_quit  = QtWidgets.QPushButton("Quit")
        glcmd.addWidget(b_calib, 0, 0); glcmd.addWidget(b_stop,  0, 1)
        glcmd.addWidget(b_load,  1, 0); glcmd.addWidget(b_ov,    1, 1)
        glcmd.addWidget(b_quit,  2, 0, 1, 2)
        b_calib.clicked.connect(lambda: self.shared.set_cmd("start_calib"))
        b_stop .clicked.connect(lambda: self.shared.set_cmd("stop_calib"))
        b_load .clicked.connect(self._choose_and_load_model)
        b_ov   .clicked.connect(lambda: self.shared.set_cmd("toggle_overlay"))
        b_quit .clicked.connect(lambda: self.shared.set_cmd("quit"))
        v.addWidget(grpCmd)

        # Visualization
        grp = QtWidgets.QGroupBox("Visualization"); v.addWidget(grp)
        gl = QtWidgets.QGridLayout(grp)
        self.cb_mesh        = QtWidgets.QCheckBox("Face Mesh");            self.cb_mesh.setChecked(False)
        self.cb_iris        = QtWidgets.QCheckBox("Iris centers");         self.cb_iris.setChecked(True)
        self.cb_iris_quad   = QtWidgets.QCheckBox("Iris 4-edges");         self.cb_iris_quad.setChecked(True)
        self.cb_axes        = QtWidgets.QCheckBox('Eye axes (fixed length; u_hat, v_hat)');   self.cb_axes.setChecked(True)
        self.cb_axes_s      = QtWidgets.QCheckBox('Eye axes (eye scaled length; s_u, s_v)');  self.cb_axes_s.setChecked(False)
        self.cb_uvvec       = QtWidgets.QCheckBox("u, v vectors");         self.cb_uvvec.setChecked(False)
        self.cb_uvvec_big   = QtWidgets.QCheckBox("u, v vectors (bigger)");self.cb_uvvec_big.setChecked(True)
        self.sb_uv_gain     = QtWidgets.QDoubleSpinBox(); self.sb_uv_gain.setRange(0.1,100.0); self.sb_uv_gain.setSingleStep(0.1); self.sb_uv_gain.setDecimals(1); self.sb_uv_gain.setValue(25.0)
        self.cb_cnt_pts     = QtWidgets.QCheckBox("Eye contour points");   self.cb_cnt_pts.setChecked(False)
        self.cb_cnt_edges   = QtWidgets.QCheckBox("Eye contour edges");    self.cb_cnt_edges.setChecked(True)
        # 패치 시각화 토글 (썸네일 삭제, ROI 박스만 유지)
        self.cb_patch_boxes = QtWidgets.QCheckBox("Eye patch ROI boxes");  self.cb_patch_boxes.setChecked(False)

        gl.addWidget(self.cb_mesh,       0,0)
        gl.addWidget(self.cb_iris,       1,0); gl.addWidget(self.cb_iris_quad,  1,1)
        gl.addWidget(self.cb_axes,       2,0); gl.addWidget(self.cb_axes_s,     2,1)
        gl.addWidget(self.cb_uvvec,      3,0); gl.addWidget(self.cb_uvvec_big,  3,1)
        gl.addWidget(QtWidgets.QLabel("u, v vectors bigger gain"), 4,0); gl.addWidget(self.sb_uv_gain, 4,1)
        gl.addWidget(self.cb_cnt_pts,    5,0); gl.addWidget(self.cb_cnt_edges,  5,1)
        gl.addWidget(self.cb_patch_boxes,6,0)

        # Smoothing
        grp2 = QtWidgets.QGroupBox("Smoothing Factors"); v.addWidget(grp2)
        g2 = QtWidgets.QGridLayout(grp2)
        g2.addWidget(QtWidgets.QLabel("OneEuro mincutoff"), 0,0)
        self.sb_minc = QtWidgets.QDoubleSpinBox(); self.sb_minc.setRange(0.01,2.0); self.sb_minc.setDecimals(3); self.sb_minc.setSingleStep(0.01); self.sb_minc.setValue(0.20)
        g2.addWidget(self.sb_minc,0,1)
        g2.addWidget(QtWidgets.QLabel("OneEuro beta"), 1,0)
        self.sb_beta = QtWidgets.QDoubleSpinBox(); self.sb_beta.setRange(0.000,0.100); self.sb_beta.setDecimals(3); self.sb_beta.setSingleStep(0.001); self.sb_beta.setValue(0.003)
        g2.addWidget(self.sb_beta,1,1)
        g2.addWidget(QtWidgets.QLabel("OneEuro dcutoff"), 2,0)
        self.sb_dcut = QtWidgets.QDoubleSpinBox(); self.sb_dcut.setRange(0.10,5.0); self.sb_dcut.setDecimals(2); self.sb_dcut.setSingleStep(0.05); self.sb_dcut.setValue(1.00)
        g2.addWidget(self.sb_dcut,2,1)
        g2.addWidget(QtWidgets.QLabel("EMA α"), 3,0)
        self.sb_ema = QtWidgets.QDoubleSpinBox(); self.sb_ema.setRange(0.0,1.0); self.sb_ema.setSingleStep(0.01); self.sb_ema.setValue(0.8)
        g2.addWidget(self.sb_ema,3,1)

        self.show()
        # 단축키 Q: 캘리브 중지
        q_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self)
        q_shortcut.activated.connect(lambda: self.shared.set_cmd("stop_calib"))

        # 바인딩
        self.sb_minc.valueChanged  .connect(lambda val: self._set("oe_mincutoff", float(val)))
        self.sb_beta.valueChanged  .connect(lambda val: self._set("oe_beta", float(val)))
        self.sb_dcut.valueChanged  .connect(lambda val: self._set("oe_dcutoff", float(val)))
        self.sb_ema.valueChanged   .connect(lambda val: self._set("ema_alpha", float(val)))
        self.cb_mesh.toggled       .connect(lambda v_: self._set("vis_mesh", v_))
        self.cb_iris.toggled       .connect(lambda v_: self._set("vis_iris", v_))
        self.cb_iris_quad.toggled  .connect(lambda v_: self._set("vis_iris_quad", v_))
        self.cb_axes.toggled       .connect(lambda v_: self._set("vis_eye_axes", v_))
        self.cb_axes_s.toggled     .connect(lambda v_: self._set("vis_eye_axes_scaled", v_))
        self.cb_uvvec.toggled      .connect(lambda v_: self._set("vis_uv_vectors", v_))
        self.cb_uvvec_big.toggled  .connect(lambda v_: self._set("vis_uv_vectors_bigger", v_))
        self.sb_uv_gain.valueChanged.connect(lambda val: self._set("uv_bigger_gain", float(val)))
        self.cb_cnt_pts.toggled    .connect(lambda v_: self._set("vis_eye_contour_pts",   v_))
        self.cb_cnt_edges.toggled  .connect(lambda v_: self._set("vis_eye_contour_edges", v_))
        self.cb_patch_boxes.toggled.connect(lambda v_: self._set("vis_eye_patch_boxes", v_))

    def _set(self,name,val):
        with self.shared.lock: setattr(self.shared,name,val)
    def _choose_and_load_model(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select model file",MODELS_DIR,"Model files (*.npz *.pkl);;All files (*)")
        if fname: self.shared.set_load_model(fname)

# -------------------- FaceMesh 빌더 --------------------
def build_facemesh(args):
    sig = inspect.signature(mp_face_mesh.FaceMesh.__init__)
    supported = set(sig.parameters.keys())
    kw = {
        "max_num_faces": 1,
        "static_image_mode": False,
        "refine_landmarks": args.mp_refine_landmarks,
        "min_detection_confidence": args.mp_min_det,
        "min_tracking_confidence": args.mp_min_track,
    }
    kw = {k:v for k,v in kw.items() if k in supported}
    return mp_face_mesh.FaceMesh(**kw)

# -------------------- 워커 --------------------
class GazeWorker(threading.Thread):
    def __init__(self, shared: SharedState, args):
        super().__init__(daemon=True)
        self.shared=shared; self.args=args
        self.stop_flag=threading.Event()
        self.oe = OneEuro(mincutoff=float(args.oe_mincutoff), beta=float(args.oe_beta), dcutoff=float(args.oe_dcutoff))
        self.ema_last=None

        # 패치 피처 구성
        self.use_patches = bool(args.use_patches)
        self.patch_w = int(args.patch_w); self.patch_h = int(args.patch_h)
        self.patch_scale_w = float(args.patch_scale_w)
        self.patch_scale_h = float(args.patch_scale_h)
        self.patch_norm = "z" if args.patch_norm == "z" else None
        self.patch_clahe = bool(args.patch_clahe)

        self.feat_names = _build_feature_names(self.use_patches, self.patch_w, self.patch_h)

        # 초기 캘리브(실제 시작 시 다시 생성)
        self.calib=Calibrator(shared.screen_w, shared.screen_h, 0,0, args.margin, args.per_point, args.delay_time,
                              feature_names=self.feat_names)

        # su/sv 안정화를 위한 히스토리(옵션)
        self.su_hist_L = deque(maxlen=120); self.sv_hist_L = deque(maxlen=120)
        self.su_hist_R = deque(maxlen=120); self.sv_hist_R = deque(maxlen=120)

    def _start_new_calibration(self):
        with self.shared.lock:
            rows=int(self.shared.calib_rows); cols=int(self.shared.calib_cols)
            perp=float(self.shared.calib_per_point); margin=float(self.shared.calib_margin); dsec=float(self.shared.calib_delay_sec)
        self.calib = Calibrator(self.shared.screen_w, self.shared.screen_h, rows, cols, margin, perp, dsec,
                                feature_names=self.feat_names)
        self.calib.begin()
        if dsec >= perp:
            with self.shared.lock:
                self.shared.status = "Warning: delay ≥ per-point"
                self.shared.substatus = "이 포인트에서는 수집시간이 0초가 됩니다."

    def _stop_calibration(self, save=False):
        if self.calib and self.calib.collecting: self.calib.collecting=False
        with self.shared.lock:
            self.shared.calibrating=False; self.shared.calib_target=None
            self.shared.substatus = "Calibration stopped" if not save else "Calibration saved"

    @staticmethod
    def _draw_points(img, pts, color, r=2):
        for (x,y) in pts: cv2.circle(img,(int(x),int(y)), r, color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_edges(img, connections, lms, W, H, color=(0,200,255), thickness=1):
        for a,b in connections:
            pa=(int(lms[a].x*W), int(lms[a].y*H))
            pb=(int(lms[b].x*W), int(lms[b].y*H))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)

    def _build_fused_feature(self, frame_bgr, uL, vL, uR, vR,
                             cL, ax1L, ax2L, suL, svL,
                             cR, ax1R, ax2R, suR, svR):
        f12 = _feat_vector_12d(uL, vL, uR, vR)
        if not self.use_patches:
            return f12, None, None, None, None  # (feat, pL, triL, pR, triR)

        # half sizes in pixels (RMS scale 기반 스케일 × 게인)
        half_w_L = suL * self.patch_scale_w
        half_h_L = svL * self.patch_scale_h
        half_w_R = suR * self.patch_scale_w
        half_h_R = svR * self.patch_scale_h

        # 좌/우 패치 추출 (PCA 축 정렬 워핑)
        pL, triL = _warp_oriented_patch(frame_bgr, cL, ax1L, ax2L, half_w_L, half_h_L, self.patch_w, self.patch_h)
        pR, triR = _warp_oriented_patch(frame_bgr, cR, ax1R, ax2R, half_w_R, half_h_R, self.patch_w, self.patch_h)

        # 그레이스케일 + (선택)CLAHE + 정규화 → 1D 벡터
        gL = cv2.cvtColor(pL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(pR, cv2.COLOR_BGR2GRAY)
        vL = _patch_to_vec(gL, do_clahe=self.patch_clahe, norm=self.patch_norm)
        vR = _patch_to_vec(gR, do_clahe=self.patch_clahe, norm=self.patch_norm)

        return np.concatenate([f12, vL, vR], axis=0).astype(np.float32), pL, triL, pR, triR

    def run(self):
        # --- 카메라 오픈 ---
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print("Could not open webcam."); return

        # 입력 품질(가능하면 MJPG 고정)
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.args.cam_fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.args.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.cam_h)
        cap.set(cv2.CAP_PROP_FPS,          self.args.cam_fps)

        # 실제 협상된 값 로깅
        real_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps= cap.get(cv2.CAP_PROP_FPS)
        fourcc_v = int(cap.get(cv2.CAP_PROP_FOURCC))
        fc_str = "".join([chr((fourcc_v >> 8*i) & 0xFF) for i in range(4)])
        print(f"[CAM] {real_w}x{real_h} @ {real_fps:.1f} FPS, FourCC={fc_str}")

        # FaceMesh
        with build_facemesh(self.args) as face_mesh:
            while not self.stop_flag.is_set():
                with self.shared.lock:
                    vis_mesh       = self.shared.vis_mesh
                    vis_iris       = self.shared.vis_iris
                    vis_iris_quad  = self.shared.vis_iris_quad
                    vis_axes       = self.shared.vis_eye_axes
                    vis_axes_s     = self.shared.vis_eye_axes_scaled
                    vis_uvvec      = self.shared.vis_uv_vectors
                    vis_uvvec_big  = self.shared.vis_uv_vectors_bigger
                    uv_gain        = float(self.shared.uv_bigger_gain)
                    vis_cnt_pts    = self.shared.vis_eye_contour_pts
                    vis_cnt_edges  = self.shared.vis_eye_contour_edges
                    vis_patch_boxes= self.shared.vis_eye_patch_boxes
                    ema_a          = float(self.shared.ema_alpha)
                    self.oe.mincutoff=float(self.args.oe_mincutoff)
                    self.oe.beta     =float(self.args.oe_beta)
                    self.oe.dcutoff  =float(self.args.oe_dcutoff)

                ok, frame = cap.read()
                if not ok: continue
                H, W = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
                res = face_mesh.process(rgb)
                out = frame.copy()

                gaze_feat=None; pL=pR=None; triL=triR=None

                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    (uL, vL), icL, cL, ax1L, ax2L, suL, svL, duL, dvL, l_eye_pts = _eye_uv_ex(lms, LEFT_EYE_ALL_IDXS,  LEFT_IRIS_IDXS,  W, H)
                    (uR, vR), icR, cR, ax1R, ax2R, suR, svR, duR, dvR, r_eye_pts = _eye_uv_ex(lms, RIGHT_EYE_ALL_IDXS, RIGHT_IRIS_IDXS, W, H)

                    # 패치 + 12D 결합 피처 생성
                    gaze_feat, pL, triL, pR, triR = self._build_fused_feature(out, uL, vL, uR, vR,
                                                                              cL, ax1L, ax2L, suL, svL,
                                                                              cR, ax1R, ax2R, suR, svR)

                    # ---- 시각화 (기존 유지) ----
                    if vis_mesh:
                        mp_drawing.draw_landmarks(out, res.multi_face_landmarks[0],
                            mp_face_mesh.FACEMESH_TESSELATION, None,
                            mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(out, res.multi_face_landmarks[0],
                            mp_face_mesh.FACEMESH_CONTOURS, None,
                            mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(out, res.multi_face_landmarks[0],
                            mp_face_mesh.FACEMESH_IRISES, None,
                            mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                    if vis_iris:
                        cv2.circle(out,(int(icL[0]),int(icL[1])),3,(255,255,0),-1)
                        cv2.circle(out,(int(icR[0]),int(icR[1])),3,(255,255,0),-1)

                    if vis_iris_quad:
                        ptsL = np.array([(lms[i].x * W, lms[i].y * H) for i in LEFT_IRIS_IDXS],  dtype=np.float32)
                        ptsR = np.array([(lms[i].x * W, lms[i].y * H) for i in RIGHT_IRIS_IDXS], dtype=np.float32)
                        for pts, color in [(ptsL, (255,255,0)), (ptsR, (255,255,0))]:
                            ordered = _order_quad_clockwise(pts)
                            poly    = ordered.astype(np.int32).reshape(-1,1,2)
                            cv2.polylines(out, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

                    if vis_axes:
                        # 왼쪽
                        L_L=25
                        cv2.line(out,(int(cL[0]-ax1L[0]*L_L),int(cL[1]-ax1L[1]*L_L)),
                                     (int(cL[0]+ax1L[0]*L_L),int(cL[1]+ax1L[1]*L_L)),(0,225,0),1,cv2.LINE_AA)
                        cv2.line(out,(int(cL[0]-ax2L[0]*L_L),int(cL[1]-ax2L[1]*L_L)),
                                     (int(cL[0]+ax2L[0]*L_L),int(cL[1]+ax2L[1]*L_L)),(0,255,0),1,cv2.LINE_AA)
                        # 오른쪽
                        L_R=25
                        cv2.line(out,(int(cR[0]-ax1R[0]*L_R),int(cR[1]-ax1R[1]*L_R)),
                                     (int(cR[0]+ax1R[0]*L_R),int(cR[1]+ax1R[1]*L_R)),(0,0,255),1,cv2.LINE_AA)
                        cv2.line(out,(int(cR[0]-ax2R[0]*L_R),int(cR[1]-ax2R[1]*L_R)),
                                     (int(cR[0]+ax2R[0]*L_R),int(cR[1]+ax2R[1]*L_R)),(0,0,255),1,cv2.LINE_AA)

                    if vis_axes_s:
                        for c,a1,a2,su,sv in [(cL,ax1L,ax2L,suL,svL),(cR,ax1R,ax2R,suR,svR)]:
                            cv2.line(out,(int(c[0]-a1[0]*su),int(c[1]-a1[1]*su)),(int(c[0]+a1[0]*su),int(c[1]+a1[1]*su)),(255,0,255),1,cv2.LINE_AA)
                            cv2.line(out,(int(c[0]-a2[0]*sv),int(c[1]-a2[1]*sv)),(int(c[0]+a2[0]*sv),int(c[1]+a2[1]*sv)),(0,255,255),1,cv2.LINE_AA)

                    if vis_uvvec or vis_uvvec_big:
                        for (c,a1,a2,su,sv,u,v) in [(cL,ax1L,ax2L,suL,svL,uL,vL),(cR,ax1R,ax2R,suR,svR,uR,vR)]:
                            base=(int(c[0]),int(c[1]))
                            if vis_uvvec:
                                vec_u = a1*(u*su); vec_v = a2*(v*sv)
                                cv2.arrowedLine(out, base, (base[0]+int(vec_u[0]), base[1]+int(vec_u[1])), (255,0,255),2,tipLength=0.3)
                                cv2.arrowedLine(out, base, (base[0]+int(vec_v[0]), base[1]+int(vec_v[1])), (0,255,255),2,tipLength=0.3)
                            if vis_uvvec_big and uv_gain>0.0:
                                big_u=a1*(u*su*uv_gain); big_v=a2*(v*sv*uv_gain)
                                cv2.arrowedLine(out, base, (base[0]+int(big_u[0]), base[1]+int(big_u[1])), (255,0,255),3,tipLength=0.25)
                                cv2.arrowedLine(out, base, (base[0]+int(big_v[0]), base[1]+int(big_v[1])), (0,255,255),3,tipLength=0.25)

                    if vis_cnt_pts:
                        for pt in l_eye_pts: cv2.circle(out,(int(pt[0]),int(pt[1])),2,(0,255,0),-1,cv2.LINE_AA)
                        for pt in r_eye_pts: cv2.circle(out,(int(pt[0]),int(pt[1])),2,(0,0,255),-1,cv2.LINE_AA)
                    if vis_cnt_edges:
                        self._draw_edges(out, mp_face_mesh.FACEMESH_LEFT_EYE,  lms, W, H, color=(0,255,0), thickness=1)
                        self._draw_edges(out, mp_face_mesh.FACEMESH_RIGHT_EYE, lms, W, H, color=(0,0,255), thickness=1)

                    # 패치 ROI 박스 시각화 (썸네일은 삭제됨)
                    if self.use_patches and vis_patch_boxes:
                        if triL is not None: _draw_oriented_box(out, triL, color=(0,200,255), thickness=1)
                        if triR is not None: _draw_oriented_box(out, triR, color=(0,200,255), thickness=1)

                # 명령 처리
                if self.shared.pop_cmd("quit"): self.stop_flag.set(); break
                if self.shared.pop_cmd("start_calib"): self._start_new_calibration()
                if self.shared.pop_cmd("stop_calib"):  self._stop_calibration(save=False)
                if self.shared.pop_cmd("load_model"):
                    path=self.shared.pop_load_path()
                    if path:
                        try:
                            self.calib.load_linear_model(path)
                            with self.shared.lock: self.shared.status="Model loaded"
                            print(f"[Model] Loaded: {path}")
                        except Exception as e:
                            with self.shared.lock: self.shared.status="Model load failed"
                            print("Model load failed:", e)
                if self.shared.pop_cmd("toggle_overlay"):
                    with self.shared.lock: self.shared.overlay_enabled = not self.shared.overlay_enabled

                # 모드별 처리
                if self.calib.collecting:
                    target_px = self.calib.current_target_px()
                    ready = (time.time() - self.calib.start_t) >= float(self.calib.delay_sec)
                    finished=False
                    if gaze_feat is not None: finished = self.calib.feed(gaze_feat, t_now=time.time())
                    with self.shared.lock:
                        self.shared.calibrating=True; self.shared.calib_target=target_px; self.shared.calib_ready  = bool(ready)
                        self.shared.status=f"Calibration {self.calib.idx+1}/{self.calib.n_points()}"; self.shared.substatus="표시되는 원(고리)의 중심을 응시하세요"
                        self.shared.cross=None
                    if finished:
                        self.calib.save_model_pkl()
                        ds_path = self.calib.save_dataset_npz(DATA_DIR, {
                            "rows": self.calib.rows, "cols": self.calib.cols, "margin": float(self.calib.margin),
                            "per_point_sec": float(self.calib.per_point_sec), "delay_sec": float(self.calib.delay_sec), "camera_index": int(self.args.camera),
                            "mirror_preview": bool(self.args.mirror_preview),
                            "use_patches": bool(self.use_patches), "patch_w": int(self.patch_w), "patch_h": int(self.patch_h),
                            "patch_scale_w": float(self.patch_scale_w), "patch_scale_h": float(self.patch_scale_h),
                            "patch_norm": self.patch_norm or "none", "patch_clahe": bool(self.patch_clahe)
                        })
                        with self.shared.lock:
                            self.shared.calibrating=False; self.shared.calib_target=None; self.shared.calib_ready  = False
                            self.shared.status="Calibrated & saved data"; self.shared.substatus=f"Saved: {os.path.basename(ds_path)}"
                else:
                    px=None
                    if (gaze_feat is not None) and self.calib.has_model():
                        pred=np.array(self.calib.predict(gaze_feat), dtype=np.float32)
                        oe=self.oe.filter(pred, t=time.time())
                        if self.ema_last is None: self.ema_last=oe.copy()
                        else:
                            a=float(self.shared.ema_alpha)
                            self.ema_last = a*self.ema_last + (1.0-a)*oe
                        sm=self.ema_last; px=(int(sm[0]),int(sm[1]))
                    with self.shared.lock:
                        self.shared.calibrating=False; self.shared.calib_target=None
                        self.shared.cross=px; self.shared.status="Gaze Overlay"; self.shared.substatus="Use Control Panel"

                # 프리뷰 창(선택)
                if self.args.webcam_window:
                    disp=out
                    if self.args.mirror_preview: disp=cv2.flip(disp,1)
                    cv2.imshow('MediaPipe Face Mesh', disp)
                    k=cv2.waitKey(1)&0xFF
                    if k==27: self.stop_flag.set(); break
                    elif k==ord('c'): self.shared.set_cmd("start_calib")
                    elif k==ord('s'): self.shared.set_cmd("stop_calib")
                    elif k==ord('o'): self.shared.set_cmd("toggle_overlay")
                    elif k==ord('q'): self.shared.set_cmd("stop_calib")   # q로 캘리브 종료
                else:
                    time.sleep(0.001)

        cap.release(); cv2.destroyAllWindows()

    def stop(self): self.stop_flag.set()

# -------------------- 인자 --------------------
def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe FaceMesh + PyQt gaze overlay (12D + optional eye patches)")
    # 기존 인자
    p.add_argument("--grid", type=str, default="", help="예: '4,8' 또는 '4x8'")
    p.add_argument("--rows", type=int, default=0); p.add_argument("--cols", type=int, default=0)
    p.add_argument("--margin", type=float, default=0.03, help="그리드 외곽 여백")
    p.add_argument("--per_point", type=float, default=2.0, help="점당 응시 시간(초)")
    p.add_argument("--delay_time", type=float, default=0.5, help="포인트 이동 후 데이터 수집 지연(초)")
    p.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    p.add_argument("--webcam_window", action="store_true", default=True)
    p.add_argument("--no-webcam_window", dest="webcam_window", action="store_false")
    p.add_argument("--mirror_preview", dest="mirror_preview", action="store_true")
    p.add_argument("--no-mirror_preview", dest="mirror_preview", action="store_false")
    p.set_defaults(mirror_preview=True)
    p.add_argument("--ema_a", type=float, default=0.8)
    p.add_argument("--oe_mincutoff", type=float, default=0.20)
    p.add_argument("--oe_beta", type=float, default=0.003)
    p.add_argument("--oe_dcutoff", type=float, default=1.0)

    # MediaPipe
    p.add_argument("--mp_min_det", type=float, default=0.75, help="min_detection_confidence (0~1)")
    p.add_argument("--mp_min_track", type=float, default=0.75, help="min_tracking_confidence (0~1)")
    p.add_argument("--mp_refine_landmarks", action="store_true", default=True,
                   help="홍채/입술 정밀 랜드마크 사용(정확도↑, 연산↑)")

    # 카메라 입력 품질
    p.add_argument("--cam_w",   type=int, default=1920, help="캡처 가로 해상도")
    p.add_argument("--cam_h",   type=int, default=1080, help="캡처 세로 해상도")
    p.add_argument("--cam_fps", type=int, default=30, help="캡처 FPS")
    p.add_argument("--cam_fourcc", type=str, default="MJPG", help="코덱 FourCC (예: MJPG, YUYV)")

    # 패치 피처 옵션
    p.add_argument("--use_patches", action="store_true", default=True,
                   help="좌/우 눈 패치 벡터를 12D에 결합")
    p.add_argument("--patch_w", type=int, default=10, help="패치 가로(px) — 예: 10 (10x12=120D)")
    p.add_argument("--patch_h", type=int, default=12, help="패치 세로(px) — 예: 12")
    p.add_argument("--patch_scale_w", type=float, default=2.5, help="패치 반폭 = su * scale")
    p.add_argument("--patch_scale_h", type=float, default=2.0, help="패치 반높이 = sv * scale")
    p.add_argument("--patch_norm", type=str, default="z", choices=["z","none"], help="패치 벡터 정규화")
    p.add_argument("--patch_clahe", action="store_true", default=True, help="CLAHE 적용")

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

    init_rows, init_cols = 0, 0
    if args.grid:
        m = re.match(r'^\s*(\d+)\s*[,xX]\s*(\d+)\s*$', args.grid)
        if m: init_rows, init_cols = int(m.group(1)), int(m.group(2))
    if args.rows: init_rows = args.rows
    if args.cols: init_cols = args.cols
    if init_rows <= 0: init_rows = shared.calib_rows
    if init_cols <= 0: init_cols = shared.calib_cols
    with shared.lock:
        shared.calib_rows = init_rows; shared.calib_cols = init_cols
        shared.calib_per_point = float(args.per_point); shared.calib_delay_sec = float(args.delay_time); shared.calib_margin = float(args.margin)
        shared.ema_alpha = float(args.ema_a)
        shared.oe_mincutoff=float(args.oe_mincutoff); shared.oe_beta=float(args.oe_beta); shared.oe_dcutoff=float(args.oe_dcutoff)

    worker = GazeWorker(shared, args); worker.start()
    ret = app.exec_()
    worker.stop(); worker.join(timeout=1.0)
    sys.exit(ret)

if __name__ == "__main__":
    main()
