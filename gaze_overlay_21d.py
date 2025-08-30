# gaze_overlay_21d_conf.py
# - 21D 특징 기반 실시간 시선추정 + PyQt 오버레이
# - "Load Model"에서 .pt(MLP) 또는 .npz/.pkl(선형) 로드 지원
# - 토치 모델 사용 시 빨간 링 옆에 confidence 표시
# - 저장/로드 경로는 data/

import os
import sys, time, argparse, re, threading, json, math
import numpy as np
import cv2
import mediapipe as mp
import pickle
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
from PyQt5 import QtCore, QtGui, QtWidgets

# -------------------- 저장 경로 상수 --------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------- MediaPipe aliases ----------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS_IDXS  = [474, 475, 476, 477]
RIGHT_IRIS_IDXS = [469, 470, 471, 472]

def _unique_idxs(connections):
    s = set()
    for a, b in connections:
        s.add(a); s.add(b)
    return sorted(list(s))

LEFT_EYE_IDXS  = _unique_idxs(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDXS = _unique_idxs(mp_face_mesh.FACEMESH_RIGHT_EYE)

FEATURE_NAMES = [
    # base 12
    "uL","vL","uR","vR",
    "uL2","vL2","uR2","vR2",
    "uL_vL","uR_vR","uL_uR","vL_vR",
    # +9 extra
    "roll_sin","roll_cos",
    "io_norm",
    "earL","earR",
    "irisRL","irisRR",
    "face_w_norm","face_h_norm"
]

# ---------- 수학/기하 유틸 ----------
def _pca_axes(pts: np.ndarray):
    c = pts.mean(axis=0)
    X = pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    ax1, ax2 = Vt[0], Vt[1]
    w = 2 * (np.sqrt(np.mean((X @ ax1)**2)) + 1e-6)
    h = 2 * (np.sqrt(np.mean((X @ ax2)**2)) + 1e-6)
    return c, ax1, ax2, w, h

def _landmarks_xy(landmarks, idxs, W, H):
    return np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in idxs], dtype=np.float32)

def _eye_metrics(landmarks, eye_idxs, iris_idxs, W, H):
    eye_pts  = _landmarks_xy(landmarks, eye_idxs, W, H)
    iris_pts = _landmarks_xy(landmarks, iris_idxs, W, H)
    c, ax1, ax2, w, h = _pca_axes(eye_pts)
    ic = iris_pts.mean(axis=0)
    u = float(np.dot(ic - c, ax1) / (w/2.0))
    v = float(np.dot(ic - c, ax2) / (h/2.0))
    iris_r = float(np.mean(np.linalg.norm(iris_pts - ic, axis=1)))
    denom = 0.25 * (w + h) + 1e-6
    iris_r_norm = float(iris_r / denom)
    ear = float(h / (w + 1e-6))
    ang = float(np.arctan2(ax1[1], ax1[0]))
    return {
        "u": u, "v": v, "c": c, "ax1": ax1, "ax2": ax2, "w": float(w), "h": float(h),
        "iris_r_norm": iris_r_norm, "ear": ear, "ang": ang
    }

def _compose_features(mL, mR, face_w, face_h, frame_W, frame_H):
    uL, vL, uR, vR = mL["u"], mL["v"], mR["u"], mR["v"]
    base12 = [
        uL, vL, uR, vR,
        uL*uL, vL*vL, uR*uR, vR*vR,
        uL*vL, uR*vR, uL*uR, vL*vR
    ]
    ax = mL["ax1"] + mR["ax1"]
    n  = np.linalg.norm(ax) + 1e-6
    ux = ax / n
    roll_sin = float(ux[1])
    roll_cos = float(ux[0])
    io = float(np.linalg.norm(mR["c"] - mL["c"]))
    io_norm = float(io / max(face_w, 1.0))
    earL, earR = mL["ear"], mR["ear"]
    irisRL, irisRR = mL["iris_r_norm"], mR["iris_r_norm"]
    face_w_norm = float(face_w / max(frame_W, 1.0))
    face_h_norm = float(face_h / max(frame_H, 1.0))
    extras = [roll_sin, roll_cos, io_norm, earL, earR, irisRL, irisRR, face_w_norm, face_h_norm]
    return np.array(base12 + extras, dtype=np.float32)

class Ridge2D:
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.W = None
        self.b = None
        self.in_dim = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N, D = X.shape
        self.in_dim = D
        Xb = np.hstack([X, np.ones((N,1), dtype=np.float64)])
        I = np.eye(D+1, dtype=np.float64); I[-1, -1] = 0.0
        XtX = Xb.T @ Xb
        A = XtX + self.alpha * I
        Wb = np.linalg.pinv(A) @ (Xb.T @ Y)
        self.W = Wb[:-1, :].T.astype(np.float32)
        self.b = Wb[-1, :].T.astype(np.float32)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.W is None: raise RuntimeError("Linear model not ready")
        return (self.W @ X.T).T + self.b

# --------- Torch Predictor (optional) ----------
_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
except Exception as e:
    _TORCH_AVAILABLE = False

class TorchPredictor:
    """
    PyTorch MLP(mu, logvar) 로드/추론
    학습 스크립트(train_gaze_mlp.py)와 동일 구조
    """
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=(64,64), dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(dropout)]
                last = h
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(last, 4)

        def forward(self, x):
            h = self.backbone(x)
            out = self.head(h)
            mu = out[:, :2]
            log_var = out[:, 2:].clamp_(-8.0, 2.0)
            return mu, log_var

    def __init__(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        self.device = torch.device("cpu")
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.in_dim = None
        self.target_norm = "per_screen"

    @staticmethod
    def load(path:str):
        obj = TorchPredictor()
        ckpt = torch.load(path, map_location="cpu")
        in_dim = int(ckpt["in_dim"])
        hidden = tuple(int(x) for x in ckpt.get("hidden", (64,64)))
        dropout = float(ckpt.get("dropout", 0.1))
        obj.model = TorchPredictor.MLP(in_dim, hidden, dropout)
        obj.model.load_state_dict(ckpt["model_state"], strict=True)
        obj.model.eval()
        obj.in_dim = in_dim
        obj.x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
        obj.x_std  = torch.tensor(ckpt["x_std"], dtype=torch.float32)
        obj.target_norm = ckpt.get("target_norm", "per_screen")
        return obj

    @torch.no_grad()
    def predict(self, feat21: np.ndarray, screen_w:int, screen_h:int):
        """
        입력: feat21 (D,) float32
        출력: (x_pix, y_pix, conf[0..1])
        """
        x = torch.tensor(feat21, dtype=torch.float32).unsqueeze(0)
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Torch model expects dim={self.in_dim}, got {x.shape[1]}")
        x = (x - self.x_mean) / self.x_std
        mu, log_var = self.model(x)
        mu = mu[0].cpu().numpy()
        var = np.exp(log_var[0].cpu().numpy())

        if self.target_norm == "per_screen":
            x_pix = float(mu[0] * screen_w)
            y_pix = float(mu[1] * screen_h)
            # 표준편차를 픽셀로 환산
            std_x_pix = float(math.sqrt(var[0]) * screen_w)
            std_y_pix = float(math.sqrt(var[1]) * screen_h)
        else:
            # 픽셀 타깃으로 학습된 경우(현재 스크립트는 쓰지 않음)
            x_pix = float(mu[0]); y_pix = float(mu[1])
            std_x_pix = float(math.sqrt(var[0]))
            std_y_pix = float(math.sqrt(var[1]))

        # 신뢰도 계산: 화면 대각선 대비 예측 표준편차의 상대 크기
        diag = math.sqrt(screen_w**2 + screen_h**2)
        r_std = math.sqrt(std_x_pix**2 + std_y_pix**2)  # 방사 표준편차(픽셀)
        # 8% 대각선에 해당하는 표준편차면 conf≈e^{-1}~0.37
        scale = 0.08 * diag + 1e-6
        conf = math.exp(-(r_std/scale)**2)
        conf = max(0.0, min(1.0, conf))
        return int(np.clip(x_pix, 0, screen_w-1)), int(np.clip(y_pix, 0, screen_h-1)), conf

# ---------- 캘리브 포인트 ----------
def make_grid_points(rows:int, cols:int, margin:float=0.10, order:str="serpentine"):
    rows = max(1, rows); cols = max(1, cols)
    margin = max(0.0, min(0.45, margin))
    xs = np.linspace(margin, 1.0 - margin, cols)
    ys = np.linspace(margin, 1.0 - margin, rows)
    points = []
    for r, y in enumerate(ys):
        row = [(x, y) for x in xs]
        if order == "serpentine" and (r % 2 == 1):
            row = row[::-1]
        points.extend(row)
    return points

FIVE_POINTS = [(0.5,0.5), (0.15,0.15), (0.85,0.15), (0.85,0.85), (0.15,0.85)]

class Calibrator:
    def __init__(self, screen_w, screen_h, rows=0, cols=0, margin=0.10, per_point_sec=0.9):
        self.sw, self.sh = screen_w, screen_h
        self.rows, self.cols, self.margin = rows, cols, margin
        if rows and cols:
            self.points_norm = make_grid_points(rows, cols, margin, "serpentine")
        else:
            self.points_norm = FIVE_POINTS[:]
        self.per_point_sec = per_point_sec
        self.reset()
        self.model = Ridge2D(alpha=10.0)
        self.torch_model = None  # TorchPredictor

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
        self.reset()
        self.collecting = True
        self.start_t = time.time()

    def feed(self, feat, t_now=None):
        if not self.collecting: return False
        tx, ty = self.current_target_px()
        self.samples_X.append(np.array(feat, dtype=np.float32))
        self.samples_Y.append(np.array([tx, ty], dtype=np.float32))
        self.samples_T.append(float(time.time() if t_now is None else t_now))
        self.samples_IDX.append(int(self.idx))
        if (time.time() - self.start_t) >= self.per_point_sec:
            self.idx += 1
            self.start_t = time.time()
            if self.idx >= len(self.points_norm):
                self.model.fit(np.stack(self.samples_X), np.stack(self.samples_Y))
                self.collecting = False
                return True
        return False

    def has_linear(self):  return (self.model.W is not None)
    def has_torch(self):   return (self.torch_model is not None)

    def predict_linear(self, feat):
        y = self.model.predict(np.array([feat], dtype=np.float32))[0]
        x = int(np.clip(y[0], 0, self.sw - 1))
        y = int(np.clip(y[1], 0, self.sh - 1))
        return x, y

    def load_linear_model(self, path):
        if path.lower().endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            if "W" in d and "b" in d:
                self.model.W = d["W"].astype(np.float32)
                self.model.b = d["b"].astype(np.float32)
                self.model.in_dim = self.model.W.shape[1]
                return True
            else:
                raise ValueError("npz에 'W','b' 키가 필요합니다.")
        elif path.lower().endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model.W = np.asarray(data["W"], dtype=np.float32)
            self.model.b = np.asarray(data["b"], dtype=np.float32)
            self.model.in_dim = self.model.W.shape[1]
            return True
        else:
            raise ValueError("지원하지 않는 포맷입니다. (.npz, .pkl)")

    def load_torch(self, path):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 미설치 상태입니다.")
        self.torch_model = TorchPredictor.load(path)
        return True

    def save_model_pkl(self, path=None):
        if path is None:
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, "calib_gaze.pkl")
        with open(path, "wb") as f:
            pickle.dump({"W": self.model.W, "b": self.model.b, "screen": (self.sw, self.sh)}, f)

    def save_dataset_npz(self, out_dir=DATA_DIR, meta_extra=None):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"gaze_samples_{ts}.npz")
        X = np.stack(self.samples_X).astype(np.float32)
        Y = np.stack(self.samples_Y).astype(np.float32)
        # T는 float64로 저장(정밀도 보장)
        T = np.array(self.samples_T, dtype=np.float64)
        IDX = np.array(self.samples_IDX, dtype=np.int32)
        meta = {
            "rows": self.rows, "cols": self.cols, "margin": float(self.margin),
            "per_point_sec": float(self.per_point_sec), "n_points": int(len(self.points_norm)),
            "timestamp": ts
        }
        if meta_extra: meta.update(meta_extra)
        np.savez(path,
                 X=X, Y=Y, T=T, pt_index=IDX,
                 feature_names=np.array(FEATURE_NAMES, dtype=object),
                 screen=np.array([self.sw, self.sh], dtype=np.int32),
                 meta=json.dumps(meta))
        return path

class OneEuro:
    def __init__(self, freq=60.0, mincutoff=0.4, beta=0.008, dcutoff=1.0):
        self.freq=freq; self.mincutoff=mincutoff; self.beta=beta; self.dcutoff=dcutoff
        self.x_prev=None; self.dx_prev=None; self.t_prev=None

    def _alpha(self, cutoff):
        tau = 1.0 / (2*np.pi*cutoff)
        te  = 1.0 / self.freq
        return 1.0 / (1.0 + tau/te)

    def filter(self, x, t=None):
        if self.t_prev is None: self.t_prev = time.time()
        if t is None: t = time.time()
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0 / dt
        if self.x_prev is None:
            self.x_prev = x.copy(); self.dx_prev = np.zeros_like(x); self.t_prev = t
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        cutoff = self.mincutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev=x_hat; self.dx_prev=dx_hat; self.t_prev=t
        return x_hat

# ---------- 공유 상태 + 커맨드 ----------
class SharedState:
    def __init__(self, sw:int, sh:int):
        self.lock = threading.Lock()
        self.screen_w = sw
        self.screen_h = sh
        self.cross = None
        self.confidence = None       # ★ 신뢰도 공유
        self.calib_target = None
        self.calibrating = False
        self.overlay_enabled = True
        self.fullscreen = False
        self.status = "Gaze Overlay"
        self.substatus = "Use Control Panel"

        self.cmd = {
            "start_calib": False,
            "load_model": False,
            "toggle_overlay": False,
            "toggle_fullscreen": False,
            "quit": False
        }
        self._load_path = None  # 모델 경로

    def set_cmd(self, name):
        with self.lock:
            if name in self.cmd: self.cmd[name] = True

    def pop_cmd(self, name):
        with self.lock:
            v = self.cmd.get(name, False)
            self.cmd[name] = False
            return v

    def set_load_model(self, path:str):
        with self.lock:
            self._load_path = path
            self.cmd["load_model"] = True

    def pop_load_path(self):
        with self.lock:
            p = self._load_path
            self._load_path = None
            return p

# ---------- PyQt 오버레이 ----------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, app: QtWidgets.QApplication):
        super().__init__(None)
        self.shared = shared
        self.app = app

        # ★ paintEvent 첫 호출 전에 기본 필드 세팅 (초기화 순서 안정)
        self.status = "Gaze Overlay"
        self.substatus = "Use Control Panel"
        self.cross = None
        self.calib_target = None
        self.confidence = None

        flags = (
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint  |
            QtCore.Qt.Tool |
            QtCore.Qt.WindowDoesNotAcceptFocus
        )
        if hasattr(QtCore.Qt, "WindowTransparentForInput"):
            flags |= QtCore.Qt.WindowTransparentForInput
        if hasattr(QtCore.Qt, "X11BypassWindowManagerHint"):
            flags |= QtCore.Qt.X11BypassWindowManagerHint

        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        scr = self.app.primaryScreen().geometry()
        self.setGeometry(scr)

        self._last_fullscreen = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self.show()

    def tick(self):
        with self.shared.lock:
            self.cross = self.shared.cross
            self.confidence = self.shared.confidence
            self.calib_target = self.shared.calib_target
            enabled = self.shared.overlay_enabled
            fs = self.shared.fullscreen
            self.status = self.shared.status
            self.substatus = self.shared.substatus

        if enabled:
            if not self.isVisible():
                self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
                self.show()
        else:
            if self.isVisible(): self.hide()

        if fs != self._last_fullscreen:
            state = self.windowState()
            if fs and not (state & QtCore.Qt.WindowFullScreen):
                self.setWindowState(state | QtCore.Qt.WindowFullScreen)
            elif not fs and (state & QtCore.Qt.WindowFullScreen):
                self.setWindowState(state & ~QtCore.Qt.WindowFullScreen)
            self._last_fullscreen = fs

        if enabled: self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self.status:
            pen = QtGui.QPen(QtGui.QColor(200,200,200,230), 2); p.setPen(pen)
            p.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold)); p.drawText(30, 50, self.status)
        if self.substatus:
            pen = QtGui.QPen(QtGui.QColor(160,160,160,200), 1); p.setPen(pen)
            p.setFont(QtGui.QFont("Arial", 12)); p.drawText(30, 80, self.substatus)
        if self.calib_target is not None:
            tx, ty = self.calib_target
            pen = QtGui.QPen(QtGui.QColor(255,165,0,240), 4); p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush); p.drawEllipse(QtCore.QPointF(tx, ty), 16, 16)
        if self.cross is not None:
            x, y = self.cross
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 230), 4)
            p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
            radius = 14
            p.drawEllipse(QtCore.QPointF(x, y), radius, radius)
            # ★ confidence 텍스트
            if self.confidence is not None:
                p.setPen(QtGui.QPen(QtGui.QColor(255,255,255,220), 1))
                p.setFont(QtGui.QFont("Arial", 12))
                txt = f"conf {int(self.confidence*100):d}%"
                p.drawText(x + radius + 8, y - radius - 6, txt)

# ---------- 컨트롤 패널 ----------
class ControlPanel(QtWidgets.QWidget):
    def __init__(self, shared: SharedState):
        super().__init__()
        self.shared = shared
        self.setWindowTitle("Gaze Control")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setFixedSize(320, 210)

        layout = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Ready"); layout.addWidget(self.lbl)

        btns = QtWidgets.QGridLayout(); layout.addLayout(btns)
        b_calib = QtWidgets.QPushButton("Start Calibration")
        b_load  = QtWidgets.QPushButton("Load Model (.pt/.npz/.pkl)")
        b_ov    = QtWidgets.QPushButton("Toggle Overlay")
        b_fs    = QtWidgets.QPushButton("Fullscreen")
        b_quit  = QtWidgets.QPushButton("Quit")
        btns.addWidget(b_calib, 0,0); btns.addWidget(b_load, 0,1)
        btns.addWidget(b_ov,    1,0); btns.addWidget(b_fs,   1,1)
        btns.addWidget(b_quit,  2,0,1,2)

        b_calib.clicked.connect(lambda: self.shared.set_cmd("start_calib"))
        b_load.clicked.connect(self._choose_and_load_model)
        b_ov.clicked.connect(lambda: self.shared.set_cmd("toggle_overlay"))
        b_fs.clicked.connect(lambda: self.shared.set_cmd("toggle_fullscreen"))
        b_quit.clicked.connect(lambda: self.shared.set_cmd("quit"))

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(200)
        self.show()

    def _choose_and_load_model(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model file", DATA_DIR,
            "Model files (*.pt *.npz *.pkl);;All files (*)"
        )
        if fname:
            self.shared.set_load_model(fname)

    def tick(self):
        with self.shared.lock:
            txt = self.shared.status
        self.lbl.setText(txt)

# ---------- 시선 워커 ----------
class GazeWorker(threading.Thread):
    def __init__(self, shared: SharedState, args):
        super().__init__(daemon=True)
        self.shared = shared
        self.args = args
        self.stop_flag = threading.Event()
        self.use_torch = False  # 어떤 모델을 사용할지 표시

    def run(self):
        # 카메라 오픈 (V4L2 우선)
        cap = cv2.VideoCapture(self.args.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.args.camera, cv2.CAP_ANY)
            if not cap.isOpened():
                print("Could not open webcam."); return

        rows, cols = self.args.rows, self.args.cols
        if self.args.grid:
            m = re.match(r'^\s*(\d+)\s*[,xX]\s*(\d+)\s*$', self.args.grid)
            if m: rows, cols = int(m.group(1)), int(m.group(2))
            else: print("[Warn] --grid 형식 예: 4,8 또는 4x8")

        calib = Calibrator(self.shared.screen_w, self.shared.screen_h,
                           rows=rows, cols=cols, margin=self.args.margin,
                           per_point_sec=self.args.per_point)
        smoother = OneEuro(mincutoff=0.4, beta=0.008, dcutoff=1.0)

        with mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

            uL = vL = uR = vR = None

            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok: continue
                H, W = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
                res = face_mesh.process(rgb)
                frame_out = frame.copy()

                gaze_feat = None
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    xs = np.array([p.x for p in lms], dtype=np.float32) * W
                    ys = np.array([p.y for p in lms], dtype=np.float32) * H
                    face_w = float(xs.max() - xs.min())
                    face_h = float(ys.max() - ys.min())

                    mL = _eye_metrics(lms, LEFT_EYE_IDXS, LEFT_IRIS_IDXS, W, H)
                    mR = _eye_metrics(lms, RIGHT_EYE_IDXS, RIGHT_IRIS_IDXS, W, H)
                    uL, vL, uR, vR = mL["u"], mL["v"], mR["u"], mR["v"]
                    gaze_feat = _compose_features(mL, mR, face_w, face_h, W, H)

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

                # ---- 커맨드 처리 ----
                if self.shared.pop_cmd("quit"): self.stop_flag.set(); break

                if self.shared.pop_cmd("start_calib"):
                    calib.begin()

                if self.shared.pop_cmd("load_model"):
                    path = self.shared.pop_load_path()
                    if path:
                        try:
                            if path.lower().endswith(".pt"):
                                calib.load_torch(path)
                                self.use_torch = True
                                msg = f"Loaded Torch model"
                            else:
                                calib.load_linear_model(path)
                                self.use_torch = False
                                msg = f"Loaded Linear model"
                            print(f"[Model] {msg}: {path}")
                            with self.shared.lock:
                                self.shared.status = msg
                        except Exception as e:
                            print("Model load failed:", e)
                            with self.shared.lock:
                                self.shared.status = "Model load failed"

                if self.shared.pop_cmd("toggle_overlay"):
                    with self.shared.lock: self.shared.overlay_enabled = not self.shared.overlay_enabled
                if self.shared.pop_cmd("toggle_fullscreen"):
                    with self.shared.lock: self.shared.fullscreen = not self.shared.fullscreen

                # ---- 캘리브/런타임 ----
                now_t = time.time()
                if calib.collecting:
                    target_px = calib.current_target_px()
                    finished = False
                    if gaze_feat is not None: finished = calib.feed(gaze_feat, t_now=now_t)

                    with self.shared.lock:
                        self.shared.calibrating = True
                        self.shared.calib_target = target_px
                        self.shared.status = f"Calibration {calib.idx+1}/{calib.n_points()}"
                        self.shared.substatus = "오렌지 점을 응시하세요 (자동 진행)"
                        self.shared.cross = None
                        self.shared.confidence = None

                    if finished:
                        calib.save_model_pkl()
                        meta_extra = {
                            "grid": self.args.grid, "rows": rows, "cols": cols,
                            "margin": float(self.args.margin),
                            "camera_index": int(self.args.camera),
                            "mirror_preview": bool(self.args.mirror_preview)
                        }
                        ds_path = calib.save_dataset_npz(out_dir=DATA_DIR, meta_extra=meta_extra)
                        print(f"[Data] Saved dataset: {ds_path}")

                        with self.shared.lock:
                            self.shared.calibrating = False
                            self.shared.calib_target = None
                            self.shared.status = "Calibrated & saved data"
                            self.shared.substatus = f"Saved: {os.path.basename(ds_path)}"
                else:
                    px = None; conf = None
                    if gaze_feat is not None:
                        try:
                            if self.use_torch and calib.has_torch():
                                x, y, conf = calib.torch_model.predict(gaze_feat, self.shared.screen_w, self.shared.screen_h)
                                px = (x, y)
                            elif calib.has_linear():
                                pred = calib.predict_linear(gaze_feat)
                                if not hasattr(self, "_last"): self._last = np.array(pred, dtype=np.float32)
                                self._last = OneEuro().filter(np.array(pred, dtype=np.float32), t=now_t)
                                px = (int(self._last[0]), int(self._last[1]))
                                conf = None  # 선형 모델은 신뢰도 추정 없음
                        except ValueError as e:
                            with self.shared.lock:
                                self.shared.status = "Model/feature dim mismatch"
                                self.shared.substatus = str(e)

                    with self.shared.lock:
                        self.shared.calibrating = False
                        self.shared.calib_target = None
                        self.shared.cross = px
                        self.shared.confidence = conf
                        self.shared.status = "Gaze Overlay"
                        self.shared.substatus = "Use Control Panel"

                # ---- OpenCV 창 표시 ----
                if self.args.webcam_window:
                    disp = frame_out
                    if self.args.mirror_preview:
                        disp = cv2.flip(disp, 1)
                    if uL is not None and uR is not None:
                        txt = f"L({uL:+.2f},{vL:+.2f}) R({uR:+.2f},{vR:+.2f})"
                        cv2.putText(disp, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    else:
                        cv2.putText(disp, "No face", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.imshow('MediaPipe Face Mesh', disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: self.stop_flag.set(); break
                    elif key == ord('c'): self.shared.set_cmd("start_calib")
                    elif key == ord('o'): self.shared.set_cmd("toggle_overlay")
                    elif key == ord('f'): self.shared.set_cmd("toggle_fullscreen")
                else:
                    time.sleep(0.001)

        cap.release()
        cv2.destroyAllWindows()

    def stop(self): self.stop_flag.set()

# ---------- 인자 ----------
def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe FaceMesh + PyQt overlay with Torch confidence")
    p.add_argument("--grid", type=str, default="", help="예: '4,8' 또는 '4x8' → 4행×8열 그리드. 생략 시 5점")
    p.add_argument("--rows", type=int, default=0, help="--grid 대신 직접 행 지정")
    p.add_argument("--cols", type=int, default=0, help="--grid 대신 직접 열 지정")
    p.add_argument("--margin", type=float, default=0.03, help="그리드 외곽 여백(0.0~0.45)")
    p.add_argument("--per_point", type=float, default=0.9, help="점당 응시 시간(초)")
    p.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    p.add_argument("--webcam_window", action="store_true", default=True, help="(기본) OpenCV 창 사용")
    p.add_argument("--no-webcam_window", dest="webcam_window", action="store_false", help="OpenCV 창 끄기")
    p.add_argument("--mirror_preview", dest="mirror_preview", action="store_true",
                   help="프리뷰를 셀피(좌우반전)로 표시 (기본 ON)")
    p.add_argument("--no-mirror_preview", dest="mirror_preview", action="store_false",
                   help="프리뷰 좌우반전 끄기")
    p.set_defaults(mirror_preview=True)
    return p.parse_args()

# ---------- 메인 ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen().geometry()
    SCREEN_W, SCREEN_H = screen.width(), screen.height()

    shared = SharedState(SCREEN_W, SCREEN_H)
    overlay = OverlayWindow(shared, app)
    panel = ControlPanel(shared)

    args = parse_args()

    worker = GazeWorker(shared, args); worker.start()
    ret = app.exec_()

    worker.stop(); worker.join(timeout=1.0)
    sys.exit(ret)

if __name__ == "__main__":
    main()
