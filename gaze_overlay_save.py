# gaze_overlay_pyqt.py
# - 투명/클릭스루(PyQt) 오버레이 + 컨트롤 패널
# - 프리뷰는 기본 '셀피(좌우반전)'로 표시하되, L/R 텍스트는 비반전 상태로 정상 표기
# - 캘리브 종료 시 data/ 폴더에 학습용 데이터 .npz 자동 저장
# - "Load Model" 버튼으로 외부 학습 결과(.npz/.pkl; W,b) 로드
# - Ubuntu 22.04 / conda(PyQt5) 권장. Wayland에선 QT_QPA_PLATFORM=xcb 권장.

import os
import sys, time, argparse, re, threading, json
import numpy as np
import cv2
import mediapipe as mp
import pickle
from datetime import datetime

# X11/xcb 강제(이미 외부에서 설정했다면 그 값을 사용)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5 import QtCore, QtGui, QtWidgets

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
    "uL","vL","uR","vR",
    "uL2","vL2","uR2","vR2",
    "uL_vL","uR_vR","uL_uR","vL_vR"
]

# ---------- 수학 유틸 ----------
def _pca_axes(pts: np.ndarray):
    c = pts.mean(axis=0)
    X = pts - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ax1, ax2 = Vt[0], Vt[1]
    w = 2 * (np.sqrt(np.mean((X @ ax1)**2)) + 1e-6)
    h = 2 * (np.sqrt(np.mean((X @ ax2)**2)) + 1e-6)
    return c, ax1, ax2, w, h

def _iris_center(landmarks, idxs, W, H):
    pts = np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)

def _eye_uv(landmarks, eye_idxs, iris_idxs, W, H):
    eye_pts = np.array([(landmarks[i].x * W, landmarks[i].y * H) for i in eye_idxs], dtype=np.float32)
    c, ax1, ax2, w, h = _pca_axes(eye_pts)
    ic = _iris_center(landmarks, iris_idxs, W, H)
    u = float(np.dot(ic - c, ax1) / (w/2.0))
    v = float(np.dot(ic - c, ax2) / (h/2.0))
    return (u, v), ic

def _feat_vector(uL, vL, uR, vR):
    return np.array([
        uL, vL, uR, vR,
        uL*uL, vL*vL, uR*uR, vR*vR,
        uL*vL, uR*vR, uL*uR, vL*vR
    ], dtype=np.float32)

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
        I[-1, -1] = 0.0
        XtX = Xb.T @ Xb
        A = XtX + self.alpha * I
        Wb = np.linalg.pinv(A) @ (Xb.T @ Y)
        self.W = Wb[:-1, :].T
        self.b = Wb[-1, :].T

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (self.W @ X.T).T + self.b

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

    def reset(self):
        self.idx = 0
        self.collecting = False
        self.samples_X, self.samples_Y = [], []
        self.start_t = None

    def n_points(self): return len(self.points_norm)

    def current_target_px(self):
        nx, ny = self.points_norm[self.idx]
        return int(nx * self.sw), int(ny * self.sh)

    def begin(self):
        self.reset()
        self.collecting = True
        self.start_t = time.time()

    def feed(self, feat):
        if not self.collecting: return False
        tx, ty = self.current_target_px()
        self.samples_X.append(np.array(feat, dtype=np.float32))
        self.samples_Y.append(np.array([tx, ty], dtype=np.float32))
        if (time.time() - self.start_t) >= self.per_point_sec:
            self.idx += 1
            self.start_t = time.time()
            if self.idx >= len(self.points_norm):
                # 학습(선형) + 종료
                self.model.fit(np.stack(self.samples_X), np.stack(self.samples_Y))
                self.collecting = False
                return True
        return False

    def has_model(self):  return (self.model.W is not None)

    def predict(self, feat):
        y = self.model.predict(np.array([feat], dtype=np.float32))[0]
        x = int(np.clip(y[0], 0, self.sw - 1))
        y = int(np.clip(y[1], 0, self.sh - 1))
        return x, y

    def save_model_pkl(self, path="calib_gaze.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"W": self.model.W, "b": self.model.b, "screen": (self.sw, self.sh)}, f)

    def save_dataset_npz(self, out_dir="data", meta_extra=None):
        """
        캘리브로 모은 (X,Y) 샘플을 .npz로 저장. 경로 반환.
        파일: data/gaze_samples_YYYYmmdd_HHMMSS.npz
          - X: float32 [N, D] (D=len(FEATURE_NAMES))
          - Y: float32 [N, 2]  (화면 픽셀)
          - feature_names: object [D]
          - screen: int32 [2] (w,h)
          - meta: json(str)
        """
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"gaze_samples_{ts}.npz")
        X = np.stack(self.samples_X).astype(np.float32)
        Y = np.stack(self.samples_Y).astype(np.float32)
        meta = {
            "rows": self.rows,
            "cols": self.cols,
            "margin": float(self.margin),
            "per_point_sec": float(self.per_point_sec),
            "n_points": int(len(self.points_norm)),
            "timestamp": ts
        }
        if meta_extra: meta.update(meta_extra)
        np.savez(path,
                 X=X, Y=Y,
                 feature_names=np.array(FEATURE_NAMES, dtype=object),
                 screen=np.array([self.sw, self.sh], dtype=np.int32),
                 meta=json.dumps(meta))
        return path

    def load_linear_model(self, path):
        """
        외부 학습 결과 로드 (지원 포맷)
          - .npz : keys 'W','b' (float)
          - .pkl : dict {'W':..., 'b':...}
        """
        if path.lower().endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            if "W" in d and "b" in d:
                self.model.W = d["W"].astype(np.float32)
                self.model.b = d["b"].astype(np.float32)
                return True
            else:
                raise ValueError("npz에 'W','b' 키가 필요합니다.")
        elif path.lower().endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model.W = np.asarray(data["W"], dtype=np.float32)
            self.model.b = np.asarray(data["b"], dtype=np.float32)
            return True
        else:
            raise ValueError("지원하지 않는 포맷입니다. (.npz, .pkl)")

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

# ---------- PyQt 오버레이(클릭스루) ----------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, app: QtWidgets.QApplication):
        super().__init__(None)
        self.shared = shared
        self.app = app

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
        # if self.cross is not None:
        #     x, y = self.cross
        #     pen = QtGui.QPen(QtGui.QColor(255,255,0,230), 2); p.setPen(pen)
        #     size = 16; p.drawLine(x - size, y, x + size, y); p.drawLine(x, y - size, x, y + size)
        
        if self.cross is not None:
            x, y = self.cross
            # 빨간색 원형 "고리"
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 230), 4)  # 색/두께
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)                      # 속이 빈 원
            radius = 14                                       # 반지름(px) — 취향껏 조절
            p.drawEllipse(QtCore.QPointF(x, y), radius, radius)        
                
        

# ---------- 컨트롤 패널 ----------
class ControlPanel(QtWidgets.QWidget):
    def __init__(self, shared: SharedState):
        super().__init__()
        self.shared = shared
        self.setWindowTitle("Gaze Control")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setFixedSize(300, 200)

        layout = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Ready"); layout.addWidget(self.lbl)

        btns = QtWidgets.QGridLayout(); layout.addLayout(btns)
        b_calib = QtWidgets.QPushButton("Start Calibration")
        b_load  = QtWidgets.QPushButton("Load Model")
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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model file", "",
            "Model files (*.npz *.pkl);;All files (*)"
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

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
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
                    (uL, vL), _ = _eye_uv(lms, LEFT_EYE_IDXS, LEFT_IRIS_IDXS, W, H)
                    (uR, vR), _ = _eye_uv(lms, RIGHT_EYE_IDXS, RIGHT_IRIS_IDXS, W, H)
                    gaze_feat = _feat_vector(uL, vL, uR, vR)

                    # 메쉬는 원본 프레임에 그린다 → 이후 프리뷰에서 필요시 좌우반전
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
                            calib.load_linear_model(path)
                            print(f"[Model] Loaded: {path}")
                            with self.shared.lock:
                                self.shared.status = f"Model loaded"
                        except Exception as e:
                            print("Model load failed:", e)
                            with self.shared.lock:
                                self.shared.status = f"Model load failed"

                if self.shared.pop_cmd("toggle_overlay"):
                    with self.shared.lock: self.shared.overlay_enabled = not self.shared.overlay_enabled
                if self.shared.pop_cmd("toggle_fullscreen"):
                    with self.shared.lock: self.shared.fullscreen = not self.shared.fullscreen

                # ---- 캘리브/런타임 ----
                if calib.collecting:
                    target_px = calib.current_target_px()
                    finished = False
                    if gaze_feat is not None: finished = calib.feed(gaze_feat)

                    with self.shared.lock:
                        self.shared.calibrating = True
                        self.shared.calib_target = target_px
                        self.shared.status = f"Calibration {calib.idx+1}/{calib.n_points()}"
                        self.shared.substatus = "오렌지 점을 응시하세요 (자동 진행)"
                        self.shared.cross = None

                    if finished:
                        # 1) 선형 모델 백업(pkl)
                        calib.save_model_pkl("calib_gaze.pkl")
                        # 2) 데이터셋 저장(npz)
                        meta_extra = {
                            "grid": self.args.grid,
                            "rows": rows,
                            "cols": cols,
                            "margin": float(self.args.margin),
                            "camera_index": int(self.args.camera),
                            "mirror_preview": bool(self.args.mirror_preview)
                        }
                        ds_path = calib.save_dataset_npz(out_dir="data", meta_extra=meta_extra)
                        print(f"[Data] Saved dataset: {ds_path}")

                        with self.shared.lock:
                            self.shared.calibrating = False
                            self.shared.calib_target = None
                            self.shared.status = f"Calibrated & saved data"
                            self.shared.substatus = f"Saved: {os.path.basename(ds_path)}"
                else:
                    px = None
                    if gaze_feat is not None and calib.has_model():
                        pred = calib.predict(gaze_feat)
                        if not hasattr(self, "_last"): self._last = np.array(pred, dtype=np.float32)
                        self._last = smoother.filter(np.array(pred, dtype=np.float32))
                        px = (int(self._last[0]), int(self._last[1]))
                    with self.shared.lock:
                        self.shared.calibrating = False
                        self.shared.calib_target = None
                        self.shared.cross = px
                        self.shared.status = "Gaze Overlay"
                        self.shared.substatus = "Use Control Panel"

                # ---- OpenCV 창 표시 ----
                if self.args.webcam_window:
                    disp = frame_out
                    if self.args.mirror_preview:
                        disp = cv2.flip(disp, 1)  # 얼굴은 셀피로
                    # 텍스트는 프리뷰 후 그리기(글자 거울반전 방지)
                    if uL is not None and uR is not None:
                        txt = f"L({uL:+.2f},{vL:+.2f}) R({uR:+.2f},{vR:+.2f})"
                        cv2.putText(disp, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    else:
                        cv2.putText(disp, "No face", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    cv2.imshow('MediaPipe Face Mesh', disp)
                    # 키 입력은 유지(원하면 여기 삭제 가능)
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
    p = argparse.ArgumentParser(description="MediaPipe FaceMesh + PyQt transparent click-through gaze overlay")
    p.add_argument("--grid", type=str, default="", help="예: '4,8' 또는 '4x8' → 4행×8열 그리드. 생략 시 5점")
    p.add_argument("--rows", type=int, default=0, help="--grid 대신 직접 행 지정")
    p.add_argument("--cols", type=int, default=0, help="--grid 대신 직접 열 지정")
    p.add_argument("--margin", type=float, default=0.10, help="그리드 외곽 여백(0.0~0.45)")
    p.add_argument("--per_point", type=float, default=0.9, help="점당 응시 시간(초)")
    p.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    p.add_argument("--webcam_window", action="store_true", default=True, help="(기본) OpenCV 창 사용")
    p.add_argument("--no-webcam_window", dest="webcam_window", action="store_false", help="OpenCV 창 끄기")

    # 기본값: 프리뷰 미러 ON (필요하면 --no-mirror_preview)
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
