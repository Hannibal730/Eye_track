# a_with_gaze.py
# Ubuntu 22.04 + Python + MediaPipe FaceMesh
# - 기존 Face Mesh 웹캠 데모 유지
# - 캘리브레이션: 사용자 지정 '행,열' 그리드 또는 5점 패턴
# - 실시간 시선→화면 픽셀 좌표 추정 + 십자선 표시

import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
import argparse, re

# 화면 해상도 얻기 (tkinter 없으면 기본값)
try:
    import tkinter as tk
    _root = tk.Tk(); _root.withdraw()
    SCREEN_W = _root.winfo_screenwidth()
    SCREEN_H = _root.winfo_screenheight()
    _root.destroy()
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080  # 필요 시 수정

# --- MediaPipe aliases (원본 유지) ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# --- 랜드마크 인덱스 ---
LEFT_IRIS_IDXS  = [474, 475, 476, 477]
RIGHT_IRIS_IDXS = [469, 470, 471, 472]

def _unique_idxs(connections):
    s = set()
    for a,b in connections:
        s.add(a); s.add(b)
    return sorted(list(s))

LEFT_EYE_IDXS  = _unique_idxs(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDXS = _unique_idxs(mp_face_mesh.FACEMESH_RIGHT_EYE)

# --- 눈 로컬 좌표계 계산 ---
def _pca_axes(pts: np.ndarray):
    """
    pts: (N,2) pixel 좌표
    return: 중심 c, 단위 주성분 축 ax1/ax2, 가로/세로 스케일 w/h(약 2*표준편차)
    """
    c = pts.mean(axis=0)
    X = pts - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ax1, ax2 = Vt[0], Vt[1]  # unit vectors
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
    return (u, v), ic, c, (ax1, ax2)

def _feat_vector(uL, vL, uR, vR):
    # 간단한 2차 특징 (바이어스는 회귀에서 처리)
    return np.array([
        uL, vL, uR, vR,
        uL*uL, vL*vL, uR*uR, vR*vR,
        uL*vL, uR*vR, uL*uR, vL*vR
    ], dtype=np.float32)

# --- 아주 가벼운 Ridge 회귀 (x,y 동시) ---
class Ridge2D:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None  # (2, D)
        self.b = None  # (2,)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)  # (N, D)
        Y = np.asarray(Y, dtype=np.float32)  # (N, 2)
        N, D = X.shape
        Xb = np.hstack([X, np.ones((N,1), dtype=np.float32)])  # bias 확장
        I = np.eye(D+1, dtype=np.float32)
        I[-1,-1] = 0.0  # bias 항은 L2 패널티 제외
        XtX = Xb.T @ Xb
        A = XtX + self.alpha * I
        Wb = np.linalg.pinv(A) @ (Xb.T @ Y)  # (D+1,2)
        self.W = Wb[:-1,:].T  # (2,D)
        self.b = Wb[-1,:].T   # (2,)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (self.W @ X.T).T + self.b

# --- 캘리브레이션 그리드 생성 ---
def make_grid_points(rows:int, cols:int, margin:float=0.1, order:str="serpentine"):
    """
    rows x cols 그리드를 [0..1] 정규좌표로 생성. 화면 모서리를 피하기 위해 margin 사용.
    order = 'serpentine'이면 지그재그 스캔(눈의 이동 폭 최소화).
    """
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
    return points  # list of (nx, ny)

# --- 5점 기본 패턴 ---
FIVE_POINTS = [(0.5,0.5), (0.15,0.15), (0.85,0.15), (0.85,0.85), (0.15,0.85)]

# ---  캘리브레이션 관리자 ---
class Calibrator:
    def __init__(self, screen_w, screen_h, rows=0, cols=0, margin=0.10, per_point_sec=0.9):
        self.sw, self.sh = screen_w, screen_h
        self.per_point_sec = per_point_sec
        if rows and cols:
            self.points_norm = make_grid_points(rows, cols, margin, order="serpentine")
        else:
            self.points_norm = FIVE_POINTS[:]  # 기본 5점
        self.idx = 0
        self.collecting = False
        self.samples_X, self.samples_Y = [], []
        self.start_t = None
        self.model = Ridge2D(alpha=10.0)

    def n_points(self): return len(self.points_norm)

    def current_target_px(self):
        nx, ny = self.points_norm[self.idx]
        return int(nx * self.sw), int(ny * self.sh)

    def begin(self):
        self.idx = 0
        self.collecting = True
        self.samples_X.clear(); self.samples_Y.clear()
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
                self.model.fit(np.stack(self.samples_X), np.stack(self.samples_Y))
                self.collecting = False
                return True
        return False

    def is_running(self): return self.collecting
    def has_model(self):  return (self.model.W is not None)

    def save(self, path="calib_gaze.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"W": self.model.W, "b": self.model.b, "screen": (self.sw, self.sh)}, f)

    def load(self, path="calib_gaze.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model.W = data["W"]; self.model.b = data["b"]
            scr = data.get("screen", None)
            if scr: self.sw, self.sh = scr
        return True

    def predict(self, feat):
        y = self.model.predict(np.array([feat], dtype=np.float32))[0]
        x = int(np.clip(y[0], 0, self.sw-1))
        y = int(np.clip(y[1], 0, self.sh-1))
        return x, y

# --- 간단한 One-Euro 필터(2D) ---
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

# --- Overlay 창(OpenCV) ---
class Overlay:
    def __init__(self, screen_w, screen_h):
        self.sw, self.sh = screen_w, screen_h
        self.enabled = True
        self.fullscreen = False
        cv2.namedWindow("Gaze Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Overlay", min(self.sw, 960), min(self.sh, 540))
        cv2.moveWindow("Gaze Overlay", 50, 50)

    def toggle(self): self.enabled = not self.enabled
    def set_fullscreen(self, fs: bool):
        self.fullscreen = fs
        cv2.setWindowProperty("Gaze Overlay", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fs else cv2.WINDOW_NORMAL)

    def show(self, cross=None, calib_target=None, text=None, subtext=None):
        if not self.enabled: return
        canvas = np.zeros((self.sh, self.sw, 3), dtype=np.uint8)
        if cross is not None:
            x, y = cross
            cv2.drawMarker(canvas, (x, y), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        if calib_target is not None:
            tx, ty = calib_target
            cv2.circle(canvas, (tx, ty), 16, (0,128,255), thickness=3)
        if text:
            cv2.putText(canvas, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
        if subtext:
            cv2.putText(canvas, subtext, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 2)
        cv2.imshow("Gaze Overlay", canvas)

# --- (원본) 정적 이미지 데모(유지, 기본 비활성) ---
def run_static_images_demo():
    IMAGE_FILES = []
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# --- (확장) 웹캠 + 시선 캘리브레이션 + 화면 십자선 표시 ---
def run_webcam_with_gaze(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open webcam."); return

    overlay = Overlay(SCREEN_W, SCREEN_H)
    rows, cols = args.rows, args.cols
    if args.grid:
        m = re.match(r'^\s*(\d+)\s*[,xX]\s*(\d+)\s*$', args.grid)
        if m:
            rows, cols = int(m.group(1)), int(m.group(2))
        else:
            print("[Warn] --grid 형식이 잘못되었습니다. 예: --grid 4,8 또는 4x8")
    calib = Calibrator(SCREEN_W, SCREEN_H, rows=rows, cols=cols,
                       margin=args.margin, per_point_sec=args.per_point)
    smoother = OneEuro(mincutoff=0.4, beta=0.008, dcutoff=1.0)
    last_pred = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            H, W = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            frame_out = image.copy()

            gaze_feat = None
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                (uL, vL), iL, cL, axesL = _eye_uv(lms, LEFT_EYE_IDXS, LEFT_IRIS_IDXS, W, H)
                (uR, vR), iR, cR, axesR = _eye_uv(lms, RIGHT_EYE_IDXS, RIGHT_IRIS_IDXS, W, H)
                gaze_feat = _feat_vector(uL, vL, uR, vR)

                # (원본 유지) 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image=frame_out,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame_out,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame_out,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                # 디버그 텍스트
                cv2.putText(frame_out, f"L({uL:+.2f},{vL:+.2f}) R({uR:+.2f},{vR:+.2f})",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # 캘리브레이션 중
            if calib.is_running():
                target_px = calib.current_target_px()
                overlay.set_fullscreen(True)  # 점이 분명히 보이도록 전체화면
                prog = f"Calibration {calib.idx+1}/{calib.n_points()}"
                hint = "오렌지 점을 응시하세요 (자동 진행)"
                overlay.show(cross=None, calib_target=target_px, text=prog, subtext=hint)
                if gaze_feat is not None:
                    finished = calib.feed(gaze_feat)
                    if finished:
                        calib.save("calib_gaze.pkl")
                        print(f"[Calib] Finished ({calib.n_points()} points) and saved to calib_gaze.pkl")
            else:
                # 일반 런타임
                overlay.set_fullscreen(False)
                if gaze_feat is not None and calib.has_model():
                    px = calib.predict(gaze_feat)
                    if last_pred is None:
                        last_pred = np.array(px, dtype=np.float32)
                    else:
                        last_pred = smoother.filter(np.array(px, dtype=np.float32))
                    overlay.show(cross=(int(last_pred[0]), int(last_pred[1])),
                                 text="Gaze Overlay", subtext="c: 캘리브 | l: 로드 | o: 토글 | f: 전체화면")
                else:
                    overlay.show(text="Gaze Overlay (미보정)", subtext="c: 캘리브 시작")

            # (원본) 셀피 뷰
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame_out, 1))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                break
            elif key == ord('c'):
                print("[Calib] Starting calibration... Look at the orange dot.")
                calib.begin()
            elif key == ord('l'):
                try:
                    calib.load("calib_gaze.pkl")
                    print("[Calib] Loaded calib_gaze.pkl")
                except Exception as e:
                    print("Load failed:", e)
            elif key == ord('o'):
                overlay.toggle()
            elif key == ord('f'):
                overlay.set_fullscreen(not overlay.fullscreen)

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe FaceMesh + Gaze-to-Screen with configurable calibration grid")
    p.add_argument("--grid", type=str, default="",
                   help="예: '4,8' 또는 '4x8' → 4행×8열 그리드. 생략 시 5점 패턴")
    p.add_argument("--rows", type=int, default=0, help="--grid 대신 직접 행 지정")
    p.add_argument("--cols", type=int, default=0, help="--grid 대신 직접 열 지정")
    p.add_argument("--margin", type=float, default=0.10, help="그리드 외곽 여백(0.0~0.45), 기본 0.10")
    p.add_argument("--per_point", type=float, default=0.9, help="점당 응시 시간(초)")
    p.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # (원본) 정적 이미지 데모는 기본 비활성화
    # run_static_images_demo()
    run_webcam_with_gaze(args)
