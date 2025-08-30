import numpy as np, json
d = np.load("data/gaze_samples_20250830_180039.npz", allow_pickle=True)
T = d["T"]               # (N,) 수집 시각(초)
IDX = d["pt_index"]      # (N,) 캘리브 타깃 인덱스
X = d["X"]; Y = d["Y"]   # (N,21), (N,2)


# 전체 샘플링 레이트(중간 끊김 제외한 중앙값 기준)
dt = np.diff(T)
eff_fps = 1.0 / np.median(dt[dt>0])
print("effective FPS ~", round(eff_fps, 2), "Hz")
# effective FPS ~ 14.89 Hz


# 점(타깃)별 샘플 수
unique, counts = np.unique(IDX, return_counts=True)
print("per-point samples:", dict(zip(unique.tolist(), counts.tolist())))

# 대충 예상치와 비교: len(unique) * per_point * eff_fps
# per-point samples: {0: 17, 1: 15, 2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 15}