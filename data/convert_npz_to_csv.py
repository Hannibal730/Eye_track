#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_npz_to_csv.py
- gaze_overlay_12d.py/21d.py가 저장한 NPZ( X, Y, T, pt_index, screen, meta )를
  같은 경로에 CSV(+메타 JSON)로 변환
- 여러 파일을 한 번에 처리하려면 --glob 사용
"""

import os, glob, argparse, json
import numpy as np

# pandas가 있으면 사용(권장), 없으면 표준 csv로 기록
try:
    import pandas as pd
    _USE_PANDAS = True
except Exception:
    import csv
    _USE_PANDAS = False

def npz_to_dataframe_dict(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    # --- 필드 읽기(없으면 건너뜀) ---
    X   = d["X"] if "X" in d else None          # (N,D)
    Y   = d["Y"] if "Y" in d else None          # (N,2)
    T   = d["T"] if "T" in d else None          # (N,)
    IDX = d["pt_index"] if "pt_index" in d else None  # (N,)
    feat_names = d["feature_names"].tolist() if "feature_names" in d else None
    screen = d["screen"].tolist() if "screen" in d else None
    meta_raw = d["meta"].item() if "meta" in d else "{}"

    # --- 행 개수 결정 ---
    N = None
    if Y is not None:
        N = len(Y)
    elif X is not None:
        N = len(X)
    else:
        raise ValueError(f"{os.path.basename(npz_path)}: 'Y'나 'X'가 필요합니다.")

    # --- 딕셔너리로 데이터 구성 ---
    data = {}
    if X is not None:
        X = np.asarray(X)
        D = X.shape[1]
        if feat_names is not None and len(feat_names) == D:
            cols = [str(c) for c in feat_names]
        else:
            cols = [f"X_{i}" for i in range(D)]
        for j, name in enumerate(cols):
            data[name] = X[:, j].astype(np.float32)

    if Y is not None:
        Y = np.asarray(Y)
        data["Y_x"] = Y[:, 0].astype(np.float32)
        data["Y_y"] = Y[:, 1].astype(np.float32)

    if T is not None:
        data["T"] = np.asarray(T, dtype=np.float64)

    if IDX is not None:
        data["pt_index"] = np.asarray(IDX, dtype=np.int32)

    if screen is not None and isinstance(screen, (list, tuple)) and len(screen) == 2:
        data["screen_w"] = np.repeat(int(screen[0]), N)
        data["screen_h"] = np.repeat(int(screen[1]), N)

    # --- 메타 정리 ---
    try:
        meta_obj = json.loads(meta_raw) if isinstance(meta_raw, (str, bytes)) else {}
    except Exception:
        meta_obj = {"raw_meta": str(meta_raw)}

    info = {
        "source_npz": os.path.basename(npz_path),
        "num_rows": int(N),
        "columns": list(data.keys()),
        "feature_names": feat_names,
        "screen": screen,
        "meta": meta_obj,
    }
    return data, info

def write_csv(out_path, data):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    headers = list(data.keys())
    N = len(next(iter(data.values()))) if data else 0

    if _USE_PANDAS:
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({k: np.asarray(v) for k, v in data.items()})
        df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        # 표준 라이브러리 csv 사용
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i in range(N):
                row = [data[h][i] for h in headers]
                writer.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="*.npz",
                    help="변환할 NPZ 글롭 패턴 (기본: 현재 폴더 *.npz)")
    ap.add_argument("--outdir", type=str, default="",
                    help="출력 폴더(기본: 입력 파일과 같은 폴더)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    for p in paths:
        data, info = npz_to_dataframe_dict(p)
        base = os.path.splitext(os.path.basename(p))[0]
        out_dir = args.outdir or os.path.dirname(p)
        csv_path  = os.path.join(out_dir, base + ".csv")
        json_path = os.path.join(out_dir, base + "_meta.json")

        write_csv(csv_path, data)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[OK] {os.path.basename(p)} -> {os.path.basename(csv_path)} "
              f"({info['num_rows']} rows)")

if __name__ == "__main__":
    main()
