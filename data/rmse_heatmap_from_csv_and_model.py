#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmse_heatmap_from_csv_and_model.py
- gaze CSV(uL,vL,uR,vR, 파생 피처, Y_x,Y_y, pt_index, ...)와
  선형 모델(.pkl/.npz)을 받아 pt_index별 RMSE(px) 히트맵을 만든다.

지원 모델 포맷
1) pkl: {"W": (2,D), "b": (2,), ...}  # Ridge2D 저장 포맷
2) npz: (a) W(2,D), b(2,)   또는  (b) W(13,2)  # 12D + bias(1) 합친 형태
"""

import argparse, os, re, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATS = ["uL","vL","uR","vR","uL2","vL2","uR2","vR2","uL_vL","uR_vR","uL_uR","vL_vR"]

def infer_rows_cols(csv_path, n_unique_pts):
    m = re.search(r'_(\d+)\s*[xX]\s*(\d+)_', os.path.basename(csv_path))
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        if r*c == n_unique_pts:
            return r, c
    # fallback: 약수로 근사
    N = int(n_unique_pts)
    best = None
    for r in range(1, int(np.sqrt(N))+1):
        if N % r == 0:
            c = N // r
            score = abs(c - r)
            if best is None or score < best[0]:
                best = (score, r, c)
    return (best[1], best[2]) if best else (1, N)

def inv_serpentine(idx, cols):
    r = idx // cols
    c_in_row = idx % cols
    if r % 2 == 0:
        c = c_in_row
    else:
        c = cols - 1 - c_in_row
    return r, c

def load_linear_model(model_path):
    """
    반환: pred_fn(X)  # X: (N,12) -> (N,2)
    지원:
      - pkl: dict에 "W"(2,D), "b"(2,)  (D=12)
      - npz: ("W"(2,D) & "b"(2,)) 또는 "W"(13,2) 단독(마지막이 bias)
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pkl":
        with open(model_path, "rb") as f:
            d = pickle.load(f)
        W = np.array(d["W"], dtype=np.float32)  # (2,D)
        b = np.array(d["b"], dtype=np.float32)  # (2,)
        assert W.ndim == 2 and b.ndim == 1 and W.shape[0] == 2, "Unexpected W/b shape in pkl"
        D = W.shape[1]
        def _pred(X):
            X = np.asarray(X, dtype=np.float32)  # (N,D)
            return (X @ W.T) + b  # (N,2)
        return _pred

    elif ext == ".npz":
        d = np.load(model_path, allow_pickle=True)
        if "W" in d and "b" in d:  # (2,D) + (2,)
            W = np.array(d["W"], dtype=np.float32)
            b = np.array(d["b"], dtype=np.float32)
            assert W.shape[0] == 2 and b.shape[0] == 2
            def _pred(X):
                X = np.asarray(X, dtype=np.float32)
                return (X @ W.T) + b
            return _pred
        elif "W" in d and "b" not in d:
            W = np.array(d["W"])
            # 케이스: 합쳐진 가중치(13×2): [12D | 1(bias)]
            if W.ndim == 2 and W.shape[0] in (12,13) and W.shape[1] == 2:
                if W.shape[0] == 12:
                    W12 = W.astype(np.float32)   # (12,2)
                    def _pred(X):
                        X = np.asarray(X, dtype=np.float32)
                        return X @ W12           # bias가 없다면 그대로
                    return _pred
                else:
                    W13 = W.astype(np.float32)   # (13,2)
                    def _pred(X):
                        X = np.asarray(X, dtype=np.float32)
                        ones = np.ones((X.shape[0],1), dtype=np.float32)
                        Xb = np.hstack([X, ones])  # (N,13)
                        return Xb @ W13           # (N,2)
                    return _pred
        raise ValueError("npz 모델에서 지원되는 키 조합(W/b)을 찾지 못했습니다.")
    else:
        raise ValueError("지원 확장자: .pkl, .npz")

def build_X_from_df(df):
    # CSV에 파생항이 이미 있으면 그대로 사용, 없으면 계산
    need = set(FEATS)
    has = set(df.columns)
    X = pd.DataFrame(index=df.index)

    # 기본 4개
    for k in ["uL","vL","uR","vR"]:
        if k not in df.columns:
            raise ValueError(f"CSV에 {k} 열이 필요합니다.")
        X[k] = df[k].astype(np.float32)

    # 파생항: 있으면 사용, 없으면 계산
    X["uL2"]   = df["uL"].astype(np.float32)**2 if "uL2" not in has else df["uL2"].astype(np.float32)
    X["vL2"]   = df["vL"].astype(np.float32)**2 if "vL2" not in has else df["vL2"].astype(np.float32)
    X["uR2"]   = df["uR"].astype(np.float32)**2 if "uR2" not in has else df["uR2"].astype(np.float32)
    X["vR2"]   = df["vR"].astype(np.float32)**2 if "vR2" not in has else df["vR2"].astype(np.float32)
    X["uL_vL"] = (df["uL"]*df["vL"]).astype(np.float32) if "uL_vL" not in has else df["uL_vL"].astype(np.float32)
    X["uR_vR"] = (df["uR"]*df["vR"]).astype(np.float32) if "uR_vR" not in has else df["uR_vR"].astype(np.float32)
    X["uL_uR"] = (df["uL"]*df["uR"]).astype(np.float32) if "uL_uR" not in has else df["uL_uR"].astype(np.float32)
    X["vL_vR"] = (df["vL"]*df["vR"]).astype(np.float32) if "vL_vR" not in has else df["vL_vR"].astype(np.float32)

    # 순서 맞추기
    X = X[FEATS].astype(np.float32).values  # (N,12)
    return X

def heatmap(array2d, title, out_png, cmap="viridis", vmin=None, vmax=None):
    H = array2d
    cm = plt.cm.get_cmap(cmap).copy()
    cm.set_bad(color="#dddddd")  # NaN 회색

    plt.figure(figsize=(max(6, H.shape[1]*0.4), max(5, H.shape[0]*0.4)))
    im = plt.imshow(H, origin="upper", interpolation="nearest", cmap=cm, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("col"); plt.ylabel("row")
    plt.grid(color="white", linestyle="-", linewidth=0.5, which="both")
    plt.xticks(np.arange(H.shape[1])); plt.yticks(np.arange(H.shape[0]))
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[saved] {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, help=".pkl 또는 .npz 선형 모델")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--outdir", default="gaze_model_eval")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    for col in ["Y_x","Y_y","pt_index"]:
        if col not in df.columns:
            raise ValueError(f"CSV에 {col} 열이 필요합니다.")
    y_true = df[["Y_x","Y_y"]].values.astype(np.float32)
    pt = df["pt_index"].astype(int).values

    # 특징행렬
    X = build_X_from_df(df)

    # 모델 로드 & 예측
    pred_fn = load_linear_model(args.model)
    y_pred = pred_fn(X).astype(np.float32)  # (N,2)

    # 에러
    dx = y_pred[:,0] - y_true[:,0]
    dy = y_pred[:,1] - y_true[:,1]
    err = np.sqrt(dx*dx + dy*dy)  # per-sample RMSE(=거리)

    # pt별 집계
    n_unique = int(np.unique(pt).size)
    rows, cols = (args.rows, args.cols)
    if rows is None or cols is None:
        rows, cols = infer_rows_cols(args.csv, n_unique)

    # 준비
    H_rmse  = np.full((rows, cols), np.nan, dtype=np.float32)
    H_bx    = np.full((rows, cols), np.nan, dtype=np.float32)  # mean bias x
    H_by    = np.full((rows, cols), np.nan, dtype=np.float32)  # mean bias y
    H_count = np.zeros((rows, cols), dtype=np.int32)

    # 집계 루프
    for pid in np.unique(pt):
        mask = (pt == pid)
        if not np.any(mask): continue
        r, c = inv_serpentine(int(pid), cols)
        ex = dx[mask]; ey = dy[mask]; ed = err[mask]
        rmse = float(np.sqrt(np.mean(ex*ex + ey*ey)))  # = sqrt(mean(d^2))
        bx = float(np.mean(ex)); by = float(np.mean(ey))
        H_rmse[r, c]  = rmse
        H_bx[r, c]    = bx
        H_by[r, c]    = by
        H_count[r, c] = int(mask.sum())

    # 저장: 요약 CSV
    out_rows = []
    for r in range(rows):
        for c in range(cols):
            # serpentine 순서의 pt_index도 같이 저장
            ridx = r * cols + (c if (r % 2 == 0) else (cols - 1 - c))
            out_rows.append({
                "row": r, "col": c, "pt_index": ridx,
                "rmse_px": float(H_rmse[r,c]) if np.isfinite(H_rmse[r,c]) else np.nan,
                "bias_x": float(H_bx[r,c]) if np.isfinite(H_bx[r,c]) else np.nan,
                "bias_y": float(H_by[r,c]) if np.isfinite(H_by[r,c]) else np.nan,
                "count": int(H_count[r,c]),
            })
    pd.DataFrame(out_rows).to_csv(os.path.join(args.outdir, "model_eval_summary.csv"), index=False)
    print(f"[saved] {os.path.join(args.outdir, 'model_eval_summary.csv')}")

    # 히트맵들
    heatmap(H_rmse,  "RMSE (px) — lower is better",   os.path.join(args.outdir, "heat_rmse.png"))
    heatmap(H_bx,    "Mean bias X (px)",             os.path.join(args.outdir, "heat_bias_x.png"), cmap="coolwarm")
    heatmap(H_by,    "Mean bias Y (px)",             os.path.join(args.outdir, "heat_bias_y.png"), cmap="coolwarm")
    heatmap(H_count, "Sample count per point",       os.path.join(args.outdir, "heat_count.png"))

    print("Done.")

if __name__ == "__main__":
    main()
