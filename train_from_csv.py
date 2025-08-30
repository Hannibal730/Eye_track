#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gaze_dl.py  (cleaning steps 2,4,5,6 removed)

CSV(uL.., Y_x, Y_y, [T], [pt_index], [screen_w], [screen_h])를 읽어
- 정제: NaN/Inf 제거 + (선택) 분위수 기반 클리핑만 적용
- 분할: random / grid / time
- 모델: PyTorch MLP(이분산; mu_x, mu_y, logvar_x, logvar_y)
- 저장: 안전 체크포인트(.pt) + 메타(.json)

예)
python train_gaze_dl.py --csv data/gaze_samples_*.csv --val grid --qclip 0.005 --epochs 180 --out models_mlp
"""

import os, argparse, glob, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

FEATURE_COLS_12D = [
    "uL","vL","uR","vR","uL2","vL2","uR2","vR2","uL_vL","uR_vR","uL_uR","vL_vR"
]

def load_csv(paths):
    if isinstance(paths, str): paths = [paths]
    dfs = []
    for p in paths:
        for fp in glob.glob(p):
            dfs.append(pd.read_csv(fp))
    if not dfs:
        raise FileNotFoundError("CSV를 찾지 못했습니다.")
    df = pd.concat(dfs, ignore_index=True)

    # 피처 자동 감지
    feat_cols = [c for c in FEATURE_COLS_12D if c in df.columns]
    if not feat_cols:
        feat_cols = [c for c in df.columns if c.startswith("X_")]
    if not feat_cols:
        raise ValueError("피처 컬럼(uL.. 또는 X_*)이 없습니다.")
    if not {"Y_x","Y_y"}.issubset(df.columns):
        raise ValueError("Y_x, Y_y 컬럼이 필요합니다.")

    # 숫자 변환 + NaN/Inf 제거
    for c in feat_cols + ["Y_x","Y_y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols+["Y_x","Y_y"])
    dropped_nan = before - len(df)

    screen = None
    if {"screen_w","screen_h"}.issubset(df.columns):
        screen = [int(df["screen_w"].iloc[0]), int(df["screen_h"].iloc[0])]

    info = {"feat_cols": feat_cols, "screen": screen, "dropped_nan": int(dropped_nan)}
    return df, info

# (남기는 유일한 추가 정제) 분위수 기반 클리핑
def clip_by_quantile(df, cols, q=0.005):
    if q <= 0.0: return df, 0
    before = len(df)
    low, high = q, 1.0 - q
    for c in cols:
        ql, qh = df[c].quantile(low), df[c].quantile(high)
        df = df[(df[c] >= ql) & (df[c] <= qh)]
    return df, before - len(df)

def split_data(df, feat_cols, val_mode="random", test_size=0.2):
    N = len(df); idx = np.arange(N)
    rng = np.random.default_rng(SEED)

    if val_mode == "grid" and "pt_index" in df.columns:
        groups = df["pt_index"].values.astype(int)
        uniq = np.unique(groups)
        n_va = max(1, int(round(len(uniq)*test_size)))
        va_groups = set(rng.choice(uniq, size=n_va, replace=False))
        va_mask = np.array([g in va_groups for g in groups], bool)
        tr_idx, va_idx = idx[~va_mask], idx[va_mask]
    elif val_mode == "time" and "T" in df.columns:
        df_sorted = df.sort_values("T").reset_index(drop=True)
        cut = int(round((1.0 - test_size) * len(df_sorted)))
        tr_idx = df_sorted.index[:cut].to_numpy()
        va_idx = df_sorted.index[cut:].to_numpy()
    else:
        rng.shuffle(idx); n_va = int(round(test_size*N))
        va_idx, tr_idx = idx[:n_va], idx[n_va:]

    X = df[feat_cols].values.astype(np.float32)
    Y = df[["Y_x","Y_y"]].values.astype(np.float32)
    return X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx]

class NDArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, mu=None, st=None):
        self.X = X; self.Y = Y
        if mu is None: mu = X.mean(axis=0, keepdims=True)
        if st is None:
            st = X.std(axis=0, keepdims=True); st[st<1e-6] = 1.0
        self.mu = mu.astype(np.float32); self.st = st.astype(np.float32)
        self.Xn = (X - self.mu) / self.st
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.Xn[i]), torch.from_numpy(self.Y[i])

class MLP(nn.Module):
    def __init__(self, d, hidden=(128,128), dropout=0.1):
        super().__init__()
        layers=[]; in_dim=d
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(in_dim, 4)  # [mu_x, mu_y, logvar_x, logvar_y]
    def forward(self, x):
        o = self.head(self.backbone(x))
        mu, logv = o[:, :2], o[:, 2:].clamp(-8.0, 2.0)
        return mu, logv

def nll_gauss(mu, logv, y):
    invv = torch.exp(-logv)
    return 0.5 * ((y - mu)**2 * invv + logv).sum(dim=1).mean()

@torch.no_grad()
def rmse_px(mu, y):
    return torch.sqrt(((mu - y)**2).sum(dim=1)).mean().item()

def as_np32(x):
    if isinstance(x, np.ndarray): return x.astype(np.float32)
    return x.detach().cpu().to(dtype=torch.float32).numpy()

def train(args):
    set_seed(SEED)
    df, info = load_csv(args.csv)
    feat_cols = info["feat_cols"]

    if args.qclip > 0:
        df, k = clip_by_quantile(df, feat_cols + ["Y_x","Y_y"], q=args.qclip)
        print(f"[clean] quantile clip q={args.qclip} removed {k} rows")
    print(f"[info] rows after cleaning: {len(df)} (NaN/Inf dropped: {info['dropped_nan']})")

    Xtr, Ytr, Xva, Yva = split_data(df, feat_cols, val_mode=args.val, test_size=args.test_size)

    mu = Xtr.mean(axis=0, keepdims=True)
    st = Xtr.std(axis=0, keepdims=True); st[st<1e-6] = 1.0
    tr_set = NDArrayDataset(Xtr, Ytr, mu, st)
    va_set = NDArrayDataset(Xva, Yva, mu, st)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_set, batch_size=2048, shuffle=False)

    D = len(feat_cols)
    model = MLP(D, tuple(args.hidden), args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"nll": 1e12, "rmse": 1e12, "state": None}
    for ep in range(1, args.epochs+1):
        model.train(); tot = 0.0; nobs=0
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            mu_t, logv_t = model(xb)
            loss = nll_gauss(mu_t, logv_t, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)*len(xb); nobs += len(xb)
        tr_nll = tot/max(1,nobs)

        model.eval(); va_nll=0.0; mcnt=0; mu_all=[]; y_all=[]
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                mu_t, logv_t = model(xb)
                va_nll += float(nll_gauss(mu_t, logv_t, yb))*len(xb)
                mcnt += len(xb); mu_all.append(mu_t.cpu()); y_all.append(yb.cpu())
        va_nll /= max(1,mcnt)
        mu_cat = torch.cat(mu_all, dim=0); y_cat = torch.cat(y_all, dim=0)
        va_rmse = rmse_px(mu_cat, y_cat)
        print(f"[{ep:03d}] train NLL {tr_nll:.4f} | val NLL {va_nll:.4f} | val RMSE(px) {va_rmse:.2f}")

        if va_nll < best["nll"] - 1e-5:
            best["nll"] = va_nll
            best["state"] = {k: v.detach().cpu() for k,v in model.state_dict().items()}
        if va_rmse < best["rmse"]:
            best["rmse"] = va_rmse

    os.makedirs(args.out, exist_ok=True)
    if best["state"] is None:
        best["state"] = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    ckpt = {
        "model_state": best["state"],
        "in_dim": int(D),
        "hidden": list(args.hidden),
        "dropout": float(args.dropout),
        "x_mean": as_np32(mu).ravel().tolist(),
        "x_std":  as_np32(st).ravel().tolist(),
        "val_nll": float(best["nll"]),
        "val_rmse_px": float(best["rmse"]),
        "feat_cols": list(feat_cols),
        "screen": info["screen"],
        "arch": "MLP(mu,logvar)"
    }
    torch.save(ckpt, os.path.join(args.out, "mlp_12d.pt"))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"val_nll": ckpt["val_nll"], "val_rmse_px": ckpt["val_rmse_px"],
                   "feat_cols": ckpt["feat_cols"], "screen": ckpt["screen"]},
                  f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved {args.out}/mlp_12d.pt (val RMSE {best['rmse']:.2f}px)")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, default="models/mlp")
    p.add_argument("--val", type=str, default="random", choices=["random","grid","time"])
    p.add_argument("--test_size", type=float, default=0.2)
    # 정제(남기는 건 분위수 클리핑만)
    p.add_argument("--qclip", type=float, default=0.0, help="분위수 양끝 제거 비율(예: 0.005=0.5%)")
    # 모델
    p.add_argument("--hidden", type=int, nargs="*", default=[128,128, 64, 32])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
