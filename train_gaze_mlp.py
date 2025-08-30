#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gaze_mlp.py
- data/gaze_samples_*.npz (21D 특징)을 모아서 PyTorch MLP로 학습
- 타깃: 화면 정규화 좌표 y_norm = [x/sw, y/sh] (0~1)
- 손실: Diagonal Gaussian NLL → 모델이 μ, logσ²를 출력 (불확실도 예측)
- 결과 저장: data/gaze_mlp.pt
"""

import os, glob, json, math, argparse, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_GLOB = os.path.join(DATA_DIR, "gaze_samples_*.npz")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_npz_list(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files

def stack_npz(files):
    Xs, Ys, SWs, SHs = [], [], [], []
    dims = None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        X = d["X"]; Y = d["Y"]; screen = d["screen"]
        # 스크린은 (w,h)
        sw, sh = int(screen[0]), int(screen[1])

        # 특징 차원 확인(21D 기대). 다르면 스킵(구 12D 파일 등)
        if dims is None: dims = X.shape[1]
        if X.shape[1] != dims:
            print(f"[Warn] Skip {os.path.basename(fp)}: dim {X.shape[1]} != {dims}")
            continue

        # 유효 샘플만 추가
        if len(X) == 0: 
            print(f"[Warn] Empty: {fp}"); 
            continue
        Xs.append(X.astype(np.float32))
        Ys.append(Y.astype(np.float32))
        SWs.append(np.full((len(X),), sw, dtype=np.float32))
        SHs.append(np.full((len(X),), sh, dtype=np.float32))

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    SW = np.concatenate(SWs, axis=0)
    SH = np.concatenate(SHs, axis=0)
    return X, Y, SW, SH, dims

class GazeDataset(Dataset):
    def __init__(self, X, Y, SW, SH, x_mean=None, x_std=None):
        # 타깃을 화면 정규화
        Yn = np.empty_like(Y, dtype=np.float32)
        Yn[:,0] = Y[:,0] / SW
        Yn[:,1] = Y[:,1] / SH

        self.X = X.astype(np.float32)
        self.Yn = Yn
        self.SW = SW.astype(np.float32)
        self.SH = SH.astype(np.float32)

        # 특징 표준화 통계
        if x_mean is None or x_std is None:
            self.x_mean = self.X.mean(axis=0)
            self.x_std  = self.X.std(axis=0) + 1e-6
        else:
            self.x_mean = x_mean.astype(np.float32)
            self.x_std  = x_std.astype(np.float32)

        # 적용
        self.X = (self.X - self.x_mean) / self.x_std

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.Yn[i], self.SW[i], self.SH[i])

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64,64), dropout=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(dropout)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, 4)  # [mu_x, mu_y, log_var_x, log_var_y]

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        mu = out[:, :2]
        log_var = out[:, 2:]
        # 안정성: log_var를 적당히 클램프
        log_var = torch.clamp(log_var, min=-8.0, max=2.0)
        return mu, log_var

def nll_gaussian(mu, log_var, target):
    # diag Gaussian NLL
    var = torch.exp(log_var)
    return 0.5 * ( ((target - mu)**2 / var) + log_var ).sum(dim=1).mean()

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for X, Yn, _, _ in loader:
        X = X.to(device); Yn = Yn.to(device)
        mu, log_var = model(X)
        loss = nll_gaussian(mu, log_var, Yn)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    # RMSE(픽셀) 계산을 위해 SW/SH 필요
    total_rmse = 0.0
    total_n = 0
    for X, Yn, SW, SH in loader:
        X = X.to(device); Yn = Yn.to(device)
        mu, log_var = model(X)
        loss = nll_gaussian(mu, log_var, Yn)
        total_loss += loss.item() * len(X)

        # 픽셀 단위 RMSE
        mu_pix_x = (mu[:,0].cpu().numpy() * SW.numpy())
        mu_pix_y = (mu[:,1].cpu().numpy() * SH.numpy())
        tgt_pix_x = (Yn[:,0].cpu().numpy() * SW.numpy())
        tgt_pix_y = (Yn[:,1].cpu().numpy() * SH.numpy())
        err = np.sqrt((mu_pix_x - tgt_pix_x)**2 + (mu_pix_y - tgt_pix_y)**2)
        total_rmse += err.sum()
        total_n += len(err)

    return total_loss / len(loader.dataset), total_rmse / max(1,total_n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default=DEFAULT_GLOB, help="npz pattern")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--hidden", type=str, default="64,64")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=os.path.join(DATA_DIR, "gaze_mlp.pt"))
    args = ap.parse_args()

    files = load_npz_list(args.glob)
    print(f"[Info] Found {len(files)} files")
    X, Y, SW, SH, D = stack_npz(files)
    print(f"[Info] Total samples: {len(X)}  (feat dim={D})")

    # 셔플 & 분할
    idx = np.arange(len(X)); np.random.shuffle(idx)
    n_val = int(len(X) * args.val_ratio)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]

    Xtr, Ytr, SWtr, SHtr = X[tr_idx], Y[tr_idx], SW[tr_idx], SH[tr_idx]
    Xva, Yva, SWva, SHva = X[val_idx], Y[val_idx], SW[val_idx], SH[val_idx]

    # 훈련셋으로 표준화 통계 산출 → train/val에 동일 적용
    tmp_ds = GazeDataset(Xtr, Ytr, SWtr, SHtr)  # mean/std 자동 산출
    x_mean, x_std = tmp_ds.x_mean, tmp_ds.x_std

    ds_tr = GazeDataset(Xtr, Ytr, SWtr, SHtr, x_mean, x_std)
    ds_va = GazeDataset(Xva, Yva, SWva, SHva, x_mean, x_std)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, drop_last=False)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=D, hidden=hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best = {"loss": float("inf"), "rmse": float("inf")}
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        va_loss, va_rmse = eval_epoch(model, dl_va, device)
        print(f"[{ep:03d}] train {tr_loss:.4f} | val {va_loss:.4f} | val RMSE(pix) {va_rmse:.2f}")
        if va_loss < best["loss"]:
            best = {"loss": va_loss, "rmse": va_rmse}
            # 저장
            os.makedirs(DATA_DIR, exist_ok=True)
            torch.save({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "model_state": model.state_dict(),
                "in_dim": D,
                "x_mean": x_mean,
                "x_std": x_std,
                "hidden": hidden,
                "dropout": args.dropout,
                "loss_val": float(va_loss),
                "rmse_val_pix": float(va_rmse),
                "target_norm": "per_screen",   # y가 0~1 정규화로 학습됨
                "arch": "MLP(mu,logvar)"
            }, args.out)
            print(f"  -> saved: {args.out}")

if __name__ == "__main__":
    main()
