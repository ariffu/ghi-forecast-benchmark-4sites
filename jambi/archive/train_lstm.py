#!/usr/bin/env python3
"""
train_lstm.py
=============
LSTM dual-stream untuk prediksi GHI 1 jam ke depan — Stasiun Jambi.

Arsitektur:
  Stream 1 — Sequence (LSTM):
    Input: (batch, 18, 3) → [ghi, kt, delta_ghi] per timestep
    - ghi_lag: tersedia lag 1..18 (semua valid karena window validity)
    - kt_lag, delta_ghi_lag: tersedia lag 1..6 (zero-padded untuk lag 7..18)
    2-layer Bidirectional LSTM, hidden=128, dropout=0.2
    Output: (batch, 256)

  Stream 2 — Static (MLP):
    Input: anchor + future clearsky + cyclical time + meteo
    Output: (batch, 64)

  Fusion → Dense(128) → Dense(64) → Dense(6) → GHI t+1..t+6

FIX v2:
  - Hapus delta_kt dan sun_alt dari sequence (tidak ada di dataset_clean → std=0
    → StandardScaler menghasilkan NaN → loss=NaN dari epoch 1)
  - Pakai masked Huber loss (aman terhadap target NaN)
  - Target scaler fit hanya pada baris non-NaN
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

warnings.filterwarnings("ignore")

DATASET_DIR = Path(__file__).parent / "dataset_clean"
OUTPUT_DIR  = Path(__file__).parent / "models_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON  = 6
LOOKBACK = 18
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── HYPERPARAMETER ───────────────────────────────────────────────────────────
CFG = dict(
    lstm_hidden   = 128,
    lstm_layers   = 2,
    lstm_dropout  = 0.20,
    mlp_hidden    = 128,
    mlp_dropout   = 0.10,
    fusion_hidden = 128,
    batch_size    = 512,
    lr            = 1e-3,
    epochs        = 200,
    patience      = 25,
    lr_patience   = 10,
    lr_factor     = 0.5,
    min_lr        = 1e-5,
    grad_clip     = 1.0,
)

# ─── FITUR SEQUENCE ──────────────────────────────────────────────────────────
# CATATAN: hanya fitur yang benar-benar ada di dataset_clean yang boleh masuk.
# delta_kt_lag dan sun_alt_lag TIDAK ADA → std=0 → StandardScaler → NaN.
# ghi_lag 1..18   : penuh (window validity)
# kt_lag 1..6     : ada; lag 7..18 zero-padded
# delta_ghi_lag 1..6: ada; lag 7..18 zero-padded
SEQ_FEAT_NAMES = ["ghi", "kt", "delta_ghi"]   # 3 fitur, TIDAK ada delta_kt/sun_alt

# ─── FITUR STATIC ────────────────────────────────────────────────────────────
STATIC_BASE = [
    "anchor_ghi", "anchor_kt", "anchor_sun_alt", "anchor_sun_az",
    "month_sin", "month_cos", "hour_sin", "hour_cos",
]
FUTURE_FEATS = (
      [f"sun_alt_h{h}"       for h in range(1, HORIZON + 1)]
    + [f"clearsky_ghi_h{h}"  for h in range(1, HORIZON + 1)]
)
METEO_FEATS = [
    "temp_air", "rh", "pressure", "ws", "ws_max", "wd", "rain", "cloud_oktas"
]
TARGET_COLS = [f"ghi_h{h}" for h in range(1, HORIZON + 1)]


# ─── BUILD ARRAYS ─────────────────────────────────────────────────────────────
def build_sequences(df: pd.DataFrame) -> np.ndarray:
    """
    Bangun array (n, 18, 3) dari kolom lag.
    Urutan: lag18 (tertua) → lag1 (terbaru) = urutan kausal untuk LSTM.
    """
    n   = len(df)
    seq = np.zeros((n, LOOKBACK, len(SEQ_FEAT_NAMES)), dtype=np.float32)

    for k in range(1, LOOKBACK + 1):
        step = LOOKBACK - k   # lag18 → index 0, lag1 → index 17

        # GHI: tersedia semua lag 1..18
        seq[:, step, 0] = df[f"ghi_lag{k}"].values

        # KT: tersedia lag 1..6 saja
        col_kt = f"kt_lag{k}"
        if col_kt in df.columns:
            v = df[col_kt].values.astype(np.float32)
            seq[:, step, 1] = np.nan_to_num(v, nan=0.0)
        # else: sudah 0.0 dari np.zeros

        # delta_ghi: tersedia lag 1..6 saja
        col_dg = f"delta_ghi_lag{k}"
        if col_dg in df.columns:
            v = df[col_dg].values.astype(np.float32)
            seq[:, step, 2] = np.nan_to_num(v, nan=0.0)

    # Pastikan tidak ada NaN tersisa
    seq = np.nan_to_num(seq, nan=0.0)
    return seq


def get_static(df: pd.DataFrame, use_meteo: bool = True) -> np.ndarray:
    cols = STATIC_BASE + FUTURE_FEATS
    if use_meteo:
        cols = cols + METEO_FEATS
    available = [c for c in cols if c in df.columns]
    arr = df[available].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def get_targets(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COLS].values.astype(np.float32)


# ─── DATASET ──────────────────────────────────────────────────────────────────
class GHIDataset(Dataset):
    def __init__(self, seq: np.ndarray, static: np.ndarray, targets: np.ndarray):
        self.seq    = torch.tensor(seq,     dtype=torch.float32)
        self.static = torch.tensor(static,  dtype=torch.float32)
        self.y      = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq[idx], self.static[idx], self.y[idx]


# ─── MODEL ────────────────────────────────────────────────────────────────────
class DualStreamGHI(nn.Module):
    def __init__(self, seq_feat: int, static_feat: int, cfg=None):
        super().__init__()
        c = cfg or CFG

        # Stream 1: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = seq_feat,
            hidden_size   = c["lstm_hidden"],
            num_layers    = c["lstm_layers"],
            batch_first   = True,
            dropout       = c["lstm_dropout"] if c["lstm_layers"] > 1 else 0,
            bidirectional = True,
        )
        lstm_out_dim = c["lstm_hidden"] * 2  # bidirectional

        # Stream 2: Static MLP
        self.static_mlp = nn.Sequential(
            nn.Linear(static_feat, c["mlp_hidden"]),
            nn.LayerNorm(c["mlp_hidden"]),
            nn.GELU(),
            nn.Dropout(c["mlp_dropout"]),
            nn.Linear(c["mlp_hidden"], 64),
            nn.GELU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(lstm_out_dim + 64, c["fusion_hidden"]),
            nn.LayerNorm(c["fusion_hidden"]),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(c["fusion_hidden"], 64),
            nn.GELU(),
            nn.Linear(64, HORIZON),
        )

    def forward(self, seq, static):
        lstm_out, _ = self.lstm(seq)
        lstm_rep    = lstm_out[:, -1, :]        # output langkah terakhir (paling recent)
        static_rep  = self.static_mlp(static)
        out         = self.fusion(torch.cat([lstm_rep, static_rep], dim=1))
        return out


# ─── LOSS (MASKED HUBER) ──────────────────────────────────────────────────────
def masked_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss yang mengabaikan elemen NaN di target."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0).sum()   # gradient tetap terhubung
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")


# ─── NORMALISASI TARGET ───────────────────────────────────────────────────────
def fit_target_scaler(y_tr: np.ndarray) -> StandardScaler:
    """Fit StandardScaler hanya pada baris di mana semua 6 target valid."""
    valid_rows = ~np.any(np.isnan(y_tr), axis=1)
    scaler = StandardScaler()
    scaler.fit(y_tr[valid_rows])
    return scaler


def transform_targets(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Transform, pertahankan NaN."""
    nan_mask = np.isnan(y)
    y_scaled = scaler.transform(np.nan_to_num(y, nan=0.0)).astype(np.float32)
    y_scaled[nan_mask] = float("nan")
    return y_scaled


# ─── TRAINING UTILITIES ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, grad_clip) -> float:
    model.train()
    total_loss = 0.0
    for seq, static, y in loader:
        seq, static, y = seq.to(DEVICE), static.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(seq, static)
        loss = masked_huber(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader) -> tuple:
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for seq, static, y in loader:
        seq, static, y = seq.to(DEVICE), static.to(DEVICE), y.to(DEVICE)
        pred = model(seq, static)
        total_loss += masked_huber(pred, y).item() * len(y)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    loss  = total_loss / len(loader.dataset)
    preds = np.vstack(all_pred)
    trues = np.vstack(all_true)
    return loss, preds, trues


def compute_metrics(y_true_n, y_pred_n, scaler_y) -> list:
    y_true = scaler_y.inverse_transform(np.nan_to_num(y_true_n, nan=0.0))
    y_pred = scaler_y.inverse_transform(y_pred_n)
    results = []
    for h in range(HORIZON):
        yt, yp = y_true[:, h], y_pred[:, h]
        mask   = ~np.isnan(y_true_n[:, h])   # baris yang valid di target asli
        if mask.sum() < 5:
            results.append(dict(h=h+1, r2=np.nan, rmse=np.nan, mae=np.nan))
            continue
        results.append(dict(
            h    = h + 1,
            r2   = r2_score(yt[mask], yp[mask]),
            rmse = np.sqrt(mean_squared_error(yt[mask], yp[mask])),
            mae  = mean_absolute_error(yt[mask], yp[mask]),
        ))
    return results


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────
def train_stage(stage: int, all_records: list) -> None:
    print(f"\n{'='*60}")
    print(f"  Stage {stage} LSTM")
    print(f"{'='*60}")

    use_meteo = stage >= 2
    df_tr = pd.read_parquet(DATASET_DIR / f"jambi_clean_s{stage}_train.parquet")
    df_va = pd.read_parquet(DATASET_DIR / f"jambi_clean_s{stage}_val.parquet")
    df_te = pd.read_parquet(DATASET_DIR / f"jambi_clean_s{stage}_test.parquet")

    # ── Build raw arrays ──────────────────────────────────────────────────────
    seq_tr = build_sequences(df_tr)
    seq_va = build_sequences(df_va)
    seq_te = build_sequences(df_te)

    sta_tr = get_static(df_tr, use_meteo)
    sta_va = get_static(df_va, use_meteo)
    sta_te = get_static(df_te, use_meteo)

    y_tr = get_targets(df_tr)
    y_va = get_targets(df_va)
    y_te = get_targets(df_te)

    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)

    # ── Verifikasi tidak ada NaN di sequence ──────────────────────────────────
    assert not np.any(np.isnan(seq_tr)), "NaN di seq_tr!"
    assert not np.any(np.isnan(seq_va)), "NaN di seq_va!"
    assert not np.any(np.isnan(seq_te)), "NaN di seq_te!"

    # ── Normalisasi sequence ──────────────────────────────────────────────────
    # Reshape ke (n×18, 3), fit, reshape kembali
    seq_2d_tr = seq_tr.reshape(-1, seq_tr.shape[2])
    scaler_seq = StandardScaler()
    seq_2d_tr  = scaler_seq.fit_transform(seq_2d_tr)

    # Cek: tidak ada std=0
    if np.any(scaler_seq.scale_ < 1e-8):
        bad = np.where(scaler_seq.scale_ < 1e-8)[0]
        print(f"  ⚠  Kolom dengan std≈0 di sequence: {[SEQ_FEAT_NAMES[i] for i in bad]} — di-skip normalisasi")
        for i in bad:
            scaler_seq.scale_[i] = 1.0

    seq_tr = seq_2d_tr.reshape(n_tr, LOOKBACK, -1).astype(np.float32)
    seq_va = scaler_seq.transform(seq_va.reshape(-1, seq_va.shape[2])).reshape(n_va, LOOKBACK, -1).astype(np.float32)
    seq_te = scaler_seq.transform(seq_te.reshape(-1, seq_te.shape[2])).reshape(n_te, LOOKBACK, -1).astype(np.float32)

    # ── Normalisasi static ────────────────────────────────────────────────────
    scaler_sta = StandardScaler()
    sta_tr = scaler_sta.fit_transform(sta_tr).astype(np.float32)
    sta_va = scaler_sta.transform(sta_va).astype(np.float32)
    sta_te = scaler_sta.transform(sta_te).astype(np.float32)

    # ── Normalisasi target (fit hanya pada baris non-NaN) ─────────────────────
    scaler_y  = fit_target_scaler(y_tr)
    y_tr_n    = transform_targets(y_tr, scaler_y)
    y_va_n    = transform_targets(y_va, scaler_y)
    y_te_n    = transform_targets(y_te, scaler_y)

    static_feat_dim = sta_tr.shape[1]
    seq_feat_dim    = seq_tr.shape[2]
    print(f"  Seq shape : {seq_tr.shape} | Static dim: {static_feat_dim}")
    print(f"  Device    : {DEVICE}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_dl = DataLoader(GHIDataset(seq_tr, sta_tr, y_tr_n),
                          batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(GHIDataset(seq_va, sta_va, y_va_n),
                          batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    test_dl  = DataLoader(GHIDataset(seq_te, sta_te, y_te_n),
                          batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model     = DualStreamGHI(seq_feat=seq_feat_dim, static_feat=static_feat_dim, cfg=CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=CFG["lr_factor"],
        patience=CFG["lr_patience"], min_lr=CFG["min_lr"]
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params    : {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = np.inf
    patience_cnt  = 0
    best_weights  = None
    t_start       = time.time()

    print(f"\n  {'Epoch':>5} | {'Train':>8} | {'Val':>8} | {'LR':>8} | Time")
    print(f"  {'─'*5}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+------")

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss = train_epoch(model, train_dl, optimizer, CFG["grad_clip"])
        va_loss, _, _ = evaluate(model, val_dl)

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_cnt  = 0
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1 or patience_cnt == 0:
            elapsed = time.time() - t_start
            print(f"  {epoch:5d} | {tr_loss:8.5f} | {va_loss:8.5f} | {lr_now:8.6f} | {elapsed:.0f}s")

        if patience_cnt >= CFG["patience"]:
            print(f"  Early stop epoch {epoch}")
            break

    # ── Evaluasi best weights ─────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    torch.save(best_weights, OUTPUT_DIR / f"lstm_s{stage}_best.pt")

    _, pred_va, true_va = evaluate(model, val_dl)
    _, pred_te, true_te = evaluate(model, test_dl)

    metrics_va = compute_metrics(true_va, pred_va, scaler_y)
    metrics_te = compute_metrics(true_te, pred_te, scaler_y)

    print(f"\n  {'Horizon':>10} | {'Val R²':>8} | {'Val RMSE':>9} | {'Test R²':>8} | {'Test RMSE':>10}")
    print(f"  {'─'*10}-+-{'─'*8}-+-{'─'*9}-+-{'─'*8}-+-{'─'*10}")
    for mva, mte in zip(metrics_va, metrics_te):
        h = mva["h"]
        print(f"  t+{h} ({h*10:2d}min) | {mva['r2']:8.4f} | {mva['rmse']:9.1f} | "
              f"{mte['r2']:8.4f} | {mte['rmse']:10.1f}")

    mean_va = np.nanmean([m["r2"] for m in metrics_va])
    mean_te = np.nanmean([m["r2"] for m in metrics_te])
    print(f"  {'MEAN':>10} | {mean_va:8.4f} | {'':9} | {mean_te:8.4f}")

    for split, mlist in [("val", metrics_va), ("test", metrics_te)]:
        for m in mlist:
            all_records.append(dict(stage=stage, split=split, **m))


# ─── RINGKASAN ────────────────────────────────────────────────────────────────
def print_final_summary(lstm_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("PERBANDINGAN LSTM vs LightGBM — Stasiun Jambi")
    print("=" * 65)

    lgbm_path = Path(__file__).parent / "models_clean" / "metrics_clean.csv"
    if lgbm_path.exists():
        lgbm_df = pd.read_csv(lgbm_path)
        for split, yr in [("val", "2024"), ("test", "2025")]:
            print(f"\n  {split.upper()} {yr} — Mean R² per Stage:")
            print(f"  {'Stage':>8} | {'LightGBM':>10} | {'LSTM':>8} | {'Δ(LSTM-LGBM)':>14}")
            for s in [1, 2]:
                lgbm_r2 = lgbm_df[(lgbm_df.stage == s) & (lgbm_df.split == split)]["r2"].mean() \
                          if "stage" in lgbm_df.columns else np.nan
                lstm_r2 = lstm_df[(lstm_df.stage == s) & (lstm_df.split == split)]["r2"].mean()
                delta   = lstm_r2 - lgbm_r2
                print(f"  {s:>8} | {lgbm_r2:10.4f} | {lstm_r2:8.4f} | {delta:+14.4f}")
    else:
        print("  (LightGBM metrics tidak ditemukan)")

    print("\n  Per-horizon Stage 2 LSTM:")
    for split, yr in [("val", "2024"), ("test", "2025")]:
        s2 = lstm_df[(lstm_df.stage == 2) & (lstm_df.split == split)].sort_values("h")
        if s2.empty:
            continue
        r2s = " | ".join(f"h+{int(row.h)}: {row.r2:.4f}" for _, row in s2.iterrows())
        print(f"  {split.upper()} {yr}: {r2s} | Mean={s2.r2.mean():.4f}")

    print("=" * 65)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LSTM Dual-Stream — GHI Forecasting Jambi (v2 fixed)")
    print(f"  Device    : {DEVICE}")
    print(f"  Seq feats : {SEQ_FEAT_NAMES}")
    print(f"  LSTM      : {CFG['lstm_layers']} layer × {CFG['lstm_hidden']} hidden, BiLSTM")
    print(f"  Epochs    : {CFG['epochs']} max, patience={CFG['patience']}")
    print(f"  Loss      : Masked Huber (delta=1.0)")
    print("=" * 60)

    records = []
    for stage in [1, 2]:
        train_stage(stage, records)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "metrics_lstm.csv", index=False)
    print_final_summary(df)
    print(f"\nModels → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
