#!/usr/bin/env python3
"""
R8 Arm B — JAMBI: GBM vs Deep Learning Fair-Play Comparison
Port dari bengkulu_ghi_julius/train_ghi_1h_bengkulu_R8_armB.py (commit 836a397).

Fair-play rules (identik semua model):
  - Features: F1 lean-50 (sama persis dgn R1/Arm A Jambi — dibangun ulang
    lewat builder train_ghi_1h_jambi_R1_benchmark.build_dataset())
  - Split: train<2024, val 2024, test 2025
  - Early stopping: val only (patience=30 DL, 150 GBM)
  - Scaler fit di train saja (DL); DL seeds = 3 (mean +/- std)
  - Input DL: flat 50-feature vector (seq_len=1) — info sama dgn GBM
  - Target: point t+60

Bug fixes yang dibawa dari template (JANGAN diubah):
  1. LSTM: out1, _ = lstm1(x); _, (h2, _) = lstm2(out1)  — pass OUTPUT sequence
  2. MLP: flat_size = n_features (seq_len=1), bukan n_features*18
  3. DB & kolom milik Jambi sendiri (time col = ts, bukan ts_wib)

Catatan koordinat: builder R1 Jambi memakai lat=-1.5833, lon=103.6667 —
dipertahankan agar fitur Arm B identik bit-per-bit dgn R1/Arm A.

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_jambi_R8_armB.py
"""
import sys
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# Dataset harmonis R1 Jambi (50 fitur F1 + target + smart_persist + sun_gt5_t60)
from train_ghi_1h_jambi_R1_benchmark import (
    build_dataset, FEATURES as F1_FEATURES, TARGET_POINT, TIME_COL,
)

OUTPUT_DIR = Path("outputs_R8_jambi")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42
DL_SEEDS = [0, 1, 2]

assert len(F1_FEATURES) == 50


# ---------------------------------------------------------------------------
# DL Architectures — identik template Bengkulu (bug-fixed)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        _, (h2, _) = self.lstm2(out1)
        return self.fc(self.dropout(h2[-1]))


class MLPModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x[:, -1])


class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=8, nlayers=4):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Split / GBM / DL / metrics — identik template
# ---------------------------------------------------------------------------
def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


def train_catboost(x_tr, y_tr, x_va, y_va):
    m = CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_va.astype(float).values, y_va.astype(float).values),
          early_stopping_rounds=150)
    return m


def train_lgbm(x_tr, y_tr, x_va, y_va):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("m", lgb.LGBMRegressor(
            objective="regression", n_estimators=6000, learning_rate=0.02,
            num_leaves=39, min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5,
            colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
            random_state=RANDOM_STATE, n_jobs=-1, force_col_wise=True, verbosity=-1,
        )),
    ])
    pipe.fit(x_tr, y_tr,
             m__eval_set=[(pipe.named_steps["imp"].fit_transform(x_va), y_va)],
             m__eval_metric="rmse",
             m__callbacks=[lgb.early_stopping(150, verbose=False)])
    return pipe


def train_dl(model, train_loader, val_loader, device, patience=30, max_epochs=150):
    opt = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    wait = 0
    best_state = None

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(), yb).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_dl(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).squeeze().cpu().numpy())
    return np.clip(np.concatenate(preds), PRED_MIN, PRED_MAX)


def metrics(y_true, y_pred, y_sp):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_sp = float(np.sqrt(mean_squared_error(y_true, y_sp)))
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / rmse_sp if rmse_sp > 0 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Building harmonised dataset (R1 builder, Jambi)...", flush=True)
    df = build_dataset()

    df_use = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    print(f"Rows: {len(df_use):,}", flush=True)

    tr_m, va_m, te_m = split_masks(df_use)
    print(f"train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}", flush=True)

    f1_avail = [f for f in F1_FEATURES if f in df_use.columns]
    x_tr = df_use.loc[tr_m, f1_avail]
    x_va = df_use.loc[va_m, f1_avail]
    x_te = df_use.loc[te_m, f1_avail]
    y_tr = df_use.loc[tr_m, TARGET_POINT]
    y_va = df_use.loc[va_m, TARGET_POINT]
    y_te = df_use.loc[te_m, TARGET_POINT]
    y_sp = np.clip(df_use.loc[te_m, "smart_persist"].values, PRED_MIN, PRED_MAX)

    results = []

    print(f"\n{'='*60}\nGBM MODELS\n{'='*60}", flush=True)

    print("  Training CatBoost...", flush=True)
    cb = train_catboost(x_tr, y_tr, x_va, y_va)
    pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
    m = metrics(y_te, pred, y_sp)
    m.update({"model": "catboost", "seed": 0, "type": "GBM"})
    results.append(m)
    print(f"  CatBoost  R2={m['r2']:.4f}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}  iter={cb.get_best_iteration()}", flush=True)

    print("  Training LightGBM...", flush=True)
    lgb_pipe = train_lgbm(x_tr, y_tr, x_va, y_va)
    pred = np.clip(lgb_pipe.predict(x_te), PRED_MIN, PRED_MAX)
    m = metrics(y_te, pred, y_sp)
    m.update({"model": "lgbm", "seed": 0, "type": "GBM"})
    results.append(m)
    print(f"  LightGBM  R2={m['r2']:.4f}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}", flush=True)

    print(f"\n{'='*60}\nDL MODELS  ({len(DL_SEEDS)} seeds)\n{'='*60}", flush=True)
    print("  (seq_len=1 — DL sees same 50 pre-processed features as GBM)", flush=True)

    imputer = SimpleImputer(strategy="median")
    x_tr_imp = imputer.fit_transform(x_tr)
    x_va_imp = imputer.transform(x_va)
    x_te_imp = imputer.transform(x_te)

    scaler = StandardScaler()
    x_tr_sc = scaler.fit_transform(x_tr_imp)
    x_va_sc = scaler.transform(x_va_imp)
    x_te_sc = scaler.transform(x_te_imp)

    n_feat = x_tr_sc.shape[1]

    def to_tensor(arr):
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

    xt_t = to_tensor(x_tr_sc)
    xv_t = to_tensor(x_va_sc)
    xe_t = to_tensor(x_te_sc)
    yt_t = torch.from_numpy(y_tr.values.astype(np.float32))
    yv_t = torch.from_numpy(y_va.values.astype(np.float32))
    ye_t = torch.zeros(len(y_te))

    train_loader = DataLoader(TensorDataset(xt_t, yt_t), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(xv_t, yv_t), batch_size=256)
    test_loader = DataLoader(TensorDataset(xe_t, ye_t), batch_size=256)

    dl_models = [
        ("lstm", LSTMModel),
        ("mlp", MLPModel),
        ("transformer", TransformerModel),
    ]

    for model_name, ModelCls in dl_models:
        seed_results = []
        for seed in DL_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ModelCls(n_feat).to(device)
            model = train_dl(model, train_loader, val_loader, device)
            pred = predict_dl(model, test_loader, device)
            m = metrics(y_te, pred, y_sp)
            m.update({"model": model_name, "seed": seed, "type": "DL"})
            results.append(m)
            seed_results.append(m["r2"])
            print(f"    {model_name} seed {seed}: R2={m['r2']:.4f}", flush=True)

        mean_r2 = np.mean(seed_results)
        std_r2 = np.std(seed_results)
        print(f"  {model_name:<12s}  R2={mean_r2:.4f} +/- {std_r2:.4f}", flush=True)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "arm_B_results.csv", index=False)

    summary_rows = []
    for mname, grp in df_res.groupby("model"):
        summary_rows.append({
            "model": mname,
            "type": grp["type"].iloc[0],
            "r2_mean": round(grp["r2"].mean(), 4),
            "r2_std": round(grp["r2"].std(), 4),
            "mae_mean": round(grp["mae"].mean(), 1),
            "rmse_mean": round(grp["rmse"].mean(), 1),
            "n_seeds": len(grp),
        })
    df_sum = pd.DataFrame(summary_rows).sort_values("r2_mean", ascending=False)
    df_sum.to_csv(OUTPUT_DIR / "arm_B_summary.csv", index=False)

    print(f"\n{'='*60}\nSUMMARY — R8 Arm B JAMBI (point t+60, test 2025)\n{'='*60}", flush=True)
    cb_r2 = df_res.loc[df_res["model"] == "catboost", "r2"].values[0]
    lgb_r2 = df_res.loc[df_res["model"] == "lgbm", "r2"].values[0]
    print(f"  GBM: CatBoost R2={cb_r2:.4f} | LightGBM R2={lgb_r2:.4f} (delta {lgb_r2-cb_r2:+.4f})", flush=True)
    for mname in ["lstm", "mlp", "transformer"]:
        grp = df_res[df_res["model"] == mname]
        print(f"  DL {mname:<12s} R2={grp['r2'].mean():.4f} +/- {grp['r2'].std():.4f}  "
              f"(delta vs CatBoost: {grp['r2'].mean()-cb_r2:+.4f})", flush=True)
    print(f"\n  -> outputs: {OUTPUT_DIR}/arm_B_results.csv, arm_B_summary.csv", flush=True)


if __name__ == "__main__":
    main()
