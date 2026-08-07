#!/usr/bin/env python3
"""
R8 Arm B — Bengkulu: GBM vs Deep Learning Fair-Play Comparison

Fair-play rules (identical for all models):
  - Features: F1 lean-50 (same as R1/Arm A/C)
  - Split: train<2024, val 2024, test 2025 (no test leakage)
  - Early stopping: on val only (patience=30 for DL, 150 for GBM)
  - Scaler: fit on train only (DL)
  - DL seeds: 3 (mean ± std reported)
  - Input: flat 50-feature vector (seq_len=1) — DL sees same info as GBM
  - Target: point t+60 (consistent across all R8 arms)

Bug fixes vs previous comprehensive script:
  - LSTM: now passes output sequence to lstm2, not (wrongly reshaped) hidden state
  - MLP: flat_size = n_features (seq_len=1), not n_features * 18
  - Uses Bengkulu DB and column names, not Kalbar

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_bengkulu_R8_armB.py
"""

import os
from pathlib import Path
import warnings

import duckdb
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_NAME       = "bengkulu"
ATTACH_ALIAS  = "bengkulu_db"
LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")
OUTPUT_DIR    = Path("outputs_R8_bengkulu")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -3.8607
STATION_LON_DEG  = 102.3381
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42
DL_SEEDS  = [0, 1, 2]   # 3 seeds → mean ± std

TARGET_POINT = "ghi_point_t60"

# ---------------------------------------------------------------------------
# F1 features (50 — identical to R1 / Arm A / Arm C)
# ---------------------------------------------------------------------------
F1_FEATURES = [
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m",
    "accel_ghi_20m",
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
    "clp_cot",
    "clp_cot_lag_10m", "clp_cot_lag_20m",
    "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m",
    "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "month_sin", "month_cos",
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
assert len(F1_FEATURES) == 50

# ---------------------------------------------------------------------------
# DL Architectures (fair-play, seq_len=1 — sees same info as GBM)
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """2-layer LSTM (128→64). Bug fixed: passes lstm1 OUTPUT to lstm2,
    not the incorrectly-reshaped hidden state from the original script."""
    def __init__(self, n_features):
        super().__init__()
        self.lstm1   = nn.LSTM(n_features, 128, batch_first=True, dropout=0.2)
        self.lstm2   = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(64, 1)

    def forward(self, x):                    # x: (batch, seq_len, n_features)
        out1, _  = self.lstm1(x)            # out1: (batch, seq_len, 128)
        _, (h2, _) = self.lstm2(out1)       # h2:  (1, batch, 64)
        return self.fc(self.dropout(h2[-1]))  # (batch, 1)


class MLPModel(nn.Module):
    """3-layer MLP. Bug fixed: flat_size = n_features (not n_features*18)
    because seq_len=1 here (DL receives same flat 50-feat vector as GBM)."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):          # x: (batch, seq_len, n_features)
        return self.net(x[:, -1])  # take last (only) timestep → (batch, 1)


class TransformerModel(nn.Module):
    """Transformer encoder (8 heads, 4 layers, d_model=64). Mean-pooling over seq."""
    def __init__(self, n_features, d_model=64, nhead=8, nlayers=4):
        super().__init__()
        self.embed       = nn.Linear(n_features, d_model)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.fc          = nn.Linear(d_model, 1)

    def forward(self, x):                    # x: (batch, seq_len, n_features)
        x = self.embed(x)                    # (batch, seq_len, d_model)
        x = self.transformer(x)             # (batch, seq_len, d_model)
        x = x.mean(dim=1)                   # (batch, d_model)
        return self.fc(x)                   # (batch, 1)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN.")
    os.environ["motherduck_token"] = token


def connect_data():
    con = duckdb.connect(database=":memory:")
    if LOCAL_DB_PATH.exists():
        con.execute(
            "ATTACH '" + LOCAL_DB_PATH.as_posix()
            + "' AS " + ATTACH_ALIAS + " (READ_ONLY)"
        )
        print("Data source: LOCAL")
    else:
        require_token()
        con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
        print("Data source: MOTHERDUCK")
    return con


# ---------------------------------------------------------------------------
# SQL (same as R1)
# ---------------------------------------------------------------------------
def build_sql():
    return """
    WITH with_kt AS (
        SELECT *,
            ghi_now / GREATEST(
                1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0
            ) AS kt_point
        FROM bengkulu_db.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
    ), with_windows AS (
        SELECT *,
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
            LAG(kt_point, 1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point, 2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(kt_point, 3) OVER (ORDER BY ts_wib) AS kt_lag_30m,
            LAG(kt_point, 6) OVER (ORDER BY ts_wib) AS kt_lag_60m,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
                AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                AS kt_roll30m_std
        FROM with_kt
    )
    SELECT * FROM with_windows
    WHERE is_model_ready = 1
      AND has_continuous_3h_history = 1
      AND ghi_now BETWEEN 0 AND 1400
    ORDER BY ts_wib
    """


# ---------------------------------------------------------------------------
# Feature engineering (identical to R1)
# ---------------------------------------------------------------------------
def solar_elevation_deg(timestamps):
    idx  = pd.DatetimeIndex(timestamps)
    doy  = idx.dayofyear.values.astype(float)
    h    = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    ha   = (h + 4.0 * (STATION_LON_DEG - WIB_MERIDIAN_DEG) / 60.0 - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(STATION_LAT_DEG)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(STATION_LAT_DEG)) * np.cos(np.deg2rad(decl))
             * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def clearsky(elev_deg):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(elev_deg)), 0.0)


def add_features(df):
    out = df.copy()
    ts  = pd.DatetimeIndex(out[TIME_COL])

    cs_now     = clearsky(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    out["accel_ghi_20m"]     = out["ghi_now"]  - 2.0 * out["ghi_lag_10m"]     + out["ghi_lag_20m"]
    out["accel_kt_20m"]      = out["kt_now"]   - 2.0 * out["kt_lag_10m"]      + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"]  - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    ts_t60      = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60    = solar_elevation_deg(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"]   = clearsky(elev_t60)

    cs_steps = [clearsky(solar_elevation_deg(out[TIME_COL] + pd.Timedelta(minutes=s * 10)))
                for s in range(1, 7)]
    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * np.column_stack(cs_steps).mean(axis=1)

    out[TARGET_POINT]  = out["target_ghi_1h_ahead"].copy()
    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))
    return out


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


# ---------------------------------------------------------------------------
# GBM trainers (same hyperparams as R1)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# DL trainer
# ---------------------------------------------------------------------------
def train_dl(model, train_loader, val_loader, device, patience=30, max_epochs=150):
    opt       = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    wait      = 0
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
            best_loss  = val_loss
            wait       = 0
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, y_sp):
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_sp = float(np.sqrt(mean_squared_error(y_true, y_sp)))
    return {
        "r2":          round(float(r2_score(y_true, y_pred)), 4),
        "mae":         round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse":        round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / rmse_sp if rmse_sp > 0 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    con = connect_data()
    print("Loading data...")
    df = con.execute(build_sql()).fetchdf()
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    df_use = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    print(f"Rows: {len(df_use):,}")

    tr_m, va_m, te_m = split_masks(df_use)
    print(f"train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

    f1_avail = [f for f in F1_FEATURES if f in df_use.columns]
    x_tr = df_use.loc[tr_m, f1_avail]
    x_va = df_use.loc[va_m, f1_avail]
    x_te = df_use.loc[te_m, f1_avail]
    y_tr = df_use.loc[tr_m, TARGET_POINT]
    y_va = df_use.loc[va_m, TARGET_POINT]
    y_te = df_use.loc[te_m, TARGET_POINT]
    y_sp = np.clip(df_use.loc[te_m, "smart_persist"].values, PRED_MIN, PRED_MAX)

    results = []

    # -------------------------------------------------------------------
    # GBM
    # -------------------------------------------------------------------
    print(f"\n{'='*60}\nGBM MODELS\n{'='*60}")

    print("  Training CatBoost...")
    cb   = train_catboost(x_tr, y_tr, x_va, y_va)
    pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
    m    = metrics(y_te, pred, y_sp)
    m.update({"model": "catboost", "seed": 0, "type": "GBM"})
    results.append(m)
    print(f"  CatBoost  R²={m['r2']:.4f}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}  iter={cb.get_best_iteration()}")

    print("  Training LightGBM...")
    lgb_pipe = train_lgbm(x_tr, y_tr, x_va, y_va)
    pred     = np.clip(lgb_pipe.predict(x_te), PRED_MIN, PRED_MAX)
    m        = metrics(y_te, pred, y_sp)
    m.update({"model": "lgbm", "seed": 0, "type": "GBM"})
    results.append(m)
    print(f"  LightGBM  R²={m['r2']:.4f}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}")

    # -------------------------------------------------------------------
    # DL: impute + scale on train
    # -------------------------------------------------------------------
    print(f"\n{'='*60}\nDL MODELS  ({len(DL_SEEDS)} seeds)\n{'='*60}")
    print("  (seq_len=1 — DL sees same 50 pre-processed features as GBM)")

    imputer = SimpleImputer(strategy="median")
    x_tr_imp = imputer.fit_transform(x_tr)
    x_va_imp = imputer.transform(x_va)
    x_te_imp = imputer.transform(x_te)

    scaler   = StandardScaler()
    x_tr_sc  = scaler.fit_transform(x_tr_imp)
    x_va_sc  = scaler.transform(x_va_imp)
    x_te_sc  = scaler.transform(x_te_imp)

    n_feat   = x_tr_sc.shape[1]

    # Tensors (seq_len = 1)
    def to_tensor(arr):
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)  # (n, 1, features)

    xt_t = to_tensor(x_tr_sc)
    xv_t = to_tensor(x_va_sc)
    xe_t = to_tensor(x_te_sc)
    yt_t = torch.from_numpy(y_tr.values.astype(np.float32))
    yv_t = torch.from_numpy(y_va.values.astype(np.float32))
    ye_t = torch.zeros(len(y_te))  # placeholder for DataLoader

    train_loader = DataLoader(TensorDataset(xt_t, yt_t), batch_size=128, shuffle=True)
    val_loader   = DataLoader(TensorDataset(xv_t, yv_t), batch_size=256)
    test_loader  = DataLoader(TensorDataset(xe_t, ye_t), batch_size=256)

    dl_models = [
        ("lstm",        LSTMModel),
        ("mlp",         MLPModel),
        ("transformer", TransformerModel),
    ]

    for model_name, ModelCls in dl_models:
        seed_results = []
        for seed in DL_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ModelCls(n_feat).to(device)
            model = train_dl(model, train_loader, val_loader, device)
            pred  = predict_dl(model, test_loader, device)
            m     = metrics(y_te, pred, y_sp)
            m.update({"model": model_name, "seed": seed, "type": "DL"})
            results.append(m)
            seed_results.append(m["r2"])

        mean_r2 = np.mean(seed_results)
        std_r2  = np.std(seed_results)
        print(f"  {model_name:<12s}  R²={mean_r2:.4f} ± {std_r2:.4f}  "
              f"(seeds: {[f'{r:.4f}' for r in seed_results]})")

    # -------------------------------------------------------------------
    # Save and report
    # -------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "arm_B_results.csv", index=False)

    # Summary: GBM (seed 0) vs DL (mean ± std)
    summary_rows = []
    for mname, grp in df_res.groupby("model"):
        row = {
            "model":     mname,
            "type":      grp["type"].iloc[0],
            "r2_mean":   round(grp["r2"].mean(), 4),
            "r2_std":    round(grp["r2"].std(), 4),
            "mae_mean":  round(grp["mae"].mean(), 1),
            "rmse_mean": round(grp["rmse"].mean(), 1),
            "n_seeds":   len(grp),
        }
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows).sort_values("r2_mean", ascending=False)
    df_sum.to_csv(OUTPUT_DIR / "arm_B_summary.csv", index=False)

    print(f"\n{'='*60}\nSUMMARY — R8 Arm B (target: point t+60, test 2025)\n{'='*60}")
    cb_r2  = df_res.loc[df_res["model"] == "catboost", "r2"].values[0]
    lgb_r2 = df_res.loc[df_res["model"] == "lgbm",     "r2"].values[0]
    print(f"  GBM best: CatBoost R²={cb_r2:.4f}")
    print(f"  GBM 2nd:  LightGBM R2={lgb_r2:.4f}  (delta vs CB: {lgb_r2-cb_r2:+.4f})")

    for mname in ["lstm", "mlp", "transformer"]:
        grp  = df_res[df_res["model"] == mname]
        mean = grp["r2"].mean()
        std  = grp["r2"].std()
        delta_vs_cb = mean - cb_r2
        print(f"  DL {mname:<12s} R2={mean:.4f} +/- {std:.4f}  (delta vs CatBoost: {delta_vs_cb:+.4f})")

    print(f"\n  → outputs: {OUTPUT_DIR}/arm_B_results.csv, arm_B_summary.csv")
    print(f"  → These are the actual numbers for Tabel 2c in paper")


if __name__ == "__main__":
    main()
