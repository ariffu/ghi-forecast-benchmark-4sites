"""
Eksperimen: apakah menambahkan DHI (sekarang sudah bersih, 0% pelanggaran DHI>GHI)
sebagai fitur -- termasuk diffuse fraction kd_now=DHI/GHI, indikator klasik kondisi
langit di literatur solar engineering -- menaikkan R2 dibanding model produksi
(train_ghi_1h_stacked_raw.py, R2=0.7234 tanpa DHI sama sekali).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PARQUET = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_direct.parquet"

df = pd.read_parquet(PARQUET)
df = df[df["anchor_valid"]].copy().sort_values("timestamp_wib").reset_index(drop=True)
df["year"] = df["timestamp_wib"].dt.year

df["kd_now"] = np.where(df["ghi_final"] > 0, df["dhi_final"] / df["ghi_final"], np.nan)
gap_ok10 = df["timestamp_wib"].diff(1).dt.total_seconds().div(60).between(5, 15)
gap_ok30 = df["timestamp_wib"].diff(3).dt.total_seconds().div(60).between(25, 35)
df["dhi_lag10m"] = np.where(gap_ok10, df["dhi_final"].shift(1), np.nan)
df["kd_lag10m"] = np.where(gap_ok10, df["kd_now"].shift(1), np.nan)
df["delta_dhi_30m"] = np.where(gap_ok30, df["dhi_final"] - df["dhi_final"].shift(3), np.nan)

BASE_FEATURES = [
    "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean", "clp_cloud_present_int",
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "sun_altitude",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "AOD_500nm", "angstrom_440_870", "precipitable_water_cm",
    "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m",
    "kt_lag10m", "kt_lag20m", "kt_lag30m", "kt_lag60m",
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std",
    "delta_kt_10m", "delta_kt_30m", "clot_lag10m", "clot_lag30m", "delta_clot_30m", "delta_ghi_30m",
    "sun_altitude_future", "ghi_clearsky_future", "hour_sin_future", "hour_cos_future",
]
DHI_FEATURES = ["dhi_final", "kd_now", "dhi_lag10m", "kd_lag10m", "delta_dhi_30m"]
TARGET = "ghi_target_60m"

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)

SEEDS = [42, 7, 123]
LGB_PARAMS = dict(n_estimators=2000, num_leaves=127, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, verbosity=-1)

def fit_bagged(feats, y_col):
    preds_val, preds_test = [], []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train[feats], train[y_col], eval_set=[(val[feats], val[y_col])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds_val.append(m.predict(val[feats]))
        preds_test.append(m.predict(test[feats]))
    return np.mean(preds_val, axis=0), np.mean(preds_test, axis=0)

def run_experiment(feats, label):
    train["resid"] = train[TARGET] - train["ghi_final"]
    val["resid"] = val[TARGET] - val["ghi_final"]
    pv_d, pt_d = fit_bagged(feats, TARGET)
    pv_r, pt_r = fit_bagged(feats, "resid")
    pv_r_ghi = pv_r + val["ghi_final"].values
    pt_r_ghi = pt_r + test["ghi_final"].values
    meta = Ridge(alpha=1.0)
    meta.fit(np.column_stack([pv_d, pv_r_ghi]), val[TARGET])
    pred_stack = np.clip(meta.predict(np.column_stack([pt_d, pt_r_ghi])), 0, 1400)
    r2 = r2_score(test[TARGET], pred_stack)
    mae = mean_absolute_error(test[TARGET], pred_stack)
    print(f"{label:40s} n_feat={len(feats):3d}  R2={r2:.4f}  MAE={mae:7.2f}")
    return r2

print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}\n")
r2_base = run_experiment(BASE_FEATURES, "Baseline (tanpa DHI)")
r2_dhi = run_experiment(BASE_FEATURES + DHI_FEATURES, "Baseline + fitur DHI/kd")
print(f"\nSelisih R2 = {r2_dhi - r2_base:+.4f}")
