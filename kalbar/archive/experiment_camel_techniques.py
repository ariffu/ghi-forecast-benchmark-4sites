"""
4 teknik dari transkrip CAMEL AI yang GENUINELY berbeda (bukan trik resolusi/agregasi
yang sudah kita bongkar), diuji di resolusi 10-menit asli, split sama (2022-23/2024/2025):
1. VMD (Variational Mode Decomposition) -- dekomposisi sinyal jadi mode oskilasi
2. ghi_clearsky_residual -- fitur fisis tambahan (selisih absolut, bukan rasio kt)
3. Huber loss objective -- robust regression terhadap outlier
4. Analog Ensemble (kNN-based) -- paradigma instance-based, bukan parametrik
"""
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from vmdpy import VMD

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
TARGET = "ghi_target_60m"

con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT g.*, c.CLOT_std, c.CLER_23_std, c.cloud_present AS clp_cloud_present_bool,
           m.pressure_hpa, s.cloud_cover_oktas, a.clear_sky AS arp_clear_sky
    FROM training_ghi_1h_direct g
    LEFT JOIN clp_pontianak c ON g.timestamp_wib = c.timestamp_wib + INTERVAL 10 MINUTE
    LEFT JOIN arp_pontianak a ON g.timestamp_wib = a.timestamp_wib + INTERVAL 10 MINUTE
    LEFT JOIN meteorologi_kalbar_10m m ON g.timestamp_wib = m.timestamp_wib
    LEFT JOIN synop_radiasi_jam s ON date_trunc('hour', g.timestamp_wib) = s.timestamp_wib
    WHERE g.anchor_valid
    ORDER BY g.timestamp_wib
""").df()
con.close()
df["year"] = df["timestamp_wib"].dt.year

FEATURES_BASE = [
    "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean", "clp_cloud_present_int",
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "sun_altitude",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "AOD_500nm", "angstrom_440_870", "precipitable_water_cm",
    "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m",
    "kt_lag10m", "kt_lag20m", "kt_lag30m", "kt_lag60m",
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std",
    "delta_kt_10m", "delta_kt_30m", "clot_lag10m", "clot_lag30m", "delta_clot_30m", "delta_ghi_30m",
    "sun_altitude_future", "ghi_clearsky_future", "hour_sin_future", "hour_cos_future",
    "CLOT_std", "CLER_23_std", "pressure_hpa", "cloud_cover_oktas",
]
for c in FEATURES_BASE:
    df[c] = df[c].astype("float32")
df[FEATURES_BASE] = df[FEATURES_BASE].fillna(df[FEATURES_BASE].median())

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)
print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}\n")

def fit_lgb(feats, y_train_col="__target__", objective="regression", **kw):
    m = lgb.LGBMRegressor(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, verbosity=-1, objective=objective, **kw)
    m.fit(train[feats], train[TARGET], eval_set=[(val[feats], val[TARGET])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = m.predict(test[feats])
    r2 = r2_score(test[TARGET], pred)
    mae = mean_absolute_error(test[TARGET], pred)
    return r2, mae

print("=== 0. Baseline (LightGBM, 43 fitur, objective standar) ===")
r2_base, mae_base = fit_lgb(FEATURES_BASE)
print(f"  Baseline                                          R2={r2_base:.4f}  MAE={mae_base:.2f}")

print("\n=== 1. VMD (Variational Mode Decomposition) pada sinyal kt ===")
kt_series = df.sort_values("timestamp_wib")["kt"].fillna(df["kt"].median()).values
alpha, tau, K, DC, init, tol = 2000, 0, 5, 0, 1, 1e-7
u, u_hat, omega = VMD(kt_series, alpha, tau, K, DC, init, tol)
n_out = u.shape[1]
print(f"  VMD selesai: {K} mode diekstrak dari sinyal kt ({len(kt_series)} titik input, {n_out} titik output)")
df_sorted = df.sort_values("timestamp_wib").reset_index(drop=True).iloc[:n_out].copy()
for k in range(K):
    df_sorted[f"vmd_mode{k}"] = u[k]
df_vmd = df.merge(df_sorted[["timestamp_wib"] + [f"vmd_mode{k}" for k in range(K)]], on="timestamp_wib", how="left")
VMD_FEATURES = FEATURES_BASE + [f"vmd_mode{k}" for k in range(K)]
train_vmd = df_vmd[df_vmd["year"].isin([2022, 2023])].reset_index(drop=True)
val_vmd = df_vmd[df_vmd["year"] == 2024].reset_index(drop=True)
test_vmd = df_vmd[df_vmd["year"] == 2025].reset_index(drop=True)
m_vmd = lgb.LGBMRegressor(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, verbosity=-1)
m_vmd.fit(train_vmd[VMD_FEATURES], train_vmd[TARGET], eval_set=[(val_vmd[VMD_FEATURES], val_vmd[TARGET])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
pred_vmd = m_vmd.predict(test_vmd[VMD_FEATURES])
r2_vmd = r2_score(test_vmd[TARGET], pred_vmd)
mae_vmd = mean_absolute_error(test_vmd[TARGET], pred_vmd)
print(f"  Baseline + 5 mode VMD sbg fitur                   R2={r2_vmd:.4f}  MAE={mae_vmd:.2f}")

print("\n=== 2. Fitur fisis: ghi_clearsky_residual & kt x sun_altitude interaction ===")
df["ghi_clearsky_residual"] = df["ghi_final"] - df["ghi_clearsky"]
df["kt_altitude_interaction"] = df["kt"] * df["sun_altitude"]
FEATURES_PHYS = FEATURES_BASE + ["ghi_clearsky_residual", "kt_altitude_interaction"]
train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)
r2_phys, mae_phys = fit_lgb(FEATURES_PHYS)
print(f"  + ghi_clearsky_residual + kt_altitude_interaction R2={r2_phys:.4f}  MAE={mae_phys:.2f}")

print("\n=== 3. Huber loss objective (robust terhadap outlier) ===")
r2_huber, mae_huber = fit_lgb(FEATURES_BASE, objective="huber", alpha=0.9)
print(f"  Huber loss                                        R2={r2_huber:.4f}  MAE={mae_huber:.2f}")

print("\n=== 4. Analog Ensemble (kNN-based, instance-based, k=30) ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[FEATURES_BASE])
X_test_scaled = scaler.transform(test[FEATURES_BASE])
nn = NearestNeighbors(n_neighbors=30, n_jobs=-1)
nn.fit(X_train_scaled)
dist, idx = nn.kneighbors(X_test_scaled)
y_train_arr = train[TARGET].values
pred_analog = y_train_arr[idx].mean(axis=1)
r2_analog = r2_score(test[TARGET], pred_analog)
mae_analog = mean_absolute_error(test[TARGET], pred_analog)
print(f"  Analog Ensemble (k=30 neighbor)                   R2={r2_analog:.4f}  MAE={mae_analog:.2f}")

print("\n=== RINGKASAN ===")
results = [
    ("Baseline (43 fitur)", r2_base, mae_base),
    ("+ VMD modes (5 mode)", r2_vmd, mae_vmd),
    ("+ ghi_clearsky_residual + interaction", r2_phys, mae_phys),
    ("Huber loss objective", r2_huber, mae_huber),
    ("Analog Ensemble (kNN k=30)", r2_analog, mae_analog),
]
for name, r2, mae in sorted(results, key=lambda x: -x[1]):
    print(f"  {name:42s} R2={r2:.4f}  MAE={mae:.2f}")
print(f"\n  (Pembanding) Ensemble tuned+weighted (note_13)    R2=0.7270  MAE=120.35")
