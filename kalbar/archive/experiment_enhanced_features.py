"""
Eksperimen lanjutan: tambah fitur baru yang belum dieksplorasi (CLOT_std, CLER_23_std,
pressure_hpa, cloud_cover_oktas SYNOP) ke dataset training_ghi_1h_direct, lalu bandingkan
LightGBM baseline (39 fitur, R2=0.7234) vs enhanced (+4 fitur baru).
"""
import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
OUT_PARQUET = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_enhanced.parquet"

con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT g.*, c.CLOT_std, c.CLER_23_std, m.pressure_hpa, s.cloud_cover_oktas
    FROM training_ghi_1h_direct g
    LEFT JOIN clp_pontianak c ON g.timestamp_wib = c.timestamp_wib + INTERVAL 10 MINUTE
    LEFT JOIN meteorologi_kalbar_10m m ON g.timestamp_wib = m.timestamp_wib
    LEFT JOIN synop_radiasi_jam s ON date_trunc('hour', g.timestamp_wib) = s.timestamp_wib
    WHERE g.anchor_valid
    ORDER BY g.timestamp_wib
""").df()
con.close()

print(f"n_rows={len(df)}  cakupan CLOT_std={df['CLOT_std'].notna().mean()*100:.1f}%  "
      f"cakupan oktas={df['cloud_cover_oktas'].notna().mean()*100:.1f}%")

df.to_parquet(OUT_PARQUET, index=False)
df["year"] = df["timestamp_wib"].dt.year

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
NEW_FEATURES = ["CLOT_std", "CLER_23_std", "pressure_hpa", "cloud_cover_oktas"]
TARGET = "ghi_target_60m"

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)

SEEDS = [42, 7, 123]
LGB_PARAMS = dict(n_estimators=2000, num_leaves=127, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, verbosity=-1)

def fit_bagged(feats, y_col):
    pv, pt = [], []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train[feats], train[y_col], eval_set=[(val[feats], val[y_col])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        pv.append(m.predict(val[feats]))
        pt.append(m.predict(test[feats]))
    return np.mean(pv, axis=0), np.mean(pt, axis=0)

def run_stacked(feats, label):
    train["resid"] = train[TARGET] - train["ghi_final"]
    val["resid"] = val[TARGET] - val["ghi_final"]
    pv_d, pt_d = fit_bagged(feats, TARGET)
    pv_r, pt_r = fit_bagged(feats, "resid")
    pv_r_ghi = pv_r + val["ghi_final"].values
    pt_r_ghi = pt_r + test["ghi_final"].values
    meta = Ridge(alpha=1.0)
    meta.fit(np.column_stack([pv_d, pv_r_ghi]), val[TARGET])
    pred = np.clip(meta.predict(np.column_stack([pt_d, pt_r_ghi])), 0, 1400)
    r2 = r2_score(test[TARGET], pred)
    mae = mean_absolute_error(test[TARGET], pred)
    print(f"{label:40s} n_feat={len(feats):3d}  R2={r2:.4f}  MAE={mae:7.2f}")
    return r2, pred

print()
r2_base, _ = run_stacked(BASE_FEATURES, "Baseline (39 fitur)")
r2_enh, pred_enh = run_stacked(BASE_FEATURES + NEW_FEATURES, "Enhanced (+CLOT_std/CLER_std/pressure/oktas)")
print(f"\nSelisih R2 = {r2_enh - r2_base:+.4f}")
