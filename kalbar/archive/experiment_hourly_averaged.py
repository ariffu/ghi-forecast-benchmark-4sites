"""
Replikasi pendekatan CAMEL AI: agregasi data 10-menit menjadi resolusi per-jam
(rata-rata/std/min/max dalam jam), lalu prediksi rata-rata GHI jam berikutnya.
Tujuan: cek apakah genuinely R2 naik di data KITA dengan teknik yang SAMA, dan
pahami betul mekanismenya (bukan sekadar percaya klaim CAMEL).
"""
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import r2_score, mean_absolute_error

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"

con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT g.timestamp_wib, g.ghi_final, g.kt, g.ghi_clearsky, g.sun_altitude,
           g.temp_air_c, g.humidity_pct, g.wind_speed_ms, g.rainfall_mm,
           g.CLOT_mean, g.CLTT_mean, g.CLTH_mean, g.CLER_23_mean,
           g.AOD_500nm, g.fill_tier,
           c.CLOT_std, m.pressure_hpa, s.cloud_cover_oktas
    FROM training_ghi_1h_direct g
    LEFT JOIN clp_pontianak c ON g.timestamp_wib = c.timestamp_wib + INTERVAL 10 MINUTE
    LEFT JOIN meteorologi_kalbar_10m m ON g.timestamp_wib = m.timestamp_wib
    LEFT JOIN synop_radiasi_jam s ON date_trunc('hour', g.timestamp_wib) = s.timestamp_wib
    ORDER BY g.timestamp_wib
""").df()
con.close()

df["hour_wib"] = df["timestamp_wib"].dt.floor("h")

# --- Agregasi per jam (gaya CAMEL: mean/std/min/max) ---
hourly = df.groupby("hour_wib").agg(
    ghi_avg=("ghi_final", "mean"), ghi_std=("ghi_final", "std"),
    ghi_min=("ghi_final", "min"), ghi_max=("ghi_final", "max"),
    ghi_clearsky_avg=("ghi_clearsky", "mean"),
    kt_avg=("kt", "mean"), kt_std=("kt", "std"),
    sun_altitude_avg=("sun_altitude", "mean"),
    temp_air_c_avg=("temp_air_c", "mean"), humidity_pct_avg=("humidity_pct", "mean"),
    wind_speed_ms_avg=("wind_speed_ms", "mean"), rainfall_mm_sum=("rainfall_mm", "sum"),
    CLOT_mean_avg=("CLOT_mean", "mean"), CLTT_mean_avg=("CLTT_mean", "mean"),
    CLTH_mean_avg=("CLTH_mean", "mean"), CLER_23_mean_avg=("CLER_23_mean", "mean"),
    CLOT_std_avg=("CLOT_std", "mean"), pressure_hpa_avg=("pressure_hpa", "mean"),
    cloud_cover_oktas_avg=("cloud_cover_oktas", "mean"), AOD_500nm_avg=("AOD_500nm", "mean"),
    n_obs=("ghi_final", "count"),
    n_original=("fill_tier", lambda x: (x == "TIER_0_ORIGINAL").sum()),
).reset_index()
hourly["pct_original"] = hourly["n_original"] / hourly["n_obs"]
print(f"Total jam (semua, termasuk malam): {len(hourly)}")

hourly = hourly.sort_values("hour_wib").reset_index(drop=True)
hourly["hour"] = hourly["hour_wib"].dt.hour
hourly["doy"] = hourly["hour_wib"].dt.dayofyear
hourly["month"] = hourly["hour_wib"].dt.month
hourly["year"] = hourly["hour_wib"].dt.year
hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)
hourly["doy_sin"] = np.sin(2 * np.pi * hourly["doy"] / 365)
hourly["doy_cos"] = np.cos(2 * np.pi * hourly["doy"] / 365)

# Target: rata-rata GHI JAM BERIKUTNYA (gap harus persis 1 jam)
gap_ok = hourly["hour_wib"].diff().shift(-1).dt.total_seconds().eq(3600)
hourly["target_ghi_avg_next1h"] = np.where(gap_ok, hourly["ghi_avg"].shift(-1), np.nan)
hourly["sun_altitude_avg_next1h"] = np.where(gap_ok, hourly["sun_altitude_avg"].shift(-1), np.nan)
hourly["hour_sin_future"] = np.where(gap_ok, hourly["hour_sin"].shift(-1), np.nan)
hourly["hour_cos_future"] = np.where(gap_ok, hourly["hour_cos"].shift(-1), np.nan)
hourly["ghi_clearsky_avg_future"] = np.where(gap_ok, hourly["ghi_clearsky_avg"].shift(-1), np.nan)

# Lag 1 jam sebelumnya & delta (gaya CAMEL: delta_ghi_1h)
gap_ok_prev = hourly["hour_wib"].diff().dt.total_seconds().eq(3600)
hourly["ghi_avg_lag1h"] = np.where(gap_ok_prev, hourly["ghi_avg"].shift(1), np.nan)
hourly["kt_avg_lag1h"] = np.where(gap_ok_prev, hourly["kt_avg"].shift(1), np.nan)
hourly["delta_ghi_1h"] = hourly["ghi_avg"] - hourly["ghi_avg_lag1h"]
hourly["delta_kt_1h"] = hourly["kt_avg"] - hourly["kt_avg_lag1h"]

# Anchor valid: siang hari sekarang & jam depan, target tidak NULL, data cukup lengkap dalam jam itu
hourly["anchor_valid_hourly"] = (
    (hourly["sun_altitude_avg"] > 5) & (hourly["sun_altitude_avg_next1h"] > 5) &
    hourly["target_ghi_avg_next1h"].notna() & (hourly["n_obs"] >= 5) & hourly["ghi_avg_lag1h"].notna()
)
valid = hourly[hourly["anchor_valid_hourly"]].reset_index(drop=True)
print(f"Jam anchor_valid (siang, lengkap, ada lag & target): {len(valid)}")

FEATURES = ["ghi_avg", "ghi_std", "ghi_min", "ghi_max", "ghi_clearsky_avg", "kt_avg", "kt_std",
            "sun_altitude_avg", "temp_air_c_avg", "humidity_pct_avg", "wind_speed_ms_avg",
            "rainfall_mm_sum", "CLOT_mean_avg", "CLTT_mean_avg", "CLTH_mean_avg", "CLER_23_mean_avg",
            "CLOT_std_avg", "pressure_hpa_avg", "cloud_cover_oktas_avg", "AOD_500nm_avg",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
            "sun_altitude_avg_next1h", "hour_sin_future", "hour_cos_future", "ghi_clearsky_avg_future",
            "ghi_avg_lag1h", "kt_avg_lag1h", "delta_ghi_1h", "delta_kt_1h"]
TARGET = "target_ghi_avg_next1h"

for c in FEATURES:
    valid[c] = valid[c].astype("float32")
valid[FEATURES] = valid[FEATURES].fillna(valid[FEATURES].median())

train = valid[valid["year"].isin([2022, 2023])].reset_index(drop=True)
val = valid[valid["year"] == 2024].reset_index(drop=True)
test = valid[valid["year"] == 2025].reset_index(drop=True)
print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}\n")

m1 = lgb.LGBMRegressor(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, verbosity=-1)
m1.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
       callbacks=[lgb.early_stopping(50, verbose=False)])
p1 = m1.predict(test[FEATURES])
print(f"{'LightGBM (hourly-averaged)':35s} R2={r2_score(test[TARGET], p1):.4f}  MAE={mean_absolute_error(test[TARGET], p1):.2f}")

m2 = xgb.XGBRegressor(n_estimators=2000, max_depth=8, learning_rate=0.03, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
m2.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])], verbose=False)
p2 = m2.predict(test[FEATURES])
print(f"{'XGBoost (hourly-averaged)':35s} R2={r2_score(test[TARGET], p2):.4f}  MAE={mean_absolute_error(test[TARGET], p2):.2f}")

m3 = cb.CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.03, random_state=42,
                           verbose=False, early_stopping_rounds=50)
m3.fit(train[FEATURES], train[TARGET], eval_set=(val[FEATURES], val[TARGET]))
p3 = m3.predict(test[FEATURES])
print(f"{'CatBoost (hourly-averaged)':35s} R2={r2_score(test[TARGET], p3):.4f}  MAE={mean_absolute_error(test[TARGET], p3):.2f}")

p_ens = (p1 + p2 + p3) / 3
r2_ens = r2_score(test[TARGET], p_ens)
mae_ens = mean_absolute_error(test[TARGET], p_ens)
print(f"{'Ensemble 3-model (hourly-averaged)':35s} R2={r2_ens:.4f}  MAE={mae_ens:.2f}")

print("\nTop feature importance (LightGBM):")
imp = pd.DataFrame({"feature": FEATURES, "importance": m1.feature_importances_}).sort_values("importance", ascending=False)
print(imp.head(10).to_string(index=False))

print("\n=== RINGKASAN PERBANDINGAN ===")
print(f"  10-menit direct, genuine (sesi ini)                 R2=0.7270  MAE=120.35")
print(f"  Per-jam, point sample tanpa averaging (sesi ini)     R2=0.7265  MAE=122.15")
print(f"  Per-jam, AVERAGED (replikasi gaya CAMEL, data kita)  R2={r2_ens:.4f}  MAE={mae_ens:.2f}")
print(f"  CAMEL AI, hourly-averaged (klaim asli)               R2=0.8066  MAE=95.31")
