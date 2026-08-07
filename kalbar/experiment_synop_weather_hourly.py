"""
Uji apakah menambah data SYNOP per-jam (present_weather, cloud_low_type, past_weather)
ke model hourly-averaged (R2=0.8341 baseline) bisa menaikkan akurasi lebih lanjut.
SYNOP sudah native per-jam (bukan 10-menit), jadi tidak perlu averaging -- ini sumber
independen dari satelit (observer darat), bukan trik resolusi.
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
           c.CLOT_std, m.pressure_hpa,
           s.cloud_cover_oktas, s.present_weather, s.past_weather_w1, s.cloud_low_type,
           s.cloud_layer2_type, s.visibility_km
    FROM training_ghi_1h_direct g
    LEFT JOIN clp_pontianak c ON g.timestamp_wib = c.timestamp_wib + INTERVAL 10 MINUTE
    LEFT JOIN meteorologi_kalbar_10m m ON g.timestamp_wib = m.timestamp_wib
    LEFT JOIN synop_radiasi_jam s ON date_trunc('hour', g.timestamp_wib) = s.timestamp_wib
    ORDER BY g.timestamp_wib
""").df()
con.close()

df["hour_wib"] = df["timestamp_wib"].dt.floor("h")

# SYNOP sudah native per jam -- ambil "first" (bukan rata-rata, karena kategorikal/kode WMO)
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
    cloud_cover_oktas=("cloud_cover_oktas", "first"),
    present_weather=("present_weather", "first"),
    past_weather_w1=("past_weather_w1", "first"),
    cloud_low_type=("cloud_low_type", "first"),
    cloud_layer2_type=("cloud_layer2_type", "first"),
    visibility_km=("visibility_km", "first"),
    AOD_500nm_avg=("AOD_500nm", "mean"),
    n_obs=("ghi_final", "count"),
).reset_index().sort_values("hour_wib").reset_index(drop=True)

# Derivasi fitur dari kode cuaca WMO present_weather (00-99)
hourly["is_raining_now"] = hourly["present_weather"].between(50, 99).astype(float)
hourly["is_thunderstorm_now"] = (hourly["present_weather"].isin([17, 29, 91, 92, 93, 94, 95, 96, 97, 98, 99])).astype(float)
hourly["was_raining_recent"] = hourly["past_weather_w1"].between(5, 9).astype(float)

hourly["hour"] = hourly["hour_wib"].dt.hour
hourly["doy"] = hourly["hour_wib"].dt.dayofyear
hourly["month"] = hourly["hour_wib"].dt.month
hourly["year"] = hourly["hour_wib"].dt.year
hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)
hourly["doy_sin"] = np.sin(2 * np.pi * hourly["doy"] / 365)
hourly["doy_cos"] = np.cos(2 * np.pi * hourly["doy"] / 365)

gap_ok_next = hourly["hour_wib"].diff().shift(-1).dt.total_seconds().eq(3600)
hourly["target"] = np.where(gap_ok_next, hourly["ghi_avg"].shift(-1), np.nan)
hourly["sun_altitude_avg_future"] = np.where(gap_ok_next, hourly["sun_altitude_avg"].shift(-1), np.nan)
hourly["hour_sin_future"] = np.where(gap_ok_next, hourly["hour_sin"].shift(-1), np.nan)
hourly["hour_cos_future"] = np.where(gap_ok_next, hourly["hour_cos"].shift(-1), np.nan)
hourly["ghi_clearsky_avg_future"] = np.where(gap_ok_next, hourly["ghi_clearsky_avg"].shift(-1), np.nan)
# Forecast cuaca jam depan TIDAK tersedia saat T (future weather unknown) -- TIDAK dipakai sbg fitur, hanya present_weather di T

gap_ok_prev = hourly["hour_wib"].diff().dt.total_seconds().eq(3600)
hourly["ghi_avg_lag1h"] = np.where(gap_ok_prev, hourly["ghi_avg"].shift(1), np.nan)
hourly["kt_avg_lag1h"] = np.where(gap_ok_prev, hourly["kt_avg"].shift(1), np.nan)
hourly["delta_ghi_1h"] = hourly["ghi_avg"] - hourly["ghi_avg_lag1h"]
hourly["delta_kt_1h"] = hourly["kt_avg"] - hourly["kt_avg_lag1h"]

valid = hourly[(hourly["sun_altitude_avg"] > 5) & (hourly["sun_altitude_avg_future"] > 5) &
               hourly["target"].notna() & (hourly["n_obs"] >= 5) & hourly["ghi_avg_lag1h"].notna()].reset_index(drop=True)
print(f"n_valid={len(valid)}  cakupan present_weather={valid['present_weather'].notna().mean()*100:.1f}%  "
      f"cakupan cloud_low_type={valid['cloud_low_type'].notna().mean()*100:.1f}%")

FEATURES_BASE = ["ghi_avg", "ghi_std", "ghi_min", "ghi_max", "ghi_clearsky_avg", "kt_avg", "kt_std",
                 "sun_altitude_avg", "temp_air_c_avg", "humidity_pct_avg", "wind_speed_ms_avg",
                 "rainfall_mm_sum", "CLOT_mean_avg", "CLTT_mean_avg", "CLTH_mean_avg", "CLER_23_mean_avg",
                 "CLOT_std_avg", "pressure_hpa_avg", "cloud_cover_oktas", "AOD_500nm_avg",
                 "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
                 "sun_altitude_avg_future", "hour_sin_future", "hour_cos_future", "ghi_clearsky_avg_future",
                 "ghi_avg_lag1h", "kt_avg_lag1h", "delta_ghi_1h", "delta_kt_1h"]
FEATURES_SYNOP_NEW = ["present_weather", "past_weather_w1", "cloud_low_type", "cloud_layer2_type",
                      "visibility_km", "is_raining_now", "is_thunderstorm_now", "was_raining_recent"]
TARGET = "target"

for c in FEATURES_BASE + FEATURES_SYNOP_NEW:
    valid[c] = valid[c].astype("float32")
valid[FEATURES_BASE + FEATURES_SYNOP_NEW] = valid[FEATURES_BASE + FEATURES_SYNOP_NEW].fillna(
    valid[FEATURES_BASE + FEATURES_SYNOP_NEW].median())

train = valid[valid["year"].isin([2022, 2023])].reset_index(drop=True)
val = valid[valid["year"] == 2024].reset_index(drop=True)
test = valid[valid["year"] == 2025].reset_index(drop=True)
print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}\n")

def run_ensemble(feats, label):
    m1 = lgb.LGBMRegressor(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, verbosity=-1)
    m1.fit(train[feats], train[TARGET], eval_set=[(val[feats], val[TARGET])],
           callbacks=[lgb.early_stopping(50, verbose=False)])
    m2 = xgb.XGBRegressor(n_estimators=2000, max_depth=8, learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
    m2.fit(train[feats], train[TARGET], eval_set=[(val[feats], val[TARGET])], verbose=False)
    m3 = cb.CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.03, random_state=42,
                               verbose=False, early_stopping_rounds=50)
    m3.fit(train[feats], train[TARGET], eval_set=(val[feats], val[TARGET]))
    p_ens = (m1.predict(test[feats]) + m2.predict(test[feats]) + m3.predict(test[feats])) / 3
    r2 = r2_score(test[TARGET], p_ens)
    mae = mean_absolute_error(test[TARGET], p_ens)
    print(f"{label:55s} n_feat={len(feats):3d}  R2={r2:.4f}  MAE={mae:.2f}")
    return r2, m1

print("=== Baseline hourly-averaged (tanpa SYNOP present_weather dkk) ===")
r2_base, _ = run_ensemble(FEATURES_BASE, "Baseline (cloud_cover_oktas saja)")

print("\n=== + SYNOP present_weather, past_weather, cloud_low_type, visibility, dst ===")
r2_synop, m1_synop = run_ensemble(FEATURES_BASE + FEATURES_SYNOP_NEW, "+ SYNOP weather/cloud detail")

print("\n=== Feature importance fitur SYNOP baru (LightGBM) ===")
imp = pd.DataFrame({"feature": FEATURES_BASE + FEATURES_SYNOP_NEW, "importance": m1_synop.feature_importances_})
imp_synop = imp[imp["feature"].isin(FEATURES_SYNOP_NEW)].sort_values("importance", ascending=False)
print(imp_synop.to_string(index=False))

print(f"\n=== RINGKASAN ===")
print(f"  Baseline (cloud_cover_oktas saja)        R2={r2_base:.4f}")
print(f"  + SYNOP weather/cloud detail              R2={r2_synop:.4f}")
print(f"  Selisih = {r2_synop - r2_base:+.4f}")
