"""
Uji tren: apakah memperbesar jendela agregasi (1h, 2h, 3h, 6h) dan memprediksi
rata-rata jendela BERIKUTNYA terus menaikkan R2? Mekanisme sama dengan eksperimen
hourly-averaged sebelumnya (averaging membatalkan noise) -- pertanyaannya apakah
makin besar jendela = makin tinggi R2 secara konsisten, dan apa konsekuensi
praktisnya (horizon riil yang diprediksi makin kasar/tidak granular).
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

def run_window(window_hours, label):
    freq = f"{window_hours}h"
    d = df.copy()
    d["bin"] = d["timestamp_wib"].dt.floor(freq)
    agg = d.groupby("bin").agg(
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
    ).reset_index().sort_values("bin").reset_index(drop=True)

    agg["hour"] = agg["bin"].dt.hour
    agg["doy"] = agg["bin"].dt.dayofyear
    agg["month"] = agg["bin"].dt.month
    agg["year"] = agg["bin"].dt.year
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24)
    agg["doy_sin"] = np.sin(2 * np.pi * agg["doy"] / 365)
    agg["doy_cos"] = np.cos(2 * np.pi * agg["doy"] / 365)

    step_sec = window_hours * 3600
    gap_ok_next = agg["bin"].diff().shift(-1).dt.total_seconds().eq(step_sec)
    agg["target"] = np.where(gap_ok_next, agg["ghi_avg"].shift(-1), np.nan)
    agg["sun_altitude_avg_future"] = np.where(gap_ok_next, agg["sun_altitude_avg"].shift(-1), np.nan)
    agg["hour_sin_future"] = np.where(gap_ok_next, agg["hour_sin"].shift(-1), np.nan)
    agg["hour_cos_future"] = np.where(gap_ok_next, agg["hour_cos"].shift(-1), np.nan)
    agg["ghi_clearsky_avg_future"] = np.where(gap_ok_next, agg["ghi_clearsky_avg"].shift(-1), np.nan)

    gap_ok_prev = agg["bin"].diff().dt.total_seconds().eq(step_sec)
    agg["ghi_avg_lag"] = np.where(gap_ok_prev, agg["ghi_avg"].shift(1), np.nan)
    agg["kt_avg_lag"] = np.where(gap_ok_prev, agg["kt_avg"].shift(1), np.nan)
    agg["delta_ghi"] = agg["ghi_avg"] - agg["ghi_avg_lag"]
    agg["delta_kt"] = agg["kt_avg"] - agg["kt_avg_lag"]

    valid_mask = ((agg["sun_altitude_avg"] > 5) & (agg["sun_altitude_avg_future"] > 5) &
                  agg["target"].notna() & (agg["n_obs"] >= 5) & agg["ghi_avg_lag"].notna())
    valid = agg[valid_mask].reset_index(drop=True)

    FEATURES = ["ghi_avg", "ghi_std", "ghi_min", "ghi_max", "ghi_clearsky_avg", "kt_avg", "kt_std",
                "sun_altitude_avg", "temp_air_c_avg", "humidity_pct_avg", "wind_speed_ms_avg",
                "rainfall_mm_sum", "CLOT_mean_avg", "CLTT_mean_avg", "CLTH_mean_avg", "CLER_23_mean_avg",
                "CLOT_std_avg", "pressure_hpa_avg", "cloud_cover_oktas_avg", "AOD_500nm_avg",
                "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
                "sun_altitude_avg_future", "hour_sin_future", "hour_cos_future", "ghi_clearsky_avg_future",
                "ghi_avg_lag", "kt_avg_lag", "delta_ghi", "delta_kt"]
    TARGET = "target"
    for c in FEATURES:
        valid[c] = valid[c].astype("float32")
    valid[FEATURES] = valid[FEATURES].fillna(valid[FEATURES].median())

    train = valid[valid["year"].isin([2022, 2023])].reset_index(drop=True)
    val = valid[valid["year"] == 2024].reset_index(drop=True)
    test = valid[valid["year"] == 2025].reset_index(drop=True)
    if len(train) < 50 or len(test) < 20:
        print(f"  [{label}] data terlalu sedikit (train={len(train)}, test={len(test)}), skip")
        return None

    m1 = lgb.LGBMRegressor(n_estimators=1500, num_leaves=63, learning_rate=0.02, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, verbosity=-1)
    m1.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
           callbacks=[lgb.early_stopping(50, verbose=False)])
    m2 = xgb.XGBRegressor(n_estimators=1500, max_depth=7, learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
    m2.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])], verbose=False)
    m3 = cb.CatBoostRegressor(iterations=1500, depth=7, learning_rate=0.03, random_state=42,
                               verbose=False, early_stopping_rounds=50)
    m3.fit(train[FEATURES], train[TARGET], eval_set=(val[FEATURES], val[TARGET]))

    p_ens = (m1.predict(test[FEATURES]) + m2.predict(test[FEATURES]) + m3.predict(test[FEATURES])) / 3
    r2 = r2_score(test[TARGET], p_ens)
    mae = mean_absolute_error(test[TARGET], p_ens)
    print(f"  [{label}] n_train={len(train):6d} n_test={len(test):5d}  R2={r2:.4f}  MAE={mae:6.2f}")
    return dict(window=label, r2=r2, mae=mae, n_test=len(test))

print("=== Tren R2 vs ukuran jendela agregasi (prediksi rata-rata jendela berikutnya) ===\n")
results = []
for w, label in [(1, "1 jam"), (2, "2 jam"), (3, "3 jam"), (6, "6 jam")]:
    res = run_window(w, label)
    if res:
        results.append(res)

print("\n=== RINGKASAN ===")
print(f"  {'Jendela':10s} {'n_test':>8s} {'R2':>8s} {'MAE':>8s}")
for r in results:
    print(f"  {r['window']:10s} {r['n_test']:8d} {r['r2']:8.4f} {r['mae']:8.2f}")
print(f"\n  (Pembanding) 10-menit direct genuine     R2=0.7270  MAE=120.35")
