"""
Resolusi per-jam seperti CAMEL AI, TAPI bukan rata-rata 6 titik 10-menit dalam jam
tersebut (yang menghaluskan noise) -- melainkan ambil titik observasi PERSIS di jam
bulat (menit=0) saja. Ini mengubah cadence sampling, BUKAN menghaluskan data.
Target tetap ghi_target_60m (nilai aktual T+60, bukan rata-rata).
"""
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import r2_score, mean_absolute_error

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
df["minute"] = df["timestamp_wib"].dt.minute

print(f"Total anchor_valid (semua menit, 10-menit cadence): {len(df)}")
df_hourly = df[df["minute"] == 0].reset_index(drop=True)
print(f"Subset titik observasi tepat jam bulat (menit=0): {len(df_hourly)} "
      f"({len(df_hourly)/len(df)*100:.1f}% dari total -- harus ~1/6 jika seimbang)")

tier_w = {"TIER_0_ORIGINAL": 1.0, "TIER_1_ML_FILLED": 0.85, "TIER_2_ASRS_FILLED": 0.75, "TIER_4_CONSOLIDATED": 0.7}

FEATURES = [
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

def prep(frame):
    frame = frame.copy()
    frame["w_tier"] = frame["fill_tier"].map(tier_w).fillna(0.6)
    frame["w_quality"] = 0.5 + 0.5 * frame["quality_score"].fillna(0.7)
    clp_cloud = frame["clp_cloud_present_bool"].fillna(False)
    arp_clear = frame["arp_clear_sky"].fillna(False)
    contradict = (clp_cloud & arp_clear) | (~clp_cloud & ~arp_clear & frame["arp_clear_sky"].notna())
    frame["w_consistency"] = np.where(contradict, 0.7, 1.0)
    frame["w_aeronet"] = np.where(frame["AOD_500nm"].notna(), 1.15, 1.0)
    frame["sample_weight"] = frame["w_tier"] * frame["w_quality"] * frame["w_consistency"] * frame["w_aeronet"]
    for c in FEATURES:
        frame[c] = frame[c].astype("float32")
    frame[FEATURES] = frame[FEATURES].fillna(frame[FEATURES].median())
    return frame

def run(frame, label):
    frame = prep(frame)
    train = frame[frame["year"].isin([2022, 2023])].reset_index(drop=True)
    val = frame[frame["year"] == 2024].reset_index(drop=True)
    test = frame[frame["year"] == 2025].reset_index(drop=True)
    print(f"  [{label}] n_train={len(train)} n_val={len(val)} n_test={len(test)}")

    tuned_params = dict(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.615,
                         colsample_bytree=0.654, min_child_samples=20, reg_alpha=0.5, reg_lambda=0.5)
    m1 = lgb.LGBMRegressor(random_state=42, verbosity=-1, **tuned_params)
    m1.fit(train[FEATURES], train[TARGET], sample_weight=train["sample_weight"].values,
           eval_set=[(val[FEATURES], val[TARGET])], callbacks=[lgb.early_stopping(50, verbose=False)])
    m2 = xgb.XGBRegressor(n_estimators=2000, max_depth=8, learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
    m2.fit(train[FEATURES], train[TARGET], sample_weight=train["sample_weight"].values,
           eval_set=[(val[FEATURES], val[TARGET])], verbose=False)
    m3 = cb.CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.03, random_state=42,
                               verbose=False, early_stopping_rounds=50)
    m3.fit(train[FEATURES], train[TARGET], sample_weight=train["sample_weight"].values,
           eval_set=(val[FEATURES], val[TARGET]))

    p1, p2, p3 = m1.predict(test[FEATURES]), m2.predict(test[FEATURES]), m3.predict(test[FEATURES])
    p_ens = (p1 + p2 + p3) / 3
    r2 = r2_score(test[TARGET], p_ens)
    mae = mean_absolute_error(test[TARGET], p_ens)
    print(f"  [{label}] Ensemble R2={r2:.4f}  MAE={mae:.2f}")
    return r2, mae, len(test)

print("\n=== A. Resolusi 10-menit penuh (semua menit, pembanding) ===")
r2_full, mae_full, n_full = run(df, "10-menit penuh")

print("\n=== B. Resolusi per-jam, titik observasi tepat jam bulat (menit=0), TANPA averaging ===")
r2_hourly_point, mae_hourly_point, n_hourly = run(df_hourly, "Jam bulat (point sample)")

print("\n=== RINGKASAN ===")
print(f"  10-menit penuh (semua menit)        n_test={n_full:6d}  R2={r2_full:.4f}  MAE={mae_full:.2f}")
print(f"  Per-jam, point sample (menit=0)      n_test={n_hourly:6d}  R2={r2_hourly_point:.4f}  MAE={mae_hourly_point:.2f}")
print(f"  Selisih R2 = {r2_hourly_point - r2_full:+.4f}")
print(f"\n  (Pembanding) CAMEL AI hourly-AVERAGED (smoothing 6 titik/jam): R2=0.8066")
