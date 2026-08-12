"""
3 usulan terakhir digabung jadi satu eksperimen:
1. Sample weighting berbasis kualitas data (quality_score, fill_tier, konsistensi CLP/ARP, AERONET)
2. Random search hyperparameter LightGBM (Optuna gagal install -> random search manual, 25 trial)
3. Feature pruning (buang fitur importance~0: rainfall_mm, clp_cloud_present_int, hour_sin/cos_future, angstrom_440_870)
4. Bonus: bandingkan latih-di-kt-lalu-rescale vs latih-langsung-di-raw-GHI
"""
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
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

# --- 1. Sample weight ---
tier_w = {"TIER_0_ORIGINAL": 1.0, "TIER_1_ML_FILLED": 0.85, "TIER_2_ASRS_FILLED": 0.75, "TIER_4_CONSOLIDATED": 0.7}
df["w_tier"] = df["fill_tier"].map(tier_w).fillna(0.6)
df["w_quality"] = 0.5 + 0.5 * df["quality_score"].fillna(0.7)
clp_cloud = df["clp_cloud_present_bool"].fillna(False)
arp_clear = df["arp_clear_sky"].fillna(False)
contradict = (clp_cloud & arp_clear) | (~clp_cloud & ~arp_clear & df["arp_clear_sky"].notna())
df["w_consistency"] = np.where(contradict, 0.7, 1.0)
df["w_aeronet"] = np.where(df["AOD_500nm"].notna(), 1.15, 1.0)
df["sample_weight"] = df["w_tier"] * df["w_quality"] * df["w_consistency"] * df["w_aeronet"]
print(f"sample_weight: mean={df['sample_weight'].mean():.3f} min={df['sample_weight'].min():.3f} max={df['sample_weight'].max():.3f}")

FEATURES_FULL = [
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
DROP_WEAK = ["rainfall_mm", "clp_cloud_present_int", "hour_sin_future", "hour_cos_future", "angstrom_440_870"]
FEATURES_PRUNED = [f for f in FEATURES_FULL if f not in DROP_WEAK]

for c in FEATURES_FULL:
    df[c] = df[c].astype("float32")
df[FEATURES_FULL] = df[FEATURES_FULL].fillna(df[FEATURES_FULL].median())

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)

def fit_eval(feats, use_weight, params, label):
    kwargs = dict(random_state=42, verbosity=-1, **params)
    m = lgb.LGBMRegressor(**kwargs)
    sw = train["sample_weight"].values if use_weight else None
    m.fit(train[feats], train[TARGET], sample_weight=sw,
          eval_set=[(val[feats], val[TARGET])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = m.predict(test[feats])
    r2 = r2_score(test[TARGET], pred)
    mae = mean_absolute_error(test[TARGET], pred)
    print(f"{label:55s} R2={r2:.4f}  MAE={mae:7.2f}")
    return r2, m

BASE_PARAMS = dict(n_estimators=2000, num_leaves=127, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)

print("\n=== Baseline ulang (cek konsistensi) ===")
r2_base, _ = fit_eval(FEATURES_FULL, False, BASE_PARAMS, "Baseline (43 fitur, tanpa weight)")

print("\n=== 1. Sample weighting ===")
r2_w, _ = fit_eval(FEATURES_FULL, True, BASE_PARAMS, "+ Sample weighting kualitas data")

print("\n=== 3. Feature pruning ===")
r2_pruned, _ = fit_eval(FEATURES_PRUNED, False, BASE_PARAMS, f"Pruned ({len(FEATURES_PRUNED)} fitur, buang 5 lemah)")
r2_pruned_w, _ = fit_eval(FEATURES_PRUNED, True, BASE_PARAMS, f"Pruned + weighting")

print("\n=== 2. Random search hyperparameter (25 trial, fitur pruned + weight) ===")
rng = np.random.default_rng(42)
best_r2, best_params, best_model = -np.inf, None, None
results_log = []
for trial in range(25):
    params = dict(
        n_estimators=int(rng.choice([800, 1200, 1600, 2000, 2500])),
        num_leaves=int(rng.choice([31, 63, 95, 127, 180, 255])),
        learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08])),
        subsample=float(rng.uniform(0.6, 0.95)),
        colsample_bytree=float(rng.uniform(0.6, 0.95)),
        min_child_samples=int(rng.choice([10, 20, 40, 80])),
        reg_alpha=float(rng.choice([0, 0.1, 0.5, 1.0])),
        reg_lambda=float(rng.choice([0, 0.1, 0.5, 1.0])),
    )
    m = lgb.LGBMRegressor(random_state=42, verbosity=-1, **params)
    m.fit(train[FEATURES_PRUNED], train[TARGET], sample_weight=train["sample_weight"].values,
          eval_set=[(val[FEATURES_PRUNED], val[TARGET])], callbacks=[lgb.early_stopping(40, verbose=False)])
    pred_val = m.predict(val[FEATURES_PRUNED])
    r2_val = r2_score(val[TARGET], pred_val)
    results_log.append((r2_val, params))
    if r2_val > best_r2:
        best_r2, best_params, best_model = r2_val, params, m

print(f"  Trial terbaik (val R2={best_r2:.4f}): {best_params}")
pred_test_tuned = best_model.predict(test[FEATURES_PRUNED])
r2_tuned = r2_score(test[TARGET], pred_test_tuned)
mae_tuned = mean_absolute_error(test[TARGET], pred_test_tuned)
print(f"{'Tuned (random search 25 trial, pruned+weight)':55s} R2={r2_tuned:.4f}  MAE={mae_tuned:7.2f}")

print("\n=== 4. Bonus: latih-di-kt-lalu-rescale vs langsung-raw-GHI ===")
m_kt = lgb.LGBMRegressor(random_state=42, verbosity=-1, **BASE_PARAMS)
m_kt.fit(train[FEATURES_FULL], train["kt_target_60m"], eval_set=[(val[FEATURES_FULL], val["kt_target_60m"])],
         callbacks=[lgb.early_stopping(50, verbose=False)])
pred_kt = np.clip(m_kt.predict(test[FEATURES_FULL]), 0, 1.5)
pred_kt_rescaled = pred_kt * test["ghi_clearsky_future"].values
r2_kt_rescaled = r2_score(test[TARGET], pred_kt_rescaled)
mae_kt_rescaled = mean_absolute_error(test[TARGET], pred_kt_rescaled)
print(f"{'Latih-di-kt lalu rescale x ghi_clearsky_future':55s} R2={r2_kt_rescaled:.4f}  MAE={mae_kt_rescaled:7.2f}")
print(f"{'Latih-langsung-raw-GHI (baseline awal)':55s} R2=0.7234  MAE=121.34")

print("\n=== RINGKASAN SEMUA ===")
summary = [
    ("Baseline (43 fitur)", r2_base),
    ("+ Sample weighting", r2_w),
    ("Feature pruned (38 fitur)", r2_pruned),
    ("Pruned + weighting", r2_pruned_w),
    ("Tuned (random search + pruned + weight)", r2_tuned),
    ("Latih-di-kt lalu rescale", r2_kt_rescaled),
    ("Ensemble 3-boosting (sesi lalu)", 0.7264),
]
for name, r2 in sorted(summary, key=lambda x: -x[1]):
    print(f"  {name:45s} R2={r2:.4f}")
