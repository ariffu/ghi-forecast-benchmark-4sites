"""
Usaha terakhir menembus R2=0.9 di resolusi 10-menit GENUINE (bukan trik hourly-aggregation
atau multi-horizon-averaging yang sudah dibongkar di CAMEL AI / GHI Vault audit).
Teknik yang BELUM dicoba sebelumnya di sesi ini:
1. CatBoost dengan native categorical features (hour_bucket, season, ghi_level, weather_pattern)
2. Filter kualitas lebih ketat (TIER_0/1 + quality_score>=0.9) -- sama resolusi, subset lebih bersih
3. Mega-ensemble: gabung semua model terbaik (boosting tuned+weighted + catboost categorical)
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

# Sample weight (sama seperti eksperimen sebelumnya)
tier_w = {"TIER_0_ORIGINAL": 1.0, "TIER_1_ML_FILLED": 0.85, "TIER_2_ASRS_FILLED": 0.75, "TIER_4_CONSOLIDATED": 0.7}
df["w_tier"] = df["fill_tier"].map(tier_w).fillna(0.6)
df["w_quality"] = 0.5 + 0.5 * df["quality_score"].fillna(0.7)
clp_cloud = df["clp_cloud_present_bool"].fillna(False)
arp_clear = df["arp_clear_sky"].fillna(False)
contradict = (clp_cloud & arp_clear) | (~clp_cloud & ~arp_clear & df["arp_clear_sky"].notna())
df["w_consistency"] = np.where(contradict, 0.7, 1.0)
df["w_aeronet"] = np.where(df["AOD_500nm"].notna(), 1.15, 1.0)
df["sample_weight"] = df["w_tier"] * df["w_quality"] * df["w_consistency"] * df["w_aeronet"]

# --- Fitur kategorikal baru (belum pernah dicoba) ---
df["hour"] = df["timestamp_wib"].dt.hour
df["hour_bucket"] = pd.cut(df["hour"], bins=[0, 8, 11, 14, 17, 24],
                            labels=["pagi_awal", "pagi", "siang", "sore_awal", "sore"], include_lowest=True).astype(str)
df["season"] = df["month"].map({12: "hujan", 1: "hujan", 2: "hujan", 3: "transisi", 4: "transisi",
                                  5: "kering", 6: "kering", 7: "kering", 8: "kering", 9: "transisi",
                                  10: "transisi", 11: "hujan"})
df["ghi_level"] = pd.cut(df["ghi_final"], bins=[-1, 100, 300, 600, 2000],
                          labels=["sangat_rendah", "rendah", "sedang", "tinggi"]).astype(str)
df["weather_pattern"] = np.where(
    df["CLOT_mean"] < 5, "cerah", np.where(df["CLOT_mean"] < 15, "sebagian_berawan", "berawan_tebal"))

FEATURES_NUM = [
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
FEATURES_CAT = ["hour_bucket", "season", "ghi_level", "weather_pattern"]
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT

for c in FEATURES_NUM:
    df[c] = df[c].astype("float32")
df[FEATURES_NUM] = df[FEATURES_NUM].fillna(df[FEATURES_NUM].median())
for c in FEATURES_CAT:
    df[c] = df[c].astype(str)

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)

print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}\n")

# --- 1. CatBoost dengan native categorical ---
cat_idx = [FEATURES_ALL.index(c) for c in FEATURES_CAT]
m_cat = cb.CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.02, random_state=42,
                              verbose=False, early_stopping_rounds=80, cat_features=cat_idx)
m_cat.fit(train[FEATURES_ALL], train[TARGET], sample_weight=train["sample_weight"].values,
          eval_set=(val[FEATURES_ALL], val[TARGET]))
pred_cat = m_cat.predict(test[FEATURES_ALL])
r2_cat = r2_score(test[TARGET], pred_cat)
print(f"{'CatBoost + categorical native (hour_bucket dkk)':50s} R2={r2_cat:.4f}  MAE={mean_absolute_error(test[TARGET], pred_cat):.2f}")

# --- 2. Filter kualitas lebih ketat (TIER_0/1 + quality>=0.9), sama resolusi 10-menit ---
strict = df[(df["fill_tier"].isin(["TIER_0_ORIGINAL", "TIER_1_ML_FILLED"])) &
            (df["fill_tier_target"].isin(["TIER_0_ORIGINAL", "TIER_1_ML_FILLED"])) &
            (df["quality_score"] >= 0.9)].reset_index(drop=True)
train_s = strict[strict["year"].isin([2022, 2023])].reset_index(drop=True)
val_s = strict[strict["year"] == 2024].reset_index(drop=True)
test_s = strict[strict["year"] == 2025].reset_index(drop=True)
print(f"\nSubset ketat (TIER_0/1 + quality>=0.9): n_train={len(train_s)} n_val={len(val_s)} n_test={len(test_s)} "
      f"({len(strict)}/{len(df)} = {len(strict)/len(df)*100:.1f}% dari anchor_valid)")

m_strict = lgb.LGBMRegressor(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, verbosity=-1)
m_strict.fit(train_s[FEATURES_NUM], train_s[TARGET], eval_set=[(val_s[FEATURES_NUM], val_s[TARGET])],
             callbacks=[lgb.early_stopping(50, verbose=False)])
pred_strict = m_strict.predict(test_s[FEATURES_NUM])
r2_strict = r2_score(test_s[TARGET], pred_strict)
print(f"{'LightGBM, subset TIER_0/1+quality>=0.9 (resolusi tetap 10-menit)':50s} R2={r2_strict:.4f}  MAE={mean_absolute_error(test_s[TARGET], pred_strict):.2f}")

# --- 3. Mega-ensemble: boosting tuned+weighted (full features) + catboost categorical ---
tuned_params = dict(n_estimators=2000, num_leaves=63, learning_rate=0.02, subsample=0.615,
                     colsample_bytree=0.654, min_child_samples=20, reg_alpha=0.5, reg_lambda=0.5)
m1 = lgb.LGBMRegressor(random_state=42, verbosity=-1, **tuned_params)
m1.fit(train[FEATURES_NUM], train[TARGET], sample_weight=train["sample_weight"].values,
       eval_set=[(val[FEATURES_NUM], val[TARGET])], callbacks=[lgb.early_stopping(50, verbose=False)])
m2 = xgb.XGBRegressor(n_estimators=2000, max_depth=8, learning_rate=0.03, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, early_stopping_rounds=50, eval_metric="rmse")
m2.fit(train[FEATURES_NUM], train[TARGET], sample_weight=train["sample_weight"].values,
       eval_set=[(val[FEATURES_NUM], val[TARGET])], verbose=False)
m3 = cb.CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.03, random_state=42,
                           verbose=False, early_stopping_rounds=50)
m3.fit(train[FEATURES_NUM], train[TARGET], sample_weight=train["sample_weight"].values,
       eval_set=(val[FEATURES_NUM], val[TARGET]))

p1, p2, p3 = m1.predict(test[FEATURES_NUM]), m2.predict(test[FEATURES_NUM]), m3.predict(test[FEATURES_NUM])
p_mega = (p1 + p2 + p3 + pred_cat) / 4
r2_mega = r2_score(test[TARGET], p_mega)
mae_mega = mean_absolute_error(test[TARGET], p_mega)
print(f"\n{'MEGA-ENSEMBLE (LGBM+XGB+CatBoost+CatBoost-categorical)':50s} R2={r2_mega:.4f}  MAE={mae_mega:.2f}")

print("\n=== RINGKASAN ===")
print(f"  Model final sebelumnya (note_13)                        R2=0.7270")
print(f"  CatBoost + categorical native                            R2={r2_cat:.4f}")
print(f"  Subset ketat TIER_0/1+quality>=0.9 (tetap 10-menit)       R2={r2_strict:.4f}")
print(f"  Mega-ensemble 4 model                                     R2={r2_mega:.4f}")
print(f"\nTarget R2=0.90 -- {'TERCAPAI' if max(r2_cat, r2_strict, r2_mega) >= 0.9 else 'TIDAK TERCAPAI'}")
