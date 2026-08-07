"""
Eksperimen: Tambah fitur CLP baru ke training_ghi_1h_direct
Fitur baru:
  - CLOT_std          : variabilitas spasial CLOT dalam buffer (dari clp_pontianak)
  - CLER_23_coverage  : fraksi pixel CLER valid — proxy "awan aktif" (dari clp_pontianak)
  - cloud_class_code  : kelas awan ordinal -1..3 (dari clp_pontianak)
  - CLOT_median       : median CLOT lebih robust dari mean (dari clp_pontianak)
  - clot_trend_30m    : CLOT_mean(t) - CLOT_mean(t-30m) — awan menebal/menipis
  - clot_std_roll30m  : rolling std CLOT_mean 30 menit terakhir — volatilitas temporal
Target: ghi_target_avg60m (model hybrid, baseline R²=0.8606)
"""

import sys, duckdb
import pandas as pd
import numpy as np
import pickle
sys.stdout.reconfigure(encoding="utf-8")
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"

# ── 1. Tambah kolom baru via DuckDB CLI (langsung di sini via duckdb python) ──
print("=" * 60)
print("STEP 1: Tambah kolom baru ke training_ghi_1h_direct")
print("=" * 60)

con = duckdb.connect(DB_PATH)

new_cols = [
    ("CLOT_std",         "DOUBLE"),
    ("CLER_23_coverage", "DOUBLE"),
    ("cloud_class_code", "DOUBLE"),
    ("CLOT_median",      "DOUBLE"),
    ("clot_trend_30m",   "DOUBLE"),
    ("clot_std_roll30m", "DOUBLE"),
]

existing = set(r[0] for r in con.execute(
    "SELECT column_name FROM information_schema.columns WHERE table_name='training_ghi_1h_direct'"
).fetchall())

for col, dtype in new_cols:
    if col not in existing:
        con.execute(f"ALTER TABLE training_ghi_1h_direct ADD COLUMN {col} {dtype}")
        print(f"  + kolom '{col}' ditambahkan")
    else:
        print(f"  ~ kolom '{col}' sudah ada, dilewati")

# ── 2. Isi dari clp_pontianak (join langsung) ──
print("\nSTEP 2: Isi CLOT_std, CLER_23_coverage, cloud_class_code, CLOT_median dari CLP 30km")

con.execute("""
UPDATE training_ghi_1h_direct t
SET
    CLOT_std         = c.CLOT_std,
    CLER_23_coverage = c.CLER_23_coverage,
    cloud_class_code = c.cloud_class_code,
    CLOT_median      = c.CLOT_median
FROM clp_pontianak c
WHERE t.timestamp_wib = c.timestamp_wib
""")
print("  OK: 4 kolom diisi dari clp_pontianak")

# ── 3. Hitung fitur temporal dari CLOT_mean (sudah ada) ──
print("\nSTEP 3: Hitung clot_trend_30m dan clot_std_roll30m via Python rolling")

df_clot = con.execute("""
    SELECT timestamp_wib, CLOT_mean
    FROM training_ghi_1h_direct
    ORDER BY timestamp_wib
""").df()

df_clot = df_clot.sort_values("timestamp_wib").reset_index(drop=True)
df_clot["clot_trend_30m"]   = df_clot["CLOT_mean"] - df_clot["CLOT_mean"].shift(3)  # t - t-30m (3x10mnt)
df_clot["clot_std_roll30m"] = df_clot["CLOT_mean"].rolling(window=4, min_periods=2).std()

# Pastikan tidak ada bocor lintas gap waktu (cek gap > 10 mnt)
ts = pd.to_datetime(df_clot["timestamp_wib"])
gap_mask = (ts.diff() > pd.Timedelta("10min"))
df_clot.loc[gap_mask, "clot_trend_30m"]   = np.nan
df_clot.loc[gap_mask, "clot_std_roll30m"] = np.nan

print(f"  clot_trend_30m   non-null: {df_clot['clot_trend_30m'].notna().sum():,}")
print(f"  clot_std_roll30m non-null: {df_clot['clot_std_roll30m'].notna().sum():,}")

# Update ke database via temp table
con.execute("CREATE OR REPLACE TEMP TABLE tmp_clot_roll AS SELECT * FROM df_clot")
con.execute("""
UPDATE training_ghi_1h_direct t
SET
    clot_trend_30m   = r.clot_trend_30m,
    clot_std_roll30m = r.clot_std_roll30m
FROM tmp_clot_roll r
WHERE t.timestamp_wib = r.timestamp_wib
""")
print("  OK: clot_trend_30m dan clot_std_roll30m diisi")

# ── 4. Verifikasi null rate ──
print("\nSTEP 4: Verifikasi null rate (anchor_valid & sun_altitude>5)")
null_check = con.execute("""
    SELECT
        COUNT(*) AS n,
        ROUND(100.0*COUNT(CLOT_std)/COUNT(*),1)         AS pct_clot_std,
        ROUND(100.0*COUNT(CLER_23_coverage)/COUNT(*),1) AS pct_cler_cov,
        ROUND(100.0*COUNT(cloud_class_code)/COUNT(*),1) AS pct_class_code,
        ROUND(100.0*COUNT(CLOT_median)/COUNT(*),1)      AS pct_clot_med,
        ROUND(100.0*COUNT(clot_trend_30m)/COUNT(*),1)   AS pct_trend,
        ROUND(100.0*COUNT(clot_std_roll30m)/COUNT(*),1) AS pct_roll_std
    FROM training_ghi_1h_direct
    WHERE anchor_valid AND sun_altitude > 5
""").fetchone()
print(f"  N anchor_valid+day = {null_check[0]:,}")
print(f"  CLOT_std:          {null_check[1]}%")
print(f"  CLER_23_coverage:  {null_check[2]}%")
print(f"  cloud_class_code:  {null_check[3]}%")
print(f"  CLOT_median:       {null_check[4]}%")
print(f"  clot_trend_30m:    {null_check[5]}%")
print(f"  clot_std_roll30m:  {null_check[6]}%")

# ── 5. Load data training ──
print("\nSTEP 5: Load data training")
df = con.execute("""
    SELECT *
    FROM training_ghi_1h_direct
    WHERE anchor_valid
    ORDER BY timestamp_wib
""").df()
con.close()

print(f"  Total baris anchor_valid: {len(df):,}")

# ── 6. Definisi fitur ──
# Fitur baseline (43 fitur dari model sebelumnya)
BASELINE_FEATURES = [
    "sun_altitude", "sun_altitude_future", "ghi_clearsky", "ghi_clearsky_future",
    "hour_sin", "hour_cos", "hour_sin_future", "hour_cos_future",
    "doy_sin", "doy_cos",
    "ghi_final", "kt",
    "ghi_lag10m", "kt_lag10m", "ghi_lag20m", "kt_lag20m",
    "ghi_lag30m", "kt_lag30m", "ghi_lag60m", "kt_lag60m",
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std", "kt_roll60m_std",
    "delta_kt_10m", "delta_kt_30m", "delta_ghi_30m",
    "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean",
    "clp_cloud_present_int", "clot_lag10m", "clot_lag30m", "delta_clot_30m",
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm",
    "AOD_500nm", "angstrom_440_870",
]

# Fitur baru CLP
NEW_CLP_FEATURES = [
    "CLOT_std",
    "CLER_23_coverage",
    "cloud_class_code",
    "CLOT_median",
    "clot_trend_30m",
    "clot_std_roll30m",
]

ALL_FEATURES = BASELINE_FEATURES + NEW_CLP_FEATURES

# Filter fitur yang benar-benar ada di dataframe
baseline_ok = [f for f in BASELINE_FEATURES if f in df.columns]
new_ok      = [f for f in NEW_CLP_FEATURES  if f in df.columns]
all_ok      = baseline_ok + new_ok

print(f"  Baseline features: {len(baseline_ok)}/{len(BASELINE_FEATURES)}")
print(f"  New CLP features:  {len(new_ok)}/{len(NEW_CLP_FEATURES)}")
print(f"  Total features:    {len(all_ok)}")

TARGET = "ghi_target_avg60m"

# ── 7. Train/test split temporal ──
df["year"] = pd.to_datetime(df["timestamp_wib"]).dt.year
train = df[df["year"] <= 2023].copy()
val   = df[df["year"] == 2024].copy()
test  = df[df["year"] == 2025].copy()

print(f"\n  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# Drop baris dengan target NaN
train = train.dropna(subset=[TARGET])
val   = val.dropna(subset=[TARGET])
test  = test.dropna(subset=[TARGET])

# ── 8. Sample weights ──
def make_weights(df_):
    tier_map = {
        "TIER_0_ORIGINAL": 1.0,
        "TIER_1_ML_FILLED": 0.85,
        "TIER_4_CONSOLIDATED": 0.70,
    }
    w = df_["fill_tier"].map(tier_map).fillna(0.5)
    if "quality_score" in df_.columns:
        w = w * df_["quality_score"].fillna(0.7)
    return w.values

w_train = make_weights(train)

# ── 9. Latih model BASELINE (43 fitur) ──
print("\nSTEP 6: Latih model BASELINE (43 fitur)")

def fill_median(df_, feats):
    X = df_[feats].copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
    return X

X_train_base = fill_median(train, baseline_ok)
X_val_base   = fill_median(val,   baseline_ok)
X_test_base  = fill_median(test,  baseline_ok)
y_train = train[TARGET].values
y_val   = val[TARGET].values
y_test  = test[TARGET].values

lgb_base = LGBMRegressor(
    n_estimators=1200, learning_rate=0.03, num_leaves=63,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
)
xgb_base = XGBRegressor(
    n_estimators=1000, learning_rate=0.04, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)
cat_base = CatBoostRegressor(
    iterations=1000, learning_rate=0.04, depth=6,
    l2_leaf_reg=3, subsample=0.8, random_seed=42, verbose=0
)

lgb_base.fit(X_train_base, y_train, sample_weight=w_train,
             eval_set=[(X_val_base, y_val)])
xgb_base.fit(X_train_base, y_train, sample_weight=w_train,
             eval_set=[(X_val_base, y_val)], verbose=False)
cat_base.fit(X_train_base, y_train, sample_weight=w_train,
             eval_set=(X_val_base, y_val))

pred_base = (lgb_base.predict(X_test_base) +
             xgb_base.predict(X_test_base) +
             cat_base.predict(X_test_base)) / 3

r2_base  = r2_score(y_test, pred_base)
mae_base = mean_absolute_error(y_test, pred_base)
rmse_base= np.sqrt(mean_squared_error(y_test, pred_base))
print(f"  BASELINE → R²={r2_base:.4f}, MAE={mae_base:.2f}, RMSE={rmse_base:.2f}")

# ── 10. Latih model ENHANCED (+ 6 fitur CLP baru) ──
print(f"\nSTEP 7: Latih model ENHANCED (+{len(new_ok)} fitur CLP baru)")

X_train_new = fill_median(train, all_ok)
X_val_new   = fill_median(val,   all_ok)
X_test_new  = fill_median(test,  all_ok)

lgb_new = LGBMRegressor(
    n_estimators=1200, learning_rate=0.03, num_leaves=63,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
)
xgb_new = XGBRegressor(
    n_estimators=1000, learning_rate=0.04, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)
cat_new = CatBoostRegressor(
    iterations=1000, learning_rate=0.04, depth=6,
    l2_leaf_reg=3, subsample=0.8, random_seed=42, verbose=0
)

lgb_new.fit(X_train_new, y_train, sample_weight=w_train,
            eval_set=[(X_val_new, y_val)])
xgb_new.fit(X_train_new, y_train, sample_weight=w_train,
            eval_set=[(X_val_new, y_val)], verbose=False)
cat_new.fit(X_train_new, y_train, sample_weight=w_train,
            eval_set=(X_val_new, y_val))

pred_new = (lgb_new.predict(X_test_new) +
            xgb_new.predict(X_test_new) +
            cat_new.predict(X_test_new)) / 3

r2_new   = r2_score(y_test, pred_new)
mae_new  = mean_absolute_error(y_test, pred_new)
rmse_new = np.sqrt(mean_squared_error(y_test, pred_new))
print(f"  ENHANCED → R²={r2_new:.4f}, MAE={mae_new:.2f}, RMSE={rmse_new:.2f}")

# ── 11. Feature importance fitur CLP baru ──
print("\nSTEP 8: Feature importance fitur CLP baru (LightGBM)")
fi = pd.Series(lgb_new.feature_importances_, index=all_ok).sort_values(ascending=False)
print("  Top 20 fitur:")
for i, (feat, imp) in enumerate(fi.head(20).items()):
    tag = " ← BARU" if feat in new_ok else ""
    print(f"    {i+1:2d}. {feat:<30} {imp:>6}{tag}")

print("\n  Fitur CLP baru — rank & importance:")
for feat in new_ok:
    rank = fi.index.get_loc(feat) + 1
    print(f"    rank #{rank:2d}: {feat:<25} imp={fi[feat]}")

# ── 12. Ablasi: kontribusi tiap fitur baru ──
print("\nSTEP 9: Ablasi — kontribusi per fitur baru (tambah satu per satu)")
cumulative = list(baseline_ok)
r2_prev = r2_base
for feat in new_ok:
    cumulative.append(feat)
    Xtr = fill_median(train, cumulative)
    Xte = fill_median(test,  cumulative)
    m = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(Xtr, y_train, sample_weight=w_train)
    r2_i = r2_score(y_test, m.predict(Xte))
    delta = r2_i - r2_prev
    print(f"  +{feat:<25} → R²={r2_i:.4f}  Δ={delta:+.4f}")
    r2_prev = r2_i

# ── 13. Ringkasan ──
print("\n" + "=" * 60)
print("RINGKASAN HASIL")
print("=" * 60)
print(f"  Baseline  (43 fitur): R²={r2_base:.4f}, MAE={mae_base:.2f}, RMSE={rmse_base:.2f}")
print(f"  Enhanced  ({len(all_ok)} fitur): R²={r2_new:.4f},  MAE={mae_new:.2f},  RMSE={rmse_new:.2f}")
print(f"  Delta R²: {r2_new - r2_base:+.4f}")
print(f"  Delta MAE: {mae_new - mae_base:+.2f} W/m²")

# ── 14. Simpan model enhanced jika lebih baik ──
if r2_new > r2_base:
    save = {
        "models": {"lightgbm": lgb_new, "xgboost": xgb_new, "catboost": cat_new},
        "features": all_ok,
        "target": TARGET,
        "r2_test": r2_new,
        "mae_test": mae_new,
        "rmse_test": rmse_new,
        "n_features": len(all_ok),
        "new_features": new_ok,
        "baseline_r2": r2_base,
    }
    out = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_enhanced.pkl"
    with open(out, "wb") as f:
        pickle.dump(save, f)
    print(f"\n  Model enhanced DISIMPAN → {out}")
else:
    print("\n  Model enhanced TIDAK disimpan (tidak lebih baik dari baseline)")

print("\nSelesai.")
