"""
Production Training v3b — HPO independen per model
Perbaikan dari v3: tiap model (LGB, XGB, CatBoost) dioptimasi sendiri
Fitur: 49 fitur (v3), target: ghi_target_avg60m
"""

import sys, duckdb, pickle, random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sys.stdout.reconfigure(encoding="utf-8")

DB  = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
OUT = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3b.pkl"

PROD_FEATURES = [
    "ghi_final", "kt", "ghi_clearsky", "ghi_clearsky_future",
    "sun_altitude", "sun_altitude_future",
    "hour_sin", "hour_cos", "hour_sin_future", "hour_cos_future",
    "doy_sin", "doy_cos", "month",
    "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m",
    "kt_lag10m", "kt_lag20m", "kt_lag30m", "kt_lag60m",
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std",
    "delta_kt_10m", "delta_kt_30m", "delta_ghi_30m",
    "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean",
    "clp_cloud_present_int",
    "clot_lag10m", "clot_lag30m", "delta_clot_30m",
    "CLOT_std", "CLER_23_std",
    "CLER_23_coverage", "CLOT_median", "clot_std_roll30m",
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm",
    "pressure_hpa", "cloud_cover_oktas",
    "AOD_500nm", "angstrom_440_870", "precipitable_water_cm",
]

print("=" * 65)
print("Load data training")
con = duckdb.connect(DB, read_only=True)
df = con.execute("""
    SELECT * FROM training_ghi_1h_direct WHERE anchor_valid ORDER BY timestamp_wib
""").df()
con.close()

features_ok = [f for f in PROD_FEATURES if f in df.columns]
print(f"  Fitur OK: {len(features_ok)}/{len(PROD_FEATURES)}")

TARGET = "ghi_target_avg60m"
df["_year"] = pd.to_datetime(df["timestamp_wib"]).dt.year
train = df[df["_year"] <= 2023].dropna(subset=[TARGET]).copy()
val   = df[df["_year"] == 2024].dropna(subset=[TARGET]).copy()
test  = df[df["_year"] == 2025].dropna(subset=[TARGET]).copy()
print(f"  Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")

tier_w = {"TIER_0_ORIGINAL":1.0,"TIER_1_ML_FILLED":0.85,"TIER_4_CONSOLIDATED":0.70}
w_train = (train["fill_tier"].map(tier_w).fillna(0.5) *
           train["quality_score"].fillna(0.7)).values

def impute(df_, feats):
    X = df_[feats].copy()
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    return X

X_tr = impute(train, features_ok)
X_va = impute(val,   features_ok)
X_te = impute(test,  features_ok)
y_tr, y_va, y_te = train[TARGET].values, val[TARGET].values, test[TARGET].values

random.seed(42); np.random.seed(42)

# ── HPO LightGBM ──────────────────────────────────────────────────────────────
print("\nHPO LightGBM (15 trial)")
best_lgb_p, best_lgb_r2 = None, -9
for i in range(15):
    p = dict(
        n_estimators      = random.choice([1000,1200,1500,2000]),
        learning_rate     = random.choice([0.02,0.03,0.04,0.05]),
        num_leaves        = random.choice([63,95,127]),
        min_child_samples = random.choice([20,30,50]),
        subsample         = random.uniform(0.7,0.95),
        colsample_bytree  = random.uniform(0.65,0.9),
        reg_alpha         = random.choice([0.0,0.05,0.1]),
        reg_lambda        = random.choice([0.5,1.0,2.0]),
    )
    m = LGBMRegressor(**p, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr, sample_weight=w_train, eval_set=[(X_va, y_va)])
    r2 = r2_score(y_va, m.predict(X_va))
    if r2 > best_lgb_r2:
        best_lgb_r2, best_lgb_p = r2, p
        print(f"  trial {i+1:2d}: R²_val={r2:.4f} *** best  leaves={p['num_leaves']} lr={p['learning_rate']}")
    else:
        print(f"  trial {i+1:2d}: R²_val={r2:.4f}")

# ── HPO XGBoost ───────────────────────────────────────────────────────────────
print("\nHPO XGBoost (15 trial)")
best_xgb_p, best_xgb_r2 = None, -9
for i in range(15):
    p = dict(
        n_estimators     = random.choice([1000,1200,1500]),
        learning_rate    = random.choice([0.02,0.03,0.04,0.05]),
        max_depth        = random.choice([5,6,7,8]),
        subsample        = random.uniform(0.7,0.95),
        colsample_bytree = random.uniform(0.65,0.9),
        reg_alpha        = random.choice([0.0,0.05,0.1]),
        reg_lambda       = random.choice([1.0,2.0,3.0]),
        min_child_weight = random.choice([1,3,5]),
    )
    m = XGBRegressor(**p, random_state=42, verbosity=0)
    m.fit(X_tr, y_tr, sample_weight=w_train,
          eval_set=[(X_va, y_va)], verbose=False)
    r2 = r2_score(y_va, m.predict(X_va))
    if r2 > best_xgb_r2:
        best_xgb_r2, best_xgb_p = r2, p
        print(f"  trial {i+1:2d}: R²_val={r2:.4f} *** best  depth={p['max_depth']} lr={p['learning_rate']}")
    else:
        print(f"  trial {i+1:2d}: R²_val={r2:.4f}")

# ── HPO CatBoost ──────────────────────────────────────────────────────────────
print("\nHPO CatBoost (15 trial)")
best_cat_p, best_cat_r2 = None, -9
for i in range(15):
    p = dict(
        iterations   = random.choice([1000,1200,1500]),
        learning_rate= random.choice([0.02,0.03,0.04,0.05]),
        depth        = random.choice([5,6,7,8]),
        l2_leaf_reg  = random.choice([1.0,2.0,3.0,5.0]),
        subsample    = random.uniform(0.7,0.95),
    )
    m = CatBoostRegressor(**p, random_seed=42, verbose=0)
    m.fit(X_tr, y_tr, sample_weight=w_train, eval_set=(X_va, y_va))
    r2 = r2_score(y_va, m.predict(X_va))
    if r2 > best_cat_r2:
        best_cat_r2, best_cat_p = r2, p
        print(f"  trial {i+1:2d}: R²_val={r2:.4f} *** best  depth={p['depth']} lr={p['learning_rate']}")
    else:
        print(f"  trial {i+1:2d}: R²_val={r2:.4f}")

# ── Latih final dengan best params ───────────────────────────────────────────
print("\nLatih model final")
lgb = LGBMRegressor(**best_lgb_p, random_state=42, verbose=-1)
lgb.fit(X_tr, y_tr, sample_weight=w_train, eval_set=[(X_va, y_va)])

xgb = XGBRegressor(**best_xgb_p, random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr, sample_weight=w_train,
        eval_set=[(X_va, y_va)], verbose=False)

cat = CatBoostRegressor(**best_cat_p, random_seed=42, verbose=0)
cat.fit(X_tr, y_tr, sample_weight=w_train, eval_set=(X_va, y_va))

r2_lgb = r2_score(y_te, lgb.predict(X_te))
r2_xgb = r2_score(y_te, xgb.predict(X_te))
r2_cat = r2_score(y_te, cat.predict(X_te))
pred_ens = (lgb.predict(X_te) + xgb.predict(X_te) + cat.predict(X_te)) / 3
r2_ens   = r2_score(y_te, pred_ens)
mae_ens  = mean_absolute_error(y_te, pred_ens)
rmse_ens = np.sqrt(mean_squared_error(y_te, pred_ens))

print(f"  LightGBM : R²={r2_lgb:.4f}")
print(f"  XGBoost  : R²={r2_xgb:.4f}")
print(f"  CatBoost : R²={r2_cat:.4f}")
print(f"  Ensemble : R²={r2_ens:.4f}, MAE={mae_ens:.2f}, RMSE={rmse_ens:.2f}")

# ── Simpan ────────────────────────────────────────────────────────────────────
save = {
    "models"    : {"lightgbm": lgb, "xgboost": xgb, "catboost": cat},
    "features"  : features_ok,
    "target"    : TARGET,
    "r2_test"   : r2_ens,
    "mae_test"  : mae_ens,
    "rmse_test" : rmse_ens,
    "r2_per_model": {"lightgbm": r2_lgb, "xgboost": r2_xgb, "catboost": r2_cat},
    "best_params": {"lightgbm": best_lgb_p, "xgboost": best_xgb_p, "catboost": best_cat_p},
}
with open(OUT, "wb") as f:
    pickle.dump(save, f)

PROD_R2 = 0.8606
print(f"\n{'=' * 65}")
print("RINGKASAN AKHIR")
print(f"{'=' * 65}")
print(f"  Production v2  : R²=0.8606, MAE=77.47  (43 fitur, HPO shared)")
print(f"  Model baru v3b : R²={r2_ens:.4f}, MAE={mae_ens:.2f}  ({len(features_ok)} fitur, HPO per-model)")
print(f"  Delta R²       : {r2_ens - PROD_R2:+.4f}")
print(f"  Disimpan ke    : {OUT}")
