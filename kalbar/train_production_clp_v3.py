"""
Production Training v3 — Model Hybrid avg60m dengan fitur CLP lengkap
Perubahan dari model sebelumnya (R²=0.8606):
  RESTORE: CLER_23_std, pressure_hpa, cloud_cover_oktas (hilang dari tabel)
  TAMBAH : CLER_23_coverage, CLOT_median, clot_std_roll30m (baru, terbukti +R²)
  SKIP   : cloud_class_code, clot_trend_30m (ablasi negatif/marginal)
Target  : ghi_target_avg60m
Metode  : ensemble LightGBM+XGBoost+CatBoost, sample-weighted, random search HPO
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
OUT = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3.pkl"

print("=" * 65)
print("STEP 1: Restore & tambah kolom yang hilang")
print("=" * 65)

con = duckdb.connect(DB)

existing = set(r[0] for r in con.execute(
    "SELECT column_name FROM information_schema.columns WHERE table_name='training_ghi_1h_direct'"
).fetchall())

# --- 1a. CLER_23_std dari clp_pontianak ---
if "CLER_23_std" not in existing:
    con.execute("ALTER TABLE training_ghi_1h_direct ADD COLUMN CLER_23_std DOUBLE")
    con.execute("""
        UPDATE training_ghi_1h_direct t
        SET CLER_23_std = c.CLER_23_std
        FROM clp_pontianak c
        WHERE t.timestamp_wib = c.timestamp_wib
    """)
    print("  + CLER_23_std ditambahkan dari clp_pontianak")
else:
    print("  ~ CLER_23_std sudah ada")

# --- 1b. pressure_hpa & cloud_cover_oktas dari synop_radiasi_jam (hourly -> 10-min LOCF) ---
for col in ["pressure_hpa", "cloud_cover_oktas"]:
    if col not in existing:
        con.execute(f"ALTER TABLE training_ghi_1h_direct ADD COLUMN {col} DOUBLE")
        print(f"  + kolom {col} ditambahkan (akan diisi dari SYNOP)")

# Cek apakah perlu diisi
n_pressure = con.execute(
    "SELECT COUNT(*) FROM training_ghi_1h_direct WHERE pressure_hpa IS NOT NULL"
).fetchone()[0]

if n_pressure == 0:
    print("  Mengisi pressure_hpa dan cloud_cover_oktas dari synop_radiasi_jam (LOCF hourly->10mnt)...")

    # Load SYNOP jam-jaman
    df_synop = con.execute("""
        SELECT
            timestamp_wib,
            pressure_qff_mb AS pressure_hpa,
            cloud_cover_oktas
        FROM synop_radiasi_jam
        WHERE timestamp_wib IS NOT NULL
        ORDER BY timestamp_wib
    """).df()
    df_synop["timestamp_wib"] = pd.to_datetime(df_synop["timestamp_wib"])

    # Load timestamp training
    df_ts = con.execute(
        "SELECT timestamp_wib FROM training_ghi_1h_direct ORDER BY timestamp_wib"
    ).df()
    df_ts["timestamp_wib"] = pd.to_datetime(df_ts["timestamp_wib"])

    # Merge-asof: join SYNOP ke setiap 10-min (last obs carried forward)
    df_merged = pd.merge_asof(
        df_ts.sort_values("timestamp_wib"),
        df_synop.sort_values("timestamp_wib"),
        on="timestamp_wib",
        direction="backward",
        tolerance=pd.Timedelta("61min")
    )

    # Update via temp table
    con.execute("CREATE OR REPLACE TEMP TABLE tmp_synop AS SELECT * FROM df_merged")
    con.execute("""
        UPDATE training_ghi_1h_direct t
        SET
            pressure_hpa      = s.pressure_hpa,
            cloud_cover_oktas = s.cloud_cover_oktas
        FROM tmp_synop s
        WHERE t.timestamp_wib = s.timestamp_wib
    """)
    n_filled = con.execute(
        "SELECT COUNT(*) FROM training_ghi_1h_direct WHERE pressure_hpa IS NOT NULL"
    ).fetchone()[0]
    print(f"  OK: pressure_hpa diisi {n_filled:,} baris")
else:
    print(f"  ~ pressure_hpa sudah terisi ({n_pressure:,} baris)")

print("\nSTEP 2: Verifikasi null rate semua fitur (anchor_valid & sun_altitude>5)")
checks = [
    "CLOT_std","CLER_23_std","CLER_23_coverage","CLOT_median",
    "clot_std_roll30m","pressure_hpa","cloud_cover_oktas"
]
for c in checks:
    pct = con.execute(f"""
        SELECT ROUND(100.0*COUNT({c})/COUNT(*),1)
        FROM training_ghi_1h_direct WHERE anchor_valid AND sun_altitude>5
    """).fetchone()[0]
    print(f"  {c:<25} {pct}%")

print("\nSTEP 3: Load data training")
df = con.execute("""
    SELECT * FROM training_ghi_1h_direct
    WHERE anchor_valid
    ORDER BY timestamp_wib
""").df()
con.close()

print(f"  Total baris: {len(df):,}")

# ── Fitur production v3 ──────────────────────────────────────────────────────
PROD_FEATURES = [
    # Radiasi & clearsky
    "ghi_final", "kt", "ghi_clearsky", "ghi_clearsky_future",
    # Geometri matahari
    "sun_altitude", "sun_altitude_future",
    # Waktu
    "hour_sin", "hour_cos", "hour_sin_future", "hour_cos_future",
    "doy_sin", "doy_cos", "month",
    # Lag GHI & kT
    "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m",
    "kt_lag10m", "kt_lag20m", "kt_lag30m", "kt_lag60m",
    # Rolling & delta
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std",
    "delta_kt_10m", "delta_kt_30m", "delta_ghi_30m",
    # CLP — fitur lama
    "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean",
    "clp_cloud_present_int",
    "clot_lag10m", "clot_lag30m", "delta_clot_30m",
    # CLP — restore (ada di production model sebelumnya)
    "CLOT_std", "CLER_23_std",
    # CLP — BARU (terbukti positif di ablasi)
    "CLER_23_coverage", "CLOT_median", "clot_std_roll30m",
    # Meteo
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm",
    # SYNOP (restore)
    "pressure_hpa", "cloud_cover_oktas",
    # Aerosol
    "AOD_500nm", "angstrom_440_870", "precipitable_water_cm",
]

features_ok = [f for f in PROD_FEATURES if f in df.columns]
features_miss = [f for f in PROD_FEATURES if f not in df.columns]
print(f"  Fitur OK: {len(features_ok)}/{len(PROD_FEATURES)}")
if features_miss:
    print(f"  Fitur hilang: {features_miss}")

TARGET = "ghi_target_avg60m"

# ── Split temporal ───────────────────────────────────────────────────────────
df["_year"] = pd.to_datetime(df["timestamp_wib"]).dt.year
train = df[df["_year"] <= 2023].dropna(subset=[TARGET]).copy()
val   = df[df["_year"] == 2024].dropna(subset=[TARGET]).copy()
test  = df[df["_year"] == 2025].dropna(subset=[TARGET]).copy()
print(f"  Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")

# ── Sample weights ────────────────────────────────────────────────────────────
tier_w = {"TIER_0_ORIGINAL":1.0,"TIER_1_ML_FILLED":0.85,"TIER_4_CONSOLIDATED":0.70}
w_train = (train["fill_tier"].map(tier_w).fillna(0.5) *
           train["quality_score"].fillna(0.7)).values

# ── Impute ────────────────────────────────────────────────────────────────────
def impute(df_, feats):
    X = df_[feats].copy()
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    return X

X_tr  = impute(train, features_ok)
X_val = impute(val,   features_ok)
X_te  = impute(test,  features_ok)
y_tr, y_val, y_te = train[TARGET].values, val[TARGET].values, test[TARGET].values

# ── Random Search HPO (LightGBM) ──────────────────────────────────────────────
print("\nSTEP 4: Random Search HPO — LightGBM (20 trial)")
random.seed(42)
np.random.seed(42)

def sample_lgb():
    return dict(
        n_estimators   = random.choice([800,1000,1200,1500]),
        learning_rate  = random.choice([0.02,0.03,0.04,0.05]),
        num_leaves     = random.choice([31,63,95,127]),
        min_child_samples = random.choice([20,30,50]),
        subsample      = random.uniform(0.7,0.95),
        colsample_bytree = random.uniform(0.6,0.9),
        reg_alpha      = random.choice([0.0,0.05,0.1,0.2]),
        reg_lambda     = random.choice([0.5,1.0,2.0,3.0]),
    )

best_p, best_r2 = None, -9
for i in range(20):
    p = sample_lgb()
    m = LGBMRegressor(**p, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr, sample_weight=w_train,
          eval_set=[(X_val, y_val)], callbacks=None)
    r2 = r2_score(y_val, m.predict(X_val))
    if r2 > best_r2:
        best_r2, best_p = r2, p
        print(f"  trial {i+1:2d}: R²_val={r2:.4f} *** best")
    else:
        print(f"  trial {i+1:2d}: R²_val={r2:.4f}")

print(f"\n  Best params: {best_p}")

# ── Latih 3 model dengan best params ─────────────────────────────────────────
print("\nSTEP 5: Latih ensemble 3 model (best HPO)")

lgb = LGBMRegressor(**best_p, random_state=42, verbose=-1)
lgb.fit(X_tr, y_tr, sample_weight=w_train, eval_set=[(X_val, y_val)])

xgb = XGBRegressor(
    n_estimators=best_p["n_estimators"], learning_rate=best_p["learning_rate"],
    max_depth=6, subsample=best_p["subsample"],
    colsample_bytree=best_p["colsample_bytree"],
    reg_alpha=best_p["reg_alpha"], reg_lambda=best_p["reg_lambda"],
    random_state=42, verbosity=0
)
xgb.fit(X_tr, y_tr, sample_weight=w_train,
        eval_set=[(X_val, y_val)], verbose=False)

cat = CatBoostRegressor(
    iterations=best_p["n_estimators"], learning_rate=best_p["learning_rate"],
    depth=6, l2_leaf_reg=best_p["reg_lambda"],
    subsample=best_p["subsample"], random_seed=42, verbose=0
)
cat.fit(X_tr, y_tr, sample_weight=w_train, eval_set=(X_val, y_val))

# Per-model score
for name, m in [("LightGBM",lgb),("XGBoost",xgb),("CatBoost",cat)]:
    r2 = r2_score(y_te, m.predict(X_te))
    print(f"  {name:<10}: R²={r2:.4f}")

# Ensemble
pred_ens = (lgb.predict(X_te) + xgb.predict(X_te) + cat.predict(X_te)) / 3
r2_ens  = r2_score(y_te, pred_ens)
mae_ens = mean_absolute_error(y_te, pred_ens)
rmse_ens= np.sqrt(mean_squared_error(y_te, pred_ens))
print(f"  Ensemble   : R²={r2_ens:.4f}, MAE={mae_ens:.2f}, RMSE={rmse_ens:.2f}")

# ── Feature importance top-20 ─────────────────────────────────────────────────
print("\nSTEP 6: Feature importance (LightGBM) — Top 25")
fi = pd.Series(lgb.feature_importances_, index=features_ok).sort_values(ascending=False)
new_feats = ["CLER_23_std","CLER_23_coverage","CLOT_median","clot_std_roll30m",
             "pressure_hpa","cloud_cover_oktas"]
for i,(f,v) in enumerate(fi.head(25).items()):
    tag = " <- RESTORE/BARU" if f in new_feats else ""
    print(f"  {i+1:2d}. {f:<30} {v:>6}{tag}")

# ── Simpan model ──────────────────────────────────────────────────────────────
print("\nSTEP 7: Simpan model")
PROD_R2 = 0.8606
save = {
    "models"    : {"lightgbm": lgb, "xgboost": xgb, "catboost": cat},
    "features"  : features_ok,
    "target"    : TARGET,
    "r2_test"   : r2_ens,
    "mae_test"  : mae_ens,
    "rmse_test" : rmse_ens,
    "n_features": len(features_ok),
    "best_lgb_params": best_p,
    "vs_previous_r2" : r2_ens - PROD_R2,
}
with open(OUT, "wb") as f:
    pickle.dump(save, f)

print(f"\n{'=' * 65}")
print("RINGKASAN AKHIR")
print(f"{'=' * 65}")
print(f"  Model sebelumnya (v2): R²=0.8606, MAE=77.47, RMSE=109.21  (43 fitur)")
print(f"  Model baru (v3)      : R²={r2_ens:.4f}, MAE={mae_ens:.2f}, RMSE={rmse_ens:.2f}  ({len(features_ok)} fitur)")
print(f"  Delta R²             : {r2_ens - PROD_R2:+.4f}")
print(f"  Disimpan ke          : {OUT}")
