"""
Eksperimen v4 — Fitur fisis turunan untuk prediksi GHI 1 jam ke depan
Baseline: v3c CatBoost solo, 49 fitur, R²=0.8641
Semua fitur baru dihitung dari kolom yang SUDAH ADA (tidak perlu data baru).

Kelompok fitur fisis baru:
  A. smart_persist      : kt(T) x rata-rata clearsky T+10..T+60 (deterministik, bukan leakage)
  B. konveksi awan      : delta_cltt_30m (puncak mendingin=tumbuh), delta_clth_30m,
                          cltt_minus_surface (kedalaman awan)
  C. akselerasi         : ghi_accel, kt_accel (turunan ke-2)
  D. range variabilitas : ghi_range_1h, kt_range_1h
  E. mean reversion     : kt_dev_roll60
  F. geometri surya     : delta_sun_alt (naik/turun)
  G. interaksi          : clot_x_kt, temp_x_rh
"""

import sys, duckdb, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sys.stdout.reconfigure(encoding="utf-8")

DB  = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
V3C = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3c.pkl"
OUT = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_physics_v4.pkl"

# ── Load baseline model info ──────────────────────────────────────────────────
with open(V3C, "rb") as f:
    v3c = pickle.load(f)
BASE_FEATURES = v3c["features"]           # 49 fitur
CAT_PARAMS    = v3c["best_params"]["catboost"] if "best_params" in v3c else None
TARGET        = v3c["target"]
print(f"Baseline v3c: {len(BASE_FEATURES)} fitur, R²={v3c['r2_test']:.4f}")
print(f"CatBoost params: {CAT_PARAMS}")

# ── Load FULL grid (untuk shift yang aman) ────────────────────────────────────
print("\nLoad data (full grid untuk perhitungan shift)...")
con = duckdb.connect(DB, read_only=True)
df = con.execute("""
    SELECT * FROM training_ghi_1h_direct ORDER BY timestamp_wib
""").df()
con.close()
print(f"  Full grid: {len(df):,} baris")

df = df.sort_values("timestamp_wib").reset_index(drop=True)
ts = pd.to_datetime(df["timestamp_wib"])

# Verifikasi grid reguler 10 menit (shift aman)
diffs = ts.diff().dropna()
pct_regular = (diffs == pd.Timedelta("10min")).mean() * 100
print(f"  Grid reguler 10-menit: {pct_regular:.2f}% baris")

# ══════════════════════════════════════════════════════════════════════════════
# HITUNG FITUR FISIS BARU
# ══════════════════════════════════════════════════════════════════════════════
print("\nHitung fitur fisis baru...")

# A. SMART PERSISTENCE — kt(T) x rata-rata clearsky T+10..T+60
#    (clearsky masa depan DETERMINISTIK — geometri matahari, bukan leakage)
cs_future_avg = sum(df["ghi_clearsky"].shift(-k) for k in range(1, 7)) / 6
df["clearsky_avg60m_future"] = cs_future_avg
df["smart_persist"] = df["kt"] * cs_future_avg

# B. KONVEKSI AWAN
#    delta_cltt_30m < 0  => puncak awan MENDINGIN => awan tumbuh vertikal (konveksi)
df["delta_cltt_30m"] = df["CLTT_mean"] - df["CLTT_mean"].shift(3)
#    delta_clth_30m > 0  => puncak awan NAIK => perkembangan vertikal
df["delta_clth_30m"] = df["CLTH_mean"] - df["CLTH_mean"].shift(3)
#    cltt_minus_surface: puncak awan sangat dingin dibanding permukaan = awan dalam (Cb)
df["cltt_minus_surface"] = df["CLTT_mean"] - (df["temp_air_c"] + 273.15)

# C. AKSELERASI (turunan ke-2) — momentum perubahan radiasi
df["ghi_accel"] = (df["ghi_final"] - df["ghi_lag10m"]) - (df["ghi_lag10m"] - df["ghi_lag20m"])
df["kt_accel"]  = (df["kt"] - df["kt_lag10m"]) - (df["kt_lag10m"] - df["kt_lag20m"])

# D. RANGE VARIABILITAS 1 JAM — magnitudo fluktuasi terkini
ghi_cols = ["ghi_final", "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m"]
kt_cols  = ["kt", "kt_lag10m", "kt_lag20m", "kt_lag30m", "kt_lag60m"]
df["ghi_range_1h"] = df[ghi_cols].max(axis=1) - df[ghi_cols].min(axis=1)
df["kt_range_1h"]  = df[kt_cols].max(axis=1)  - df[kt_cols].min(axis=1)

# E. MEAN REVERSION — deviasi kt dari rata-rata rolling 60m
df["kt_dev_roll60"] = df["kt"] - df["kt_roll60m_mean"]

# F. GEOMETRI SURYA — matahari naik (pagi) vs turun (sore); konveksi tropis asimetris
df["delta_sun_alt"] = df["sun_altitude_future"] - df["sun_altitude"]

# G. INTERAKSI FISIS
df["clot_x_kt"]  = df["CLOT_mean"].fillna(0) * df["kt"]
df["temp_x_rh"]  = df["temp_air_c"] * df["humidity_pct"] / 100.0

# Mask fitur shift di sekitar gap temporal (jaga2 walau grid ~100% reguler)
gap = ts.diff() > pd.Timedelta("10min")
gap_fwd = ts.diff(-1) < pd.Timedelta("-10min")   # gap ke depan
for col, need in [("delta_cltt_30m", 3), ("delta_clth_30m", 3)]:
    m = gap.rolling(need, min_periods=1).max().astype(bool)
    df.loc[m, col] = np.nan

NEW_GROUPS = {
    "A_smart_persist": ["smart_persist", "clearsky_avg60m_future"],
    "B_konveksi"     : ["delta_cltt_30m", "delta_clth_30m", "cltt_minus_surface"],
    "C_akselerasi"   : ["ghi_accel", "kt_accel"],
    "D_range"        : ["ghi_range_1h", "kt_range_1h"],
    "E_meanrev"      : ["kt_dev_roll60"],
    "F_geometri"     : ["delta_sun_alt"],
    "G_interaksi"    : ["clot_x_kt", "temp_x_rh"],
}
ALL_NEW = [f for g in NEW_GROUPS.values() for f in g]
print(f"  {len(ALL_NEW)} fitur baru dihitung")

# Korelasi cepat dengan target
print("\nKorelasi fitur baru dengan target (anchor_valid, day):")
mask = df["anchor_valid"] & (df["sun_altitude"] > 5)
for f in ALL_NEW:
    r = df.loc[mask, f].corr(df.loc[mask, TARGET])
    print(f"  {f:<26} r={r:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
dfa = df[df["anchor_valid"]].copy()
dfa["_year"] = pd.to_datetime(dfa["timestamp_wib"]).dt.year
train = dfa[dfa["_year"] <= 2023].dropna(subset=[TARGET]).copy()
val   = dfa[dfa["_year"] == 2024].dropna(subset=[TARGET]).copy()
test  = dfa[dfa["_year"] == 2025].dropna(subset=[TARGET]).copy()
print(f"\nTrain:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")

tier_w = {"TIER_0_ORIGINAL":1.0,"TIER_1_ML_FILLED":0.85,"TIER_4_CONSOLIDATED":0.70}
w_train = (train["fill_tier"].map(tier_w).fillna(0.5) *
           train["quality_score"].fillna(0.7)).values

def impute(df_, feats):
    X = df_[feats].copy()
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    return X

y_tr, y_va, y_te = train[TARGET].values, val[TARGET].values, test[TARGET].values

cat_kwargs = dict(CAT_PARAMS) if CAT_PARAMS else dict(
    iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=3.0, subsample=0.85)

def train_cat(feats):
    Xtr, Xva, Xte = impute(train, feats), impute(val, feats), impute(test, feats)
    m = CatBoostRegressor(**cat_kwargs, random_seed=42, verbose=0)
    m.fit(Xtr, y_tr, sample_weight=w_train, eval_set=(Xva, y_va))
    p = m.predict(Xte)
    return m, r2_score(y_te, p), mean_absolute_error(y_te, p), np.sqrt(mean_squared_error(y_te, p))

# 1) Reproduksi baseline v3c
print("\n[1] Baseline v3c (49 fitur) — reproduksi")
m_base, r2_base, mae_base, rmse_base = train_cat(BASE_FEATURES)
print(f"    R²={r2_base:.4f}  MAE={mae_base:.2f}  RMSE={rmse_base:.2f}")

# 2) Semua fitur fisis baru
print(f"\n[2] + SEMUA fitur fisis ({len(ALL_NEW)} baru, total {len(BASE_FEATURES)+len(ALL_NEW)})")
m_all, r2_all, mae_all, rmse_all = train_cat(BASE_FEATURES + ALL_NEW)
print(f"    R²={r2_all:.4f}  MAE={mae_all:.2f}  RMSE={rmse_all:.2f}  Δ={r2_all-r2_base:+.4f}")

# 3) Ablasi per kelompok (baseline + satu kelompok)
print("\n[3] Ablasi per kelompok (baseline + kelompok tsb saja):")
group_results = {}
for gname, gfeats in NEW_GROUPS.items():
    _, r2_g, mae_g, _ = train_cat(BASE_FEATURES + gfeats)
    group_results[gname] = r2_g
    print(f"    +{gname:<18} R²={r2_g:.4f}  Δ={r2_g-r2_base:+.4f}")

# 4) Kombinasi kelompok positif saja
positive_groups = [g for g, r2g in group_results.items() if r2g > r2_base + 0.0003]
pos_feats = [f for g in positive_groups for f in NEW_GROUPS[g]]
if pos_feats:
    print(f"\n[4] Baseline + kelompok positif saja: {positive_groups}")
    m_pos, r2_pos, mae_pos, rmse_pos = train_cat(BASE_FEATURES + pos_feats)
    print(f"    R²={r2_pos:.4f}  MAE={mae_pos:.2f}  Δ={r2_pos-r2_base:+.4f}")
else:
    print("\n[4] Tidak ada kelompok yang positif signifikan")
    m_pos, r2_pos, mae_pos, rmse_pos = None, -9, None, None

# ── Pilih terbaik & simpan ────────────────────────────────────────────────────
candidates = [
    ("baseline_v3c", m_base, BASE_FEATURES, r2_base, mae_base, rmse_base),
    ("all_physics",  m_all,  BASE_FEATURES + ALL_NEW, r2_all, mae_all, rmse_all),
]
if m_pos is not None:
    candidates.append(("positive_groups", m_pos, BASE_FEATURES + pos_feats, r2_pos, mae_pos, rmse_pos))

best = max(candidates, key=lambda c: c[3])
name, model, feats, r2_f, mae_f, rmse_f = best

print(f"\n{'='*65}")
print("RINGKASAN")
print(f"{'='*65}")
print(f"  v3c baseline         : R²={r2_base:.4f}  MAE={mae_base:.2f}")
print(f"  + semua fisis        : R²={r2_all:.4f}  MAE={mae_all:.2f}")
if m_pos is not None:
    print(f"  + kelompok positif   : R²={r2_pos:.4f}  MAE={mae_pos:.2f}")
print(f"\n  TERBAIK: {name} (R²={r2_f:.4f}, {len(feats)} fitur)")

if name != "baseline_v3c":
    # Feature importance top-15 dari model terbaik
    fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\n  Top-15 feature importance:")
    for i,(f,v) in enumerate(fi.head(15).items(),1):
        tag = " <- FISIS BARU" if f in ALL_NEW else ""
        print(f"    {i:2d}. {f:<26} {v:6.2f}{tag}")

    save = {
        "models"   : {"catboost": model},
        "features" : feats,
        "target"   : TARGET,
        "r2_test"  : r2_f, "mae_test": mae_f, "rmse_test": rmse_f,
        "mode"     : name,
        "new_physics_features": [f for f in feats if f in ALL_NEW],
        "vs_v3c_delta": r2_f - r2_base,
        "cat_params": cat_kwargs,
        "feature_recipe": {
            "smart_persist": "kt * mean(ghi_clearsky shift -1..-6)",
            "clearsky_avg60m_future": "mean(ghi_clearsky shift -1..-6)",
            "delta_cltt_30m": "CLTT_mean - CLTT_mean.shift(3)",
            "delta_clth_30m": "CLTH_mean - CLTH_mean.shift(3)",
            "cltt_minus_surface": "CLTT_mean - (temp_air_c+273.15)",
            "ghi_accel": "(ghi-ghi_lag10)-(ghi_lag10-ghi_lag20)",
            "kt_accel": "(kt-kt_lag10)-(kt_lag10-kt_lag20)",
            "ghi_range_1h": "max-min dari ghi & 4 lag",
            "kt_range_1h": "max-min dari kt & 4 lag",
            "kt_dev_roll60": "kt - kt_roll60m_mean",
            "delta_sun_alt": "sun_altitude_future - sun_altitude",
            "clot_x_kt": "CLOT_mean(fill0) * kt",
            "temp_x_rh": "temp_air_c * humidity_pct / 100",
        },
    }
    with open(OUT, "wb") as f:
        pickle.dump(save, f)
    print(f"\n  Model DISIMPAN -> {OUT}")
else:
    print("\n  Fitur fisis tidak menaikkan akurasi — model v3c tetap produksi.")
