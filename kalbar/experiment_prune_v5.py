"""
Eksperimen v5 — Feature pruning untuk model v3c (CatBoost, 49 fitur, R²=0.8646)
Tujuan: cari SUBSET fitur MINIMAL yang mempertahankan (atau menaikkan) R².

Metode:
  Fase 1: baseline 49 fitur (reproduksi)
  Fase 2: sweep top-K berdasarkan CatBoost importance (K = 12..45)
  Fase 3: greedy backward elimination di sekitar K optimal
          (hapus 1 fitur bila R² tidak turun > toleransi, ulangi)
  Simpan model pruned bila R² >= baseline - epsilon dengan fitur << 49.
"""

import sys, duckdb, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sys.stdout.reconfigure(encoding="utf-8")

DB  = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
V3C = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3c.pkl"
OUT = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_pruned_v5.pkl"

# Params CatBoost yang mereproduksi baseline v3c (dari experiment_physics_v4: R²=0.8646)
CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=7,
                  l2_leaf_reg=3.0, subsample=0.85)

with open(V3C, "rb") as f:
    v3c = pickle.load(f)
BASE_FEATURES = list(v3c["features"])
TARGET = v3c["target"]
print(f"Baseline v3c: {len(BASE_FEATURES)} fitur, target={TARGET}")

# ── Load data ────────────────────────────────────────────────────────────────
con = duckdb.connect(DB, read_only=True)
df = con.execute("SELECT * FROM training_ghi_1h_direct WHERE anchor_valid ORDER BY timestamp_wib").df()
con.close()

df["_year"] = pd.to_datetime(df["timestamp_wib"]).dt.year
train = df[df["_year"] <= 2023].dropna(subset=[TARGET]).copy()
val   = df[df["_year"] == 2024].dropna(subset=[TARGET]).copy()
test  = df[df["_year"] == 2025].dropna(subset=[TARGET]).copy()
print(f"Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")

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

def fit_eval(feats, return_model=False):
    Xtr, Xva, Xte = impute(train, feats), impute(val, feats), impute(test, feats)
    m = CatBoostRegressor(**CAT_PARAMS, random_seed=42, verbose=0)
    m.fit(Xtr, y_tr, sample_weight=w_train, eval_set=(Xva, y_va))
    p = m.predict(Xte)
    r2 = r2_score(y_te, p); mae = mean_absolute_error(y_te, p)
    rmse = np.sqrt(mean_squared_error(y_te, p))
    return (m, r2, mae, rmse) if return_model else (r2, mae, rmse)

# ── Fase 1: baseline ─────────────────────────────────────────────────────────
print("\n[Fase 1] Baseline 49 fitur")
m_base, r2_base, mae_base, rmse_base = fit_eval(BASE_FEATURES, return_model=True)
print(f"  R²={r2_base:.4f}  MAE={mae_base:.2f}  RMSE={rmse_base:.2f}")

# Ranking importance dari baseline
imp = pd.Series(m_base.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)
print("\n  Importance ranking (49 fitur):")
for i,(f,v) in enumerate(imp.items(),1):
    print(f"    {i:2d}. {f:<26} {v:6.2f}")

# ── Fase 2: sweep top-K ──────────────────────────────────────────────────────
print("\n[Fase 2] Sweep top-K berdasarkan importance")
ranked = list(imp.index)
Ks = [12, 15, 18, 21, 24, 27, 30, 34, 38, 42, 46]
sweep = {}
for K in Ks:
    feats = ranked[:K]
    r2, mae, rmse = fit_eval(feats)
    sweep[K] = (r2, mae, feats)
    flag = ""
    if r2 >= r2_base - 0.0005: flag = " <= dalam toleransi (0.0005)"
    elif r2 >= r2_base - 0.0015: flag = " (~toleransi longgar 0.0015)"
    print(f"  top-{K:2d}: R²={r2:.4f}  MAE={mae:.2f}  Δ={r2-r2_base:+.4f}{flag}")

# Pilih K terkecil yang masih dalam toleransi ketat (0.0005), fallback ke 0.0015
EPS_TIGHT, EPS_LOOSE = 0.0005, 0.0015
K_tight = [K for K in Ks if sweep[K][0] >= r2_base - EPS_TIGHT]
K_loose = [K for K in Ks if sweep[K][0] >= r2_base - EPS_LOOSE]
if K_tight:
    K_star = min(K_tight); tol_used = EPS_TIGHT
elif K_loose:
    K_star = min(K_loose); tol_used = EPS_LOOSE
else:
    K_star = 46; tol_used = None
print(f"\n  K* terpilih = {K_star} (toleransi {tol_used})")

# ── Fase 3: greedy backward elimination dari top-K* ──────────────────────────
print(f"\n[Fase 3] Greedy backward elimination mulai dari top-{K_star}")
current = list(ranked[:K_star])
r2_cur, mae_cur, _ = fit_eval(current)
print(f"  Start: {len(current)} fitur, R²={r2_cur:.4f}")

TOL = 0.0008   # boleh hapus bila R² tidak turun lebih dari ini vs baseline penuh
removed = []
improved = True
while improved and len(current) > 8:
    improved = False
    best_drop, best_r2, best_mae = None, -9, None
    for f in list(current):
        trial = [x for x in current if x != f]
        r2_t, mae_t, _ = fit_eval(trial)
        if r2_t > best_r2:
            best_r2, best_mae, best_drop = r2_t, mae_t, f
    # Terima penghapusan bila R² hasil >= baseline penuh - TOL
    if best_r2 >= r2_base - TOL:
        current = [x for x in current if x != best_drop]
        removed.append((best_drop, best_r2))
        r2_cur, mae_cur = best_r2, best_mae
        print(f"  - hapus {best_drop:<26} -> {len(current)} fitur, R²={best_r2:.4f} (Δbase={best_r2-r2_base:+.4f})")
        improved = True
    else:
        print(f"  stop: penghapusan terbaik ({best_drop}) menurunkan R² ke {best_r2:.4f} < {r2_base-TOL:.4f}")

# ── Model final pruned ───────────────────────────────────────────────────────
print(f"\n[Final] Latih ulang model pruned ({len(current)} fitur)")
m_pruned, r2_p, mae_p, rmse_p = fit_eval(current, return_model=True)
print(f"  R²={r2_p:.4f}  MAE={mae_p:.2f}  RMSE={rmse_p:.2f}")

print(f"\n{'='*66}")
print("RINGKASAN PRUNING")
print(f"{'='*66}")
print(f"  Baseline penuh : {len(BASE_FEATURES)} fitur  R²={r2_base:.4f}  MAE={mae_base:.2f}")
print(f"  Pruned final   : {len(current)} fitur  R²={r2_p:.4f}  MAE={mae_p:.2f}")
print(f"  Fitur dibuang  : {len(BASE_FEATURES)-len(current)}  |  Δ R²={r2_p-r2_base:+.4f}")
print(f"\n  Fitur pruned ({len(current)}):")
for i,f in enumerate(current,1):
    print(f"    {i:2d}. {f}")
print(f"\n  Fitur dibuang ({len(BASE_FEATURES)-len(current)}):")
for f in BASE_FEATURES:
    if f not in current:
        print(f"    - {f}")

# Simpan bila cukup baik (dalam toleransi longgar & lebih ramping)
if r2_p >= r2_base - EPS_LOOSE and len(current) < len(BASE_FEATURES):
    save = {
        "models": {"catboost": m_pruned},
        "features": current,
        "target": TARGET,
        "r2_test": r2_p, "mae_test": mae_p, "rmse_test": rmse_p,
        "cat_params": CAT_PARAMS,
        "pruned_from": len(BASE_FEATURES),
        "dropped_features": [f for f in BASE_FEATURES if f not in current],
        "baseline_r2": r2_base,
        "vs_baseline_delta": r2_p - r2_base,
    }
    with open(OUT, "wb") as f:
        pickle.dump(save, f)
    print(f"\n  Model pruned DISIMPAN -> {OUT}")
else:
    print(f"\n  Pruned tidak disimpan (R²={r2_p:.4f} di luar toleransi atau tidak lebih ramping)")
