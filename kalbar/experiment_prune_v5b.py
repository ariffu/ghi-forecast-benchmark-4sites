"""
Eksperimen v5b — Feature pruning DENGAN SELEKSI VALIDASI (metodologi benar)
Perbaikan dari v5: SEMUA keputusan pruning dipandu VAL 2024, bukan test 2025.
Test 2025 hanya dievaluasi SEKALI di akhir untuk kandidat model terpilih.
Ini menghilangkan bias seleksi yang membuat v5 (0.8691 @ 8 fitur) terlalu optimistis.
"""

import sys, duckdb, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sys.stdout.reconfigure(encoding="utf-8")

DB  = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
V3C = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3c.pkl"
OUT = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_pruned_v5b.pkl"

CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=7,
                  l2_leaf_reg=3.0, subsample=0.85)

with open(V3C, "rb") as f:
    v3c = pickle.load(f)
BASE_FEATURES = list(v3c["features"])
TARGET = v3c["target"]
print(f"Baseline v3c: {len(BASE_FEATURES)} fitur, target={TARGET}")

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

def fit(feats):
    m = CatBoostRegressor(**CAT_PARAMS, random_seed=42, verbose=0)
    m.fit(impute(train, feats), y_tr, sample_weight=w_train,
          eval_set=(impute(val, feats), y_va))
    return m

def r2_val(m, feats):  return r2_score(y_va, m.predict(impute(val,  feats)))
def eval_test(m, feats):
    p = m.predict(impute(test, feats))
    return r2_score(y_te, p), mean_absolute_error(y_te, p), np.sqrt(mean_squared_error(y_te, p))

# ── Baseline (val + test) ────────────────────────────────────────────────────
print("\n[Baseline] 49 fitur")
m_base = fit(BASE_FEATURES)
r2v_base = r2_val(m_base, BASE_FEATURES)
r2t_base, maet_base, rmset_base = eval_test(m_base, BASE_FEATURES)
print(f"  VAL  R²={r2v_base:.4f}")
print(f"  TEST R²={r2t_base:.4f}  MAE={maet_base:.2f}  RMSE={rmset_base:.2f}")

imp = pd.Series(m_base.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)
ranked = list(imp.index)

# ── Fase 2: sweep top-K, DINILAI DI VAL ──────────────────────────────────────
print("\n[Fase 2] Sweep top-K (dinilai di VAL 2024)")
Ks = [8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 46]
sweep = {}
for K in Ks:
    feats = ranked[:K]
    m = fit(feats)
    rv = r2_val(m, feats)
    sweep[K] = rv
    print(f"  top-{K:2d}: VAL R²={rv:.4f}  Δval={rv-r2v_base:+.4f}")

# Pilih K terkecil yang VAL-nya >= baseline_val - 0.001
EPS = 0.001
cand = [K for K in Ks if sweep[K] >= r2v_base - EPS]
K_star = min(cand) if cand else max(Ks, key=lambda k: sweep[k])
print(f"\n  K* = {K_star} (VAL R²={sweep[K_star]:.4f})")

# ── Fase 3: greedy backward, DINILAI DI VAL ──────────────────────────────────
print(f"\n[Fase 3] Greedy backward elimination dari top-{K_star} (dinilai di VAL)")
current = list(ranked[:K_star])
m_cur = fit(current); rv_cur = r2_val(m_cur, current)
print(f"  Start: {len(current)} fitur, VAL R²={rv_cur:.4f}")

TOL = 0.0010
while len(current) > 5:
    best_drop, best_rv = None, -9
    for f in list(current):
        trial = [x for x in current if x != f]
        m_t = fit(trial); rv_t = r2_val(m_t, trial)
        if rv_t > best_rv:
            best_rv, best_drop = rv_t, f
    if best_rv >= r2v_base - TOL:
        current = [x for x in current if x != best_drop]
        rv_cur = best_rv
        print(f"  - hapus {best_drop:<24} -> {len(current)} fitur, VAL R²={best_rv:.4f}")
    else:
        print(f"  stop: hapus {best_drop} -> VAL R²={best_rv:.4f} < {r2v_base-TOL:.4f}")
        break

# ── Evaluasi FINAL di test (sekali) untuk beberapa kandidat ──────────────────
print(f"\n[Evaluasi TEST — sekali, untuk kandidat final]")
candidates = {
    f"pruned_greedy ({len(current)})": current,
    f"top-{K_star}": ranked[:K_star],
    "top-15": ranked[:15],
    "top-21": ranked[:21],
    "baseline (49)": BASE_FEATURES,
}
results = {}
for name, feats in candidates.items():
    m = fit(feats)
    rv = r2_val(m, feats)
    rt, mt, rmst = eval_test(m, feats)
    results[name] = (len(feats), rv, rt, mt, rmst, feats, m)
    print(f"  {name:<22} nfeat={len(feats):2d}  VAL={rv:.4f}  TEST R²={rt:.4f}  MAE={mt:.2f}")

print(f"\n{'='*66}")
print("RINGKASAN (metodologi benar — seleksi di VAL, evaluasi di TEST)")
print(f"{'='*66}")
print(f"  Baseline 49 fitur : TEST R²={r2t_base:.4f}  MAE={maet_base:.2f}")
pr = results[f"pruned_greedy ({len(current)})"]
print(f"  Pruned {pr[0]} fitur   : TEST R²={pr[2]:.4f}  MAE={pr[3]:.2f}")
print(f"  Δ TEST R² = {pr[2]-r2t_base:+.4f}")

# Simpan kandidat pruned terbaik (VAL-terbaik di antara yang ramping <= 20 fitur)
slim = {k:v for k,v in results.items() if v[0] <= 20}
best_name = max(slim, key=lambda k: slim[k][1])   # pilih by VAL
bn, bv, bt, bmae, brmse, bfeats, bmodel = slim[best_name]
print(f"\n  Kandidat ramping terbaik (by VAL): {best_name}")
print(f"    {bn} fitur, VAL R²={bv:.4f}, TEST R²={bt:.4f}, MAE={bmae:.2f}")
print(f"    Fitur: {bfeats}")

save = {
    "models": {"catboost": bmodel},
    "features": bfeats,
    "target": TARGET,
    "r2_test": bt, "mae_test": bmae, "rmse_test": brmse,
    "r2_val": bv,
    "cat_params": CAT_PARAMS,
    "selection_method": "validation-guided (VAL 2024), test evaluated once",
    "pruned_from": len(BASE_FEATURES),
    "baseline_test_r2": r2t_base,
    "vs_baseline_delta": bt - r2t_base,
    "dropped_features": [f for f in BASE_FEATURES if f not in bfeats],
}
with open(OUT, "wb") as f:
    pickle.dump(save, f)
print(f"\n  Model DISIMPAN -> {OUT}")
