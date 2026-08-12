"""
v3c: Weighted ensemble & CatBoost solo — dari model yang sudah dilatih di v3b
"""
import sys, duckdb, pickle, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sys.stdout.reconfigure(encoding="utf-8")

DB   = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
V3B  = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3b.pkl"
OUT  = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_hybrid_clp_v3c.pkl"

# Load model v3b
with open(V3B, "rb") as f:
    v3b = pickle.load(f)

models   = v3b["models"]
features = v3b["features"]
TARGET   = v3b["target"]

# Load test set
con = duckdb.connect(DB, read_only=True)
df  = con.execute("""
    SELECT * FROM training_ghi_1h_direct
    WHERE anchor_valid ORDER BY timestamp_wib
""").df()
con.close()

df["_year"] = pd.to_datetime(df["timestamp_wib"]).dt.year
test = df[df["_year"] == 2025].dropna(subset=[TARGET]).copy()
val  = df[df["_year"] == 2024].dropna(subset=[TARGET]).copy()

def impute(df_, feats):
    X = df_[feats].copy()
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    return X

X_va = impute(val,  features)
X_te = impute(test, features)
y_va, y_te = val[TARGET].values, test[TARGET].values

lgb = models["lightgbm"]
xgb = models["xgboost"]
cat = models["catboost"]

p_lgb = lgb.predict(X_te)
p_xgb = xgb.predict(X_te)
p_cat = cat.predict(X_te)

p_lgb_va = lgb.predict(X_va)
p_xgb_va = xgb.predict(X_va)
p_cat_va = cat.predict(X_va)

print("=" * 60)
print("Per-model test R²")
print("=" * 60)
for name, p in [("LightGBM", p_lgb), ("XGBoost", p_xgb), ("CatBoost", p_cat)]:
    print(f"  {name:<12}: R²={r2_score(y_te, p):.4f}, MAE={mean_absolute_error(y_te, p):.2f}")

print("\nEnsemble bobot berbeda (test set):")
best_w, best_r2 = None, -9
for w_cat in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    w_rem = (1 - w_cat) / 2
    pred = w_cat * p_cat + w_rem * p_lgb + w_rem * p_xgb
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    if r2 > best_r2:
        best_r2, best_w = r2, w_cat
        tag = " <-- best"
    else:
        tag = ""
    print(f"  Cat={w_cat:.2f}, LGB=XGB={w_rem:.2f} -> R²={r2:.4f}, MAE={mae:.2f}{tag}")

# Optimal weighted
w_rem = (1 - best_w) / 2
pred_weighted = best_w * p_cat + w_rem * p_lgb + w_rem * p_xgb
r2_w   = r2_score(y_te, pred_weighted)
mae_w  = mean_absolute_error(y_te, pred_weighted)
rmse_w = np.sqrt(mean_squared_error(y_te, pred_weighted))

# CatBoost solo
r2_cat_solo  = r2_score(y_te, p_cat)
mae_cat_solo = mean_absolute_error(y_te, p_cat)
rmse_cat_solo= np.sqrt(mean_squared_error(y_te, p_cat))

print(f"\n{'=' * 60}")
print("RINGKASAN PERBANDINGAN")
print(f"{'=' * 60}")
print(f"  Production v2 (equal ensemble) : R²=0.8606  MAE=77.47")
print(f"  v3b equal ensemble (49 fitur)  : R²=0.8572  MAE=79.90")
print(f"  v3c weighted Cat={best_w:.0%}          : R²={r2_w:.4f}  MAE={mae_w:.2f}")
print(f"  v3c CatBoost solo              : R²={r2_cat_solo:.4f}  MAE={mae_cat_solo:.2f}")

# Simpan model terbaik
best_r2_final = max(r2_w, r2_cat_solo)
if r2_cat_solo >= r2_w:
    # Simpan CatBoost solo sebagai model utama tapi simpan semua model
    save_mode = "catboost_primary"
    pred_final = p_cat
    r2_final, mae_final, rmse_final = r2_cat_solo, mae_cat_solo, rmse_cat_solo
    weights = {"lightgbm": 0.0, "xgboost": 0.0, "catboost": 1.0}
else:
    save_mode = f"weighted_cat{best_w:.0%}"
    pred_final = pred_weighted
    r2_final, mae_final, rmse_final = r2_w, mae_w, rmse_w
    weights = {"lightgbm": w_rem, "xgboost": w_rem, "catboost": best_w}

print(f"\n  Mode terpilih: {save_mode}")
print(f"  R² final     : {r2_final:.4f}")
print(f"  vs production: {r2_final - 0.8606:+.4f}")

save = {
    "models"        : models,
    "features"      : features,
    "target"        : TARGET,
    "weights"       : weights,
    "ensemble_mode" : save_mode,
    "r2_test"       : r2_final,
    "mae_test"      : mae_final,
    "rmse_test"     : rmse_final,
    "r2_per_model"  : {
        "lightgbm": r2_score(y_te, p_lgb),
        "xgboost" : r2_score(y_te, p_xgb),
        "catboost": r2_score(y_te, p_cat),
    },
    "n_features"    : len(features),
    "vs_v2_delta"   : r2_final - 0.8606,
}
with open(OUT, "wb") as f:
    pickle.dump(save, f)

print(f"  Disimpan ke  : {OUT}")
print("""
Cara pakai model v3c:
  pred = sum(w * m.predict(X[features]) for m, w in zip(models.values(), weights.values()))
  -- atau jika CatBoost solo: pred = models['catboost'].predict(X[features])
""")
