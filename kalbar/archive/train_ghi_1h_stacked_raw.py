"""
Model produksi prediksi GHI MENTAH (W/m2) 1 jam ke depan (direct, 1-langkah) — Kalbar.
Target = ghi_target_60m_raw (lebih sesuai kebutuhan praktis: W/m2, bukan kt).
Metodologi: LightGBM direct + LightGBM residual + 3-seed bagging + Ridge stacking.
"""
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PARQUET = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_direct.parquet"
MODEL_OUT = r"C:\Users\ariff\DuckDB_kalbar\model_ghi_1h_direct_stacked_raw.pkl"

df = pd.read_parquet(PARQUET)
df = df[df["anchor_valid"]].copy()
df["year"] = df["timestamp_wib"].dt.year

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
]
TARGET = "ghi_target_60m"

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)
print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}")

train["resid"] = train[TARGET] - train["ghi_final"]
val["resid"] = val[TARGET] - val["ghi_final"]

SEEDS = [42, 7, 123]
LGB_PARAMS = dict(
    n_estimators=2000, num_leaves=127, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, verbosity=-1,
)

def fit_bagged(X_tr, y_tr, X_val, y_val, X_test):
    preds_val, preds_test = [], []
    models = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds_val.append(m.predict(X_val))
        preds_test.append(m.predict(X_test))
        models.append(m)
    return np.mean(preds_val, axis=0), np.mean(preds_test, axis=0), models

print("Training direct model (3-seed bagging)...")
pred_direct_val, pred_direct_test, models_direct = fit_bagged(
    train[FEATURES], train[TARGET], val[FEATURES], val[TARGET], test[FEATURES]
)

print("Training residual model (3-seed bagging)...")
pred_resid_val, pred_resid_test, models_resid = fit_bagged(
    train[FEATURES], train["resid"], val[FEATURES], val["resid"], test[FEATURES]
)
pred_resid_val_ghi = pred_resid_val + val["ghi_final"].values
pred_resid_test_ghi = pred_resid_test + test["ghi_final"].values

meta_X_val = np.column_stack([pred_direct_val, pred_resid_val_ghi])
meta_X_test = np.column_stack([pred_direct_test, pred_resid_test_ghi])
meta = Ridge(alpha=1.0)
meta.fit(meta_X_val, val[TARGET])
pred_stacked_test = meta.predict(meta_X_test)
pred_stacked_test = np.clip(pred_stacked_test, 0, 1400)

def eval_r2(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label:30s} R2={r2:.4f}  MAE={mae:7.2f}  RMSE={rmse:7.2f}")
    return r2, mae, rmse

print()
r2_persist = eval_r2(test[TARGET], test["ghi_final"], "Persistence baseline")
r2_direct = eval_r2(test[TARGET], pred_direct_test, "Direct LGBM (bagged)")
r2_resid = eval_r2(test[TARGET], pred_resid_test_ghi, "Residual LGBM (bagged)")
r2_stack = eval_r2(test[TARGET], pred_stacked_test, "Stacked (direct+residual)")

skill = (r2_stack[0] - r2_persist[0]) / (1 - r2_persist[0])
print(f"\nSkill score vs persistence (stacked) = {skill:.4f}")
print(f"Ridge stacking weights: direct={meta.coef_[0]:.3f} residual={meta.coef_[1]:.3f} intercept={meta.intercept_:.3f}")

test_eval = test.copy()
test_eval["pred"] = pred_stacked_test
test_eval["regime"] = pd.cut(test_eval["CLOT_mean"].fillna(-1), bins=[-2, -0.01, 5, 15, 1e9],
                              labels=["no_data", "clear(COT<5)", "partly_cloudy(5-15)", "overcast(COT>15)"])
print("\nR2 per regime awan (stacked model, test 2025):")
for reg, g in test_eval.groupby("regime", observed=True):
    if len(g) > 30:
        print(f"  {reg:22s} n={len(g):6d}  R2={r2_score(g[TARGET], g['pred']):.4f}  MAE={mean_absolute_error(g[TARGET], g['pred']):.2f}")

with open(MODEL_OUT, "wb") as f:
    pickle.dump({
        "models_direct": models_direct, "models_resid": models_resid, "meta": meta,
        "features": FEATURES, "target": TARGET,
        "test_r2_stacked": r2_stack[0], "test_mae_stacked": r2_stack[1],
    }, f)
print(f"\nModel disimpan: {MODEL_OUT}")
