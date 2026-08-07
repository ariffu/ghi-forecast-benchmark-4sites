#!/usr/bin/env python3
"""
R8 Arm B v2 -- JAMBI: GBM vs Deep Learning, dataset v2.

Data source (DIPASTIKAN, sesuai instruksi user 2026-07-25):
    jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb
    -> jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025

Ini menggantikan run v1 lama (`outputs_R8_Jambi/arm_B_summary.csv`, 18 Juli,
n=9.511). GBM (CatBoost + LightGBM) dijalankan di bagian ini. Bagian DL
(LSTM/MLP/Transformer, 3-seed) BELUM dijalankan -- torch tidak terpasang di
sandbox ini dan instalasi paket sempat dihentikan user, jadi bagian DL
menunggu keputusan/izin eksplisit sebelum dicoba lagi (lihat catatan di
akhir skrip / laporan sesi).

Run:
    python train_ghi_1h_jambi_R8_armB_v2.py
"""
import pickle
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ghi_1h_jambi_R1_benchmark_v2 import (
    connect_data, build_sql, add_features, FEATURES as F1_FEATURES,
    TARGET_POINT, TIME_COL, LOCAL_DB_PATH, SCHEMA, TABLE,
)

OUTPUT_DIR = Path("outputs_R8_jambi_v2")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
PRED_MIN, PRED_MAX = 0.0, 1400.0
DATA_CKPT_PATH = OUTPUT_DIR / "arm_B_v2_data.pkl"
CKPT_PATH = OUTPUT_DIR / "arm_B_v2_checkpoint.pkl"

assert len(F1_FEATURES) == 50


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp("2024-01-01"),
            (ts >= pd.Timestamp("2024-01-01")) & (ts < pd.Timestamp("2025-01-01")),
            ts >= pd.Timestamp("2025-01-01"))


def train_catboost(x_tr, y_tr, x_va, y_va):
    m = CatBoostRegressor(iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
                           loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
                           thread_count=-1, allow_writing_files=False)
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_va.astype(float).values, y_va.astype(float).values),
          early_stopping_rounds=150)
    return m


def train_lgbm(x_tr, y_tr, x_va, y_va):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("m", lgb.LGBMRegressor(
            objective="regression", n_estimators=6000, learning_rate=0.02,
            num_leaves=39, min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5,
            colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
            random_state=RANDOM_STATE, n_jobs=-1, force_col_wise=True, verbosity=-1)),
    ])
    pipe.fit(x_tr, y_tr,
             m__eval_set=[(pipe.named_steps["imp"].fit_transform(x_va), y_va)],
             m__eval_metric="rmse",
             m__callbacks=[lgb.early_stopping(150, verbose=False)])
    return pipe


def metrics(y_true, y_pred, y_sp):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_sp = float(np.sqrt(mean_squared_error(y_true, y_sp)))
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / rmse_sp if rmse_sp > 0 else 0.0, 4),
    }


def load_or_build_data():
    if DATA_CKPT_PATH.exists():
        print(f"Loading cached data from {DATA_CKPT_PATH}")
        with open(DATA_CKPT_PATH, "rb") as f:
            return pickle.load(f)
    print(f"DB reference: {LOCAL_DB_PATH.resolve()}  (schema={SCHEMA}, table={TABLE})")
    con = connect_data()
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)
    df_use = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    print(f"Rows: {len(df_use):,}")
    with open(DATA_CKPT_PATH, "wb") as f:
        pickle.dump(df_use, f)
    return df_use


def main():
    df_use = load_or_build_data()
    tr_m, va_m, te_m = split_masks(df_use)
    print(f"train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

    x_tr = df_use.loc[tr_m, F1_FEATURES]
    x_va = df_use.loc[va_m, F1_FEATURES]
    x_te = df_use.loc[te_m, F1_FEATURES]
    y_tr = df_use.loc[tr_m, TARGET_POINT]
    y_va = df_use.loc[va_m, TARGET_POINT]
    y_te = df_use.loc[te_m, TARGET_POINT]
    y_sp = np.clip(df_use.loc[te_m, "smart_persist"].values, PRED_MIN, PRED_MAX)

    if CKPT_PATH.exists():
        with open(CKPT_PATH, "rb") as f:
            results = pickle.load(f)
    else:
        results = []

    done_models = {r["model"] for r in results}

    if "catboost" not in done_models:
        print("Training CatBoost...")
        cb = train_catboost(x_tr, y_tr, x_va, y_va)
        pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
        m = metrics(y_te, pred, y_sp)
        m.update({"model": "catboost", "seed": 0, "type": "GBM"})
        results.append(m)
        with open(CKPT_PATH, "wb") as f:
            pickle.dump(results, f)
        print(f"  CatBoost R2={m['r2']:.4f} MAE={m['mae']:.1f} iter={cb.get_best_iteration()}")
        return
    elif "lgbm" not in done_models:
        print("Training LightGBM...")
        lgb_pipe = train_lgbm(x_tr, y_tr, x_va, y_va)
        pred = np.clip(lgb_pipe.predict(x_te), PRED_MIN, PRED_MAX)
        m = metrics(y_te, pred, y_sp)
        m.update({"model": "lgbm", "seed": 0, "type": "GBM"})
        results.append(m)
        with open(CKPT_PATH, "wb") as f:
            pickle.dump(results, f)
        print(f"  LightGBM R2={m['r2']:.4f} MAE={m['mae']:.1f}")
        return

    print("\n=== GBM PORTION DONE ===")
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "arm_B_gbm_results_v2.csv", index=False)
    for r in results:
        print(f"  {r['model']}: R2={r['r2']:.4f}")
    print(f"\nSaved -> {OUTPUT_DIR}/arm_B_gbm_results_v2.csv")
    print("\nNOTE: bagian DL (LSTM/MLP/Transformer, 3-seed) BELUM dijalankan -- "
          "torch tidak terpasang di sandbox, instalasi sempat dihentikan user. "
          "Menunggu izin/keputusan sebelum dicoba lagi.")


if __name__ == "__main__":
    main()
