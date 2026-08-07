#!/usr/bin/env python3
"""
R8 Arm C — JAMBI: Top-K Pruning dari F1 lean-50 (replikasi protokol Bengkulu tervalidasi)

Konteks (diagnosa Bengkulu 2026-07-18, diag_ablation.csv):
  - Superset dgn radiasi mentah (DNI/DHI) justru MERUGIKAN (-0.02..-0.03) di bawah
    distribution shift -> pruning yang valid dilakukan DARI F1 lean-50, bukan superset.
  - Bengkulu: top-20 = 0.792 (identik full-50); top-30 = 0.7912; top-40 = 0.7917.

Protokol (identik Bengkulu):
  1. Train CatBoost full F1-50 (params R1: 4000 it, lr 0.02, d8, RMSE, ES 150 di val)
  2. Ranking fitur by PredictionValuesChange importance
  3. Sweep top-K, K = 10, 15, 20, 25, 30, 35, 40, 50(full)
  4. Laporkan K minimal dgn R2 >= full - 0.001 (epsilon)
  Target: point t+60. Split/filter identik R1.

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_jambi_R8_armC.py
"""
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from train_ghi_1h_jambi_R1_benchmark import (
    build_dataset, FEATURES as F1_FEATURES, TARGET_POINT, TIME_COL,
)

OUTPUT_DIR = Path("outputs_R8_jambi")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42
K_SWEEP = [10, 15, 20, 25, 30, 35, 40]
EPSILON = 0.001

assert len(F1_FEATURES) == 50


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


def train_catboost(x_tr, y_tr, x_va, y_va):
    m = CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_va.astype(float).values, y_va.astype(float).values),
          early_stopping_rounds=150)
    return m


def metrics(y_true, y_pred, y_sp):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_sp = float(np.sqrt(mean_squared_error(y_true, y_sp)))
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / rmse_sp if rmse_sp > 0 else 0.0, 4),
    }


def main():
    print("R8 Arm C — Jambi: top-K pruning dari F1 lean-50", flush=True)
    print("Building harmonised dataset (R1 builder)...", flush=True)
    df = build_dataset()
    df_use = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    tr_m, va_m, te_m = split_masks(df_use)
    print(f"Rows: {len(df_use):,}  (train={tr_m.sum():,} val={va_m.sum():,} test={te_m.sum():,})", flush=True)

    x_tr = df_use.loc[tr_m, F1_FEATURES]
    x_va = df_use.loc[va_m, F1_FEATURES]
    x_te = df_use.loc[te_m, F1_FEATURES]
    y_tr = df_use.loc[tr_m, TARGET_POINT]
    y_va = df_use.loc[va_m, TARGET_POINT]
    y_te = df_use.loc[te_m, TARGET_POINT]
    y_sp = np.clip(df_use.loc[te_m, "smart_persist"].values, PRED_MIN, PRED_MAX)

    # 1-2. Full F1 + importance ranking
    print("\nTraining CatBoost full F1-50...", flush=True)
    cb_full = train_catboost(x_tr, y_tr, x_va, y_va)
    pred = np.clip(cb_full.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
    m_full = metrics(y_te, pred, y_sp)
    print(f"  full-50: R2={m_full['r2']:.4f}  MAE={m_full['mae']:.1f}", flush=True)

    imp = pd.Series(
        cb_full.get_feature_importance(type="PredictionValuesChange"),
        index=F1_FEATURES,
    ).sort_values(ascending=False)
    imp.to_csv(OUTPUT_DIR / "arm_C_importance.csv", header=["importance"])
    print("\nTop 20 importance:", flush=True)
    for feat, v in imp.head(20).items():
        print(f"  {v:6.2f}  {feat}", flush=True)

    # 3. Top-K sweep
    print("\nTop-K sweep:", flush=True)
    rows = [dict(K=50, **m_full)]
    for K in K_SWEEP:
        feats_k = imp.head(K).index.tolist()
        cb_k = train_catboost(df_use.loc[tr_m, feats_k], y_tr,
                              df_use.loc[va_m, feats_k], y_va)
        pred_k = np.clip(cb_k.predict(df_use.loc[te_m, feats_k].astype(float).values),
                         PRED_MIN, PRED_MAX)
        m_k = metrics(y_te, pred_k, y_sp)
        rows.append(dict(K=K, **m_k))
        print(f"  K={K:3d}: R2={m_k['r2']:.4f}  MAE={m_k['mae']:.1f}  "
              f"(delta vs full: {m_k['r2']-m_full['r2']:+.4f})", flush=True)

    res = pd.DataFrame(rows).sort_values("K")
    res.to_csv(OUTPUT_DIR / "arm_C_topk_sweep.csv", index=False)

    # 4. K minimal dalam epsilon
    ok = res[(res["K"] < 50) & (res["r2"] >= m_full["r2"] - EPSILON)]
    if len(ok):
        k_min = int(ok["K"].min())
        r2_min = float(ok.loc[ok["K"] == k_min, "r2"].iloc[0])
        print(f"\nK minimal (R2 >= full-{EPSILON}): K={k_min} (R2={r2_min:.4f} vs full {m_full['r2']:.4f})", flush=True)
        top_feats = imp.head(k_min).index.tolist()
        pd.Series(top_feats).to_csv(OUTPUT_DIR / "arm_C_pruned_features.csv",
                                    index=False, header=["feature"])
    else:
        print(f"\nTidak ada K < 50 dalam epsilon {EPSILON} dari full — pertahankan 50 fitur.", flush=True)

    print(f"\n-> outputs: {OUTPUT_DIR}/arm_C_topk_sweep.csv, arm_C_importance.csv", flush=True)


if __name__ == "__main__":
    main()
