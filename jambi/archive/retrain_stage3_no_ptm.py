#!/usr/bin/env python3
"""
retrain_stage3_no_ptm.py
=========================
Retrain khusus untuk menangani gap ptm_cloud_opacity & AERONET di 2025 test.

Strategi:
  Stage 3b — Stage 3 tanpa fitur ptm_* dan aeronet_*
             → Dapat dievaluasi di test 2025 secara fair
             → Dibanding Stage 2 untuk melihat kontribusi cloud_op_lag

  Juga: re-run Stage 1+2+3 dengan LR turun (0.01) dan early_stop lebih longgar (150)
        untuk memastikan tidak ada underfitting akibat early stopping terlalu cepat.

Output:
  models/s3b_h{h}_model.txt         ← Stage 3 tanpa ptm/aeronet
  models/s3b_h{h}_importance.csv
  models/metrics_tuned.csv          ← semua stage dengan LR=0.01
  models/metrics_comparison.txt     ← perbandingan lengkap
"""

import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR  = Path(__file__).parent / "models"
HORIZON     = 6

# ─── Hyperparameter: turun LR, longgar early stop ─────────────────────────────
LGBM_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 3000,
    learning_rate     = 0.01,         # diturunkan dari 0.03 → lebih halus
    num_leaves        = 255,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)
EARLY_STOPPING_ROUNDS = 150   # lebih longgar dari 80

# Fitur yang di-drop di Stage 3b (tidak tersedia di 2025)
PTM_AERONET_COLS = (
    ["ptm_cloud_opacity", "ptm_precipitable_water"]
    + [f"cloud_op_lag{k}" for k in range(1, 19)]
    + ["aeronet_aod500", "aeronet_ae", "aeronet_pwv_cm",
       "aeronet_aod500_std", "aeronet_smoke_flag", "aeronet_n_obs"]
)

EXCLUDE_FROM_FEATURES = (
    {"anchor_ts"}
    | {f"ghi_h{h}" for h in range(1, HORIZON + 1)}
    | {f"kt_h{h}"  for h in range(1, HORIZON + 1)}
)


def load_split(stage: int, split: str) -> pd.DataFrame:
    return pd.read_parquet(DATASET_DIR / f"jambi_ghi_s{stage}_{split}.parquet")


def get_feature_cols(df: pd.DataFrame, drop_extra=None) -> list:
    cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    if drop_extra:
        cols = [c for c in cols if c not in drop_extra]
    return cols


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10:
        return dict(n=len(yt), r2=np.nan, rmse=np.nan, mae=np.nan)
    return dict(
        n    = len(yt),
        r2   = r2_score(yt, yp),
        rmse = np.sqrt(mean_squared_error(yt, yp)),
        mae  = mean_absolute_error(yt, yp),
    )


def train_one_stage(label: str, stage_src: int, drop_cols: list,
                    records: list, save_prefix: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")

    train = load_split(stage_src, "train")
    val   = load_split(stage_src, "val")
    test  = load_split(stage_src, "test")

    feat_cols = get_feature_cols(train, drop_extra=drop_cols)
    print(f"  Fitur: {len(feat_cols)}")

    X_tr = train[feat_cols].values.astype(float)
    X_va = val[feat_cols].values.astype(float)
    X_te = test[feat_cols].values.astype(float)

    for h in range(1, HORIZON + 1):
        col = f"ghi_h{h}"
        t0  = time.time()

        y_tr = train[col].values.astype(float)
        y_va = val[col].values.astype(float)
        y_te = test[col].values.astype(float)

        mask_tr = ~np.isnan(y_tr)
        mask_va = ~np.isnan(y_va)

        dtrain = lgb.Dataset(X_tr[mask_tr], label=y_tr[mask_tr],
                             feature_name=feat_cols, free_raw_data=False)
        dval   = lgb.Dataset(X_va[mask_va], label=y_va[mask_va],
                             feature_name=feat_cols, free_raw_data=False,
                             reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS, dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        best_iter = model.best_iteration
        p_va = model.predict(X_va, num_iteration=best_iter)
        p_te = model.predict(X_te, num_iteration=best_iter)

        m_va = compute_metrics(y_va, p_va)
        m_te = compute_metrics(y_te, p_te)

        elapsed = time.time() - t0
        print(f"  h+{h} ({h*10:2d}m) | iter={best_iter:4d} | "
              f"Val  R²={m_va['r2']:.4f} RMSE={m_va['rmse']:.1f} | "
              f"Test R²={m_te['r2']:.4f} RMSE={m_te['rmse']:.1f} | {elapsed:.1f}s")

        # Simpan model
        model.save_model(str(OUTPUT_DIR / f"{save_prefix}_h{h}_model.txt"))

        # Feature importance
        imp = pd.DataFrame({
            "feature"          : feat_cols,
            "importance_gain"  : model.feature_importance("gain"),
            "importance_split" : model.feature_importance("split"),
        }).sort_values("importance_gain", ascending=False)
        imp.to_csv(OUTPUT_DIR / f"{save_prefix}_h{h}_importance.csv", index=False)

        for split_name, m in [("val", m_va), ("test", m_te)]:
            records.append(dict(
                label      = label,
                save_prefix= save_prefix,
                horizon_h  = h,
                horizon_min= h * 10,
                split      = split_name,
                **m,
                best_iter  = best_iter,
            ))


def print_comparison(df: pd.DataFrame) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PERBANDINGAN LENGKAP — GHI Forecasting Jambi (LR=0.01, early_stop=150)")
    lines.append("=" * 70)

    labels = df["label"].unique()

    for split in ["val", "test"]:
        lines.append(f"\n{'─'*70}")
        lines.append(f"  {split.upper()} {'(2024)' if split=='val' else '(2025)'}")
        lines.append(f"{'─'*70}")
        header = f"  {'Label':30s} | " + " | ".join([f"h+{h}({h*10}m)" for h in range(1, HORIZON+1)]) + " | Mean"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for lbl in labels:
            sub = df[(df.label == lbl) & (df.split == split)].sort_values("horizon_h")
            if sub.empty:
                continue
            r2s  = [f"{row.r2:.4f}" for _, row in sub.iterrows()]
            mean = sub.r2.mean()
            lines.append(f"  {lbl:30s} | " + " | ".join(r2s) + f" | {mean:.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("KESIMPULAN:")
    lines.append("  Stage 3b = Stage 3 tanpa ptm/aeronet → dapat dievaluasi di test 2025")
    lines.append("  Bandingkan Stage 3b vs Stage 2 di test untuk melihat nilai cloud_op_lag")
    lines.append("  Bandingkan Stage 3 vs Stage 3b di val untuk melihat nilai ptm_cloud_opacity")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    print("=" * 65)
    print("RETRAIN + TUNING — LR=0.01, early_stop=150")
    print("=" * 65)

    # Stage 1 (tuned)
    train_one_stage("Stage 1 — Radiasi (tuned)",
                    stage_src=1, drop_cols=[],
                    records=records, save_prefix="s1t")

    # Stage 2 (tuned)
    train_one_stage("Stage 2 — + Meteo (tuned)",
                    stage_src=2, drop_cols=[],
                    records=records, save_prefix="s2t")

    # Stage 3 (tuned, dengan ptm/aeronet)
    train_one_stage("Stage 3 — + PTM + AERONET (tuned)",
                    stage_src=3, drop_cols=[],
                    records=records, save_prefix="s3t")

    # Stage 3b — tanpa ptm/aeronet (untuk evaluasi 2025)
    train_one_stage("Stage 3b — + cloud_op_lag tanpa ptm/aeronet",
                    stage_src=3, drop_cols=PTM_AERONET_COLS,
                    records=records, save_prefix="s3b")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "metrics_tuned.csv", index=False, encoding="utf-8")

    comparison = print_comparison(df)
    print(comparison)

    with open(OUTPUT_DIR / "metrics_comparison.txt", "w", encoding="utf-8") as f:
        f.write(comparison)

    print(f"\nModels saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()