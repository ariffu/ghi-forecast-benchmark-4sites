#!/usr/bin/env python3
"""
train_lgbm.py
=============
Training LightGBM baseline prediksi GHI — Stasiun Jambi.

6 model terpisah per stage, satu per horizon (t+1..t+6 × 10 menit = 1 jam):
  Stage 1 — Radiasi + sun position saja
  Stage 2 — + Meteorologi
  Stage 3 — + PTM cloud opacity lag + AERONET aerosol

Referensi Kalbar:
  n_estimators=1000-2000, lr=0.03, num_leaves=255, early_stopping=80 rounds
  R² ceiling tropis ~0.80 (clear regime: ~0.92, overcast: ~0.60)

Output per stage:
  models/s{N}_h{h}_model.txt       ← model LightGBM teks
  models/s{N}_h{h}_importance.csv  ← feature importance (gain)
  models/metrics_all.csv           ← ringkasan R²/RMSE/MAE semua stage+horizon
  models/metrics_summary.txt       ← laporan ringkas siap baca
"""

import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR  = Path(__file__).parent / "models"

HORIZON = 6

# LightGBM hyperparameters (dari pelajaran kalbar)
LGBM_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 2000,         # max; early stopping akan berhenti lebih awal
    learning_rate     = 0.03,
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
EARLY_STOPPING_ROUNDS = 80

# Kolom yang BUKAN fitur
EXCLUDE_FROM_FEATURES = (
    {"anchor_ts"}
    | {f"ghi_h{h}" for h in range(1, HORIZON + 1)}
    | {f"kt_h{h}"  for h in range(1, HORIZON + 1)}
)

STAGES = [1, 2, 3]


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_split(stage: int, split: str) -> pd.DataFrame:
    path = DATASET_DIR / f"jambi_ghi_s{stage}_{split}.parquet"
    return pd.read_parquet(path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]


def metrics(y_true, y_pred, label: str) -> dict:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10:
        return {"label": label, "n": len(yt), "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    return {"label": label, "n": len(yt), "r2": r2, "rmse": rmse, "mae": mae}


def rmse_score(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


# ─── TRAINING PER STAGE ───────────────────────────────────────────────────────
def train_stage(stage: int, all_records: list) -> None:
    print(f"\n{'='*60}")
    print(f"STAGE {stage}")
    print(f"{'='*60}")

    train = load_split(stage, "train")
    val   = load_split(stage, "val")
    test  = load_split(stage, "test")

    feat_cols = get_feature_cols(train)
    print(f"  Fitur    : {len(feat_cols)} kolom")
    print(f"  Train    : {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Periksa NULL rate fitur kunci di test
    key_feats = ["ptm_cloud_opacity", "aeronet_aod500", "temp_air"]
    for kf in key_feats:
        if kf in train.columns:
            nr = test[kf].isna().mean()
            if nr > 0.5:
                print(f"  ⚠  {kf} NULL di test: {nr:.0%} — fitur tidak efektif untuk 2025")

    X_train = train[feat_cols].values.astype(float)
    X_val   = val[feat_cols].values.astype(float)
    X_test  = test[feat_cols].values.astype(float)

    for h in range(1, HORIZON + 1):
        target_col = f"ghi_h{h}"
        t0 = time.time()

        y_train = train[target_col].values.astype(float)
        y_val   = val[target_col].values.astype(float)
        y_test  = test[target_col].values.astype(float)

        # Filter baris dengan target valid
        mask_tr = ~np.isnan(y_train)
        mask_va = ~np.isnan(y_val)

        dtrain = lgb.Dataset(X_train[mask_tr], label=y_train[mask_tr],
                             feature_name=feat_cols, free_raw_data=False)
        dval   = lgb.Dataset(X_val[mask_va],   label=y_val[mask_va],
                             feature_name=feat_cols, free_raw_data=False,
                             reference=dtrain)

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),   # senyap
        ]

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration
        elapsed   = time.time() - t0

        # Prediksi
        pred_val  = model.predict(X_val,  num_iteration=best_iter)
        pred_test = model.predict(X_test, num_iteration=best_iter)

        m_val  = metrics(y_val,  pred_val,  f"s{stage}_h{h}_val")
        m_test = metrics(y_test, pred_test, f"s{stage}_h{h}_test")

        print(f"  h+{h} ({(h*10):2d}min) | best_iter={best_iter:4d} | "
              f"Val  R²={m_val['r2']:.4f}  RMSE={m_val['rmse']:.1f}  MAE={m_val['mae']:.1f} | "
              f"Test R²={m_test['r2']:.4f}  RMSE={m_test['rmse']:.1f}  MAE={m_test['mae']:.1f} | "
              f"{elapsed:.1f}s")

        # Simpan model
        model_path = OUTPUT_DIR / f"s{stage}_h{h}_model.txt"
        model.save_model(str(model_path))

        # Feature importance (gain)
        imp = pd.DataFrame({
            "feature"  : feat_cols,
            "importance_gain"  : model.feature_importance(importance_type="gain"),
            "importance_split" : model.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
        imp.to_csv(OUTPUT_DIR / f"s{stage}_h{h}_importance.csv", index=False)

        # Catat metrics
        for m in [m_val, m_test]:
            all_records.append({
                "stage"    : stage,
                "horizon_h": h,
                "horizon_min": h * 10,
                "split"    : "val"  if "val"  in m["label"] else "test",
                "n"        : m["n"],
                "r2"       : m["r2"],
                "rmse"     : m["rmse"],
                "mae"      : m["mae"],
                "best_iter": best_iter,
            })


# ─── RINGKASAN ────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append("\n" + "="*65)
    lines.append("RINGKASAN HASIL — LightGBM GHI Forecasting Jambi")
    lines.append("="*65)

    for stage in STAGES:
        lines.append(f"\nStage {stage}:")
        lines.append(f"  {'Horizon':>8} | {'Val R²':>8} | {'Val RMSE':>9} | {'Test R²':>8} | {'Test RMSE':>10}")
        lines.append(f"  {'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")

        for h in range(1, HORIZON + 1):
            row_val  = df[(df.stage == stage) & (df.horizon_h == h) & (df.split == "val")]
            row_test = df[(df.stage == stage) & (df.horizon_h == h) & (df.split == "test")]
            if row_val.empty or row_test.empty:
                continue
            rv, rt = row_val.iloc[0], row_test.iloc[0]
            lines.append(
                f"  t+{h} ({h*10:2d}min) | {rv.r2:8.4f} | {rv.rmse:9.1f} | "
                f"{rt.r2:8.4f} | {rt.rmse:10.1f}"
            )

        # Rata-rata
        sv = df[(df.stage == stage) & (df.split == "val")]
        st = df[(df.stage == stage) & (df.split == "test")]
        lines.append(
            f"  {'MEAN':>8}   | {sv.r2.mean():8.4f} | {sv.rmse.mean():9.1f} | "
            f"{st.r2.mean():8.4f} | {st.rmse.mean():10.1f}"
        )

    # Perbandingan antar stage (val)
    lines.append("\n\nPerbandingan Stage (Val 2024, mean semua horizon):")
    lines.append(f"  {'Stage':>6} | {'R² mean':>8} | {'RMSE mean':>10} | {'MAE mean':>9}")
    for s in STAGES:
        sv = df[(df.stage == s) & (df.split == "val")]
        lines.append(
            f"  {s:>6} | {sv.r2.mean():8.4f} | {sv.rmse.mean():10.1f} | {sv.mae.mean():9.1f}"
        )

    lines.append("\n\nCatatan:")
    lines.append("  - Test 2025: ptm_cloud_opacity & aeronet 100% NULL → Stage 3 ≈ Stage 1 di test")
    lines.append("  - Val 2024 : ptm NULL=0%, aeronet NULL=53% → gunakan val untuk bandingkan Stage 3 vs 1")
    lines.append("  - R² ceiling tropis ~0.80 (kalbar benchmark)")
    lines.append("  - Clear-sky regime biasanya R²>0.90 jika dipisah")
    lines.append("="*65)

    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("LightGBM Training — GHI Forecasting Jambi")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Params  : lr={LGBM_PARAMS['learning_rate']}, num_leaves={LGBM_PARAMS['num_leaves']}, "
          f"max_iter={LGBM_PARAMS['n_estimators']}, early_stop={EARLY_STOPPING_ROUNDS}")

    all_records = []
    t_total = time.time()

    for stage in STAGES:
        train_stage(stage, all_records)

    # Simpan metrics
    metrics_df = pd.DataFrame(all_records)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_all.csv", index=False)

    # Ringkasan
    summary = print_summary(metrics_df)
    print(summary)

    with open(OUTPUT_DIR / "metrics_summary.txt", "w") as f:
        f.write(summary)

    elapsed = time.time() - t_total
    print(f"\nTotal training time: {elapsed/60:.1f} menit")
    print(f"Models saved to    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()