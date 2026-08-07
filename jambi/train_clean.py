#!/usr/bin/env python3
"""
train_clean.py
==============
Training final model GHI Jambi — dataset bersih tanpa leakage.

Membandingkan:
  Stage 1 clean  — radiasi + sun position saja
  Stage 2 clean  — + meteorologi
  Stage 3 clean  — + ptm_cloud_prev (jam sebelumnya) + aeronet

Target: lihat apakah R² konsisten antara Val 2024 dan Test 2025
(Stage 3 leaky sebelumnya: val=0.57 tapi test=0.21 — tidak stabil)
"""

import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

DATASET_DIR = Path(__file__).parent / "dataset_clean"
OUTPUT_DIR  = Path(__file__).parent / "models_clean"
HORIZON     = 6

LGBM_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 3000,
    learning_rate     = 0.01,
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
EARLY_STOPPING = 150

EXCL_COLS = (
    {"anchor_ts"}
    | {f"ghi_h{h}" for h in range(1, HORIZON + 1)}
    | {f"kt_h{h}"  for h in range(1, HORIZON + 1)}
)


def load(stage, split):
    return pd.read_parquet(DATASET_DIR / f"jambi_clean_s{stage}_{split}.parquet")


def feat_cols(df):
    return [c for c in df.columns if c not in EXCL_COLS]


def metrics(y_true, y_pred):
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 5:
        return dict(n=0, r2=np.nan, rmse=np.nan, mae=np.nan)
    return dict(n=len(yt),
                r2=r2_score(yt, yp),
                rmse=np.sqrt(mean_squared_error(yt, yp)),
                mae=mean_absolute_error(yt, yp))


def train_stage(stage, records):
    print(f"\n{'='*55}")
    print(f"  Stage {stage} (clean)")
    print(f"{'='*55}")
    tr = load(stage, "train")
    va = load(stage, "val")
    te = load(stage, "test")

    fc = feat_cols(tr)
    print(f"  Fitur: {len(fc)}")

    # Cek NULL rate fitur kunci di test
    for col in ["ptm_cloud_prev", "aeronet_aod500"]:
        if col in fc:
            nr = te[col].isna().mean()
            if nr > 0.3:
                print(f"  ⚠  {col} NULL di test: {nr:.0%}")

    Xtr = tr[fc].values.astype(float)
    Xva = va[fc].values.astype(float)
    Xte = te[fc].values.astype(float)

    for h in range(1, HORIZON + 1):
        col = f"ghi_h{h}"
        t0  = time.time()

        ytr = tr[col].values.astype(float)
        yva = va[col].values.astype(float)
        yte = te[col].values.astype(float)

        mtr = ~np.isnan(ytr)
        mva = ~np.isnan(yva)

        dtrain = lgb.Dataset(Xtr[mtr], label=ytr[mtr], feature_name=fc, free_raw_data=False)
        dval   = lgb.Dataset(Xva[mva], label=yva[mva], feature_name=fc,
                             free_raw_data=False, reference=dtrain)

        model = lgb.train(LGBM_PARAMS, dtrain, valid_sets=[dval], callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ])

        best = model.best_iteration
        pva  = model.predict(Xva, num_iteration=best)
        pte  = model.predict(Xte, num_iteration=best)

        mva_ = metrics(yva, pva)
        mte_ = metrics(yte, pte)
        elapsed = time.time() - t0

        print(f"  h+{h} ({h*10:2d}m) | iter={best:4d} | "
              f"Val R²={mva_['r2']:.4f} RMSE={mva_['rmse']:.1f} | "
              f"Test R²={mte_['r2']:.4f} RMSE={mte_['rmse']:.1f} | {elapsed:.1f}s")

        model.save_model(str(OUTPUT_DIR / f"clean_s{stage}_h{h}.txt"))

        # Feature importance
        imp = pd.DataFrame({
            "feature"  : fc,
            "gain"     : model.feature_importance("gain"),
            "split"    : model.feature_importance("split"),
        }).sort_values("gain", ascending=False)
        imp["pct"] = imp["gain"] / imp["gain"].sum() * 100
        imp.to_csv(OUTPUT_DIR / f"clean_s{stage}_h{h}_imp.csv", index=False)

        for sp, m in [("val", mva_), ("test", mte_)]:
            records.append(dict(stage=stage, h=h, split=sp, best_iter=best, **m))


def summary(df):
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("HASIL FINAL — GHI Forecasting Jambi (Clean, No Leakage)")
    lines.append("=" * 70)

    stages = {
        1: "Radiasi + sun pos",
        2: "+ Meteo",
        3: "+ PTM prev-hour + AERONET",
    }

    for split, yr in [("val", "2024"), ("test", "2025")]:
        lines.append(f"\n[{split.upper()} {yr}]")
        lines.append(f"  {'Stage':30s}|" + "|".join(f"h+{h}({h*10}m)" for h in range(1,7)) + "| Mean R²")
        for s, label in stages.items():
            sub = df[(df.stage==s) & (df.split==split)].sort_values("h")
            if sub.empty: continue
            r2s  = "|".join(f"{r:.4f}" for r in sub.r2)
            mean = sub.r2.mean()
            lines.append(f"  {label:30s}|{r2s}| {mean:.4f}")

    lines.append("\n[PERBANDINGAN ΔR² — Stage 3 vs Stage 1, clean]")
    for split, yr in [("val","2024"), ("test","2025")]:
        s1 = df[(df.stage==1) & (df.split==split)].r2.mean()
        s2 = df[(df.stage==2) & (df.split==split)].r2.mean()
        s3 = df[(df.stage==3) & (df.split==split)].r2.mean()
        lines.append(f"  {split.upper()} {yr}: "
                     f"S1={s1:.4f} | S2={s2:.4f} (Δ{s2-s1:+.4f}) | "
                     f"S3={s3:.4f} (Δ{s3-s1:+.4f} dari S1, Δ{s3-s2:+.4f} dari S2)")

    lines.append("\n[KONSISTENSI val vs test]")
    for s, label in stages.items():
        va = df[(df.stage==s)&(df.split=="val")].r2.mean()
        te = df[(df.stage==s)&(df.split=="test")].r2.mean()
        diff = te - va
        ok = "✓" if abs(diff) < 0.03 else "⚠ tidak konsisten"
        lines.append(f"  S{s}: val={va:.4f} test={te:.4f} Δ={diff:+.4f} {ok}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    print("=" * 60)
    print("TRAIN CLEAN MODEL — Jambi GHI Forecast (no leakage)")
    print("=" * 60)

    for s in [1, 2, 3]:
        train_stage(s, records)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "metrics_clean.csv", index=False)

    result = summary(df)
    print(result)
    with open(OUTPUT_DIR / "metrics_clean_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nModels → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
