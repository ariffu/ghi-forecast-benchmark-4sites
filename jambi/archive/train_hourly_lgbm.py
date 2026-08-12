#!/usr/bin/env python3
"""
train_hourly_lgbm.py
=====================
Training LightGBM untuk dataset GHI perjam Jambi.

Dataset: dataset_hourly/ (output dari build_ghi_hourly_dataset.py)
Target : ghi_next = rata-rata GHI jam berikutnya (W/m²)

Perbandingan vs model 10-menit:
  - 10-menit: R² ~0.517, target = GHI rata-rata 10 menit ke depan (sampai 60 menit)
  - Perjam  : target = GHI rata-rata JAM BERIKUTNYA (lebih smooth → ekspektasi R² lebih tinggi)
  - cloud_oktas SYNOP real-time → bukan reanalysis, tidak ada leakage

Stages:
  Stage 1 — Solar only (ghi, kt, clearsky, sun position, lag 1-3h)
  Stage 2 — + Cloud SYNOP (oktas, cloud type, cloud base, present_weather)
  Stage 3 — + Meteorologi (temp, RH, pressure, wind, rain, visibility)
"""

import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

DATASET_DIR = Path(__file__).parent / "dataset_hourly"
OUTPUT_DIR  = Path(__file__).parent / "models_hourly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "ghi_next"

LGBM_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 3000,
    learning_rate     = 0.01,
    num_leaves        = 127,       # lebih kecil karena dataset ~7K train
    min_child_samples = 10,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)
EARLY_STOP = 150

# ─── FEATURE GROUPS ───────────────────────────────────────────────────────────
# Stage 1: Solar only
SOLAR_FEATS = [
    "ghi_h", "ghi_last", "ghi_std", "ghi_std_norm",
    "clearsky_h", "clearsky_pvlib_h",
    "sun_alt_h", "sun_az_h", "sun_alt_pvlib_h",
    "kt_h", "kt_last",
    "ghi_lag1", "ghi_lag2", "ghi_lag3",
    "ghi_last_lag1", "ghi_std_lag1",
    "kt_lag1", "kt_lag2", "kt_lag3",
    "delta_ghi_1h", "delta_kt_1h",
    # Future sun geometry (no leakage: hanya geometri, bukan GHI aktual)
    "clearsky_pvlib_next", "sun_alt_pvlib_next", "sun_az_pvlib_next",
    # Cyclical time
    "month_sin", "month_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

# Stage 2: + Cloud SYNOP
CLOUD_FEATS = [
    "cloud_oktas", "cloud_base_m",
    "cloud_low_type", "cloud_med_type", "cloud_high_type",
    "present_weather",
    "cloud_oktas_lag1", "cloud_oktas_lag2", "cloud_oktas_lag3",
    "delta_cloud_1h",
    "cloud_clearsky_ratio", "ghi_cloud_interact",
    "is_cb", "is_cu", "is_sc",
    "sky_clear", "sky_scattered", "sky_broken", "sky_overcast",
]

# Stage 3: + Meteo
METEO_FEATS = [
    "temp_air", "rh", "pressure", "ws", "ws_max",
    "rain_sum", "rain_flag",
    "visibility_km", "pressure_tend_3h", "rain_6h",
    "temp_lag1", "rh_lag1", "ws_lag1",
    "rain_lag1", "rain_lag2", "rain_prev_flag",
    "vis_lag1",
    "delta_temp_1h", "delta_rh_1h",
]

STAGE_FEATS = {
    1: SOLAR_FEATS,
    2: SOLAR_FEATS + CLOUD_FEATS,
    3: SOLAR_FEATS + CLOUD_FEATS + METEO_FEATS,
}
STAGE_LABELS = {
    1: "Solar only",
    2: "Solar + Cloud SYNOP",
    3: "Solar + Cloud + Meteo",
}


def load_split(split):
    return pd.read_parquet(DATASET_DIR / f"jambi_hourly_{split}.parquet")


def metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return dict(n=0, r2=np.nan, rmse=np.nan, mae=np.nan)
    return dict(
        n    = len(yt),
        r2   = r2_score(yt, yp),
        rmse = np.sqrt(mean_squared_error(yt, yp)),
        mae  = mean_absolute_error(yt, yp),
    )


def persistence_baseline(tr, va, te):
    """Persistence: GHI(t+1) ≈ GHI(t), Smart: scale dengan clearsky ratio."""
    print("  [BASELINE PERSISTENCE]")
    for name, df in [("val", va), ("test", te)]:
        y   = df[TARGET_COL].values
        yp  = df["ghi_h"].values   # persistence
        # Smart persistence: scale dengan clearsky ratio
        r_cs = np.where(
            df["clearsky_pvlib_h"].values > 10,
            df["clearsky_pvlib_next"].values / df["clearsky_pvlib_h"].values,
            1.0
        )
        yp_smart = np.clip(yp * r_cs, 0, None)
        m_p = metrics(y, yp)
        m_s = metrics(y, yp_smart)
        print(f"  {name:5s}: Persist R²={m_p['r2']:.4f} RMSE={m_p['rmse']:.1f} | "
              f"Smart-Persist R²={m_s['r2']:.4f} RMSE={m_s['rmse']:.1f}")
    print()


def train_stage(stage, records):
    print(f"\n{'='*58}")
    print(f"  Stage {stage}: {STAGE_LABELS[stage]}")
    print(f"{'='*58}")

    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")

    # Fitur yang tersedia dalam dataset
    req_feats = STAGE_FEATS[stage]
    fc = [c for c in req_feats if c in tr.columns]
    missing = [c for c in req_feats if c not in tr.columns]
    if missing:
        print(f"  ⚠  Fitur tidak ada (skip): {missing[:5]}{'...' if len(missing)>5 else ''}")
    print(f"  Fitur: {len(fc)}")

    # Cek null rate fitur cloud di test
    for col in ["cloud_oktas", "visibility_km", "pressure_tend_3h"]:
        if col in fc:
            nr = te[col].isna().mean()
            if nr > 0.1:
                print(f"  ⚠  {col} NULL di test: {nr:.1%}")

    Xtr = tr[fc].values.astype(float)
    Xva = va[fc].values.astype(float)
    Xte = te[fc].values.astype(float)
    ytr = tr[TARGET_COL].values.astype(float)
    yva = va[TARGET_COL].values.astype(float)
    yte = te[TARGET_COL].values.astype(float)

    mtr = ~np.isnan(ytr)
    mva = ~np.isnan(yva)

    t0 = time.time()
    dtrain = lgb.Dataset(Xtr[mtr], label=ytr[mtr], feature_name=fc, free_raw_data=False)
    dval   = lgb.Dataset(Xva[mva], label=yva[mva], feature_name=fc,
                         free_raw_data=False, reference=dtrain)

    model = lgb.train(
        LGBM_PARAMS, dtrain, valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best = model.best_iteration
    pva  = model.predict(Xva, num_iteration=best)
    pte  = model.predict(Xte, num_iteration=best)

    mva_ = metrics(yva, pva)
    mte_ = metrics(yte, pte)
    elapsed = time.time() - t0

    print(f"  iter={best} | elapsed={elapsed:.1f}s")
    print(f"  Val  R²={mva_['r2']:.4f} RMSE={mva_['rmse']:.1f} MAE={mva_['mae']:.1f}")
    print(f"  Test R²={mte_['r2']:.4f} RMSE={mte_['rmse']:.1f} MAE={mte_['mae']:.1f}")

    model.save_model(str(OUTPUT_DIR / f"hourly_s{stage}.txt"))

    # Feature importance
    imp = pd.DataFrame({
        "feature": fc,
        "gain"   : model.feature_importance("gain"),
        "split"  : model.feature_importance("split"),
    }).sort_values("gain", ascending=False)
    imp["pct"] = imp["gain"] / imp["gain"].sum() * 100
    imp.to_csv(OUTPUT_DIR / f"hourly_s{stage}_imp.csv", index=False)

    # Top 15 fitur
    print(f"\n  Top 15 fitur:")
    for _, row in imp.head(15).iterrows():
        print(f"    {row['feature']:35s} {row['pct']:5.2f}%")

    for sp, m in [("val", mva_), ("test", mte_)]:
        records.append(dict(stage=stage, label=STAGE_LABELS[stage], split=sp,
                            best_iter=best, **m))

    return pva, pte, yva, yte


def residual_analysis(stage, pva, pte, yva, yte, va, te):
    """Analisis error per regime cloud."""
    print(f"\n  [ANALISIS RESIDUAL — Stage {stage}]")
    for name, pred, true, df in [("val", pva, yva, va), ("test", pte, yte, te)]:
        if "cloud_oktas" not in df.columns:
            continue
        res = pd.DataFrame({
            "y_true" : true,
            "y_pred" : pred,
            "oktas"  : df["cloud_oktas"].values,
            "ghi_h"  : df["ghi_h"].values,
        }).dropna()

        print(f"\n  {name.upper()} — R² per cloud regime (oktas):")
        bins = [(0, 2, "Clear 0-2"), (3, 5, "Scattered 3-5"),
                (6, 7, "Broken 6-7"), (8, 9, "Overcast 8+")]
        for lo, hi, lbl in bins:
            sub = res[(res.oktas >= lo) & (res.oktas <= hi)]
            if len(sub) < 10:
                continue
            r2_v = r2_score(sub.y_true, sub.y_pred)
            rmse_v = np.sqrt(mean_squared_error(sub.y_true, sub.y_pred))
            pct = len(sub) / len(res) * 100
            print(f"    {lbl:20s}: n={len(sub):4d} ({pct:4.1f}%) R²={r2_v:.4f} RMSE={rmse_v:.1f}")


def summary(records):
    df = pd.DataFrame(records)
    lines = []
    lines.append("\n" + "=" * 68)
    lines.append("HASIL FINAL — GHI Hourly Forecasting Jambi")
    lines.append("Dataset: perjam, Target: GHI rata-rata jam berikutnya")
    lines.append("=" * 68)

    for split, yr in [("val", "2024"), ("test", "2025")]:
        lines.append(f"\n[{split.upper()} {yr}]")
        lines.append(f"  {'Stage':30s} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8} | Iter")
        sub = df[df.split == split].sort_values("stage")
        for _, row in sub.iterrows():
            lines.append(f"  {row['label']:30s} | {row['r2']:8.4f} | "
                         f"{row['rmse']:8.1f} | {row['mae']:8.1f} | {row['best_iter']:.0f}")

    lines.append("\n[KONSISTENSI val vs test]")
    for s in df.stage.unique():
        va = df[(df.stage==s)&(df.split=="val")]["r2"].values[0]
        te = df[(df.stage==s)&(df.split=="test")]["r2"].values[0]
        diff = te - va
        ok = "✓" if abs(diff) < 0.05 else "⚠ gap besar"
        lines.append(f"  S{s}: val={va:.4f} test={te:.4f} Δ={diff:+.4f} {ok}")

    lines.append("\n[PERBANDINGAN vs MODEL 10-MENIT]")
    lines.append("  10-menit LightGBM S2: val=0.5061 test=0.5171 (horizon 10-60 menit)")
    for sp in ["val", "test"]:
        sub = df[(df.stage==2)&(df.split==sp)]
        if not sub.empty:
            r2 = sub["r2"].values[0]
            delta = r2 - (0.5061 if sp=="val" else 0.5171)
            lines.append(f"  Hourly S2 {sp:4s}: R²={r2:.4f} Δ={delta:+.4f} vs 10-menit")

    lines.append("=" * 68)
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("TRAIN HOURLY MODEL — GHI Jambi (SYNOP Cloud + Meteo)")
    print("=" * 60)

    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")
    print(f"  Train: {len(tr):,}  Val: {len(va):,}  Test: {len(te):,}")
    print(f"  Fitur tersedia: {len([c for c in tr.columns if c not in ['hour_wib','hour_next',TARGET_COL]])}")

    # Persistence baseline
    persistence_baseline(tr, va, te)

    records = []
    pred_results = {}

    for s in [1, 2, 3]:
        pva, pte, yva, yte = train_stage(s, records)
        pred_results[s] = (pva, pte, yva, yte)

    # Analisis residual Stage 3
    for s in [2, 3]:
        pva, pte, yva, yte = pred_results[s]
        residual_analysis(s, pva, pte, yva, yte, va, te)

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUTPUT_DIR / "metrics_hourly.csv", index=False)

    result = summary(records)
    print(result)
    with open(OUTPUT_DIR / "metrics_hourly_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nModels → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
