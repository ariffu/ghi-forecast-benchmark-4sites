#!/usr/bin/env python3
"""
train_multistep_lgbm.py
========================
LightGBM multi-horizon 6 jam ke depan — GHI perjam Jambi.

Dataset: dataset_multistep/ (output build_ghi_multistep_dataset.py)
Horizons: h+1..h+6 (6 model per stage)
Stages:
  S1 — Solar only (GHI lag 18h + clearsky futures)
  S2 — + Cloud SYNOP (cloud lag 18h + cloud regime)
  S3 — + Meteo (temp, RH, pressure, wind, rain, visibility)
  S4 — S3 + is_wet_season (global model dengan fitur musim)

Novelty:
  Analisis R² per musim (Hujan vs Kemarau) untuk setiap stage & horizon
  Insight: fitur apa yang paling penting di masing-masing musim
  Season-specific models: train terpisah per musim → bandingkan vs global
"""

import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

DATASET_DIR = Path(__file__).parent / "dataset_multistep"
OUTPUT_DIR  = Path(__file__).parent / "models_multistep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON   = 6
LAG_HOURS = 18
TARGET_COLS = [f"ghi_h{h}" for h in range(1, HORIZON + 1)]

LGBM_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 3000,
    learning_rate     = 0.01,
    num_leaves        = 127,
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
SOLAR_FEATS = (
    ["ghi_h", "ghi_last", "ghi_std", "ghi_std_norm", "kt_h", "kt_last",
     "clearsky_h", "clearsky_pvlib_h", "sun_alt_h", "sun_az_h", "sun_alt_pvlib_h",
     "dni_h", "ghi_mean_3h", "ghi_mean_6h",
     "delta_ghi_1h", "delta_ghi_3h", "delta_kt_1h"]
    + [f"ghi_lag{k}"      for k in range(1, LAG_HOURS + 1)]
    + [f"ghi_last_lag{k}" for k in range(1, 7)]
    + [f"kt_lag{k}"       for k in range(1, 7)]
    + [f"clearsky_pvlib_h{h}" for h in range(1, HORIZON + 1)]
    + [f"sun_alt_pvlib_h{h}"  for h in range(1, HORIZON + 1)]
    + [f"sun_az_pvlib_h{h}"   for h in range(1, HORIZON + 1)]
    + ["month_sin", "month_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)

CLOUD_FEATS = (
    ["cloud_oktas", "cloud_base_m", "cloud_low_type", "cloud_med_type",
     "cloud_high_type", "present_weather",
     "cloud_mean_6h", "cloud_mean_12h", "cloud_mean_18h", "cloud_trend_6h",
     "delta_cloud_1h", "delta_cloud_3h",
     "sky_clear", "sky_scattered", "sky_broken", "sky_overcast", "is_cb"]
    + [f"cloud_lag{k}" for k in range(1, LAG_HOURS + 1)]
)

METEO_FEATS = (
    ["temp_air", "rh", "pressure", "ws", "rain_sum", "rain_flag",
     "visibility_km", "pressure_tend_3h", "rain_6h",
     "delta_temp_1h", "delta_rh_1h"]
    + [f"temp_lag{k}" for k in range(1, 7)]
    + [f"rh_lag{k}"   for k in range(1, 7)]
    + [f"ws_lag{k}"   for k in range(1, 4)]
    + [f"rain_lag{k}" for k in range(1, 7)]
)

STAGE_FEATS = {
    1: SOLAR_FEATS,
    2: SOLAR_FEATS + CLOUD_FEATS,
    3: SOLAR_FEATS + CLOUD_FEATS + METEO_FEATS,
    4: SOLAR_FEATS + CLOUD_FEATS + METEO_FEATS + ["is_wet_season"],
}
STAGE_LABELS = {
    1: "Solar 18h lag",
    2: "+ Cloud SYNOP 18h",
    3: "+ Meteo",
    4: "+ is_wet_season",
}


def load_split(split):
    return pd.read_parquet(DATASET_DIR / f"jambi_ms_{split}.parquet")


def metrics(y_true, y_pred, label=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return dict(n=0, r2=np.nan, rmse=np.nan, mae=np.nan)
    return dict(n=len(yt), r2=r2_score(yt, yp),
                rmse=np.sqrt(mean_squared_error(yt, yp)),
                mae=mean_absolute_error(yt, yp))


def persistence_baseline(va, te):
    """Persistence dan smart persistence per horizon."""
    print("\n" + "=" * 62)
    print("  BASELINE PERSISTENCE (6 horizon)")
    print("=" * 62)
    for name, df in [("VAL 2024", va), ("TEST 2025", te)]:
        print(f"\n  {name}")
        print(f"  {'Horizon':>8} | {'Persist R²':>10} | {'Smart R²':>10} | {'RMSE':>8}")
        for h in range(1, HORIZON + 1):
            y  = df[f"ghi_h{h}"].values
            yp = df["ghi_h"].values    # persistence = anchor GHI
            # Smart: scale dengan clearsky ratio
            cs_h = df[f"clearsky_pvlib_h{h}"].values
            cs_0 = df["clearsky_pvlib_h"].values
            ratio = np.where(cs_0 > 10, cs_h / cs_0, 1.0)
            yp_sm = np.clip(yp * ratio, 0, None)
            m_p = metrics(y, yp)
            m_s = metrics(y, yp_sm)
            print(f"  h+{h} ({h:2d}h ahead) | {m_p['r2']:10.4f} | {m_s['r2']:10.4f} | {m_p['rmse']:8.1f}")


def train_stage(stage, tr, va, te, records_global):
    """Train 6 model per stage (1 per horizon)."""
    print(f"\n{'='*62}")
    print(f"  Stage {stage}: {STAGE_LABELS[stage]}")
    print(f"{'='*62}")

    req  = STAGE_FEATS[stage]
    fc   = [c for c in req if c in tr.columns]
    miss = [c for c in req if c not in tr.columns]
    if miss:
        print(f"  ⚠  Fitur skip: {miss[:3]}{'...' if len(miss)>3 else ''}")
    print(f"  Fitur: {len(fc)}")

    Xtr = tr[fc].values.astype(float)
    Xva = va[fc].values.astype(float)
    Xte = te[fc].values.astype(float)

    print(f"\n  {'Horizon':>8} | {'Val R²':>8} | {'RMSE':>8} | {'Test R²':>8} | {'RMSE':>8} | Iter")
    print(f"  {'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-----")

    preds_va, preds_te, trues_va, trues_te = {}, {}, {}, {}

    for h in range(1, HORIZON + 1):
        col = f"ghi_h{h}"
        t0  = time.time()

        ytr = tr[col].values.astype(float)
        yva = va[col].values.astype(float)
        yte = te[col].values.astype(float)

        mtr = ~np.isnan(ytr); mva_ = ~np.isnan(yva)
        dtr = lgb.Dataset(Xtr[mtr], label=ytr[mtr], feature_name=fc, free_raw_data=False)
        dva = lgb.Dataset(Xva[mva_], label=yva[mva_], feature_name=fc,
                          free_raw_data=False, reference=dtr)

        model = lgb.train(LGBM_PARAMS, dtr, valid_sets=[dva], callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(-1),
        ])

        best = model.best_iteration
        pva  = model.predict(Xva, num_iteration=best)
        pte  = model.predict(Xte, num_iteration=best)

        mva = metrics(yva, pva)
        mte = metrics(yte, pte)

        print(f"  h+{h} ({h:2d}h)    | {mva['r2']:8.4f} | {mva['rmse']:8.1f} | "
              f"{mte['r2']:8.4f} | {mte['rmse']:8.1f} | {best}")

        model.save_model(str(OUTPUT_DIR / f"ms_s{stage}_h{h}.txt"))

        # Feature importance
        imp = pd.DataFrame({
            "feature": fc,
            "gain"   : model.feature_importance("gain"),
            "split"  : model.feature_importance("split"),
        }).sort_values("gain", ascending=False)
        imp["pct"] = imp["gain"] / imp["gain"].sum() * 100
        imp.to_csv(OUTPUT_DIR / f"ms_s{stage}_h{h}_imp.csv", index=False)

        preds_va[h] = pva;  trues_va[h] = yva
        preds_te[h] = pte;  trues_te[h] = yte

        for sp, m in [("val", mva), ("test", mte)]:
            records_global.append(dict(
                stage=stage, label=STAGE_LABELS[stage],
                horizon=h, split=sp, best_iter=best, **m))

    return preds_va, preds_te, trues_va, trues_te


def seasonal_analysis(stage, preds_va, preds_te, trues_va, trues_te, va, te):
    """R² per musim Hujan/Kemarau per horizon."""
    print(f"\n  [ANALISIS MUSIM — Stage {stage}: {STAGE_LABELS[stage]}]")

    results = []
    for split, preds, trues, df_s in [
        ("val",  preds_va, trues_va, va),
        ("test", preds_te, trues_te, te)
    ]:
        for musim_label, mask_fn in [
            ("Hujan",   lambda d: d["is_wet_season"].values == 1),
            ("Kemarau", lambda d: d["is_wet_season"].values == 0),
        ]:
            mask = mask_fn(df_s)
            n_total = mask.sum()
            if n_total < 20:
                continue
            print(f"\n  {split.upper()} — {musim_label} (n={n_total})")
            print(f"  {'Horizon':>8} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8}")
            r2_list = []
            for h in range(1, HORIZON + 1):
                yt = trues[h][mask]
                yp = preds[h][mask]
                m  = metrics(yt, yp)
                r2_list.append(m["r2"])
                print(f"  h+{h} ({h:2d}h)   | {m['r2']:8.4f} | {m['rmse']:8.1f} | {m['mae']:8.1f}")
            print(f"  {'MEAN':>8}   | {np.nanmean(r2_list):8.4f}")
            results.append(dict(stage=stage, split=split, musim=musim_label,
                                r2_mean=np.nanmean(r2_list),
                                r2_per_h=r2_list))
    return results


def feature_importance_by_season(stage, tr, va, te):
    """Top fitur penting berbeda antara Hujan vs Kemarau (h+1 dan h+3)."""
    print(f"\n  [TOP FITUR PER MUSIM — Stage {stage}, h+1 & h+3]")

    req = STAGE_FEATS[stage]
    fc  = [c for c in req if c in tr.columns]

    for musim_label, mask_col_val in [("Hujan", 1.0), ("Kemarau", 0.0)]:
        for h in [1, 3]:
            col = f"ghi_h{h}"
            # Train pada data musim tsb saja
            tr_m = tr[tr["is_wet_season"] == mask_col_val]
            va_m = va[va["is_wet_season"] == mask_col_val]
            if len(tr_m) < 100 or len(va_m) < 20:
                continue

            Xtr_m = tr_m[fc].values.astype(float)
            Xva_m = va_m[fc].values.astype(float)
            ytr_m = tr_m[col].values.astype(float)
            yva_m = va_m[col].values.astype(float)

            mtr_m = ~np.isnan(ytr_m)
            mva_m = ~np.isnan(yva_m)

            # Lebih sedikit pohon untuk speed
            p = LGBM_PARAMS.copy()
            p["n_estimators"] = 1000
            dtr = lgb.Dataset(Xtr_m[mtr_m], label=ytr_m[mtr_m], feature_name=fc)
            dva = lgb.Dataset(Xva_m[mva_m], label=yva_m[mva_m], reference=dtr)
            model = lgb.train(p, dtr, valid_sets=[dva], callbacks=[
                lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])

            imp = pd.DataFrame({"feature": fc,
                                 "gain": model.feature_importance("gain")})
            imp["pct"] = imp["gain"] / imp["gain"].sum() * 100
            imp = imp.sort_values("gain", ascending=False)
            imp.to_csv(OUTPUT_DIR / f"ms_s{stage}_h{h}_{musim_label.lower()}_imp.csv", index=False)

            top5 = imp.head(5)
            pva_m = model.predict(Xva_m)
            r2 = metrics(yva_m, pva_m)["r2"]
            print(f"\n  {musim_label}, h+{h} (n_train={mtr_m.sum()}, val R²={r2:.4f}):")
            for _, row in top5.iterrows():
                print(f"    {row['feature']:35s} {row['pct']:.2f}%")


def season_specific_models(tr, va, te, records_season):
    """Train model terpisah per musim (S3 features), bandingkan vs global."""
    print(f"\n{'='*62}")
    print(f"  SEASON-SPECIFIC MODELS (Stage 3, h+1..h+6)")
    print(f"{'='*62}")

    req = STAGE_FEATS[3]  # gunakan S3 features (tanpa is_wet_season)
    fc  = [c for c in req if c in tr.columns]

    print(f"\n  {'Musim':12} | {'Horizon':>8} | {'Val R²':>8} | {'Test R²':>8} | Δ vs Global")

    # Load global S3 results
    global_path = OUTPUT_DIR / "metrics_multistep.csv"

    for musim_label, mask_val in [("Hujan", 1.0), ("Kemarau", 0.0)]:
        tr_m = tr[tr["is_wet_season"] == mask_val]
        va_m = va[va["is_wet_season"] == mask_val]
        te_m = te[te["is_wet_season"] == mask_val]

        if len(tr_m) < 100:
            print(f"  {musim_label}: data terlalu sedikit (n={len(tr_m)})")
            continue

        Xtr = tr_m[fc].values.astype(float)
        Xva = va_m[fc].values.astype(float)
        Xte = te_m[fc].values.astype(float)

        for h in range(1, HORIZON + 1):
            col = f"ghi_h{h}"
            ytr = tr_m[col].values.astype(float)
            yva = va_m[col].values.astype(float)
            yte = te_m[col].values.astype(float)

            mtr_ = ~np.isnan(ytr); mva_ = ~np.isnan(yva)
            dtr  = lgb.Dataset(Xtr[mtr_], label=ytr[mtr_], feature_name=fc)
            dva  = lgb.Dataset(Xva[mva_], label=yva[mva_], reference=dtr)

            model = lgb.train(LGBM_PARAMS, dtr, valid_sets=[dva], callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(-1)])

            pva = model.predict(Xva)
            pte = model.predict(Xte)
            mva = metrics(yva, pva)
            mte = metrics(yte, pte)

            print(f"  {musim_label:12} | h+{h} ({h:2d}h) | {mva['r2']:8.4f} | {mte['r2']:8.4f}")

            records_season.append(dict(
                musim=musim_label, horizon=h,
                val_r2=mva["r2"], test_r2=mte["r2"]))


def summary(records):
    df = pd.DataFrame(records)
    lines = []
    lines.append("\n" + "=" * 72)
    lines.append("HASIL FINAL — GHI Multi-Horizon (6 jam) + Analisis Musim")
    lines.append("Dataset: perjam, lag 18 jam, horizon h+1..h+6")
    lines.append("=" * 72)

    for split, yr in [("val", "2024"), ("test", "2025")]:
        lines.append(f"\n[{split.upper()} {yr}] — Mean R² per Stage")
        header = f"  {'Stage':25s} | " + " | ".join(f"h+{h}({h}h)" for h in range(1, 7)) + " | MEAN"
        lines.append(header)
        for s in sorted(df.stage.unique()):
            sub = df[(df.stage == s) & (df.split == split)].sort_values("horizon")
            lbl = sub.iloc[0]["label"] if len(sub) else ""
            r2s = [f"{row.r2:.4f}" for _, row in sub.iterrows()]
            mean = sub.r2.mean()
            lines.append(f"  {lbl:25s} | " + " | ".join(r2s) + f" | {mean:.4f}")

    lines.append("\n[KONSISTENSI val vs test — Stage 3]")
    for h in range(1, HORIZON + 1):
        va = df[(df.stage==3)&(df.split=="val")&(df.horizon==h)]["r2"].values
        te = df[(df.stage==3)&(df.split=="test")&(df.horizon==h)]["r2"].values
        if len(va) and len(te):
            diff = te[0] - va[0]
            ok = "✓" if abs(diff) < 0.06 else "⚠"
            lines.append(f"  h+{h}: val={va[0]:.4f} test={te[0]:.4f} Δ={diff:+.4f} {ok}")

    lines.append("\n[vs MODEL 1 JAM (hasil sebelumnya)]")
    lines.append("  1-jam LightGBM S2: val=0.7575 test=0.7844")
    for sp in ["val", "test"]:
        sub = df[(df.stage==3)&(df.split==sp)]
        if not sub.empty:
            r2h1 = sub[sub.horizon==1]["r2"].values[0] if len(sub[sub.horizon==1]) else np.nan
            r2mean = sub["r2"].mean()
            lines.append(f"  Multi-step S3 {sp}: h+1 R²={r2h1:.4f} | mean R²={r2mean:.4f}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    print("=" * 62)
    print("TRAIN MULTI-STEP 6H — GHI Jambi + Analisis Musim")
    print("=" * 62)

    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")
    print(f"  Train: {len(tr):,}  Val: {len(va):,}  Test: {len(te):,}")

    print(f"\n  Distribusi musim:")
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        hw = (d["musim"] == "Hujan").sum()
        km = (d["musim"] == "Kemarau").sum()
        print(f"    {name:5s}: Hujan={hw:4d} ({hw/len(d):.0%})  Kemarau={km:4d} ({km/len(d):.0%})")

    # Baseline
    persistence_baseline(va, te)

    # Training per stage
    records_global  = []
    pred_store = {}

    for s in [1, 2, 3, 4]:
        pva, pte, tva, tte = train_stage(s, tr, va, te, records_global)
        pred_store[s] = (pva, pte, tva, tte)

    # Analisis musim (Stage 3 dan 4)
    seasonal_records = []
    for s in [3, 4]:
        pva, pte, tva, tte = pred_store[s]
        seas_res = seasonal_analysis(s, pva, pte, tva, tte, va, te)
        seasonal_records.extend(seas_res)

    # Feature importance per musim (Stage 3)
    feature_importance_by_season(3, tr, va, te)

    # Season-specific models
    season_spec_records = []
    season_specific_models(tr, va, te, season_spec_records)

    # Simpan metrics
    df_global = pd.DataFrame(records_global)
    df_global.to_csv(OUTPUT_DIR / "metrics_multistep.csv", index=False, encoding="utf-8")

    if season_spec_records:
        pd.DataFrame(season_spec_records).to_csv(
            OUTPUT_DIR / "metrics_season_specific.csv", index=False, encoding="utf-8")

    # Summary
    result = summary(records_global)
    print(result)
    with open(OUTPUT_DIR / "metrics_multistep_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nModels → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
