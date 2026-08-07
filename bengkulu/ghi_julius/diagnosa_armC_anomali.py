#!/usr/bin/env python3
"""
Diagnostik: mengapa F_super (84 fitur) R2=0.763 < F1 (50 fitur) R2=0.792

Tiga hipotesis:
  H1. NaN rate AWS/radiasi berbeda train vs test (sensor outage 2025)
  H2. Collinearity collapse — ghi_now diranking rendah (#80/84) sehingga
      top-K sweep mengganti GHI-history dengan fitur yang kurang stabil
  H3. Feature leakage atau bug konstruksi fitur turunan (ratio instability)

Script ini:
  1. Cek NaN rate per grup fitur — train vs test
  2. Cek importance ghi_now di F_super vs F1
  3. Ablasi fokus: F1 + radiasi saja (tanpa AWS) → apakah turun?
  4. Ablasi: F1 + AWS saja (tanpa radiasi) → berapa R2? (harusnya ~0.792 per Arm A)
  5. Konfirmasi: apa F1 top-20 by F_super importance → berapa R2?
"""

import os
from pathlib import Path
import warnings
import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")
ATTACH_ALIAS  = "bengkulu_db"
OUTPUT_DIR    = Path("outputs_R8_bengkulu")
OUTPUT_DIR.mkdir(exist_ok=True)
TIME_COL = "ts_wib"
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
STATION_LAT_DEG, STATION_LON_DEG, WIB_MERIDIAN_DEG = -3.8607, 102.3381, 105.0
PRED_MIN, PRED_MAX = 0.0, 1400.0

F1_FEATURES = [
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m", "accel_ghi_20m",
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean", "accel_kt_20m",
    "clp_cot", "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean", "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]

F_RADIATION = ["dhi_now", "dni_now", "reflected_now", "nett_rad_now",
               "dhi_roll_180m_mean", "dni_roll_180m_mean"]
F_RAD_DERIVED = ["solar_elev_sin_clip", "clear_sky_ghi_now", "dhi_fraction", "dni_fraction"]
F_AWS = ["aws_pressure_hpa", "aws_pressure_lag_60m", "aws_ws_roll_180m_mean",
         "aws_rh_roll_180m_mean", "aws_temp_roll_180m_mean",
         "aws_temp_c", "aws_rh_pct", "aws_ws_avg", "aws_rain_mm"]

def connect():
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{LOCAL_DB_PATH.as_posix()}' AS {ATTACH_ALIAS} (READ_ONLY)")
    return con

def build_sql():
    return """
    WITH with_kt AS (
        SELECT *, ghi_now / GREATEST(1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0) AS kt_point
        FROM bengkulu_db.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
    ), with_windows AS (
        SELECT *,
            LAG(clp_cot,1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot,2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot,3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            LAG(ghi_now,2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
            LAG(ghi_now,5) OVER (ORDER BY ts_wib) AS ghi_lag_50m,
            LAG(kt_point,1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point,2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(kt_point,3) OVER (ORDER BY ts_wib) AS kt_lag_30m,
            LAG(kt_point,6) OVER (ORDER BY ts_wib) AS kt_lag_60m,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std
        FROM with_kt
    )
    SELECT * FROM with_windows
    WHERE is_model_ready=1 AND has_continuous_3h_history=1 AND ghi_now BETWEEN 0 AND 1400
    ORDER BY ts_wib
    """

def clearsky(e): return 1100.0 * np.maximum(np.sin(np.deg2rad(e)), 0.0)

def solar_elev(ts):
    idx = pd.DatetimeIndex(ts)
    doy = idx.dayofyear.values.astype(float)
    h   = idx.hour.values.astype(float) + idx.minute.values.astype(float)/60.0
    decl = 23.45 * np.sin(np.deg2rad(360*(284+doy)/365))
    ha   = (h + 4*(STATION_LON_DEG - WIB_MERIDIAN_DEG)/60 - 12)*15
    sin_e = (np.sin(np.deg2rad(STATION_LAT_DEG))*np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(STATION_LAT_DEG))*np.cos(np.deg2rad(decl))*np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e,-1,1)))

def add_features(df):
    out = df.copy()
    cs_now = clearsky(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)
    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["accel_ghi_20m"]      = out["ghi_now"]  - 2*out["ghi_lag_10m"]      + out["ghi_lag_20m"]
    out["accel_kt_20m"]       = out["kt_now"]   - 2*out["kt_lag_10m"]       + out["kt_lag_20m"]
    out["accel_clp_cot_20m"]  = out["clp_cot"]  - 2*out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    ts = pd.DatetimeIndex(out[TIME_COL])
    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    out["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    ts_t60 = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60 = solar_elev(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"]   = clearsky(elev_t60)
    cs_steps = [clearsky(solar_elev(out[TIME_COL]+pd.Timedelta(minutes=s*10))) for s in range(1,7)]
    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * np.column_stack(cs_steps).mean(axis=1)
    # F_super extras
    es = np.maximum(np.sin(np.deg2rad(out["solar_elev_deg"].values.astype(float))), 0.02)
    out["solar_elev_sin_clip"] = es
    out["clear_sky_ghi_now"]   = 1100.0 * es
    out["dhi_fraction"] = out["dhi_now"]  / np.maximum(out["ghi_now"], 20.0)
    out["dni_fraction"] = out["dni_now"]  / np.maximum(out["ghi_now"], 20.0)
    out["target"] = out["target_ghi_1h_ahead"].copy()
    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))
    return out

def split(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))

def cb(seed=42):
    return CatBoostRegressor(iterations=4000, learning_rate=0.02, depth=8,
                             l2_leaf_reg=3.0, loss_function="RMSE",
                             random_seed=seed, verbose=False, thread_count=-1,
                             allow_writing_files=False)

def fit_eval(feats, x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp, label):
    avail = [f for f in feats if f in x_tr.columns]
    m = cb()
    m.fit(x_tr[avail].astype(float).values, y_tr.astype(float).values,
          eval_set=(x_va[avail].astype(float).values, y_va.astype(float).values),
          early_stopping_rounds=150)
    pred = np.clip(m.predict(x_te[avail].astype(float).values), PRED_MIN, PRED_MAX)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    sp_rmse = float(np.sqrt(np.mean((y_te - y_sp)**2)))
    rmse    = float(np.sqrt(np.mean((y_te - pred)**2)))
    skill   = 1 - rmse/sp_rmse if sp_rmse > 0 else 0
    print(f"  {label:<40s}  n={len(avail):2d}  R2={r2:.4f}  MAE={mae:.1f}  iter={m.get_best_iteration()}")
    return r2, len(avail), m

def main():
    con = connect()
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)
    df_use = df[df["target"].between(0,1400) & df["sun_gt5_t60"]].copy()
    tr, va, te = split(df_use)
    print(f"train={tr.sum():,}  val={va.sum():,}  test={te.sum():,}")

    ALL_EXTRA = F_RADIATION + F_RAD_DERIVED + F_AWS
    all_cols = F1_FEATURES + ALL_EXTRA
    avail_cols = [c for c in all_cols if c in df_use.columns]

    x_tr = df_use.loc[tr, avail_cols]
    x_va = df_use.loc[va, avail_cols]
    x_te = df_use.loc[te, avail_cols]
    y_tr, y_va, y_te = df_use.loc[tr,"target"], df_use.loc[va,"target"], df_use.loc[te,"target"]
    y_sp = np.clip(df_use.loc[te,"smart_persist"].values, PRED_MIN, PRED_MAX)

    # -----------------------------------------------------------------------
    # H1: NaN rate per feature grup — train vs test
    # -----------------------------------------------------------------------
    print("\n=== H1: NaN rate per grup fitur (train vs test) ===")
    groups = {
        "F1_core_GHI":    [c for c in ["ghi_now","ghi_lag_10m","ghi_lag_30m","ghi_lag_60m",
                                         "ghi_lag_120m","ghi_lag_180m","kt_now"] if c in df_use.columns],
        "F1_CLP":         [c for c in ["clp_cot","clp_cth_m","clp_ctt_k","clp_cer"] if c in df_use.columns],
        "F1_deterministic":[c for c in ["smart_persist","ghi_cs_t60","elev_sin_t60"] if c in df_use.columns],
        "RAD_raw":        [c for c in F_RADIATION if c in df_use.columns],
        "RAD_derived":    [c for c in F_RAD_DERIVED if c in df_use.columns],
        "AWS_rolling":    [c for c in ["aws_ws_roll_180m_mean","aws_rh_roll_180m_mean",
                                        "aws_temp_roll_180m_mean","aws_pressure_lag_60m"] if c in df_use.columns],
        "AWS_instant":    [c for c in ["aws_temp_c","aws_rh_pct","aws_ws_avg","aws_rain_mm",
                                        "aws_pressure_hpa"] if c in df_use.columns],
    }
    rows = []
    for grp, cols in groups.items():
        if not cols: continue
        nan_tr = df_use.loc[tr, cols].isna().mean().mean()
        nan_te = df_use.loc[te, cols].isna().mean().mean()
        flag = " <<< SHIFT" if abs(nan_te - nan_tr) > 0.05 else ""
        print(f"  {grp:<22s}  train_NaN={nan_tr:.3f}  test_NaN={nan_te:.3f}{flag}")
        rows.append({"group": grp, "train_nan": round(nan_tr,4), "test_nan": round(nan_te,4),
                     "shift": round(nan_te-nan_tr,4)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/"diag_nan_rates.csv", index=False)

    # Also check ghi_now importance shift — key collinearity check
    print("\n  [Collinearity check: ghi_now correlation with clear_sky_ghi_now, kt_now, dhi_now in test]")
    te_df = df_use.loc[te]
    for c in ["clear_sky_ghi_now", "kt_now", "dhi_now", "dni_now"]:
        if c in te_df.columns:
            corr = te_df["ghi_now"].corr(te_df[c].astype(float))
            print(f"    corr(ghi_now, {c}) = {corr:.4f}")

    # -----------------------------------------------------------------------
    # H2: Ablation — isolate which group causes the drop
    # -----------------------------------------------------------------------
    print("\n=== H2: Ablasi grup — R2 dengan berbagai kombinasi fitur ===")
    results = []

    r2_f1, n, _ = fit_eval(F1_FEATURES, x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp,
                            "F1 baseline (50)")
    results.append({"label":"F1 baseline", "n":n, "r2":round(r2_f1,4)})

    r2_f1r, n, _ = fit_eval(F1_FEATURES + F_RADIATION,
                              x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp,
                              "F1 + RAD_raw (56)")
    results.append({"label":"F1+RAD_raw", "n":n, "r2":round(r2_f1r,4)})

    r2_f1rd, n, _ = fit_eval(F1_FEATURES + F_RADIATION + F_RAD_DERIVED,
                               x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp,
                               "F1 + RAD_raw + RAD_derived (60)")
    results.append({"label":"F1+RAD+derived", "n":n, "r2":round(r2_f1rd,4)})

    r2_f1aws, n, _ = fit_eval(F1_FEATURES + F_AWS,
                                x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp,
                                "F1 + AWS only (59)")
    results.append({"label":"F1+AWS", "n":n, "r2":round(r2_f1aws,4)})

    r2_super, n, m_super = fit_eval(avail_cols,
                                     x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp,
                                     f"F_super all ({len(avail_cols)})")
    results.append({"label":"F_super_all", "n":n, "r2":round(r2_super,4)})

    # -----------------------------------------------------------------------
    # H3: Does the sweep problem come from collinearity masking ghi_now?
    # Train F_super, get importance, check rank of ghi_now and GHI history
    # -----------------------------------------------------------------------
    print("\n=== H3: Rank ghi_now dan GHI-history dalam F_super importance ===")
    imp = pd.DataFrame({"feature": avail_cols,
                         "importance": m_super.get_feature_importance()}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1

    ghi_hist = ["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m",
                "ghi_lag_120m","ghi_lag_180m","ghi_roll_30m_mean","ghi_roll_60m_mean"]
    print(f"  {'Feature':<30s}  Rank  Importance")
    for _, row in imp[imp["feature"].isin(ghi_hist)].iterrows():
        print(f"  {row['feature']:<30s}  #{int(row['rank']):<4d}  {row['importance']:.2f}")

    # Bonus: train top-K of F1 importance (not F_super importance) — do we recover 0.792?
    print("\n  [Bonus: Arm C sweep using F1 importance order instead of F_super order]")
    m_f1 = cb()
    f1_avail = [f for f in F1_FEATURES if f in x_tr.columns]
    m_f1.fit(x_tr[f1_avail].astype(float).values, y_tr.astype(float).values,
             eval_set=(x_va[f1_avail].astype(float).values, y_va.astype(float).values),
             early_stopping_rounds=150)
    f1_imp = pd.DataFrame({"feature": f1_avail, "imp": m_f1.get_feature_importance()}) \
               .sort_values("imp", ascending=False)["feature"].tolist()

    for k in [20, 30, 40]:
        feats_k = f1_imp[:k]
        r2k, nk, _ = fit_eval(feats_k, x_tr, y_tr, x_va, y_va, x_te, y_te, y_sp,
                               f"F1 top-{k} by F1-importance")
        results.append({"label": f"F1_top{k}_by_F1imp", "n": nk, "r2": round(r2k, 4)})

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== RINGKASAN DIAGNOSTIK ===")
    df_res = pd.DataFrame(results)
    for _, row in df_res.iterrows():
        delta = row["r2"] - r2_f1
        print(f"  {row['label']:<35s}  n={row['n']:2d}  R2={row['r2']:.4f}  vs_F1={delta:+.4f}")
    df_res.to_csv(OUTPUT_DIR/"diag_ablation.csv", index=False)

    print("\n  Kesimpulan yang diharapkan:")
    print("  - Jika F1+RAD_raw turun -> radiasi mentah (DHI/DNI) adalah culprit")
    print("  - Jika F1+AWS turun     -> AWS distribution shift adalah culprit")
    print("  - Jika F1+RAD+derived turun lebih jauh -> dhi_fraction/dni_fraction ratio instabil")
    print("  - Jika NaN_AWS test >> train -> konfirmasi H1 (sensor outage 2025)")
    print(f"\n  Output: {OUTPUT_DIR}/diag_nan_rates.csv, diag_ablation.csv")

if __name__ == "__main__":
    main()
