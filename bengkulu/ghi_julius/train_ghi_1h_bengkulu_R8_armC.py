#!/usr/bin/env python3
"""
R8 Arm C — Bengkulu: Feature Pruning to Close Point-Target Gap

Problem:
  §3.2 lean F1 (50 features, no AWS/SYNOP) achieves R2=0.789 on test-2025.
  v10 full-featured (43 features, includes SYNOP/AWS/radiation) = R2=0.821.
  Gap = -0.032. Root cause: F1 lacks radiation components + CLP extended.

This script:
  1. Builds F_super (~85 features): F1 + radiation (DHI/DNI/reflected/nett_rad)
     + CLP extended (CTH lags, height-temp interaction, delta-180m)
     + AWS-pressure + derived interactions
     — deliberately EXCLUDES SYNOP for cross-location compatibility
  2. Trains CatBoost with F_super -> "no-SYNOP ceiling"
  3. Ranks features by CatBoost PredictionValuesChange importance
  4. Sweeps top-K (K = 10, 15, 20, 25, 30, 35, 40, 50, 65, all)
  5. Reports: minimal K where R2 >= ceiling - 0.001 (epsilon=0.001)

Key question: Can radiation components + CLP extended close the -0.032 gap
without SYNOP, maintaining cross-location compatibility?

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_bengkulu_R8_armC.py
"""

import os
from pathlib import Path
import warnings

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_NAME       = "bengkulu"
ATTACH_ALIAS  = "bengkulu_db"
LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")
OUTPUT_DIR    = Path("outputs_R8_bengkulu")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -3.8607
STATION_LON_DEG  = 102.3381
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

TARGET_POINT = "ghi_point_t60"
DELTA_POINT  = "delta_point"

# K values for top-K sweep
K_SWEEP = [10, 15, 20, 25, 30, 35, 40, 50, 65]

# ---------------------------------------------------------------------------
# F1 baseline (50 features — §3.2 lean recipe, same as R1)
# ---------------------------------------------------------------------------
F1_FEATURES = [
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m",
    "accel_ghi_20m",
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
    "clp_cot",
    "clp_cot_lag_10m", "clp_cot_lag_20m",
    "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m",
    "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "month_sin", "month_cos",
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
assert len(F1_FEATURES) == 50

# ---------------------------------------------------------------------------
# F_super: F1 + extended features (~85 total)
# Groups added (all from view or derived — no SYNOP):
#   R: radiation components (DHI, DNI, reflected, nett_rad)
#   D: derived radiation fractions + clearsky reference
#   G: GHI extended (50m lag, 60m range, 180m range, ramp ratio, 60m accel)
#   C: CLP extended (CTH lags, roll, delta-180m, height-temp interaction)
#   P: AWS pressure only (pressure is physical, not surface-met-biased)
#   W: AWS aggregates (wind/RH/temp roll — kept for importance test)
#   A: AWS instantaneous (to confirm low importance vs Arm A finding)
# ---------------------------------------------------------------------------
F_EXTRA_RADIATION = [
    "dhi_now", "dni_now", "reflected_now", "nett_rad_now",
    "dhi_roll_180m_mean", "dni_roll_180m_mean",
]

F_EXTRA_DERIVED = [
    "solar_elev_sin_clip",    # max(sin(elev), 0.02)
    "clear_sky_ghi_now",      # 1100 * solar_elev_sin_clip
    "dhi_fraction",           # dhi_now / max(ghi_now, 20)
    "dni_fraction",           # dni_now / max(ghi_now, 20)
]

F_EXTRA_GHI = [
    "ghi_lag_50m",            # LAG(ghi_now, 5) via SQL
    "ghi_roll_60m_max",       # from view
    "ghi_roll_60m_min",       # from view
    "ghi_roll_180m_max",      # from view
    "ghi_roll_180m_min",      # from view
    "ghi_roll_180m_range",    # derived: max - min
    "ghi_roll_60m_range",     # derived: max - min
    "ghi_ramp_ratio_60m",     # derived: delta_60m / |lag_60m|
    "accel_ghi_60m",          # derived: 60m 2nd difference
    "accel_kt_60m",           # derived: 60m kt 2nd difference
]

F_EXTRA_CLP = [
    "clp_cth_lag_60m",        # from view
    "clp_cth_roll_180m_mean", # from view
    "clp_cth_delta_180m",     # derived: cth - cth_roll_180m
    "cloud_height_temp_interaction",  # derived: cth * ctt
]

F_EXTRA_AWS = [
    "aws_pressure_hpa",       # surface pressure (semi-physical)
    "aws_pressure_lag_60m",   # pressure trend
    "aws_ws_roll_180m_mean",  # wind history (3h)
    "aws_rh_roll_180m_mean",  # humidity history (3h)
    "aws_temp_roll_180m_mean",# temperature history (3h)
    "aws_temp_c",             # current temp
    "aws_rh_pct",             # current humidity
    "aws_ws_avg",             # current wind
    "aws_rain_mm",            # rainfall (near-zero importance expected)
]

F_EXTRA_OTHER = [
    "daylight_flag",          # from view
]

F_SUPER = (F1_FEATURES + F_EXTRA_RADIATION + F_EXTRA_DERIVED
           + F_EXTRA_GHI + F_EXTRA_CLP + F_EXTRA_AWS + F_EXTRA_OTHER)
print(f"F_super size: {len(F_SUPER)} features")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN.")
    os.environ["motherduck_token"] = token


def connect_data():
    con = duckdb.connect(database=":memory:")
    if LOCAL_DB_PATH.exists():
        con.execute(
            "ATTACH '" + LOCAL_DB_PATH.as_posix()
            + "' AS " + ATTACH_ALIAS + " (READ_ONLY)"
        )
        print("Data source: LOCAL")
    else:
        require_token()
        con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
        print("Data source: MOTHERDUCK")
    return con


# ---------------------------------------------------------------------------
# SQL — same as R1 + ghi_lag_50m
# ---------------------------------------------------------------------------
def build_sql():
    return """
    WITH with_kt AS (
        SELECT
            *,
            ghi_now / GREATEST(
                1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0
            ) AS kt_point
        FROM bengkulu_db.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
    ), with_windows AS (
        SELECT
            *,
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
            LAG(ghi_now, 5) OVER (ORDER BY ts_wib) AS ghi_lag_50m,
            LAG(kt_point, 1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point, 2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(kt_point, 3) OVER (ORDER BY ts_wib) AS kt_lag_30m,
            LAG(kt_point, 6) OVER (ORDER BY ts_wib) AS kt_lag_60m,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
                AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                AS kt_roll30m_std,
            LEAD(ghi_now, 6) OVER (ORDER BY ts_wib) AS ghi_lead_60m_pt
        FROM with_kt
    )
    SELECT *
    FROM with_windows
    WHERE is_model_ready = 1
      AND has_continuous_3h_history = 1
      AND ghi_now BETWEEN 0 AND 1400
    ORDER BY ts_wib
    """


# ---------------------------------------------------------------------------
# Astronomical helpers
# ---------------------------------------------------------------------------
def solar_elevation_deg(timestamps, lat=STATION_LAT_DEG, lon=STATION_LON_DEG,
                        meridian=WIB_MERIDIAN_DEG):
    idx = pd.DatetimeIndex(timestamps)
    doy = idx.dayofyear.values.astype(float)
    h   = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    solar_t = h + 4.0 * (lon - meridian) / 60.0
    ha = (solar_t - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def clearsky_simple(elev_deg):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(elev_deg)), 0.0)


# ---------------------------------------------------------------------------
# Feature engineering (F1 + all extras)
# ---------------------------------------------------------------------------
def add_features(df):
    out = df.copy()
    ts = pd.DatetimeIndex(out[TIME_COL])

    # F1 derived
    cs_now = clearsky_simple(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    out["accel_ghi_20m"]      = out["ghi_now"]  - 2.0 * out["ghi_lag_10m"]      + out["ghi_lag_20m"]
    out["accel_kt_20m"]       = out["kt_now"]   - 2.0 * out["kt_lag_10m"]       + out["kt_lag_20m"]
    out["accel_clp_cot_20m"]  = out["clp_cot"]  - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy  / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy  / 365.25)

    ts_t60 = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60 = solar_elevation_deg(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"]   = clearsky_simple(elev_t60)

    cs_steps = [clearsky_simple(solar_elevation_deg(out[TIME_COL] + pd.Timedelta(minutes=s*10)))
                for s in range(1, 7)]
    out["ghi_cs_avg_t10_t60"] = np.column_stack(cs_steps).mean(axis=1)
    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_avg_t10_t60"]

    # Extra derived features
    elev_sin_now = np.maximum(np.sin(np.deg2rad(out["solar_elev_deg"].values.astype(float))), 0.02)
    out["solar_elev_sin_clip"] = elev_sin_now
    out["clear_sky_ghi_now"]   = 1100.0 * elev_sin_now
    out["dhi_fraction"]        = out["dhi_now"]  / np.maximum(out["ghi_now"], 20.0)
    out["dni_fraction"]        = out["dni_now"]  / np.maximum(out["ghi_now"], 20.0)

    # GHI extended
    out["ghi_roll_180m_range"] = out["ghi_roll_180m_max"] - out["ghi_roll_180m_min"]
    out["ghi_roll_60m_range"]  = out["ghi_roll_60m_max"]  - out["ghi_roll_60m_min"]
    out["ghi_ramp_ratio_60m"]  = out["ghi_delta_60m"] / np.maximum(out["ghi_lag_60m"].abs(), 20.0)
    out["accel_ghi_60m"]       = out["ghi_now"] - 2.0 * out["ghi_lag_30m"] + out["ghi_lag_60m"]
    out["accel_kt_60m"]        = out["kt_now"]  - 2.0 * out["kt_lag_30m"]  + out["kt_lag_60m"]

    # CLP extended
    out["clp_cth_delta_180m"]         = out["clp_cth_m"] - out["clp_cth_roll_180m_mean"]
    out["cloud_height_temp_interaction"] = out["clp_cth_m"] * out["clp_ctt_k"]

    # Targets
    out[TARGET_POINT] = out["target_ghi_1h_ahead"].copy()
    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))
    out[DELTA_POINT]   = out[TARGET_POINT] - out["ghi_now"]

    return out


# ---------------------------------------------------------------------------
# Split masks
# ---------------------------------------------------------------------------
def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


# ---------------------------------------------------------------------------
# CatBoost helpers
# ---------------------------------------------------------------------------
def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )


def fit_cb(m, x_tr, y_tr, x_es, y_es):
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_es.astype(float).values, y_es.astype(float).values),
          early_stopping_rounds=150)
    return m


def eval_metrics(y_true, y_pred, sp):
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp_rmse = float(np.sqrt(mean_squared_error(y_true, sp)))
    return {
        "r2":          round(float(r2_score(y_true, y_pred)), 4),
        "mae":         round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse":        round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / sp_rmse, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    con = connect_data()
    print("Loading data...")
    df = con.execute(build_sql()).fetchdf()
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    # Check which F_super columns are actually available
    avail = [f for f in F_SUPER if f in df.columns]
    missing = [f for f in F_SUPER if f not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} features missing from view: {missing}")
    print(f"F_super available: {len(avail)} / {len(F_SUPER)}")

    # Filter: valid point target + sun > 5° at t+60
    df_use = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    print(f"Rows: {len(df_use):,}")

    tr_m, va_m, te_m = split_masks(df_use)
    print(f"train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

    x_tr = df_use.loc[tr_m, avail]
    x_va = df_use.loc[va_m, avail]
    x_te = df_use.loc[te_m, avail]
    y_tr = df_use.loc[tr_m, TARGET_POINT]
    y_va = df_use.loc[va_m, TARGET_POINT]
    y_te = df_use.loc[te_m, TARGET_POINT]
    sp_te = np.clip(df_use.loc[te_m, "smart_persist"].values, PRED_MIN, PRED_MAX)
    ghi_now_te = df_use.loc[te_m, "ghi_now"].values

    # -------------------------------------------------------------------
    # Step 1: Train on full F_super → ceiling
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"STEP 1: CatBoost F_super ({len(avail)} features) — ceiling run")
    print(f"{'='*65}")

    cb_super = catboost_model()
    fit_cb(cb_super, x_tr, y_tr, x_va, y_va)
    best_it_super = cb_super.get_best_iteration()

    pred_super = np.clip(cb_super.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
    m_super = eval_metrics(y_te, pred_super, sp_te)
    print(f"  F_super ceiling: R2={m_super['r2']:.4f}  MAE={m_super['mae']:.1f}  "
          f"iter={best_it_super}  skill={m_super['skill_vs_sp']:.4f}")
    print(f"  vs F1 baseline (R1): R2=0.7920  gap_from_ceiling={(m_super['r2']-0.7920):+.4f}")
    print(f"  vs v10 full (0.8212): gap={(m_super['r2']-0.8212):+.4f}")

    # Feature importance ranking
    feat_imp = pd.DataFrame({
        "feature": avail,
        "importance": cb_super.get_feature_importance(),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_imp["rank"] = feat_imp.index + 1
    feat_imp.to_csv(OUTPUT_DIR / "arm_C_feature_importance.csv", index=False)

    print(f"\n  Top 20 features:")
    for _, row in feat_imp.head(20).iterrows():
        in_f1 = "F1" if row["feature"] in F1_FEATURES else "NEW"
        print(f"    {row['rank']:2d}. [{in_f1}] {row['feature']:<35s}  {row['importance']:7.2f}")

    # -------------------------------------------------------------------
    # Step 2: F1-only baseline (self-consistency check vs R1)
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("STEP 2: F1 baseline (self-consistency vs R1)")
    print(f"{'='*65}")

    f1_avail = [f for f in F1_FEATURES if f in df_use.columns]
    cb_f1 = catboost_model()
    fit_cb(cb_f1,
           df_use.loc[tr_m, f1_avail], y_tr,
           df_use.loc[va_m, f1_avail], y_va)
    pred_f1 = np.clip(cb_f1.predict(df_use.loc[te_m, f1_avail].astype(float).values),
                      PRED_MIN, PRED_MAX)
    m_f1 = eval_metrics(y_te, pred_f1, sp_te)
    print(f"  F1 (50 feat): R2={m_f1['r2']:.4f}  MAE={m_f1['mae']:.1f}  "
          f"[R1 reference: 0.7920]")

    CEILING = m_super["r2"]
    EPSILON  = 0.001

    # -------------------------------------------------------------------
    # Step 3: Top-K sweep
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"STEP 3: Top-K sweep  (target: R2 >= {CEILING:.4f} - {EPSILON} = {CEILING-EPSILON:.4f})")
    print(f"{'='*65}")

    sweep_rows = []
    ranked_features = feat_imp["feature"].tolist()

    for k in K_SWEEP + [len(avail)]:
        feats_k = ranked_features[:k]
        avail_k = [f for f in feats_k if f in df_use.columns]
        if len(avail_k) < 5:
            continue

        cb_k = catboost_model()
        fit_cb(cb_k,
               df_use.loc[tr_m, avail_k], y_tr,
               df_use.loc[va_m, avail_k], y_va)
        best_it_k = cb_k.get_best_iteration()

        pred_k = np.clip(cb_k.predict(df_use.loc[te_m, avail_k].astype(float).values),
                         PRED_MIN, PRED_MAX)
        m_k = eval_metrics(y_te, pred_k, sp_te)
        delta_vs_ceiling = m_k["r2"] - CEILING
        delta_vs_f1      = m_k["r2"] - m_f1["r2"]
        delta_vs_v10     = m_k["r2"] - 0.8212
        meets_threshold  = m_k["r2"] >= (CEILING - EPSILON)

        flag = "<-- OPTIMAL" if meets_threshold else ""
        print(f"  top-{k:3d}: R2={m_k['r2']:.4f}  MAE={m_k['mae']:.1f}  "
              f"iter={best_it_k:4d}  vs_ceiling={delta_vs_ceiling:+.4f}  "
              f"vs_F1={delta_vs_f1:+.4f}  vs_v10={delta_vs_v10:+.4f}  {flag}")

        sweep_rows.append({
            "k": k, "r2": m_k["r2"], "mae": m_k["mae"], "rmse": m_k["rmse"],
            "skill_vs_sp": m_k["skill_vs_sp"],
            "best_iter": best_it_k,
            "delta_vs_ceiling": round(delta_vs_ceiling, 4),
            "delta_vs_f1": round(delta_vs_f1, 4),
            "delta_vs_v10": round(delta_vs_v10, 4),
            "meets_threshold": meets_threshold,
            "top_features": ",".join(avail_k),
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUTPUT_DIR / "arm_C_topk_sweep.csv", index=False)

    # -------------------------------------------------------------------
    # Step 4: Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("SUMMARY — R8 Arm C Bengkulu (point target)")
    print(f"{'='*65}")
    print(f"  F1 baseline (lean-50) R2  : {m_f1['r2']:.4f}  (R1 reference: 0.7920)")
    print(f"  F_super ceiling R2        : {CEILING:.4f}  ({len(avail)} features, no SYNOP)")
    print(f"  v10 full (with SYNOP/AWS) : 0.8212")
    gap_closed = CEILING - m_f1["r2"]
    gap_remaining = 0.8212 - CEILING
    print(f"  Gap F1 -> F_super         : +{gap_closed:.4f}")
    print(f"  Remaining gap to v10      : {gap_remaining:+.4f}")

    # Find minimal K meeting threshold
    optimal_rows = sweep_df[sweep_df["meets_threshold"]]
    if not optimal_rows.empty:
        opt = optimal_rows.iloc[0]
        opt_feats = opt["top_features"].split(",")
        # Classify each optimal feature
        new_in_opt = [f for f in opt_feats if f not in F1_FEATURES]
        in_f1_opt  = [f for f in opt_feats if f in F1_FEATURES]
        print(f"\n  MINIMAL OPTIMAL SET: top-{int(opt['k'])} features")
        print(f"    F1 features kept : {len(in_f1_opt)} / {len(F1_FEATURES)}")
        print(f"    New features added: {len(new_in_opt)}")
        print(f"    New features: {new_in_opt}")
        print(f"    R2 = {opt['r2']:.4f}  vs F1 +{opt['delta_vs_f1']:.4f}  vs v10 {opt['delta_vs_v10']:+.4f}")

        # Save optimal feature list
        pd.Series(opt_feats, name="feature").to_csv(
            OUTPUT_DIR / f"arm_C_optimal_top{int(opt['k'])}.csv", index=False)
    else:
        print("\n  No K met threshold — F_super ceiling itself is below EPSILON threshold")

    # Ceiling comparison to v10 and R1
    print(f"\n  === Interpretation ===")
    if CEILING >= 0.8212 - EPSILON:
        print(f"  Gap vs v10 CLOSED: F_super ({CEILING:.4f}) >= v10 (0.8212) - eps")
        print(f"  -> Extended features (no SYNOP) suffice to match v10 accuracy")
    elif CEILING >= 0.8212 - 0.005:
        print(f"  Gap vs v10 NEARLY CLOSED: residual {0.8212 - CEILING:.4f} (< 0.005)")
        print(f"  -> SYNOP contribution is only ~{0.8212-CEILING:.4f} R2 — practically negligible")
    else:
        print(f"  Gap vs v10 partially closed: F_super {CEILING:.4f}, v10 0.8212")
        print(f"  -> Remaining gap {0.8212-CEILING:.4f} attributable to SYNOP features")

    print(f"\n  Outputs -> {OUTPUT_DIR}/")
    print(f"    arm_C_feature_importance.csv  ({len(avail)} features ranked)")
    print(f"    arm_C_topk_sweep.csv          ({len(sweep_df)} K values)")


if __name__ == "__main__":
    main()
