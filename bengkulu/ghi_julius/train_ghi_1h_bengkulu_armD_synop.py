#!/usr/bin/env python3
"""
Arm D — SYNOP cloud features + F1 baseline (Bengkulu GHI 1-hour-ahead).

Investigates whether Bengkulu's complete SYNOP CL/CM/CH cloud data
contributes to its R²=0.792 advantage over other sites.

Configurations:
  CONFIG_F1   : 50 F1 features only (re-confirms baseline R²)
  CONFIG_F1D  : 50 F1 + 9 SYNOP cloud features (F1+D)

SYNOP JOIN strategy:
  - synop_bengkulu is native hourly 07:00-19:00 WIB
  - training data is 10-minute
  - JOIN on date_trunc('hour', ts_wib) = synop.ts_wib
  - NaN rows handled by SimpleImputer(median) in LGBM pipeline
  - CatBoost: .astype(float) converts pd.NA → np.nan (accepted natively)

SYNOP features (from synop_null_audit.csv fill rates in synop_bengkulu):
  Tier 1 — high fill (>50% in synop table):
    cloud_low_base_1          → syn_cloud_low_base   (99.72%) [m, CL presence]
    cloud_med_base_1          → syn_cloud_med_base   (98.60%) [m, CM presence]
    cloud_high_base_1         → syn_cloud_high_base  (93.79%) [m, CH presence]
    cloud_elevation_angle_ec_1→ syn_cloud_elev_angle (82.26%) [deg]
    cloud_low_peak_1          → syn_cloud_low_peak   (67.26%) [m]
    cloud_low_base_2          → syn_cloud_low_base2  (52.30%) [m, 2nd CL group]
  Tier 2 — partial fill (20-50%):
    cloud_layer_1_amt_oktas_ns→ syn_layer1_oktas     (35.57%) [oktas 0-8]
    cloud_layer_2_amt_oktas_ns→ syn_layer2_oktas     (23.35%) [oktas 0-8]
  Tier 3 — sparse (<25%):
    cloud_layer_3_amt_oktas_ns→ syn_layer3_oktas     ( 7.89%) [oktas 0-8]

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_bengkulu_armD_synop.py
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
OUTPUT_DIR    = Path("outputs_armD_bengkulu_synop")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -3.8607
STATION_LON_DEG  = 102.3381
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

REF_R2_LGBM   = 0.7891  # R1 reference — LGBM residual, point target (outputs_R1_bengkulu/ghi_1h_R1_results.csv)
REF_R2_CB     = 0.7920  # R1 reference — CatBoost direct, point target
REF_ROW_COUNT = 105051  # total rows after R1 filter (is_model_ready=1, continuity, ghi in [0,1400])

# Set True to re-confirm baseline within this run (takes ~10 min extra).
# Set False to skip — use REF_R2_* above and only train the SYNOP config.
RUN_BASELINE = False

# ---------------------------------------------------------------------------
# §3.2 F1 feature set (50 features, identical to R1_benchmark)
# ---------------------------------------------------------------------------
FEATURES_GHI = [
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m",
    "accel_ghi_20m",
]
FEATURES_KT = [
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
]
FEATURES_CLP = [
    "clp_cot",
    "clp_cot_lag_10m", "clp_cot_lag_20m",
    "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m",
    "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
]
FEATURES_TIME = [
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "month_sin", "month_cos",
]
FEATURES_FUTURE = [
    "ghi_cs_t60",
    "elev_sin_t60",
    "smart_persist",
    "smart_persist_avg",
]

FEATURES_F1 = (FEATURES_GHI + FEATURES_KT + FEATURES_CLP
               + FEATURES_TIME + FEATURES_FUTURE)
assert len(FEATURES_F1) == 50, f"Expected 50 F1 features, got {len(FEATURES_F1)}"

# ---------------------------------------------------------------------------
# SYNOP Arm D features
# ---------------------------------------------------------------------------
SYNOP_TIER1 = [
    "syn_cloud_low_base",    # cloud_low_base_1:          99.72%
    "syn_cloud_med_base",    # cloud_med_base_1:          98.60%
    "syn_cloud_high_base",   # cloud_high_base_1:         93.79%
    "syn_cloud_elev_angle",  # cloud_elevation_angle_ec_1:82.26%
    "syn_cloud_low_peak",    # cloud_low_peak_1:          67.26%
    "syn_cloud_low_base2",   # cloud_low_base_2:          52.30%
]
SYNOP_TIER2 = [
    "syn_layer1_oktas",      # cloud_layer_1_amt_oktas_ns:35.57%
    "syn_layer2_oktas",      # cloud_layer_2_amt_oktas_ns:23.35%
]
SYNOP_TIER3 = [
    "syn_layer3_oktas",      # cloud_layer_3_amt_oktas_ns: 7.89%
]
SYNOP_FEATURES = SYNOP_TIER1 + SYNOP_TIER2 + SYNOP_TIER3

FEATURES_F1D = FEATURES_F1 + SYNOP_FEATURES  # 59 features

TARGET_POINT = "ghi_point_t60"
TARGET_AVG   = "ghi_avg_t10_t60"
DELTA_POINT  = "delta_point"
DELTA_AVG    = "delta_avg"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN environment variable.")
    os.environ["motherduck_token"] = token


def connect_data():
    con = duckdb.connect(database=":memory:")
    if LOCAL_DB_PATH.exists():
        con.execute(
            "ATTACH '" + LOCAL_DB_PATH.as_posix()
            + "' AS " + ATTACH_ALIAS + " (READ_ONLY)"
        )
        print("Data source: LOCAL  (" + str(LOCAL_DB_PATH) + ")")
    else:
        require_token()
        con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
        print("Data source: MOTHERDUCK")
    return con


# ---------------------------------------------------------------------------
# SQL — F1 + SYNOP JOIN
# The SYNOP join uses date_trunc('hour') to match each 10-min row to the
# synop observation at the top of that hour.  Rows in hours without SYNOP
# (e.g. non-observation hours or missing obs) get NULL → imputed downstream.
# QUALIFY dedup ensures one SYNOP row per hour if duplicates exist.
# ---------------------------------------------------------------------------
def build_sql(include_synop=True):
    synop_select = ""
    synop_join   = ""
    if include_synop:
        synop_select = """
       ,s.cloud_low_base_1              AS syn_cloud_low_base
       ,s.cloud_med_base_1              AS syn_cloud_med_base
       ,s.cloud_high_base_1             AS syn_cloud_high_base
       ,s.cloud_elevation_angle_ec_1    AS syn_cloud_elev_angle
       ,s.cloud_low_peak_1              AS syn_cloud_low_peak
       ,s.cloud_low_base_2              AS syn_cloud_low_base2
       ,s.cloud_layer_1_amt_oktas_ns    AS syn_layer1_oktas
       ,s.cloud_layer_2_amt_oktas_ns    AS syn_layer2_oktas
       ,s.cloud_layer_3_amt_oktas_ns    AS syn_layer3_oktas"""
        synop_join = """
    LEFT JOIN (
        SELECT
            date_trunc('hour', ts_wib)       AS obs_hour,
            MIN(cloud_low_base_1)             AS cloud_low_base_1,
            MIN(cloud_med_base_1)             AS cloud_med_base_1,
            MIN(cloud_high_base_1)            AS cloud_high_base_1,
            MIN(cloud_elevation_angle_ec_1)   AS cloud_elevation_angle_ec_1,
            MIN(cloud_low_peak_1)             AS cloud_low_peak_1,
            MIN(cloud_low_base_2)             AS cloud_low_base_2,
            MIN(cloud_layer_1_amt_oktas_ns)   AS cloud_layer_1_amt_oktas_ns,
            MIN(cloud_layer_2_amt_oktas_ns)   AS cloud_layer_2_amt_oktas_ns,
            MIN(cloud_layer_3_amt_oktas_ns)   AS cloud_layer_3_amt_oktas_ns
        FROM bengkulu_db.bengkulu_sch.synop_bengkulu
        GROUP BY date_trunc('hour', ts_wib)
    ) s ON date_trunc('hour', w.ts_wib) = s.obs_hour"""

    return f"""
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
            LEAD(ghi_now, 1) OVER (ORDER BY ts_wib) AS ghi_lead_10m,
            LEAD(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lead_20m,
            LEAD(ghi_now, 3) OVER (ORDER BY ts_wib) AS ghi_lead_30m,
            LEAD(ghi_now, 4) OVER (ORDER BY ts_wib) AS ghi_lead_40m,
            LEAD(ghi_now, 5) OVER (ORDER BY ts_wib) AS ghi_lead_50m,
            LEAD(ghi_now, 6) OVER (ORDER BY ts_wib) AS ghi_lead_60m
        FROM with_kt
    )
    SELECT w.*{synop_select}
    FROM with_windows w{synop_join}
    WHERE w.is_model_ready = 1
      AND w.has_continuous_3h_history = 1
      AND w.ghi_now BETWEEN 0 AND 1400
    ORDER BY w.ts_wib
    """


# ---------------------------------------------------------------------------
# Astronomical helpers
# ---------------------------------------------------------------------------
def solar_elevation_deg(timestamps, lat=STATION_LAT_DEG, lon=STATION_LON_DEG,
                        meridian=WIB_MERIDIAN_DEG):
    idx  = pd.DatetimeIndex(timestamps)
    doy  = idx.dayofyear.values.astype(float)
    h    = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    ha   = (h + 4.0 * (lon - meridian) / 60.0 - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def clearsky_simple(elev_deg):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(elev_deg)), 0.0)


# ---------------------------------------------------------------------------
# Feature engineering (identical to R1_benchmark.py)
# ---------------------------------------------------------------------------
def add_features(df):
    out = df.copy()
    ts  = pd.DatetimeIndex(out[TIME_COL])

    cs_now = clearsky_simple(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    out["accel_ghi_20m"]     = out["ghi_now"] - 2.0 * out["ghi_lag_10m"]     + out["ghi_lag_20m"]
    out["accel_kt_20m"]      = out["kt_now"]  - 2.0 * out["kt_lag_10m"]      + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    ts_t60 = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60 = solar_elevation_deg(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"]   = clearsky_simple(elev_t60)

    cs_steps = []
    for step in range(1, 7):
        ts_f = out[TIME_COL] + pd.Timedelta(minutes=step * 10)
        cs_steps.append(clearsky_simple(solar_elevation_deg(ts_f)))
    out["ghi_cs_avg_t10_t60"] = np.column_stack(cs_steps).mean(axis=1)

    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_avg_t10_t60"]

    out[TARGET_POINT] = out["target_ghi_1h_ahead"].copy()

    lead_cols = ["ghi_lead_10m", "ghi_lead_20m", "ghi_lead_30m",
                 "ghi_lead_40m", "ghi_lead_50m", "ghi_lead_60m"]
    leads     = out[lead_cols]
    all_valid = (leads.notna().all(axis=1)
                 & leads.apply(lambda c: c.between(0, 1400)).all(axis=1))
    out[TARGET_AVG] = np.where(all_valid, leads.mean(axis=1), np.nan)

    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))

    out[DELTA_POINT] = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG]   = out[TARGET_AVG]   - out["ghi_now"]

    return out


# ---------------------------------------------------------------------------
# SYNOP coverage report
# ---------------------------------------------------------------------------
def report_synop_coverage(df, features=SYNOP_FEATURES):
    print("\n--- SYNOP join coverage (% non-null in full dataset) ---")
    for col in features:
        if col in df.columns:
            n_ok  = df[col].notna().sum()
            pct   = 100.0 * n_ok / len(df)
            tier  = ("T1" if col in SYNOP_TIER1
                     else "T2" if col in SYNOP_TIER2
                     else "T3")
            print(f"  [{tier}] {col:<28}: {n_ok:7,} / {len(df):,}  ({pct:5.1f}%)")
        else:
            print(f"  [??] {col:<28}: COLUMN NOT FOUND IN DATA")

    # Check per split
    ts = df[TIME_COL]
    tr = ts <  pd.Timestamp(TRAIN_END)
    va = (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END))
    te = ts >= pd.Timestamp(VALID_END)
    ref_col = "syn_cloud_low_base"
    if ref_col in df.columns:
        for mask, lbl in [(tr, "train"), (va, "val"), (te, "test2025")]:
            sub = df.loc[mask, ref_col]
            pct = 100.0 * sub.notna().sum() / max(len(sub), 1)
            print(f"  {lbl:<10}  {ref_col} coverage: {pct:.1f}%")


# ---------------------------------------------------------------------------
# Models (identical hyperparameters to R1_benchmark.py)
# ---------------------------------------------------------------------------
def lgbm_pipe(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=6000, learning_rate=0.02,
        num_leaves=39, min_child_samples=70,
        reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
        random_state=seed, n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("m",   reg),
    ])


def fit_lgbm(pipe, x_tr, y_tr, x_es, y_es):
    pipe.fit(x_tr, y_tr,
             m__eval_set=[(x_es, y_es)],
             m__eval_metric="rmse",
             m__callbacks=[lgb.early_stopping(150, verbose=False)])
    return pipe


def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )


def fit_catboost(m, x_tr, y_tr, x_es, y_es):
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_es.astype(float).values, y_es.astype(float).values),
          early_stopping_rounds=150)
    return m


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, sp_pred, model, target, config):
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp_rmse = float(np.sqrt(mean_squared_error(y_true, sp_pred)))
    skill = (1.0 - rmse / sp_rmse) if sp_rmse > 0 else float("nan")
    return {
        "config": config, "model": model, "target": target, "n": len(y_true),
        "r2": round(r2, 4), "mae": round(mae, 1),
        "rmse": round(rmse, 1), "skill_vs_sp": round(skill, 4),
    }


# ---------------------------------------------------------------------------
# Train + evaluate one feature configuration
# ---------------------------------------------------------------------------
def run_config(df, features, config_name):
    print(f"\n{'='*64}")
    print(f"CONFIG: {config_name}  ({len(features)} features)")
    print(f"{'='*64}")

    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()

    ts = df_pt[TIME_COL]
    tr_m = ts <  pd.Timestamp(TRAIN_END)
    va_m = (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END))
    te_m = ts >= pd.Timestamp(VALID_END)

    print(f"  train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

    # Confirm all features exist
    missing = [f for f in features if f not in df_pt.columns]
    if missing:
        print(f"  WARNING — missing features: {missing}")
        features = [f for f in features if f in df_pt.columns]
        print(f"  Proceeding with {len(features)} features")

    x_tr = df_pt.loc[tr_m, features]
    x_va = df_pt.loc[va_m, features]
    x_te = df_pt.loc[te_m, features]
    y_tr = df_pt.loc[tr_m, TARGET_POINT]
    y_va = df_pt.loc[va_m, TARGET_POINT]
    y_te = df_pt.loc[te_m, TARGET_POINT]
    yd_tr = df_pt.loc[tr_m, DELTA_POINT]
    yd_va = df_pt.loc[va_m, DELTA_POINT]

    ghi_now_te = df_pt.loc[te_m, "ghi_now"].values
    sp_te = np.clip(df_pt.loc[te_m, "smart_persist"].values, PRED_MIN, PRED_MAX)

    results = []

    # Smart-persistence (same for all configs)
    r = compute_metrics(y_te, sp_te, sp_te, "smart_persistence", "point_t60", config_name)
    results.append(r)
    print(f"  smart_persistence  R2={r['r2']:.4f}  MAE={r['mae']:.1f}")

    # PRIMARY: LightGBM residual
    print("  Training LightGBM residual...")
    lgbm = lgbm_pipe()
    fit_lgbm(lgbm, x_tr, yd_tr, x_va, yd_va)
    best_it = lgbm.named_steps["m"].best_iteration_
    lgbm_pred = np.clip(ghi_now_te + lgbm.predict(x_te), PRED_MIN, PRED_MAX)
    r = compute_metrics(y_te, lgbm_pred, sp_te, "lgbm_residual", "point_t60", config_name)
    r["best_iter"] = best_it
    results.append(r)
    dr = r["r2"] - REF_R2_LGBM
    print(f"  lgbm_residual  iter={best_it:4d}  R2={r['r2']:.4f}  "
          f"dR2={dr:+.4f} vs ref {REF_R2_LGBM}  MAE={r['mae']:.1f}  skill={r['skill_vs_sp']:.4f}")

    # SENSITIVITY: CatBoost direct
    print("  Training CatBoost direct...")
    cb_m = catboost_model()
    fit_catboost(cb_m, x_tr, y_tr, x_va, y_va)
    best_it_cb = cb_m.get_best_iteration()
    cb_pred = np.clip(cb_m.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
    r = compute_metrics(y_te, cb_pred, sp_te, "catboost_direct", "point_t60", config_name)
    r["best_iter"] = best_it_cb
    results.append(r)
    dr = r["r2"] - REF_R2_CB
    print(f"  catboost_direct  iter={best_it_cb:4d}  R2={r['r2']:.4f}  "
          f"dR2={dr:+.4f} vs ref {REF_R2_CB}  MAE={r['mae']:.1f}  skill={r['skill_vs_sp']:.4f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    con = connect_data()

    # -----------------------------------------------------------------------
    # CONFIG A: F1 baseline (50 features, no SYNOP)
    # Optionally re-confirms R²=0.792 CB / 0.789 LGBM (RUN_BASELINE=True).
    # Default: skip and use stored reference values.
    # -----------------------------------------------------------------------
    if RUN_BASELINE:
        print("\nLoading F1 baseline data (no SYNOP)...")
        df_f1 = con.execute(build_sql(include_synop=False)).fetchdf()
        df_f1[TIME_COL] = pd.to_datetime(df_f1[TIME_COL])
        df_f1 = add_features(df_f1)
        print(f"Total rows (F1): {len(df_f1):,}")
        results_f1 = run_config(df_f1, FEATURES_F1, "F1_baseline_50feat")
    else:
        print(f"\nSkipping F1 baseline retrain (RUN_BASELINE=False).")
        print(f"Using stored reference: LGBM={REF_R2_LGBM}  CatBoost={REF_R2_CB}")
        results_f1 = pd.DataFrame([
            {"config": "F1_baseline_50feat", "model": "lgbm_residual",
             "target": "point_t60", "n": 22711, "r2": REF_R2_LGBM,
             "mae": 96.7, "rmse": 137.9, "skill_vs_sp": 0.2583, "best_iter": 400},
            {"config": "F1_baseline_50feat", "model": "catboost_direct",
             "target": "point_t60", "n": 22711, "r2": REF_R2_CB,
             "mae": 96.5, "rmse": 137.0, "skill_vs_sp": 0.2633, "best_iter": 959},
        ])

    # -----------------------------------------------------------------------
    # CONFIG B: F1 + SYNOP (59 features)
    # -----------------------------------------------------------------------
    print("\nLoading F1+SYNOP data (with SYNOP join)...")
    df_f1d = con.execute(build_sql(include_synop=True)).fetchdf()
    df_f1d[TIME_COL] = pd.to_datetime(df_f1d[TIME_COL])
    df_f1d = add_features(df_f1d)
    print(f"Total rows (F1+SYNOP): {len(df_f1d):,}")

    # Verify join didn't multiply rows (compare to known R1 baseline count)
    if len(df_f1d) != REF_ROW_COUNT:
        print(f"  WARNING: row count {len(df_f1d):,} != expected {REF_ROW_COUNT:,} "
              f"(diff = {len(df_f1d) - REF_ROW_COUNT:+,})")
        print("  Check for duplicate SYNOP rows per hour in synop_bengkulu.")
    else:
        print(f"  Row count OK: {len(df_f1d):,} (matches R1 baseline)")

    report_synop_coverage(df_f1d)

    results_f1d = run_config(df_f1d, FEATURES_F1D, "F1D_synop_59feat")

    con.close()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    all_results = pd.concat([results_f1, results_f1d], ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "armD_results.csv", index=False)

    print(f"\n{'='*64}")
    print("SUMMARY — test 2025 (point target, R² comparison)")
    print(f"{'='*64}")

    summary = all_results[
        (all_results["model"].isin(["lgbm_residual", "catboost_direct"]))
        & (all_results["target"] == "point_t60")
    ][["config", "model", "r2", "mae", "rmse", "skill_vs_sp"]].copy()

    # Add delta vs F1 baseline
    ref_vals = {}
    for _, row in summary.iterrows():
        if row["config"] == "F1_baseline_50feat":
            ref_vals[row["model"]] = row["r2"]

    summary["dR2_vs_F1"] = summary.apply(
        lambda r: r["r2"] - ref_vals.get(r["model"], float("nan")), axis=1
    )
    summary["dR2_vs_ref"] = summary.apply(
        lambda r: (r["r2"] - REF_R2_LGBM) if "lgbm" in r["model"]
                  else (r["r2"] - REF_R2_CB), axis=1
    )

    print(summary.to_string(index=False))

    print(f"\nRef: LGBM={REF_R2_LGBM}  CatBoost={REF_R2_CB}")
    print(f"Noise floor (from Banten §2.3 precedent): ±0.003")
    print(f"\nConclusion guidance:")
    for _, row in summary[summary["config"] == "F1D_synop_59feat"].iterrows():
        dr = row["dR2_vs_F1"]
        if abs(dr) < 0.003:
            verdict = "SYNOP tidak berkontribusi signifikan (noise floor ±0.003)"
        elif dr >= 0.003:
            verdict = "SYNOP BERKONTRIBUSI — keunggulan Bengkulu sebagian dari data"
        else:
            verdict = "SYNOP merugikan — aneh, periksa NaN imputation"
        print(f"  [{row['model']}] dR2={dr:+.4f} → {verdict}")

    print(f"\nAll outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
