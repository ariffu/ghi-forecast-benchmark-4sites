#!/usr/bin/env python3
"""
Calibration layer for direct 60-minute-ahead GHI prediction in Bengkulu.

Goal:
    Fix underprediction in high irradiance conditions, especially target GHI > 900 W/m2
    and midday/high-sun periods, without leaking future target into features.

Approach:
    1. Train base direct LightGBM on 2021-2023.
    2. Predict validation 2024 and test 2025.
    3. Fit calibration layers ONLY on validation predictions from 2024:
        - isotonic calibration: raw_prediction -> actual GHI
        - bin-bias calibration: median residual by raw prediction bin
        - high-GHI residual calibrator: LightGBM residual correction for high predicted / high clear-sky cases
    4. Apply calibration to test 2025.
    5. Compare raw direct vs calibrated outputs by overall metrics and high-GHI segments.

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow lightgbm tqdm

Run:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal if using setx
    python calibrate_direct_ghi_60m_high_ghi.py

Outputs:
    outputs_calibrated_60m/metrics.csv
    outputs_calibrated_60m/metrics_by_segment.csv
    outputs_calibrated_60m/predictions_test_calibrated.csv
    outputs_calibrated_60m/high_ghi_worst_errors.csv
    outputs_calibrated_60m/diagnostics.png
    outputs_calibrated_60m/models/base_direct_regularized.joblib
    outputs_calibrated_60m/models/calibrator_isotonic.joblib
    outputs_calibrated_60m/models/calibrator_high_residual.joblib
    outputs_calibrated_60m/models/calibrator_bin_bias.joblib
"""

import os
from pathlib import Path
import warnings

import duckdb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
OUTPUT_DIR = Path("outputs_calibrated_60m")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "target_ghi_60m"
TARGET_TS_COL = "target_ts_60m"
PRED_MIN = 0.0
PRED_MAX = 1400.0
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Thresholds are intentionally based on available-at-prediction-time values, not actual target.
HIGH_PRED_THRESHOLD = 650.0
HIGH_CLEARSKY_THRESHOLD = 850.0
HIGH_SOLAR_ELEV_THRESHOLD = 30.0
HIGH_HOURS = list(range(8, 15))

FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "daylight_flag", "sun_above_5deg_flag",
    "ghi_now", "dhi_now", "dni_now", "reflected_now", "nett_rad_now", "solar_elev_deg",
    "asrs_n_obs_1min", "asrs_ok_obs", "aws_temp_c", "aws_temp_min_c", "aws_temp_max_c",
    "aws_rh_pct", "aws_pressure_hpa", "aws_ws_avg", "aws_ws_max", "aws_wd_deg", "aws_rain_mm",
    "aws_sr_avg_w_m2", "clp_cot", "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "clp_clear_flag", "clp_thin_cloud_flag", "clp_moderate_cloud_flag", "clp_thick_cloud_flag",
    "synop_temp_c", "synop_dewpoint_c", "synop_rh_pct", "synop_wind_speed", "synop_wind_dir_deg",
    "synop_visibility", "synop_rainfall_24h_mm", "synop_solar_rad_24h",
    "ghi_lag_10m", "ghi_lag_30m", "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "dhi_lag_60m", "dni_lag_60m", "aws_temp_lag_60m", "aws_rh_lag_60m", "aws_pressure_lag_60m",
    "clp_cot_lag_60m", "clp_cth_lag_60m", "ghi_roll_30m_mean", "ghi_roll_30m_min", "ghi_roll_30m_max",
    "ghi_roll_30m_std", "ghi_roll_60m_mean", "ghi_roll_60m_min", "ghi_roll_60m_max", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_min", "ghi_roll_180m_max", "ghi_roll_180m_std",
    "dhi_roll_180m_mean", "dni_roll_180m_mean", "aws_temp_roll_180m_mean", "aws_rh_roll_180m_mean",
    "aws_ws_roll_180m_mean", "aws_rain_sum_180m", "clp_cot_roll_180m_mean", "clp_cth_roll_180m_mean",
    "ghi_delta_10m", "ghi_delta_60m", "aws_temp_delta_60m", "aws_rh_delta_60m",
    "solar_elev_sin", "solar_elev_sin_clip", "clear_sky_ghi_now", "clear_sky_ghi_target_60m",
    "kt_now", "ghi_to_aws_sr_ratio", "dhi_fraction", "dni_fraction", "diffuse_to_global_ratio",
    "temp_rh_interaction", "vpd_proxy", "wind_u", "wind_v",
    "clp_cot_delta_60m", "clp_cth_delta_60m", "clp_cot_delta_180m", "clp_cth_delta_180m",
    "ghi_roll_180m_range", "ghi_roll_60m_range", "ghi_ramp_ratio_60m", "aws_temp_range",
    "cloud_opacity_proxy", "cloud_height_temp_interaction"
]

CALIBRATION_FEATURES = [
    "pred_raw", "pred_raw_sq", "pred_clear_ratio", "raw_minus_now", "ghi_now", "solar_elev_deg",
    "target_solar_elev_deg_60m", "clear_sky_ghi_target_60m", "kt_now", "hour", "month",
    "hour_sin", "hour_cos", "ghi_delta_10m", "ghi_delta_60m", "ghi_roll_60m_mean", "ghi_roll_180m_mean",
    "ghi_roll_60m_range", "ghi_roll_180m_range", "dhi_fraction", "dni_fraction", "aws_rh_pct",
    "aws_temp_c", "aws_pressure_hpa", "aws_ws_avg", "clp_cot", "clp_cth_m", "clp_ctt_k",
    "clp_cloud_present", "clp_clear_flag", "clp_thin_cloud_flag", "clp_moderate_cloud_flag", "clp_thick_cloud_flag",
    "cloud_opacity_proxy", "clp_cot_delta_60m", "clp_cth_delta_60m"
]


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN environment variable.")
    os.environ["motherduck_token"] = token


def connect_motherduck():
    require_token()
    con = duckdb.connect(database=":memory:")
    con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
    return con


def build_sql():
    # Same direct 60m dataset construction used in explore_direct_ghi_60m_bengkulu.py.
    return """
    WITH master AS (
        SELECT m.*
        FROM bengkulu_db.bengkulu_sch.bengkulu_master_10min_quality_final m
        WHERE m.ts_wib >= TIMESTAMP '2021-01-01'
          AND m.ts_wib < TIMESTAMP '2026-01-01'
    ), feat AS (
        SELECT
            m.ts_wib,
            m.ts_wib + INTERVAL '60 minutes' AS target_ts_60m,
            LEAD(m.ts_wib, 6) OVER (ORDER BY m.ts_wib) AS observed_target_ts_60m,
            LEAD(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS target_ghi_60m,
            LEAD(m.asrs_solar_elev_deg, 6) OVER (ORDER BY m.ts_wib) AS target_solar_elev_deg_60m,
            m.year, m.month, m.day, m.hour, m.minute,
            SIN(2 * PI() * m.hour / 24.0) AS hour_sin,
            COS(2 * PI() * m.hour / 24.0) AS hour_cos,
            SIN(2 * PI() * m.month / 12.0) AS month_sin,
            COS(2 * PI() * m.month / 12.0) AS month_cos,
            CASE WHEN m.asrs_solar_elev_deg > 0 THEN 1 ELSE 0 END AS daylight_flag,
            CASE WHEN m.asrs_solar_elev_deg > 5 THEN 1 ELSE 0 END AS sun_above_5deg_flag,
            m.asrs_ghi_w_m2 AS ghi_now,
            m.asrs_dhi_w_m2 AS dhi_now,
            m.asrs_dni_w_m2 AS dni_now,
            m.asrs_reflected_w_m2 AS reflected_now,
            m.asrs_nett_rad_w_m2 AS nett_rad_now,
            m.asrs_solar_elev_deg AS solar_elev_deg,
            m.asrs_n_obs_1min, m.asrs_ok_obs,
            m.aws_temp_c, m.aws_temp_min_c, m.aws_temp_max_c, m.aws_rh_pct, m.aws_pressure_hpa,
            m.aws_ws_avg, m.aws_ws_max, m.aws_wd_deg, m.aws_rain_mm, m.aws_sr_avg_w_m2,
            m.clp_cot, m.clp_cth_m, m.clp_ctt_k, m.clp_cer,
            CAST(m.clp_cloud_present AS INT) AS clp_cloud_present,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%clear%' THEN 1 ELSE 0 END AS clp_clear_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%thin%' THEN 1 ELSE 0 END AS clp_thin_cloud_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%moderate%' THEN 1 ELSE 0 END AS clp_moderate_cloud_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%thick%' THEN 1 ELSE 0 END AS clp_thick_cloud_flag,
            m.synop_temp_c, m.synop_dewpoint_c, m.synop_rh_pct, m.synop_wind_speed, m.synop_wind_dir_deg,
            m.synop_visibility, m.synop_rainfall_24h_mm, m.synop_solar_rad_24h,
            m.has_asrs, m.has_aws, m.has_clp, m.has_synop, m.master_qc_status,
            m.asrs_qc_status, m.aws_qc_status, m.clp_qc_status, m.synop_qc_status,
            LAG(m.asrs_ghi_w_m2, 1) OVER (ORDER BY m.ts_wib) AS ghi_lag_10m,
            LAG(m.asrs_ghi_w_m2, 3) OVER (ORDER BY m.ts_wib) AS ghi_lag_30m,
            LAG(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS ghi_lag_60m,
            LAG(m.asrs_ghi_w_m2, 12) OVER (ORDER BY m.ts_wib) AS ghi_lag_120m,
            LAG(m.asrs_ghi_w_m2, 18) OVER (ORDER BY m.ts_wib) AS ghi_lag_180m,
            LAG(m.asrs_dhi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS dhi_lag_60m,
            LAG(m.asrs_dni_w_m2, 6) OVER (ORDER BY m.ts_wib) AS dni_lag_60m,
            LAG(m.aws_temp_c, 6) OVER (ORDER BY m.ts_wib) AS aws_temp_lag_60m,
            LAG(m.aws_rh_pct, 6) OVER (ORDER BY m.ts_wib) AS aws_rh_lag_60m,
            LAG(m.aws_pressure_hpa, 6) OVER (ORDER BY m.ts_wib) AS aws_pressure_lag_60m,
            LAG(m.clp_cot, 6) OVER (ORDER BY m.ts_wib) AS clp_cot_lag_60m,
            LAG(m.clp_cth_m, 6) OVER (ORDER BY m.ts_wib) AS clp_cth_lag_60m,
            AVG(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_mean,
            MIN(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_min,
            MAX(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_max,
            STDDEV_SAMP(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_std,
            AVG(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_mean,
            MIN(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_min,
            MAX(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_max,
            STDDEV_SAMP(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_std,
            AVG(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
            MIN(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_min,
            MAX(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_max,
            STDDEV_SAMP(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
            AVG(m.asrs_dhi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS dhi_roll_180m_mean,
            AVG(m.asrs_dni_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS dni_roll_180m_mean,
            AVG(m.aws_temp_c) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_temp_roll_180m_mean,
            AVG(m.aws_rh_pct) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_rh_roll_180m_mean,
            AVG(m.aws_ws_avg) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_ws_roll_180m_mean,
            SUM(COALESCE(m.aws_rain_mm, 0)) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_rain_sum_180m,
            AVG(m.clp_cot) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
            AVG(m.clp_cth_m) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cth_roll_180m_mean,
            m.asrs_ghi_w_m2 - LAG(m.asrs_ghi_w_m2, 1) OVER (ORDER BY m.ts_wib) AS ghi_delta_10m,
            m.asrs_ghi_w_m2 - LAG(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS ghi_delta_60m,
            m.aws_temp_c - LAG(m.aws_temp_c, 6) OVER (ORDER BY m.ts_wib) AS aws_temp_delta_60m,
            m.aws_rh_pct - LAG(m.aws_rh_pct, 6) OVER (ORDER BY m.ts_wib) AS aws_rh_delta_60m
        FROM master m
    )
    SELECT * FROM feat
    WHERE observed_target_ts_60m = target_ts_60m
      AND target_ts_60m < TIMESTAMP '2026-01-01'
      AND target_ghi_60m IS NOT NULL
      AND ghi_now IS NOT NULL
    ORDER BY ts_wib
    """


def add_engineered(df):
    out = df.copy()
    elev_rad = np.deg2rad(out["solar_elev_deg"].astype(float))
    elev_sin = np.sin(elev_rad)
    out["solar_elev_sin"] = elev_sin
    out["solar_elev_sin_clip"] = np.maximum(elev_sin, 0.02)
    out["clear_sky_ghi_now"] = 1100.0 * out["solar_elev_sin_clip"]
    target_sin = np.maximum(np.sin(np.deg2rad(out["target_solar_elev_deg_60m"].astype(float))), 0.02)
    out["clear_sky_ghi_target_60m"] = 1100.0 * target_sin
    out["kt_now"] = out["ghi_now"] / np.maximum(out["clear_sky_ghi_now"], 20.0)
    out["ghi_to_aws_sr_ratio"] = out["ghi_now"] / np.maximum(out["aws_sr_avg_w_m2"], 20.0)
    out["dhi_fraction"] = out["dhi_now"] / np.maximum(out["ghi_now"], 20.0)
    out["dni_fraction"] = out["dni_now"] / np.maximum(out["ghi_now"], 20.0)
    out["diffuse_to_global_ratio"] = out["dhi_now"] / np.maximum(out["ghi_now"], 20.0)
    out["temp_rh_interaction"] = out["aws_temp_c"] * out["aws_rh_pct"]
    out["vpd_proxy"] = out["aws_temp_c"] * (100.0 - out["aws_rh_pct"]) / 100.0
    wd_rad = np.deg2rad(out["aws_wd_deg"].astype(float))
    out["wind_u"] = out["aws_ws_avg"] * np.sin(wd_rad)
    out["wind_v"] = out["aws_ws_avg"] * np.cos(wd_rad)
    out["clp_cot_delta_60m"] = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cth_delta_60m"] = out["clp_cth_m"] - out["clp_cth_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["clp_cth_delta_180m"] = out["clp_cth_m"] - out["clp_cth_roll_180m_mean"]
    out["ghi_roll_180m_range"] = out["ghi_roll_180m_max"] - out["ghi_roll_180m_min"]
    out["ghi_roll_60m_range"] = out["ghi_roll_60m_max"] - out["ghi_roll_60m_min"]
    out["ghi_ramp_ratio_60m"] = out["ghi_delta_60m"] / np.maximum(out["ghi_lag_60m"].abs(), 20.0)
    out["aws_temp_range"] = out["aws_temp_max_c"] - out["aws_temp_min_c"]
    out["cloud_opacity_proxy"] = out["clp_cot"] * out["clp_cloud_present"].fillna(0)
    out["cloud_height_temp_interaction"] = out["clp_cth_m"] * out["clp_ctt_k"]
    return out


def ready_mask(df):
    return (
        (df[TARGET_COL].between(0, 1400)) &
        (df["ghi_now"].between(0, 1400)) &
        (df["daylight_flag"] == 1) &
        (df["target_solar_elev_deg_60m"] > 0) &
        (df["has_asrs"] == True) &
        (df["has_aws"] == True) &
        (df["asrs_qc_status"].isin(["ok", "warn_partial_asrs_qc"])) &
        (df["aws_qc_status"] == "ok") &
        (df["ghi_lag_180m"].notna()) &
        (df["ghi_roll_180m_mean"].notna())
    )


def split_masks(df):
    mask = ready_mask(df)
    train = mask & (df[TIME_COL] < pd.Timestamp("2024-01-01"))
    valid = mask & (df[TIME_COL] >= pd.Timestamp("2024-01-01")) & (df[TIME_COL] < pd.Timestamp("2025-01-01"))
    test = mask & (df[TIME_COL] >= pd.Timestamp("2025-01-01"))
    return train, valid, test


def make_base_model():
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2600,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=90,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.25,
        reg_lambda=3.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])


def make_high_condition(df, pred_col="pred_raw"):
    return (
        (df[pred_col] >= HIGH_PRED_THRESHOLD) |
        ((df["clear_sky_ghi_target_60m"] >= HIGH_CLEARSKY_THRESHOLD) & (df["solar_elev_deg"] >= HIGH_SOLAR_ELEV_THRESHOLD) & (df["hour"].isin(HIGH_HOURS)))
    )


def add_calibration_features(df, pred_col="pred_raw"):
    out = df.copy()
    out["pred_raw"] = out[pred_col]
    out["pred_raw_sq"] = out["pred_raw"] ** 2
    out["pred_clear_ratio"] = out["pred_raw"] / np.maximum(out["clear_sky_ghi_target_60m"], 20.0)
    out["raw_minus_now"] = out["pred_raw"] - out["ghi_now"]
    return out


def fit_bin_bias(valid_df):
    work = valid_df.copy()
    bins = [-1, 100, 300, 500, 650, 800, 900, 1000, 1200, 1500]
    work["pred_bin"] = pd.cut(work["pred_raw"], bins=bins, include_lowest=True)
    bias = work.groupby("pred_bin")["residual_raw"].median().fillna(0)
    return {"bins": bins, "bias": bias}


def apply_bin_bias(df, bin_model):
    bins = bin_model["bins"]
    bias = bin_model["bias"]
    pred_bin = pd.cut(df["pred_raw"], bins=bins, include_lowest=True)
    correction = pred_bin.map(bias).astype(float).fillna(0).values
    return clip_ghi(df["pred_raw"].values + correction)


def fit_high_residual_calibrator(valid_df):
    high = make_high_condition(valid_df, "pred_raw")
    train_df = valid_df[high].copy()
    if len(train_df) < 500:
        train_df = valid_df.copy()
    y = np.clip(train_df["residual_raw"].values, -350, 350)
    reg = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])
    pipe.fit(train_df[CALIBRATION_FEATURES], y)
    return pipe


def apply_high_residual_calibrator(df, calibrator, strength=0.75):
    high = make_high_condition(df, "pred_raw")
    correction = calibrator.predict(df[CALIBRATION_FEATURES])
    correction = np.clip(correction, -250, 300)
    out = df["pred_raw"].values.copy()
    out[high.values] = out[high.values] + strength * correction[high.values]
    return clip_ghi(out)


def metric_row(y, pred, model_name, persistence_rmse):
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    mbe = float(np.mean(pred - y))
    skill = 1.0 - rmse / persistence_rmse if persistence_rmse > 0 else np.nan
    if model_name == "persistence":
        skill = 0.0
    return {"model": model_name, "n_rows": len(y), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def evaluate(pred_df, pred_cols):
    y = pred_df[TARGET_COL].values
    persistence_rmse = float(np.sqrt(mean_squared_error(y, pred_df["persistence"].values)))
    return pd.DataFrame([metric_row(y, pred_df[col].values, col, persistence_rmse) for col in pred_cols])


def segment_metrics(pred_df, pred_cols):
    rows = []
    pred_df = pred_df.copy()
    pred_df["target_ghi_bin"] = pd.cut(pred_df[TARGET_COL], bins=[0, 100, 300, 600, 900, 1400], labels=["0-100", "100-300", "300-600", "600-900", "900+"])
    pred_df["pred_ghi_bin"] = pd.cut(pred_df["pred_raw"], bins=[0, 100, 300, 600, 900, 1400], labels=["0-100", "100-300", "300-600", "600-900", "900+"])
    pred_df["solar_segment"] = pd.cut(pred_df["solar_elev_deg"], bins=[-90, 15, 35, 90], labels=["low", "medium", "high"])
    for seg_col in ["target_ghi_bin", "pred_ghi_bin", "hour", "solar_segment", "month", "has_clp"]:
        for seg_val, group in pred_df.groupby(seg_col, dropna=False):
            if len(group) < 30:
                continue
            persistence_rmse = float(np.sqrt(mean_squared_error(group[TARGET_COL], group["persistence"])))
            for col in pred_cols:
                row = metric_row(group[TARGET_COL].values, group[col].values, col, persistence_rmse)
                row["segment_col"] = seg_col
                row["segment_val"] = str(seg_val)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading direct 60m data...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_TS_COL] = pd.to_datetime(df[TARGET_TS_COL])
    df = add_engineered(df)
    train_mask, valid_mask, test_mask = split_masks(df)
    print("Rows train " + str(int(train_mask.sum())) + " valid " + str(int(valid_mask.sum())) + " test " + str(int(test_mask.sum())))

    print("Training base direct_regularized_all model...")
    base = make_base_model()
    base.fit(
        df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET_COL],
        model__eval_set=[(df.loc[valid_mask, FEATURES], df.loc[valid_mask, TARGET_COL])],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(120, verbose=False)]
    )

    valid_df = df.loc[valid_mask].copy()
    test_df = df.loc[test_mask].copy()
    valid_df["pred_raw"] = clip_ghi(base.predict(valid_df[FEATURES]))
    test_df["pred_raw"] = clip_ghi(base.predict(test_df[FEATURES]))
    valid_df["persistence"] = clip_ghi(valid_df["ghi_now"].values)
    test_df["persistence"] = clip_ghi(test_df["ghi_now"].values)
    valid_df["residual_raw"] = valid_df[TARGET_COL] - valid_df["pred_raw"]
    test_df["residual_raw"] = test_df[TARGET_COL] - test_df["pred_raw"]
    valid_df = add_calibration_features(valid_df, "pred_raw")
    test_df = add_calibration_features(test_df, "pred_raw")

    print("Fitting calibration layers on validation 2024 predictions only...")
    iso = IsotonicRegression(y_min=0, y_max=1400, out_of_bounds="clip")
    iso.fit(valid_df["pred_raw"].values, valid_df[TARGET_COL].values)
    bin_model = fit_bin_bias(valid_df)
    high_cal = fit_high_residual_calibrator(valid_df)

    for work_df in [valid_df, test_df]:
        work_df["pred_isotonic"] = clip_ghi(iso.predict(work_df["pred_raw"].values))
        work_df["pred_bin_bias"] = apply_bin_bias(work_df, bin_model)
        work_df["pred_high_residual_cal"] = apply_high_residual_calibrator(work_df, high_cal, strength=0.75)
        work_df["pred_high_residual_cal_full"] = apply_high_residual_calibrator(work_df, high_cal, strength=1.0)
        work_df["pred_blend_raw_highcal_50"] = clip_ghi(0.5 * work_df["pred_raw"].values + 0.5 * work_df["pred_high_residual_cal"].values)
        work_df["pred_blend_iso_highcal_50"] = clip_ghi(0.5 * work_df["pred_isotonic"].values + 0.5 * work_df["pred_high_residual_cal"].values)

    pred_cols = [
        "persistence", "pred_raw", "pred_isotonic", "pred_bin_bias", "pred_high_residual_cal",
        "pred_high_residual_cal_full", "pred_blend_raw_highcal_50", "pred_blend_iso_highcal_50"
    ]
    valid_metrics = evaluate(valid_df, pred_cols)
    valid_metrics["split"] = "valid_2024"
    test_metrics = evaluate(test_df, pred_cols)
    test_metrics["split"] = "test_2025"
    metrics_df = pd.concat([valid_metrics, test_metrics], ignore_index=True)
    seg_df = segment_metrics(test_df, pred_cols)

    best_model = test_metrics.sort_values("rmse").iloc[0]["model"]
    test_df["best_prediction"] = test_df[best_model]
    test_df["best_error"] = test_df["best_prediction"] - test_df[TARGET_COL]
    test_df["raw_error"] = test_df["pred_raw"] - test_df[TARGET_COL]
    worst_df = test_df.reindex(test_df["best_error"].abs().sort_values(ascending=False).index).head(300)

    print("Calibration metrics")
    print(metrics_df.sort_values(["split", "rmse"]).to_string(index=False))
    print("High-GHI segment comparison")
    high_seg = seg_df[(seg_df["segment_col"] == "target_ghi_bin") & (seg_df["segment_val"] == "900+")].sort_values("rmse")
    print(high_seg.to_string(index=False))
    print("Best calibrated model: " + best_model)

    metrics_path = OUTPUT_DIR / "metrics.csv"
    seg_path = OUTPUT_DIR / "metrics_by_segment.csv"
    pred_path = OUTPUT_DIR / "predictions_test_calibrated.csv"
    worst_path = OUTPUT_DIR / "high_ghi_worst_errors.csv"
    plot_path = OUTPUT_DIR / "diagnostics.png"
    metrics_df.to_csv(metrics_path, index=False)
    seg_df.to_csv(seg_path, index=False)
    test_df.to_csv(pred_path, index=False)
    worst_df.to_csv(worst_path, index=False)
    joblib.dump({"pipeline": base, "features": FEATURES, "target": TARGET_COL}, MODEL_DIR / "base_direct_regularized.joblib")
    joblib.dump(iso, MODEL_DIR / "calibrator_isotonic.joblib")
    joblib.dump(high_cal, MODEL_DIR / "calibrator_high_residual.joblib")
    joblib.dump(bin_model, MODEL_DIR / "calibrator_bin_bias.joblib")

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plot_metrics = test_metrics.sort_values("rmse")
    sns.barplot(data=plot_metrics, y="model", x="r2")
    plt.axvline(0.9, color="red", linestyle="--", linewidth=1)
    plt.title("Calibrated direct 60m - Test R2")
    plt.xlabel("R2")
    plt.ylabel("")

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=test_df, x=TARGET_COL, y="pred_raw", s=8, alpha=0.25, label="raw")
    sns.scatterplot(data=test_df, x=TARGET_COL, y=best_model, s=8, alpha=0.25, label="calibrated")
    plt.plot([0, 1200], [0, 1200], color="black", linewidth=1)
    plt.title("Raw vs calibrated prediction")
    plt.xlabel("Actual GHI t+60m")
    plt.ylabel("Predicted")

    plt.subplot(2, 2, 3)
    high_df = test_df[test_df[TARGET_COL] >= 900]
    if len(high_df) > 0:
        sns.kdeplot(high_df["pred_raw"] - high_df[TARGET_COL], label="raw high-GHI")
        sns.kdeplot(high_df[best_model] - high_df[TARGET_COL], label="calibrated high-GHI")
    plt.title("Error distribution for target GHI >= 900")
    plt.xlabel("Prediction error")
    plt.legend()

    plt.subplot(2, 2, 4)
    by_hour = seg_df[(seg_df["segment_col"] == "hour") & (seg_df["model"].isin(["pred_raw", best_model]))].copy()
    sns.lineplot(data=by_hour, x="segment_val", y="rmse", hue="model", marker="o")
    plt.title("RMSE by hour")
    plt.xlabel("Hour")
    plt.ylabel("RMSE")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("Saved outputs:")
    for p in [metrics_path, seg_path, pred_path, worst_path, plot_path]:
        print(str(p))
    print("Models saved under: " + str(MODEL_DIR))


if __name__ == "__main__":
    main()
