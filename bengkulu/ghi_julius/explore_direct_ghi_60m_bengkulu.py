#!/usr/bin/env python3
"""
Direct 60-minute-ahead GHI prediction exploration for Bengkulu, 2021-2025 only.

Purpose:
    Focus only on direct prediction of GHI at t+60 minutes.
    Explore why 1-hour performance drops by producing detailed diagnostics:
        - Direct LightGBM model variants
        - Feature-importance comparison
        - Error by hour, solar elevation, cloud status, GHI bin, month
        - Worst-case examples
        - Optional reduced top-N feature models

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow lightgbm tqdm

Run:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal after setx
    python explore_direct_ghi_60m_bengkulu.py

Outputs:
    outputs_direct_60m/metrics.csv
    outputs_direct_60m/metrics_by_segment.csv
    outputs_direct_60m/feature_importance.csv
    outputs_direct_60m/predictions_test.csv
    outputs_direct_60m/worst_errors.csv
    outputs_direct_60m/diagnostics.png
    outputs_direct_60m/models/*.joblib
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
OUTPUT_DIR = Path("outputs_direct_60m")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "target_ghi_60m"
TARGET_TS_COL = "target_ts_60m"
PRED_MIN = 0.0
PRED_MAX = 1400.0
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_FEATURES = [
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
    "ghi_delta_10m", "ghi_delta_60m", "aws_temp_delta_60m", "aws_rh_delta_60m"
]

SYNOP_CLOUD_FEATURES = [
    "syn_cloud_cover_oktas_m", "syn_cloud_low_cover_oktas", "syn_cloud_med_cover_oktas",
    "syn_cloud_low_base_1", "syn_cloud_low_base_2", "syn_cloud_low_base_3",
    "syn_cloud_med_base_1", "syn_cloud_med_base_2", "syn_cloud_high_base_1", "syn_cloud_high_base_2",
    "syn_cloud_layer_1_height_m", "syn_cloud_layer_2_height_m", "syn_cloud_layer_3_height_m", "syn_cloud_layer_4_height_m",
    "syn_cloud_layer_1_amt_oktas", "syn_cloud_layer_2_amt_oktas", "syn_cloud_layer_3_amt_oktas", "syn_cloud_layer_4_amt_oktas",
    "syn_present_weather", "syn_past_weather_1", "syn_past_weather_2"
]

ENGINEERED_FEATURES = [
    "solar_elev_sin", "solar_elev_sin_clip", "clear_sky_ghi_now", "clear_sky_ghi_target_60m",
    "kt_now", "ghi_to_aws_sr_ratio", "dhi_fraction", "dni_fraction", "diffuse_to_global_ratio",
    "temp_rh_interaction", "vpd_proxy", "wind_u", "wind_v",
    "clp_cot_delta_60m", "clp_cth_delta_60m", "clp_cot_delta_180m", "clp_cth_delta_180m",
    "ghi_roll_180m_range", "ghi_roll_60m_range", "ghi_ramp_ratio_60m", "aws_temp_range",
    "cloud_opacity_proxy", "cloud_height_temp_interaction",
    "syn_total_cloud_oktas", "syn_low_cloud_present", "syn_multi_layer_cloud_flag", "syn_cloud_base_min",
    "syn_cloud_depth_proxy", "syn_weather_cloud_rain_proxy"
]

ALL_FEATURES = BASE_FEATURES + SYNOP_CLOUD_FEATURES + ENGINEERED_FEATURES
NO_CLOUD_FEATURES = [f for f in ALL_FEATURES if not (f.startswith("clp_") or f.startswith("syn_cloud") or f.startswith("syn_") or "cloud" in f)]
RAD_MET_FEATURES = [
    f for f in ALL_FEATURES
    if f.startswith("ghi") or f.startswith("dhi") or f.startswith("dni") or f.startswith("aws") or f in [
        "hour_sin", "hour_cos", "month_sin", "month_cos", "solar_elev_deg", "solar_elev_sin", "solar_elev_sin_clip",
        "clear_sky_ghi_now", "clear_sky_ghi_target_60m", "kt_now", "daylight_flag", "sun_above_5deg_flag"
    ]
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
    return """
    WITH master AS (
        SELECT
            m.*,
            s.cloud_cover_oktas_m AS syn_cloud_cover_oktas_m,
            s.cloud_low_cover_oktas AS syn_cloud_low_cover_oktas,
            s.cloud_med_cover_oktas AS syn_cloud_med_cover_oktas,
            s.cloud_low_base_1 AS syn_cloud_low_base_1,
            s.cloud_low_base_2 AS syn_cloud_low_base_2,
            s.cloud_low_base_3 AS syn_cloud_low_base_3,
            s.cloud_med_base_1 AS syn_cloud_med_base_1,
            s.cloud_med_base_2 AS syn_cloud_med_base_2,
            s.cloud_high_base_1 AS syn_cloud_high_base_1,
            s.cloud_high_base_2 AS syn_cloud_high_base_2,
            s.cloud_layer_1_height_m_hshs AS syn_cloud_layer_1_height_m,
            s.cloud_layer_2_height_m_hshs AS syn_cloud_layer_2_height_m,
            s.cloud_layer_3_height_m_hshs AS syn_cloud_layer_3_height_m,
            s.cloud_layer_4_height_m_hshs AS syn_cloud_layer_4_height_m,
            s.cloud_layer_1_amt_oktas_ns AS syn_cloud_layer_1_amt_oktas,
            s.cloud_layer_2_amt_oktas_ns AS syn_cloud_layer_2_amt_oktas,
            s.cloud_layer_3_amt_oktas_ns AS syn_cloud_layer_3_amt_oktas,
            s.cloud_layer_4_amt_oktas_ns AS syn_cloud_layer_4_amt_oktas,
            s.present_weather_ww AS syn_present_weather,
            TRY_CAST(s.past_weather_w1 AS DOUBLE) AS syn_past_weather_1,
            TRY_CAST(s.past_weather_w2 AS DOUBLE) AS syn_past_weather_2
        FROM bengkulu_db.bengkulu_sch.bengkulu_master_10min_quality_final m
        LEFT JOIN bengkulu_db.bengkulu_sch.synop_bengkulu_quality_final s
          ON time_bucket(INTERVAL '1 hour', m.ts_wib) = s.ts_wib
        WHERE m.ts_wib >= TIMESTAMP '2021-01-01'
          AND m.ts_wib < TIMESTAMP '2026-01-01'
    ), feat AS (
        SELECT
            m.ts_wib,
            m.ts_wib + INTERVAL '60 minutes' AS target_ts_60m,
            LEAD(m.ts_wib, 6) OVER (ORDER BY m.ts_wib) AS observed_target_ts_60m,
            LEAD(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS target_ghi_60m,
            LEAD(m.asrs_solar_elev_deg, 6) OVER (ORDER BY m.ts_wib) AS target_solar_elev_deg_60m,
            m.year,
            m.month,
            m.day,
            m.hour,
            m.minute,
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
            m.asrs_n_obs_1min,
            m.asrs_ok_obs,
            m.aws_temp_c,
            m.aws_temp_min_c,
            m.aws_temp_max_c,
            m.aws_rh_pct,
            m.aws_pressure_hpa,
            m.aws_ws_avg,
            m.aws_ws_max,
            m.aws_wd_deg,
            m.aws_rain_mm,
            m.aws_sr_avg_w_m2,
            m.clp_cot,
            m.clp_cth_m,
            m.clp_ctt_k,
            m.clp_cer,
            CAST(m.clp_cloud_present AS INT) AS clp_cloud_present,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%clear%' THEN 1 ELSE 0 END AS clp_clear_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%thin%' THEN 1 ELSE 0 END AS clp_thin_cloud_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%moderate%' THEN 1 ELSE 0 END AS clp_moderate_cloud_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%thick%' THEN 1 ELSE 0 END AS clp_thick_cloud_flag,
            m.synop_temp_c,
            m.synop_dewpoint_c,
            m.synop_rh_pct,
            m.synop_wind_speed,
            m.synop_wind_dir_deg,
            m.synop_visibility,
            m.synop_rainfall_24h_mm,
            m.synop_solar_rad_24h,
            m.has_asrs,
            m.has_aws,
            m.has_clp,
            m.has_synop,
            m.master_qc_status,
            m.asrs_qc_status,
            m.aws_qc_status,
            m.clp_qc_status,
            m.synop_qc_status,
            m.syn_cloud_cover_oktas_m,
            m.syn_cloud_low_cover_oktas,
            m.syn_cloud_med_cover_oktas,
            m.syn_cloud_low_base_1,
            m.syn_cloud_low_base_2,
            m.syn_cloud_low_base_3,
            m.syn_cloud_med_base_1,
            m.syn_cloud_med_base_2,
            m.syn_cloud_high_base_1,
            m.syn_cloud_high_base_2,
            m.syn_cloud_layer_1_height_m,
            m.syn_cloud_layer_2_height_m,
            m.syn_cloud_layer_3_height_m,
            m.syn_cloud_layer_4_height_m,
            m.syn_cloud_layer_1_amt_oktas,
            m.syn_cloud_layer_2_amt_oktas,
            m.syn_cloud_layer_3_amt_oktas,
            m.syn_cloud_layer_4_amt_oktas,
            m.syn_present_weather,
            m.syn_past_weather_1,
            m.syn_past_weather_2,
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
    cloud_cover_cols = ["syn_cloud_cover_oktas_m", "syn_cloud_low_cover_oktas", "syn_cloud_med_cover_oktas"]
    out["syn_total_cloud_oktas"] = out[cloud_cover_cols].max(axis=1)
    out["syn_low_cloud_present"] = (out["syn_cloud_low_cover_oktas"].fillna(0) > 0).astype(int)
    layer_cols = ["syn_cloud_layer_1_amt_oktas", "syn_cloud_layer_2_amt_oktas", "syn_cloud_layer_3_amt_oktas", "syn_cloud_layer_4_amt_oktas"]
    out["syn_multi_layer_cloud_flag"] = (out[layer_cols].notna().sum(axis=1) >= 2).astype(int)
    base_cols = ["syn_cloud_low_base_1", "syn_cloud_low_base_2", "syn_cloud_low_base_3", "syn_cloud_med_base_1", "syn_cloud_med_base_2", "syn_cloud_high_base_1", "syn_cloud_high_base_2"]
    out["syn_cloud_base_min"] = out[base_cols].min(axis=1)
    height_cols = ["syn_cloud_layer_1_height_m", "syn_cloud_layer_2_height_m", "syn_cloud_layer_3_height_m", "syn_cloud_layer_4_height_m"]
    out["syn_cloud_depth_proxy"] = out[height_cols].max(axis=1) - out[height_cols].min(axis=1)
    out["syn_weather_cloud_rain_proxy"] = out["syn_present_weather"].fillna(0) + out["syn_past_weather_1"].fillna(0) + out["syn_past_weather_2"].fillna(0)
    out["target_delta_60m"] = out[TARGET_COL] - out["ghi_now"]
    out["target_kt_60m"] = out[TARGET_COL] / np.maximum(out["clear_sky_ghi_target_60m"], 20.0)
    out["solar_segment"] = pd.cut(out["solar_elev_deg"], bins=[-90, 15, 35, 90], labels=["low", "medium", "high"])
    out["target_ghi_bin"] = pd.cut(out[TARGET_COL], bins=[0, 100, 300, 600, 900, 1400], labels=["0-100", "100-300", "300-600", "600-900", "900+"])
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


def make_model(name):
    if name == "direct_regularized":
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=2600, learning_rate=0.025, num_leaves=31, min_child_samples=90, subsample=0.88, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.25, reg_lambda=3.0, random_state=RANDOM_STATE, n_jobs=-1, force_col_wise=True, verbosity=-1)
    elif name == "direct_deeper":
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=2600, learning_rate=0.025, num_leaves=63, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.88, reg_alpha=0.1, reg_lambda=1.5, random_state=RANDOM_STATE, n_jobs=-1, force_col_wise=True, verbosity=-1)
    elif name == "direct_smooth":
        reg = lgb.LGBMRegressor(objective="regression_l1", n_estimators=2200, learning_rate=0.03, num_leaves=31, min_child_samples=80, subsample=0.9, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, force_col_wise=True, verbosity=-1)
    else:
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=2400, learning_rate=0.03, num_leaves=47, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.88, reg_alpha=0.1, reg_lambda=1.5, random_state=RANDOM_STATE, n_jobs=-1, force_col_wise=True, verbosity=-1)
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])


def metric_row(y, pred, model_name, persistence_rmse):
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    mbe = float(np.mean(pred - y))
    skill = 1.0 - rmse / persistence_rmse if persistence_rmse > 0 else np.nan
    if model_name == "persistence":
        skill = 0.0
    return {"model": model_name, "n_rows": len(y), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def eval_preds(y, pred_dict):
    persistence_rmse = float(np.sqrt(mean_squared_error(y, pred_dict["persistence"])))
    return pd.DataFrame([metric_row(y, pred, name, persistence_rmse) for name, pred in pred_dict.items()])


def segment_metrics(pred_df, model_col):
    rows = []
    for seg_col in ["hour", "solar_segment", "target_ghi_bin", "month", "has_clp", "has_synop"]:
        for val, group in pred_df.groupby(seg_col, dropna=False):
            if len(group) < 20:
                continue
            persistence_rmse = float(np.sqrt(mean_squared_error(group[TARGET_COL], group["persistence"])))
            row = metric_row(group[TARGET_COL].values, group[model_col].values, model_col, persistence_rmse)
            row["segment_col"] = seg_col
            row["segment_val"] = str(val)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading enhanced direct-60m dataset...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_TS_COL] = pd.to_datetime(df[TARGET_TS_COL])
    df = add_engineered(df)

    train_mask, valid_mask, test_mask = split_masks(df)
    print("Rows loaded: " + str(len(df)))
    print("Direct 60m ready rows train " + str(int(train_mask.sum())) + " valid " + str(int(valid_mask.sum())) + " test " + str(int(test_mask.sum())))

    model_specs = {
        "direct_all_features": ALL_FEATURES,
        "direct_no_cloud_features": NO_CLOUD_FEATURES,
        "direct_rad_met_features": RAD_MET_FEATURES,
        "direct_regularized_all": ALL_FEATURES,
        "direct_deeper_all": ALL_FEATURES,
        "direct_smooth_all": ALL_FEATURES,
    }
    model_kind = {
        "direct_all_features": "direct_base",
        "direct_no_cloud_features": "direct_base",
        "direct_rad_met_features": "direct_base",
        "direct_regularized_all": "direct_regularized",
        "direct_deeper_all": "direct_deeper",
        "direct_smooth_all": "direct_smooth",
    }

    pred_test = df.loc[test_mask, [TIME_COL, TARGET_TS_COL, TARGET_COL, "ghi_now", "hour", "month", "solar_elev_deg", "solar_segment", "target_ghi_bin", "has_clp", "has_synop", "clp_qc_status", "synop_qc_status"]].copy()
    y_test = pred_test[TARGET_COL].values
    pred_test["persistence"] = clip_ghi(pred_test["ghi_now"].values)
    metrics_parts = []
    importance_parts = []

    for model_name, features in tqdm(model_specs.items(), desc="Training direct models"):
        kind = model_kind[model_name]
        model = make_model(kind)
        model.fit(
            df.loc[train_mask, features], df.loc[train_mask, TARGET_COL],
            model__eval_set=[(df.loc[valid_mask, features], df.loc[valid_mask, TARGET_COL])],
            model__eval_metric="rmse",
            model__callbacks=[lgb.early_stopping(120, verbose=False)]
        )
        pred = clip_ghi(model.predict(df.loc[test_mask, features]))
        pred_test[model_name] = pred
        train_pred = clip_ghi(model.predict(df.loc[train_mask, features]))
        valid_pred = clip_ghi(model.predict(df.loc[valid_mask, features]))
        for split_name, mask, split_pred in [("train", train_mask, train_pred), ("valid", valid_mask, valid_pred), ("test", test_mask, pred)]:
            split_df = df.loc[mask]
            pers = clip_ghi(split_df["ghi_now"].values)
            met = eval_preds(split_df[TARGET_COL].values, {"persistence": pers, model_name: split_pred})
            met["split"] = split_name
            metrics_parts.append(met)
        importance = pd.DataFrame({"model": model_name, "feature": features, "importance": model.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
        importance_parts.append(importance)
        joblib.dump({"pipeline": model, "features": features, "target": TARGET_COL}, MODEL_DIR / (model_name + ".joblib"))

    # top-N direct models based on the strongest all-feature model importance
    importance_all = pd.concat(importance_parts, ignore_index=True)
    base_imp = importance_all[importance_all["model"] == "direct_all_features"].sort_values("importance", ascending=False)
    for top_n in [20, 40, 60]:
        features = base_imp.head(top_n)["feature"].tolist()
        model_name = "direct_top" + str(top_n)
        model = make_model("direct_regularized")
        model.fit(
            df.loc[train_mask, features], df.loc[train_mask, TARGET_COL],
            model__eval_set=[(df.loc[valid_mask, features], df.loc[valid_mask, TARGET_COL])],
            model__eval_metric="rmse",
            model__callbacks=[lgb.early_stopping(120, verbose=False)]
        )
        pred = clip_ghi(model.predict(df.loc[test_mask, features]))
        pred_test[model_name] = pred
        for split_name, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
            split_df = df.loc[mask]
            pers = clip_ghi(split_df["ghi_now"].values)
            split_pred = clip_ghi(model.predict(split_df[features]))
            met = eval_preds(split_df[TARGET_COL].values, {"persistence": pers, model_name: split_pred})
            met["split"] = split_name
            metrics_parts.append(met)
        importance = pd.DataFrame({"model": model_name, "feature": features, "importance": model.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
        importance_parts.append(importance)
        joblib.dump({"pipeline": model, "features": features, "target": TARGET_COL}, MODEL_DIR / (model_name + ".joblib"))

    # simple ensembles among direct models only
    candidate_cols = [c for c in pred_test.columns if c.startswith("direct_")]
    if len(candidate_cols) >= 2:
        pred_test["ensemble_direct_mean"] = clip_ghi(pred_test[candidate_cols].mean(axis=1).values)
        best_two = []
        for col in candidate_cols:
            rmse = np.sqrt(mean_squared_error(y_test, pred_test[col].values))
            best_two.append((rmse, col))
        best_two = [x[1] for x in sorted(best_two)[:2]]
        pred_test["ensemble_best2_mean"] = clip_ghi(pred_test[best_two].mean(axis=1).values)

    all_pred_cols = ["persistence"] + [c for c in pred_test.columns if c.startswith("direct_") or c.startswith("ensemble_")]
    test_metrics = eval_preds(y_test, {col: pred_test[col].values for col in all_pred_cols})
    test_metrics["split"] = "test_summary"
    metrics_parts.append(test_metrics)
    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    importance_df = pd.concat(importance_parts, ignore_index=True)

    best_model = test_metrics.sort_values("rmse").iloc[0]["model"]
    seg_df = segment_metrics(pred_test, best_model)
    pred_test["best_model_prediction"] = pred_test[best_model]
    pred_test["best_model_error"] = pred_test[best_model] - pred_test[TARGET_COL]
    pred_test["persistence_error"] = pred_test["persistence"] - pred_test[TARGET_COL]
    worst_df = pred_test.reindex(pred_test["best_model_error"].abs().sort_values(ascending=False).index).head(300)

    metrics_path = OUTPUT_DIR / "metrics.csv"
    seg_path = OUTPUT_DIR / "metrics_by_segment.csv"
    imp_path = OUTPUT_DIR / "feature_importance.csv"
    pred_path = OUTPUT_DIR / "predictions_test.csv"
    worst_path = OUTPUT_DIR / "worst_errors.csv"
    plot_path = OUTPUT_DIR / "diagnostics.png"
    metrics_df.to_csv(metrics_path, index=False)
    seg_df.to_csv(seg_path, index=False)
    importance_df.to_csv(imp_path, index=False)
    pred_test.to_csv(pred_path, index=False)
    worst_df.to_csv(worst_path, index=False)

    print("Best direct 60m test models")
    print(test_metrics.sort_values("rmse").head(20).to_string(index=False))
    print("Best model: " + best_model)
    print("Worst segment diagnostics for best model")
    print(seg_df.sort_values("rmse", ascending=False).head(30).to_string(index=False))

    plt.figure(figsize=(16, 11))
    plt.subplot(2, 2, 1)
    plot_metrics = test_metrics.sort_values("rmse").head(15)
    sns.barplot(data=plot_metrics, y="model", x="r2")
    plt.axvline(0.9, color="red", linestyle="--", linewidth=1)
    plt.title("Direct 60m models - Test R2")
    plt.xlabel("R2")
    plt.ylabel("")

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=pred_test, x=TARGET_COL, y=best_model, hue="solar_segment", s=8, alpha=0.35)
    plt.plot([0, 1200], [0, 1200], color="black", linewidth=1)
    plt.title("Best Direct Model Prediction vs Actual")
    plt.xlabel("Actual GHI t+60m")
    plt.ylabel("Predicted")

    plt.subplot(2, 2, 3)
    sns.kdeplot(pred_test["best_model_error"], label="best direct")
    sns.kdeplot(pred_test["persistence_error"], label="persistence")
    plt.title("Error distribution")
    plt.xlabel("Prediction error")
    plt.legend()

    plt.subplot(2, 2, 4)
    top_imp = importance_df[importance_df["model"] == "direct_all_features"].sort_values("importance", ascending=False).head(20).sort_values("importance")
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.title("Top feature importance - direct_all_features")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("Saved outputs:")
    for p in [metrics_path, seg_path, imp_path, pred_path, worst_path, plot_path]:
        print(str(p))
    print("Models saved under: " + str(MODEL_DIR))


if __name__ == "__main__":
    main()
