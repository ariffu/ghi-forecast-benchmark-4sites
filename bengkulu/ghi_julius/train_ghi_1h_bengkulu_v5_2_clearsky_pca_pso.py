#!/usr/bin/env python3
"""
V5.2 Bengkulu GHI forecasting benchmark.

Focus:
    1. 2021-2025 only, 2026 excluded.
    2. Multi-horizon benchmark: 10m, 30m, 60m ahead.
    3. Clear-sky-index target using solar elevation proxy.
    4. Adds SYNOP cloud properties from synop_bengkulu_quality_final.
    5. Tests feature reduction:
        - LightGBM top-feature selection
        - PCA on numeric features
        - Optional lightweight PSO-style binary feature selection on top features

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow lightgbm tqdm

Run:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal
    python train_ghi_1h_bengkulu_v5_clearsky_pca_pso.py

Outputs:
    outputs_v5_2_clearsky/ghi_v5_metrics.csv
    outputs_v5_2_clearsky/ghi_v5_best_by_horizon.csv
    outputs_v5_2_clearsky/ghi_v5_predictions_test.csv
    outputs_v5_2_clearsky/ghi_v5_feature_importance.csv
    outputs_v5_2_clearsky/ghi_v5_diagnostics.png
    outputs_v5_2_clearsky/models/*.joblib
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
OUTPUT_DIR = Path("outputs_v5_2_clearsky")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL_TEMPLATE = "target_ghi_{horizon}m"
PRED_MIN = 0.0
PRED_MAX = 1400.0
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42
HORIZONS = [10, 30, 60]
RUN_PSO = False
PSO_TOP_N = 45
PSO_PARTICLES = 18
PSO_ITERATIONS = 18

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
    "solar_elev_sin", "solar_elev_sin_clip", "clear_sky_ghi_now", "kt_now",
    "ghi_to_aws_sr_ratio", "dhi_fraction", "dni_fraction", "diffuse_to_global_ratio",
    "temp_rh_interaction", "vpd_proxy", "wind_u", "wind_v",
    "clp_cot_delta_60m", "clp_cth_delta_60m", "clp_cot_delta_180m", "clp_cth_delta_180m",
    "ghi_roll_180m_range", "ghi_roll_60m_range", "ghi_ramp_ratio_60m", "aws_temp_range",
    "cloud_opacity_proxy", "cloud_height_temp_interaction",
    "syn_total_cloud_oktas", "syn_low_cloud_present", "syn_multi_layer_cloud_flag", "syn_cloud_base_min",
    "syn_cloud_depth_proxy", "syn_weather_cloud_rain_proxy"
]

FEATURES = BASE_FEATURES + SYNOP_CLOUD_FEATURES + ENGINEERED_FEATURES


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


def safe_div(num, den, floor=1e-6):
    return num / np.maximum(np.abs(den), floor)


def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN environment variable.")
    os.environ["motherduck_token"] = token
    return token


def connect_motherduck():
    require_token()
    con = duckdb.connect(database=":memory:")
    con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
    return con


def build_sql():
    horizon_selects = []
    for horizon in HORIZONS:
        steps = int(horizon / 10)
        horizon_selects.append("LEAD(m.ts_wib, " + str(steps) + ") OVER (ORDER BY m.ts_wib) AS observed_target_ts_" + str(horizon) + "m")
        horizon_selects.append("m.ts_wib + INTERVAL '" + str(horizon) + " minutes' AS target_ts_" + str(horizon) + "m")
        horizon_selects.append("LEAD(m.asrs_ghi_w_m2, " + str(steps) + ") OVER (ORDER BY m.ts_wib) AS target_ghi_" + str(horizon) + "m")
        horizon_selects.append("LEAD(m.asrs_solar_elev_deg, " + str(steps) + ") OVER (ORDER BY m.ts_wib) AS target_solar_elev_deg_" + str(horizon) + "m")
    horizon_sql = ",\n        ".join(horizon_selects)
    sql_text = """
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
    ), with_targets AS (
        SELECT
            m.ts_wib,
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
            m.aws_rh_pct - LAG(m.aws_rh_pct, 6) OVER (ORDER BY m.ts_wib) AS aws_rh_delta_60m,
            """ + horizon_sql + """
        FROM master m
    )
    SELECT *
    FROM with_targets
    WHERE ts_wib >= TIMESTAMP '2021-01-01'
      AND ts_wib < TIMESTAMP '2026-01-01'
    """
    return sql_text


def load_data(con):
    df = con.execute(build_sql()).fetchdf()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    for horizon in HORIZONS:
        df["target_ts_" + str(horizon) + "m"] = pd.to_datetime(df["target_ts_" + str(horizon) + "m"])
        df["observed_target_ts_" + str(horizon) + "m"] = pd.to_datetime(df["observed_target_ts_" + str(horizon) + "m"])
    return df


def add_engineered_features(df):
    out = df.copy()
    elev_rad = np.deg2rad(out["solar_elev_deg"].astype(float))
    elev_sin = np.sin(elev_rad)
    out["solar_elev_sin"] = elev_sin
    out["solar_elev_sin_clip"] = np.maximum(elev_sin, 0.02)
    out["clear_sky_ghi_now"] = 1100.0 * out["solar_elev_sin_clip"]
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
    syn_cloud_cols = ["syn_cloud_cover_oktas_m", "syn_cloud_low_cover_oktas", "syn_cloud_med_cover_oktas"]
    out["syn_total_cloud_oktas"] = out[syn_cloud_cols].max(axis=1)
    out["syn_low_cloud_present"] = (out["syn_cloud_low_cover_oktas"].fillna(0) > 0).astype(int)
    layer_cols = ["syn_cloud_layer_1_amt_oktas", "syn_cloud_layer_2_amt_oktas", "syn_cloud_layer_3_amt_oktas", "syn_cloud_layer_4_amt_oktas"]
    out["syn_multi_layer_cloud_flag"] = (out[layer_cols].notna().sum(axis=1) >= 2).astype(int)
    base_cols = ["syn_cloud_low_base_1", "syn_cloud_low_base_2", "syn_cloud_low_base_3", "syn_cloud_med_base_1", "syn_cloud_med_base_2", "syn_cloud_high_base_1", "syn_cloud_high_base_2"]
    out["syn_cloud_base_min"] = out[base_cols].min(axis=1)
    height_cols = ["syn_cloud_layer_1_height_m", "syn_cloud_layer_2_height_m", "syn_cloud_layer_3_height_m", "syn_cloud_layer_4_height_m"]
    out["syn_cloud_depth_proxy"] = out[height_cols].max(axis=1) - out[height_cols].min(axis=1)
    out["syn_weather_cloud_rain_proxy"] = out["syn_present_weather"].fillna(0) + out["syn_past_weather_1"].fillna(0) + out["syn_past_weather_2"].fillna(0)
    for horizon in HORIZONS:
        target_elev = out["target_solar_elev_deg_" + str(horizon) + "m"]
        target_sin = np.maximum(np.sin(np.deg2rad(target_elev.astype(float))), 0.02)
        out["clear_sky_ghi_target_" + str(horizon) + "m"] = 1100.0 * target_sin
        out["target_kt_" + str(horizon) + "m"] = out["target_ghi_" + str(horizon) + "m"] / np.maximum(out["clear_sky_ghi_target_" + str(horizon) + "m"], 20.0)
        out["target_delta_" + str(horizon) + "m"] = out["target_ghi_" + str(horizon) + "m"] - out["ghi_now"]
    return out


def horizon_ready_mask(df, horizon):
    return (
        (df["observed_target_ts_" + str(horizon) + "m"] == df["target_ts_" + str(horizon) + "m"]) &
        (df["target_ts_" + str(horizon) + "m"] < pd.Timestamp("2026-01-01")) &
        (df["target_ghi_" + str(horizon) + "m"].between(0, 1400)) &
        (df["ghi_now"].between(0, 1400)) &
        (df["daylight_flag"] == 1) &
        (df["target_solar_elev_deg_" + str(horizon) + "m"] > 0) &
        (df["has_asrs"] == True) &
        (df["has_aws"] == True) &
        (df["asrs_qc_status"].isin(["ok", "warn_partial_asrs_qc"])) &
        (df["aws_qc_status"] == "ok") &
        (df["ghi_lag_180m"].notna()) &
        (df["ghi_roll_180m_mean"].notna())
    )


def split_masks(df, mask):
    train = mask & (df[TIME_COL] < pd.Timestamp(TRAIN_END))
    valid = mask & (df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))
    test = mask & (df[TIME_COL] >= pd.Timestamp(VALID_END))
    return train, valid, test


def make_lgbm(kind):
    if kind == "kt":
        params = dict(n_estimators=2200, learning_rate=0.025, num_leaves=31, min_child_samples=70, reg_alpha=0.15, reg_lambda=2.5)
    elif kind == "residual":
        params = dict(n_estimators=2200, learning_rate=0.025, num_leaves=31, min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5)
    else:
        params = dict(n_estimators=2200, learning_rate=0.03, num_leaves=47, min_child_samples=60, reg_alpha=0.1, reg_lambda=1.5)
    reg = lgb.LGBMRegressor(
        objective="regression",
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.88,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
        **params
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])


def make_pca_model(kind, n_components=0.95):
    model = make_lgbm(kind).named_steps["model"]
    model.set_params(n_estimators=900)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("model", model)
    ])


def metric_row(y_true, y_pred, model_name, horizon, persistence_rmse=None):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    mbe = float(np.mean(y_pred - y_true))
    skill = 0.0 if model_name == "persistence" else np.nan
    if persistence_rmse and persistence_rmse > 0:
        skill = 1.0 - rmse / persistence_rmse
    return {"horizon_min": horizon, "model": model_name, "n_rows": len(y_true), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def evaluate(y_true, pred_dict, horizon):
    rows = []
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, pred_dict["persistence"])))
    for key, val in pred_dict.items():
        rows.append(metric_row(y_true, val, key, horizon, persistence_rmse))
    return pd.DataFrame(rows)


def top_features_from_importance(model, features, top_n=45):
    imp = model.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
    return imp_df.head(min(top_n, len(imp_df)))["feature"].tolist(), imp_df


def pso_select_features(df, train_mask, valid_mask, candidate_features, target_col):
    rng = np.random.default_rng(RANDOM_STATE)
    n_feat = len(candidate_features)
    positions = rng.random((PSO_PARTICLES, n_feat)) > 0.5
    velocities = rng.normal(0, 0.2, size=(PSO_PARTICLES, n_feat))
    personal_best = positions.copy()
    personal_scores = np.full(PSO_PARTICLES, np.inf)
    global_best = positions[0].copy()
    global_score = np.inf

    x_train_all = df.loc[train_mask, candidate_features]
    y_train = df.loc[train_mask, target_col]
    x_valid_all = df.loc[valid_mask, candidate_features]
    y_valid = df.loc[valid_mask, target_col]

    def score(mask):
        if mask.sum() < 5:
            return np.inf
        feats = [candidate_features[idx] for idx in range(n_feat) if mask[idx]]
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=350, learning_rate=0.05, num_leaves=15, min_child_samples=80, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, force_col_wise=True)
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])
        pipe.fit(x_train_all[feats], y_train)
        pred = pipe.predict(x_valid_all[feats])
        penalty = 0.002 * mask.sum()
        return float(np.sqrt(mean_squared_error(y_valid, pred))) + penalty

    for _ in tqdm(range(PSO_ITERATIONS), desc="PSO feature selection"):
        for p_idx in range(PSO_PARTICLES):
            current_score = score(positions[p_idx])
            if current_score < personal_scores[p_idx]:
                personal_scores[p_idx] = current_score
                personal_best[p_idx] = positions[p_idx].copy()
            if current_score < global_score:
                global_score = current_score
                global_best = positions[p_idx].copy()
        r1 = rng.random((PSO_PARTICLES, n_feat))
        r2 = rng.random((PSO_PARTICLES, n_feat))
        velocities = 0.65 * velocities + 1.2 * r1 * (personal_best.astype(float) - positions.astype(float)) + 1.2 * r2 * (global_best.astype(float) - positions.astype(float))
        probs = 1.0 / (1.0 + np.exp(-velocities))
        positions = rng.random((PSO_PARTICLES, n_feat)) < probs
    selected = [candidate_features[idx] for idx in range(n_feat) if global_best[idx]]
    if len(selected) < 5:
        selected = candidate_features[:min(15, len(candidate_features))]
    return selected, global_score


def train_horizon(df, horizon):
    mask = horizon_ready_mask(df, horizon)
    train_mask, valid_mask, test_mask = split_masks(df, mask)
    print("horizon " + str(horizon) + " rows train " + str(int(train_mask.sum())) + " valid " + str(int(valid_mask.sum())) + " test " + str(int(test_mask.sum())))
    target_ghi = "target_ghi_" + str(horizon) + "m"
    target_delta = "target_delta_" + str(horizon) + "m"
    target_kt = "target_kt_" + str(horizon) + "m"
    clear_target = "clear_sky_ghi_target_" + str(horizon) + "m"

    direct = make_lgbm("direct")
    residual = make_lgbm("residual")
    kt_model = make_lgbm("kt")

    direct.fit(df.loc[train_mask, FEATURES], df.loc[train_mask, target_ghi], model__eval_set=[(df.loc[valid_mask, FEATURES], df.loc[valid_mask, target_ghi])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(100, verbose=False)])
    residual.fit(df.loc[train_mask, FEATURES], df.loc[train_mask, target_delta], model__eval_set=[(df.loc[valid_mask, FEATURES], df.loc[valid_mask, target_delta])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(100, verbose=False)])
    kt_model.fit(df.loc[train_mask, FEATURES], df.loc[train_mask, target_kt], model__eval_set=[(df.loc[valid_mask, FEATURES], df.loc[valid_mask, target_kt])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(100, verbose=False)])

    top_feats, importance_df = top_features_from_importance(kt_model, FEATURES, top_n=45)
    kt_top = make_lgbm("kt")
    kt_top.fit(df.loc[train_mask, top_feats], df.loc[train_mask, target_kt], model__eval_set=[(df.loc[valid_mask, top_feats], df.loc[valid_mask, target_kt])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(100, verbose=False)])

    kt_pca = make_pca_model("kt", n_components=0.95)
    kt_pca.fit(df.loc[train_mask, FEATURES], df.loc[train_mask, target_kt])

    pso_model = None
    pso_feats = []
    pso_score = np.nan
    if RUN_PSO:
        pso_candidates = top_feats[:min(PSO_TOP_N, len(top_feats))]
        pso_feats, pso_score = pso_select_features(df, train_mask, valid_mask, pso_candidates, target_kt)
        pso_model = make_lgbm("kt")
        pso_model.fit(df.loc[train_mask, pso_feats], df.loc[train_mask, target_kt], model__eval_set=[(df.loc[valid_mask, pso_feats], df.loc[valid_mask, target_kt])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(100, verbose=False)])

    test_df = df.loc[test_mask].copy()
    y = test_df[target_ghi].values
    persistence = clip_ghi(test_df["ghi_now"].values)
    pred_direct = clip_ghi(direct.predict(test_df[FEATURES]))
    pred_residual = clip_ghi(test_df["ghi_now"].values + residual.predict(test_df[FEATURES]))
    pred_kt = clip_ghi(kt_model.predict(test_df[FEATURES]) * test_df[clear_target].values)
    pred_kt_top = clip_ghi(kt_top.predict(test_df[top_feats]) * test_df[clear_target].values)
    pred_kt_pca = clip_ghi(kt_pca.predict(test_df[FEATURES]) * test_df[clear_target].values)
    pred_dict = {
        "persistence": persistence,
        "lgbm_direct_ghi": pred_direct,
        "lgbm_residual_ghi": pred_residual,
        "lgbm_clearsky_kt": pred_kt,
        "lgbm_clearsky_kt_top45": pred_kt_top,
        "lgbm_clearsky_kt_pca95": pred_kt_pca,
        "blend_50_residual_50_kt": clip_ghi(0.5 * pred_residual + 0.5 * pred_kt),
        "blend_50_direct_50_kt": clip_ghi(0.5 * pred_direct + 0.5 * pred_kt),
        "blend_50_persistence_50_kt": clip_ghi(0.5 * persistence + 0.5 * pred_kt),
    }
    if pso_model is not None:
        pred_dict["lgbm_clearsky_kt_pso"] = clip_ghi(pso_model.predict(test_df[pso_feats]) * test_df[clear_target].values)

    metrics = evaluate(y, pred_dict, horizon)
    metrics["n_train"] = int(train_mask.sum())
    metrics["n_valid"] = int(valid_mask.sum())
    metrics["n_test"] = int(test_mask.sum())
    metrics["pso_score"] = pso_score
    metrics["pso_n_features"] = len(pso_feats)

    pred_out = test_df[[TIME_COL, "target_ts_" + str(horizon) + "m", target_ghi, "ghi_now", "solar_elev_deg", clear_target, "has_clp", "has_synop"]].copy()
    pred_out["horizon_min"] = horizon
    for key, val in pred_dict.items():
        pred_out[key] = val
    importance_df["horizon_min"] = horizon
    importance_df["pso_selected"] = importance_df["feature"].isin(pso_feats) if pso_feats else False

    joblib.dump({"pipeline": direct, "features": FEATURES, "horizon": horizon}, MODEL_DIR / ("h" + str(horizon) + "_direct_ghi.joblib"))
    joblib.dump({"pipeline": residual, "features": FEATURES, "horizon": horizon}, MODEL_DIR / ("h" + str(horizon) + "_residual_ghi.joblib"))
    joblib.dump({"pipeline": kt_model, "features": FEATURES, "horizon": horizon, "target": "kt"}, MODEL_DIR / ("h" + str(horizon) + "_clearsky_kt.joblib"))
    joblib.dump({"pipeline": kt_top, "features": top_feats, "horizon": horizon, "target": "kt"}, MODEL_DIR / ("h" + str(horizon) + "_clearsky_kt_top45.joblib"))
    joblib.dump({"pipeline": kt_pca, "features": FEATURES, "horizon": horizon, "target": "kt"}, MODEL_DIR / ("h" + str(horizon) + "_clearsky_kt_pca95.joblib"))
    if pso_model is not None:
        joblib.dump({"pipeline": pso_model, "features": pso_feats, "horizon": horizon, "target": "kt"}, MODEL_DIR / ("h" + str(horizon) + "_clearsky_kt_pso.joblib"))
    return metrics, pred_out, importance_df


def main():
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading enhanced master data with SYNOP cloud properties...")
    print("V5.2 patch: cyclic time features are derived; PCA model is fitted without eval_set because eval_set bypasses sklearn PCA transforms.")
    df = load_data(con)
    con.close()
    df = add_engineered_features(df)
    print("Rows loaded: " + str(len(df)))
    print("Date range: " + str(df[TIME_COL].min()) + " to " + str(df[TIME_COL].max()))
    print("Split policy: train=2021-2023, validation=2024, test=2025; 2026 excluded.")
    print("PSO enabled: " + str(RUN_PSO))

    metric_parts = []
    pred_parts = []
    imp_parts = []
    for horizon in HORIZONS:
        metrics, preds, imps = train_horizon(df, horizon)
        metric_parts.append(metrics)
        pred_parts.append(preds)
        imp_parts.append(imps)
        print("Best for horizon " + str(horizon) + "m")
        print(metrics.sort_values("rmse").head(8).to_string(index=False))

    metrics_df = pd.concat(metric_parts, ignore_index=True)
    predictions_df = pd.concat(pred_parts, ignore_index=True)
    importance_df = pd.concat(imp_parts, ignore_index=True)
    best_df = metrics_df.sort_values(["horizon_min", "rmse"]).groupby("horizon_min").head(5).reset_index(drop=True)

    metrics_path = OUTPUT_DIR / "ghi_v5_metrics.csv"
    best_path = OUTPUT_DIR / "ghi_v5_best_by_horizon.csv"
    pred_path = OUTPUT_DIR / "ghi_v5_predictions_test.csv"
    imp_path = OUTPUT_DIR / "ghi_v5_feature_importance.csv"
    plot_path = OUTPUT_DIR / "ghi_v5_diagnostics.png"

    metrics_df.to_csv(metrics_path, index=False)
    best_df.to_csv(best_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    importance_df.to_csv(imp_path, index=False)

    print("Best by horizon")
    print(best_df.to_string(index=False))

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=best_df, x="horizon_min", y="r2", hue="model")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.title("Top Models by Horizon - R2")
    plt.ylabel("R2")

    plt.subplot(2, 2, 2)
    sns.barplot(data=best_df, x="horizon_min", y="rmse", hue="model")
    plt.title("Top Models by Horizon - RMSE")
    plt.ylabel("RMSE")

    plt.subplot(2, 2, 3)
    for horizon in HORIZONS:
        subset = metrics_df[(metrics_df["horizon_min"] == horizon)].sort_values("rmse").head(4)
        plt.scatter(subset["rmse"], subset["r2"], label=str(horizon) + "m", s=80)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xlabel("RMSE")
    plt.ylabel("R2")
    plt.title("RMSE vs R2")
    plt.legend()

    plt.subplot(2, 2, 4)
    top_imp = importance_df[importance_df["horizon_min"] == 60].sort_values("importance", ascending=False).head(20).sort_values("importance")
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.title("Top Feature Importance, 60m KT model")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("Saved outputs:")
    for path in [metrics_path, best_path, pred_path, imp_path, plot_path]:
        print(str(path))
    print("Models saved under: " + str(MODEL_DIR))


if __name__ == "__main__":
    main()
