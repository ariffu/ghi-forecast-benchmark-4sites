#!/usr/bin/env python3
"""
V9 Bengkulu GHI 1-hour-ahead forecasting - techniques ported from Training_Banten.

Builds on v8 (40 pruned features, R2=0.8198) and adds the two levers that the
Banten project (Documents/Balai Obsidian/Balai/Training_Banten, notes 08-09)
found to genuinely help BEYOND lag/rolling features, validated there by leave-
one-group-out ablation rather than just literature claims:

    Lever 1 - feature engineering (+0.0024 R2 in Banten):
        a. Acceleration (2nd difference) of clp_cot / kt / ghi over 20m - Banten's
           ablation found this contributes standalone (+0.0009), distinct info
           from the 1st-difference ("trend") features Bengkulu already has.
        b. "Smart persistence" as an input FEATURE (not just an evaluation
           baseline): kt_now x clearsky_GHI_at_target_time. This is NOT leakage -
           the target's solar elevation is purely deterministic astronomy (known
           in advance from the clock time alone), same principle as v5.2's
           clear_sky_ghi_target_60m. Banten ranked this feature #5 by importance.
        Banten explicitly found rolling-MEAN, EWMA, and an explicit clearsky-index
        feature to be REDUNDANT (slightly negative effect) once lags already
        exist - consistent with Bengkulu's own v6/v8 findings (kt_lag/roll,
        wavelet features all gave ~0 net benefit). Only added what Banten's own
        ablation marked as genuinely additive.

    Lever 2 - cross-family ensemble diversity (+0.0024 more R2 in Banten):
        Banten's key insight: simple averaging across DIFFERENT model families
        (tree boosters + neural) beats averaging variants of the SAME family,
        because errors are more independent across families. Bengkulu's v6/v8
        "ensemble" so far was bagging/stacking of LightGBM variants only - not
        genuinely diverse. This version adds CatBoost, XGBoost, and a small MLP
        alongside LightGBM, then compares simple-average vs the existing Ridge
        stack.

Data source: runs fully OFFLINE against the local synced copy
C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb if present (no token needed); falls
back to MotherDuck (md:bengkulu, requires MOTHERDUCK_TOKEN) only if that file is
missing.

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib lightgbm pywt catboost xgboost

Run:
    python train_ghi_1h_bengkulu_v9_banten_techniques.py

Outputs:
    outputs_v9_banten/ghi_1h_v9_metrics.csv
    outputs_v9_banten/ghi_1h_v9_predictions_test.csv
    outputs_v9_banten/ghi_1h_v9_feature_importance.csv
    outputs_v9_banten/ghi_1h_v9_diagnostics.png
    outputs_v9_banten/models/*.joblib
"""

import os
from pathlib import Path
import warnings

import catboost as cb
import duckdb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
OUTPUT_DIR = Path("outputs_v9_banten")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "target_ghi_1h_ahead"
DELTA_TARGET_COL = "target_delta_ghi_1h"
PRED_MIN = 0.0
PRED_MAX = 1400.0
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42

# Bengkulu city approx coordinates (used only for the deterministic smart-persistence
# clearsky proxy at the target time - no measured data, no leakage).
STATION_LAT_DEG = -3.8667
STATION_LON_DEG = 102.3415
WIB_MERIDIAN_DEG = 105.0

np.random.seed(RANDOM_STATE)

# --- v8's 40 pruned features, unchanged ---
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
    "ghi_delta_10m", "ghi_delta_60m", "aws_temp_delta_60m", "aws_rh_delta_60m",
]

NEW_FINE_LAG_FEATURES = [
    "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cth_lag_30m",
    "ghi_lag_20m", "ghi_lag_50m",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std",
]

WAVELET_FEATURES = [
    "kt_wavelet_trend", "kt_wavelet_detail1_energy", "kt_wavelet_detail2_energy",
]

SYNOP_CLOUD_FEATURES = [
    "syn_cloud_cover_oktas_m", "syn_cloud_low_cover_oktas", "syn_cloud_med_cover_oktas",
    "syn_cloud_low_base_1", "syn_cloud_low_base_2", "syn_cloud_low_base_3",
    "syn_cloud_med_base_1", "syn_cloud_med_base_2", "syn_cloud_high_base_1", "syn_cloud_high_base_2",
    "syn_cloud_layer_1_height_m", "syn_cloud_layer_2_height_m", "syn_cloud_layer_3_height_m", "syn_cloud_layer_4_height_m",
    "syn_cloud_layer_1_amt_oktas", "syn_cloud_layer_2_amt_oktas", "syn_cloud_layer_3_amt_oktas", "syn_cloud_layer_4_amt_oktas",
    "syn_present_weather", "syn_past_weather_1", "syn_past_weather_2",
]

ENGINEERED_FEATURES = [
    "solar_elev_sin", "solar_elev_sin_clip", "clear_sky_ghi_now", "kt_now",
    "ghi_to_aws_sr_ratio", "dhi_fraction", "dni_fraction",
    "temp_rh_interaction", "vpd_proxy", "wind_u", "wind_v",
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cth_delta_60m",
    "clp_cot_delta_180m", "clp_cth_delta_180m",
    "ghi_roll_180m_range", "ghi_roll_60m_range", "ghi_ramp_ratio_60m", "aws_temp_range",
    "cloud_opacity_proxy", "cloud_height_temp_interaction",
    "syn_total_cloud_oktas", "syn_low_cloud_present", "syn_multi_layer_cloud_flag", "syn_cloud_base_min",
    "syn_cloud_depth_proxy", "syn_weather_cloud_rain_proxy",
]

ALL_CANDIDATE_FEATURES = BASE_FEATURES + NEW_FINE_LAG_FEATURES + SYNOP_CLOUD_FEATURES + ENGINEERED_FEATURES + WAVELET_FEATURES

PRUNED_40_FEATURES = [
    "clp_cot", "kt_now", "dni_now", "hour_sin", "solar_elev_deg",
    "solar_elev_sin_clip", "month_cos", "clp_cot_delta_10m", "clp_cot_delta_180m", "clp_cer",
    "month_sin", "ghi_now", "nett_rad_now", "dni_fraction", "aws_pressure_lag_60m",
    "synop_dewpoint_c", "dhi_roll_180m_mean", "ghi_roll_180m_max", "clp_cot_delta_30m", "aws_pressure_hpa",
    "clp_cot_roll_180m_mean", "cloud_height_temp_interaction", "aws_ws_roll_180m_mean", "clp_cot_lag_10m", "ghi_lag_180m",
    "synop_temp_c", "clp_cth_roll_180m_mean", "dni_roll_180m_mean", "ghi_roll_180m_min", "temp_rh_interaction",
    "clp_ctt_k", "clp_cth_m", "clp_cth_delta_180m", "synop_wind_dir_deg", "clp_cot_delta_60m",
    "reflected_now", "ghi_delta_60m", "aws_rh_roll_180m_mean", "aws_temp_roll_180m_mean", "ghi_roll_180m_std",
]
assert len(PRUNED_40_FEATURES) == 40

# --- NEW (Banten levers): acceleration + smart-persistence ---
BANTEN_FEATURES = [
    "accel_clp_cot_20m", "accel_kt_20m", "accel_ghi_20m", "smart_persist_60m",
]

FEATURES = PRUNED_40_FEATURES + BANTEN_FEATURES


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")


def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN environment variable.")
    os.environ["motherduck_token"] = token
    return token


def connect_data():
    con = duckdb.connect(database=":memory:")
    if LOCAL_DB_PATH.exists():
        con.execute("ATTACH '" + LOCAL_DB_PATH.as_posix() + "' AS " + ATTACH_ALIAS + " (READ_ONLY)")
        print("Data source: LOCAL  (" + str(LOCAL_DB_PATH) + ")")
    else:
        require_token()
        con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
        print("Data source: MOTHERDUCK (md:" + DB_NAME + ") - local file not found")
    return con


def build_sql():
    return """
    WITH joined AS (
        SELECT
            t.*,
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
        FROM bengkulu_db.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025 t
        LEFT JOIN bengkulu_db.bengkulu_sch.synop_bengkulu_quality_final s
          ON time_bucket(INTERVAL '1 hour', t.ts_wib) = s.ts_wib
    ), with_kt AS (
        SELECT
            *,
            ghi_now / GREATEST(1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0) AS kt_point
        FROM joined
    ), with_windows AS (
        SELECT
            *,
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            LAG(clp_cth_m, 3) OVER (ORDER BY ts_wib) AS clp_cth_lag_30m,
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
            LAG(ghi_now, 5) OVER (ORDER BY ts_wib) AS ghi_lag_50m,
            LAG(kt_point, 1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point, 2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(kt_point, 3) OVER (ORDER BY ts_wib) AS kt_lag_30m,
            LAG(kt_point, 6) OVER (ORDER BY ts_wib) AS kt_lag_60m,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std,
            COUNT(*) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS kt_window_n,
            LIST(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS kt_window_180m
        FROM with_kt
    )
    SELECT *
    FROM with_windows
    WHERE is_model_ready = 1
      AND has_continuous_3h_history = 1
      AND target_ghi_1h_ahead BETWEEN 0 AND 1400
      AND ghi_now BETWEEN 0 AND 1400
    ORDER BY ts_wib
    """


def load_data(con):
    df = con.execute(build_sql()).fetchdf()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["target_ts_wib"] = pd.to_datetime(df["target_ts_wib"])
    return df


def solar_elevation_deg(timestamps, lat_deg=STATION_LAT_DEG, lon_deg=STATION_LON_DEG):
    """Deterministic solar elevation from clock time alone (no measured data) -
    standard declination + hour-angle formula, WIB clock corrected to local solar
    time via longitude offset from the WIB=UTC+7 standard meridian (105E). Equation
    of time ignored (±15min/year wobble, negligible for a smart-persistence proxy).
    Used to project clearsky GHI at the FUTURE target time - legitimate because
    solar position is fully predictable from the clock, unlike cloud cover."""
    doy = pd.DatetimeIndex(timestamps).dayofyear.values.astype(float)
    hour_decimal = pd.DatetimeIndex(timestamps).hour.values.astype(float) + pd.DatetimeIndex(timestamps).minute.values.astype(float) / 60.0
    decl_deg = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    solar_time = hour_decimal + 4.0 * (lon_deg - WIB_MERIDIAN_DEG) / 60.0
    hour_angle_deg = (solar_time - 12.0) * 15.0
    lat_rad = np.deg2rad(lat_deg)
    decl_rad = np.deg2rad(decl_deg)
    ha_rad = np.deg2rad(hour_angle_deg)
    sin_elev = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


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
    out["temp_rh_interaction"] = out["aws_temp_c"] * out["aws_rh_pct"]
    out["vpd_proxy"] = out["aws_temp_c"] * (100.0 - out["aws_rh_pct"]) / 100.0
    wd_rad = np.deg2rad(out["aws_wd_deg"].astype(float))
    out["wind_u"] = out["aws_ws_avg"] * np.sin(wd_rad)
    out["wind_v"] = out["aws_ws_avg"] * np.cos(wd_rad)
    out["clp_cot_delta_10m"] = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"] = out["clp_cot"] - out["clp_cot_lag_30m"]
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

    out[DELTA_TARGET_COL] = out[TARGET_COL] - out["ghi_now"]

    # --- Banten Lever 1a: acceleration (2nd difference) over 20 minutes ---
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2.0 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_ghi_20m"] = out["ghi_now"] - 2.0 * out["ghi_lag_10m"] + out["ghi_lag_20m"]

    # --- Banten Lever 1b: smart persistence = kt_now projected onto the TARGET
    # time's clearsky envelope (deterministic astronomy, not measured -> no leakage) ---
    elev_target_deg = solar_elevation_deg(out["target_ts_wib"])
    clear_sky_ghi_target = 1100.0 * np.maximum(np.sin(np.deg2rad(elev_target_deg)), 0.02)
    out["smart_persist_60m"] = clip_ghi(out["kt_now"] * clear_sky_ghi_target)
    return out


def add_wavelet_features(df, wavelet="db2", level=2, window=18):
    n = len(df)
    trend = np.full(n, np.nan)
    detail1_energy = np.full(n, np.nan)
    detail2_energy = np.full(n, np.nan)
    windows = df["kt_window_180m"].values
    window_n = df["kt_window_n"].values
    for i in range(n):
        if window_n[i] < window:
            continue
        w = np.asarray(windows[i], dtype=float)
        if len(w) < window or np.isnan(w).any():
            continue
        cA2, cD2, cD1 = pywt.wavedec(w, wavelet, level=level, mode="periodization")
        trend[i] = cA2[-1]
        detail1_energy[i] = float(np.std(cD1))
        detail2_energy[i] = float(np.std(cD2))
    df["kt_wavelet_trend"] = trend
    df["kt_wavelet_detail1_energy"] = detail1_energy
    df["kt_wavelet_detail2_energy"] = detail2_energy
    return df


def split_masks(df):
    train = df[TIME_COL] < pd.Timestamp(TRAIN_END)
    valid = (df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))
    test = df[TIME_COL] >= pd.Timestamp(VALID_END)
    return train, valid, test


def make_lgbm_residual(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=6000, learning_rate=0.02, num_leaves=39,
        min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5, colsample_bytree=0.82,
        subsample=0.85, subsample_freq=1, random_state=seed, n_jobs=-1,
        force_col_wise=True, verbosity=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])


def fit_lgbm(pipe, x_train, y_train, x_valid, y_valid):
    pipe.fit(
        x_train, y_train,
        model__eval_set=[(x_valid, y_valid)],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return pipe


def make_catboost():
    return cb.CatBoostRegressor(
        iterations=4000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
        early_stopping_rounds=150,
    )


def make_xgboost():
    return xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, max_depth=7, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.5,
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=150, eval_metric="rmse",
    )


def make_mlp():
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-3, learning_rate_init=1e-3, max_iter=500,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", mlp),
    ])


def metric_row(y_true, y_pred, model_name, persistence_rmse=None):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    mbe = float(np.mean(y_pred - y_true))
    skill = 0.0 if model_name == "persistence" else np.nan
    if persistence_rmse and persistence_rmse > 0:
        skill = 1.0 - rmse / persistence_rmse
    return {"model": model_name, "n_rows": len(y_true), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def evaluate(y_true, pred_dict, split_name):
    rows = []
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, pred_dict["persistence"])))
    for key, val in pred_dict.items():
        row = metric_row(y_true, val, key, persistence_rmse)
        row["split"] = split_name
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    con = connect_data()
    print("Loading base rollback table + SYNOP cloud-layer join + fine lags...")
    df = load_data(con)
    con.close()
    df = add_engineered_features(df)
    print("Computing causal wavelet decomposition of kt (180m trailing window)...")
    df = add_wavelet_features(df)
    print("Rows loaded: " + str(len(df)))
    print("Features: " + str(len(FEATURES)) + " (40 pruned + 4 Banten-lever)")

    train_mask, valid_mask, test_mask = split_masks(df)
    print("n_train=" + str(int(train_mask.sum())) + " n_valid=" + str(int(valid_mask.sum())) + " n_test=" + str(int(test_mask.sum())))

    x_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET_COL]
    x_valid, y_valid = df.loc[valid_mask, FEATURES], df.loc[valid_mask, TARGET_COL]
    x_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET_COL]
    yd_train = df.loc[train_mask, DELTA_TARGET_COL]
    yd_valid = df.loc[valid_mask, DELTA_TARGET_COL]

    print("\n--- Training 4 diverse model families (Banten Lever 2) ---")

    print("[1/4] LightGBM residual...")
    lgbm = make_lgbm_residual()
    fit_lgbm(lgbm, x_train, yd_train, x_valid, yd_valid)
    print("  best_iteration_ =", lgbm.named_steps["model"].best_iteration_)

    print("[2/4] CatBoost (direct target)...")
    catb = make_catboost()
    catb.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    print("  best_iteration_ =", catb.get_best_iteration())

    print("[3/4] XGBoost (direct target)...")
    xgbm = make_xgboost()
    xgbm.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    print("  best_iteration_ =", xgbm.best_iteration)

    print("[4/4] MLP (direct target, small net for diversity not raw accuracy)...")
    mlp = make_mlp()
    mlp.fit(x_train, y_train)
    print("  n_iter_ =", mlp.named_steps["model"].n_iter_)

    def predict_all(x, ghi_now_vals):
        return {
            "lgbm_residual": clip_ghi(ghi_now_vals + lgbm.predict(x)),
            "catboost": clip_ghi(catb.predict(x)),
            "xgboost": clip_ghi(xgbm.predict(x)),
            "mlp": clip_ghi(mlp.predict(x)),
        }

    preds_valid = predict_all(x_valid, df.loc[valid_mask, "ghi_now"].values)
    preds_test = predict_all(x_test, df.loc[test_mask, "ghi_now"].values)

    simple_avg_valid = np.mean(list(preds_valid.values()), axis=0)
    simple_avg_test = np.mean(list(preds_test.values()), axis=0)

    # Ridge stack (same recipe as v6/v8, now fed by 4 diverse families instead of LGBM variants only)
    stack_x_valid = np.column_stack(list(preds_valid.values()) + [df.loc[valid_mask, "ghi_now"].values])
    stack_x_test = np.column_stack(list(preds_test.values()) + [df.loc[test_mask, "ghi_now"].values])
    meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    meta.fit(stack_x_valid, df.loc[valid_mask, TARGET_COL].values)
    stacked_test = clip_ghi(meta.predict(stack_x_test))
    print("\nRidge stack weights [lgbm, catboost, xgboost, mlp, ghi_now]:", np.round(meta.coef_, 3))

    pred_dict_test = {
        "persistence": clip_ghi(df.loc[test_mask, "ghi_now"].values),
        **preds_test,
        "simple_avg_4family": clip_ghi(simple_avg_test),
        "ridge_stack_4family": stacked_test,
    }
    metrics_test = evaluate(y_test.values, pred_dict_test, "test")

    metrics_path = OUTPUT_DIR / "ghi_1h_v9_metrics.csv"
    metrics_test.to_csv(metrics_path, index=False)

    print("\n=== TEST SET RESULTS (2025 holdout) ===")
    print(metrics_test.sort_values("rmse").to_string(index=False))

    pred_out = df.loc[test_mask, [TIME_COL, "target_ts_wib", TARGET_COL, "ghi_now", "solar_elev_deg", "clp_cot"]].copy()
    for key, val in pred_dict_test.items():
        pred_out[key] = val
    pred_out.to_csv(OUTPUT_DIR / "ghi_1h_v9_predictions_test.csv", index=False)

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance_lgbm": lgbm.named_steps["model"].feature_importances_,
        "importance_catboost": catb.get_feature_importance(),
        "importance_xgboost": xgbm.feature_importances_,
    }).sort_values("importance_catboost", ascending=False)
    imp.to_csv(OUTPUT_DIR / "ghi_1h_v9_feature_importance.csv", index=False)

    joblib.dump({"pipeline": lgbm, "features": FEATURES}, MODEL_DIR / "lgbm_residual.joblib")
    catb.save_model(str(MODEL_DIR / "catboost.cbm"))
    joblib.dump({"model": xgbm, "features": FEATURES}, MODEL_DIR / "xgboost.joblib")
    joblib.dump({"pipeline": mlp, "features": FEATURES}, MODEL_DIR / "mlp.joblib")
    joblib.dump({"meta": meta, "input_order": list(preds_test.keys()) + ["ghi_now"]}, MODEL_DIR / "ridge_stack_4family.joblib")

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=metrics_test, x="model", y="r2")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.title("Test R2 by model (2025 holdout)")

    plt.subplot(2, 2, 2)
    sns.barplot(data=metrics_test, x="model", y="mae")
    plt.xticks(rotation=30, ha="right")
    plt.title("Test MAE by model")

    plt.subplot(2, 2, 3)
    top_imp = imp.head(20).sort_values("importance_catboost")
    plt.barh(top_imp["feature"], top_imp["importance_catboost"])
    plt.title("Top 20 feature importance (CatBoost)")

    plt.subplot(2, 2, 4)
    sample = pred_out.sample(min(3000, len(pred_out)), random_state=RANDOM_STATE)
    plt.scatter(sample[TARGET_COL], sample["ridge_stack_4family"], s=4, alpha=0.3)
    plt.plot([0, 1400], [0, 1400], "r--", linewidth=1)
    plt.xlabel("Actual GHI (t+60m)")
    plt.ylabel("Predicted GHI (ridge_stack_4family)")
    plt.title("4-family stack: actual vs predicted (test)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ghi_1h_v9_diagnostics.png", dpi=160)
    plt.close()

    print("\nSaved outputs under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
