#!/usr/bin/env python3
"""
V11 Bengkulu GHI forecasting - "hybrid resolution" target framing ported from
Training_Banten note 09 ("Target: rata-rata GHI 60 menit ke depan, framing
hybrid 10-mnt -> rata-rata jam").

Banten's production target is NOT the instantaneous GHI value exactly at
t+60m (what v1-v10 here predict) - it is the AVERAGE of GHI over the next
hour, computed FROM 10-minute-resolution data (6 points averaged), while the
INPUT features stay at full 10-minute resolution (lags/rolling/deltas/accel
down to 10-20m). This script asks the user's exact question: does keeping
that fine-resolution variability in the inputs pay off when the prediction
target itself is a smoothed hourly average, for Bengkulu's data?

Three models are compared, all single LightGBM-residual, same train/valid/test
split as v6-v10:
    (A) reference  - existing point target (target_ghi_1h_ahead), v10's 43
                      features, re-evaluated on the row-set intersected with
                      (B)/(C) below for a fair n_rows comparison.
    (B) hybrid      - NEW target_ghi_1h_avg (mean of LEAD 1..6 x ghi_now, i.e.
                      t+10..t+60 averaged) + v10's full 43 features (includes
                      sub-60m fine lags/deltas/accel).
    (C) coarse-only - same target_ghi_1h_avg, but with the 6 sub-60-minute
                      fine-resolution features removed (COARSE_FEATURES, 37
                      features) - isolates whether fine input variability adds
                      value once the target itself is already smoothed.

IMPORTANT CAVEAT (stated explicitly, same way Banten's own note 09 frames it):
(A) vs (B)/(C) is NOT an apples-to-apples "did accuracy improve" comparison -
predicting a 6-point average is an inherently easier statistical task (lower
target variance) than predicting one noisy instantaneous point, regardless of
model skill. The R2 gain from point->average reflects target smoothing, not
forecasting skill. The genuinely informative comparison is (B) vs (C), same
target, same rows, only input resolution differs - that isolates whether 10-
minute variability is worth keeping.

Data source: runs fully OFFLINE against the local synced copy
C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb if present (no token needed); falls
back to MotherDuck (md:bengkulu, requires MOTHERDUCK_TOKEN) only if that file is
missing. Run sync_motherduck_to_local.py in DuckDB_bengkulu to refresh the local copy.

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib lightgbm pywt

Run:
    python train_ghi_1h_bengkulu_v11_hybrid_avg_target.py

Outputs:
    outputs_v11_hybrid_avg/ghi_1h_v11_metrics.csv
    outputs_v11_hybrid_avg/ghi_1h_v11_predictions_test.csv
"""

import os
from pathlib import Path
import warnings

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pywt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
OUTPUT_DIR = Path("outputs_v11_hybrid_avg")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "target_ghi_1h_ahead"
DELTA_TARGET_COL = "target_delta_ghi_1h"
KT_TARGET_COL = "target_kt_1h"
PRED_MIN = 0.0
PRED_MAX = 1400.0
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42
BAG_SEEDS = [42, 202, 777]

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

# 2nd-difference (acceleration) of GHI/kT/cloud optical thickness over 20 minutes -
# the one Banten-ported lever that empirically generalized to Bengkulu. Computed
# from lags that are already present in NEW_FINE_LAG_FEATURES (clp_cot_lag_10/20m)
# plus kt_lag_10/20m and ghi_lag_20m, so no new SQL is required.
ACCEL_FEATURES = ["accel_clp_cot_20m", "accel_kt_20m", "accel_ghi_20m"]

ALL_CANDIDATE_FEATURES = BASE_FEATURES + NEW_FINE_LAG_FEATURES + SYNOP_CLOUD_FEATURES + ENGINEERED_FEATURES + WAVELET_FEATURES + ACCEL_FEATURES

# Top-40 by combined (importance_direct + importance_residual) from the v6 run.
# See outputs_v6_stacked/feature_pruning_sweep.csv for the full K-vs-accuracy sweep.
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
assert len(PRUNED_40_FEATURES) == 40, "expected exactly 40 pruned features"
assert all(f in ALL_CANDIDATE_FEATURES for f in PRUNED_40_FEATURES), "pruned feature not found in candidate pool"

FEATURES = PRUNED_40_FEATURES + ACCEL_FEATURES
assert len(FEATURES) == 43

# Features that need < 60-minute resolution to compute (sub-hourly lags/deltas/
# acceleration). Removing these from FEATURES isolates whether fine-resolution
# input variability matters once the TARGET is already a smoothed hourly average.
FINE_RESOLUTION_FEATURES = [
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_lag_10m",
    "accel_clp_cot_20m", "accel_kt_20m", "accel_ghi_20m",
]
assert all(f in FEATURES for f in FINE_RESOLUTION_FEATURES)
COARSE_FEATURES = [f for f in FEATURES if f not in FINE_RESOLUTION_FEATURES]
assert len(COARSE_FEATURES) == 43 - len(FINE_RESOLUTION_FEATURES)

AVG_TARGET_COL = "target_ghi_1h_avg"
AVG_DELTA_TARGET_COL = "target_delta_ghi_1h_avg"
LEAD_STEPS = [1, 2, 3, 4, 5, 6]  # t+10m .. t+60m


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
    """Attach as ATTACH_ALIAS so build_sql() is identical either way. Prefers the
    local synced copy (offline, no token needed); falls back to MotherDuck if the
    local file is missing. See sync_motherduck_to_local.py in DuckDB_bengkulu to
    refresh the local copy."""
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
    # NOTE: all LAG/rolling window functions below are computed over the FULL
    # continuous joined series (no WHERE filter yet), so they are always
    # calendar-correct (e.g. kt_lag_10m really is t-10min) even at day
    # boundaries. The model-ready filter is applied LAST, after all windows.
    # An earlier version of this script computed these windows AFTER the
    # filter, which could silently pull a stale value across a multi-hour
    # overnight gap near sunrise. Fixed here.
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
            LIST(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS kt_window_180m,
            LEAD(ghi_now, 1) OVER (ORDER BY ts_wib) AS ghi_lead_10m,
            LEAD(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lead_20m,
            LEAD(ghi_now, 3) OVER (ORDER BY ts_wib) AS ghi_lead_30m,
            LEAD(ghi_now, 4) OVER (ORDER BY ts_wib) AS ghi_lead_40m,
            LEAD(ghi_now, 5) OVER (ORDER BY ts_wib) AS ghi_lead_50m,
            LEAD(ghi_now, 6) OVER (ORDER BY ts_wib) AS ghi_lead_60m
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
    out[KT_TARGET_COL] = out[TARGET_COL] / np.maximum(out["clear_sky_ghi_now"], 20.0)

    # Acceleration (2nd difference) over 20 minutes - ported from Training_Banten,
    # the one lever that empirically generalized here (see module docstring).
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2.0 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_ghi_20m"] = out["ghi_now"] - 2.0 * out["ghi_lag_10m"] + out["ghi_lag_20m"]

    # Banten-style hybrid target: average GHI over t+10..t+60 (6 future 10-min
    # points), computed from the continuous unfiltered LEAD series. A row is
    # only valid for this target if all 6 future points exist and are in-range
    # (same physical bounds as the existing point target).
    lead_cols = ["ghi_lead_10m", "ghi_lead_20m", "ghi_lead_30m", "ghi_lead_40m", "ghi_lead_50m", "ghi_lead_60m"]
    leads = out[lead_cols]
    in_range = leads.apply(lambda c: c.between(0, 1400))
    out["avg_target_valid"] = leads.notna().all(axis=1) & in_range.all(axis=1)
    out[AVG_TARGET_COL] = leads.mean(axis=1)
    out[AVG_DELTA_TARGET_COL] = out[AVG_TARGET_COL] - out["ghi_now"]
    return out


def add_wavelet_features(df, wavelet="db2", level=2, window=18):
    """Causal 2-level wavelet decomposition of the trailing 180-minute kt_point
    window (oldest..now). Only valid for rows whose window is full (kt_window_n
    == window) and has no NaNs - those should already be guaranteed by the
    has_continuous_3h_history filter applied in SQL before the windows were
    cut, but we guard explicitly anyway."""
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


def make_lgbm(kind, seed=RANDOM_STATE):
    if kind == "residual":
        params = dict(n_estimators=6000, learning_rate=0.02, num_leaves=39, min_child_samples=70,
                      reg_alpha=0.2, reg_lambda=2.5, colsample_bytree=0.82, subsample=0.85)
    elif kind == "kt":
        params = dict(n_estimators=6000, learning_rate=0.02, num_leaves=39, min_child_samples=70,
                      reg_alpha=0.2, reg_lambda=2.5, colsample_bytree=0.82, subsample=0.85)
    else:
        params = dict(n_estimators=6000, learning_rate=0.02, num_leaves=55, min_child_samples=55,
                      reg_alpha=0.1, reg_lambda=1.5, colsample_bytree=0.85, subsample=0.88)
    reg = lgb.LGBMRegressor(
        objective="regression",
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
        **params
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])


def fit_with_early_stop(pipe, x_train, y_train, x_valid, y_valid):
    pipe.fit(
        x_train, y_train,
        model__eval_set=[(x_valid, y_valid)],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return pipe


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


def fit_residual_and_eval(df, feature_list, target_col, delta_col, train_mask, valid_mask, test_mask, label):
    x_train = df.loc[train_mask, feature_list]
    x_valid = df.loc[valid_mask, feature_list]
    x_test = df.loc[test_mask, feature_list]
    yd_train = df.loc[train_mask, delta_col]
    yd_valid = df.loc[valid_mask, delta_col]
    y_test = df.loc[test_mask, target_col]

    model = make_lgbm("residual")
    fit_with_early_stop(model, x_train, yd_train, x_valid, yd_valid)
    pred_test = clip_ghi(df.loc[test_mask, "ghi_now"].values + model.predict(x_test))

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"  [{label}] n_features={len(feature_list)} best_iter={model.named_steps['model'].best_iteration_} "
          f"R2={r2:.6f} MAE={mae:.3f} RMSE={rmse:.3f}")
    return {"config": label, "n_features": len(feature_list), "n_rows": int(test_mask.sum()),
            "r2": r2, "mae": mae, "rmse": rmse}, model, pred_test


def main():
    con = connect_data()
    print("Loading base rollback table + SYNOP cloud-layer join + fine lags + future leads...")
    df = load_data(con)
    con.close()
    df = add_engineered_features(df)
    print("Computing causal wavelet decomposition of kt (180m trailing window)...")
    df = add_wavelet_features(df)
    print("Rows loaded (before avg-target validity filter): " + str(len(df)))

    n_before = len(df)
    df = df[df["avg_target_valid"]].reset_index(drop=True)
    print(f"Rows after requiring a full valid 6-point future average: {len(df)} (dropped {n_before - len(df)} "
          f"tail-of-series rows lacking a complete t+10..t+60 lookahead)")
    print("Date range: " + str(df[TIME_COL].min()) + " to " + str(df[TIME_COL].max()))
    print("Split: train<2024, 2024<=valid<2025, test>=2025")

    train_mask, valid_mask, test_mask = split_masks(df)
    print("n_train=" + str(int(train_mask.sum())) + " n_valid=" + str(int(valid_mask.sum())) + " n_test=" + str(int(test_mask.sum())))

    print("\n--- (A) reference: point target (target_ghi_1h_ahead), 43 features, same row-set ---")
    row_a, model_a, pred_a = fit_residual_and_eval(
        df, FEATURES, TARGET_COL, DELTA_TARGET_COL, train_mask, valid_mask, test_mask, "A_point_43feat")

    print("\n--- (B) hybrid: avg target (target_ghi_1h_avg), 43 features (fine+coarse) ---")
    row_b, model_b, pred_b = fit_residual_and_eval(
        df, FEATURES, AVG_TARGET_COL, AVG_DELTA_TARGET_COL, train_mask, valid_mask, test_mask, "B_avg_43feat_hybrid")

    print("\n--- (C) coarse-only: avg target (target_ghi_1h_avg), 37 features (sub-60m removed) ---")
    row_c, model_c, pred_c = fit_residual_and_eval(
        df, COARSE_FEATURES, AVG_TARGET_COL, AVG_DELTA_TARGET_COL, train_mask, valid_mask, test_mask, "C_avg_37feat_coarse")

    summary = pd.DataFrame([row_a, row_b, row_c])
    print("\n=== SUMMARY (test 2025 holdout) ===")
    print(summary.to_string(index=False))
    print("\nNOTE: A vs B/C is NOT apples-to-apples (different prediction target - averaging")
    print("inherently lowers target variance). The informative comparison is B vs C: same")
    print("target_ghi_1h_avg, same rows, only input resolution differs.")
    delta_bc = row_b["r2"] - row_c["r2"]
    if delta_bc > 0.001:
        print(f"-> B beats C by {delta_bc:+.4f} R2: fine-resolution (10-20m) input variability DOES help "
              f"once the target is a smoothed hourly average.")
    elif delta_bc < -0.001:
        print(f"-> C beats B by {-delta_bc:+.4f} R2: fine-resolution input features are NOT needed "
              f"(possibly add noise) once the target is already smoothed.")
    else:
        print(f"-> B and C are statistically indistinguishable (delta={delta_bc:+.4f}): fine-resolution "
              f"input variability does not add value once the target is smoothed.")

    summary.to_csv(OUTPUT_DIR / "ghi_1h_v11_metrics.csv", index=False)

    pred_out = df.loc[test_mask, [TIME_COL, "target_ts_wib", TARGET_COL, AVG_TARGET_COL, "ghi_now", "solar_elev_deg", "clp_cot"]].copy()
    pred_out["pred_A_point_43feat"] = pred_a
    pred_out["pred_B_avg_43feat_hybrid"] = pred_b
    pred_out["pred_C_avg_37feat_coarse"] = pred_c
    pred_out.to_csv(OUTPUT_DIR / "ghi_1h_v11_predictions_test.csv", index=False)

    print("\nSaved outputs under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
