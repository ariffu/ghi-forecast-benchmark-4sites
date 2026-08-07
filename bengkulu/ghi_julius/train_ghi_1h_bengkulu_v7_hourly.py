#!/usr/bin/env python3
"""
V7 Bengkulu GHI 1-hour-ahead forecasting - HOURLY resolution variant of v6.

Identical pipeline, features, model, and train(2021-23)/valid(2024)/test(2025) split
as v6 (train_ghi_1h_bengkulu_v6_stacked.py), with exactly ONE change: instead of
training on every 10-minute row and predicting target_ghi_1h_ahead (6 steps ahead in
the 10-min series), this version keeps only the row nearest to each clock hour
(minute(ts_wib) == 0) and predicts the same target_ghi_1h_ahead column, which for an
on-the-hour row is exactly 1 step ahead in the new hourly-cadence series.

All lag/rolling columns (ghi_lag_60m, ghi_roll_180m_mean, the v6 fine lags, kt_lag/roll,
wavelet features, etc.) are left untouched - they were computed on the full continuous
10-minute series before this row subset is applied, so they remain calendar-correct
(e.g. ghi_lag_60m on an hourly row still correctly means "GHI 60 minutes before this
row"). Only the ROW COUNT changes (~1/6th of v6), not the feature definitions - this
isolates the effect of "resolution/cadence" from any confound of different features.

Purpose: test whether resampling to hourly cadence (closest-to-the-hour snapshot,
matching how vault note "lstm bengkulu hourly.md" was built) improves single-step
1h-ahead accuracy vs the 10-min/6-step setup in v6.

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib lightgbm pywt

Run:
    setx MOTHERDUCK_TOKEN "your_token_here"
    python train_ghi_1h_bengkulu_v7_hourly.py

Outputs:
    outputs_v7_hourly/ghi_1h_v7_metrics.csv
    outputs_v7_hourly/ghi_1h_v7_predictions_test.csv
    outputs_v7_hourly/ghi_1h_v7_feature_importance.csv
    outputs_v7_hourly/ghi_1h_v7_diagnostics.png
    outputs_v7_hourly/models/*.joblib
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
import pywt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
OUTPUT_DIR = Path("outputs_v7_hourly")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

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

FEATURES = BASE_FEATURES + NEW_FINE_LAG_FEATURES + SYNOP_CLOUD_FEATURES + ENGINEERED_FEATURES + WAVELET_FEATURES


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


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
            LIST(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS kt_window_180m
        FROM with_kt
    )
    SELECT *
    FROM with_windows
    WHERE is_model_ready = 1
      AND has_continuous_3h_history = 1
      AND target_ghi_1h_ahead BETWEEN 0 AND 1400
      AND ghi_now BETWEEN 0 AND 1400
      AND EXTRACT(minute FROM ts_wib) = 0
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

    elev_rad_target = np.deg2rad(out["solar_elev_deg"].astype(float))  # placeholder, target elevation not in base table
    out[DELTA_TARGET_COL] = out[TARGET_COL] - out["ghi_now"]
    out[KT_TARGET_COL] = out[TARGET_COL] / np.maximum(out["clear_sky_ghi_now"], 20.0)
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


def main():
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading base rollback table + SYNOP cloud-layer join + fine lags...")
    df = load_data(con)
    con.close()
    df = add_engineered_features(df)
    print("Computing causal wavelet decomposition of kt (180m trailing window)...")
    df = add_wavelet_features(df)
    print("Rows loaded: " + str(len(df)))
    print("Date range: " + str(df[TIME_COL].min()) + " to " + str(df[TIME_COL].max()))
    print("Split: train<2024, 2024<=valid<2025, test>=2025")

    train_mask, valid_mask, test_mask = split_masks(df)
    print("n_train=" + str(int(train_mask.sum())) + " n_valid=" + str(int(valid_mask.sum())) + " n_test=" + str(int(test_mask.sum())))

    x_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET_COL]
    x_valid, y_valid = df.loc[valid_mask, FEATURES], df.loc[valid_mask, TARGET_COL]
    x_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET_COL]

    yd_train = df.loc[train_mask, DELTA_TARGET_COL]
    yd_valid = df.loc[valid_mask, DELTA_TARGET_COL]

    print("Training direct LightGBM (early stopping on valid RMSE)...")
    direct = make_lgbm("direct")
    fit_with_early_stop(direct, x_train, y_train, x_valid, y_valid)
    print("  best_iteration_ =", direct.named_steps["model"].best_iteration_)

    print("Training residual LightGBM (target = future - now)...")
    residual = make_lgbm("residual")
    fit_with_early_stop(residual, x_train, yd_train, x_valid, yd_valid)
    print("  best_iteration_ =", residual.named_steps["model"].best_iteration_)

    print("Training seed-bagged direct models (" + str(len(BAG_SEEDS)) + " seeds)...")
    bag_models = []
    for seed in BAG_SEEDS:
        m = make_lgbm("direct", seed=seed)
        fit_with_early_stop(m, x_train, y_train, x_valid, y_valid)
        bag_models.append(m)

    def bagged_predict(x):
        preds = np.mean([clip_ghi(m.predict(x)) for m in bag_models], axis=0)
        return preds

    # --- Build validation-set base predictions for stacking (no leakage: meta trained only on valid) ---
    persistence_valid = clip_ghi(df.loc[valid_mask, "ghi_now"].values)
    direct_valid = clip_ghi(direct.predict(x_valid))
    residual_valid = clip_ghi(df.loc[valid_mask, "ghi_now"].values + residual.predict(x_valid))
    bagged_valid = bagged_predict(x_valid)

    stack_x_valid = np.column_stack([
        persistence_valid, direct_valid, residual_valid, bagged_valid,
        df.loc[valid_mask, "solar_elev_deg"].fillna(0).values,
        df.loc[valid_mask, "clp_cot"].fillna(0).values,
    ])
    meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    meta.fit(stack_x_valid, df.loc[valid_mask, TARGET_COL].values)

    # --- Test set evaluation ---
    persistence_test = clip_ghi(df.loc[test_mask, "ghi_now"].values)
    direct_test = clip_ghi(direct.predict(x_test))
    residual_test = clip_ghi(df.loc[test_mask, "ghi_now"].values + residual.predict(x_test))
    bagged_test = bagged_predict(x_test)
    stack_x_test = np.column_stack([
        persistence_test, direct_test, residual_test, bagged_test,
        df.loc[test_mask, "solar_elev_deg"].fillna(0).values,
        df.loc[test_mask, "clp_cot"].fillna(0).values,
    ])
    stacked_test = clip_ghi(meta.predict(stack_x_test))

    pred_dict_test = {
        "persistence": persistence_test,
        "lgbm_direct": direct_test,
        "lgbm_residual": residual_test,
        "lgbm_direct_bagged": bagged_test,
        "stacked_meta": stacked_test,
        "blend_50_direct_50_bagged": clip_ghi(0.5 * direct_test + 0.5 * bagged_test),
    }
    metrics_test = evaluate(y_test.values, pred_dict_test, "test")

    # Also report train/valid for the headline models for overfitting diagnostics
    pred_dict_train = {
        "persistence": clip_ghi(df.loc[train_mask, "ghi_now"].values),
        "lgbm_direct": clip_ghi(direct.predict(x_train)),
        "lgbm_residual": clip_ghi(df.loc[train_mask, "ghi_now"].values + residual.predict(x_train)),
        "lgbm_direct_bagged": bagged_predict(x_train),
    }
    metrics_train = evaluate(y_train.values, pred_dict_train, "train")

    pred_dict_valid = {
        "persistence": persistence_valid,
        "lgbm_direct": direct_valid,
        "lgbm_residual": residual_valid,
        "lgbm_direct_bagged": bagged_valid,
    }
    metrics_valid = evaluate(y_valid.values, pred_dict_valid, "valid")

    metrics_df = pd.concat([metrics_train, metrics_valid, metrics_test], ignore_index=True)
    metrics_path = OUTPUT_DIR / "ghi_1h_v7_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n=== TEST SET RESULTS (2025 holdout) ===")
    print(metrics_test.sort_values("rmse").to_string(index=False))

    pred_out = df.loc[test_mask, [TIME_COL, "target_ts_wib", TARGET_COL, "ghi_now", "solar_elev_deg", "clp_cot"]].copy()
    for key, val in pred_dict_test.items():
        pred_out[key] = val
    pred_out.to_csv(OUTPUT_DIR / "ghi_1h_v7_predictions_test.csv", index=False)

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance_direct": direct.named_steps["model"].feature_importances_,
        "importance_residual": residual.named_steps["model"].feature_importances_,
    }).sort_values("importance_direct", ascending=False)
    imp.to_csv(OUTPUT_DIR / "ghi_1h_v7_feature_importance.csv", index=False)

    joblib.dump({"pipeline": direct, "features": FEATURES}, MODEL_DIR / "direct_model.joblib")
    joblib.dump({"pipeline": residual, "features": FEATURES}, MODEL_DIR / "residual_model.joblib")
    joblib.dump({"pipelines": bag_models, "features": FEATURES}, MODEL_DIR / "bagged_direct_models.joblib")
    joblib.dump({"meta": meta, "input_order": ["persistence", "direct", "residual", "bagged", "solar_elev_deg", "clp_cot"]}, MODEL_DIR / "stacked_meta.joblib")

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
    top_imp = imp.head(20).sort_values("importance_direct")
    plt.barh(top_imp["feature"], top_imp["importance_direct"])
    plt.title("Top 20 feature importance (direct model)")

    plt.subplot(2, 2, 4)
    sample = pred_out.sample(min(3000, len(pred_out)), random_state=RANDOM_STATE)
    plt.scatter(sample[TARGET_COL], sample["stacked_meta"], s=4, alpha=0.3)
    plt.plot([0, 1400], [0, 1400], "r--", linewidth=1)
    plt.xlabel("Actual GHI (t+60m)")
    plt.ylabel("Predicted GHI (stacked_meta)")
    plt.title("Stacked meta: actual vs predicted (test)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ghi_1h_v7_diagnostics.png", dpi=160)
    plt.close()

    print("\nSaved outputs under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
