#!/usr/bin/env python3
"""
Walk-forward 5-fold cross-validation for the v10 model recipe
(43 features, single LightGBM residual — identical recipe to
train_ghi_1h_bengkulu_v10_accel_lean.py).

Purpose: produce the walk-forward R² that pairs with the test-2025
holdout R²=0.8212 for Table 2a.  Running both from the same 43-feature
residual recipe gives a consistent, apples-to-apples pair of numbers.

Fold design (5 folds, 6-month test windows, purely chronological):
  Fold 1 — train: <2023-01-01,  test: 2023-01-01..2023-07-01
  Fold 2 — train: <2023-07-01,  test: 2023-07-01..2024-01-01
  Fold 3 — train: <2024-01-01,  test: 2024-01-01..2024-07-01
  Fold 4 — train: <2024-07-01,  test: 2024-07-01..2025-01-01
  Fold 5 — train: <2025-01-01,  test: 2025-01-01..end      ← same as holdout
Each fold uses the last 3 months of its training window as the early-
stopping validation set (no leakage: those rows are excluded from the
effective training set sent to LightGBM).

Data source: local bengkulu.duckdb (offline, no token) with MotherDuck
fallback — same connect_data() pattern as v10.

Run:
    python train_ghi_1h_bengkulu_v10_walkforward.py

Outputs:
    outputs_v10_walkforward/ghi_1h_v10_wf_fold_results.csv   per-fold metrics
    outputs_v10_walkforward/ghi_1h_v10_wf_summary.csv        mean ± std across folds
"""

import os
from pathlib import Path
import warnings

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config — identical to v10
# ---------------------------------------------------------------------------
DB_NAME       = "bengkulu"
ATTACH_ALIAS  = "bengkulu_db"
LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")
OUTPUT_DIR    = Path("outputs_v10_walkforward")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL        = "ts_wib"
TARGET_COL      = "target_ghi_1h_ahead"
DELTA_TARGET    = "target_delta_ghi_1h"
RANDOM_STATE    = 42
PRED_MIN, PRED_MAX = 0.0, 1400.0

# 5 walk-forward folds: (test_start, test_end)  — train = all rows before test_start
# test_end=None means "to the end of dataset"
FOLDS = [
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", None),
]
# Early-stopping window: last N months of each fold's training block
EARLY_STOP_MONTHS = 3

# ---------------------------------------------------------------------------
# Feature lists — identical to v10
# ---------------------------------------------------------------------------
PRUNED_40_FEATURES = [
    "clp_cot",             "kt_now",              "dni_now",             "hour_sin",
    "solar_elev_deg",      "solar_elev_sin_clip",  "month_cos",           "clp_cot_delta_10m",
    "clp_cot_delta_180m",  "clp_cer",             "month_sin",           "ghi_now",
    "nett_rad_now",        "dni_fraction",        "aws_pressure_lag_60m","synop_dewpoint_c",
    "dhi_roll_180m_mean",  "ghi_roll_180m_max",   "clp_cot_delta_30m",   "aws_pressure_hpa",
    "clp_cot_roll_180m_mean","cloud_height_temp_interaction","aws_ws_roll_180m_mean","clp_cot_lag_10m",
    "ghi_lag_180m",        "synop_temp_c",        "clp_cth_roll_180m_mean","dni_roll_180m_mean",
    "ghi_roll_180m_min",   "temp_rh_interaction", "clp_ctt_k",           "clp_cth_m",
    "clp_cth_delta_180m",  "synop_wind_dir_deg",  "clp_cot_delta_60m",   "reflected_now",
    "ghi_delta_60m",       "aws_rh_roll_180m_mean","aws_temp_roll_180m_mean","ghi_roll_180m_std",
]
ACCEL_FEATURES = ["accel_clp_cot_20m", "accel_kt_20m", "accel_ghi_20m"]
FEATURES = PRUNED_40_FEATURES + ACCEL_FEATURES
assert len(FEATURES) == 43

# ---------------------------------------------------------------------------
# DB connection
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
            "ATTACH '" + LOCAL_DB_PATH.as_posix() + "' AS " + ATTACH_ALIAS + " (READ_ONLY)"
        )
        print("Data source: LOCAL  (" + str(LOCAL_DB_PATH) + ")")
    else:
        require_token()
        con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
        print("Data source: MOTHERDUCK (md:" + DB_NAME + ")")
    return con


# ---------------------------------------------------------------------------
# SQL + feature engineering — identical to v10
# ---------------------------------------------------------------------------
def build_sql():
    return """
    WITH joined AS (
        SELECT
            t.*,
            s.cloud_cover_oktas_m        AS syn_cloud_cover_oktas_m,
            s.cloud_low_cover_oktas      AS syn_cloud_low_cover_oktas,
            s.cloud_med_cover_oktas      AS syn_cloud_med_cover_oktas,
            s.cloud_low_base_1           AS syn_cloud_low_base_1,
            s.cloud_low_base_2           AS syn_cloud_low_base_2,
            s.cloud_low_base_3           AS syn_cloud_low_base_3,
            s.cloud_med_base_1           AS syn_cloud_med_base_1,
            s.cloud_med_base_2           AS syn_cloud_med_base_2,
            s.cloud_high_base_1          AS syn_cloud_high_base_1,
            s.cloud_high_base_2          AS syn_cloud_high_base_2,
            s.cloud_layer_1_height_m_hshs AS syn_cloud_layer_1_height_m,
            s.cloud_layer_2_height_m_hshs AS syn_cloud_layer_2_height_m,
            s.cloud_layer_3_height_m_hshs AS syn_cloud_layer_3_height_m,
            s.cloud_layer_4_height_m_hshs AS syn_cloud_layer_4_height_m,
            s.cloud_layer_1_amt_oktas_ns  AS syn_cloud_layer_1_amt_oktas,
            s.cloud_layer_2_amt_oktas_ns  AS syn_cloud_layer_2_amt_oktas,
            s.cloud_layer_3_amt_oktas_ns  AS syn_cloud_layer_3_amt_oktas,
            s.cloud_layer_4_amt_oktas_ns  AS syn_cloud_layer_4_amt_oktas,
            s.present_weather_ww          AS syn_present_weather,
            TRY_CAST(s.past_weather_w1 AS DOUBLE) AS syn_past_weather_1,
            TRY_CAST(s.past_weather_w2 AS DOUBLE) AS syn_past_weather_2
        FROM bengkulu_db.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025 t
        LEFT JOIN bengkulu_db.bengkulu_sch.synop_bengkulu_quality_final s
          ON time_bucket(INTERVAL '1 hour', t.ts_wib) = s.ts_wib
    ), with_kt AS (
        SELECT *,
            ghi_now / GREATEST(1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0) AS kt_point
        FROM joined
    ), with_windows AS (
        SELECT *,
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib)  AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib)  AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib)  AS clp_cot_lag_30m,
            LAG(clp_cth_m, 3) OVER (ORDER BY ts_wib) AS clp_cth_lag_30m,
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib)  AS ghi_lag_20m,
            LAG(ghi_now, 5) OVER (ORDER BY ts_wib)  AS ghi_lag_50m,
            LAG(kt_point, 1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point, 2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(kt_point, 3) OVER (ORDER BY ts_wib) AS kt_lag_30m,
            LAG(kt_point, 6) OVER (ORDER BY ts_wib) AS kt_lag_60m,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std
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


def add_engineered_features(df):
    out = df.copy()
    elev_rad = np.deg2rad(out["solar_elev_deg"].astype(float))
    elev_sin = np.sin(elev_rad)
    out["solar_elev_sin"]      = elev_sin
    out["solar_elev_sin_clip"] = np.maximum(elev_sin, 0.02)
    out["clear_sky_ghi_now"]   = 1100.0 * out["solar_elev_sin_clip"]
    out["kt_now"]              = out["ghi_now"] / np.maximum(out["clear_sky_ghi_now"], 20.0)
    out["ghi_to_aws_sr_ratio"] = out["ghi_now"] / np.maximum(out["aws_sr_avg_w_m2"], 20.0)
    out["dhi_fraction"]        = out["dhi_now"] / np.maximum(out["ghi_now"], 20.0)
    out["dni_fraction"]        = out["dni_now"] / np.maximum(out["ghi_now"], 20.0)
    out["temp_rh_interaction"] = out["aws_temp_c"] * out["aws_rh_pct"]
    out["vpd_proxy"]           = out["aws_temp_c"] * (100.0 - out["aws_rh_pct"]) / 100.0
    wd_rad = np.deg2rad(out["aws_wd_deg"].astype(float))
    out["wind_u"] = out["aws_ws_avg"] * np.sin(wd_rad)
    out["wind_v"] = out["aws_ws_avg"] * np.cos(wd_rad)
    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cth_delta_60m"]  = out["clp_cth_m"] - out["clp_cth_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["clp_cth_delta_180m"] = out["clp_cth_m"] - out["clp_cth_roll_180m_mean"]
    out["ghi_roll_180m_range"] = out["ghi_roll_180m_max"] - out["ghi_roll_180m_min"]
    out["ghi_roll_60m_range"]  = out["ghi_roll_60m_max"]  - out["ghi_roll_60m_min"]
    out["ghi_ramp_ratio_60m"]  = out["ghi_delta_60m"] / np.maximum(out["ghi_lag_60m"].abs(), 20.0)
    out["aws_temp_range"]      = out["aws_temp_max_c"] - out["aws_temp_min_c"]
    out["cloud_opacity_proxy"] = out["clp_cot"] * out["clp_cloud_present"].fillna(0)
    out["cloud_height_temp_interaction"] = out["clp_cth_m"] * out["clp_ctt_k"]
    syn_cols = ["syn_cloud_cover_oktas_m", "syn_cloud_low_cover_oktas", "syn_cloud_med_cover_oktas"]
    out["syn_total_cloud_oktas"]  = out[syn_cols].max(axis=1)
    out["syn_low_cloud_present"]  = (out["syn_cloud_low_cover_oktas"].fillna(0) > 0).astype(int)
    layer_cols = ["syn_cloud_layer_1_amt_oktas", "syn_cloud_layer_2_amt_oktas",
                  "syn_cloud_layer_3_amt_oktas", "syn_cloud_layer_4_amt_oktas"]
    out["syn_multi_layer_cloud_flag"] = (out[layer_cols].notna().sum(axis=1) >= 2).astype(int)
    base_cols = ["syn_cloud_low_base_1", "syn_cloud_low_base_2", "syn_cloud_low_base_3",
                 "syn_cloud_med_base_1", "syn_cloud_med_base_2",
                 "syn_cloud_high_base_1", "syn_cloud_high_base_2"]
    out["syn_cloud_base_min"]   = out[base_cols].min(axis=1)
    height_cols = ["syn_cloud_layer_1_height_m", "syn_cloud_layer_2_height_m",
                   "syn_cloud_layer_3_height_m", "syn_cloud_layer_4_height_m"]
    out["syn_cloud_depth_proxy"] = out[height_cols].max(axis=1) - out[height_cols].min(axis=1)
    out["syn_weather_cloud_rain_proxy"] = (
        out["syn_present_weather"].fillna(0) +
        out["syn_past_weather_1"].fillna(0) +
        out["syn_past_weather_2"].fillna(0)
    )
    out[DELTA_TARGET] = out[TARGET_COL] - out["ghi_now"]
    # Acceleration features (2nd difference, 20 min window)
    out["accel_clp_cot_20m"] = out["clp_cot"]  - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    out["accel_kt_20m"]      = out["kt_now"]   - 2.0 * out["kt_lag_10m"]      + out["kt_lag_20m"]
    out["accel_ghi_20m"]     = out["ghi_now"]  - 2.0 * out["ghi_lag_10m"]     + out["ghi_lag_20m"]
    return out


# ---------------------------------------------------------------------------
# Model factory — identical hyperparams to v10 residual
# ---------------------------------------------------------------------------
def make_residual_pipe(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=6000, learning_rate=0.02,
        num_leaves=39, min_child_samples=70,
        reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85,
        subsample_freq=1, random_state=seed,
        n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model",   reg),
    ])


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------
def run_fold(df, test_start, test_end, fold_idx):
    ts = df[TIME_COL]
    ts_start = pd.Timestamp(test_start)
    ts_end   = pd.Timestamp(test_end) if test_end else pd.Timestamp("2099-01-01")

    train_mask = ts < ts_start
    test_mask  = (ts >= ts_start) & (ts < ts_end)

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    if len(df_train) < 5000 or len(df_test) < 100:
        print(f"  Fold {fold_idx}: insufficient rows — skipping")
        return None

    # Early-stopping val = last EARLY_STOP_MONTHS months of training block
    es_cutoff = ts_start - pd.DateOffset(months=EARLY_STOP_MONTHS)
    es_mask_train = df_train[TIME_COL] < es_cutoff
    es_mask_val   = df_train[TIME_COL] >= es_cutoff

    df_tr  = df_train[es_mask_train]
    df_es  = df_train[es_mask_val]

    x_tr,  yd_tr  = df_tr[FEATURES],  df_tr[DELTA_TARGET]
    x_es,  yd_es  = df_es[FEATURES],  df_es[DELTA_TARGET]
    x_test, y_test = df_test[FEATURES], df_test[TARGET_COL]

    pipe = make_residual_pipe()
    pipe.fit(
        x_tr, yd_tr,
        model__eval_set=[(x_es, yd_es)],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    best_iter = pipe.named_steps["model"].best_iteration_

    residual_pred = pipe.predict(x_test)
    ghi_pred = np.clip(df_test["ghi_now"].values + residual_pred, PRED_MIN, PRED_MAX)

    r2   = float(r2_score(y_test, ghi_pred))
    mae  = float(mean_absolute_error(y_test, ghi_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, ghi_pred)))

    test_label = test_start[:7] + ".." + (test_end[:7] if test_end else "end")
    print(f"  Fold {fold_idx} [{test_label}]  n_train={len(df_tr):6d}  n_es={len(df_es):5d}  "
          f"n_test={len(df_test):5d}  best_iter={best_iter:4d}  "
          f"R²={r2:.4f}  MAE={mae:.1f}  RMSE={rmse:.1f}")

    return {
        "fold": fold_idx,
        "test_period": test_label,
        "n_train_eff": len(df_tr),
        "n_es": len(df_es),
        "n_test": len(df_test),
        "best_iter": best_iter,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }


def main():
    con = connect_data()
    print("Loading data...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_engineered_features(df)
    print(f"Total rows: {len(df):,}  |  {df[TIME_COL].min().date()} to {df[TIME_COL].max().date()}\n")

    print("Walk-forward 5-fold (lgbm_residual, 43 features):")
    print("-" * 90)

    results = []
    for idx, (test_start, test_end) in enumerate(FOLDS, start=1):
        row = run_fold(df, test_start, test_end, idx)
        if row:
            results.append(row)

    fold_df = pd.DataFrame(results)
    fold_df.to_csv(OUTPUT_DIR / "ghi_1h_v10_wf_fold_results.csv", index=False)

    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY (lgbm_residual, v10, 43 features)")
    print("=" * 60)
    mean_r2   = fold_df["r2"].mean()
    std_r2    = fold_df["r2"].std()
    mean_mae  = fold_df["mae"].mean()
    std_mae   = fold_df["mae"].std()
    mean_rmse = fold_df["rmse"].mean()
    std_rmse  = fold_df["rmse"].std()

    print(f"  R²   : {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  MAE  : {mean_mae:.1f} ± {std_mae:.1f} W/m²")
    print(f"  RMSE : {mean_rmse:.1f} ± {std_rmse:.1f} W/m²")
    print(f"\n  (Fold 5 = 2025 holdout, cf. v10 test R2~0.8212; small diff from ES window size)")

    summary = pd.DataFrame([{
        "metric": "r2",   "mean": mean_r2,   "std": std_r2},
        {"metric": "mae",  "mean": mean_mae,  "std": std_mae},
        {"metric": "rmse", "mean": mean_rmse, "std": std_rmse},
    ])
    summary.to_csv(OUTPUT_DIR / "ghi_1h_v10_wf_summary.csv", index=False)
    print(f"\nSaved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
