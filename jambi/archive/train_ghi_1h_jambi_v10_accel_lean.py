#!/usr/bin/env python3
"""
V10 recipe ported to JAMBI (Prioritas B, 2026-07-25) -- replicate Bengkulu's
best single-model result (43 features = 40 pruned + 3 acceleration, single
LightGBM residual, R2 test=0.8212 at Bengkulu) on Jambi's new v2 rollback
table (ghi_forecast_1h_train_3h_rollback_2021_2025 in
jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb), which was built to
mirror Bengkulu's 102-column schema 1:1 (see 09_Audit_Volume_Data_Jambi.md).

Feature-availability audit (2026-07-25): ALL 43 v10 features are directly
computable from Jambi's v2 table -- same base columns as Bengkulu. The SYNOP
cloud-layer join + wavelet features present in the Bengkulu script are
DROPPED here because inspection of PRUNED_40_FEATURES confirmed neither is
actually consumed by any of the 43 selected features. So this port is a
near-exact copy: same FEATURES list, same hyperparameters, same split -- only
the data source + the extra SQL lags needed (clp_cot_lag_10/20/30m,
kt_lag_10/20m, ghi_lag_20m; everything else is either a base column already
present in Jambi's table or a pandas-side transform identical to Bengkulu's
add_engineered_features()).

Run:
    python train_ghi_1h_jambi_v10_accel_lean.py
"""
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

LOCAL_DB_PATH = Path("jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb")
SCHEMA = "jambi_sch"
TABLE = "ghi_forecast_1h_train_3h_rollback_2021_2025"
OUTPUT_DIR = Path("outputs_v10_jambi")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "target_ghi_1h_ahead"
DELTA_TARGET_COL = "target_delta_ghi_1h"
TARGET_AVG_COL = "ghi_avg_t10_t60"
DELTA_AVG_COL = "delta_avg"
PRED_MIN, PRED_MAX = 0.0, 1400.0
TRAIN_END, VALID_END = "2024-01-01", "2025-01-01"
RANDOM_STATE = 42

# Identical to Bengkulu v10 (train_ghi_1h_bengkulu_v10_accel_lean.py)
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
ACCEL_FEATURES = ["accel_clp_cot_20m", "accel_kt_20m", "accel_ghi_20m"]
FEATURES = PRUNED_40_FEATURES + ACCEL_FEATURES
assert len(FEATURES) == 43


def connect_data():
    con = duckdb.connect(database=":memory:")
    con.execute(f"ATTACH '{LOCAL_DB_PATH.as_posix()}' AS jambi_db (READ_ONLY)")
    print(f"Data source: LOCAL ({LOCAL_DB_PATH})")
    return con


def build_sql():
    return f"""
    WITH with_kt AS (
        SELECT
            *,
            ghi_now / GREATEST(1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0) AS kt_point
        FROM jambi_db.{SCHEMA}.{TABLE}
    ), with_windows AS (
        SELECT
            *,
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            LAG(kt_point, 1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point, 2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
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


def add_engineered_features(df):
    out = df.copy()
    elev_rad = np.deg2rad(out["solar_elev_deg"].astype(float))
    elev_sin = np.sin(elev_rad)
    out["solar_elev_sin_clip"] = np.maximum(elev_sin, 0.02)
    clear_sky_ghi_now = 1100.0 * out["solar_elev_sin_clip"]
    out["kt_now"] = out["ghi_now"] / np.maximum(clear_sky_ghi_now, 20.0)
    out["dni_fraction"] = out["dni_now"] / np.maximum(out["ghi_now"], 20.0)
    out["temp_rh_interaction"] = out["aws_temp_c"] * out["aws_rh_pct"]
    out["clp_cot_delta_10m"] = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"] = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"] = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["clp_cth_delta_180m"] = out["clp_cth_m"] - out["clp_cth_roll_180m_mean"]
    out["cloud_height_temp_interaction"] = out["clp_cth_m"] * out["clp_ctt_k"]
    out[DELTA_TARGET_COL] = out[TARGET_COL] - out["ghi_now"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2.0 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_ghi_20m"] = out["ghi_now"] - 2.0 * out["ghi_lag_10m"] + out["ghi_lag_20m"]
    lead_cols = ["ghi_lead_10m", "ghi_lead_20m", "ghi_lead_30m", "ghi_lead_40m", "ghi_lead_50m", "ghi_lead_60m"]
    leads = out[lead_cols]
    avg_valid = leads.notna().all(axis=1) & leads.apply(lambda c: c.between(0, 1400)).all(axis=1)
    out[TARGET_AVG_COL] = np.where(avg_valid, leads.mean(axis=1), np.nan)
    out[DELTA_AVG_COL] = out[TARGET_AVG_COL] - out["ghi_now"]
    return out


def clip_ghi(v):
    return np.clip(v, PRED_MIN, PRED_MAX)


def make_lgbm(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=6000, learning_rate=0.02, num_leaves=39,
        min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5, colsample_bytree=0.82,
        subsample=0.85, subsample_freq=1, random_state=seed, n_jobs=-1,
        force_col_wise=True, verbosity=-1,
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


def metric_row(y_true, y_pred, model_name, persistence_rmse=None, target_name="point"):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    mbe = float(np.mean(y_pred - y_true))
    skill = 0.0 if model_name == "persistence" else np.nan
    if persistence_rmse and persistence_rmse > 0:
        skill = 1.0 - rmse / persistence_rmse
    return {"model": model_name, "target": target_name, "n_rows": len(y_true), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def split_masks(df):
    train = df[TIME_COL] < pd.Timestamp(TRAIN_END)
    valid = (df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))
    test = df[TIME_COL] >= pd.Timestamp(VALID_END)
    return train, valid, test


def main():
    con = connect_data()
    print("Loading Jambi v2 rollback table + v10 SQL window features...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_engineered_features(df)
    print(f"Rows loaded: {len(df):,}")
    print(f"Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")

    train_mask, valid_mask, test_mask = split_masks(df)
    print(f"n_train={int(train_mask.sum()):,} n_valid={int(valid_mask.sum()):,} n_test={int(test_mask.sum()):,}")

    x_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET_COL]
    x_valid, y_valid = df.loc[valid_mask, FEATURES], df.loc[valid_mask, TARGET_COL]
    x_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET_COL]
    yd_train = df.loc[train_mask, DELTA_TARGET_COL]
    yd_valid = df.loc[valid_mask, DELTA_TARGET_COL]

    print("Training single LightGBM residual model (v10 recipe, no ensemble)...")
    residual = make_lgbm()
    fit_with_early_stop(residual, x_train, yd_train, x_valid, yd_valid)
    print("  best_iteration_ =", residual.named_steps["model"].best_iteration_)

    persistence_test = clip_ghi(df.loc[test_mask, "ghi_now"].values)
    residual_test = clip_ghi(df.loc[test_mask, "ghi_now"].values + residual.predict(x_test))
    persistence_rmse = float(np.sqrt(mean_squared_error(y_test.values, persistence_test)))

    rows = [
        metric_row(y_test.values, persistence_test, "persistence", target_name="point"),
        metric_row(y_test.values, residual_test, "lgbm_residual", persistence_rmse, target_name="point"),
    ]

    # --- v11-style supplementary check: same 43-feature recipe, avg_t10_t60 target ---
    # NOTE: per 06_Perbandingan_4_Lokasi.md (koreksi metodologi 2026-07-24), avg_t10_t60
    # is NOT the primary result -- this run exists only to test whether the "+0.15-0.22 R2
    # from averaging" pattern found at Bengkulu generalizes, per Prioritas B item 2.
    df_avg = df[df[TARGET_AVG_COL].notna()].copy()
    tr_a, va_a, te_a = split_masks(df_avg)
    xa_tr, ya_tr = df_avg.loc[tr_a, FEATURES], df_avg.loc[tr_a, TARGET_AVG_COL]
    xa_va, ya_va = df_avg.loc[va_a, FEATURES], df_avg.loc[va_a, TARGET_AVG_COL]
    xa_te, ya_te = df_avg.loc[te_a, FEATURES], df_avg.loc[te_a, TARGET_AVG_COL]
    yda_tr = df_avg.loc[tr_a, DELTA_AVG_COL]
    yda_va = df_avg.loc[va_a, DELTA_AVG_COL]
    print(f"\n[avg_t10_t60] n_train={len(xa_tr):,} n_valid={len(xa_va):,} n_test={len(xa_te):,}")
    residual_avg = make_lgbm()
    fit_with_early_stop(residual_avg, xa_tr, yda_tr, xa_va, yda_va)
    print("  best_iteration_ =", residual_avg.named_steps["model"].best_iteration_)
    persistence_avg_test = clip_ghi(df_avg.loc[te_a, "ghi_now"].values)
    residual_avg_test = clip_ghi(df_avg.loc[te_a, "ghi_now"].values + residual_avg.predict(xa_te))
    persistence_avg_rmse = float(np.sqrt(mean_squared_error(ya_te.values, persistence_avg_test)))
    rows.append(metric_row(ya_te.values, persistence_avg_test, "persistence", target_name="avg"))
    rows.append(metric_row(ya_te.values, residual_avg_test, "lgbm_residual", persistence_avg_rmse, target_name="avg"))

    metrics_df = pd.DataFrame(rows)
    metrics_df["split"] = "test"
    metrics_df.to_csv(OUTPUT_DIR / "ghi_1h_v10_jambi_metrics.csv", index=False)

    print("\n=== TEST SET RESULTS (2025 holdout) ===")
    print(metrics_df.to_string(index=False))

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": residual.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(OUTPUT_DIR / "ghi_1h_v10_jambi_feature_importance.csv", index=False)

    print(f"\nSaved outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
