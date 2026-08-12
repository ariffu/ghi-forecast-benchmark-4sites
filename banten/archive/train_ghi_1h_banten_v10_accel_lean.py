#!/usr/bin/env python3
"""
V10 recipe ported to BANTEN (Prioritas B, 2026-07-25) -- replicate Bengkulu's
best single-model result (43 features = 40 pruned + 3 acceleration, single
LightGBM residual, R2 test=0.8212 at Bengkulu) on Banten's solar_features_base
(banten.duckdb).

Feature-availability audit (2026-07-25): Banten's raw table HAS all the raw
columns v10 needs (dni, dhi, net_rad_wm2, reflected_rad_wm2, temp, rh,
pressure, ws, dewpoint_c, wind_dir_deg, cloud_top_height, cloud_top_temp,
cloud_eff_radius -- see solar_features_base, coverage 89-100%), but the
existing Banten R1 script (train_ghi_1h_banten_R1_benchmark.py) only computes
30/60m rolling windows, not the 180m windows v10 needs for GHI/DHI/DNI/AWS.
This script extends that SQL with the missing 180m rolls + 60m pressure lag.

Substitutions (documented, not hidden):
  - synop_temp_c / synop_dewpoint_c / synop_wind_dir_deg -- Banten's fused
    solar_features_base has ONE station-level temp/dewpoint/wind-dir series
    (no separate AWS-vs-SYNOP split like Bengkulu/Jambi), so synop_temp_c
    reuses the same 'temp' column as aws_temp_c (this is a genuine data
    limitation, not an error -- Banten's raw sources were already fused
    upstream, see 08_Standardisasi_Data_Mentah.md).

Run:
    python train_ghi_1h_banten_v10_accel_lean.py
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

DB_PATH = "banten.duckdb"
OUTPUT_DIR = Path("outputs_v10_banten")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "ghi_point_t60"
DELTA_TARGET_COL = "delta_point"
TARGET_AVG_COL = "ghi_avg_t10_t60"
DELTA_AVG_COL = "delta_avg"
PRED_MIN, PRED_MAX = 0.0, 1400.0
TRAIN_END, VALID_END = "2024-01-01", "2025-01-01"
RANDOM_STATE = 42

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


def build_sql():
    return """
    WITH base AS (
        SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, dni AS dni_now, dhi AS dhi_now,
               net_rad_wm2 AS nett_rad_now, reflected_rad_wm2 AS reflected_now,
               elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m,
               cloud_top_temp AS clp_ctt_k, cloud_eff_radius AS clp_cer,
               temp AS aws_temp_c, rh AS aws_rh_pct, pressure AS aws_pressure_hpa, ws AS aws_ws_avg,
               dewpoint_c AS synop_dewpoint_c, temp AS synop_temp_c, wind_dir_deg AS synop_wind_dir_deg
        FROM solar_features_base
    ), with_kt AS (
        SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base
    ), w AS (
        SELECT *,
          LAG(ghi_now,18) OVER o AS ghi_lag_180m,
          LAG(ghi_now,1) OVER o AS ghi_lag_10m, LAG(ghi_now,2) OVER o AS ghi_lag_20m,
          LAG(ghi_now,6) OVER o AS ghi_lag_60m,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
          MIN(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_min,
          MAX(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_max,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
          AVG(dhi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS dhi_roll_180m_mean,
          AVG(dni_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS dni_roll_180m_mean,
          AVG(aws_temp_c) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_temp_roll_180m_mean,
          AVG(aws_rh_pct) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_rh_roll_180m_mean,
          AVG(aws_ws_avg) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_ws_roll_180m_mean,
          LAG(aws_pressure_hpa,6) OVER o AS aws_pressure_lag_60m,
          LAG(kt_point,1) OVER o AS kt_lag_10m, LAG(kt_point,2) OVER o AS kt_lag_20m,
          LAG(clp_cot,1) OVER o AS clp_cot_lag_10m, LAG(clp_cot,2) OVER o AS clp_cot_lag_20m,
          LAG(clp_cot,3) OVER o AS clp_cot_lag_30m, LAG(clp_cot,6) OVER o AS clp_cot_lag_60m,
          AVG(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
          AVG(clp_cth_m) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cth_roll_180m_mean,
          LEAD(ghi_now,1) OVER o AS ghi_lead_10m, LEAD(ghi_now,2) OVER o AS ghi_lead_20m,
          LEAD(ghi_now,3) OVER o AS ghi_lead_30m, LEAD(ghi_now,4) OVER o AS ghi_lead_40m,
          LEAD(ghi_now,5) OVER o AS ghi_lead_50m, LEAD(ghi_now,6) OVER o AS ghi_lead_60m
        FROM with_kt WINDOW o AS (ORDER BY ts_wib)
    )
    SELECT * FROM w WHERE solar_elev_deg>5 AND ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib
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
    out["ghi_delta_60m"] = out["ghi_now"] - out["ghi_lag_60m"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2.0 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_ghi_20m"] = out["ghi_now"] - 2.0 * out["ghi_lag_10m"] + out["ghi_lag_20m"]
    hh = pd.DatetimeIndex(out[TIME_COL]).hour.values.astype(float) + pd.DatetimeIndex(out[TIME_COL]).minute.values.astype(float) / 60.0
    mo = pd.DatetimeIndex(out[TIME_COL]).month.values.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hh / 24)
    out["month_sin"] = np.sin(2 * np.pi * mo / 12)
    out["month_cos"] = np.cos(2 * np.pi * mo / 12)
    out[TARGET_COL] = out["ghi_lead_60m"]
    out[DELTA_TARGET_COL] = out[TARGET_COL] - out["ghi_now"]
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


def metric_row(y_true, y_pred, model_name, persistence_rmse=None, target_name="point", persistence_rmse_name=None):
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
    con = duckdb.connect(DB_PATH, read_only=True)
    print("Loading Banten data + extended v10 180m windows...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_engineered_features(df)
    df = df[df[TARGET_COL].between(0, 1400)].copy()
    print(f"Rows: {len(df):,} | {df[TIME_COL].min()}..{df[TIME_COL].max()}")

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
        metric_row(y_test.values, persistence_test, "persistence", persistence_rmse_name="point"),
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
    metrics_df.to_csv(OUTPUT_DIR / "ghi_1h_v10_banten_metrics.csv", index=False)

    print("\n=== TEST SET RESULTS (2025 holdout) ===")
    print(metrics_df.to_string(index=False))

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": residual.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(OUTPUT_DIR / "ghi_1h_v10_banten_feature_importance.csv", index=False)

    print(f"\nSaved outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
