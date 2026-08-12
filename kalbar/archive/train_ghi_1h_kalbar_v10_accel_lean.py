#!/usr/bin/env python3
"""
V10 recipe ported to KALBAR (Prioritas B, 2026-07-25) -- replicate Bengkulu's
best single-model result (43 features = 40 pruned + 3 acceleration, single
LightGBM residual, R2 test=0.8212 at Bengkulu) on Kalbar.

Feature-availability audit (2026-07-25): Kalbar's existing training table
(training_ghi_1h_direct, 66 cols, used by R1/R8) does NOT carry the raw
DHI/DNI/reflected/net-rad/AWS-pressure/AWS-wind values or any 180m rolling
windows the v10 recipe needs -- it only has lag/roll up to 60m for kt and up
to lag60m for ghi. So this script builds a FRESH SQL query straight from
Kalbar's raw 10-minute tables (bypassing training_ghi_1h_direct entirely),
mirroring the same LAG/window approach used for Bengkulu/Jambi/Banten:
  - solar_kalbar_10m       -> ghi_final, dni_final, dhi_final, reflected_rad,
                               net_rad, sun_altitude (10-min, 210,384 rows)
  - meteorologi_kalbar_10m -> suhu_avg_c, rh_percent, pressure_hpa,
                               wind_speed_ms, wind_dir_deg (10-min, same join
                               key/row count as solar_kalbar_10m -- verified
                               1:1 by timestamp_wib)
  - clp_pontianak_20km     -> CLOT_mean/CLTT_mean/CLTH_mean/CLER_23_mean
                               (cloud optical thickness/top-temp/top-height/
                               eff-radius, 10-min but only ~48% coverage vs
                               solar/meteo -- LEFT JOIN, gaps handled by the
                               model's median imputer, consistent with how
                               Kalbar's official training table already
                               handles CLP gaps)
  - synop_unified          -> temp_dewpoint_c (hourly, LEFT JOIN via
                               time_bucket, same pattern as Bengkulu's SYNOP
                               cloud-layer join)

Substitution note: synop_temp_c reuses meteorologi_kalbar_10m's suhu_avg_c
(no separate hourly SYNOP dry-bulb reading joined in, to keep this a LEFT
JOIN on one hourly table only) -- same style of substitution documented for
Banten. synop_wind_dir_deg reuses meteorologi's wind_dir_deg (10-min AWS
wind direction) rather than synop_unified's, since coverage is far higher.

Target: ghi_final at t+60 via LEAD(6) on solar_kalbar_10m's own 10-min
series (the genuine-lead approach already verified this session for Kalbar's
per-horizon walk-forward, NOT the anchor_valid-derived target in
training_ghi_1h_direct).

Run:
    python train_ghi_1h_kalbar_v10_accel_lean.py
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

DB_PATH = "kalbar_local.db"
OUTPUT_DIR = Path("outputs_v10_kalbar")
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
        SELECT
            s.timestamp_wib AS ts_wib,
            s.ghi_final AS ghi_now, s.dni_final AS dni_now, s.dhi_final AS dhi_now,
            s.reflected_rad AS reflected_now, s.net_rad AS nett_rad_now,
            s.sun_altitude AS solar_elev_deg,
            m.suhu_avg_c AS aws_temp_c, m.rh_percent AS aws_rh_pct,
            m.pressure_hpa AS aws_pressure_hpa, m.wind_speed_ms AS aws_ws_avg,
            m.wind_dir_deg AS synop_wind_dir_deg, m.suhu_avg_c AS synop_temp_c,
            c.CLOT_mean AS clp_cot, c.CLTH_mean AS clp_cth_m, c.CLTT_mean AS clp_ctt_k, c.CLER_23_mean AS clp_cer,
            sy.temp_dewpoint_c AS synop_dewpoint_c
        FROM solar_kalbar_10m s
        JOIN meteorologi_kalbar_10m m ON s.timestamp_wib = m.timestamp_wib
        LEFT JOIN clp_pontianak_20km c ON s.timestamp_wib = c.timestamp
        LEFT JOIN synop_unified sy ON time_bucket(INTERVAL '1 hour', s.timestamp_wib) = sy.timestamp_wib
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
    idx = pd.DatetimeIndex(out[TIME_COL])
    hh = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    mo = idx.month.values.astype(float)
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
    con = duckdb.connect(DB_PATH, read_only=True)
    print("Loading Kalbar raw 10-min tables + v10 SQL features (fresh build)...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_engineered_features(df)
    df = df[df[TARGET_COL].between(0, 1400)].copy()
    print(f"Rows: {len(df):,} | {df[TIME_COL].min()}..{df[TIME_COL].max()}")
    print(f"CLP coverage (clp_cot non-null): {100.0*df['clp_cot'].notna().mean():.1f}%")

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
    metrics_df.to_csv(OUTPUT_DIR / "ghi_1h_v10_kalbar_metrics.csv", index=False)

    print("\n=== TEST SET RESULTS (2025 holdout) ===")
    print(metrics_df.to_string(index=False))

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": residual.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(OUTPUT_DIR / "ghi_1h_v10_kalbar_feature_importance.csv", index=False)

    print(f"\nSaved outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
