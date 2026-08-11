#!/usr/bin/env python3
"""
R8 Arm A — Bengkulu: Meteorological Redundancy Sensitivity Test

Research question: Does adding AWS meteorology (T, RH, P, wind, rain) to the
§3.2 lean F1 baseline improve GHI forecasting accuracy?

Protocol (identical to Kalbar R8 Arm A):
  F1 (50 features):  §3.2 lean baseline — same as R1 benchmark (no AWS/SYNOP)
  F2 (55 features):  F1 + {aws_temp_c, aws_rh_pct, aws_ws_avg, aws_rain_mm,
                           aws_pressure_hpa}
  Models: CatBoost + LightGBM
  Split:  train < 2024-01-01 | val 2024 | test 2025
  Filter: sun > 5 deg at anchor AND at t+60

R1 benchmark (F1 baseline) already ran separately:
  Point  — LGBM: R2=0.7891 | CatBoost: R2=0.7920
  Avg    — LGBM: R2=0.8994 | CatBoost: R2=0.9004

This script re-runs F1 for self-consistency and adds F2, then reports delta R2.

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_bengkulu_R8_armA.py
"""

import os
from pathlib import Path
import warnings

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_NAME       = "bengkulu"
ATTACH_ALIAS  = "bengkulu_db"
LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")
OUTPUT_DIR    = Path("outputs_R8_bengkulu")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -3.8607
STATION_LON_DEG  = 102.3381
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
F1_FEATURES = [
    # GHI dynamics (16)
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m",
    "accel_ghi_20m",
    # kt dynamics (9)
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
    # CLP (15)
    "clp_cot",
    "clp_cot_lag_10m", "clp_cot_lag_20m",
    "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m",
    "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    # time cyclic (6)
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "month_sin", "month_cos",
    # future deterministic (4)
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
assert len(F1_FEATURES) == 50

# F2 adds 5 AWS meteorological features
AWS_FEATURES = ["aws_temp_c", "aws_rh_pct", "aws_ws_avg", "aws_rain_mm", "aws_pressure_hpa"]
F2_FEATURES  = F1_FEATURES + AWS_FEATURES
assert len(F2_FEATURES) == 55

TARGET_POINT = "ghi_point_t60"
TARGET_AVG   = "ghi_avg_t10_t60"
DELTA_POINT  = "delta_point"
DELTA_AVG    = "delta_avg"


# ---------------------------------------------------------------------------
# Database
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
            "ATTACH '" + LOCAL_DB_PATH.as_posix()
            + "' AS " + ATTACH_ALIAS + " (READ_ONLY)"
        )
        print("Data source: LOCAL  (" + str(LOCAL_DB_PATH) + ")")
    else:
        require_token()
        con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
        print("Data source: MOTHERDUCK")
    return con


# ---------------------------------------------------------------------------
# SQL — same pattern as R1 (SELECT * + fine lags + LEAD)
# AWS columns are included via SELECT * from the view
# ---------------------------------------------------------------------------
def build_sql():
    return """
    WITH with_kt AS (
        SELECT
            *,
            ghi_now / GREATEST(
                1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0
            ) AS kt_point
        FROM bengkulu_db.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
    ), with_windows AS (
        SELECT
            *,
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
            LAG(kt_point, 1) OVER (ORDER BY ts_wib) AS kt_lag_10m,
            LAG(kt_point, 2) OVER (ORDER BY ts_wib) AS kt_lag_20m,
            LAG(kt_point, 3) OVER (ORDER BY ts_wib) AS kt_lag_30m,
            LAG(kt_point, 6) OVER (ORDER BY ts_wib) AS kt_lag_60m,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
                AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                AS kt_roll30m_std,
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
      AND ghi_now BETWEEN 0 AND 1400
    ORDER BY ts_wib
    """


# ---------------------------------------------------------------------------
# Astronomical helpers
# ---------------------------------------------------------------------------
def solar_elevation_deg(timestamps, lat=STATION_LAT_DEG, lon=STATION_LON_DEG,
                        meridian=WIB_MERIDIAN_DEG):
    idx = pd.DatetimeIndex(timestamps)
    doy = idx.dayofyear.values.astype(float)
    h   = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    solar_t = h + 4.0 * (lon - meridian) / 60.0
    ha = (solar_t - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def clearsky_simple(elev_deg):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(elev_deg)), 0.0)


# ---------------------------------------------------------------------------
# Feature engineering (same as R1)
# ---------------------------------------------------------------------------
def add_features(df):
    out = df.copy()
    ts = pd.DatetimeIndex(out[TIME_COL])

    cs_now = clearsky_simple(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    out["accel_ghi_20m"]      = out["ghi_now"]  - 2.0 * out["ghi_lag_10m"]      + out["ghi_lag_20m"]
    out["accel_kt_20m"]       = out["kt_now"]   - 2.0 * out["kt_lag_10m"]       + out["kt_lag_20m"]
    out["accel_clp_cot_20m"]  = out["clp_cot"]  - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy  / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy  / 365.25)

    ts_t60 = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60 = solar_elevation_deg(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"]   = clearsky_simple(elev_t60)

    cs_steps = []
    for step in range(1, 7):
        ts_f = out[TIME_COL] + pd.Timedelta(minutes=step * 10)
        cs_steps.append(clearsky_simple(solar_elevation_deg(ts_f)))
    out["ghi_cs_avg_t10_t60"] = np.column_stack(cs_steps).mean(axis=1)

    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_avg_t10_t60"]

    out[TARGET_POINT] = out["target_ghi_1h_ahead"].copy()

    lead_cols = ["ghi_lead_10m", "ghi_lead_20m", "ghi_lead_30m",
                 "ghi_lead_40m", "ghi_lead_50m", "ghi_lead_60m"]
    leads     = out[lead_cols]
    all_valid = (leads.notna().all(axis=1)
                 & leads.apply(lambda c: c.between(0, 1400)).all(axis=1))
    out[TARGET_AVG] = np.where(all_valid, leads.mean(axis=1), np.nan)

    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))
    out[DELTA_POINT]   = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG]     = out[TARGET_AVG]   - out["ghi_now"]

    return out


# ---------------------------------------------------------------------------
# Split masks
# ---------------------------------------------------------------------------
def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def lgbm_pipe(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=6000, learning_rate=0.02,
        num_leaves=39, min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
        random_state=seed, n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("m",   reg),
    ])


def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )


def fit_lgbm(pipe, x_tr, y_tr, x_es, y_es):
    pipe.fit(x_tr, y_tr,
             m__eval_set=[(x_es, y_es)],
             m__eval_metric="rmse",
             m__callbacks=[lgb.early_stopping(150, verbose=False)])
    return pipe


def fit_catboost(m, x_tr, y_tr, x_es, y_es):
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_es.astype(float).values, y_es.astype(float).values),
          early_stopping_rounds=150)
    return m


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, sp_pred):
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp_rmse = float(np.sqrt(mean_squared_error(y_true, sp_pred)))
    return {
        "r2":          round(float(r2_score(y_true, y_pred)), 4),
        "mae":         round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse":        round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / sp_rmse if sp_rmse > 0 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    con = connect_data()
    print("Loading data...")
    df = con.execute(build_sql()).fetchdf()
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    # Check AWS columns availability
    missing_aws = [c for c in AWS_FEATURES if c not in df.columns]
    if missing_aws:
        print(f"WARNING: AWS columns not found in view: {missing_aws}")
        print("F2 run will be skipped for missing columns.")

    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna()             & df["sun_gt5_t60"]].copy()

    print(f"Rows point target: {len(df_pt):,}")
    print(f"Rows avg target  : {len(df_av):,}")

    all_results = []

    for (df_use, target_col, delta_col, sp_col, tgt_name) in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist",     "point_t60"),
        (df_av, TARGET_AVG,   DELTA_AVG,   "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n{'='*70}")
        print(f"TARGET: {tgt_name}")
        print(f"{'='*70}")

        tr_m, va_m, te_m = split_masks(df_use)
        y_te  = df_use.loc[te_m, target_col]
        sp_te = np.clip(df_use.loc[te_m, sp_col].values, PRED_MIN, PRED_MAX)
        ghi_now_te = df_use.loc[te_m, "ghi_now"].values

        for feat_name, feat_set in [("F1", F1_FEATURES), ("F2", F2_FEATURES)]:
            avail = [f for f in feat_set if f in df_use.columns]
            if len(avail) < len(feat_set):
                skipped = set(feat_set) - set(avail)
                print(f"  [{feat_name}] Warning: {len(skipped)} features missing: {skipped}")
            if len(avail) < 40:
                print(f"  [{feat_name}] Too many features missing — skip")
                continue

            x_tr = df_use.loc[tr_m, avail]
            x_va = df_use.loc[va_m, avail]
            x_te = df_use.loc[te_m, avail]
            y_tr = df_use.loc[tr_m, target_col]
            y_va = df_use.loc[va_m, target_col]
            yd_tr = df_use.loc[tr_m, delta_col]
            yd_va = df_use.loc[va_m, delta_col]

            print(f"\n  [{feat_name} — {len(avail)} features]")

            # CatBoost (direct)
            cb = catboost_model()
            fit_catboost(cb, x_tr, y_tr, x_va, y_va)
            cb_pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
            m = compute_metrics(y_te, cb_pred, sp_te)
            m.update({"lokasi": "bengkulu", "arm": "A", "features": feat_name,
                      "n_feat": len(avail), "model": "catboost", "target": tgt_name})
            all_results.append(m)
            print(f"    CatBoost: R2={m['r2']:.4f}  MAE={m['mae']:.1f}  skill={m['skill_vs_sp']:.4f}")

            # LightGBM (residual)
            lgbm = lgbm_pipe()
            fit_lgbm(lgbm, x_tr, yd_tr, x_va, yd_va)
            best_it = lgbm.named_steps["m"].best_iteration_
            lgbm_pred = np.clip(ghi_now_te + lgbm.predict(x_te), PRED_MIN, PRED_MAX)
            m = compute_metrics(y_te, lgbm_pred, sp_te)
            m.update({"lokasi": "bengkulu", "arm": "A", "features": feat_name,
                      "n_feat": len(avail), "model": "lgbm", "target": tgt_name,
                      "best_iter": best_it})
            all_results.append(m)
            print(f"    LightGBM: R2={m['r2']:.4f}  MAE={m['mae']:.1f}  iter={best_it}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "arm_A_results.csv", index=False)

    # Print delta analysis
    print(f"\n{'='*70}")
    print("ARM A DELTA R2 SUMMARY (F2 - F1)")
    print(f"{'='*70}")

    for model_name in ["catboost", "lgbm"]:
        print(f"\n  Model: {model_name}")
        for tgt in ["point_t60", "avg_t10_t60"]:
            sub = results_df[(results_df["model"] == model_name) & (results_df["target"] == tgt)]
            if len(sub) < 2:
                continue
            r2_f1 = sub.loc[sub["features"] == "F1", "r2"].values
            r2_f2 = sub.loc[sub["features"] == "F2", "r2"].values
            if len(r2_f1) and len(r2_f2):
                delta = r2_f2[0] - r2_f1[0]
                flag = ("HIGH" if delta > 0.015
                        else "MEDIUM" if delta > 0.005
                        else "NEGLIGIBLE (<0.5%)")
                print(f"    {tgt:15s}: F1={r2_f1[0]:.4f}  F2={r2_f2[0]:.4f}  "
                      f"delta={delta:+.4f}  -> {flag}")

    # Arm B summary (from R1 — no re-run needed)
    print(f"\n{'='*70}")
    print("ARM B SUMMARY (from R1 benchmark — no re-run)")
    print(f"{'='*70}")
    print("  point_t60  : CatBoost R2=0.7920  LGBM R2=0.7891  delta=+0.0029 (CB wins)")
    print("  avg_t10_t60: CatBoost R2=0.9004  LGBM R2=0.8994  delta=+0.0010 (CB wins)")
    print("  Interpretation: CatBoost consistently superior — consistent with Kalbar")

    print(f"\nAll outputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
