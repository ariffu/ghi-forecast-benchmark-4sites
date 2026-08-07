#!/usr/bin/env python3
"""
R1 Harmonised Benchmark — Bengkulu GHI 1-hour-ahead forecasting.

Implements the single standardised configuration anchoring Table 2a/2b
across four sites (Bengkulu, Banten, Kalbar, Jambi).

  Model    : LightGBM residual (PRIMARY) | CatBoost direct (SENSITIVITY)
  Features : §3.2 lean recipe — 50 features, no surface-met / SYNOP:
               GHI dynamics  — now + 5 lags + 6 rolls + 2 deltas + accel
               kt  dynamics  — now + 4 lags + 3 rolls + accel
               CLP           — CLOT now + 4 lags + 4 deltas + roll + accel
                                + CTH, CTT, CER, cloud_present
               Time cyclic   — hour / doy / month  (sin + cos)
               Future determ — clearsky t+60, cos-theta t+60, smart-persist (x2)
  Targets  : (a) GHI point  — instantaneous GHI at t+60 min
              (b) GHI avg   — mean GHI over t+10..t+60 (6 x 10-min steps)
  Data     : 2021-01-01 to 2025-12-31 (local bengkulu.duckdb, offline)
  Split    : train < 2024-01-01 | val 2024 | test 2025
  Filter   : sun_altitude > 5 deg at anchor AND at t+60
  Metrics  : R2, MAE, RMSE, skill vs smart-persistence (test 2025)
             + walk-forward 5-fold R2 (LGBM x point target)

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_bengkulu_R1_benchmark.py
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
OUTPUT_DIR    = Path("outputs_R1_bengkulu")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -3.8607
STATION_LON_DEG  = 102.3381
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"   # train = rows with ts_wib < TRAIN_END (i.e. <= 2023-12-31)
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

# Walk-forward folds: (test_start, test_end)
FOLDS = [
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", None),
]
ES_MONTHS = 3   # months at end of each training block used for early-stopping val

# ---------------------------------------------------------------------------
# §3.2 Feature set  (50 features — no surface-met, no SYNOP)
# ---------------------------------------------------------------------------
# Group A — GHI dynamics (16)
FEATURES_GHI = [
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m",
    "accel_ghi_20m",
]
# Group B — kt dynamics (9)
FEATURES_KT = [
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
]
# Group C — CLP (15)
FEATURES_CLP = [
    "clp_cot",
    "clp_cot_lag_10m", "clp_cot_lag_20m",
    "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m",
    "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
]
# Group D — time cyclic (6)
FEATURES_TIME = [
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "month_sin", "month_cos",
]
# Group E — future deterministic (4)
FEATURES_FUTURE = [
    "ghi_cs_t60",        # clearsky GHI at t+60  (astronomy only, no leakage)
    "elev_sin_t60",      # sin(solar elevation at t+60)
    "smart_persist",     # kt_now * ghi_cs_t60          (point SP baseline)
    "smart_persist_avg", # kt_now * mean(ghi_cs t+10..t+60)  (avg SP baseline)
]

FEATURES = FEATURES_GHI + FEATURES_KT + FEATURES_CLP + FEATURES_TIME + FEATURES_FUTURE
assert len(FEATURES) == 50, f"Expected 50 features, got {len(FEATURES)}"

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
# SQL  — follows v10/v11 pattern:
#   CTE 1 : add kt_point to ALL rows (no filter)
#   CTE 2 : add fine lags / kt windows / LEAD (no filter) so LAGs are
#            calendar-correct even at overnight/gap boundaries
#   Final  : apply model-ready filter LAST
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
            -- fine CLP lags (view only stores 60m and roll-180m)
            LAG(clp_cot, 1) OVER (ORDER BY ts_wib) AS clp_cot_lag_10m,
            LAG(clp_cot, 2) OVER (ORDER BY ts_wib) AS clp_cot_lag_20m,
            LAG(clp_cot, 3) OVER (ORDER BY ts_wib) AS clp_cot_lag_30m,
            -- 20-min GHI lag (view stores 10/30/60/120/180m)
            LAG(ghi_now, 2) OVER (ORDER BY ts_wib) AS ghi_lag_20m,
            -- kt lags and rolling stats
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
            -- LEAD for average target  (t+10 .. t+60)
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
# Astronomical helpers (same formula as v9/v10)
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
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(df):
    out = df.copy()
    ts = pd.DatetimeIndex(out[TIME_COL])

    # kt_now (clearness index at anchor)
    cs_now = clearsky_simple(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    # CLP deltas
    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    # Acceleration (2nd difference, 20 min window)
    out["accel_ghi_20m"]      = out["ghi_now"]  - 2.0 * out["ghi_lag_10m"]      + out["ghi_lag_20m"]
    out["accel_kt_20m"]       = out["kt_now"]   - 2.0 * out["kt_lag_10m"]       + out["kt_lag_20m"]
    out["accel_clp_cot_20m"]  = out["clp_cot"]  - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    # Day-of-year cyclic (view has hour/month sin-cos but not doy)
    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy  / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy  / 365.25)

    # Future deterministic features (pure astronomy — no leakage)
    ts_t60 = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60 = solar_elevation_deg(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"]   = clearsky_simple(elev_t60)

    # Average clearsky t+10..t+60 (used in smart-persist for avg target)
    cs_steps = []
    for step in range(1, 7):
        ts_f = out[TIME_COL] + pd.Timedelta(minutes=step * 10)
        cs_steps.append(clearsky_simple(solar_elevation_deg(ts_f)))
    out["ghi_cs_avg_t10_t60"] = np.column_stack(cs_steps).mean(axis=1)

    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_avg_t10_t60"]

    # Targets
    out[TARGET_POINT] = out["target_ghi_1h_ahead"].copy()

    lead_cols = ["ghi_lead_10m", "ghi_lead_20m", "ghi_lead_30m",
                 "ghi_lead_40m", "ghi_lead_50m", "ghi_lead_60m"]
    leads     = out[lead_cols]
    all_valid = (leads.notna().all(axis=1)
                 & leads.apply(lambda c: c.between(0, 1400)).all(axis=1))
    out[TARGET_AVG] = np.where(all_valid, leads.mean(axis=1), np.nan)

    # Extra filter: sun > 5 deg at t+60 (view already enforces anchor sun > 5)
    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))

    # Residual targets for LightGBM
    out[DELTA_POINT] = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG]   = out[TARGET_AVG]   - out["ghi_now"]

    return out


# ---------------------------------------------------------------------------
# Split masks
# ---------------------------------------------------------------------------
def split_masks(df):
    ts = df[TIME_COL]
    train = ts <  pd.Timestamp(TRAIN_END)
    valid = (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END))
    test  = ts >= pd.Timestamp(VALID_END)
    return train, valid, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def lgbm_pipe(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=6000, learning_rate=0.02,
        num_leaves=39, min_child_samples=70,
        reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
        random_state=seed, n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("m",   reg),
    ])


def fit_lgbm(pipe, x_tr, y_tr, x_es, y_es):
    pipe.fit(x_tr, y_tr,
             m__eval_set=[(x_es, y_es)],
             m__eval_metric="rmse",
             m__callbacks=[lgb.early_stopping(150, verbose=False)])
    return pipe


def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )


def fit_catboost(m, x_tr, y_tr, x_es, y_es):
    # .astype(float) converts pandas <NA> to np.nan, which CatBoost accepts
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_es.astype(float).values, y_es.astype(float).values),
          early_stopping_rounds=150)
    return m


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, sp_pred, model, target):
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp_rmse = float(np.sqrt(mean_squared_error(y_true, sp_pred)))
    skill = (1.0 - rmse / sp_rmse) if sp_rmse > 0 else float("nan")
    return {
        "model": model, "target": target, "n": len(y_true),
        "r2": round(r2, 4), "mae": round(mae, 1),
        "rmse": round(rmse, 1), "skill_vs_sp": round(skill, 4),
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

    print(f"Total rows     : {len(df):,}")
    print(f"Date range     : {df[TIME_COL].min().date()} to {df[TIME_COL].max().date()}")

    # Per-target subsets (also enforce sun > 5 deg at t+60)
    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna()             & df["sun_gt5_t60"]].copy()

    print(f"Rows point target : {len(df_pt):,}")
    print(f"Rows avg target   : {len(df_av):,}")

    results = []

    for (df_use, target_col, delta_col, sp_col, tgt_name) in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist",     "point_t60"),
        (df_av, TARGET_AVG,   DELTA_AVG,   "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n{'='*60}")
        print(f"TARGET: {tgt_name}")
        print(f"{'='*60}")

        tr_m, va_m, te_m = split_masks(df_use)
        print(f"  train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

        x_tr  = df_use.loc[tr_m, FEATURES]
        x_va  = df_use.loc[va_m, FEATURES]
        x_te  = df_use.loc[te_m, FEATURES]
        y_tr  = df_use.loc[tr_m, target_col]
        y_va  = df_use.loc[va_m, target_col]
        y_te  = df_use.loc[te_m, target_col]
        yd_tr = df_use.loc[tr_m, delta_col]
        yd_va = df_use.loc[va_m, delta_col]

        ghi_now_te = df_use.loc[te_m, "ghi_now"].values
        sp_te = np.clip(df_use.loc[te_m, sp_col].values, PRED_MIN, PRED_MAX)

        # Smart-persistence baseline
        r = compute_metrics(y_te, sp_te, sp_te, "smart_persistence", tgt_name)
        results.append(r)
        print(f"  smart_persistence  R2={r['r2']:.4f}  MAE={r['mae']:.1f}")

        # PRIMARY: LightGBM residual
        print("  Training LightGBM residual...")
        lgbm = lgbm_pipe()
        fit_lgbm(lgbm, x_tr, yd_tr, x_va, yd_va)
        best_it = lgbm.named_steps["m"].best_iteration_

        lgbm_pred = np.clip(ghi_now_te + lgbm.predict(x_te), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, lgbm_pred, sp_te, "lgbm_residual", tgt_name)
        r["best_iter"] = best_it
        results.append(r)
        print(f"  lgbm_residual  iter={best_it:4d}  R2={r['r2']:.4f}  "
              f"MAE={r['mae']:.1f}  skill={r['skill_vs_sp']:.4f}")

        # SENSITIVITY: CatBoost direct
        print("  Training CatBoost direct...")
        cb = catboost_model()
        fit_catboost(cb, x_tr, y_tr, x_va, y_va)
        best_it_cb = cb.get_best_iteration()

        cb_pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, cb_pred, sp_te, "catboost_direct", tgt_name)
        r["best_iter"] = best_it_cb
        results.append(r)
        print(f"  catboost_direct iter={best_it_cb:4d}  R2={r['r2']:.4f}  "
              f"MAE={r['mae']:.1f}  skill={r['skill_vs_sp']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "ghi_1h_R1_results.csv", index=False)

    # -----------------------------------------------------------------------
    # Walk-forward 5-fold  (LightGBM residual x point target only)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("WALK-FORWARD 5-FOLD  (lgbm_residual x point_t60)")
    print(f"{'='*60}")

    wf_rows = []
    for fold_idx, (test_start, test_end) in enumerate(FOLDS, 1):
        ts_start = pd.Timestamp(test_start)
        ts_end   = pd.Timestamp(test_end) if test_end else pd.Timestamp("2099-01-01")

        ts_col = df_pt[TIME_COL]
        tr_all = df_pt[ts_col < ts_start]
        te_wf  = df_pt[(ts_col >= ts_start) & (ts_col < ts_end)]

        if len(tr_all) < 5000 or len(te_wf) < 100:
            print(f"  Fold {fold_idx}: too few rows, skip")
            continue

        es_cut = ts_start - pd.DateOffset(months=ES_MONTHS)
        tr_eff = tr_all[tr_all[TIME_COL] <  es_cut]
        tr_es  = tr_all[tr_all[TIME_COL] >= es_cut]

        pipe = lgbm_pipe()
        fit_lgbm(pipe,
                 tr_eff[FEATURES], tr_eff[DELTA_POINT],
                 tr_es[FEATURES],  tr_es[DELTA_POINT])
        best_it = pipe.named_steps["m"].best_iteration_

        ghi_now_wf = te_wf["ghi_now"].values
        sp_wf   = np.clip(te_wf["smart_persist"].values, PRED_MIN, PRED_MAX)
        y_te_wf = te_wf[TARGET_POINT].values
        pred_wf = np.clip(ghi_now_wf + pipe.predict(te_wf[FEATURES]), PRED_MIN, PRED_MAX)

        r2_wf   = float(r2_score(y_te_wf, pred_wf))
        mae_wf  = float(mean_absolute_error(y_te_wf, pred_wf))
        rmse_wf = float(np.sqrt(mean_squared_error(y_te_wf, pred_wf)))
        sp_rmse = float(np.sqrt(mean_squared_error(y_te_wf, sp_wf)))
        skill_wf = 1.0 - rmse_wf / sp_rmse if sp_rmse > 0 else float("nan")

        period = test_start[:7] + ".." + (test_end[:7] if test_end else "end")
        print(f"  Fold {fold_idx} [{period}]  n_tr={len(tr_eff):6,}  n_es={len(tr_es):5,}  "
              f"n_te={len(te_wf):5,}  iter={best_it:4d}  "
              f"R2={r2_wf:.4f}  MAE={mae_wf:.1f}  skill={skill_wf:.4f}")

        wf_rows.append({
            "fold": fold_idx, "period": period,
            "n_train_eff": len(tr_eff), "n_es": len(tr_es), "n_test": len(te_wf),
            "best_iter": best_it,
            "r2": r2_wf, "mae": mae_wf, "rmse": rmse_wf, "skill_vs_sp": skill_wf,
        })

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUTPUT_DIR / "ghi_1h_R1_wf_folds.csv", index=False)

    print("\n--- Walk-forward summary ---")
    for col in ["r2", "mae", "rmse", "skill_vs_sp"]:
        print(f"  {col:<14}: {wf_df[col].mean():.4f} +/- {wf_df[col].std():.4f}")
    wf_df.describe().to_csv(OUTPUT_DIR / "ghi_1h_R1_wf_summary.csv")

    print(f"\n--- HEADLINE RESULTS (test 2025) ---")
    print(results_df[["model", "target", "r2", "mae", "rmse", "skill_vs_sp"]].to_string(index=False))
    print(f"\nAll outputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
