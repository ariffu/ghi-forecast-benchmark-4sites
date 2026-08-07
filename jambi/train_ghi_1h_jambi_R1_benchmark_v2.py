#!/usr/bin/env python3
"""
R1 Harmonised Benchmark v2 -- JAMBI, memakai tabel baru 24-jam-penuh
(ghi_forecast_1h_train_3h_rollback_2021_2025, lihat build_ghi_forecast_1h_rollback_jambi.py
dan Restrukturisasi/09_Audit_Volume_Data_Jambi.md).

Port 1:1 dari train_ghi_1h_bengkulu_R1_benchmark.py -- fitur (50), model, split,
filter, metrik SAMA PERSIS, hanya path DB + koordinat stasiun yang beda. Ini
menggantikan train_ghi_1h_jambi_R1_benchmark.py (v1) yang sumbernya
dfm_with_clp_stats.parquet (dipangkas siang hari, lihat audit 09).

  FAST_MODE=True (default) -- boosting rounds dikurangi supaya selesai cepat
  untuk validasi awal pipeline baru. Set FAST_MODE=False untuk angka final
  (n_estimators/iterations penuh seperti skrip Bengkulu asli, akan makan waktu
  jauh lebih lama).

Run:
    python train_ghi_1h_jambi_R1_benchmark_v2.py
"""
import os
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

FAST_MODE = True   # lihat docstring

LOCAL_DB_PATH = Path("jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb")
SCHEMA        = "jambi_sch"
TABLE         = "ghi_forecast_1h_train_3h_rollback_2021_2025"
OUTPUT_DIR    = Path("outputs_R1_jambi_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -1.5833
STATION_LON_DEG  = 103.6667
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

FOLDS = [
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", None),
]
ES_MONTHS = 3

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
FEATURES_KT = [
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
]
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
FEATURES_TIME = [
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "month_sin", "month_cos",
]
FEATURES_FUTURE = [
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
FEATURES = FEATURES_GHI + FEATURES_KT + FEATURES_CLP + FEATURES_TIME + FEATURES_FUTURE
assert len(FEATURES) == 50, f"Expected 50 features, got {len(FEATURES)}"

TARGET_POINT = "ghi_point_t60"
TARGET_AVG   = "ghi_avg_t10_t60"
DELTA_POINT  = "delta_point"
DELTA_AVG    = "delta_avg"


def connect_data():
    con = duckdb.connect(database=":memory:")
    con.execute(f"ATTACH '{LOCAL_DB_PATH.as_posix()}' AS jambi_db (READ_ONLY)")
    print(f"Data source: LOCAL  ({LOCAL_DB_PATH})")
    return con


def build_sql():
    return f"""
    WITH with_kt AS (
        SELECT
            *,
            ghi_now / GREATEST(
                1100.0 * GREATEST(SIN(RADIANS(solar_elev_deg)), 0.02), 20.0
            ) AS kt_point
        FROM jambi_db.{SCHEMA}.{TABLE}
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
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
            AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
            STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std,
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


def add_features(df):
    out = df.copy()
    ts = pd.DatetimeIndex(out[TIME_COL])

    cs_now = clearsky_simple(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    out["accel_ghi_20m"]     = out["ghi_now"] - 2.0 * out["ghi_lag_10m"] + out["ghi_lag_20m"]
    out["accel_kt_20m"]      = out["kt_now"]  - 2.0 * out["kt_lag_10m"]  + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    doy = ts.dayofyear.values.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

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
    leads = out[lead_cols]
    all_valid = (leads.notna().all(axis=1)
                 & leads.apply(lambda c: c.between(0, 1400)).all(axis=1))
    out[TARGET_AVG] = np.where(all_valid, leads.mean(axis=1), np.nan)

    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))

    out[DELTA_POINT] = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG]   = out[TARGET_AVG]   - out["ghi_now"]

    return out


def split_masks(df):
    ts = df[TIME_COL]
    train = ts < pd.Timestamp(TRAIN_END)
    valid = (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END))
    test = ts >= pd.Timestamp(VALID_END)
    return train, valid, test


def lgbm_pipe(seed=RANDOM_STATE):
    n_est = 800 if FAST_MODE else 6000
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=n_est, learning_rate=0.05 if FAST_MODE else 0.02,
        num_leaves=39, min_child_samples=70,
        reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
        random_state=seed, n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("m", reg),
    ])


def fit_lgbm(pipe, x_tr, y_tr, x_es, y_es):
    patience = 50 if FAST_MODE else 150
    pipe.fit(x_tr, y_tr,
             m__eval_set=[(x_es, y_es)],
             m__eval_metric="rmse",
             m__callbacks=[lgb.early_stopping(patience, verbose=False)])
    return pipe


def catboost_model(seed=RANDOM_STATE):
    iters = 800 if FAST_MODE else 4000
    return CatBoostRegressor(
        iterations=iters, learning_rate=0.05 if FAST_MODE else 0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )


def fit_catboost(m, x_tr, y_tr, x_es, y_es):
    patience = 50 if FAST_MODE else 150
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_es.astype(float).values, y_es.astype(float).values),
          early_stopping_rounds=patience)
    return m


def compute_metrics(y_true, y_pred, sp_pred, model, target):
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp_rmse = float(np.sqrt(mean_squared_error(y_true, sp_pred)))
    skill = (1.0 - rmse / sp_rmse) if sp_rmse > 0 else float("nan")
    return {
        "model": model, "target": target, "n": len(y_true),
        "r2": round(r2, 4), "mae": round(mae, 1),
        "rmse": round(rmse, 1), "skill_vs_sp": round(skill, 4),
    }


def main():
    t0 = time.time()
    print(f"FAST_MODE={FAST_MODE}")
    con = connect_data()
    print("Loading data...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    print(f"  loaded in {time.time()-t0:.1f}s")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    print(f"Total rows     : {len(df):,}")
    print(f"Date range     : {df[TIME_COL].min().date()} to {df[TIME_COL].max().date()}")

    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna() & df["sun_gt5_t60"]].copy()

    print(f"Rows point target : {len(df_pt):,}")
    print(f"Rows avg target   : {len(df_av):,}")

    results = []
    for (df_use, target_col, delta_col, sp_col, tgt_name) in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist", "point_t60"),
        (df_av, TARGET_AVG, DELTA_AVG, "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n=== TARGET: {tgt_name} ===  t={time.time()-t0:.1f}s")
        tr_m, va_m, te_m = split_masks(df_use)
        print(f"  train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

        x_tr = df_use.loc[tr_m, FEATURES]
        x_va = df_use.loc[va_m, FEATURES]
        x_te = df_use.loc[te_m, FEATURES]
        y_tr = df_use.loc[tr_m, target_col]
        y_va = df_use.loc[va_m, target_col]
        y_te = df_use.loc[te_m, target_col]
        yd_tr = df_use.loc[tr_m, delta_col]
        yd_va = df_use.loc[va_m, delta_col]

        ghi_now_te = df_use.loc[te_m, "ghi_now"].values
        sp_te = np.clip(df_use.loc[te_m, sp_col].values, PRED_MIN, PRED_MAX)

        r = compute_metrics(y_te, sp_te, sp_te, "smart_persistence", tgt_name)
        results.append(r)
        print(f"  smart_persistence  R2={r['r2']:.4f}  MAE={r['mae']:.1f}")

        lgbm = lgbm_pipe()
        fit_lgbm(lgbm, x_tr, yd_tr, x_va, yd_va)
        best_it = lgbm.named_steps["m"].best_iteration_
        lgbm_pred = np.clip(ghi_now_te + lgbm.predict(x_te), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, lgbm_pred, sp_te, "lgbm_residual", tgt_name)
        r["best_iter"] = best_it
        results.append(r)
        print(f"  lgbm_residual  iter={best_it}  R2={r['r2']:.4f}  MAE={r['mae']:.1f}  "
              f"skill={r['skill_vs_sp']:.4f}  t={time.time()-t0:.1f}s")

        cb = catboost_model()
        fit_catboost(cb, x_tr, y_tr, x_va, y_va)
        best_it_cb = cb.get_best_iteration()
        cb_pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, cb_pred, sp_te, "catboost_direct", tgt_name)
        r["best_iter"] = best_it_cb
        results.append(r)
        print(f"  catboost_direct iter={best_it_cb}  R2={r['r2']:.4f}  MAE={r['mae']:.1f}  "
              f"skill={r['skill_vs_sp']:.4f}  t={time.time()-t0:.1f}s")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "ghi_1h_R1_results.csv", index=False)

    print(f"\n--- HEADLINE RESULTS (test 2025, FAST_MODE={FAST_MODE}) ---")
    print(results_df[["model", "target", "r2", "mae", "rmse", "skill_vs_sp"]].to_string(index=False))
    print(f"\nAll outputs -> {OUTPUT_DIR}/  total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
