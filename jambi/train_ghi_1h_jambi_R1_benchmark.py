#!/usr/bin/env python3
"""
R1 Harmonised Benchmark — JAMBI GHI 1-hour-ahead forecasting.
Port 1:1 dari bengkulu_ghi_julius/train_ghi_1h_bengkulu_R1_benchmark.py —
konfigurasi, fitur (50), model, filter, metrik, dan walk-forward identik.

  Model    : LightGBM residual (PRIMARY) | CatBoost direct (SENSITIVITY)
  Features : resep §3.2 — 50 fitur, tanpa surface-met / SYNOP
  Targets  : (a) GHI titik t+60; (b) GHI rata-rata t+10..t+60
  Data     : dfm_with_clp_stats.parquet (10-menit, 2021-2025)
  Split    : train <2024 | val 2024 | test 2025
  Filter   : elev>5 deg di anchor & t+60 (astronomi sederhana, sama dgn Bengkulu);
             riwayat 3 jam kontinu (18 step x 600 dtk +/-30 dtk); GHI 0-1400
  Metrics  : R2, MAE, RMSE, skill = 1 - RMSE/RMSE_SP (test 2025)
             + walk-forward 5-fold (LGBM residual x point target)

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_jambi_R1_benchmark.py
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs_R1_jambi")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -1.5833
STATION_LON_DEG  = 103.6667
WIB_MERIDIAN_DEG = 105.0

TIME_COL  = "ts"
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
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


def build_dataset():
    df = pd.read_parquet("dfm_with_clp_stats.parquet")
    df = df.sort_values("ts").reset_index(drop=True)

    out = pd.DataFrame()
    out[TIME_COL] = pd.to_datetime(df["ts"])
    out["ghi_now"] = df["ghi_consolidated"].astype(float)
    out["clp_cot"] = df["cloud_optical_thickness"].astype(float)
    out["clp_cth_m"] = df["cloud_top_height_m"].astype(float)
    out["clp_ctt_k"] = df["cloud_top_temp_c"].astype(float) + 273.15
    out["clp_cer"] = df["cloud_eff_radius_um"].astype(float)
    out["clp_cloud_present"] = df["sat_cloud_present"].astype(float)

    # Kontinuitas grid 10-menit (+/-30 dtk) — pengganti has_continuous_3h_history
    # ts adalah datetime64[us] — bagi 1e6 utk detik (BUKAN 1e9; lihat memory environment)
    ts_sec = (out[TIME_COL].values.astype("int64") / 1_000_000).astype(np.int64)
    dstep = np.diff(ts_sec)
    ok_step = np.abs(dstep - 600) <= 30                      # step i-1 -> i valid
    ok_step = np.concatenate([[False], ok_step])             # align ke index baris
    ok_cum = pd.Series(ok_step.astype(int))
    cont_3h_hist = ok_cum.rolling(18, min_periods=18).sum().eq(18)   # 18 step ke belakang
    ok_fwd = pd.Series(np.concatenate([ok_step[1:], [False]]).astype(int))
    cont_1h_fwd = ok_fwd.rolling(6, min_periods=6).sum().shift(-5).eq(6)  # 6 step ke depan
    out["has_continuous_3h_history"] = cont_3h_hist.values
    out["has_continuous_1h_forward"] = cont_1h_fwd.values

    # Astronomi sederhana (harmonis Bengkulu)
    out["solar_elev_deg"] = solar_elevation_deg(out[TIME_COL])
    cs_now = clearsky_simple(out["solar_elev_deg"].values)
    out["kt_now"] = out["ghi_now"].values / np.maximum(
        1100.0 * np.maximum(np.sin(np.deg2rad(out["solar_elev_deg"].values)), 0.02), 20.0)

    # Lags/rolls/deltas GHI (row-based — sah karena filter kontinuitas)
    g = out["ghi_now"]
    for n, name in [(1, "10m"), (2, "20m"), (3, "30m"), (6, "60m"), (12, "120m"), (18, "180m")]:
        out[f"ghi_lag_{name}"] = g.shift(n)
    out["ghi_roll_30m_mean"] = g.rolling(3).mean()
    out["ghi_roll_30m_std"] = g.rolling(3).std()
    out["ghi_roll_60m_mean"] = g.rolling(6).mean()
    out["ghi_roll_60m_std"] = g.rolling(6).std()
    out["ghi_roll_180m_mean"] = g.rolling(18).mean()
    out["ghi_roll_180m_std"] = g.rolling(18).std()
    out["ghi_delta_10m"] = g - out["ghi_lag_10m"]
    out["ghi_delta_60m"] = g - out["ghi_lag_60m"]
    out["accel_ghi_20m"] = g - 2.0 * out["ghi_lag_10m"] + out["ghi_lag_20m"]

    k = out["kt_now"]
    for n, name in [(1, "10m"), (2, "20m"), (3, "30m"), (6, "60m")]:
        out[f"kt_lag_{name}"] = k.shift(n)
    out["kt_roll30m_mean"] = k.rolling(3).mean()
    out["kt_roll30m_std"] = k.rolling(3).std()
    out["kt_roll60m_mean"] = k.rolling(6).mean()
    out["accel_kt_20m"] = k - 2.0 * out["kt_lag_10m"] + out["kt_lag_20m"]

    c = out["clp_cot"]
    for n, name in [(1, "10m"), (2, "20m"), (3, "30m"), (6, "60m")]:
        out[f"clp_cot_lag_{name}"] = c.shift(n)
    out["clp_cot_roll_180m_mean"] = c.rolling(18, min_periods=9).mean()
    out["clp_cot_delta_10m"] = c - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"] = c - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"] = c - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = c - out["clp_cot_roll_180m_mean"]
    out["accel_clp_cot_20m"] = c - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    ts = pd.DatetimeIndex(out[TIME_COL])
    out["hour_sin"] = np.sin(2 * np.pi * (ts.hour + ts.minute / 60.0) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * (ts.hour + ts.minute / 60.0) / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * ts.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * ts.month / 12.0)

    # Future deterministic
    ts_t60 = out[TIME_COL] + pd.Timedelta(minutes=60)
    elev_t60 = solar_elevation_deg(ts_t60)
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"] = clearsky_simple(elev_t60)
    cs_steps = []
    for step in range(1, 7):
        ts_f = out[TIME_COL] + pd.Timedelta(minutes=step * 10)
        cs_steps.append(clearsky_simple(solar_elevation_deg(ts_f)))
    out["ghi_cs_avg_t10_t60"] = np.column_stack(cs_steps).mean(axis=1)
    out["smart_persist"] = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_avg_t10_t60"]

    # Targets via LEAD (sah karena filter kontinuitas 1h forward)
    leads = pd.DataFrame({f"lead{n}": g.shift(-n) for n in range(1, 7)})
    out[TARGET_POINT] = leads["lead6"]
    all_valid = leads.notna().all(axis=1) & leads.apply(
        lambda col: col.between(0, 1400)).all(axis=1)
    out[TARGET_AVG] = np.where(all_valid, leads.mean(axis=1), np.nan)

    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))
    out[DELTA_POINT] = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG] = out[TARGET_AVG] - out["ghi_now"]

    # Filter model-ready (paralel dgn view Bengkulu)
    mask = (out["has_continuous_3h_history"]
            & out["has_continuous_1h_forward"]
            & (out["solar_elev_deg"] > 5.0)
            & out["ghi_now"].between(0, 1400))
    return out[mask].reset_index(drop=True)


def split_masks(df):
    ts = df[TIME_COL]
    train = ts < pd.Timestamp(TRAIN_END)
    valid = (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END))
    test = ts >= pd.Timestamp(VALID_END)
    return train, valid, test


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
        ("m", reg),
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
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_es.astype(float).values, y_es.astype(float).values),
          early_stopping_rounds=150)
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
    print("Building harmonised dataset (Jambi)...", flush=True)
    df = build_dataset()
    print(f"Total rows     : {len(df):,}", flush=True)
    print(f"Date range     : {df[TIME_COL].min().date()} to {df[TIME_COL].max().date()}", flush=True)

    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna() & df["sun_gt5_t60"]].copy()
    print(f"Rows point target : {len(df_pt):,}", flush=True)
    print(f"Rows avg target   : {len(df_av):,}", flush=True)

    results = []
    for (df_use, target_col, delta_col, sp_col, tgt_name) in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist", "point_t60"),
        (df_av, TARGET_AVG, DELTA_AVG, "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n=== TARGET: {tgt_name} ===", flush=True)
        tr_m, va_m, te_m = split_masks(df_use)
        print(f"  train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}", flush=True)

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
        print(f"  smart_persistence  R2={r['r2']:.4f}  MAE={r['mae']:.1f}", flush=True)

        lgbm = lgbm_pipe()
        fit_lgbm(lgbm, x_tr, yd_tr, x_va, yd_va)
        best_it = lgbm.named_steps["m"].best_iteration_
        lgbm_pred = np.clip(ghi_now_te + lgbm.predict(x_te), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, lgbm_pred, sp_te, "lgbm_residual", tgt_name)
        r["best_iter"] = best_it
        results.append(r)
        print(f"  lgbm_residual  iter={best_it}  R2={r['r2']:.4f}  MAE={r['mae']:.1f}  "
              f"skill={r['skill_vs_sp']:.4f}", flush=True)

        cb = catboost_model()
        fit_catboost(cb, x_tr, y_tr, x_va, y_va)
        best_it_cb = cb.get_best_iteration()
        cb_pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, cb_pred, sp_te, "catboost_direct", tgt_name)
        r["best_iter"] = best_it_cb
        results.append(r)
        print(f"  catboost_direct iter={best_it_cb}  R2={r['r2']:.4f}  MAE={r['mae']:.1f}  "
              f"skill={r['skill_vs_sp']:.4f}", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "ghi_1h_R1_results.csv", index=False)

    print("\n=== WALK-FORWARD 5-FOLD (lgbm_residual x point_t60) ===", flush=True)
    wf_rows = []
    for fold_idx, (test_start, test_end) in enumerate(FOLDS, 1):
        ts_start = pd.Timestamp(test_start)
        ts_end = pd.Timestamp(test_end) if test_end else pd.Timestamp("2099-01-01")
        ts_col = df_pt[TIME_COL]
        tr_all = df_pt[ts_col < ts_start]
        te_wf = df_pt[(ts_col >= ts_start) & (ts_col < ts_end)]
        if len(tr_all) < 5000 or len(te_wf) < 100:
            print(f"  Fold {fold_idx}: too few rows, skip", flush=True)
            continue
        es_cut = ts_start - pd.DateOffset(months=ES_MONTHS)
        tr_eff = tr_all[tr_all[TIME_COL] < es_cut]
        tr_es = tr_all[tr_all[TIME_COL] >= es_cut]

        pipe = lgbm_pipe()
        fit_lgbm(pipe, tr_eff[FEATURES], tr_eff[DELTA_POINT],
                 tr_es[FEATURES], tr_es[DELTA_POINT])
        best_it = pipe.named_steps["m"].best_iteration_

        ghi_now_wf = te_wf["ghi_now"].values
        sp_wf = np.clip(te_wf["smart_persist"].values, PRED_MIN, PRED_MAX)
        y_te_wf = te_wf[TARGET_POINT].values
        pred_wf = np.clip(ghi_now_wf + pipe.predict(te_wf[FEATURES]), PRED_MIN, PRED_MAX)

        r2_wf = float(r2_score(y_te_wf, pred_wf))
        mae_wf = float(mean_absolute_error(y_te_wf, pred_wf))
        rmse_wf = float(np.sqrt(mean_squared_error(y_te_wf, pred_wf)))
        sp_rmse = float(np.sqrt(mean_squared_error(y_te_wf, sp_wf)))
        skill_wf = 1.0 - rmse_wf / sp_rmse if sp_rmse > 0 else float("nan")

        period = test_start[:7] + ".." + (test_end[:7] if test_end else "end")
        print(f"  Fold {fold_idx} [{period}]  n_tr={len(tr_eff):,}  n_te={len(te_wf):,}  "
              f"iter={best_it}  R2={r2_wf:.4f}  MAE={mae_wf:.1f}  skill={skill_wf:.4f}", flush=True)
        wf_rows.append({
            "fold": fold_idx, "period": period,
            "n_train_eff": len(tr_eff), "n_es": len(tr_es), "n_test": len(te_wf),
            "best_iter": best_it,
            "r2": r2_wf, "mae": mae_wf, "rmse": rmse_wf, "skill_vs_sp": skill_wf,
        })

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUTPUT_DIR / "ghi_1h_R1_wf_folds.csv", index=False)
    print("\n--- Walk-forward summary ---", flush=True)
    for col in ["r2", "mae", "rmse", "skill_vs_sp"]:
        print(f"  {col:<14}: {wf_df[col].mean():.4f} +/- {wf_df[col].std():.4f}", flush=True)
    wf_df.describe().to_csv(OUTPUT_DIR / "ghi_1h_R1_wf_summary.csv")

    print("\n--- HEADLINE RESULTS (test 2025) ---", flush=True)
    print(results_df[["model", "target", "r2", "mae", "rmse", "skill_vs_sp"]].to_string(index=False), flush=True)
    print(f"\nAll outputs -> {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
