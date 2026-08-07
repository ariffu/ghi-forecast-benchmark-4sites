#!/usr/bin/env python3
"""
R1 Harmonised Benchmark — KALBAR GHI 1-hour-ahead forecasting.
Konfigurasi seragam dengan bengkulu_ghi_julius & DuckDB_jambi & Duckdb_Banten.

  Model    : LightGBM residual (PRIMARY) | CatBoost direct (SENSITIVITY)
  Features : resep §3.2 — 50 fitur lean (GHI history + CLP dynamics + time + future)
  Targets  : (a) GHI titik t+60 (ghi_target_60m); (b) GHI rata-rata t+10..t+60 (ghi_target_avg60m)
  Data     : training_ghi_1h_direct (2022-2025, 10-menit) — Kalbar mulai 2022, bukan 2021
  Split    : train <2024-01-01 | val 2024 | test 2025
  Filter   : sun_altitude > 5° di anchor & di t+60 (astronomi); GHI 0-1400; anchor_valid=true
  Metrics  : R², MAE, RMSE, skill = 1 - RMSE/RMSE_SP (test 2025)
             + walk-forward 5-fold (LGBM residual × point target)

Catatan Kalbar:
  - Data mulai 2022 (bukan 2021); anchor_valid untuk filter gap ±30dtk
  - Column naming: ghi_final (vs ghi_now), ghi_lag10m (vs ghi_lag_10m), dst
  - sun_altitude_future sudah pre-computed (deterministik, t+60)
  - Dua target: ghi_target_60m (titik) & ghi_target_avg60m (rata-rata t+10..t+60)

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_kalbar_R1_benchmark.py
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

OUTPUT_DIR = Path("outputs_R1_kalbar")
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
STATION_LAT_DEG  = -0.0356
STATION_LON_DEG  = 109.3384
WIB_MERIDIAN_DEG = 105.0

TIME_COL  = "timestamp_wib"
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

# Fitur 50-lean (mapping Kalbar columns → standard names)
FEATURES_GHI = [
    "ghi_now",           # ghi_final
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",  # ghi_lag10m, ghi_lag20m, ghi_lag30m
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",  # 120m & 180m derived via rolling
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",  # 180m rolling
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
    "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",  # 60m derived via shift
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",  # derived
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
    h = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    ha = ((h + 4.0 * (lon - meridian) / 60.0) - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl)) +
             np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1, 1)))


def clearsky_simple(e):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(e)), 0.0)


def add_features(df):
    out = df.copy()

    # Rename untuk konsistensi naming
    out["ghi_now"] = out["ghi_final"]
    out["kt_now"] = out["kt"]
    out["clp_cot"] = out["CLOT_mean"]
    out["clp_cth_m"] = out["CLTH_mean"]
    out["clp_ctt_k"] = out["CLTT_mean"]
    out["clp_cer"] = out["CLER_23_mean"]
    out["clp_cloud_present"] = out["clp_cloud_present_int"].astype(float)

    # Rename lag & rolling (Kalbar uses different naming)
    out["ghi_lag_10m"] = out["ghi_lag10m"]
    out["ghi_lag_20m"] = out["ghi_lag20m"]
    out["ghi_lag_30m"] = out["ghi_lag30m"]
    out["ghi_lag_60m"] = out["ghi_lag60m"]
    out["kt_lag_10m"] = out["kt_lag10m"]
    out["kt_lag_20m"] = out["kt_lag20m"]
    out["kt_lag_30m"] = out["kt_lag30m"]
    out["kt_lag_60m"] = out["kt_lag60m"]
    out["kt_roll30m_mean"] = out["kt_roll30m_mean"]
    out["kt_roll30m_std"] = out["kt_roll30m_std"]
    out["kt_roll60m_mean"] = out["kt_roll60m_mean"]
    out["kt_roll60m_std"] = out["kt_roll60m_std"]
    out["clp_cot_lag_10m"] = out["clot_lag10m"]

    # clp_cot_lag_20m: interpolasi dari lag10m, lag30m
    out["clp_cot_lag_20m"] = out["clot_lag10m"] * 0.67 + out["clp_cot"] * 0.33
    out["clp_cot_lag_30m"] = out["clot_lag30m"]
    out["clp_cot_lag_60m"] = out["clp_cot"].shift(6)  # 6 × 10-min = 60m

    # Derived features (lags via shift, rolling stats)
    out["ghi_lag_120m"] = out["ghi_now"].shift(12)  # 12 × 10-min steps = 120 min
    out["ghi_lag_180m"] = out["ghi_now"].shift(18)  # 18 × 10-min steps = 180 min
    out["ghi_roll_30m_mean"] = out["ghi_now"].rolling(window=3, center=False).mean()
    out["ghi_roll_30m_std"] = out["ghi_now"].rolling(window=3, center=False).std()
    out["ghi_roll_60m_mean"] = out["ghi_now"].rolling(window=6, center=False).mean()
    out["ghi_roll_60m_std"] = out["ghi_now"].rolling(window=6, center=False).std()
    out["ghi_roll_180m_mean"] = out["ghi_now"].rolling(window=18, center=False).mean()
    out["ghi_roll_180m_std"] = out["ghi_now"].rolling(window=18, center=False).std()
    out["clp_cot_roll_180m_mean"] = out["clp_cot"].rolling(window=18, center=False).mean()

    # Delta features
    out["clp_cot_delta_10m"] = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"] = out["delta_clot_30m"]
    out["clp_cot_delta_60m"] = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["ghi_delta_10m"] = out["ghi_now"] - out["ghi_lag_10m"]
    out["ghi_delta_60m"] = out["ghi_now"] - out["ghi_lag_60m"]

    # Acceleration features
    out["accel_ghi_20m"] = out["ghi_now"] - 2 * out["ghi_lag_10m"] + out["ghi_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    # Time cyclic (Kalbar sudah punya hour_sin/cos, doy_sin/cos, tp perlu month_sin/cos)
    ts = pd.DatetimeIndex(out[TIME_COL])
    mo = ts.month.values.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * mo / 12)
    out["month_cos"] = np.cos(2 * np.pi * mo / 12)

    # Future features (deterministic)
    out["ghi_cs_t60"] = out["ghi_clearsky_future"]
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(out["sun_altitude_future"])), 0.0)

    # Smart persistence
    out["smart_persist"] = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out.get("ghi_cs_avg_t10_t60", out["ghi_cs_t60"])

    # Target
    out[TARGET_POINT] = out["ghi_target_60m"].copy()
    out[TARGET_AVG] = out["ghi_target_avg60m"].copy()
    out[DELTA_POINT] = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG] = out[TARGET_AVG] - out["ghi_now"]

    return out


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


def lgbm_pipe(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=6000,
        learning_rate=0.02,
        num_leaves=39,
        min_child_samples=70,
        reg_alpha=0.2,
        reg_lambda=2.5,
        colsample_bytree=0.82,
        subsample=0.85,
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("m", reg)])


def fit_lgbm(pipe, xt, yt, xe, ye):
    pipe.fit(
        xt, yt,
        m__eval_set=[(xe, ye)],
        m__eval_metric="rmse",
        m__callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    return pipe


def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(
        iterations=4000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


def fit_catboost(m, xt, yt, xe, ye):
    m.fit(
        xt.astype(float).values,
        yt.astype(float).values,
        eval_set=(xe.astype(float).values, ye.astype(float).values),
        early_stopping_rounds=150,
    )
    return m


def metr(y, p, sp, model, target):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spr = float(np.sqrt(mean_squared_error(y, sp)))
    return {
        "model": model,
        "target": target,
        "n": len(y),
        "r2": round(float(r2_score(y, p)), 4),
        "mae": round(float(mean_absolute_error(y, p)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / spr if spr > 0 else float("nan"), 4),
    }


def main():
    import duckdb
    print("Loading Kalbar data...")
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT * FROM training_ghi_1h_direct
        WHERE anchor_valid AND ghi_final BETWEEN 0 AND 1400
        ORDER BY timestamp_wib
    """).df()
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    print(f"Total rows: {len(df):,} | {df[TIME_COL].min().date()}..{df[TIME_COL].max().date()}")

    # Filter sun_altitude > 5°
    df_pt = df[
        df[TARGET_POINT].between(0, 1400) &
        (df["sun_altitude"] > 5.0) &
        (df["sun_altitude_future"] > 5.0)
    ].copy()
    df_av = df[
        df[TARGET_AVG].notna() &
        (df["sun_altitude"] > 5.0) &
        (df["sun_altitude_future"] > 5.0)
    ].copy()

    print(f"Rows after filter: point={len(df_pt):,} avg={len(df_av):,}")

    results = []
    for du, tc, dc, spc, tg in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist", "point_t60"),
        (df_av, TARGET_AVG, DELTA_AVG, "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n{'='*56}\nTARGET: {tg}\n{'='*56}")
        tm, vm, em = split_masks(du)
        print(f"  train={tm.sum():,} val={vm.sum():,} test={em.sum():,}")

        xt, xv, xe = du.loc[tm, FEATURES], du.loc[vm, FEATURES], du.loc[em, FEATURES]
        yt, yv, ye = du.loc[tm, tc], du.loc[vm, tc], du.loc[em, tc]
        ydt, ydv = du.loc[tm, dc], du.loc[vm, dc]
        gne = du.loc[em, "ghi_now"].values
        spe = np.clip(du.loc[em, spc].values, PRED_MIN, PRED_MAX)

        # Smart persistence baseline
        results.append(metr(ye, spe, spe, "smart_persistence", tg))
        print(f"  SP R2={results[-1]['r2']:.4f} MAE={results[-1]['mae']:.1f}")

        # LGBM residual
        lg = lgbm_pipe()
        fit_lgbm(lg, xt, ydt, xv, ydv)
        bi = lg.named_steps["m"].best_iteration_
        pr = np.clip(gne + lg.predict(xe), PRED_MIN, PRED_MAX)
        r = metr(ye, pr, spe, "lgbm_residual", tg)
        r["best_iter"] = bi
        results.append(r)
        print(f"  LGBM iter={bi} R2={r['r2']:.4f} MAE={r['mae']:.1f} skill={r['skill_vs_sp']:.4f}")

        # CatBoost direct
        cb = catboost_model()
        fit_catboost(cb, xt, yt, xv, yv)
        bic = cb.get_best_iteration()
        cp = np.clip(cb.predict(xe.astype(float).values), PRED_MIN, PRED_MAX)
        r = metr(ye, cp, spe, "catboost_direct", tg)
        r["best_iter"] = bic
        results.append(r)
        print(f"  CatB iter={bic} R2={r['r2']:.4f} MAE={r['mae']:.1f} skill={r['skill_vs_sp']:.4f}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "ghi_1h_R1_results.csv", index=False)

    print(f"\n{'='*56}\nWALK-FORWARD 5-FOLD (lgbm_residual x point)\n{'='*56}")
    wf = []
    for fi, (t0, t1) in enumerate(FOLDS, 1):
        s = pd.Timestamp(t0)
        e = pd.Timestamp(t1) if t1 else pd.Timestamp("2099-01-01")
        c = df_pt[TIME_COL]
        tra = df_pt[c < s]
        te = df_pt[(c >= s) & (c < e)]

        if len(tra) < 5000 or len(te) < 100:
            print(f"  Fold {fi}: skip (train={len(tra)}, test={len(te)})")
            continue

        cut = s - pd.DateOffset(months=ES_MONTHS)
        tre = tra[tra[TIME_COL] < cut]
        tes = tra[tra[TIME_COL] >= cut]

        p = lgbm_pipe()
        fit_lgbm(p, tre[FEATURES], tre[DELTA_POINT], tes[FEATURES], tes[DELTA_POINT])
        bi = p.named_steps["m"].best_iteration_

        pr = np.clip(te["ghi_now"].values + p.predict(te[FEATURES]), PRED_MIN, PRED_MAX)
        yv = te[TARGET_POINT].values
        sp = np.clip(te["smart_persist"].values, PRED_MIN, PRED_MAX)

        rmse = float(np.sqrt(mean_squared_error(yv, pr)))
        spr = float(np.sqrt(mean_squared_error(yv, sp)))

        per = t0[:7] + ".." + (t1[:7] if t1 else "end")
        wf.append({
            "fold": fi,
            "period": per,
            "n_train_eff": len(tre),
            "n_test": len(te),
            "best_iter": bi,
            "r2": float(r2_score(yv, pr)),
            "mae": float(mean_absolute_error(yv, pr)),
            "rmse": rmse,
            "skill_vs_sp": 1.0 - rmse / spr if spr > 0 else float("nan"),
        })
        print(f"  Fold {fi} [{per}] n_tr={len(tre):,} n_te={len(te):,} iter={bi} R2={wf[-1]['r2']:.4f} MAE={wf[-1]['mae']:.1f} skill={wf[-1]['skill_vs_sp']:.4f}")

    wfdf = pd.DataFrame(wf)
    wfdf.to_csv(OUTPUT_DIR / "ghi_1h_R1_wf_folds.csv", index=False)

    print("\n--- WF summary ---")
    for c in ["r2", "mae", "rmse", "skill_vs_sp"]:
        print(f"  {c:<12}: {wfdf[c].mean():.4f} +/- {wfdf[c].std():.4f}")

    print("\n--- HEADLINE (test 2025) ---")
    print(pd.DataFrame(results)[["model", "target", "r2", "mae", "rmse", "skill_vs_sp"]].to_string(index=False))
    print(f"\nOutputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
