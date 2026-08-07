#!/usr/bin/env python3
"""
R1 Kalbar — re-evaluasi dengan anchor §2.3 (tanpa filter anchor_valid).

Tujuan: verifikasi apakah R² berubah signifikan saat menggunakan set anchor
homogen §2.3 vs pipeline (anchor_valid=1, yang mensyaratkan CLP quality).

Modifikasi vs R1 (train_ghi_1h_kalbar_R1_benchmark.py):
  - SQL: hapus 'anchor_valid', filter hanya sun_altitude>5 & sun_altitude_future>5
  - Python: filter sun_altitude>5 & sun_altitude_future>5 (sudah identik R1)
  - Semua lain IDENTIK: fitur 50, hyperparameter, split

Catatan:
  - §2.3 test target: 22,627 baris
  - Pipeline test aktual: 21,386 baris
  - Script ini akan menghasilkan set antara (tanpa anchor_valid, tanpa CLP check)
    yang mendekati tapi tidak identik dengan §2.3 murni (karena tidak ada
    continuity check per-row di training_ghi_1h_direct)

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_kalbar_R1_sec23_anchor.py
"""
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

OUTPUT_DIR = Path("outputs_R1_kalbar_sec23")
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

# Referensi untuk perbandingan
REF_PIPELINE_TOTAL = 81_851
REF_PIPELINE_TEST  = 21_386
REF_SEC23_TOTAL    = 90_579
REF_SEC23_TEST     = 22_627
REF_R2_LGBM        = 0.7217   # LightGBM R1 (angka di paper, per-horizon t+60)
REF_R2_CB          = 0.728    # CatBoost R1

FOLDS = [
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", None),
]
ES_MONTHS = 3

# Fitur 50-lean (identik R1 Kalbar)
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
    "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
]
FEATURES_TIME = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
FEATURES_FUTURE = ["ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg"]
FEATURES = FEATURES_GHI + FEATURES_KT + FEATURES_CLP + FEATURES_TIME + FEATURES_FUTURE
assert len(FEATURES) == 50, f"Expected 50 features, got {len(FEATURES)}"

TARGET_POINT = "ghi_point_t60"
TARGET_AVG   = "ghi_avg_t10_t60"
DELTA_POINT  = "delta_point"
DELTA_AVG    = "delta_avg"


def add_features(df):
    out = df.copy()

    out["ghi_now"] = out["ghi_final"]
    out["kt_now"] = out["kt"]
    out["clp_cot"] = out["CLOT_mean"]
    out["clp_cth_m"] = out["CLTH_mean"]
    out["clp_ctt_k"] = out["CLTT_mean"]
    out["clp_cer"] = out["CLER_23_mean"]
    out["clp_cloud_present"] = out["clp_cloud_present_int"].astype(float)

    out["ghi_lag_10m"] = out["ghi_lag10m"]
    out["ghi_lag_20m"] = out["ghi_lag20m"]
    out["ghi_lag_30m"] = out["ghi_lag30m"]
    out["ghi_lag_60m"] = out["ghi_lag60m"]
    out["kt_lag_10m"] = out["kt_lag10m"]
    out["kt_lag_20m"] = out["kt_lag20m"]
    out["kt_lag_30m"] = out["kt_lag30m"]
    out["kt_lag_60m"] = out["kt_lag60m"]
    out["clp_cot_lag_10m"] = out["clot_lag10m"]
    out["clp_cot_lag_20m"] = out["clot_lag10m"] * 0.67 + out["clp_cot"] * 0.33
    out["clp_cot_lag_30m"] = out["clot_lag30m"]
    out["clp_cot_lag_60m"] = out["clp_cot"].shift(6)

    out["ghi_lag_120m"] = out["ghi_now"].shift(12)
    out["ghi_lag_180m"] = out["ghi_now"].shift(18)
    out["ghi_roll_30m_mean"] = out["ghi_now"].rolling(window=3, center=False).mean()
    out["ghi_roll_30m_std"] = out["ghi_now"].rolling(window=3, center=False).std()
    out["ghi_roll_60m_mean"] = out["ghi_now"].rolling(window=6, center=False).mean()
    out["ghi_roll_60m_std"] = out["ghi_now"].rolling(window=6, center=False).std()
    out["ghi_roll_180m_mean"] = out["ghi_now"].rolling(window=18, center=False).mean()
    out["ghi_roll_180m_std"] = out["ghi_now"].rolling(window=18, center=False).std()
    out["clp_cot_roll_180m_mean"] = out["clp_cot"].rolling(window=18, center=False).mean()

    out["clp_cot_delta_10m"] = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"] = out["delta_clot_30m"]
    out["clp_cot_delta_60m"] = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["ghi_delta_10m"] = out["ghi_now"] - out["ghi_lag_10m"]
    out["ghi_delta_60m"] = out["ghi_now"] - out["ghi_lag_60m"]

    out["accel_ghi_20m"] = out["ghi_now"] - 2 * out["ghi_lag_10m"] + out["ghi_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    ts = pd.DatetimeIndex(out[TIME_COL])
    mo = ts.month.values.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * mo / 12)
    out["month_cos"] = np.cos(2 * np.pi * mo / 12)

    out["ghi_cs_t60"] = out["ghi_clearsky_future"]
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(out["sun_altitude_future"])), 0.0)
    out["smart_persist"] = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out.get("ghi_cs_avg_t10_t60", out["ghi_cs_t60"])

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
        objective="regression", n_estimators=6000, learning_rate=0.02,
        num_leaves=39, min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
        random_state=seed, n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("m",   reg)])


def fit_lgbm(p, xt, yt, xe, ye):
    p.fit(xt, yt, m__eval_set=[(xe, ye)], m__eval_metric="rmse",
          m__callbacks=[lgb.early_stopping(150, verbose=False)])
    return p


def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(
        iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )


def fit_catboost(m, xt, yt, xe, ye):
    m.fit(xt.astype(float).values, yt.astype(float).values,
          eval_set=(xe.astype(float).values, ye.astype(float).values),
          early_stopping_rounds=150)
    return m


def metr(y, p, sp, model, target):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    spr  = float(np.sqrt(mean_squared_error(y, sp)))
    return {
        "model": model, "target": target, "n": len(y),
        "r2": round(float(r2_score(y, p)), 4),
        "mae": round(float(mean_absolute_error(y, p)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / spr if spr > 0 else float("nan"), 4),
    }


def main():
    print("=" * 65)
    print("  KALBAR R1 — §2.3 ANCHOR SET VERIFICATION")
    print("=" * 65)
    print(f"  SQL filter: sun_altitude>5 & sun_altitude_future>5 (TANPA anchor_valid)")
    print(f"  Referensi pipeline: total={REF_PIPELINE_TOTAL:,}  test={REF_PIPELINE_TEST:,}")
    print(f"  Target §2.3:        total={REF_SEC23_TOTAL:,}  test={REF_SEC23_TEST:,}")
    print()

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT * FROM training_ghi_1h_direct
        WHERE sun_altitude > 5
          AND sun_altitude_future > 5
          AND ghi_final BETWEEN 0 AND 1400
        ORDER BY timestamp_wib
    """).df()
    # CATATAN: anchor_valid dihapus sengaja (§2.3 tidak mensyaratkan CLP quality)
    # Namun continuity 3h tidak bisa dicek di sini (tidak ada flag per-row).
    # Rows tanpa CLP data akan punya NaN di CLOT_mean → SimpleImputer handle.
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    total = len(df)
    tm, vm, em = split_masks(df)
    print(f"  Total rows (§2.3 SQL): {total:,}")
    print(f"  Split: train={tm.sum():,}  val={vm.sum():,}  test={em.sum():,}")
    delta = total - REF_PIPELINE_TOTAL
    print(f"  Δ vs pipeline: {delta:+,} ({100*delta/REF_PIPELINE_TOTAL:+.1f}%)")
    print(f"  Target §2.3 test: {REF_SEC23_TEST:,}  script test: {em.sum():,}")
    print()

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
    print(f"  Rows setelah filter target: point={len(df_pt):,} avg={len(df_av):,}")

    results = []
    for du, tc, dc, spc, tg in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist",     "point_t60"),
        (df_av, TARGET_AVG,   DELTA_AVG,   "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n{'='*60}")
        print(f"TARGET: {tg}")
        tmm, vmm, emm = split_masks(du)
        print(f"  train={tmm.sum():,}  val={vmm.sum():,}  test={emm.sum():,}")

        xt, xv, xe = du.loc[tmm, FEATURES], du.loc[vmm, FEATURES], du.loc[emm, FEATURES]
        yt, yv, ye = du.loc[tmm, tc], du.loc[vmm, tc], du.loc[emm, tc]
        ydt, ydv = du.loc[tmm, dc], du.loc[vmm, dc]
        gne = du.loc[emm, "ghi_now"].values
        spe = np.clip(du.loc[emm, spc].values, PRED_MIN, PRED_MAX)

        # Cek NaN rate CLP di §2.3 set (vs pipeline)
        nan_clp = du[FEATURES_CLP[:4]].isna().mean().mean()
        print(f"  CLP NaN rate (§2.3): {nan_clp:.3f}  "
              f"(jika tinggi → rows tanpa CLP akan imputed)")

        print("  Training LightGBM residual...")
        lg = lgbm_pipe()
        fit_lgbm(lg, xt, ydt, xv, ydv)
        bi = lg.named_steps["m"].best_iteration_
        pr = np.clip(gne + lg.predict(xe), PRED_MIN, PRED_MAX)
        r = metr(ye, pr, spe, "lgbm_residual", tg)
        r["best_iter"] = bi
        results.append(r)
        print(f"  LGBM iter={bi}  R2={r['r2']:.4f}  dR2={r['r2']-REF_R2_LGBM:+.4f}")

        print("  Training CatBoost direct...")
        cb = catboost_model()
        fit_catboost(cb, xt, yt, xv, yv)
        bic = cb.get_best_iteration()
        cp = np.clip(cb.predict(xe.astype(float).values), PRED_MIN, PRED_MAX)
        r = metr(ye, cp, spe, "catboost_direct", tg)
        r["best_iter"] = bic
        results.append(r)
        print(f"  CatB iter={bic}  R2={r['r2']:.4f}  dR2={r['r2']-REF_R2_CB:+.4f}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "sec23_anchor_results.csv", index=False)

    print(f"\n{'='*65}")
    print("  RINGKASAN PERBANDINGAN R² (§2.3 vs pipeline R1)")
    print(f"{'='*65}")
    for _, row in pd.DataFrame(results)[pd.DataFrame(results)["target"] == "point_t60"].iterrows():
        ref = REF_R2_CB if row["model"] == "catboost_direct" else REF_R2_LGBM
        dr2 = row["r2"] - ref
        verdict = ("✓ AMAN (<±0.003 noise floor)" if abs(dr2) < 0.003
                   else "⚠ SIGNIFIKAN — pertimbangkan update §4 Results")
        print(f"  {row['model']:20s}  R2={row['r2']:.4f}  dR2={dr2:+.4f}  {verdict}")

    print(f"\n  Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
