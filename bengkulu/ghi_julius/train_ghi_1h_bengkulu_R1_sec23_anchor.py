#!/usr/bin/env python3
"""
R1 Bengkulu — re-evaluasi dengan anchor §2.3 (elev_anchor > 5°).

Tujuan: verifikasi apakah R² berubah signifikan saat menggunakan set anchor
homogen §2.3 (elev > 5° di ANCHOR DAN t+60) vs pipeline (is_model_ready=1,
elev_t60>5° saja).

Modifikasi vs R1:
  - WHERE: hapus 'is_model_ready = 1', tambah 'solar_elev_deg > 5'
  - Semua lain IDENTIK: fitur (50), hyperparameter, split

Catatan:
  - Jika view sudah pre-filter ke is_model_ready=1, row count akan LEBIH KECIL
    dari pipeline (intersection). Script akan melaporkan ini.
  - Jika view belum pre-filter, row count akan BERBEDA dari pipeline karena
    kita menambah/mengurangi baris.

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_bengkulu_R1_sec23_anchor.py
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
# Config  (identik dengan R1)
# ---------------------------------------------------------------------------
LOCAL_DB_PATH = Path("C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb")
OUTPUT_DIR    = Path("outputs_R1_bengkulu_sec23")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -3.8607
STATION_LON_DEG  = 102.3381
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "ts_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

# Referensi R1 pipeline untuk perbandingan
REF_PIPELINE_TOTAL = 105_051
REF_PIPELINE_TEST  = 22_711
REF_R2_CB          = 0.792    # CatBoost R1 (angka di paper)
REF_R2_LGBM        = 0.789    # LightGBM R1 (angka di paper)

# §2.3 anchor reference (dari verify_homogen_anchor_bengkulu_kalbar.py)
REF_SEC23_TOTAL = 109_855
REF_SEC23_TEST  = 22_088

# ---------------------------------------------------------------------------
# Features (50 — identik R1)
# ---------------------------------------------------------------------------
FEATURES_GHI  = ["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m",
                  "ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
                  "ghi_roll_30m_mean","ghi_roll_30m_std",
                  "ghi_roll_60m_mean","ghi_roll_60m_std",
                  "ghi_roll_180m_mean","ghi_roll_180m_std",
                  "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m"]
FEATURES_KT   = ["kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
                  "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m"]
FEATURES_CLP  = ["clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
                  "clp_cot_lag_30m","clp_cot_lag_60m",
                  "clp_cot_delta_10m","clp_cot_delta_30m",
                  "clp_cot_delta_60m","clp_cot_delta_180m",
                  "clp_cot_roll_180m_mean","accel_clp_cot_20m",
                  "clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present"]
FEATURES_TIME = ["hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos"]
FEATURES_FUTURE = ["ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]

FEATURES = FEATURES_GHI + FEATURES_KT + FEATURES_CLP + FEATURES_TIME + FEATURES_FUTURE
assert len(FEATURES) == 50

TARGET_POINT = "ghi_point_t60"
TARGET_AVG   = "ghi_avg_t10_t60"
DELTA_POINT  = "delta_point"
DELTA_AVG    = "delta_avg"


# ---------------------------------------------------------------------------
# SQL  — §2.3 filter: hapus is_model_ready, tambah solar_elev_deg > 5
# ---------------------------------------------------------------------------
def build_sql_sec23():
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
    WHERE solar_elev_deg > 5          -- §2.3: elevasi di anchor > 5°
      AND has_continuous_3h_history = 1
      AND ghi_now BETWEEN 0 AND 1400
    ORDER BY ts_wib
    """
    # CATATAN: is_model_ready = 1 DIHAPUS sengaja (§2.3 tidak mensyaratkan CLP)
    # Jika view sudah pre-filter ke is_model_ready, baris dengan CLP=NaN tidak
    # akan muncul. Script akan melaporkan count untuk deteksi otomatis.


# ---------------------------------------------------------------------------
# Astronomical helpers (identik R1)
# ---------------------------------------------------------------------------
def solar_elevation_deg(timestamps):
    idx = pd.DatetimeIndex(timestamps)
    doy = idx.dayofyear.values.astype(float)
    h   = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    solar_t = h + 4.0 * (STATION_LON_DEG - WIB_MERIDIAN_DEG) / 60.0
    ha = (solar_t - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(STATION_LAT_DEG)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(STATION_LAT_DEG)) * np.cos(np.deg2rad(decl))
             * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def clearsky_simple(elev_deg):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(elev_deg)), 0.0)


# ---------------------------------------------------------------------------
# Feature engineering (identik R1)
# ---------------------------------------------------------------------------
def add_features(df):
    out = df.copy()
    cs_now = clearsky_simple(out["solar_elev_deg"].values.astype(float))
    out["kt_now"] = out["ghi_now"].values / np.maximum(cs_now, 20.0)

    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]

    out["accel_ghi_20m"]      = out["ghi_now"]  - 2.0*out["ghi_lag_10m"]     + out["ghi_lag_20m"]
    out["accel_kt_20m"]       = out["kt_now"]   - 2.0*out["kt_lag_10m"]      + out["kt_lag_20m"]
    out["accel_clp_cot_20m"]  = out["clp_cot"]  - 2.0*out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    ts = pd.DatetimeIndex(out[TIME_COL])
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

    lead_cols = ["ghi_lead_10m","ghi_lead_20m","ghi_lead_30m",
                 "ghi_lead_40m","ghi_lead_50m","ghi_lead_60m"]
    leads     = out[lead_cols]
    all_valid = (leads.notna().all(axis=1)
                 & leads.apply(lambda c: c.between(0, 1400)).all(axis=1))
    out[TARGET_AVG] = np.where(all_valid, leads.mean(axis=1), np.nan)

    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))

    out[DELTA_POINT] = out[TARGET_POINT] - out["ghi_now"]
    out[DELTA_AVG]   = out[TARGET_AVG]   - out["ghi_now"]

    return out


def split_masks(df):
    ts = df[TIME_COL]
    train = ts <  pd.Timestamp(TRAIN_END)
    valid = (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END))
    test  = ts >= pd.Timestamp(VALID_END)
    return train, valid, test


def lgbm_pipe(seed=RANDOM_STATE):
    reg = lgb.LGBMRegressor(
        objective="regression", n_estimators=6000, learning_rate=0.02,
        num_leaves=39, min_child_samples=70,
        reg_alpha=0.2, reg_lambda=2.5,
        colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
        random_state=seed, n_jobs=-1, force_col_wise=True, verbosity=-1,
    )
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("m",   reg)])


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
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp_rmse = float(np.sqrt(mean_squared_error(y_true, sp_pred)))
    skill = (1.0 - rmse / sp_rmse) if sp_rmse > 0 else float("nan")
    return {"model": model, "target": target, "n": len(y_true),
            "r2": round(r2, 4), "mae": round(mae, 1),
            "rmse": round(rmse, 1), "skill_vs_sp": round(skill, 4)}


def main():
    print("=" * 65)
    print("  BENGKULU R1 — §2.3 ANCHOR SET VERIFICATION")
    print("=" * 65)
    print(f"  Filter: solar_elev_deg > 5 (NOT is_model_ready=1)")
    print(f"  Referensi pipeline:  total={REF_PIPELINE_TOTAL:,}  test={REF_PIPELINE_TEST:,}")
    print(f"  Target §2.3:         total={REF_SEC23_TOTAL:,}  test={REF_SEC23_TEST:,}")
    print()

    con = duckdb.connect(database=":memory:")
    con.execute(f"ATTACH '{LOCAL_DB_PATH.as_posix()}' AS bengkulu_db (READ_ONLY)")
    print("Memuat data (§2.3 filter)...")
    df = con.execute(build_sql_sec23()).fetchdf()
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    total = len(df)
    tr_m, va_m, te_m = split_masks(df)
    print(f"  Total rows (§2.3 SQL): {total:,}")
    print(f"  Split: train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")
    print()

    # Diagnostik: apakah view pre-filter ke is_model_ready?
    if total < REF_PIPELINE_TOTAL:
        print(f"  ⚠️  {total:,} < pipeline ({REF_PIPELINE_TOTAL:,})")
        print("  → View mungkin sudah pre-filter ke is_model_ready=1.")
        print("  → Ini adalah INTERSECTION (pipeline ∩ §2.3).")
        print("  → R² yang dihasilkan = dampak penambahan elev_anchor>5° pada pipeline rows.")
    elif total > REF_PIPELINE_TOTAL:
        print(f"  ✓ {total:,} > pipeline ({REF_PIPELINE_TOTAL:,})")
        print("  → View TIDAK pre-filter is_model_ready — §2.3 menambah baris non-CLP.")
        print("  → R² yang dihasilkan = estimasi §2.3 penuh (incl. NaN CLP → imputed).")

    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna()             & df["sun_gt5_t60"]].copy()
    print(f"  Rows point target : {len(df_pt):,}")

    results = []

    for (df_use, target_col, delta_col, sp_col, tgt_name) in [
        (df_pt, TARGET_POINT, DELTA_POINT, "smart_persist",     "point_t60"),
        (df_av, TARGET_AVG,   DELTA_AVG,   "smart_persist_avg", "avg_t10_t60"),
    ]:
        print(f"\n{'='*60}")
        print(f"TARGET: {tgt_name}")
        tr_m, va_m, te_m = split_masks(df_use)
        print(f"  train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

        x_tr  = df_use.loc[tr_m, FEATURES];  x_va = df_use.loc[va_m, FEATURES]
        x_te  = df_use.loc[te_m, FEATURES]
        y_tr  = df_use.loc[tr_m, target_col]; y_va = df_use.loc[va_m, target_col]
        y_te  = df_use.loc[te_m, target_col]
        yd_tr = df_use.loc[tr_m, delta_col]; yd_va = df_use.loc[va_m, delta_col]
        ghi_now_te = df_use.loc[te_m, "ghi_now"].values
        sp_te = np.clip(df_use.loc[te_m, sp_col].values, PRED_MIN, PRED_MAX)

        print("  Training LightGBM residual...")
        lgbm = lgbm_pipe()
        fit_lgbm(lgbm, x_tr, yd_tr, x_va, yd_va)
        lgbm_pred = np.clip(ghi_now_te + lgbm.predict(x_te), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, lgbm_pred, sp_te, "lgbm_residual", tgt_name)
        r["best_iter"] = lgbm.named_steps["m"].best_iteration_
        results.append(r)
        print(f"  lgbm_residual  iter={r['best_iter']:4d}  R2={r['r2']:.4f}  "
              f"dR2_vs_R1={r['r2']-REF_R2_LGBM:+.4f}")

        print("  Training CatBoost direct...")
        cb = catboost_model()
        fit_catboost(cb, x_tr, y_tr, x_va, y_va)
        cb_pred = np.clip(cb.predict(x_te.astype(float).values), PRED_MIN, PRED_MAX)
        r = compute_metrics(y_te, cb_pred, sp_te, "catboost_direct", tgt_name)
        r["best_iter"] = cb.get_best_iteration()
        results.append(r)
        print(f"  catboost_direct iter={r['best_iter']:4d}  R2={r['r2']:.4f}  "
              f"dR2_vs_R1={r['r2']-REF_R2_CB:+.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "sec23_anchor_results.csv", index=False)

    print(f"\n{'='*65}")
    print("  RINGKASAN PERBANDINGAN R² (§2.3 vs pipeline R1)")
    print(f"{'='*65}")
    for _, row in results_df[results_df["target"] == "point_t60"].iterrows():
        ref = REF_R2_CB if row["model"] == "catboost_direct" else REF_R2_LGBM
        dr2 = row["r2"] - ref
        verdict = ("✓ AMAN (<±0.003 noise floor)" if abs(dr2) < 0.003
                   else "⚠ SIGNIFIKAN — update §4 Results diperlukan")
        print(f"  {row['model']:20s}  R2={row['r2']:.4f}  dR2={dr2:+.4f}  {verdict}")

    print(f"\n  Hasil tersimpan: {OUTPUT_DIR}/")
    print(f"\n  Catatan untuk paper:")
    point_results = results_df[results_df["target"] == "point_t60"].set_index("model")["r2"]
    for model, r2 in point_results.items():
        ref = REF_R2_CB if model == "catboost_direct" else REF_R2_LGBM
        dr2 = r2 - ref
        if abs(dr2) < 0.003:
            print(f"  [{model}] dR²={dr2:+.4f} → hanya update Tabel 1, Results §4 tidak perlu diubah")
        else:
            print(f"  [{model}] dR²={dr2:+.4f} → perlu update Results §4 dengan angka §2.3")


if __name__ == "__main__":
    main()
