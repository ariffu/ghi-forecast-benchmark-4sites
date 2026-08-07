#!/usr/bin/env python3
"""
R8 Arm A v2 -- JAMBI: Redundansi Meteo (F1 vs F2), dataset v2.

Data source (DIPASTIKAN, sesuai instruksi user 2026-07-25):
    jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb
    -> jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025

Ini menggantikan run v1 lama (`outputs_R8_Jambi/arm_A_results.csv`, 17 Juli,
n=9.511, sumber dfm_with_clp_stats.parquet yang sudah diketahui bermasalah --
lihat 09_Audit_Volume_Data_Jambi.md). Perbedaan penting dari v1: meteo (F2)
sekarang diambil LANGSUNG dari tabel v2 itu sendiri (aws_temp_c, aws_rh_pct,
aws_ws_avg, aws_rain_mm, aws_pressure_hpa -- sudah ada di skema 102-kolom),
BUKAN di-join dari parquet eksternal terpisah seperti v1 -- lebih robust,
tidak ada risiko timestamp-mismatch join.

Metodologi identik Arm A lokasi lain: F1 (50 fitur harmonis) vs F2 (F1 + 5
meteo permukaan), CatBoost, target titik & rata-rata, test 2025.

Run:
    python train_ghi_1h_jambi_R8_armA_v2.py
"""
import pickle
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ghi_1h_jambi_R1_benchmark_v2 import (
    connect_data, build_sql, add_features, FEATURES as F1_FEATURES,
    TARGET_POINT, TARGET_AVG, TIME_COL, LOCAL_DB_PATH, SCHEMA, TABLE,
)

OUTPUT_DIR = Path("outputs_R8_jambi_v2")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
PRED_MIN, PRED_MAX = 0.0, 1400.0

METEO = ["aws_temp_c", "aws_rh_pct", "aws_ws_avg", "aws_rain_mm", "aws_pressure_hpa"]
F2_FEATURES = F1_FEATURES + METEO


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp("2024-01-01"),
            (ts >= pd.Timestamp("2024-01-01")) & (ts < pd.Timestamp("2025-01-01")),
            ts >= pd.Timestamp("2025-01-01"))


def catboost_model():
    return CatBoostRegressor(iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
                              loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
                              thread_count=-1, allow_writing_files=False)


def evaluate(y_true, y_pred, y_sp):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_sp = float(np.sqrt(mean_squared_error(y_true, y_sp)))
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / rmse_sp if rmse_sp > 0 else 0.0, 4),
    }


CKPT_PATH = OUTPUT_DIR / "arm_A_v2_checkpoint.pkl"
DATA_CKPT_PATH = OUTPUT_DIR / "arm_A_v2_data.pkl"

COMBOS = [
    ("point_t60", "F1"), ("point_t60", "F2"),
    ("avg_t10_t60", "F1"), ("avg_t10_t60", "F2"),
]


def load_or_build_data():
    if DATA_CKPT_PATH.exists():
        print(f"Loading cached data from {DATA_CKPT_PATH}")
        with open(DATA_CKPT_PATH, "rb") as f:
            return pickle.load(f)
    print(f"DB reference: {LOCAL_DB_PATH.resolve()}  (schema={SCHEMA}, table={TABLE})")
    con = connect_data()
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)
    print(f"Rows loaded: {len(df):,}")
    for m in METEO:
        null_pct = df[m].isna().mean() * 100
        print(f"  {m}: null {null_pct:.1f}%")
    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna() & df["sun_gt5_t60"]].copy()
    print(f"Filtered: point={len(df_pt):,}  avg={len(df_av):,}")
    data = {"df_pt": df_pt, "df_av": df_av}
    with open(DATA_CKPT_PATH, "wb") as f:
        pickle.dump(data, f)
    return data


def run_one_combo(target_name, fs_name, df_use, target_col):
    tm, vm, em = split_masks(df_use)
    sp_col = "smart_persist" if "point" in target_name else "smart_persist_avg"
    y_sp_e = np.clip(df_use.loc[em, sp_col], PRED_MIN, PRED_MAX)
    fs = F1_FEATURES if fs_name == "F1" else F2_FEATURES

    xt, xv, xe = df_use.loc[tm, fs], df_use.loc[vm, fs], df_use.loc[em, fs]
    yt, yv, ye = df_use.loc[tm, target_col], df_use.loc[vm, target_col], df_use.loc[em, target_col]
    med = xt.median()
    xt_i, xv_i, xe_i = xt.fillna(med), xv.fillna(med), xe.fillna(med)

    cb = catboost_model()
    cb.fit(xt_i, yt, eval_set=(xv_i, yv), early_stopping_rounds=150, verbose=False)
    pred = np.clip(cb.predict(xe_i), PRED_MIN, PRED_MAX)
    m = evaluate(ye, pred, y_sp_e)
    m.update({"target": target_name, "features": fs_name, "model": "catboost",
               "n_features": len(fs), "best_iter": cb.get_best_iteration()})
    return m


def main(max_combos_this_call=1):
    data = load_or_build_data()
    df_pt, df_av = data["df_pt"], data["df_av"]

    if CKPT_PATH.exists():
        with open(CKPT_PATH, "rb") as f:
            state = pickle.load(f)
    else:
        state = {"done_combos": [], "results": []}

    remaining = [c for c in COMBOS if c not in state["done_combos"]]
    print(f"Done so far: {state['done_combos']}  Remaining: {remaining}")

    n_run = 0
    for target_name, fs_name in remaining:
        if n_run >= max_combos_this_call:
            break
        df_use = df_pt if target_name == "point_t60" else df_av
        target_col = TARGET_POINT if target_name == "point_t60" else TARGET_AVG
        print(f"\nRunning {target_name} x {fs_name} ...")
        m = run_one_combo(target_name, fs_name, df_use, target_col)
        state["results"].append(m)
        state["done_combos"].append((target_name, fs_name))
        with open(CKPT_PATH, "wb") as f:
            pickle.dump(state, f)
        print(f"  R2={m['r2']:.4f} MAE={m['mae']:.1f} best_iter={m['best_iter']}")
        n_run += 1

    if len(state["done_combos"]) == len(COMBOS):
        df_res = pd.DataFrame(state["results"])
        df_res.to_csv(OUTPUT_DIR / "arm_A_results_v2.csv", index=False)
        print("\n=== ALL DONE ===")
        print("\nDelta (F2 - F1):")
        for tgt in ["point_t60", "avg_t10_t60"]:
            sub = df_res[df_res["target"] == tgt].set_index("features")
            d = sub.loc["F2", "r2"] - sub.loc["F1", "r2"]
            print(f"  {tgt}: F1={sub.loc['F1','r2']:.4f}  F2={sub.loc['F2','r2']:.4f}  delta={d:+.4f}")
        print(f"\nSaved -> {OUTPUT_DIR}/arm_A_results_v2.csv")
    else:
        print(f"\nPaused (budget). {len(state['done_combos'])}/{len(COMBOS)} combos done -- rerun to continue.")


if __name__ == "__main__":
    main(max_combos_this_call=1)
