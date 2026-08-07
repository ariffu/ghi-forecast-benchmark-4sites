#!/usr/bin/env python3
"""
R8 BATCH — JAMBI (adaptasi dari DuckDB_kalbar/train_ghi_1h_r8_batch_template.py)

Arm A: F1 (50 fitur harmonis R1) vs F2 (F1 + 5 meteo permukaan) — CatBoost
Arm B: CatBoost direct vs LightGBM direct (fair-play, F1, target titik)

Dataset dibangun ulang memakai builder R1 Jambi (train_ghi_1h_jambi_R1_benchmark.py)
— 50 fitur standar + target ghi_point_t60 / ghi_avg_t10_t60 — lalu ditambah
5 kolom meteo (mapping Jambi: rh_pct -> humidity_pct).

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_jambi_r8.py
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Builder dataset harmonis R1 (50 fitur + target)
from train_ghi_1h_jambi_R1_benchmark import (
    build_dataset, FEATURES, TARGET_POINT, TARGET_AVG, TIME_COL,
)

LOKASI = "Jambi"
OUTPUT_DIR = Path("outputs_R8_Jambi")
OUTPUT_DIR.mkdir(exist_ok=True)

F1_FEATURES = list(FEATURES)          # 50 fitur harmonis R1
METEO = ["temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "pressure_hpa"]
F2_FEATURES = F1_FEATURES + METEO


def split_masks(df, time_col):
    ts = df[time_col]
    return (ts < pd.Timestamp("2024-01-01"),
            (ts >= pd.Timestamp("2024-01-01")) & (ts < pd.Timestamp("2025-01-01")),
            ts >= pd.Timestamp("2025-01-01"))


def lgbm_pipe():
    reg = lgb.LGBMRegressor(objective="regression", n_estimators=6000, learning_rate=0.02,
                            num_leaves=39, min_child_samples=70, reg_alpha=0.2, reg_lambda=2.5,
                            colsample_bytree=0.82, subsample=0.85, subsample_freq=1,
                            random_state=42, n_jobs=-1, force_col_wise=True, verbosity=-1)
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("m", reg)])


def catboost_model():
    return CatBoostRegressor(iterations=4000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
                             loss_function="RMSE", random_seed=42, verbose=False,
                             thread_count=-1, allow_writing_files=False)


def evaluate_model(y_true, y_pred, y_sp):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_sp = float(np.sqrt(mean_squared_error(y_true, y_sp)))
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse": round(rmse, 1),
        "skill_vs_sp": round(1.0 - rmse / rmse_sp if rmse_sp > 0 else 0.0, 4),
    }


def main():
    print("=" * 70, flush=True)
    print(f"R8 BATCH RUN — {LOKASI}", flush=True)
    print("=" * 70, flush=True)

    print("\nBuilding harmonised dataset (R1 builder)...", flush=True)
    df = build_dataset()

    # Tambah meteo dari parquet sumber (align via ts)
    src = pd.read_parquet("dfm_with_clp_stats.parquet")
    src = src.sort_values("ts").reset_index(drop=True)
    src_meteo = src[["ts", "temp_air_c", "rh_pct", "wind_speed_ms",
                     "rainfall_mm", "pressure_hpa"]].copy()
    src_meteo["ts"] = pd.to_datetime(src_meteo["ts"])
    src_meteo = src_meteo.rename(columns={"rh_pct": "humidity_pct"})
    df = df.merge(src_meteo, on="ts", how="left")

    print(f"Loaded: {len(df):,} rows", flush=True)
    for m in METEO:
        print(f"  {m}: null {df[m].isna().mean()*100:.1f}%", flush=True)

    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    df_av = df[df[TARGET_AVG].notna() & df["sun_gt5_t60"]].copy()
    print(f"Filtered: point={len(df_pt):,}, avg={len(df_av):,}", flush=True)

    # ── ARM A: F1 vs F2 (CatBoost) ─────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("ARM A: FEATURE ENGINEERING (F1 vs F2)", flush=True)
    print("=" * 70, flush=True)

    arm_a = []
    for target_name, target_col, df_use in [("point_t60", TARGET_POINT, df_pt),
                                            ("avg_t10_t60", TARGET_AVG, df_av)]:
        tm, vm, em = split_masks(df_use, TIME_COL)
        sp_col = "smart_persist" if "point" in target_name else "smart_persist_avg"
        ysp_e = np.clip(df_use.loc[em, sp_col], 0, 1400)

        for fs_name, fs in [("F1", F1_FEATURES), ("F2", F2_FEATURES)]:
            print(f"\n{target_name} x {fs_name} ({len(fs)} fitur):", flush=True)
            xt = df_use.loc[tm, fs]
            xv = df_use.loc[vm, fs]
            xe = df_use.loc[em, fs]
            yt = df_use.loc[tm, target_col]
            yv = df_use.loc[vm, target_col]
            ye = df_use.loc[em, target_col]

            med = xt.median()
            xt_imp, xv_imp, xe_imp = xt.fillna(med), xv.fillna(med), xe.fillna(med)

            cb = catboost_model()
            cb.fit(xt_imp, yt, eval_set=(xv_imp, yv), early_stopping_rounds=150, verbose=False)
            pred = np.clip(cb.predict(xe_imp), 0, 1400)
            metrics = evaluate_model(ye, pred, ysp_e)
            metrics.update({"target": target_name, "features": fs_name,
                            "model": "catboost", "n_features": len(fs),
                            "best_iter": cb.get_best_iteration()})
            arm_a.append(metrics)
            print(f"  CatBoost {fs_name}: R2={metrics['r2']:.4f}  MAE={metrics['mae']:.1f}", flush=True)

    arm_a_df = pd.DataFrame(arm_a)
    arm_a_df.to_csv(OUTPUT_DIR / "arm_A_results.csv", index=False)
    print("\n-> Saved: arm_A_results.csv", flush=True)

    # Delta summary
    print("\nArm A delta (F2 - F1):", flush=True)
    for tgt in ["point_t60", "avg_t10_t60"]:
        sub = arm_a_df[arm_a_df["target"] == tgt].set_index("features")
        if {"F1", "F2"}.issubset(sub.index):
            d = sub.loc["F2", "r2"] - sub.loc["F1", "r2"]
            print(f"  {tgt}: F1={sub.loc['F1','r2']:.4f}  F2={sub.loc['F2','r2']:.4f}  delta={d:+.4f}", flush=True)

    # ── ARM B: CatBoost vs LGBM direct (fair-play, F1, titik) ──────────────
    print("\n" + "=" * 70, flush=True)
    print("ARM B: MODEL COMPARISON (GBM only, F1, point_t60)", flush=True)
    print("=" * 70, flush=True)

    arm_b = []
    du = df_pt
    tm, vm, em = split_masks(du, TIME_COL)
    xt, xv, xe = du.loc[tm, F1_FEATURES], du.loc[vm, F1_FEATURES], du.loc[em, F1_FEATURES]
    yt, yv, ye = du.loc[tm, TARGET_POINT], du.loc[vm, TARGET_POINT], du.loc[em, TARGET_POINT]
    ysp_e = np.clip(du.loc[em, "smart_persist"], 0, 1400)

    med = xt.median()
    xt_imp, xv_imp, xe_imp = xt.fillna(med), xv.fillna(med), xe.fillna(med)

    for model_name in ["catboost", "lgbm"]:
        if model_name == "catboost":
            m = catboost_model()
            m.fit(xt_imp, yt, eval_set=(xv_imp, yv), early_stopping_rounds=150, verbose=False)
            pred = np.clip(m.predict(xe_imp), 0, 1400)
        else:
            m = lgbm_pipe()
            m.fit(xt_imp, yt, m__eval_set=[(xv_imp, yv)], m__eval_metric="rmse",
                  m__callbacks=[lgb.early_stopping(150, verbose=False)])
            pred = np.clip(m.predict(xe_imp), 0, 1400)
        metrics = evaluate_model(ye, pred, ysp_e)
        metrics.update({"model": model_name})
        arm_b.append(metrics)
        print(f"  {model_name}: R2={metrics['r2']:.4f}  MAE={metrics['mae']:.1f}", flush=True)

    pd.DataFrame(arm_b).to_csv(OUTPUT_DIR / "arm_B_results.csv", index=False)
    print("\n-> Saved: arm_B_results.csv", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"OK {LOKASI} R8 COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
