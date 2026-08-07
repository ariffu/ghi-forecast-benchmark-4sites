#!/usr/bin/env python3
"""
R8 BATCH TEMPLATE — Simplified for 4-lokasi execution
Focus: Arm A (meteo) + Arm B (GBM only) — core findings
Skip: Arm B DL (will add 5-seed in revision), Arm C (use v5b logic)

Usage: python train_ghi_1h_r8_batch_template.py --db <db_path> --lokasi <name>
"""

import sys
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path
import argparse

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────

F1_FEATURES = [
    "ghi_now", "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m", "ghi_lag_60m",
    "ghi_lag_120m", "ghi_lag_180m", "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std", "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m", "accel_ghi_20m",
    "kt_now", "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean", "accel_kt_20m",
    "clp_cot", "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean", "accel_clp_cot_20m", "clp_cth_m", "clp_ctt_k",
    "clp_cer", "clp_cloud_present",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]

F2_FEATURES = F1_FEATURES + ["temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "pressure_hpa"]

TARGET_POINT = "ghi_point_t60"
TARGET_AVG = "ghi_avg_t10_t60"

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_r8_batch(db_path, lokasi, time_col, target_point, target_avg, output_dir):
    """Run simplified R8 (Arm A + Arm B GBM)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print(f"R8 BATCH RUN — {lokasi}")
    print("="*70)

    # Load data
    print(f"\nLoading {lokasi} data...")
    try:
        con = duckdb.connect(db_path, read_only=True)
        df = con.execute(f"""
            SELECT * FROM training_ghi_1h_direct
            WHERE anchor_valid AND ghi_final BETWEEN 0 AND 1400
            ORDER BY {time_col}
        """).df()
        con.close()
    except:
        print(f"Error loading from {db_path}. Trying alternate schema...")
        con = duckdb.connect(db_path, read_only=True)
        tables = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        print(f"Available tables: {tables}")
        con.close()
        return

    df[time_col] = pd.to_datetime(df[time_col])
    print(f"Loaded: {len(df):,} rows")

    # Prepare features (minimal — assume columns exist)
    try:
        # Ensure features exist (impute if missing)
        for feat in F1_FEATURES + F2_FEATURES:
            if feat not in df.columns:
                print(f"  WARNING: {feat} not in columns, will skip")
    except:
        pass

    # Filter
    df_pt = df[(df[target_point].notna()) & (df["ghi_final"] > 0)].copy()
    df_av = df[(df[target_avg].notna()) & (df["ghi_final"] > 0)].copy()
    print(f"Filtered: point={len(df_pt):,}, avg={len(df_av):,}")

    # ─────────────────────────────────────────────────────────────────────────
    # ARM A: Feature Engineering (F1 vs F2)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ARM A: FEATURE ENGINEERING (F1 vs F2)")
    print("="*70)

    arm_a_results = []
    for target_name, target_col, df_use in [("point_t60", target_point, df_pt), ("avg_t10_t60", target_avg, df_av)]:
        tm, vm, em = split_masks(df_use, time_col)

        for feature_set_name, feature_set in [("F1", F1_FEATURES), ("F2", F2_FEATURES)]:
            print(f"\n{target_name} x {feature_set_name}:")

            # Check features exist
            missing = [f for f in feature_set if f not in df_use.columns]
            if missing:
                print(f"  Skipping — missing features: {missing[:3]}")
                continue

            xt, xv, xe = df_use.loc[tm, feature_set], df_use.loc[vm, feature_set], df_use.loc[em, feature_set]
            yt, yv, ye = df_use.loc[tm, target_col], df_use.loc[vm, target_col], df_use.loc[em, target_col]

            # Impute
            xt_imp = xt.fillna(xt.median())
            xv_imp = xv.fillna(xt.median())
            xe_imp = xe.fillna(xt.median())

            # CatBoost
            try:
                cb = catboost_model()
                cb.fit(xt_imp, yt, eval_set=(xv_imp, yv), early_stopping_rounds=150, verbose=False)
                cp = np.clip(cb.predict(xe_imp), 0, 1400)
                metrics = evaluate_model(ye, cp, np.clip(df_use.loc[em, "smart_persist" if "point" in target_name else "smart_persist_avg"], 0, 1400))
                metrics.update({"target": target_name, "features": feature_set_name, "model": "catboost"})
                arm_a_results.append(metrics)
                print(f"  CatBoost: R2={metrics['r2']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

    pd.DataFrame(arm_a_results).to_csv(output_dir / "arm_A_results.csv", index=False)
    print(f"\n-> Saved: arm_A_results.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # ARM B: GBM Only (CatBoost vs LightGBM)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ARM B: MODEL COMPARISON (GBM only)")
    print("="*70)

    arm_b_results = []
    du = df_pt
    tm, vm, em = split_masks(du, time_col)
    xt, xv, xe = du.loc[tm, F1_FEATURES], du.loc[vm, F1_FEATURES], du.loc[em, F1_FEATURES]
    yt, yv, ye = du.loc[tm, target_point], du.loc[vm, target_point], du.loc[em, target_point]
    ysp_e = np.clip(du.loc[em, "smart_persist"], 0, 1400)

    xt_imp = xt.fillna(xt.median())
    xv_imp = xv.fillna(xt.median())
    xe_imp = xe.fillna(xt.median())

    for model_name, model_cls in [("catboost", catboost_model), ("lgbm", lgbm_pipe)]:
        try:
            if model_name == "catboost":
                m = model_cls()
                m.fit(xt_imp, yt, eval_set=(xv_imp, yv), early_stopping_rounds=150, verbose=False)
                pred = np.clip(m.predict(xe_imp), 0, 1400)
            else:
                m = model_cls()
                m.fit(xt_imp, yt, m__eval_set=[(xv_imp, yv)], m__eval_metric="rmse", m__callbacks=[lgb.early_stopping(150, verbose=False)])
                pred = np.clip(m.predict(xe_imp), 0, 1400)

            metrics = evaluate_model(ye, pred, ysp_e)
            metrics.update({"model": model_name})
            arm_b_results.append(metrics)
            print(f"  {model_name}: R2={metrics['r2']:.4f}")
        except Exception as e:
            print(f"  {model_name} error: {e}")

    pd.DataFrame(arm_b_results).to_csv(output_dir / "arm_B_results.csv", index=False)
    print(f"\n-> Saved: arm_B_results.csv")

    print("\n" + "="*70)
    print(f"OK {lokasi} R8 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--lokasi", required=True, help="Lokasi name (Kalbar/Bengkulu/Jambi/Banten)")
    parser.add_argument("--time-col", default="timestamp_wib", help="Time column name")
    parser.add_argument("--target-point", default="ghi_target_60m", help="Point target column")
    parser.add_argument("--target-avg", default="ghi_target_avg60m", help="Avg target column")
    parser.add_argument("--output", default="outputs_R8", help="Output directory base")

    args = parser.parse_args()
    output_dir = f"{args.output}_{args.lokasi}"

    run_r8_batch(args.db, args.lokasi, args.time_col, args.target_point, args.target_avg, output_dir)
