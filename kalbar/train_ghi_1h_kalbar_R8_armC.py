#!/usr/bin/env python3
"""
R8 Arm C — Kalbar: Feature Pruning via Validation-Guided Greedy Elimination

Methodology:
  - Use validation 2024 for feature importance / elimination decisions
  - Evaluate final pruned model on test 2025 only (no test-set bias)
  - Target: minimize feature count while keeping ΔR² within ε=0.1% of baseline
  - Model: CatBoost (primary from Arm B)

Run:
    & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_kalbar_R8_armC.py
"""

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCAL_DB_PATH = Path(r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db")
OUTPUT_DIR    = Path(r"C:\Users\ariff\DuckDB_kalbar\outputs_R8_kalbar")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_LAT_DEG  = -0.0356
STATION_LON_DEG  = 109.3384
WIB_MERIDIAN_DEG = 105.0

TIME_COL   = "timestamp_wib"
TRAIN_END  = "2024-01-01"
VALID_END  = "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

TARGET_POINT = "ghi_point_t60"
EPSILON = 0.001  # Allow 0.1% R² loss

# F1: 50-lean baseline (identical to R1/Arm A/Arm B)
F1_FEATURES = [
    "ghi_now",
    "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m",
    "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m",
    "accel_ghi_20m",
    "kt_now",
    "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean",
    "accel_kt_20m",
    "clp_cot",
    "clp_cot_lag_10m", "clp_cot_lag_20m",
    "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m",
    "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean",
    "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "hour_sin", "hour_cos",
    "doy_sin",  "doy_cos",
    "month_sin", "month_cos",
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
assert len(F1_FEATURES) == 50


# ---------------------------------------------------------------------------
# Feature Engineering (same as R1/Arm A/Arm B)
# ---------------------------------------------------------------------------
def solar_elevation_deg(timestamps, lat=STATION_LAT_DEG, lon=STATION_LON_DEG,
                        meridian=WIB_MERIDIAN_DEG):
    idx = pd.DatetimeIndex(timestamps)
    doy = idx.dayofyear.values.astype(float)
    h   = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    ha   = (h + 4.0 * (lon - meridian) / 60.0 - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl))
             * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def clearsky(elev_deg):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(elev_deg)), 0.0)


def add_features(df):
    out = df.copy()
    ts  = pd.DatetimeIndex(out[TIME_COL])

    # Rename columns to standard names
    out["ghi_now"] = out["ghi_final"]
    out["kt_now"] = out["kt"]
    out["clp_cot"] = out["CLOT_mean"]
    out["clp_cth_m"] = out["CLTH_mean"]
    out["clp_ctt_k"] = out["CLTT_mean"]
    out["clp_cer"] = out["CLER_23_mean"]
    out["clp_cloud_present"] = out["clp_cloud_present_int"].astype(float)

    # Rename existing lag/rolling columns
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
    out["clp_cot_lag_10m"] = out["clot_lag10m"]

    # Derive missing lags and rolling stats
    out["ghi_lag_120m"] = out["ghi_now"].shift(12)
    out["ghi_lag_180m"] = out["ghi_now"].shift(18)
    out["ghi_roll_30m_mean"] = out["ghi_now"].rolling(window=3, center=False).mean()
    out["ghi_roll_30m_std"] = out["ghi_now"].rolling(window=3, center=False).std()
    out["ghi_roll_60m_mean"] = out["ghi_now"].rolling(window=6, center=False).mean()
    out["ghi_roll_60m_std"] = out["ghi_now"].rolling(window=6, center=False).std()
    out["ghi_roll_180m_mean"] = out["ghi_now"].rolling(window=18, center=False).mean()
    out["ghi_roll_180m_std"] = out["ghi_now"].rolling(window=18, center=False).std()

    # CLP lags and rolling
    out["clp_cot_lag_20m"] = out["clot_lag10m"] * 0.67 + out["clp_cot"] * 0.33
    out["clp_cot_lag_30m"] = out["clot_lag30m"]
    out["clp_cot_lag_60m"] = out["clp_cot"].shift(6)
    out["clp_cot_roll_180m_mean"] = out["clp_cot"].rolling(window=18, center=False).mean()

    # Delta features
    out["clp_cot_delta_10m"]  = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"]  = out["clot_delta_30m"] if "clot_delta_30m" in out.columns else (out["clp_cot"] - out["clp_cot_lag_30m"])
    out["clp_cot_delta_60m"]  = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["ghi_delta_10m"] = out["ghi_now"] - out["ghi_lag_10m"]
    out["ghi_delta_60m"] = out["ghi_now"] - out["ghi_lag_60m"]

    # Acceleration features
    out["accel_ghi_20m"]     = out["ghi_now"]  - 2.0 * out["ghi_lag_10m"]     + out["ghi_lag_20m"]
    out["accel_kt_20m"]      = out["kt_now"]   - 2.0 * out["kt_lag_10m"]      + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"]  - 2.0 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]

    # Time cyclic features
    doy = ts.dayofyear.values.astype(float)
    hour = ts.hour.values.astype(float) + ts.minute.values.astype(float) / 60.0
    mo = ts.month.values.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * mo / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * mo / 12.0)

    # Future features
    out["ghi_cs_t60"] = out["ghi_clearsky_future"]
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(out["sun_altitude_future"])), 0.0)
    out["smart_persist"]     = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_t60"]

    # Target
    out[TARGET_POINT] = out["ghi_target_60m"].copy()
    out["sun_gt5_t60"] = out["sun_altitude_future"] > 5.0
    return out


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


# ---------------------------------------------------------------------------
# Pruning: Validation-guided greedy backward elimination
# ---------------------------------------------------------------------------
def train_catboost_simple(x_tr, y_tr, x_va, y_va):
    """Quick training for feature importance."""
    m = CatBoostRegressor(
        iterations=1000, learning_rate=0.02, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )
    m.fit(x_tr.astype(float).values, y_tr.astype(float).values,
          eval_set=(x_va.astype(float).values, y_va.astype(float).values),
          early_stopping_rounds=100)
    return m


def greedy_backward_elimination(x_tr, y_tr, x_va, y_va, x_te, y_te, features, epsilon=0.001):
    """
    Greedy backward elimination:
      1. Train on all features
      2. Compute baseline R² on validation
      3. Iteratively remove feature with lowest importance (via permutation)
      4. Stop when ΔR²(val) exceeds epsilon OR no features left
      5. Evaluate final pruned set on test only
    """
    print(f"\n{'='*60}")
    print(f"PRUNING: Greedy Backward Elimination (eps = {epsilon*100:.2f}%)")
    print(f"{'='*60}")

    # Baseline
    m_full = train_catboost_simple(x_tr, y_tr, x_va, y_va)
    y_va_pred = np.clip(m_full.predict(x_va.astype(float).values), PRED_MIN, PRED_MAX)
    r2_baseline = r2_score(y_va, y_va_pred)
    print(f"\nBaseline (all {len(features)} features): R² = {r2_baseline:.4f}")

    remaining_features = list(features)
    eliminated = []

    iteration = 0
    while len(remaining_features) > 1:
        iteration += 1
        print(f"\n--- Iteration {iteration}: {len(remaining_features)} features remaining ---")

        # Train on remaining features
        m = train_catboost_simple(
            x_tr[remaining_features], y_tr,
            x_va[remaining_features], y_va
        )
        y_va_pred = np.clip(m.predict(x_va[remaining_features].astype(float).values), PRED_MIN, PRED_MAX)
        r2_current = r2_score(y_va, y_va_pred)

        # Feature importance (permutation-based approximation: use CatBoost's feature_importance)
        importances = m.feature_importances_
        feat_importance = dict(zip(remaining_features, importances))

        # Remove feature with lowest importance
        to_remove = min(feat_importance, key=feat_importance.get)
        print(f"  Removing: {to_remove} (importance={feat_importance[to_remove]:.4f})")
        remaining_features.remove(to_remove)
        eliminated.append(to_remove)

        # Retrain and check validation R²
        m = train_catboost_simple(
            x_tr[remaining_features], y_tr,
            x_va[remaining_features], y_va
        )
        y_va_pred = np.clip(m.predict(x_va[remaining_features].astype(float).values), PRED_MIN, PRED_MAX)
        r2_new = r2_score(y_va, y_va_pred)
        delta_r2 = r2_baseline - r2_new

        print(f"  New R2 (val): {r2_new:.4f}  Delta_R2 = {delta_r2:.4f}")

        if delta_r2 > epsilon:
            print(f"  ! Delta_R2 exceeds eps={epsilon} - STOP pruning")
            remaining_features.append(to_remove)  # add back last removed
            eliminated.pop()
            break

        if len(remaining_features) <= 1:
            print(f"  Only 1 feature left — STOP")
            break

    print(f"\n{'='*60}")
    print(f"PRUNING COMPLETE")
    print(f"  Remaining: {len(remaining_features)} features (from {len(features)})")
    pct_reduction = 100.0 * (len(features) - len(remaining_features)) / len(features)
    print(f"  Reduction: {pct_reduction:.1f}%")
    print(f"  Eliminated: {eliminated}")

    # Final evaluation on TEST (no tuning on test)
    m_final = train_catboost_simple(
        x_tr[remaining_features], y_tr,
        x_va[remaining_features], y_va
    )
    y_te_pred = np.clip(m_final.predict(x_te[remaining_features].astype(float).values), PRED_MIN, PRED_MAX)
    r2_test = r2_score(y_te, y_te_pred)
    delta_r2_test = r2_baseline - r2_test

    print(f"\nFinal on TEST 2025:")
    print(f"  R2 (test): {r2_test:.4f}")
    print(f"  Delta_R2 vs baseline: {delta_r2_test:.4f}")

    return remaining_features, r2_test, r2_baseline, delta_r2_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    con = duckdb.connect(database=str(LOCAL_DB_PATH), read_only=True)
    print("Loading Kalbar data...")
    df = con.execute("""
        SELECT * FROM training_ghi_1h_direct
        WHERE anchor_valid AND ghi_final BETWEEN 0 AND 1400
        ORDER BY timestamp_wib
    """).df()
    con.close()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)

    df_use = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    print(f"Rows: {len(df_use):,}")

    tr_m, va_m, te_m = split_masks(df_use)
    print(f"train={tr_m.sum():,}  val={va_m.sum():,}  test={te_m.sum():,}")

    f1_avail = [f for f in F1_FEATURES if f in df_use.columns]
    print(f"Features available: {len(f1_avail)}/{len(F1_FEATURES)}")

    x_tr = df_use.loc[tr_m, f1_avail]
    x_va = df_use.loc[va_m, f1_avail]
    x_te = df_use.loc[te_m, f1_avail]
    y_tr = df_use.loc[tr_m, TARGET_POINT]
    y_va = df_use.loc[va_m, TARGET_POINT]
    y_te = df_use.loc[te_m, TARGET_POINT]

    # Pruning
    pruned_features, r2_test, r2_baseline, delta_r2 = greedy_backward_elimination(
        x_tr, y_tr, x_va, y_va, x_te, y_te, f1_avail, epsilon=EPSILON
    )

    # Save results
    pruning_results = []
    for feat in f1_avail:
        pruning_results.append({
            "feature": feat,
            "selected": feat in pruned_features,
            "category": "GHI" if "ghi" in feat else ("KT" if "kt" in feat else ("CLP" if "clp" in feat else ("TIME" if any(x in feat for x in ["hour", "doy", "month"]) else "FUTURE"))),
        })

    df_pruning = pd.DataFrame(pruning_results)
    df_pruning.to_csv(OUTPUT_DIR / "arm_C_features.csv", index=False)

    summary = pd.DataFrame([{
        "location": "Kalbar",
        "n_features_baseline": len(f1_avail),
        "n_features_pruned": len(pruned_features),
        "reduction_pct": 100.0 * (len(f1_avail) - len(pruned_features)) / len(f1_avail),
        "r2_baseline_val": r2_baseline,
        "r2_pruned_test": r2_test,
        "delta_r2": delta_r2,
        "top_features": ", ".join(pruned_features[:5]),
    }])
    summary.to_csv(OUTPUT_DIR / "arm_C_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"OUTPUTS")
    print(f"{'='*60}")
    print(f"  arm_C_features.csv: {len(df_pruning)} rows (selected={df_pruning['selected'].sum()})")
    print(f"  arm_C_summary.csv")
    print(f"\n  Top 5 selected features: {', '.join(pruned_features[:5])}")


if __name__ == "__main__":
    main()
