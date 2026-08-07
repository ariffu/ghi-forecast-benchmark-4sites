#!/usr/bin/env python3
"""
ARM C — Validation-guided Feature Pruning for R8
(Wrapper around experiment_prune_v5b.py logic)

Runs pruning on F1 features using validation set, evaluates test once.
"""

import sys
import pickle
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(encoding="utf-8")

DB = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
OUTPUT_DIR = Path("outputs_R8_kalbar")
OUTPUT_DIR.mkdir(exist_ok=True)

F1_FEATURES = [  # Same as Arm A/B
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

TARGET = "ghi_target_avg60m"  # Use avg target for consistency

CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=3.0, subsample=0.85)

def main():
    print("="*70)
    print("ARM C: VALIDATION-GUIDED PRUNING")
    print("="*70)

    # Load & split
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT * FROM training_ghi_1h_direct
        WHERE anchor_valid AND ghi_final BETWEEN 0 AND 1400
        ORDER BY timestamp_wib
    """).df()
    con.close()

    df["_year"] = pd.to_datetime(df["timestamp_wib"]).dt.year
    train = df[df["_year"] <= 2023].dropna(subset=[TARGET]).copy()
    val = df[df["_year"] == 2024].dropna(subset=[TARGET]).copy()
    test = df[df["_year"] == 2025].dropna(subset=[TARGET]).copy()

    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    # Fit baseline
    def impute(df_, feats):
        X = df_[feats].copy()
        for c in X.columns:
            med = X[c].median()
            X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
        return X

    y_tr, y_va, y_te = train[TARGET].values, val[TARGET].values, test[TARGET].values

    def fit(feats):
        m = CatBoostRegressor(**CAT_PARAMS, random_seed=42, verbose=0)
        m.fit(impute(train, feats), y_tr, eval_set=(impute(val, feats), y_va))
        return m

    def r2_val(m, feats):
        return r2_score(y_va, m.predict(impute(val, feats)))

    def eval_test(m, feats):
        p = m.predict(impute(test, feats))
        return r2_score(y_te, p), mean_absolute_error(y_te, p), np.sqrt(mean_squared_error(y_te, p))

    # Baseline
    print("\n[Baseline] 50 F1 features")
    m_base = fit(F1_FEATURES)
    r2v_base = r2_val(m_base, F1_FEATURES)
    r2t_base, maet_base, rmset_base = eval_test(m_base, F1_FEATURES)
    print(f"  VAL R2={r2v_base:.4f}, TEST R2={r2t_base:.4f}")

    # Feature importance
    imp = pd.Series(m_base.feature_importances_, index=F1_FEATURES).sort_values(ascending=False)
    ranked = list(imp.index)

    # Sweep top-K
    print("\n[Sweep] top-K via VAL")
    Ks = [8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 46]
    sweep = {}
    for K in Ks:
        feats = ranked[:K]
        m = fit(feats)
        rv = r2_val(m, feats)
        sweep[K] = rv
        print(f"  top-{K:2d}: VAL R2={rv:.4f} Δ={rv-r2v_base:+.4f}")

    # Select K*
    EPS = 0.001
    cand = [K for K in Ks if sweep[K] >= r2v_base - EPS]
    K_star = min(cand) if cand else max(Ks, key=lambda k: sweep[k])
    print(f"\nK* = {K_star} (VAL R2={sweep[K_star]:.4f})")

    # Greedy backward elimination
    print(f"\n[Greedy] backward from top-{K_star}")
    current = list(ranked[:K_star])
    m_cur = fit(current)
    rv_cur = r2_val(m_cur, current)
    print(f"  Start: {len(current)} fitur, VAL R2={rv_cur:.4f}")

    TOL = 0.001
    while len(current) > 5:
        best_drop, best_rv = None, -9
        for f in list(current):
            trial = [x for x in current if x != f]
            m_t = fit(trial)
            rv_t = r2_val(m_t, trial)
            if rv_t > best_rv:
                best_rv, best_drop = rv_t, f

        if best_rv >= r2v_base - TOL:
            current = [x for x in current if x != best_drop]
            rv_cur = best_rv
            print(f"  - {best_drop:<30} -> {len(current)} fitur, VAL R2={best_rv:.4f}")
        else:
            print(f"  stop: dropping {best_drop} -> {best_rv:.4f} < {r2v_base-TOL:.4f}")
            break

    # Evaluate final
    print(f"\n[Final Evaluation]")
    m_final = fit(current)
    rv_final = r2_val(m_final, current)
    rt_final, mt_final, rmst_final = eval_test(m_final, current)

    print(f"  Pruned {len(current)} features:")
    for i, f in enumerate(current, 1):
        print(f"    {i:2d}. {f}")

    print(f"\n  VAL R2={rv_final:.4f}, TEST R2={rt_final:.4f}, MAE={mt_final:.2f}")
    print(f"  Δ TEST R2 vs baseline = {rt_final - r2t_base:+.4f}")

    # Save
    results = {
        "pruned_features": current,
        "n_pruned": len(current),
        "r2_test": rt_final,
        "mae_test": mt_final,
        "rmse_test": rmst_final,
        "r2_val": rv_final,
        "baseline_r2_test": r2t_base,
        "baseline_n_features": len(F1_FEATURES),
    }

    with open(OUTPUT_DIR / "arm_C_pruned.pkl", "wb") as f:
        pickle.dump(results, f)

    df_features = pd.DataFrame([
        {"feature": f, "importance": imp[f], "selected": f in current}
        for f in F1_FEATURES
    ]).sort_values("importance", ascending=False)

    df_features.to_csv(OUTPUT_DIR / "arm_C_features.csv", index=False)

    print(f"\n-> Saved: arm_C_pruned.pkl, arm_C_features.csv")

if __name__ == "__main__":
    main()
