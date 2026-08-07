#!/usr/bin/env python3
"""
R8 Arm C v2 -- JAMBI: Feature Pruning via Validation-Guided Greedy Backward
Elimination, dataset v2, metodologi identik Bengkulu/Kalbar/Banten (BUKAN
top-K sweep yang dipakai run v1 lama -- lihat root cause collinearity di
02_Feature_Engineering.md SS4 poin 9).

Data source (DIPASTIKAN, sesuai instruksi user 2026-07-25):
    jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb
    -> jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025

Resumable: state di-checkpoint ke arm_C_v2_checkpoint.pkl setiap iterasi.

Run:
    python train_ghi_1h_jambi_R8_armC_v2.py
"""
import pickle
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ghi_1h_jambi_R1_benchmark_v2 import (
    connect_data, build_sql, add_features, FEATURES, TARGET_POINT, TIME_COL,
    LOCAL_DB_PATH, SCHEMA, TABLE,
)

OUTPUT_DIR = Path("outputs_R8_jambi_v2")
OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_PATH = OUTPUT_DIR / "arm_C_v2_checkpoint.pkl"
DATA_CKPT_PATH = OUTPUT_DIR / "arm_C_v2_data.pkl"
EPSILON = 0.001
PRED_MIN, PRED_MAX = 0.0, 1400.0
RANDOM_STATE = 42

PRUNE_ITERS, PRUNE_LR, PRUNE_ES = 400, 0.05, 40
FINAL_ITERS, FINAL_LR, FINAL_ES = 4000, 0.02, 150


def train_cb(x_tr, y_tr, x_va, y_va, iters=PRUNE_ITERS, lr=PRUNE_LR, es=PRUNE_ES):
    mdl = CatBoostRegressor(
        iterations=iters, learning_rate=lr, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )
    mdl.fit(x_tr.astype(float).values, y_tr.astype(float).values,
            eval_set=(x_va.astype(float).values, y_va.astype(float).values),
            early_stopping_rounds=es)
    return mdl


def load_data():
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
    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()

    tm = df_pt[TIME_COL] < pd.Timestamp("2024-01-01")
    vm = (df_pt[TIME_COL] >= pd.Timestamp("2024-01-01")) & (df_pt[TIME_COL] < pd.Timestamp("2025-01-01"))
    em = df_pt[TIME_COL] >= pd.Timestamp("2025-01-01")

    result = (df_pt.loc[tm, FEATURES], df_pt.loc[vm, FEATURES], df_pt.loc[em, FEATURES],
              df_pt.loc[tm, TARGET_POINT], df_pt.loc[vm, TARGET_POINT], df_pt.loc[em, TARGET_POINT])
    with open(DATA_CKPT_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


def run_one_iteration(state, x_tr, y_tr, x_va, y_va):
    remaining = state["remaining_features"]
    if len(remaining) <= 1:
        state["done"] = True
        return state

    mdl = train_cb(x_tr[remaining], y_tr, x_va[remaining], y_va)
    pred = np.clip(mdl.predict(x_va[remaining].astype(float).values), PRED_MIN, PRED_MAX)
    r2_current = r2_score(y_va, pred)
    importances = dict(zip(remaining, mdl.feature_importances_))
    to_remove = min(importances, key=importances.get)

    trial = [f for f in remaining if f != to_remove]
    mdl2 = train_cb(x_tr[trial], y_tr, x_va[trial], y_va)
    pred2 = np.clip(mdl2.predict(x_va[trial].astype(float).values), PRED_MIN, PRED_MAX)
    r2_new = r2_score(y_va, pred2)
    delta = state["r2_baseline"] - r2_new

    state["log"].append(dict(iteration=state["iteration"] + 1, removed=to_remove,
                              importance=importances[to_remove], n_remaining_before=len(remaining),
                              r2_current=r2_current, r2_after_removal=r2_new, delta_vs_baseline=delta))

    if delta > EPSILON:
        state["done"] = True
        state["stop_reason"] = f"delta_r2={delta:.4f} > eps={EPSILON} at iter {state['iteration']+1}"
    else:
        state["remaining_features"] = trial
        state["eliminated"].append(to_remove)
        state["iteration"] += 1
        if len(trial) <= 1:
            state["done"] = True
            state["stop_reason"] = "only 1 feature left"
    return state


def main(max_iters_this_call=3):
    x_tr, x_va, x_te, y_tr, y_va, y_te = load_data()
    print(f"train={len(x_tr):,} val={len(x_va):,} test={len(x_te):,}")

    if CKPT_PATH.exists():
        state = pickle.load(open(CKPT_PATH, "rb"))
        print(f"Resumed: iteration={state['iteration']}, remaining={len(state['remaining_features'])}, done={state.get('done', False)}")
    else:
        mdl0 = train_cb(x_tr[FEATURES], y_tr, x_va[FEATURES], y_va)
        pred0 = np.clip(mdl0.predict(x_va[FEATURES].astype(float).values), PRED_MIN, PRED_MAX)
        r2_baseline = r2_score(y_va, pred0)
        state = dict(remaining_features=list(FEATURES), eliminated=[], iteration=0,
                      r2_baseline=r2_baseline, log=[], done=False, stop_reason=None)
        print(f"NEW run. Baseline ({len(FEATURES)} features) val R2 = {r2_baseline:.4f}")
        pickle.dump(state, open(CKPT_PATH, "wb"))

    if state.get("done"):
        print(f"Already DONE: {state.get('stop_reason')}. remaining={len(state['remaining_features'])}")
        finalize(state, x_tr, x_va, x_te, y_tr, y_va, y_te)
        return

    n_run = 0
    while not state.get("done") and n_run < max_iters_this_call:
        state = run_one_iteration(state, x_tr, y_tr, x_va, y_va)
        pickle.dump(state, open(CKPT_PATH, "wb"))
        n_run += 1
        last = state["log"][-1]
        print(f"iter {last['iteration']}: removed={last['removed']} "
              f"r2_after={last['r2_after_removal']:.4f} delta={last['delta_vs_baseline']:.4f} "
              f"remaining={len(state['remaining_features'])}")

    if state.get("done"):
        print(f"DONE this call: {state.get('stop_reason')}")
        finalize(state, x_tr, x_va, x_te, y_tr, y_va, y_te)
    else:
        print(f"Paused (budget). remaining={len(state['remaining_features'])}, "
              f"iteration={state['iteration']} -- rerun to continue.")


FINALIZE_CKPT_PATH = OUTPUT_DIR / "arm_C_v2_finalize_checkpoint.pkl"


def finalize(state, x_tr, x_va, x_te, y_tr, y_va, y_te):
    remaining = state["remaining_features"]

    fstate = {}
    if FINALIZE_CKPT_PATH.exists():
        fstate = pickle.load(open(FINALIZE_CKPT_PATH, "rb"))

    if "r2_test" not in fstate:
        print("Finalize step 1/2: training pruned-feature FULL config model...")
        mdl_final = train_cb(x_tr[remaining], y_tr, x_va[remaining], y_va,
                              iters=FINAL_ITERS, lr=FINAL_LR, es=FINAL_ES)
        pred_te = np.clip(mdl_final.predict(x_te[remaining].astype(float).values), PRED_MIN, PRED_MAX)
        fstate["r2_test"] = r2_score(y_te, pred_te)
        pickle.dump(fstate, open(FINALIZE_CKPT_PATH, "wb"))
        print(f"  pruned r2_test={fstate['r2_test']:.4f} -- saved checkpoint, rerun for step 2/2.")
        return
    r2_test = fstate["r2_test"]

    if "r2_base_test" not in fstate:
        print("Finalize step 2/2: training baseline (50-feature) FULL config model...")
        mdl_base_final = train_cb(x_tr[FEATURES], y_tr, x_va[FEATURES], y_va,
                                   iters=FINAL_ITERS, lr=FINAL_LR, es=FINAL_ES)
        pred_base_te = np.clip(mdl_base_final.predict(x_te[FEATURES].astype(float).values), PRED_MIN, PRED_MAX)
        fstate["r2_base_test"] = r2_score(y_te, pred_base_te)
        pickle.dump(fstate, open(FINALIZE_CKPT_PATH, "wb"))
        print(f"  baseline r2_test={fstate['r2_base_test']:.4f} -- saved checkpoint, rerun to write final outputs.")
        return
    r2_base_test = fstate["r2_base_test"]

    pd.DataFrame(state["log"]).to_csv(OUTPUT_DIR / "arm_C_v2_elimination_log.csv", index=False)
    pd.DataFrame([{"feature": f, "selected": f in remaining} for f in FEATURES]) \
        .to_csv(OUTPUT_DIR / "arm_C_v2_features.csv", index=False)
    summary = pd.DataFrame([{
        "location": "Jambi (v2)", "n_features_baseline": len(FEATURES),
        "n_features_pruned": len(remaining),
        "reduction_pct": 100.0 * (len(FEATURES) - len(remaining)) / len(FEATURES),
        "r2_baseline_val": state["r2_baseline"], "r2_baseline_test_fullconfig": r2_base_test,
        "r2_pruned_test_fullconfig": r2_test, "delta_r2_test": r2_base_test - r2_test,
        "selected_features": ", ".join(remaining),
    }])
    summary.to_csv(OUTPUT_DIR / "arm_C_v2_summary.csv", index=False)
    print(f"\n=== FINAL ===\nremaining={len(remaining)}/{len(FEATURES)} "
          f"({100*(len(FEATURES)-len(remaining))/len(FEATURES):.1f}% reduction)")
    print(f"R2 test (full-config): baseline={r2_base_test:.4f}  pruned={r2_test:.4f}  "
          f"delta={r2_base_test-r2_test:.4f}")
    print(f"Selected: {remaining}")


if __name__ == "__main__":
    main(max_iters_this_call=3)
