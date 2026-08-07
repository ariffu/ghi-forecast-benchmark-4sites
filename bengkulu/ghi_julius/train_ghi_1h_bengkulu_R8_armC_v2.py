#!/usr/bin/env python3
"""
R8 Arm C v2 -- Bengkulu: Feature Pruning via Validation-Guided Greedy Backward
Elimination, dimulai murni dari F1 (50 fitur), TIDAK mencampur DHI/DNI mentah
dengan ghi_now (root cause collinearity collapse yang membuat F_super lama void
-- lihat Restrukturisasi/02_Feature_Engineering.md §4 poin 9).

Metodologi identik dengan train_ghi_1h_kalbar_R8_armC.py (backward elimination,
BUKAN top-K dari superset), hanya beda: (a) sumber data + F1 dari skrip R1
Bengkulu, (b) CatBoost pruning-phase pakai iterasi lebih ringan (400/lr0.05/es40
vs 1000/0.02/100 di Kalbar) demi kelayakan waktu -- evaluasi FINAL tetap pakai
model penuh (lihat compute_final()).

Resumable: state (remaining_features, eliminated, r2_baseline, iteration) di-
checkpoint ke arm_C_checkpoint.pkl setiap iterasi supaya bisa dilanjutkan lintas
sesi/panggilan tanpa mengulang dari nol.

Run:
    python train_ghi_1h_bengkulu_R8_armC_v2.py
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
import train_ghi_1h_bengkulu_R1_benchmark as m
m.LOCAL_DB_PATH = Path("/sessions/jolly-gifted-ritchie/mnt/DuckDB_bengkulu/bengkulu.duckdb")

OUTPUT_DIR = Path(__file__).parent / "outputs_R8_bengkulu"
OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_PATH = OUTPUT_DIR / "arm_C_checkpoint.pkl"
EPSILON = 0.001  # 0.1% R2 tolerance, same as Kalbar

# Pruning-phase (lighter, for speed) vs final-eval (full) CatBoost configs
PRUNE_ITERS, PRUNE_LR, PRUNE_ES = 400, 0.05, 40
FINAL_ITERS, FINAL_LR, FINAL_ES = 4000, 0.02, 150


def train_cb(x_tr, y_tr, x_va, y_va, iters=PRUNE_ITERS, lr=PRUNE_LR, es=PRUNE_ES):
    mdl = CatBoostRegressor(
        iterations=iters, learning_rate=lr, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=42, verbose=False,
        thread_count=-1, allow_writing_files=False,
    )
    mdl.fit(x_tr.astype(float).values, y_tr.astype(float).values,
            eval_set=(x_va.astype(float).values, y_va.astype(float).values),
            early_stopping_rounds=es)
    return mdl


def load_data():
    con = m.connect_data()
    df = con.execute(m.build_sql()).fetchdf()
    con.close()
    df[m.TIME_COL] = pd.to_datetime(df[m.TIME_COL])
    df = m.add_features(df)
    df_pt = df[df[m.TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    tr_m, va_m, te_m = m.split_masks(df_pt)
    return (df_pt.loc[tr_m, m.FEATURES], df_pt.loc[va_m, m.FEATURES], df_pt.loc[te_m, m.FEATURES],
            df_pt.loc[tr_m, m.TARGET_POINT], df_pt.loc[va_m, m.TARGET_POINT], df_pt.loc[te_m, m.TARGET_POINT])


def run_one_iteration(state, x_tr, y_tr, x_va, y_va):
    """Run exactly one elimination step; mutate+return state. Returns None if done."""
    remaining = state["remaining_features"]
    if len(remaining) <= 1:
        state["done"] = True
        return state

    mdl = train_cb(x_tr[remaining], y_tr, x_va[remaining], y_va)
    pred = np.clip(mdl.predict(x_va[remaining].astype(float).values), m.PRED_MIN, m.PRED_MAX)
    r2_current = r2_score(y_va, pred)
    importances = dict(zip(remaining, mdl.feature_importances_))
    to_remove = min(importances, key=importances.get)

    trial = [f for f in remaining if f != to_remove]
    mdl2 = train_cb(x_tr[trial], y_tr, x_va[trial], y_va)
    pred2 = np.clip(mdl2.predict(x_va[trial].astype(float).values), m.PRED_MIN, m.PRED_MAX)
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
        print(f"Resumed from checkpoint: iteration={state['iteration']}, "
              f"remaining={len(state['remaining_features'])}, done={state.get('done', False)}")
    else:
        mdl0 = train_cb(x_tr[m.FEATURES], y_tr, x_va[m.FEATURES], y_va)
        pred0 = np.clip(mdl0.predict(x_va[m.FEATURES].astype(float).values), m.PRED_MIN, m.PRED_MAX)
        r2_baseline = r2_score(y_va, pred0)
        state = dict(remaining_features=list(m.FEATURES), eliminated=[], iteration=0,
                      r2_baseline=r2_baseline, log=[], done=False, stop_reason=None)
        print(f"NEW run. Baseline (50 features) val R2 = {r2_baseline:.4f}")
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
              f"iteration={state['iteration']} -- rerun script to continue.")


def finalize(state, x_tr, x_va, x_te, y_tr, y_va, y_te):
    remaining = state["remaining_features"]
    mdl_final = train_cb(x_tr[remaining], y_tr, x_va[remaining], y_va,
                          iters=FINAL_ITERS, lr=FINAL_LR, es=FINAL_ES)
    pred_te = np.clip(mdl_final.predict(x_te[remaining].astype(float).values), m.PRED_MIN, m.PRED_MAX)
    r2_test = r2_score(y_te, pred_te)

    mdl_base_final = train_cb(x_tr[m.FEATURES], y_tr, x_va[m.FEATURES], y_va,
                               iters=FINAL_ITERS, lr=FINAL_LR, es=FINAL_ES)
    pred_base_te = np.clip(mdl_base_final.predict(x_te[m.FEATURES].astype(float).values), m.PRED_MIN, m.PRED_MAX)
    r2_base_test = r2_score(y_te, pred_base_te)

    pd.DataFrame(state["log"]).to_csv(OUTPUT_DIR / "arm_C_v2_elimination_log.csv", index=False)
    pd.DataFrame([{"feature": f, "selected": f in remaining} for f in m.FEATURES]) \
        .to_csv(OUTPUT_DIR / "arm_C_v2_features.csv", index=False)
    summary = pd.DataFrame([{
        "location": "Bengkulu", "n_features_baseline": len(m.FEATURES),
        "n_features_pruned": len(remaining),
        "reduction_pct": 100.0 * (len(m.FEATURES) - len(remaining)) / len(m.FEATURES),
        "r2_baseline_val": state["r2_baseline"], "r2_baseline_test_fullconfig": r2_base_test,
        "r2_pruned_test_fullconfig": r2_test, "delta_r2_test": r2_base_test - r2_test,
        "selected_features": ", ".join(remaining),
    }])
    summary.to_csv(OUTPUT_DIR / "arm_C_v2_summary.csv", index=False)
    print(f"\n=== FINAL ===\nremaining={len(remaining)}/{len(m.FEATURES)} "
          f"({100*(len(m.FEATURES)-len(remaining))/len(m.FEATURES):.1f}% reduction)")
    print(f"R2 test (full-config): baseline={r2_base_test:.4f}  pruned={r2_test:.4f}  "
          f"delta={r2_base_test-r2_test:.4f}")
    print(f"Selected: {remaining}")


if __name__ == "__main__":
    main(max_iters_this_call=3)
