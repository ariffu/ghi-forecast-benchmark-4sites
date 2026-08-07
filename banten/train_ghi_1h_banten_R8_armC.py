#!/usr/bin/env python3
"""
R8 Arm C -- Banten: Feature Pruning via Validation-Guided Greedy Backward
Elimination. Belum pernah dijalankan sebelumnya (lihat
Restrukturisasi/07_Status_dan_Rencana_Selanjutnya.md Prioritas A). Metodologi
identik dengan train_ghi_1h_kalbar_R8_armC.py / bengkulu_R8_armC_v2.py --
backward elimination murni dari F1 (50 fitur), BUKAN top-K dari superset.

Resumable: state di-checkpoint ke arm_C_checkpoint.pkl tiap iterasi.

Run:
    python train_ghi_1h_banten_R8_armC.py
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

DB_PATH = "banten.duckdb"
OUTPUT_DIR = Path(__file__).parent / "outputs_R8_banten"
OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_PATH = OUTPUT_DIR / "arm_C_checkpoint_v2.pkl"

STATION_LAT_DEG, STATION_LON_DEG, WIB_MERIDIAN_DEG = -6.26147, 106.7509, 105.0
TIME_COL = "ts_wib"
TRAIN_END, VALID_END = "2024-01-01", "2025-01-01"
PRED_MIN, PRED_MAX = 0.0, 1400.0
EPSILON = 0.001

PRUNE_ITERS, PRUNE_LR, PRUNE_ES = 800, 0.03, 60  # heavier than Kalbar/Bengkulu template: Banten's
# weaker baseline signal (val R2~0.68 vs ~0.80 Bengkulu) made the original lighter config
# (400/0.05/40) unstable -- verified empirically: it flagged clp_cloud_present removal as
# delta=0.0023 (false stop), but delta=-0.0001..-0.0002 with 800-1200 iters. See arm_C notes.
FINAL_ITERS, FINAL_LR, FINAL_ES = 4000, 0.02, 150

FEATURES_GHI = ["ghi_now", "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m", "ghi_lag_60m",
                "ghi_lag_120m", "ghi_lag_180m", "ghi_roll_30m_mean", "ghi_roll_30m_std",
                "ghi_roll_60m_mean", "ghi_roll_60m_std", "ghi_roll_180m_mean", "ghi_roll_180m_std",
                "ghi_delta_10m", "ghi_delta_60m", "accel_ghi_20m"]
FEATURES_KT = ["kt_now", "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
               "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean", "accel_kt_20m"]
FEATURES_CLP = ["clp_cot", "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",
                "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",
                "clp_cot_roll_180m_mean", "accel_clp_cot_20m", "clp_cth_m", "clp_ctt_k", "clp_cer",
                "clp_cloud_present"]
FEATURES_TIME = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
FEATURES_FUTURE = ["ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg"]
FEATURES = FEATURES_GHI + FEATURES_KT + FEATURES_CLP + FEATURES_TIME + FEATURES_FUTURE
assert len(FEATURES) == 50
TARGET_POINT = "ghi_point_t60"


def build_sql():
    return """
    WITH base AS (
        SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m,
               cloud_top_temp AS clp_ctt_k, cloud_eff_radius AS clp_cer,
               CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present
        FROM solar_features_base
    ), with_kt AS (
        SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base
    ), w AS (
        SELECT *,
          LAG(ghi_now,1) OVER o AS ghi_lag_10m, LAG(ghi_now,2) OVER o AS ghi_lag_20m,
          LAG(ghi_now,3) OVER o AS ghi_lag_30m, LAG(ghi_now,6) OVER o AS ghi_lag_60m,
          LAG(ghi_now,12) OVER o AS ghi_lag_120m, LAG(ghi_now,18) OVER o AS ghi_lag_180m,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
          LAG(kt_point,1) OVER o AS kt_lag_10m, LAG(kt_point,2) OVER o AS kt_lag_20m,
          LAG(kt_point,3) OVER o AS kt_lag_30m, LAG(kt_point,6) OVER o AS kt_lag_60m,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
          LAG(clp_cot,1) OVER o AS clp_cot_lag_10m, LAG(clp_cot,2) OVER o AS clp_cot_lag_20m,
          LAG(clp_cot,3) OVER o AS clp_cot_lag_30m, LAG(clp_cot,6) OVER o AS clp_cot_lag_60m,
          AVG(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
          LEAD(ghi_now,6) OVER o AS ghi_lead_60m
        FROM with_kt WINDOW o AS (ORDER BY ts_wib)
    )
    SELECT * FROM w WHERE solar_elev_deg>5 AND ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib
    """


def solar_elevation_deg(ts, lat=STATION_LAT_DEG, lon=STATION_LON_DEG, meridian=WIB_MERIDIAN_DEG):
    idx = pd.DatetimeIndex(ts)
    doy = idx.dayofyear.values.astype(float)
    h = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    ha = ((h + 4.0 * (lon - meridian) / 60.0) - 12.0) * 15.0
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1, 1)))


def clearsky_simple(e):
    return 1100.0 * np.maximum(np.sin(np.deg2rad(e)), 0.0)


def add_features(df):
    out = df.copy()
    ts = pd.DatetimeIndex(out[TIME_COL])
    out["kt_now"] = out["ghi_now"].values / np.maximum(clearsky_simple(out["solar_elev_deg"].values.astype(float)), 20.0)
    out["clp_cot_delta_10m"] = out["clp_cot"] - out["clp_cot_lag_10m"]
    out["clp_cot_delta_30m"] = out["clp_cot"] - out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"] = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["accel_ghi_20m"] = out["ghi_now"] - 2 * out["ghi_lag_10m"] + out["ghi_lag_20m"]
    out["accel_kt_20m"] = out["kt_now"] - 2 * out["kt_lag_10m"] + out["kt_lag_20m"]
    out["accel_clp_cot_20m"] = out["clp_cot"] - 2 * out["clp_cot_lag_10m"] + out["clp_cot_lag_20m"]
    out["ghi_delta_10m"] = out["ghi_now"] - out["ghi_lag_10m"]
    out["ghi_delta_60m"] = out["ghi_now"] - out["ghi_lag_60m"]
    hh = ts.hour.values.astype(float) + ts.minute.values.astype(float) / 60.0
    doy = ts.dayofyear.values.astype(float)
    mo = ts.month.values.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hh / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hh / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * mo / 12)
    out["month_cos"] = np.cos(2 * np.pi * mo / 12)
    elev_t60 = solar_elevation_deg(out[TIME_COL] + pd.Timedelta(minutes=60))
    out["elev_sin_t60"] = np.maximum(np.sin(np.deg2rad(elev_t60)), 0.0)
    out["ghi_cs_t60"] = clearsky_simple(elev_t60)
    cs = [clearsky_simple(solar_elevation_deg(out[TIME_COL] + pd.Timedelta(minutes=s * 10))) for s in range(1, 7)]
    out["ghi_cs_avg_t10_t60"] = np.column_stack(cs).mean(axis=1)
    out["smart_persist"] = out["kt_now"] * out["ghi_cs_t60"]
    out["smart_persist_avg"] = out["kt_now"] * out["ghi_cs_avg_t10_t60"]
    out[TARGET_POINT] = out["ghi_lead_60m"].copy()
    out["sun_gt5_t60"] = out["elev_sin_t60"] > np.sin(np.deg2rad(5.0))
    return out


def split_masks(df):
    ts = df[TIME_COL]
    return (ts < pd.Timestamp(TRAIN_END),
            (ts >= pd.Timestamp(TRAIN_END)) & (ts < pd.Timestamp(VALID_END)),
            ts >= pd.Timestamp(VALID_END))


def train_cb(x_tr, y_tr, x_va, y_va, iters=PRUNE_ITERS, lr=PRUNE_LR, es=PRUNE_ES):
    mdl = CatBoostRegressor(iterations=iters, learning_rate=lr, depth=8, l2_leaf_reg=3.0,
                             loss_function="RMSE", random_seed=42, verbose=False,
                             thread_count=-1, allow_writing_files=False)
    mdl.fit(x_tr.astype(float).values, y_tr.astype(float).values,
            eval_set=(x_va.astype(float).values, y_va.astype(float).values), early_stopping_rounds=es)
    return mdl


def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_features(df)
    df_pt = df[df[TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    tr_m, va_m, te_m = split_masks(df_pt)
    return (df_pt.loc[tr_m, FEATURES], df_pt.loc[va_m, FEATURES], df_pt.loc[te_m, FEATURES],
            df_pt.loc[tr_m, TARGET_POINT], df_pt.loc[va_m, TARGET_POINT], df_pt.loc[te_m, TARGET_POINT])


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


def main(max_iters_this_call=2):
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
        print(f"iter {last['iteration']}: removed={last['removed']} r2_after={last['r2_after_removal']:.4f} "
              f"delta={last['delta_vs_baseline']:.4f} remaining={len(state['remaining_features'])}")

    if state.get("done"):
        print(f"DONE this call: {state.get('stop_reason')}")
        finalize(state, x_tr, x_va, x_te, y_tr, y_va, y_te)
    else:
        print(f"Paused (budget). remaining={len(state['remaining_features'])}, iteration={state['iteration']} -- rerun to continue.")


def finalize(state, x_tr, x_va, x_te, y_tr, y_va, y_te):
    remaining = state["remaining_features"]
    mdl_final = train_cb(x_tr[remaining], y_tr, x_va[remaining], y_va, iters=FINAL_ITERS, lr=FINAL_LR, es=FINAL_ES)
    pred_te = np.clip(mdl_final.predict(x_te[remaining].astype(float).values), PRED_MIN, PRED_MAX)
    r2_test = r2_score(y_te, pred_te)
    mdl_base = train_cb(x_tr[FEATURES], y_tr, x_va[FEATURES], y_va, iters=FINAL_ITERS, lr=FINAL_LR, es=FINAL_ES)
    pred_base_te = np.clip(mdl_base.predict(x_te[FEATURES].astype(float).values), PRED_MIN, PRED_MAX)
    r2_base_test = r2_score(y_te, pred_base_te)
    pd.DataFrame(state["log"]).to_csv(OUTPUT_DIR / "arm_C_elimination_log.csv", index=False)
    pd.DataFrame([{"feature": f, "selected": f in remaining} for f in FEATURES]).to_csv(OUTPUT_DIR / "arm_C_features.csv", index=False)
    summary = pd.DataFrame([{
        "location": "Banten", "n_features_baseline": len(FEATURES), "n_features_pruned": len(remaining),
        "reduction_pct": 100.0 * (len(FEATURES) - len(remaining)) / len(FEATURES),
        "r2_baseline_val": state["r2_baseline"], "r2_baseline_test_fullconfig": r2_base_test,
        "r2_pruned_test_fullconfig": r2_test, "delta_r2_test": r2_base_test - r2_test,
        "selected_features": ", ".join(remaining),
    }])
    summary.to_csv(OUTPUT_DIR / "arm_C_summary.csv", index=False)
    print(f"\n=== FINAL === remaining={len(remaining)}/{len(FEATURES)} ({100*(len(FEATURES)-len(remaining))/len(FEATURES):.1f}% reduction)")
    print(f"R2 test (full-config): baseline={r2_base_test:.4f} pruned={r2_test:.4f} delta={r2_base_test-r2_test:.4f}")
    print(f"Selected: {remaining}")


if __name__ == "__main__":
    main(max_iters_this_call=2)
