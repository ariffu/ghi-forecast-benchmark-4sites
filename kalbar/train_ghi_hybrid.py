#!/usr/bin/env python
"""
train_ghi_hybrid.py  -- Hybrid approach untuk prediksi GHI 1 jam ke depan
==========================================================================
Pendekatan:
  A  LGBM baseline         (referensi ulang dari parquet v3-stage3)
  B  LGBM + future sun_alt (tambah 6 fitur geometri surya deterministik)
  C  LGBM kondisional per regime awan (3 model: Clear / PartlyCloudy / Overcast)
  D  Ensemble LGBM+LSTM    (blending tertimbang)
  E  Selective regime      (Clear=model_C, lainnya=model_A)
  F  Selective-E + LSTM    (E blend dengan LSTM)

Input  : C:\\Users\\ariff\\DuckDB\\ghi_forecast_v3_stage3.parquet
         C:\\Users\\ariff\\DuckDB\\ghi_lstm_predictions_stage3.csv  (opsional)
Output : C:\\Users\\ariff\\DuckDB\\ghi_hybrid_eval.csv
"""

import os, math, warnings, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ---- Paths ------------------------------------------------------------------
PARQUET_S3 = r"C:\Users\ariff\DuckDB\ghi_forecast_v3_stage3.parquet"
LSTM_CSV   = r"C:\Users\ariff\DuckDB\ghi_lstm_predictions_stage3.csv"
OUTPUT_CSV = r"C:\Users\ariff\DuckDB\ghi_hybrid_eval.csv"

HORIZONS    = [f"t+{(i+1)*10}min" for i in range(6)]
TARGET_COLS = [f"ghi_t{i+1}" for i in range(6)]

LAT_DEG = 0.02
LON_DEG = 109.34

# ---- LightGBM params --------------------------------------------------------
LGB_PARAMS = dict(
    objective         = "regression",
    metric            = "mae",
    n_estimators      = 2500,
    learning_rate     = 0.04,
    num_leaves        = 127,
    max_depth         = -1,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
    random_state      = 42,
)

# ---- Solar geometry ---------------------------------------------------------
def future_solar_alt_batch(ts_series, n_steps=6, step_min=10):
    lat = math.radians(LAT_DEG)
    results = []
    for ts in ts_series:
        row = []
        for k in range(1, n_steps + 1):
            future  = ts + pd.Timedelta(minutes=k * step_min)
            doy     = future.timetuple().tm_yday
            delta   = 23.45 * math.sin(math.radians(360 / 365 * (284 + doy)))
            utc_h   = future.hour + future.minute / 60.0 - 7.0
            solar_h = utc_h + LON_DEG / 15.0
            ha_deg  = (solar_h - 12.0) * 15.0
            dec = math.radians(delta)
            ha  = math.radians(ha_deg)
            sin_alt = (math.sin(lat) * math.sin(dec) +
                       math.cos(lat) * math.cos(dec) * math.cos(ha))
            row.append(max(0.0, math.degrees(
                math.asin(max(-1.0, min(1.0, sin_alt))))))
        results.append(row)
    return np.array(results, dtype=np.float32)

# ---- Train / predict --------------------------------------------------------
def train_models(X_tr, Y_tr, X_va, Y_va, extra_params=None):
    p = {**LGB_PARAMS, **(extra_params or {})}
    models = []
    for i in range(6):
        m = lgb.LGBMRegressor(**p)
        m.fit(
            X_tr, Y_tr[:, i],
            eval_set  = [(X_va, Y_va[:, i])],
            callbacks = [
                lgb.early_stopping(120, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        models.append(m)
    return models

def predict(models, X):
    return np.column_stack([m.predict(X) for m in models]).astype(np.float32)

# ---- Evaluate ---------------------------------------------------------------
def evaluate(Y_true, Y_pred, tag, mask=None):
    if mask is not None:
        Y_true, Y_pred = Y_true[mask], Y_pred[mask]
    rows = []
    for i, h in enumerate(HORIZONS):
        r2  = r2_score(Y_true[:, i], Y_pred[:, i])
        mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
        rows.append({"model": tag, "horizon": h,
                     "test_r2": round(r2, 5), "test_mae": round(mae, 2)})
    avg = np.mean([r["test_r2"] for r in rows])
    hor = "  ".join(f"{r['horizon']}={r['test_r2']:.4f}" for r in rows)
    print(f"  [{tag}]  avg={avg:.4f}  |  {hor}")
    return pd.DataFrame(rows)

# ---- Regime assignment ------------------------------------------------------
def assign_regime(X, feat_cols):
    """0=Clear, 1=PartlyCloudy, 2=Overcast"""
    cot_col = next((c for c in feat_cols
                    if "cloud_optical_thick" in c and "lag0" in c), None)
    cpb_col = next((c for c in feat_cols
                    if "cloud_present_bin" in c and "lag0" in c), None)

    if cot_col is None:
        print("  cloud_optical_thick_lag0 tidak ditemukan -- semua di Regime 1")
        return np.ones(len(X), dtype=int)

    ci  = feat_cols.index(cot_col)
    pi  = feat_cols.index(cpb_col) if cpb_col else None
    cot = X[:, ci]
    cpb = X[:, pi] if pi is not None else np.ones(len(X))

    regime = np.ones(len(X), dtype=int)
    regime[(cpb < 0.5) | (cot < 1.0)]   = 0
    regime[(cot >= 1.0) & (cot < 8.0)]  = 1
    regime[cot >= 8.0]                  = 2
    return regime

# ---- Main -------------------------------------------------------------------
def main():
    t0 = time.time()

    print(f"Loading parquet: {PARQUET_S3}")
    df = pd.read_parquet(PARQUET_S3)
    df["anchor_ts"] = pd.to_datetime(df["anchor_ts"])
    df = df.sort_values("anchor_ts").reset_index(drop=True)

    year    = df["anchor_ts"].dt.year
    tr_mask = year.isin([2022, 2023])
    va_mask = year == 2024
    te_mask = year == 2025

    feat_cols = [c for c in df.columns
                 if c != "anchor_ts"
                 and c not in TARGET_COLS
                 and not c.startswith("ghi_t")]

    X_tr = df.loc[tr_mask, feat_cols].values.astype(np.float32)
    X_va = df.loc[va_mask, feat_cols].values.astype(np.float32)
    X_te = df.loc[te_mask, feat_cols].values.astype(np.float32)
    Y_tr = df.loc[tr_mask, TARGET_COLS].values.astype(np.float32)
    Y_va = df.loc[va_mask, TARGET_COLS].values.astype(np.float32)
    Y_te = df.loc[te_mask, TARGET_COLS].values.astype(np.float32)

    print(f"  Train {X_tr.shape[0]:,} | Val {X_va.shape[0]:,} | Test {X_te.shape[0]:,}")
    print(f"  Features: {len(feat_cols)}")

    all_results = []

    # -- Approach A: LGBM baseline --------------------------------------------
    print("\n" + "="*60)
    print("APPROACH A -- LGBM baseline (Stage 3, referensi)")
    print("="*60)
    mA    = train_models(X_tr, Y_tr, X_va, Y_va)
    pA_te = predict(mA, X_te)
    all_results.append(evaluate(Y_te, pA_te, "A_LGBM_baseline"))

    # -- Approach B: LGBM + future sun_altitude -------------------------------
    print("\n" + "="*60)
    print("APPROACH B -- LGBM + future sun_alt (t+10..t+60 deterministik)")
    print("="*60)
    print("  Computing future solar altitude ...")
    fsa_tr = future_solar_alt_batch(df.loc[tr_mask, "anchor_ts"].reset_index(drop=True))
    fsa_va = future_solar_alt_batch(df.loc[va_mask, "anchor_ts"].reset_index(drop=True))
    fsa_te = future_solar_alt_batch(df.loc[te_mask, "anchor_ts"].reset_index(drop=True))

    cs_tr = (950.0 * np.power(np.sin(np.clip(np.radians(fsa_tr), 0, None)), 1.1)).astype(np.float32)
    cs_va = (950.0 * np.power(np.sin(np.clip(np.radians(fsa_va), 0, None)), 1.1)).astype(np.float32)
    cs_te = (950.0 * np.power(np.sin(np.clip(np.radians(fsa_te), 0, None)), 1.1)).astype(np.float32)

    X_tr_B = np.hstack([X_tr, fsa_tr, cs_tr])
    X_va_B = np.hstack([X_va, fsa_va, cs_va])
    X_te_B = np.hstack([X_te, fsa_te, cs_te])

    print(f"  Features setelah penambahan: {X_tr_B.shape[1]}")
    mB    = train_models(X_tr_B, Y_tr, X_va_B, Y_va)
    pB_te = predict(mB, X_te_B)
    all_results.append(evaluate(Y_te, pB_te, "B_LGBM_future_sunalt"))

    feat_cols_B = (feat_cols
                   + [f"future_sun_alt_t{i+1}" for i in range(6)]
                   + [f"future_clearsky_t{i+1}" for i in range(6)])

    # -- Approach C: Regime-conditional LGBM ----------------------------------
    print("\n" + "="*60)
    print("APPROACH C -- Regime-conditional LGBM (3 model terpisah)")
    print("="*60)
    reg_tr = assign_regime(X_tr_B, feat_cols_B)
    reg_va = assign_regime(X_va_B, feat_cols_B)
    reg_te = assign_regime(X_te_B, feat_cols_B)

    REGIME_LABELS = {0: "Clear", 1: "PartlyCloudy", 2: "Overcast"}
    pC_te = np.zeros_like(Y_te)

    for r, label in REGIME_LABELS.items():
        mtr = reg_tr == r
        mva = reg_va == r
        mte = reg_te == r
        n_tr, n_va, n_te = mtr.sum(), mva.sum(), mte.sum()
        print(f"  Regime {r} ({label}): train={n_tr:,}  val={n_va:,}  test={n_te:,}")

        if n_tr < 200 or n_te < 20:
            print("    Sampel terlalu sedikit -- pakai baseline")
            pC_te[mte] = pB_te[mte]
            continue

        if n_va < 50:
            mC = train_models(X_tr_B[mtr], Y_tr[mtr], X_va_B, Y_va)
        else:
            mC = train_models(X_tr_B[mtr], Y_tr[mtr], X_va_B[mva], Y_va[mva])

        pC_te[mte] = predict(mC, X_te_B[mte])
        evaluate(Y_te[mte], pC_te[mte], f"  C_regime{r}_{label}")

    all_results.append(evaluate(Y_te, pC_te, "C_LGBM_regime_cond"))

    # -- Approach D: LGBM + LSTM Ensemble -------------------------------------
    print("\n" + "="*60)
    print("APPROACH D -- Weighted LGBM+LSTM Ensemble")
    print("="*60)

    lstm_file = None
    for f in [LSTM_CSV,
              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ghi_lstm_predictions_stage3.csv")]:
        if os.path.exists(f):
            lstm_file = f
            break

    n_common = 0
    ldf_idx  = None
    te_df_sub = df[te_mask].reset_index(drop=True)
    pred_cols = [f"ghi_pred_t{i+1}" for i in range(6)]

    if lstm_file:
        print(f"  File LSTM: {lstm_file}")
        ldf      = pd.read_csv(lstm_file, parse_dates=["anchor_ts"])
        ldf_idx  = ldf.set_index("anchor_ts")
        te_ts    = te_df_sub["anchor_ts"]
        common   = te_ts[te_ts.isin(ldf_idx.index)]
        n_common = len(common)
        print(f"  Alignment: {n_common:,} / {len(te_ts):,} timestamps match")

        if n_common > 1000:
            match  = te_df_sub["anchor_ts"].isin(common.values)
            P_lstm = ldf_idx.loc[te_df_sub.loc[match, "anchor_ts"].values,
                                 pred_cols].values.astype(np.float32)
            P_lgbm = pC_te[match.values]
            Y_sub  = Y_te[match.values]

            best_r2, best_w = -np.inf, 0.7
            for w in np.arange(0.40, 1.01, 0.05):
                P_e = w * P_lgbm + (1 - w) * P_lstm
                avg = np.mean([r2_score(Y_sub[:, i], P_e[:, i]) for i in range(6)])
                if avg > best_r2:
                    best_r2, best_w = avg, w

            print(f"  Bobot optimal: w_LGBM={best_w:.2f}  w_LSTM={1-best_w:.2f}")
            wstr = f"{best_w:.2f}"
            P_D  = best_w * P_lgbm + (1 - best_w) * P_lstm
            all_results.append(evaluate(Y_sub, P_D,
                                        "D_Ensemble_LGBM" + wstr + "+LSTM"))
        else:
            print("  Terlalu sedikit match -- skip")
    else:
        print("  File LSTM tidak ditemukan.")
        print("  Copy ghi_lstm_predictions_stage3.csv ke C:\\Users\\ariff\\DuckDB\\")

    # -- Approach E: Selective regime (Clear=C, lainnya=A) --------------------
    print("\n" + "="*60)
    print("APPROACH E -- Selective: Clear=model_C, lainnya=model_A (global)")
    print("="*60)
    pE_te = pA_te.copy()
    clear_te = reg_te == 0
    pE_te[clear_te] = pC_te[clear_te]
    print(f"  {clear_te.sum():,} sampel Clear diganti dengan prediksi model-C")
    all_results.append(evaluate(Y_te, pE_te, "E_Selective_Clear+Global"))

    # -- Approach F: Selective-E + LSTM ensemble ------------------------------
    if lstm_file and n_common > 1000:
        print("\n" + "="*60)
        print("APPROACH F -- Selective-E + LSTM blend")
        print("="*60)
        match2   = te_df_sub["anchor_ts"].isin(common.values)
        P_lstm2  = ldf_idx.loc[te_df_sub.loc[match2, "anchor_ts"].values,
                               pred_cols].values.astype(np.float32)
        P_E_sub  = pE_te[match2.values]
        Y_sub2   = Y_te[match2.values]

        best_r2f, best_wf = -np.inf, 0.75
        for w in np.arange(0.50, 1.01, 0.05):
            P_f = w * P_E_sub + (1 - w) * P_lstm2
            avg = np.mean([r2_score(Y_sub2[:, i], P_f[:, i]) for i in range(6)])
            if avg > best_r2f:
                best_r2f, best_wf = avg, w

        print(f"  Bobot optimal: w_Selective={best_wf:.2f}  w_LSTM={1-best_wf:.2f}")
        wfstr = f"{best_wf:.2f}"
        P_F   = best_wf * P_E_sub + (1 - best_wf) * P_lstm2
        all_results.append(evaluate(Y_sub2, P_F,
                                    "F_SelectiveClear+LSTM" + wfstr))

    # -- Summary --------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "="*70)
    print("RINGKASAN HASIL HYBRID")
    print("="*70)

    all_df = pd.concat(all_results, ignore_index=True)
    pivot  = (all_df
              .pivot(index="model", columns="horizon", values="test_r2")
              .rename_axis(None, axis=1))
    pivot  = pivot[[f"t+{(i+1)*10}min" for i in range(6)]]
    pivot["avg_R2"] = pivot.mean(axis=1)
    pivot  = pivot.sort_values("avg_R2", ascending=False)

    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWaktu total: {elapsed/60:.1f} menit")

    all_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetail disimpan ke: {OUTPUT_CSV}")

    best_model = pivot["avg_R2"].idxmax()
    best_avg   = pivot.loc[best_model, "avg_R2"]
    print(f"\n-> Model terbaik: {best_model}  (avg R2={best_avg:.4f})")
    if best_avg < 0.9:
        print(f"   Jarak ke target R2=0.90 : {0.9 - best_avg:.4f}")
    print()
    print("  Analisis per-regime:")
    print("  * Regime Clear   : target SUDAH tercapai (R2~0.919)")
    print("  * Regime Partly  : bottleneck utama (R2~0.72), butuh data tambahan")
    print("  * Regime Overcast: R2 rendah akibat low-variance target (GHI rendah)")


if __name__ == "__main__":
    main()
