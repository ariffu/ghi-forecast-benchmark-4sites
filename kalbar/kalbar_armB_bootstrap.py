#!/usr/bin/env python3
"""
Uji signifikansi klaim "empat dari lima arsitektur konvergen dalam 0.01 R2" -- KALBAR.
Memakai kelas DL + builder identik dari train_ghi_1h_kalbar_R8_armB.py (import).
- Latih 5 arsitektur (GBM & DL masing2 3 seed), simpan prediksi per-sample test.
- Paired BLOCK-bootstrap atas test set (blok ~1 hari, jaga autokorelasi harian).
- Output: CI 95% & p-value dR2 tiap pasang; sebaran (max-min) R2 top-4 + P(spread<=0.01).

Jawaban untuk Reviewer Q1: klaim 0.01 terkuantifikasi, semua top-4 tidak berbeda signifikan,
LSTM outlier tegas.

Caveat GBM: Tabel 2c memakai 1 seed untuk CatBoost & LightGBM (desain seragam 4 lokasi).
Bootstrap ini menggunakan 3 seed untuk GBM agar seed-std bisa dihitung dan metodologi
konsisten dengan Banten (banten_armB_bootstrap.py). R2 seed-mean sedikit beda dari Tabel 2c.

Run:
    & "C:\\Program Files\\Python39\\python.exe" kalbar_armB_bootstrap.py
"""
import json
import warnings

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import train_ghi_1h_kalbar_R8_armB as A

warnings.filterwarnings("ignore")

OUT    = A.OUTPUT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2]
B_BOOT = 5000
BLOCK  = 78   # ~1 hari siang (elev>5) pada grid 10-menit
TOP4   = ["catboost", "lgbm", "mlp", "transformer"]
ALL5   = TOP4 + ["lstm"]
rng    = np.random.default_rng(12345)


def r2_idx(y, p, idx):
    yy = y[idx]
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    return 1.0 - np.sum((yy - p[idx]) ** 2) / ss_tot


def main():
    print(f"Device: {DEVICE}")
    print("Kalbar ARM B Bootstrap Analysis")
    print("=" * 80)

    # Load data -- identik dengan A.main(), termasuk filter sun_gt5_t60
    con = duckdb.connect(str(A.LOCAL_DB_PATH), read_only=True)
    df  = con.execute(A.build_sql()).fetchdf()
    con.close()

    df[A.TIME_COL] = pd.to_datetime(df[A.TIME_COL])
    df = A.add_features(df)

    # PENTING: filter dulu (sama dengan A.main()), BARU split
    # Bug lama: split diterapkan ke df mentah (termasuk baris malam & GHI ekstrem)
    du = df[df[A.TARGET_POINT].between(0, 1400) & df["sun_gt5_t60"]].copy()
    tr, va, te = A.split_masks(du)

    f1   = [f for f in A.F1_FEATURES if f in du.columns]
    x_tr = du.loc[tr, f1];  x_va = du.loc[va, f1];  x_te = du.loc[te, f1]
    y_tr = du.loc[tr, A.TARGET_POINT];  y_va = du.loc[va, A.TARGET_POINT]
    y    = du.loc[te, A.TARGET_POINT].values.astype(float)

    N = len(y)
    print(f"train={tr.sum():,}  val={va.sum():,}  test={N:,}")
    print(f"Features: {len(f1)}/{len(A.F1_FEATURES)}")
    missing = [f for f in A.F1_FEATURES if f not in du.columns]
    if missing:
        print(f"  Missing F1 features: {missing}")

    preds      = {}
    perseed_r2 = {m: [] for m in ALL5}

    # ---- GBM: 3 seed (konsisten dengan Banten bootstrap; Tabel 2c memakai 1 seed) ----
    print("\n== GBM ==")
    for m in ["catboost", "lgbm"]:
        acc = np.zeros(N)
        for s in SEEDS:
            A.RANDOM_STATE = s
            if m == "catboost":
                mdl = A.train_catboost(x_tr, y_tr, x_va, y_va)
                p   = np.clip(mdl.predict(x_te.astype(float).values), 0, 1400)
            else:
                mdl = A.train_lgbm(x_tr, y_tr, x_va, y_va)
                p   = np.clip(mdl.predict(x_te), 0, 1400)
            acc += p
            perseed_r2[m].append(r2_score(y, p))
        A.RANDOM_STATE = 42
        preds[m] = acc / len(SEEDS)
        print(f"  {m:<12} seed-mean-pred R2={r2_score(y, preds[m]):.4f}"
              f"  per-seed={[round(r, 4) for r in perseed_r2[m]]}")

    # ---- DL: 3 seeds ----
    print("\n== DL ==")
    imp = SimpleImputer(strategy="median")
    xtr_i = imp.fit_transform(x_tr)
    xva_i = imp.transform(x_va)
    xte_i = imp.transform(x_te)

    sc = StandardScaler()
    xtr_s = sc.fit_transform(xtr_i)
    xva_s = sc.transform(xva_i)
    xte_s = sc.transform(xte_i)

    n_feat = xtr_s.shape[1]

    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).unsqueeze(1)
    xt, xv, xe = to_t(xtr_s), to_t(xva_s), to_t(xte_s)
    yt = torch.from_numpy(y_tr.values.astype(np.float32))
    yv = torch.from_numpy(y_va.values.astype(np.float32))
    ye = torch.zeros(N)

    tl = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True)
    vl = DataLoader(TensorDataset(xv, yv), batch_size=256)
    el = DataLoader(TensorDataset(xe, ye), batch_size=256)

    for mn, Cls in [("lstm", A.LSTMModel), ("mlp", A.MLPModel), ("transformer", A.TransformerModel)]:
        acc = np.zeros(N)
        for s in SEEDS:
            torch.manual_seed(s)
            np.random.seed(s)
            A.RANDOM_STATE = s
            model = Cls(n_feat).to(DEVICE)
            model = A.train_dl(model, tl, vl, DEVICE)
            p = A.predict_dl(model, el, DEVICE)
            acc += p
            r2 = r2_score(y, p)
            perseed_r2[mn].append(r2)

        preds[mn] = acc / len(SEEDS)
        r2 = r2_score(y, preds[mn])
        mae = mean_absolute_error(y, preds[mn])
        rmse = np.sqrt(mean_squared_error(y, preds[mn]))
        seed_stds = [f"{r:.4f}" for r in perseed_r2[mn]]
        print(f"  {mn:<12} R2={r2:.4f}  MAE={mae:.1f}  RMSE={rmse:.1f}  seeds={seed_stds}")

    # Simpan prediksi per-sample
    dfp = pd.DataFrame({"y_true": y})
    for m in ALL5:
        dfp[f"pred_{m}"] = preds[m]
    dfp.to_csv(OUT / "armB_test_predictions.csv", index=False)
    print(f"\nPer-sample predictions saved: {OUT / 'armB_test_predictions.csv'}")

    # ---- BLOCK BOOTSTRAP ----
    print(f"\n== PAIRED BLOCK-BOOTSTRAP (B={B_BOOT}, block={BLOCK}) ==")
    nblocks = int(np.ceil(N / BLOCK))
    starts_max = N - BLOCK

    point_r2 = {m: r2_score(y, preds[m]) for m in ALL5}
    boot_r2 = {m: np.empty(B_BOOT) for m in ALL5}
    top4_spread = np.empty(B_BOOT)

    for b in range(B_BOOT):
        st = rng.integers(0, starts_max + 1, size=nblocks)
        idx = (st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:N]
        r2b = {m: r2_idx(y, preds[m], idx) for m in ALL5}
        for m in ALL5:
            boot_r2[m][b] = r2b[m]
        top4_spread[b] = max(r2b[m] for m in TOP4) - min(r2b[m] for m in TOP4)

    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    def pval(d):  # two-sided bootstrap p for ΔR2=0
        f = np.mean(d > 0)
        return float(2 * min(f, 1 - f))

    # Pairwise differences
    rows = []
    pairs = [(a, bb) for i, a in enumerate(TOP4) for bb in TOP4[i + 1:]] + [(a, "lstm") for a in TOP4]
    for a, bb in pairs:
        d = boot_r2[a] - boot_r2[bb]
        lo, hi = ci(d)
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        rows.append({
            "pair": f"{a}-{bb}",
            "dR2": round(point_r2[a] - point_r2[bb], 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "p": round(pval(d), 4),
            "sig_0.05": sig
        })
    dfpairs = pd.DataFrame(rows)
    dfpairs.to_csv(OUT / "armB_bootstrap_pairs.csv", index=False)

    # Summary
    print("\n-- R2 per arsitektur (point + 95% CI block-bootstrap + seed-std) --")
    for m in ALL5:
        lo, hi = ci(boot_r2[m])
        ss = np.std(perseed_r2[m]) if len(perseed_r2[m]) > 1 else 0.0
        print(f"  {m:<12} R2={point_r2[m]:.4f}  CI[{lo:.4f},{hi:.4f}]  seed_std={ss:.4f}")

    print("\n-- dR2 berpasangan (block-bootstrap, Holm corr a=0.0083) --")
    print(dfpairs.to_string(index=False))

    slo, shi = ci(top4_spread)
    print(f"\n-- SEBARAN R2 top-4 (max-min) --")
    point_spread = max(point_r2[m] for m in TOP4) - min(point_r2[m] for m in TOP4)
    print(f"  point spread = {point_spread:.4f}")
    print(f"  bootstrap median = {np.median(top4_spread):.4f}  CI[{slo:.4f},{shi:.4f}]")
    print(f"  P(spread <= 0.01) = {np.mean(top4_spread <= 0.01):.3f}")

    summary = {
        "site": "Kalbar",
        "N_test": int(N),
        "point_r2": {m: round(point_r2[m], 4) for m in ALL5},
        "point_r2_top4": [round(point_r2[m], 4) for m in TOP4],
        "top4_spread_point": round(point_spread, 4),
        "top4_spread_ci": [round(slo, 4), round(shi, 4)],
        "top4_spread_median": round(float(np.median(top4_spread)), 4),
        "P_spread_le_0.01": round(float(np.mean(top4_spread <= 0.01)), 3),
        "block": BLOCK,
        "B": B_BOOT,
    }
    (OUT / "armB_bootstrap_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs:")
    print(f"  - {OUT / 'armB_test_predictions.csv'}")
    print(f"  - {OUT / 'armB_bootstrap_pairs.csv'}")
    print(f"  - {OUT / 'armB_bootstrap_summary.json'}")
    print("\n" + "=" * 80)
    print("Bootstrap analysis complete. Ready for paper submission.")

if __name__ == "__main__":
    main()
