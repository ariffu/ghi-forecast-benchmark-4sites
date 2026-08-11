#!/usr/bin/env python3
"""
Uji signifikansi klaim "empat dari lima arsitektur konvergen dalam 0.01 R2" -- BENGKULU.
Memakai kelas DL + builder identik dari train_ghi_1h_bengkulu_R8_armB.py (import).

Metodologi identik Banten (banten_armB_bootstrap.py) dan Kalbar (kalbar_armB_bootstrap.py):
  - Latih 5 arsitektur (GBM & DL masing2 3 seed), simpan prediksi per-sample test.
  - Paired BLOCK-bootstrap atas test set (blok ~1 hari, jaga autokorelasi harian).
  - B=5000 resample, block=78 (~1 hari siang pada grid 10-menit).
  - Output: CI 95% & p-value dR2 tiap pasang; sebaran (max-min) R2 top-4 + P(spread<=0.01).

Caveat GBM: Tabel 2c memakai 1 seed untuk CatBoost & LightGBM (desain seragam 4 lokasi).
Bootstrap ini menggunakan 3 seed untuk GBM agar seed-std bisa dihitung dan metodologi
konsisten dengan Banten & Kalbar bootstrap. R2 seed-mean sedikit beda dari Tabel 2c.

Run (dari folder bengkulu_ghi_julius):
    & "C:\\Program Files\\Python39\\python.exe" bengkulu_armB_bootstrap.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import train_ghi_1h_bengkulu_R8_armB as A

warnings.filterwarnings("ignore")

OUT    = Path(r"C:\Users\ariff\bengkulu_ghi_julius\outputs_R8_Bengkulu")
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2]
B_BOOT = 5_000
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
    print("Bengkulu ARM B Bootstrap Analysis")
    print("=" * 80)

    # Load data via A.connect_data() -- handle ATTACH LOCAL/MotherDuck otomatis
    con = A.connect_data()
    df  = con.execute(A.build_sql()).fetchdf()
    con.close()

    df[A.TIME_COL] = pd.to_datetime(df[A.TIME_COL])
    df = A.add_features(df)

    # Filter dulu (identik A.main()), BARU split
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

    # ---- GBM: 3 seed (Tabel 2c memakai 1 seed -- caveat di docstring) ----
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
        A.RANDOM_STATE = 42   # reset ke default
        preds[m] = acc / len(SEEDS)
        print(f"  {m:<12} seed-mean-pred R2={r2_score(y, preds[m]):.4f}"
              f"  per-seed={[round(r, 4) for r in perseed_r2[m]]}")

    # ---- DL: prep identik A.main() ----
    print("\n== DL ==")
    imp   = SimpleImputer(strategy="median")
    xtr_i = imp.fit_transform(x_tr);  xva_i = imp.transform(x_va);  xte_i = imp.transform(x_te)
    sc    = StandardScaler()
    xtr_s = sc.fit_transform(xtr_i);  xva_s = sc.transform(xva_i);  xte_s = sc.transform(xte_i)
    n_feat = xtr_s.shape[1]

    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).unsqueeze(1)
    xt = to_t(xtr_s);  xv = to_t(xva_s);  xe = to_t(xte_s)
    yt = torch.from_numpy(y_tr.values.astype(np.float32))
    yv = torch.from_numpy(y_va.values.astype(np.float32))
    ye = torch.zeros(N)
    tl = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True)
    vl = DataLoader(TensorDataset(xv, yv), batch_size=256)
    el = DataLoader(TensorDataset(xe, ye), batch_size=256)

    for mn, Cls in [("lstm", A.LSTMModel), ("mlp", A.MLPModel), ("transformer", A.TransformerModel)]:
        acc = np.zeros(N)
        for s in SEEDS:
            torch.manual_seed(s);  np.random.seed(s)
            model = Cls(n_feat).to(DEVICE)
            model = A.train_dl(model, tl, vl, DEVICE)
            p     = A.predict_dl(model, el, DEVICE)
            acc  += p
            perseed_r2[mn].append(r2_score(y, p))
        preds[mn] = acc / len(SEEDS)
        print(f"  {mn:<12} seed-mean-pred R2={r2_score(y, preds[mn]):.4f}"
              f"  per-seed={[round(r, 4) for r in perseed_r2[mn]]}")

    # simpan prediksi per-sample (untuk audit / re-bootstrap tanpa retrain)
    dfp = pd.DataFrame({"y_true": y})
    for m in ALL5:
        dfp[f"pred_{m}"] = preds[m]
    dfp.to_csv(OUT / "armB_test_predictions.csv", index=False)
    print(f"\nSaved per-sample predictions -> {OUT / 'armB_test_predictions.csv'}")

    # ---- BLOCK bootstrap ----
    print(f"\n== PAIRED BLOCK-BOOTSTRAP (B={B_BOOT}, block={BLOCK}) ==")
    nblocks    = int(np.ceil(N / BLOCK))
    starts_max = N - BLOCK
    point_r2   = {m: r2_score(y, preds[m]) for m in ALL5}
    boot_r2    = {m: np.empty(B_BOOT) for m in ALL5}
    top4_spread = np.empty(B_BOOT)

    for b in range(B_BOOT):
        st  = rng.integers(0, starts_max + 1, size=nblocks)
        idx = (st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:N]
        r2b = {m: r2_idx(y, preds[m], idx) for m in ALL5}
        for m in ALL5:
            boot_r2[m][b] = r2b[m]
        top4_spread[b] = max(r2b[m] for m in TOP4) - min(r2b[m] for m in TOP4)

    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    def pval(d):
        f = np.mean(d > 0)
        return float(2 * min(f, 1 - f))

    rows  = []
    pairs = ([(a, bb) for i, a in enumerate(TOP4) for bb in TOP4[i + 1:]]
             + [(a, "lstm") for a in TOP4])
    for a, bb in pairs:
        d      = boot_r2[a] - boot_r2[bb]
        lo, hi = ci(d)
        rows.append({
            "pair":     f"{a}-{bb}",
            "dR2":      round(point_r2[a] - point_r2[bb], 4),
            "ci_lo":    round(lo, 4),
            "ci_hi":    round(hi, 4),
            "p":        round(pval(d), 4),
            "sig_0.05": "YES" if (lo > 0 or hi < 0) else "no",
        })
    dfpairs = pd.DataFrame(rows)
    dfpairs.to_csv(OUT / "armB_bootstrap_pairs.csv", index=False)

    # ---- ringkasan output ----
    print("\n-- R2 per arsitektur (point + 95% CI block-bootstrap + seed-std) --")
    for m in ALL5:
        lo, hi = ci(boot_r2[m])
        ss = np.std(perseed_r2[m])
        print(f"  {m:<12} R2={point_r2[m]:.4f}  CI[{lo:.4f},{hi:.4f}]  seed_std={ss:.4f}")

    print("\n-- dR2 berpasangan (block-bootstrap, Holm corr a=0.0083) --")
    print(dfpairs.to_string(index=False))

    slo, shi    = ci(top4_spread)
    spread_pt   = max(point_r2[m] for m in TOP4) - min(point_r2[m] for m in TOP4)
    print(f"\n-- SEBARAN R2 top-4 (max-min) --")
    print(f"  point spread     = {spread_pt:.4f}")
    print(f"  bootstrap median = {np.median(top4_spread):.4f}  CI[{slo:.4f},{shi:.4f}]")
    print(f"  P(spread <= 0.01) = {np.mean(top4_spread <= 0.01):.3f}")

    # simpan summary JSON
    summary = {
        "site": "bengkulu",
        "N_test": N,
        "point_r2": {m: round(point_r2[m], 4) for m in ALL5},
        "seed_std": {m: round(float(np.std(perseed_r2[m])), 4) for m in ALL5},
        "top4_spread_point": round(spread_pt, 4),
        "top4_spread_ci": [round(slo, 4), round(shi, 4)],
        "P_spread_le_0.01": round(float(np.mean(top4_spread <= 0.01)), 3),
        "block": BLOCK,
        "B": B_BOOT,
        "note_gbm_seeds": "3 seeds (bootstrap); Tabel 2c memakai 1 seed",
    }
    (OUT / "armB_bootstrap_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs -> {OUT}/")
    print("  armB_test_predictions.csv")
    print("  armB_bootstrap_pairs.csv")
    print("  armB_bootstrap_summary.json")


if __name__ == "__main__":
    main()
