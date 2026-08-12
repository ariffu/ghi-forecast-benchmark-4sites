#!/usr/bin/env python3
"""
diagnose_results.py
====================
Diagnosa mengapa R² rendah (~0.50 vs kalbar ~0.80).

Cek yang dilakukan:
  1. Persistence baseline — seberapa prediktif GHI(t) untuk GHI(t+h)?
  2. Smart persistence — clearsky-scaled persistence
  3. Feature importance top-20 per stage/horizon
  4. Distribusi GHI per split (shift distribusi?)
  5. Distribusi delta_ghi (seberapa volatil 10-menit?)
  6. NULL rate semua fitur lag di train/val/test
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

DATASET_DIR = Path(__file__).parent / "dataset"
MODELS_DIR  = Path(__file__).parent / "models"
HORIZON     = 6


def load_split(stage, split):
    return pd.read_parquet(DATASET_DIR / f"jambi_ghi_s{stage}_{split}.parquet")


# ─── 1. PERSISTENCE BASELINE ──────────────────────────────────────────────────
def persistence_baseline():
    """
    Persistence: GHI(t+h) ≈ GHI(t)
    Smart persistence: GHI(t+h) ≈ GHI(t) × [clearsky(t+h) / clearsky(t)]
    Ini adalah lower bound yang harus dilampaui model ML.
    """
    print("=" * 60)
    print("1. PERSISTENCE BASELINE")
    print("=" * 60)

    for split in ["val", "test"]:
        df = load_split(1, split)
        print(f"\n  {split.upper()} ({len(df):,} samples):")
        print(f"  {'Horizon':>10} | {'Persist R²':>10} | {'Smart R²':>10} | {'Persist RMSE':>12}")

        for h in range(1, HORIZON + 1):
            y_true   = df[f"ghi_h{h}"].values
            y_persist = df["anchor_ghi"].values  # GHI(t) → prediksi GHI(t+h)

            # Smart persistence: scale dengan rasio clearsky
            clr_h = df[f"clearsky_ghi_h{h}"].values
            clr_0 = df["ghi_clearsky"].values if "ghi_clearsky" in df.columns else None

            # anchor_ghi / clearsky_anchor × clearsky_h
            # Clearsky anchor bisa dihitung dari clearsky_ghi_h1 jika tidak ada
            # Kita pakai sun_alt sebagai proxy (clearsky ∝ sin(sun_alt)^0.9 )
            sin_h   = np.sin(np.radians(df[f"sun_alt_h{h}"].values))
            sin_0   = np.sin(np.radians(df["anchor_sun_alt"].values))
            # Hindari divisi near-zero
            ratio   = np.where(sin_0 > 0.05, sin_h / sin_0, 1.0)
            y_smart = y_persist * ratio

            mask = ~(np.isnan(y_true) | np.isnan(y_persist))
            r2_p  = r2_score(y_true[mask], y_persist[mask])
            rmse_p = np.sqrt(mean_squared_error(y_true[mask], y_persist[mask]))
            r2_s  = r2_score(y_true[mask], y_smart[mask])

            print(f"  t+{h} ({h*10:2d}min) | {r2_p:10.4f} | {r2_s:10.4f} | {rmse_p:12.1f}")


# ─── 2. DISTRIBUSI GHI ────────────────────────────────────────────────────────
def ghi_distribution():
    print("\n" + "=" * 60)
    print("2. DISTRIBUSI GHI PER SPLIT")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        df = load_split(1, split)
        g  = df["anchor_ghi"]
        print(f"\n  {split.upper()} ({len(df):,} samples):")
        print(f"    mean={g.mean():.1f}  std={g.std():.1f}  "
              f"p10={g.quantile(0.1):.1f}  p50={g.median():.1f}  "
              f"p90={g.quantile(0.9):.1f}  max={g.max():.1f}")
        # Distribusi per bin
        bins = pd.cut(g, bins=[0, 100, 300, 600, 900, 1200, 1500])
        print(f"    {bins.value_counts().sort_index().to_dict()}")


# ─── 3. VOLATILITAS DELTA GHI ─────────────────────────────────────────────────
def ghi_volatility():
    print("\n" + "=" * 60)
    print("3. VOLATILITAS DELTA_GHI (persen perubahan 10-menit)")
    print("=" * 60)

    df = load_split(1, "train")
    d = df["delta_ghi_lag1"].dropna()  # perubahan GHI dari t-1 ke t
    abs_d = d.abs()

    print(f"\n  Train delta_ghi_lag1 (GHI(t) - GHI(t-1)):")
    print(f"    mean_abs={abs_d.mean():.1f}  std={d.std():.1f}  "
          f"p50={abs_d.median():.1f}  p90={abs_d.quantile(0.9):.1f}  "
          f"p95={abs_d.quantile(0.95):.1f}  max={abs_d.max():.1f} W/m²")
    pct_large = (abs_d > 100).mean()
    print(f"    Fraksi |ΔGHI| > 100 W/m²: {pct_large:.1%}  ← indikator variabilitas awan")

    # Autocorrelation GHI lag
    print(f"\n  Autokorelasi GHI (seberapa prediktif GHI(t) untuk GHI(t+h)):")
    for h in range(1, HORIZON + 1):
        y_t  = df["anchor_ghi"].values
        y_th = df[f"ghi_h{h}"].values
        mask = ~(np.isnan(y_t) | np.isnan(y_th))
        corr = np.corrcoef(y_t[mask], y_th[mask])[0, 1]
        print(f"    corr(GHI_t, GHI_t+{h}) = {corr:.4f}  (t+{h*10:2d}min)")


# ─── 4. FEATURE IMPORTANCE ────────────────────────────────────────────────────
def feature_importance():
    print("\n" + "=" * 60)
    print("4. FEATURE IMPORTANCE TOP-15 (Stage 1, h+1 dan Stage 3, h+1)")
    print("=" * 60)

    for stage in [1, 3]:
        imp_path = MODELS_DIR / f"s{stage}_h1_importance.csv"
        if not imp_path.exists():
            print(f"\n  s{stage}_h1_importance.csv tidak ditemukan")
            continue
        imp = pd.read_csv(imp_path)
        total = imp["importance_gain"].sum()
        imp["pct"] = imp["importance_gain"] / total * 100
        top = imp.head(15)
        print(f"\n  Stage {stage}, h+1 (10min) — top 15 fitur:")
        for _, row in top.iterrows():
            print(f"    {row['feature']:35s} {row['importance_gain']:12.0f}  ({row['pct']:5.2f}%)")


# ─── 5. NULL RATE FITUR LAG ───────────────────────────────────────────────────
def null_rate_analysis():
    print("\n" + "=" * 60)
    print("5. NULL RATE FITUR LAG (train)")
    print("=" * 60)

    df = load_split(1, "train")
    lag_feats = [f"ghi_lag{k}" for k in range(1, 19)]
    delta_feats = [f"delta_ghi_lag{k}" for k in range(1, 19)]

    ghi_nulls   = df[lag_feats].isna().mean()
    delta_nulls = df[delta_feats].isna().mean()

    print(f"\n  ghi_lag NULL rate (harus 0 karena window validity):")
    print(f"    max={ghi_nulls.max():.4f}  mean={ghi_nulls.mean():.4f}")

    print(f"\n  delta_ghi_lag NULL rate:")
    nonzero = delta_nulls[delta_nulls > 0]
    if len(nonzero):
        print(f"    {nonzero.to_string()}")
    else:
        print(f"    max={delta_nulls.max():.4f}  mean={delta_nulls.mean():.4f}")


# ─── 6. DIAGNOSIS KESIMPULAN ──────────────────────────────────────────────────
def print_diagnosis():
    print("\n" + "=" * 60)
    print("6. DIAGNOSIS & REKOMENDASI")
    print("=" * 60)
    print("""
  Jika R²_persistence ≈ R²_LightGBM:
    → Data memang sulit diprediksi; LightGBM sudah optimal
    → Pertimbangkan: cloud regime separation, ensemble, atau LSTM+attention

  Jika R²_persistence << R²_LightGBM:
    → Model belajar sesuatu di luar persistence
    → R² rendah karena data inherently variabel (tropis konvektif)

  Jika R²_persistence > R²_LightGBM:
    → Ada bug di feature engineering (lag index off-by-one?)
    → Atau model underfitting karena early_stopping terlalu agresif

  Penanganan Stage 3 untuk 2025 (ptm/aeronet 100% NULL):
    → Opsi A: Buat model Stage 3b — sama dengan Stage 3 tapi
              drop ptm_* dan aeronet_* sebelum prediksi 2025
    → Opsi B: Gunakan Stage 2 sebagai model final untuk 2025
    → Opsi C: Retrain Stage 3 hanya pada data di mana ptm != NULL
              (train 2022-2024 termasuk val sebagai train)
  """)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    persistence_baseline()
    ghi_distribution()
    ghi_volatility()
    feature_importance()
    null_rate_analysis()
    print_diagnosis()