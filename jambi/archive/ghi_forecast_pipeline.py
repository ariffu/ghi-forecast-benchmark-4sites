"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      Solar GHI 6-Step Ahead Forecasting Pipeline — Jambi Station           ║
║      Dataset : jambi.jambi_sch.jambi_obs_combined (MotherDuck)             ║
║      Resolution : 10 menit  |  Horizon: 6 steps (60 menit ke depan)       ║
║      Lookback   : 18 steps  (180 menit = 3 jam)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Feature Phases:                                                            ║
║    Phase 1 — Solar radiation only (GHI, Kt, clearsky, solar geometry)      ║
║    Phase 2 — + Meteorologi (suhu, RH, tekanan, angin, hujan, awan)         ║
║    Phase 3 — + Aerosol & Cloud Properties (AOD, COT, CTT, CER)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Models:                                                                    ║
║    1. Persistence  — baseline statistik (naif)                              ║
║    2. Ridge        — regresi linear multi-output                            ║
║    3. LGBM         — gradient boosting (direct multi-step)                  ║
║    4. LSTM         — recurrent deep learning                                ║
║    5. Transformer  — attention-based (encoder-only)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Split: Train 2022-2023 | Val 2024 | Test 2025  (2021 dikecualikan)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cara pakai:
  python ghi_forecast_pipeline.py --token <MOTHERDUCK_TOKEN> --phases phase1 phase2 phase3

  Atau set env var MOTHERDUCK_TOKEN lalu:
  python ghi_forecast_pipeline.py --phases phase1

Install dependensi:
  pip install duckdb lightgbm scikit-learn torch pandas numpy matplotlib seaborn tqdm
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS & SETUP
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1. KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Data ──────────────────────────────────────────────
    "db":          "jambi",
    "table":       "jambi.jambi_sch.jambi_obs_combined",
    "min_sun_alt": 3.0,      # elevasi matahari minimum (derajat)
    "max_gap_min": 15,       # gap maksimum (menit) agar window tidak terputus

    # ── Window ────────────────────────────────────────────
    "window_in":  18,        # langkah lookback (18 × 10 mnt = 3 jam)
    "window_out":  6,        # langkah horizon  ( 6 × 10 mnt = 1 jam)

    # ── Split (by tahun) — 2021 dikecualikan ──────────────
    "train_years": [2022, 2023],
    "val_years":   [2024],
    "test_years":  [2025],

    # ── Hyperparams LGBM ──────────────────────────────────
    "lgbm": {
        "objective":        "regression",
        "metric":           "rmse",
        "n_estimators":     700,
        "learning_rate":    0.04,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "verbosity":        -1,
        "n_jobs":           -1,
    },

    # ── Hyperparams LSTM ──────────────────────────────────
    "lstm": {
        "hidden_size":  128,
        "num_layers":     2,
        "dropout":      0.25,
        "batch_size":   256,
        "epochs":        60,
        "lr":          1e-3,
        "patience":      12,
    },

    # ── Hyperparams Transformer ───────────────────────────
    "transformer": {
        "d_model":          64,
        "nhead":             4,
        "num_enc_layers":    3,
        "dim_feedforward": 256,
        "dropout":         0.1,
        "batch_size":      256,
        "epochs":           60,
        "lr":             1e-3,
        "patience":         12,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINISI FITUR (3 FASE)
# ══════════════════════════════════════════════════════════════════════════════
#
# Catatan: sun_altitude, sun_azimuth, ghi_clearsky pada window (t-17..t)
# digunakan sebagai konteks historis. Untuk fitur future deterministic
# (clearsky jam ke depan), tambahkan fungsi add_future_clearsky().

TARGET_COL = "ghi_consolidated"

# ── Phase 1: Solar only ───────────────────────────────────
FEAT_PHASE1 = [
    "ghi_consolidated",       # target (lag)
    "kt_consolidated",        # clearness index
    "ghi_clearsky",           # clearsky irradiance (model McClear/Ineichen)
    "dhi_consolidated",       # diffuse horizontal
    "dni_consolidated",       # direct normal
    "sun_altitude",           # elevasi matahari
    "sun_azimuth",            # azimut matahari
    "optical_air_mass",       # air mass
    # fitur waktu siklik
    "sin_hour", "cos_hour",
    "sin_doy",  "cos_doy",
]

# ── Phase 2: + Meteorologi ────────────────────────────────
FEAT_PHASE2 = FEAT_PHASE1 + [
    "temp_air_c",             # suhu udara
    "dewpoint_c",             # titik embun
    "rh_pct",                 # kelembapan relatif
    "vapour_pressure_hpa",    # tekanan uap
    "pressure_hpa",           # tekanan udara
    "wind_speed_ms",          # kecepatan angin
    "sin_wdir", "cos_wdir",   # arah angin (cyclic)
    "rainfall_mm",            # curah hujan
    "cloud_cover_oktas",      # tutupan awan (oktas)
    "cloud_cover_fraction",   # tutupan awan (fraksi)
]

# ── Phase 3: + Aerosol & Cloud Properties ─────────────────
FEAT_PHASE3 = FEAT_PHASE2 + [
    # AOD (Aerosol Optical Depth)
    "aod_best",               # AOD terbaik (gabungan ground/satelit)
    "aod_550nm",              # AOD pada 550nm
    "angstrom_exp_440_870",   # Angstrom exponent
    "precipitable_water_cm",  # water vapour column
    "beam_transmittance_500nm",
    "fine_mode_aot_proxy",    # fine mode aerosol
    "coarse_mode_aot_proxy",  # coarse mode aerosol
    # Cloud Properties (satelit)
    "cloud_optical_thickness",
    "cloud_top_temp_k",
    "cloud_top_height_m",
    "cloud_eff_radius_um",
    "sat_cloud_present",      # boolean → float
]

PHASE_FEATURES = {
    "phase1": FEAT_PHASE1,
    "phase2": FEAT_PHASE2,
    "phase3": FEAT_PHASE3,
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(token: str = None) -> pd.DataFrame:
    """
    Ambil data dari MotherDuck.
    Token bisa diberikan langsung atau via env var MOTHERDUCK_TOKEN.
    """
    if token:
        con = duckdb.connect(f"md:?motherduck_token={token}")
    else:
        con = duckdb.connect("md:")   # gunakan env MOTHERDUCK_TOKEN

    sql = f"""
    SELECT
        timestamp_wib,
        sun_altitude,
        sun_azimuth,
        optical_air_mass,
        -- Solar radiation
        ghi_consolidated,
        ghi_clearsky,
        kt_consolidated,
        dhi_consolidated,
        dni_consolidated,
        ghi_quality_flag,
        -- Meteorologi
        temp_air_c,
        dewpoint_c,
        rh_pct,
        vapour_pressure_hpa,
        pressure_hpa,
        wind_speed_ms,
        wind_dir_deg,
        rainfall_mm,
        cloud_cover_oktas,
        cloud_cover_fraction,
        -- AOD
        AOD_440nm,
        AOD_500nm,
        AOD_675nm,
        AOD_870nm,
        angstrom_exp_440_870,
        precipitable_water_cm,
        beam_transmittance_500nm,
        aod_best,
        aod_550nm,
        fine_mode_aot_proxy,
        coarse_mode_aot_proxy,
        -- Cloud Properties
        sat_cloud_present,
        cloud_optical_thickness,
        clot_std,
        cloud_top_temp_k,
        cloud_top_height_m,
        cloud_eff_radius_um
    FROM {CFG['table']}
    WHERE sun_altitude >= {CFG['min_sun_alt']}
      AND YEAR(timestamp_wib) >= 2022
    ORDER BY timestamp_wib
    """

    df = con.execute(sql).df()
    con.close()

    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    print(f"[DATA] {len(df):,} baris | "
          f"{df['timestamp_wib'].min().date()} → {df['timestamp_wib'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur waktu siklik dan derivatif."""
    df = df.copy()
    hour_frac = df["timestamp_wib"].dt.hour + df["timestamp_wib"].dt.minute / 60.0
    doy       = df["timestamp_wib"].dt.dayofyear

    df["sin_hour"] = np.sin(2 * np.pi * hour_frac / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour_frac / 24)
    df["sin_doy"]  = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"]  = np.cos(2 * np.pi * doy / 365.25)

    # Arah angin → cyclic
    wdir_rad       = np.deg2rad(df["wind_dir_deg"].fillna(0))
    df["sin_wdir"] = np.sin(wdir_rad)
    df["cos_wdir"] = np.cos(wdir_rad)

    # Boolean → float
    if "sat_cloud_present" in df.columns:
        df["sat_cloud_present"] = df["sat_cloud_present"].astype(float)

    df["year"] = df["timestamp_wib"].dt.year
    return df


def add_future_clearsky(df: pd.DataFrame, window_out: int = 6) -> pd.DataFrame:
    """
    Tambah fitur clearsky dan solar angle MENDATANG (deterministik).
    Ini valid karena nilai clearsky bisa dihitung dari ephemeris tanpa observasi.
    Berguna untuk LGBM/Ridge sebagai fitur target-time context.
    """
    df = df.copy().sort_values("timestamp_wib").reset_index(drop=True)
    for h in range(1, window_out + 1):
        df[f"ghi_clearsky_t{h}"]  = df["ghi_clearsky"].shift(-h)
        df[f"sun_altitude_t{h}"]  = df["sun_altitude"].shift(-h)
    return df


def mark_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tandai baris dengan gap waktu besar.
    Sliding window tidak boleh melintas segmen yang terputus.
    """
    df = df.sort_values("timestamp_wib").reset_index(drop=True)
    df["gap_min"] = df["timestamp_wib"].diff().dt.total_seconds().div(60).fillna(0)
    df["is_break"] = (df["gap_min"] > CFG["max_gap_min"]) | \
                     (df["timestamp_wib"].dt.date != df["timestamp_wib"].shift().dt.date)
    df["seg_id"]   = df["is_break"].cumsum()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. SLIDING WINDOW DATASET
# ══════════════════════════════════════════════════════════════════════════════
def build_windows(df: pd.DataFrame, feature_cols: list,
                  verbose: bool = True) -> dict:
    """
    Bangun dataset (X, y) dengan sliding window per segmen kontinu.

    Returns:
        X      : (N, W_IN, n_features)  — float32
        y      : (N, W_OUT)             — float32 (GHI target)
        split  : dict dengan kunci 'train', 'val', 'test'
                 masing-masing tuple (X_split, y_split)
    """
    W_IN  = CFG["window_in"]
    W_OUT = CFG["window_out"]
    total = W_IN + W_OUT

    # Pastikan semua kolom ada; kolom yang hilang diisi 0
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Kolom tidak ditemukan, diisi 0: {missing}")
        for c in missing:
            df[c] = 0.0

    # TARGET_COL (ghi_consolidated) sudah ada di feature_cols sebagai lag feature.
    # Jangan tambahkan lagi — kolom duplikat membuat seg[TARGET_COL] return DataFrame
    # bukan Series, sehingga vals_y.ndim == 2 dan np.stack(y_list) jadi 3D → crash.
    _extra    = [c for c in [TARGET_COL, "seg_id", "timestamp_wib", "year"]
                 if c not in feature_cols]
    work_cols = feature_cols + _extra
    feat_df   = df[work_cols].copy()

    # Imputasi per segmen: ffill → bfill → 0
    feat_df = (feat_df
               .groupby("seg_id", group_keys=False)
               .apply(lambda g: g.ffill().bfill().fillna(0)))

    X_list, y_list, ts_list, yr_list = [], [], [], []

    for _sid, seg in feat_df.groupby("seg_id"):
        if len(seg) < total:
            continue
        seg      = seg.reset_index(drop=True)
        vals_X   = seg[feature_cols].values.astype(np.float32)
        vals_y   = seg[TARGET_COL].values.astype(np.float32)
        timestamps = seg["timestamp_wib"].values
        years    = seg["year"].values

        for i in range(len(seg) - total + 1):
            X_list.append(vals_X[i : i + W_IN])
            y_list.append(vals_y[i + W_IN : i + W_IN + W_OUT])
            ts_list.append(timestamps[i + W_IN])    # t_0 prediksi
            yr_list.append(int(years[i + W_IN - 1]))

    X        = np.stack(X_list)    # (N, W_IN, F)
    y        = np.stack(y_list)    # (N, W_OUT)
    years_arr = np.array(yr_list)

    train_mask = np.isin(years_arr, CFG["train_years"])
    val_mask   = np.isin(years_arr, CFG["val_years"])
    test_mask  = np.isin(years_arr, CFG["test_years"])

    if verbose:
        print(f"  Windows: {len(X):,} total | "
              f"train={train_mask.sum():,}  val={val_mask.sum():,}  "
              f"test={test_mask.sum():,}")

    return {
        "X": X, "y": y,
        "timestamps": np.array(ts_list),
        "years":      years_arr,
        "train": (X[train_mask], y[train_mask]),
        "val":   (X[val_mask],   y[val_mask]),
        "test":  (X[test_mask],  y[test_mask]),
        "ts_test": np.array(ts_list)[test_mask],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALISASI
# ══════════════════════════════════════════════════════════════════════════════
def fit_scalers(data: dict):
    """Fit StandardScaler HANYA pada training data."""
    X_tr, y_tr = data["train"]
    N, T, F    = X_tr.shape

    x_sc = StandardScaler()
    x_sc.fit(X_tr.reshape(-1, F))

    y_sc = StandardScaler()
    y_sc.fit(y_tr)

    return x_sc, y_sc


def scale_split(split_data: tuple, x_sc, y_sc) -> tuple:
    X, y   = split_data
    N, T, F = X.shape
    X_s = x_sc.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
    y_s = y_sc.transform(y).astype(np.float32)
    return X_s, y_s


def apply_scalers(data: dict, x_sc, y_sc) -> dict:
    return {
        "train":   scale_split(data["train"], x_sc, y_sc),
        "val":     scale_split(data["val"],   x_sc, y_sc),
        "test":    scale_split(data["test"],  x_sc, y_sc),
        "ts_test": data["ts_test"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. METRIK EVALUASI
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    name: str = "") -> dict:
    """
    Hitung metrik per horizon dan rata-rata.
    Input dalam satuan asli (W/m²).
    """
    W_OUT   = y_true.shape[1]
    results = {"model": name}

    for h in range(W_OUT):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae  = mean_absolute_error(yt, yp)
        mbe  = float(np.mean(yp - yt))
        r2   = r2_score(yt, yp)
        mask = yt > 50  # MAPE hanya untuk GHI signifikan
        mape = 100 * np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) \
               if mask.sum() > 0 else np.nan
        nrmse = rmse / (yt.mean() + 1e-6) * 100  # nRMSE dalam %

        results[f"h{h+1}_rmse"]  = round(rmse,  2)
        results[f"h{h+1}_mae"]   = round(mae,   2)
        results[f"h{h+1}_mbe"]   = round(mbe,   2)
        results[f"h{h+1}_r2"]    = round(r2,    4)
        results[f"h{h+1}_mape"]  = round(mape,  2) if not np.isnan(mape) else np.nan
        results[f"h{h+1}_nrmse"] = round(nrmse, 2)

    results["avg_rmse"]  = round(np.mean([results[f"h{h+1}_rmse"]  for h in range(W_OUT)]), 2)
    results["avg_mae"]   = round(np.mean([results[f"h{h+1}_mae"]   for h in range(W_OUT)]), 2)
    results["avg_mbe"]   = round(np.mean([results[f"h{h+1}_mbe"]   for h in range(W_OUT)]), 2)
    results["avg_r2"]    = round(np.mean([results[f"h{h+1}_r2"]    for h in range(W_OUT)]), 4)
    results["avg_nrmse"] = round(np.mean([results[f"h{h+1}_nrmse"] for h in range(W_OUT)]), 2)
    results["avg_mape"]  = round(float(np.nanmean(
        [results[f"h{h+1}_mape"] for h in range(W_OUT)])), 2)

    return results


def print_metrics(m: dict):
    print(f"  avg_RMSE={m['avg_rmse']:.1f}  MAE={m['avg_mae']:.1f}  "
          f"nRMSE={m['avg_nrmse']:.1f}%  R²={m['avg_r2']:.3f}  "
          f"MBE={m['avg_mbe']:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 8.1 Persistence ───────────────────────────────────────────────────────────
class PersistenceModel:
    """
    Baseline naif: GHI(t+h) = GHI(t) untuk semua h.
    Bekerja pada ruang ternormalisasi — kolom pertama fitur = ghi_consolidated.
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        last_ghi = X[:, -1, 0]   # nilai GHI terakhir dalam window
        return np.tile(last_ghi.reshape(-1, 1), (1, CFG["window_out"]))


# ── 8.2 Persistence Clearsky (pers-kt) ───────────────────────────────────────
class PersistenceKtModel:
    """
    Persistence kt (clearness index): kt(t+h) = kt(t).
    Lebih baik dari persistence murni karena memperhitungkan posisi matahari.
    Membutuhkan ghi_clearsky di indeks fitur ke-2 dan kt di indeks ke-1.
    """
    def __init__(self, feat_cols: list):
        self.kt_idx  = feat_cols.index("kt_consolidated")  if "kt_consolidated"  in feat_cols else None
        self.cs_idx  = feat_cols.index("ghi_clearsky")     if "ghi_clearsky"     in feat_cols else None

    def predict(self, X: np.ndarray, X_orig: np.ndarray) -> np.ndarray:
        """X_orig: tidak ternormalisasi, untuk clearsky future."""
        if self.kt_idx is None or self.cs_idx is None:
            raise ValueError("kt_consolidated / ghi_clearsky tidak ada dalam fitur.")
        last_kt = X[:, -1, self.kt_idx]              # normalized kt
        # Untuk simplicity, gunakan clearsky t (yang sudah ternormalisasi)
        # Prediksi: kt_last × clearsky_window_mean (future clearsky tidak tersedia di window)
        cs_last = X[:, -1, self.cs_idx]
        pred_kt = np.tile(last_kt.reshape(-1, 1), (1, CFG["window_out"]))
        pred_cs = np.tile(cs_last.reshape(-1, 1), (1, CFG["window_out"]))
        return pred_kt * pred_cs   # (N, W_OUT) — tetap di ruang normalized


# ── 8.3 Ridge Regression ──────────────────────────────────────────────────────
class RidgeModel:
    """Ridge Regression multi-output menggunakan window yang di-flatten."""
    def __init__(self, alpha: float = 1.0):
        self.model = MultiOutputRegressor(Ridge(alpha=alpha), n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, T, F = X.shape
        self.model.fit(X.reshape(N, T * F), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, T, F = X.shape
        return self.model.predict(X.reshape(N, T * F))


# ── 8.4 LightGBM ──────────────────────────────────────────────────────────────
class LGBMForecast:
    """
    Direct multi-step LGBM: model terpisah untuk setiap horizon (h=1..6).
    Fitur = flatten window + statistik agregat window.
    """
    def __init__(self):
        self.models: list = []

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Ekstrak fitur dari window (N, T, F) → (N, D).
        Termasuk: flatten, mean, std, trend, nilai terakhir, recent mean.
        """
        N, T, F  = X.shape
        flat     = X.reshape(N, T * F)
        means    = X.mean(axis=1)              # (N, F)
        stds     = X.std(axis=1)               # (N, F)
        trend    = X[:, -1, :] - X[:, 0, :]   # (N, F) delta
        recent   = X[:, -3:, :].mean(axis=1)   # (N, F) mean 3 step terakhir
        last     = X[:, -1, :]                 # (N, F) step terakhir
        return np.concatenate([flat, means, stds, trend, recent, last], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        feats     = self._extract_features(X)
        feats_val = self._extract_features(X_val) if X_val is not None else None
        params    = CFG["lgbm"].copy()
        n_est     = params.pop("n_estimators")

        self.models = []
        for h in range(CFG["window_out"]):
            model = lgb.LGBMRegressor(n_estimators=n_est, **params)
            eval_set = [(feats_val, y_val[:, h])] if feats_val is not None else None
            model.fit(
                feats, y[:, h],
                eval_set=eval_set,
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.models.append(model)
            best = getattr(model, "best_iteration_", n_est)
            print(f"    h={h+1} → best_iter={best}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._extract_features(X)
        return np.column_stack([m.predict(feats) for m in self.models])

    def feature_importance(self, feat_names: list = None) -> pd.DataFrame:
        """Rata-rata feature importance semua model horizon."""
        imp_list = [m.feature_importances_ for m in self.models]
        imp_mean = np.mean(imp_list, axis=0)
        if feat_names is None:
            feat_names = [f"f{i}" for i in range(len(imp_mean))]
        return pd.DataFrame({"feature": feat_names, "importance": imp_mean}) \
                 .sort_values("importance", ascending=False)


# ── 8.5 LSTM ──────────────────────────────────────────────────────────────────
class LSTMForecast(nn.Module):
    """
    Stacked LSTM → Dense.
    Input : (B, T=18, F)
    Output: (B, 6)
    """
    def __init__(self, n_features: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.25, window_out: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.drop    = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, hidden_size // 2)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden_size // 2, window_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, T, H)
        out    = self.norm(out[:, -1])  # ambil step terakhir + LayerNorm
        out    = self.drop(out)
        out    = self.act(self.fc1(out))
        return self.fc2(out)            # (B, window_out)


# ── 8.6 Transformer ───────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerForecast(nn.Module):
    """
    Encoder-only Transformer dengan Pre-LayerNorm.
    Input : (B, T=18, F)
    Output: (B, 6)
    """
    def __init__(self, n_features: int,
                 d_model: int = 64, nhead: int = 4,
                 num_enc_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 window_in: int = 18, window_out: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=window_in + 8,
                                              dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN: lebih stabil
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)   # mean-pool over time
        self.head    = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, window_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)                        # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                           # (B, T, d_model)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        return self.head(x)                            # (B, window_out)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training loop umum untuk LSTM & Transformer ───────────────────────────────
def train_torch_model(model: nn.Module, data_sc: dict,
                      params: dict, label: str) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Training {label} on {device}")
    model = model.to(device)

    train_ds = WindowDataset(*data_sc["train"])
    val_ds   = WindowDataset(*data_sc["val"])
    train_ld = DataLoader(train_ds, batch_size=params["batch_size"],
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=params["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=False)

    opt       = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=params["epochs"], eta_min=params["lr"] * 0.01)
    criterion = nn.HuberLoss(delta=1.0)

    best_val  = float("inf")
    patience  = 0
    best_wts  = None

    for epoch in range(1, params["epochs"] + 1):
        # Train
        model.train()
        tr_losses = []
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        # Validate
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_ld:
                va_losses.append(criterion(model(xb.to(device)),
                                           yb.to(device)).item())

        tr_loss = np.mean(tr_losses)
        va_loss = np.mean(va_losses)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    ep {epoch:3d}/{params['epochs']} | "
                  f"train={tr_loss:.4f}  val={va_loss:.4f} | "
                  f"lr={sched.get_last_lr()[0]:.1e}")

        if va_loss < best_val:
            best_val  = va_loss
            best_wts  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience  = 0
        else:
            patience += 1
            if patience >= params["patience"]:
                print(f"    Early stop @ epoch {epoch}  best_val={best_val:.4f}")
                break

    model.load_state_dict(best_wts)
    return model.cpu()


def predict_torch(model: nn.Module, X: np.ndarray,
                  batch_size: int = 1024) -> np.ndarray:
    model.eval()
    dummy_y  = np.zeros((len(X), CFG["window_out"]), dtype=np.float32)
    loader   = DataLoader(WindowDataset(X, dummy_y),
                          batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb).numpy())
    return np.concatenate(preds)


# ══════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE PER FASE
# ══════════════════════════════════════════════════════════════════════════════
def run_phase(phase_name: str, df: pd.DataFrame,
              feature_cols: list, output_dir: Path) -> dict:
    """
    Jalankan full pipeline untuk satu fase fitur.
    Kembalikan dict berisi metrics, scalers, prediksi test.
    """
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  FASE : {phase_name.upper()}")
    print(f"  Fitur: {len(feature_cols)} kolom")
    print(sep)

    # 1. Build windows
    data_raw = build_windows(df, feature_cols)

    # 2. Scale
    x_sc, y_sc = fit_scalers(data_raw)
    data_sc    = apply_scalers(data_raw, x_sc, y_sc)
    N_FEAT     = len(feature_cols)
    W_IN, W_OUT = CFG["window_in"], CFG["window_out"]

    def inv_y(y_norm: np.ndarray) -> np.ndarray:
        return y_sc.inverse_transform(y_norm)

    y_test_orig = inv_y(data_sc["test"][1])
    X_test      = data_sc["test"][0]

    all_metrics  = []
    all_preds    = {}

    # ── Persistence ────────────────────────────────────────
    print(f"\n[1/5] Persistence...")
    pers   = PersistenceModel()
    p_sc   = pers.predict(X_test)
    p_orig = inv_y(p_sc)
    m = compute_metrics(y_test_orig, p_orig, f"Persistence_{phase_name}")
    all_metrics.append(m); all_preds["Persistence"] = p_orig; print_metrics(m)

    # ── Ridge ──────────────────────────────────────────────
    print(f"\n[2/5] Ridge Regression...")
    ridge  = RidgeModel(alpha=1.0)
    ridge.fit(*data_sc["train"])
    p_sc   = ridge.predict(X_test)
    p_orig = inv_y(p_sc)
    m = compute_metrics(y_test_orig, p_orig, f"Ridge_{phase_name}")
    all_metrics.append(m); all_preds["Ridge"] = p_orig; print_metrics(m)

    # ── LGBM ───────────────────────────────────────────────
    print(f"\n[3/5] LightGBM...")
    lgbm_m = LGBMForecast()
    lgbm_m.fit(*data_sc["train"],
               X_val=data_sc["val"][0],
               y_val=data_sc["val"][1])
    p_sc   = lgbm_m.predict(X_test)
    p_orig = inv_y(p_sc)
    m = compute_metrics(y_test_orig, p_orig, f"LGBM_{phase_name}")
    all_metrics.append(m); all_preds["LGBM"] = p_orig; print_metrics(m)

    # ── LSTM ───────────────────────────────────────────────
    print(f"\n[4/5] LSTM...")
    lstm_m = LSTMForecast(
        n_features=N_FEAT,
        hidden_size=CFG["lstm"]["hidden_size"],
        num_layers=CFG["lstm"]["num_layers"],
        dropout=CFG["lstm"]["dropout"],
        window_out=W_OUT,
    )
    lstm_m = train_torch_model(lstm_m, data_sc, CFG["lstm"], f"LSTM_{phase_name}")
    p_sc   = predict_torch(lstm_m, X_test)
    p_orig = inv_y(p_sc)
    m = compute_metrics(y_test_orig, p_orig, f"LSTM_{phase_name}")
    all_metrics.append(m); all_preds["LSTM"] = p_orig; print_metrics(m)

    # Simpan model LSTM
    torch.save(lstm_m.state_dict(),
               output_dir / f"lstm_{phase_name}.pt")

    # ── Transformer ────────────────────────────────────────
    print(f"\n[5/5] Transformer...")
    tf_m = TransformerForecast(
        n_features=N_FEAT,
        d_model=CFG["transformer"]["d_model"],
        nhead=CFG["transformer"]["nhead"],
        num_enc_layers=CFG["transformer"]["num_enc_layers"],
        dim_feedforward=CFG["transformer"]["dim_feedforward"],
        dropout=CFG["transformer"]["dropout"],
        window_in=W_IN, window_out=W_OUT,
    )
    tf_m   = train_torch_model(tf_m, data_sc, CFG["transformer"],
                                f"Transformer_{phase_name}")
    p_sc   = predict_torch(tf_m, X_test)
    p_orig = inv_y(p_sc)
    m = compute_metrics(y_test_orig, p_orig, f"Transformer_{phase_name}")
    all_metrics.append(m); all_preds["Transformer"] = p_orig; print_metrics(m)

    torch.save(tf_m.state_dict(),
               output_dir / f"transformer_{phase_name}.pt")

    return {
        "metrics":      all_metrics,
        "scalers":      (x_sc, y_sc),
        "y_test_orig":  y_test_orig,
        "preds":        all_preds,
        "ts_test":      data_raw["ts_test"],
        "feature_cols": feature_cols,
        "lgbm_model":   lgbm_m,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════
MODEL_COLORS = {
    "Persistence":  "#aaaaaa",
    "Ridge":        "#4daf4a",
    "LGBM":         "#ff7f00",
    "LSTM":         "#377eb8",
    "Transformer":  "#e41a1c",
}

def plot_horizon_rmse(df_res: pd.DataFrame, save_path: Path = None):
    """RMSE vs horizon per model untuk setiap fase."""
    W_OUT    = CFG["window_out"]
    horizons = [(h + 1) * 10 for h in range(W_OUT)]
    phases   = ["phase1", "phase2", "phase3"]
    labels   = ["Phase 1: Solar Only",
                "Phase 2: + Meteorologi",
                "Phase 3: + Aerosol & Cloud"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, phase, label in zip(axes, phases, labels):
        sub = df_res[df_res["model"].str.contains(phase, na=False)]
        for _, row in sub.iterrows():
            mname  = row["model"].replace(f"_{phase}", "")
            rmse_v = [row[f"h{h+1}_rmse"] for h in range(W_OUT)]
            color  = MODEL_COLORS.get(mname, "gray")
            ax.plot(horizons, rmse_v, marker="o", label=mname,
                    color=color, linewidth=2, markersize=5)
        ax.set_xlabel("Horizon (menit ke depan)")
        ax.set_ylabel("RMSE (W/m²)")
        ax.set_title(label, fontsize=10)
        ax.set_xticks(horizons)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("RMSE per Horizon — GHI 6-step Forecasting\nTest: 2025",
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_summary_heatmap(df_res: pd.DataFrame, metric: str = "avg_rmse",
                          save_path: Path = None):
    """Heatmap avg metric: model × fase."""
    df = df_res.copy()
    df["base_model"] = df["model"].str.replace(r"_phase\d", "", regex=True)
    df["phase"]      = df["model"].str.extract(r"(phase\d)")

    order = ["Persistence", "Ridge", "LGBM", "LSTM", "Transformer"]
    pivot = df.pivot(index="base_model", columns="phase", values=metric)
    pivot = pivot.reindex(index=[m for m in order if m in pivot.index])
    pivot.columns = ["Phase 1", "Phase 2", "Phase 3"]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f",
                cmap="RdYlGn_r" if "rmse" in metric or "mae" in metric else "RdYlGn",
                ax=ax, linewidths=0.5, cbar_kws={"label": metric})
    ax.set_title(f"{metric.upper()} — Test 2025\n(lebih rendah = lebih baik)",
                 fontsize=11)
    ax.set_xlabel("Feature Phase")
    ax.set_ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_sample_forecast(y_true: np.ndarray, preds_dict: dict,
                          ts_test: np.ndarray, phase_name: str,
                          n_days: int = 2, save_path: Path = None):
    """Plot contoh prediksi vs observasi untuk beberapa hari."""
    # Ambil n_days pertama dari test set
    ts_pd    = pd.to_datetime(ts_test)
    days     = pd.Series(ts_pd.date).unique()[:n_days]
    mask     = np.isin(pd.Series(ts_pd.date), days)
    y_plot   = y_true[mask]
    ts_plot  = ts_pd[mask]

    fig, axes = plt.subplots(n_days, 1, figsize=(14, 4 * n_days))
    if n_days == 1:
        axes = [axes]

    for ax, day in zip(axes, days):
        day_mask = pd.Series(ts_pd.date).values == day
        idx      = np.where(mask)[0][pd.Series(ts_pd[mask].date).values == day]
        if len(idx) == 0:
            continue
        ax.plot(ts_pd[idx], y_true[idx, 0], "k-", label="Observasi (h+1)",
                linewidth=1.5)
        for mname, color in MODEL_COLORS.items():
            if mname in preds_dict:
                ax.plot(ts_pd[idx], preds_dict[mname][idx, 0],
                        "--", color=color, label=mname, alpha=0.8)
        ax.set_title(f"{day} — {phase_name} (h=+10min)")
        ax.set_ylabel("GHI (W/m²)")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Contoh Prediksi GHI — {phase_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="GHI 6-step Forecasting Pipeline — Jambi")
    parser.add_argument("--token",        type=str, default=None,
                        help="MotherDuck token (atau set env MOTHERDUCK_TOKEN)")
    parser.add_argument("--parquet_path", type=str, default=None,
                        help="Path ke parquet hasil ghi_preprocess.py "
                             "(direkomendasikan — lebih cepat, data sudah bersih)")
    parser.add_argument("--phases",       nargs="+",
                        default=["phase1", "phase2", "phase3"],
                        choices=["phase1", "phase2", "phase3"],
                        help="Fase fitur yang dijalankan")
    parser.add_argument("--output_dir",   type=str, default="./ghi_output",
                        help="Direktori output hasil & model")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────
    if args.parquet_path:
        # ── Mode parquet: data sudah bersih dari ghi_preprocess.py ──
        print(f"\n[STEP 1] Memuat dari parquet: {args.parquet_path}")
        df = pd.read_parquet(args.parquet_path)
        df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
        # seg_id & fitur waktu sudah ada dari preprocessing
        if "seg_id" not in df.columns:
            df = mark_continuity(df)
        if "year" not in df.columns:
            df["year"] = df["timestamp_wib"].dt.year

        # Gunakan kolom _clean dari preprocessing
        from ghi_preprocess import (CLEAN_FEAT_PHASE1, CLEAN_FEAT_PHASE2,
                                     CLEAN_FEAT_PHASE3)
        active_phases = {
            "phase1": CLEAN_FEAT_PHASE1,
            "phase2": CLEAN_FEAT_PHASE2,
            "phase3": CLEAN_FEAT_PHASE3,
        }
    else:
        # ── Mode MotherDuck langsung (hanya 2022–2025) ──────
        print("\n[STEP 1] Memuat data dari MotherDuck (2022–2025)...")
        df = load_data(token=args.token)
        df = add_time_features(df)
        df = mark_continuity(df)
        active_phases = PHASE_FEATURES

    print(f"  Rows: {len(df):,} | Segmen: {df['seg_id'].nunique()} | "
          f"Tahun: {sorted(df['year'].unique())}")

    # ── Jalankan setiap fase ───────────────────────────────
    phase_results = {}
    for phase in args.phases:
        feat_cols_raw = active_phases[phase]
        feat_cols     = [c for c in feat_cols_raw if c in df.columns]
        missing_count = len(feat_cols_raw) - len(feat_cols)
        if missing_count > 0:
            print(f"  [INFO] {phase}: {missing_count} kolom tidak ditemukan, dilewati")
        phase_results[phase] = run_phase(phase, df, feat_cols, out_dir)

    # ── Kompilasi metrik ────────────────────────────────────
    print("\n[STEP FINAL] Kompilasi hasil...")
    rows = []
    for pname, res in phase_results.items():
        for m in res["metrics"]:
            rows.append(m)
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_dir / "results_all.csv", index=False)

    # ── Cetak ringkasan ─────────────────────────────────────
    summary_cols = [
        "model",
        "avg_rmse", "avg_mae", "avg_mbe", "avg_r2", "avg_nrmse", "avg_mape",
        "h1_rmse", "h2_rmse", "h3_rmse", "h6_rmse",
    ]
    print("\n" + "═" * 90)
    print("  RINGKASAN METRIK — Test Set 2025")
    print("═" * 90)
    print(df_res[summary_cols].sort_values(["model"]).to_string(index=False))

    # Best model per fase
    print("\n  ── Best model per fase (avg RMSE) ──")
    for phase in args.phases:
        sub  = df_res[df_res["model"].str.contains(phase)]
        best = sub.loc[sub["avg_rmse"].idxmin()]
        print(f"  {phase}: {best['model']}  "
              f"RMSE={best['avg_rmse']:.1f}  R²={best['avg_r2']:.3f}")

    # ── Visualisasi ─────────────────────────────────────────
    print("\n[VIZ] Membuat grafik...")
    plot_horizon_rmse(df_res, save_path=out_dir / "horizon_rmse.png")
    for metric in ["avg_rmse", "avg_r2"]:
        plot_summary_heatmap(df_res, metric=metric,
                             save_path=out_dir / f"heatmap_{metric}.png")
    for phase in args.phases:
        if phase in phase_results:
            res = phase_results[phase]
            plot_sample_forecast(
                res["y_test_orig"], res["preds"], res["ts_test"],
                phase_name=phase,
                save_path=out_dir / f"sample_forecast_{phase}.png",
            )

    print(f"\n✓ Selesai. Output tersimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
