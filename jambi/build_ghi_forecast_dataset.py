#!/usr/bin/env python3
"""
build_ghi_forecast_dataset.py
==============================
Membangun dataset ML untuk prediksi GHI 1 jam ke depan (t+1..t+6 × 10 menit)
dari jambi.solar_radiation_valid di MotherDuck.

Desain sliding window:
  Lookback : 18 lag × 10 menit = 3 jam ke belakang
  Horizon  : 6 step × 10 menit = 1 jam ke depan
  Total window: 24 bin (240 menit) harus kontinu, tanpa gap

Anchor validity:
  - ghi_source = 'asrs'   → target training hanya dari pyranometer
  - sun_altitude ≥ 10°    → di anchor DAN di t+6 (masih siang)
  - Semua 24 bin kontinu  → selisih timestamp persis 10 menit tiap langkah

Output (3 stage Parquet, masing-masing train/val/test):
  Stage 1 — Radiasi saja: ghi lag/delta, kt, sun position, clearsky
  Stage 2 — + Meteorologi: temp_air, rh, pressure, ws, rain, cloud_oktas
  Stage 3 — + Cloud/Aerosol: ptm_cloud_opacity lag, aeronet, aeronet_smoke

Split temporal:
  Train : 2022-01-01 — 2023-12-31
  Val   : 2024-01-01 — 2024-12-31
  Test  : 2025-01-01 — 2025-12-31
"""

import os
import sys
import time
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

# ─── KONFIGURASI ───────────────────────────────────────────────────────────────
MOTHERDUCK_TOKEN = os.environ.get("motherduck_token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFyaWZmdUBnbWFpbC5jb20iLCJtZFJlZ2lvbiI6ImF3cy11cy1lYXN0LTEiLCJzZXNzaW9uIjoiYXJpZmZ1LmdtYWlsLmNvbSIsInBhdCI6IktOejdCOXZjYjZKNTdUOFcyZmZhLUZscUUtSXg1YVlEYy1FZnlPVktkV0EiLCJ1c2VySWQiOiJkMzQ0MTFkMS04MGRmLTQyNTQtYWRmNy00YjZlNThmNTMzZDkiLCJpc3MiOiJtZF9wYXQiLCJyZWFkT25seSI6ZmFsc2UsInRva2VuVHlwZSI6InJlYWRfd3JpdGUiLCJpYXQiOjE3ODA3MTI3MDV9.CfUo8h9ZTvaCOpRO7e2qcGG_Dc9-0cByDd5LOvXl1fE")

# Sesuaikan OUTPUT_DIR ke folder workspace-mu jika perlu
OUTPUT_DIR = Path(__file__).parent / "dataset"

LOOKBACK     = 18       # jumlah lag bin (× 10 menit = 180 menit = 3 jam)
HORIZON      = 6        # jumlah horizon bin (× 10 menit = 60 menit = 1 jam)
SUN_ALT_MIN  = 10.0     # derajat; lebih ketat dari 5° untuk menghindari noise kt
ANCHOR_SOURCE = "asrs"  # training target hanya dari pyranometer

TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_END   = pd.Timestamp("2024-12-31 23:59:59")


# ─── 1. LOAD DATA DARI MOTHERDUCK ──────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if MOTHERDUCK_TOKEN:
        conn_str = f"md:jambi?motherduck_token={MOTHERDUCK_TOKEN}"
    else:
        conn_str = "md:jambi"  # gunakan token dari env MOTHERDUCK_TOKEN

    print("Connecting to MotherDuck (jambi)...")
    conn = duckdb.connect(conn_str)

    query = """
    SELECT
        timestamp_wib,
        ghi_final,
        ghi_source,
        kt,
        sun_altitude,
        sun_azimuth,
        ghi_clearsky,
        has_gap_before,
        n_minutes,
        data_flag_max,
        -- Meteorologi
        temp_air,
        rh,
        pressure,
        ws,
        ws_max,
        wd,
        rain,
        cloud_oktas,
        -- PTM model (cloud/clearsky proxy)
        ptm_cloud_opacity,
        ptm_precipitable_water,
        ptm_clearsky_ghi,
        ptm_ghi,
        -- AERONET (daily aerosol)
        aeronet_aod500,
        aeronet_ae,
        aeronet_pwv_cm,
        aeronet_aod500_std,
        CAST(aeronet_smoke_flag AS INTEGER) AS aeronet_smoke_flag,
        aeronet_n_obs
    FROM jambi.solar_radiation_valid
    WHERE timestamp_wib >= '2022-01-01'
      AND timestamp_wib <  '2026-01-01'
    ORDER BY timestamp_wib
    """

    t0 = time.time()
    df = conn.execute(query).df()
    conn.close()

    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    df["has_gap_before"] = df["has_gap_before"].fillna(False).astype(bool)

    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {elapsed:.1f}s")
    print(f"  Range: {df['timestamp_wib'].min()} → {df['timestamp_wib'].max()}")
    print(f"  ghi_source counts:\n{df['ghi_source'].value_counts(dropna=False).to_string()}")
    return df


# ─── 2. TAMBAH FITUR TURUNAN (SEBELUM SLIDING WINDOW) ─────────────────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah fitur siklus temporal dan rate-of-change GHI.
    Semua dikerjakan di-full series sebelum windowing agar lag δ-GHI valid.
    """
    # ── Siklus bulan dan jam (untuk seasonality) ─────────────────────────
    df["month"]     = df["timestamp_wib"].dt.month
    df["hour"]      = df["timestamp_wib"].dt.hour
    df["minute"]    = df["timestamp_wib"].dt.minute
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # jam + menit sebagai desimal (misal 10:30 → 10.5)
    df["hour_frac"] = df["hour"] + df["minute"] / 60
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_frac"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_frac"] / 24)

    # ── Rate of change (δGHI dan δkt) ────────────────────────────────────
    # Hanya valid jika tidak ada gap antar baris berurutan
    df["delta_ghi"] = df["ghi_final"].diff()
    df["delta_kt"]  = df["kt"].diff()

    # Null-kan jika ada gap sebelum baris ini
    gap_mask = df["has_gap_before"] == True
    df.loc[gap_mask, "delta_ghi"] = np.nan
    df.loc[gap_mask, "delta_kt"]  = np.nan

    # ── Clearness index normalisasi (opsional diagnostik) ─────────────────
    # kt sudah ada di solar_radiation_valid; tidak perlu dihitung ulang

    return df


# ─── 3. SLIDING WINDOW ────────────────────────────────────────────────────────
TEN_MINUTES = np.timedelta64(10, "m")


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat satu baris per anchor yang valid.

    Setiap baris mengandung:
      - ghi_lag{1..18}, kt_lag{1..18}, delta_ghi_lag{1..18}, delta_kt_lag{1..18}
      - sun_alt_lag{1..18}, cloud_op_lag{1..18}
      - sun_alt_h{1..6}, sun_az_h{1..6}, clearsky_ghi_h{1..6}  (future-known)
      - fitur statis anchor: month_sin/cos, hour_sin/cos, sun_alt/az, kt, ghi
      - fitur meteorologi anchor: temp_air, rh, pressure, ws, ws_max, wd, rain, cloud_oktas
      - fitur cloud/aerosol anchor: ptm_cloud_opacity, ptm_pw, aeronet_*
      - target: ghi_h{1..6}, kt_h{1..6}
    """
    print("\nBuilding sliding windows...")
    t0 = time.time()

    df = df.reset_index(drop=True)
    n  = len(df)

    # ── Ekstrak array numpy untuk akses cepat ──────────────────────────────
    ts           = df["timestamp_wib"].values
    ghi          = df["ghi_final"].values.astype(float)
    ghi_src      = df["ghi_source"].values
    kt           = df["kt"].values.astype(float)
    sun_alt      = df["sun_altitude"].values.astype(float)
    sun_az       = df["sun_azimuth"].values.astype(float)
    clearsky     = df["ghi_clearsky"].values.astype(float)
    has_gap      = df["has_gap_before"].values.astype(bool)
    delta_ghi    = df["delta_ghi"].values.astype(float)
    delta_kt     = df["delta_kt"].values.astype(float)

    # Meteo
    temp_air     = df["temp_air"].values.astype(float)
    rh           = df["rh"].values.astype(float)
    pressure     = df["pressure"].values.astype(float)
    ws           = df["ws"].values.astype(float)
    ws_max       = df["ws_max"].values.astype(float)
    wd           = df["wd"].values.astype(float)
    rain         = df["rain"].values.astype(float)
    cloud_oktas  = df["cloud_oktas"].values.astype(float)

    # PTM + AERONET
    ptm_cloud_op = df["ptm_cloud_opacity"].values.astype(float)
    ptm_pw       = df["ptm_precipitable_water"].values.astype(float)
    ae_aod500    = df["aeronet_aod500"].values.astype(float)
    ae_ae        = df["aeronet_ae"].values.astype(float)
    ae_pwv       = df["aeronet_pwv_cm"].values.astype(float)
    ae_aod_std   = df["aeronet_aod500_std"].values.astype(float)
    ae_smoke     = df["aeronet_smoke_flag"].values.astype(float)
    ae_nobs      = df["aeronet_n_obs"].values.astype(float)

    # Cyclical
    month_sin    = df["month_sin"].values.astype(float)
    month_cos    = df["month_cos"].values.astype(float)
    hour_sin     = df["hour_sin"].values.astype(float)
    hour_cos     = df["hour_cos"].values.astype(float)

    rows = []
    n_skip_src = n_skip_alt = n_skip_gap = 0

    for i in range(LOOKBACK, n - HORIZON):

        # ── Cek 1: Anchor validity ─────────────────────────────────────────
        if ghi_src[i] != ANCHOR_SOURCE:
            n_skip_src += 1
            continue
        if sun_alt[i] < SUN_ALT_MIN:
            n_skip_alt += 1
            continue
        if np.isnan(ghi[i]):
            continue

        # ── Cek 2: Horizon t+6 masih siang ────────────────────────────────
        if sun_alt[i + HORIZON] < SUN_ALT_MIN:
            n_skip_alt += 1
            continue

        # ── Cek 3: Kontinuitas window (timestamp persis 10 menit) ─────────
        # Lag window: i-LOOKBACK .. i  (19 titik, 18 selisih)
        lag_ts   = ts[i - LOOKBACK : i + 1]
        lag_diff = np.diff(lag_ts)
        if np.any(lag_diff != TEN_MINUTES):
            n_skip_gap += 1
            continue

        # Horizon window: i .. i+HORIZON  (7 titik, 6 selisih)
        hor_ts   = ts[i : i + HORIZON + 1]
        hor_diff = np.diff(hor_ts)
        if np.any(hor_diff != TEN_MINUTES):
            n_skip_gap += 1
            continue

        # ── Bangun baris ───────────────────────────────────────────────────
        row: dict = {
            "anchor_ts"     : ts[i],
            "anchor_ghi"    : ghi[i],
            "anchor_kt"     : kt[i],
            "anchor_sun_alt": sun_alt[i],
            "anchor_sun_az" : sun_az[i],
            # Cyclical
            "month_sin"     : month_sin[i],
            "month_cos"     : month_cos[i],
            "hour_sin"      : hour_sin[i],
            "hour_cos"      : hour_cos[i],
            # Meteo (anchor)
            "temp_air"      : temp_air[i],
            "rh"            : rh[i],
            "pressure"      : pressure[i],
            "ws"            : ws[i],
            "ws_max"        : ws_max[i],
            "wd"            : wd[i],
            "rain"          : rain[i],
            "cloud_oktas"   : cloud_oktas[i],
            # PTM (anchor)
            "ptm_cloud_opacity"      : ptm_cloud_op[i],
            "ptm_precipitable_water" : ptm_pw[i],
            # AERONET (daily)
            "aeronet_aod500"     : ae_aod500[i],
            "aeronet_ae"         : ae_ae[i],
            "aeronet_pwv_cm"     : ae_pwv[i],
            "aeronet_aod500_std" : ae_aod_std[i],
            "aeronet_smoke_flag" : ae_smoke[i],
            "aeronet_n_obs"      : ae_nobs[i],
        }

        # Lag features (k=1 → paling dekat ke anchor)
        for k in range(1, LOOKBACK + 1):
            j = i - k
            row[f"ghi_lag{k}"]       = ghi[j]
            row[f"kt_lag{k}"]        = kt[j]
            row[f"delta_ghi_lag{k}"] = delta_ghi[j + 1]  # δghi[j+1] = ghi[j+1]-ghi[j]
            row[f"delta_kt_lag{k}"]  = delta_kt[j + 1]
            row[f"sun_alt_lag{k}"]   = sun_alt[j]
            row[f"cloud_op_lag{k}"]  = ptm_cloud_op[j]   # hourly, berubah tiap 6 bin

        # Horizon features (future-known: sun position + clearsky)
        for h in range(1, HORIZON + 1):
            j = i + h
            row[f"sun_alt_h{h}"]     = sun_alt[j]
            row[f"sun_az_h{h}"]      = sun_az[j]
            row[f"clearsky_ghi_h{h}"] = clearsky[j]

        # Target (GHI dan kt di tiap horizon)
        for h in range(1, HORIZON + 1):
            j = i + h
            row[f"ghi_h{h}"] = ghi[j]
            row[f"kt_h{h}"]  = kt[j]

        rows.append(row)

    elapsed = time.time() - t0
    result  = pd.DataFrame(rows)

    print(f"  Window loop: {elapsed:.1f}s")
    print(f"  Valid windows : {len(result):,}")
    print(f"  Skipped (src) : {n_skip_src:,}")
    print(f"  Skipped (alt) : {n_skip_alt:,}")
    print(f"  Skipped (gap) : {n_skip_gap:,}")

    return result


# ─── 4. TEMPORAL SPLIT ─────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame):
    ts    = pd.to_datetime(df["anchor_ts"])
    train = df[ts <= TRAIN_END].copy()
    val   = df[(ts > TRAIN_END) & (ts <= VAL_END)].copy()
    test  = df[ts > VAL_END].copy()

    print(f"\nTemporal split:")
    print(f"  Train 2022-2023 : {len(train):,}")
    print(f"  Val   2024      : {len(val):,}")
    print(f"  Test  2025      : {len(test):,}")
    return train, val, test


# ─── 5. DEFINISI KOLOM PER STAGE ──────────────────────────────────────────────
def _lag_cols(prefix: str, n: int = LOOKBACK):
    return [f"{prefix}{k}" for k in range(1, n + 1)]

def _hor_cols(prefix: str, n: int = HORIZON):
    return [f"{prefix}{h}" for h in range(1, n + 1)]

TARGET_COLS = _lag_cols("ghi_h", HORIZON) + _lag_cols("kt_h", HORIZON)
# Alias yang lebih jelas untuk target
TARGET_COLS = (
    [f"ghi_h{h}" for h in range(1, HORIZON + 1)]
    + [f"kt_h{h}"  for h in range(1, HORIZON + 1)]
)

META_COLS = ["anchor_ts", "anchor_ghi", "anchor_sun_alt", "anchor_sun_az"]

# Stage 1: Radiasi + sun position + cyclical
STAGE1_FEATURE_COLS = (
    META_COLS
    + ["anchor_kt", "month_sin", "month_cos", "hour_sin", "hour_cos"]
    + _lag_cols("ghi_lag")
    + _lag_cols("kt_lag")
    + _lag_cols("delta_ghi_lag")
    + _lag_cols("delta_kt_lag")
    + _lag_cols("sun_alt_lag")
    + [f"sun_alt_h{h}"     for h in range(1, HORIZON + 1)]
    + [f"sun_az_h{h}"      for h in range(1, HORIZON + 1)]
    + [f"clearsky_ghi_h{h}" for h in range(1, HORIZON + 1)]
    + TARGET_COLS
)

# Stage 2: + Meteorologi
STAGE2_FEATURE_COLS = STAGE1_FEATURE_COLS + [
    "temp_air", "rh", "pressure", "ws", "ws_max", "wd", "rain", "cloud_oktas",
]

# Stage 3: + Cloud properties (ptm lag) + AERONET aerosol
STAGE3_FEATURE_COLS = STAGE2_FEATURE_COLS + (
    ["ptm_cloud_opacity", "ptm_precipitable_water"]
    + _lag_cols("cloud_op_lag")
    + ["aeronet_aod500", "aeronet_ae", "aeronet_pwv_cm",
       "aeronet_aod500_std", "aeronet_smoke_flag", "aeronet_n_obs"]
)


# ─── 6. DIAGNOSTIK ────────────────────────────────────────────────────────────
def print_diagnostics(train, val, test):
    print("\n=== Diagnostics ===")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        ghi_h1 = df["ghi_h1"]
        n_rows = len(df)
        print(f"\n{name} ({n_rows:,} samples):")
        print(f"  GHI_h1 → mean={ghi_h1.mean():.1f}  std={ghi_h1.std():.1f}  "
              f"min={ghi_h1.min():.1f}  max={ghi_h1.max():.1f}  "
              f"null={ghi_h1.isna().mean():.3f}")

        # NULL rate per feature group
        target_null = df[[f"ghi_h{h}" for h in range(1, HORIZON + 1)]].isna().mean().mean()
        ghi_lag_null = df["ghi_lag1"].isna().mean()
        temp_null = df["temp_air"].isna().mean() if "temp_air" in df.columns else float("nan")
        ptm_null  = df["ptm_cloud_opacity"].isna().mean() if "ptm_cloud_opacity" in df.columns else float("nan")
        ae_null   = df["aeronet_aod500"].isna().mean() if "aeronet_aod500" in df.columns else float("nan")

        print(f"  NULL target     : {target_null:.3f}")
        print(f"  NULL ghi_lag1   : {ghi_lag_null:.3f}")
        print(f"  NULL temp_air   : {temp_null:.3f}")
        print(f"  NULL ptm_cloud  : {ptm_null:.3f}")
        print(f"  NULL aeronet    : {ae_null:.3f}")

        # Distribusi per tahun
        ts = pd.to_datetime(df["anchor_ts"])
        yr = ts.dt.year.value_counts().sort_index()
        print(f"  Per year: {yr.to_dict()}")


# ─── 7. SIMPAN PARQUET ─────────────────────────────────────────────────────────
def save_stage(train, val, test, stage: int, cols: list):
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        available = [c for c in cols if c in split_df.columns]
        out_path  = OUTPUT_DIR / f"jambi_ghi_s{stage}_{split_name}.parquet"
        split_df[available].to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    {out_path.name}  ({len(split_df):,} rows × {len(available)} cols, {size_mb:.1f} MB)")


# ─── 8. RINGKASAN FITUR ───────────────────────────────────────────────────────
def print_feature_summary():
    print("\n=== Feature Groups (Stage 3) ===")
    groups = {
        "Anchor GHI+kt"      : ["anchor_ghi", "anchor_kt"],
        "Sun position"        : ["anchor_sun_alt", "anchor_sun_az"],
        "Cyclical"            : ["month_sin", "month_cos", "hour_sin", "hour_cos"],
        f"GHI lag ×{LOOKBACK}": _lag_cols("ghi_lag"),
        f"kt lag ×{LOOKBACK}" : _lag_cols("kt_lag"),
        f"δGHI lag ×{LOOKBACK}": _lag_cols("delta_ghi_lag"),
        f"δkt lag ×{LOOKBACK}": _lag_cols("delta_kt_lag"),
        f"sun_alt lag ×{LOOKBACK}": _lag_cols("sun_alt_lag"),
        f"Cloud opacity lag ×{LOOKBACK}": _lag_cols("cloud_op_lag"),
        f"Future sun/clearsky ×{HORIZON}": (
            _hor_cols("sun_alt_h") + _hor_cols("sun_az_h") + _hor_cols("clearsky_ghi_h")
        ),
        "Meteorologi"         : ["temp_air", "rh", "pressure", "ws", "ws_max", "wd", "rain", "cloud_oktas"],
        "PTM model"           : ["ptm_cloud_opacity", "ptm_precipitable_water"],
        "AERONET aerosol"     : ["aeronet_aod500", "aeronet_ae", "aeronet_pwv_cm",
                                  "aeronet_aod500_std", "aeronet_smoke_flag"],
        f"Target ×{HORIZON}"  : TARGET_COLS,
    }
    total_feat = 0
    for group, cols in groups.items():
        print(f"  {group:35s}: {len(cols)} cols")
        total_feat += len(cols)
    print(f"  {'TOTAL':35s}: {total_feat} cols")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("BUILD GHI FORECAST DATASET — Stasiun Jambi")
    print(f"  Lookback : {LOOKBACK} × 10 menit = {LOOKBACK * 10} menit ({LOOKBACK * 10 // 60} jam)")
    print(f"  Horizon  : {HORIZON}  × 10 menit = {HORIZON * 10} menit ({HORIZON * 10 // 60} jam)")
    print(f"  Sun alt  : ≥ {SUN_ALT_MIN}° di anchor + t+{HORIZON}")
    print(f"  Anchor src: '{ANCHOR_SOURCE}' only (pyranometer)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_data()

    # 2. Derived features
    print("\nAdding derived features...")
    df = add_derived_features(df)

    # 3. Sliding window
    windows = build_windows(df)

    if len(windows) == 0:
        print("ERROR: Tidak ada window valid! Periksa filter dan data coverage.")
        sys.exit(1)

    # 4. Temporal split
    train, val, test = temporal_split(windows)

    # 5. Diagnostics
    print_diagnostics(train, val, test)

    # 6. Feature summary
    print_feature_summary()

    # 7. Save parquet
    print("\n=== Saving Parquet ===")
    stages = [
        (1, STAGE1_FEATURE_COLS),
        (2, STAGE2_FEATURE_COLS),
        (3, STAGE3_FEATURE_COLS),
    ]
    for stage, cols in stages:
        print(f"\nStage {stage}:")
        save_stage(train, val, test, stage, cols)

    print("\n✅ Dataset build selesai.")
    print(f"   Output: {OUTPUT_DIR.resolve()}")
    print(f"   Files: 9 parquet (3 stage × 3 split)")


if __name__ == "__main__":
    main()