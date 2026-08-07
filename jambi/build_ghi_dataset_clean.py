#!/usr/bin/env python3
"""
build_ghi_dataset_clean.py
===========================
Dataset builder versi bersih — tanpa data leakage.

PERUBAHAN dari build_ghi_forecast_dataset.py:
  1. ptm_cloud_opacity diganti dengan ptm_cloud_opacity_PREV (jam sebelumnya)
     → Bukti leakage: corr(current_hour, ghi) naik monoton di dalam jam
       (korealsi -0.50 di :00 naik ke -0.56 di :50 = tanda reanalysis/hindcast)
     → Corr(prev_hour, ghi) = -0.539 ≈ sama dengan current = aman digunakan
  2. Hapus ptm_precipitable_water (dari jam yang sama, juga reanalysis)
  3. Hapus cloud_op_lag (lag ptm_cloud_opacity = leaky untuk lag dalam jam yg sama)
  4. Feature pruning: hapus fitur importance < 0.3% (sun_alt_lag, kt_lag7+, dll)
  5. Perpendek lookback fitur lambat: ghi_lag1..12, kt_lag1..6, delta_*_lag1..6

Fitur AERONET daily dipertahankan (kontribusi nol tapi tidak leaky secara material).

Output:
  dataset_clean/jambi_clean_s{1,2,3}_{train,val,test}.parquet
"""

import os
import time
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

MOTHERDUCK_TOKEN = os.environ.get("motherduck_token", "")
OUTPUT_DIR = Path(__file__).parent / "dataset_clean"

LOOKBACK     = 18
HORIZON      = 6
SUN_ALT_MIN  = 10.0
ANCHOR_SOURCE = "asrs"

TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_END   = pd.Timestamp("2024-12-31 23:59:59")


# ─── 1. LOAD: solar_radiation_valid + prev-hour ptm ──────────────────────────
def load_data() -> pd.DataFrame:
    conn_str = f"md:jambi?motherduck_token={MOTHERDUCK_TOKEN}" if MOTHERDUCK_TOKEN else "md:jambi"
    print("Connecting to MotherDuck...")
    conn = duckdb.connect(conn_str)

    # Ambil solar_radiation_valid + join csv_fixed_ptm JAM SEBELUMNYA (non-leaky)
    query = """
    SELECT
        s.timestamp_wib,
        s.ghi_final,
        s.ghi_source,
        s.kt,
        s.sun_altitude,
        s.sun_azimuth,
        s.ghi_clearsky,
        s.has_gap_before,
        s.n_minutes,
        s.data_flag_max,
        -- Meteorologi
        s.temp_air,
        s.rh,
        s.pressure,
        s.ws,
        s.ws_max,
        s.wd,
        s.rain,
        s.cloud_oktas,
        -- PTM JAM SEBELUMNYA (non-leaky)
        -- period_end = DATE_TRUNC('hour', ts) = jam mulai anchor = akhir jam sebelumnya
        ptm_prev.cloud_opacity      AS ptm_cloud_opacity_prev,
        ptm_prev.precipitable_water AS ptm_pw_prev,
        ptm_prev.ghi                AS ptm_ghi_prev,
        -- AERONET (daily aerosol)
        s.aeronet_aod500,
        s.aeronet_ae,
        s.aeronet_pwv_cm,
        CAST(s.aeronet_smoke_flag AS INTEGER) AS aeronet_smoke_flag,
        s.aeronet_n_obs
    FROM jambi.solar_radiation_valid s
    LEFT JOIN jambi.csv_fixed_ptm ptm_prev
        ON DATE_TRUNC('hour', s.timestamp_wib)
           = (ptm_prev.period_end AT TIME ZONE 'Asia/Jakarta')::TIMESTAMP
    WHERE s.timestamp_wib >= '2022-01-01'
      AND s.timestamp_wib <  '2026-01-01'
    ORDER BY s.timestamp_wib
    """

    t0  = time.time()
    df  = conn.execute(query).df()
    conn.close()

    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    df["has_gap_before"] = df["has_gap_before"].fillna(False).astype(bool)

    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Verifikasi: ptm_prev seharusnya NULL untuk jam 00:00 WIB (tidak ada jam sebelumnya)
    null_ptm = df["ptm_cloud_opacity_prev"].isna().mean()
    print(f"  ptm_cloud_opacity_prev NULL rate: {null_ptm:.3f}")
    return df


# ─── 2. DERIVED FEATURES ─────────────────────────────────────────────────────
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["month"]     = df["timestamp_wib"].dt.month
    df["hour"]      = df["timestamp_wib"].dt.hour
    df["minute"]    = df["timestamp_wib"].dt.minute
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_frac"] = df["hour"] + df["minute"] / 60
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_frac"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_frac"] / 24)

    df["delta_ghi"] = df["ghi_final"].diff()
    df["delta_kt"]  = df["kt"].diff()
    gap_mask = df["has_gap_before"] == True
    df.loc[gap_mask, "delta_ghi"] = np.nan
    df.loc[gap_mask, "delta_kt"]  = np.nan
    return df


# ─── 3. SLIDING WINDOW (versi pruned) ────────────────────────────────────────
TEN_MIN = np.timedelta64(10, "m")

# Berapa lag yang diambil per grup fitur (pruning)
GHI_LAGS    = 12   # ghi_lag1..12  (2 jam terakhir, cukup)
KT_LAGS     = 6    # kt_lag1..6
DELTA_LAGS  = 6    # delta_ghi/kt lag1..6
# sun_alt_lag DIHAPUS (low importance, info ada di sun_alt_h juga)
# cloud_op_lag DIHAPUS (leaky untuk lag dalam jam yg sama)


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding windows (clean, pruned)...")
    t0 = time.time()
    df = df.reset_index(drop=True)
    n  = len(df)

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

    temp_air     = df["temp_air"].values.astype(float)
    rh           = df["rh"].values.astype(float)
    pressure     = df["pressure"].values.astype(float)
    ws           = df["ws"].values.astype(float)
    ws_max       = df["ws_max"].values.astype(float)
    wd           = df["wd"].values.astype(float)
    rain         = df["rain"].values.astype(float)
    cloud_oktas  = df["cloud_oktas"].values.astype(float)

    # PTM JAM SEBELUMNYA (clean)
    ptm_cloud_prev = df["ptm_cloud_opacity_prev"].values.astype(float)
    ptm_pw_prev    = df["ptm_pw_prev"].values.astype(float)

    # AERONET
    ae_aod500    = df["aeronet_aod500"].values.astype(float)
    ae_ae        = df["aeronet_ae"].values.astype(float)
    ae_pwv       = df["aeronet_pwv_cm"].values.astype(float)
    ae_smoke     = df["aeronet_smoke_flag"].values.astype(float)
    ae_nobs      = df["aeronet_n_obs"].values.astype(float)

    month_sin    = df["month_sin"].values.astype(float)
    month_cos    = df["month_cos"].values.astype(float)
    hour_sin     = df["hour_sin"].values.astype(float)
    hour_cos     = df["hour_cos"].values.astype(float)

    rows = []
    n_skip_src = n_skip_alt = n_skip_gap = 0

    for i in range(LOOKBACK, n - HORIZON):
        if ghi_src[i] != ANCHOR_SOURCE:
            n_skip_src += 1; continue
        if sun_alt[i] < SUN_ALT_MIN:
            n_skip_alt += 1; continue
        if np.isnan(ghi[i]):
            continue
        if sun_alt[i + HORIZON] < SUN_ALT_MIN:
            n_skip_alt += 1; continue

        # Cek kontinuitas 10-menit
        lag_diff = np.diff(ts[i - LOOKBACK : i + 1])
        if np.any(lag_diff != TEN_MIN):
            n_skip_gap += 1; continue
        hor_diff = np.diff(ts[i : i + HORIZON + 1])
        if np.any(hor_diff != TEN_MIN):
            n_skip_gap += 1; continue

        row: dict = {
            "anchor_ts"     : ts[i],
            "anchor_ghi"    : ghi[i],
            "anchor_kt"     : kt[i],
            "anchor_sun_alt": sun_alt[i],
            "anchor_sun_az" : sun_az[i],
            "month_sin"     : month_sin[i],
            "month_cos"     : month_cos[i],
            "hour_sin"      : hour_sin[i],
            "hour_cos"      : hour_cos[i],
            # Meteo
            "temp_air"      : temp_air[i],
            "rh"            : rh[i],
            "pressure"      : pressure[i],
            "ws"            : ws[i],
            "ws_max"        : ws_max[i],
            "wd"            : wd[i],
            "rain"          : rain[i],
            "cloud_oktas"   : cloud_oktas[i],
            # PTM jam sebelumnya (clean)
            "ptm_cloud_prev": ptm_cloud_prev[i],
            "ptm_pw_prev"   : ptm_pw_prev[i],
            # AERONET daily
            "aeronet_aod500"    : ae_aod500[i],
            "aeronet_ae"        : ae_ae[i],
            "aeronet_pwv_cm"    : ae_pwv[i],
            "aeronet_smoke_flag": ae_smoke[i],
            "aeronet_n_obs"     : ae_nobs[i],
        }

        # GHI lag 1..GHI_LAGS
        for k in range(1, GHI_LAGS + 1):
            j = i - k
            row[f"ghi_lag{k}"]  = ghi[j]

        # kt lag 1..KT_LAGS
        for k in range(1, KT_LAGS + 1):
            j = i - k
            row[f"kt_lag{k}"]   = kt[j]

        # Delta lag 1..DELTA_LAGS
        for k in range(1, DELTA_LAGS + 1):
            j = i - k
            row[f"delta_ghi_lag{k}"] = delta_ghi[j + 1]
            row[f"delta_kt_lag{k}"]  = delta_kt[j + 1]

        # Extended GHI lag (lag7..18 tetap diambil tapi sebagai grup terpisah)
        for k in range(GHI_LAGS + 1, LOOKBACK + 1):
            j = i - k
            row[f"ghi_lag{k}"] = ghi[j]

        # Future known (deterministic)
        for h in range(1, HORIZON + 1):
            j = i + h
            row[f"sun_alt_h{h}"]      = sun_alt[j]
            row[f"sun_az_h{h}"]       = sun_az[j]
            row[f"clearsky_ghi_h{h}"] = clearsky[j]

        # Targets
        for h in range(1, HORIZON + 1):
            j = i + h
            row[f"ghi_h{h}"] = ghi[j]
            row[f"kt_h{h}"]  = kt[j]

        rows.append(row)

    elapsed = time.time() - t0
    result  = pd.DataFrame(rows)
    print(f"  Windows: {len(result):,} valid | Skipped: src={n_skip_src:,} alt={n_skip_alt:,} gap={n_skip_gap:,} | {elapsed:.1f}s")
    return result


# ─── 4. SPLIT ─────────────────────────────────────────────────────────────────
def temporal_split(df):
    ts    = pd.to_datetime(df["anchor_ts"])
    train = df[ts <= TRAIN_END].copy()
    val   = df[(ts > TRAIN_END) & (ts <= VAL_END)].copy()
    test  = df[ts > VAL_END].copy()
    print(f"\nSplit: Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")
    return train, val, test


# ─── 5. DEFINISI STAGE ────────────────────────────────────────────────────────
EXCL = (
    {"anchor_ts"}
    | {f"ghi_h{h}" for h in range(1, HORIZON + 1)}
    | {f"kt_h{h}"  for h in range(1, HORIZON + 1)}
)

def stage_cols(df, extra_drop=None):
    cols = [c for c in df.columns if c not in EXCL]
    if extra_drop:
        cols = [c for c in cols if c not in extra_drop]
    return cols

# Stage 1: radiasi + sun position (tanpa meteo & cloud)
METEO_COLS  = ["temp_air", "rh", "pressure", "ws", "ws_max", "wd", "rain", "cloud_oktas"]
CLOUD_COLS  = ["ptm_cloud_prev", "ptm_pw_prev",
               "aeronet_aod500", "aeronet_ae", "aeronet_pwv_cm",
               "aeronet_smoke_flag", "aeronet_n_obs"]


# ─── 6. SAVE ──────────────────────────────────────────────────────────────────
def save_all(train, val, test):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stage, drop in [
        (1, METEO_COLS + CLOUD_COLS),
        (2, CLOUD_COLS),
        (3, []),
    ]:
        fcols = stage_cols(train, extra_drop=drop)
        print(f"\nStage {stage} ({len(fcols)} cols):")
        for split_name, df in [("train", train), ("val", val), ("test", test)]:
            available = [c for c in fcols if c in df.columns]
            path = OUTPUT_DIR / f"jambi_clean_s{stage}_{split_name}.parquet"
            df[available + [f"ghi_h{h}" for h in range(1,7)] +
                           [f"kt_h{h}"  for h in range(1,7)]].to_parquet(path, index=False)
            mb = path.stat().st_size / 1024 / 1024
            print(f"  {path.name}: {len(df):,}×{len(available)+12} cols, {mb:.1f}MB")


# ─── 7. NULL RATE CHECK ───────────────────────────────────────────────────────
def check_nulls(train, val, test):
    print("\nNULL rates per split:")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name}:")
        for col in ["anchor_ghi", "ghi_lag1", "temp_air",
                    "ptm_cloud_prev", "aeronet_aod500"]:
            if col in df.columns:
                print(f"    {col:30s}: {df[col].isna().mean():.3f}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BUILD CLEAN GHI DATASET — Leakage fixed, features pruned")
    print("  PTM: prev-hour cloud opacity (non-leaky)")
    print(f"  GHI lags: 1..{LOOKBACK} | kt/delta lags: 1..{max(KT_LAGS, DELTA_LAGS)}")
    print("=" * 60)

    df      = load_data()
    df      = add_derived(df)
    windows = build_windows(df)

    if len(windows) == 0:
        print("ERROR: Tidak ada window valid!")
        return

    train, val, test = temporal_split(windows)
    check_nulls(train, val, test)
    save_all(train, val, test)
    print("\n✅ Dataset bersih selesai → dataset_clean/")


if __name__ == "__main__":
    main()
