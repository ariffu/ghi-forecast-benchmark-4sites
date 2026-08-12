#!/usr/bin/env python3
"""
build_ghi_hourly_dataset.py
============================
Build dataset GHI forecasting resolusi 1 JAM untuk Stasiun Jambi.

Perbedaan dari dataset 10-menit:
  - Anchor : rata-rata GHI 1 jam penuh (t)
  - Target : rata-rata GHI jam BERIKUTNYA (t+1)
  - Cloud  : cloud_oktas + cloud type SYNOP (real-time, no leakage)
  - Meteo  : dari SYNOP + AWS jam tsb
  - Lag    : GHI/kt/cloud 1-3 jam sebelumnya

Keunggulan vs 10-menit:
  - cloud_oktas adalah observasi real-time manusia (bukan reanalysis)
  - Averaging per jam mereduksi noise acak
  - Persistence R² hourly ~0.49 vs 0.38 untuk 10-menit
  - Target lebih smooth → model lebih mudah belajar

Output:
  dataset_hourly/
    jambi_hourly_{train,val,test}.parquet
"""

import os
import duckdb
import numpy as np
import pandas as pd
import pvlib
from pathlib import Path
from datetime import timezone

# ─── KONEKSI MOTHERDUCK ───────────────────────────────────────────────────────
MD_TOKEN = os.environ.get("MOTHERDUCK_TOKEN", "")
DB_NAME  = "jambi"
if MD_TOKEN:
    con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={MD_TOKEN}")
else:
    # Coba connect dengan token dari ~/.motherduck.token atau default
    con = duckdb.connect(f"md:{DB_NAME}")

# ─── KOORDINAT STASIUN ────────────────────────────────────────────────────────
LAT, LON, ELEV = -1.5833, 103.6667, 35.0   # Stasiun Klimatologi Jambi

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "dataset_hourly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── QUERY: AGREGASI PERJAM ───────────────────────────────────────────────────
print("Mengambil data dari MotherDuck ...")
query = """
WITH hourly_raw AS (
    -- Agregasi solar_radiation_valid ke resolusi 1 jam
    SELECT
        DATE_TRUNC('hour', timestamp_wib) AS hour_wib,
        -- GHI: mean jam, nilai terakhir (paling recent), variabilitas
        AVG(ghi)                               AS ghi_h,
        LAST(ghi ORDER BY timestamp_wib)       AS ghi_last,
        STDDEV_SAMP(ghi)                       AS ghi_std,
        AVG(ghi_clearsky)                      AS clearsky_h,
        AVG(sun_altitude)                      AS sun_alt_h,
        AVG(sun_azimuth)                       AS sun_az_h,
        AVG(kt)                                AS kt_h,
        -- Cloud dari SYNOP (observasi per jam → ambil nilai tengah atau mean)
        AVG(cloud_oktas)                       AS cloud_oktas,
        AVG(cloud_base_m)                      AS cloud_base_m,
        FIRST(cloud_low_type  ORDER BY timestamp_wib) AS cloud_low_type,
        FIRST(cloud_med_type  ORDER BY timestamp_wib) AS cloud_med_type,
        FIRST(cloud_high_type ORDER BY timestamp_wib) AS cloud_high_type,
        FIRST(present_weather ORDER BY timestamp_wib) AS present_weather,
        -- Meteo AWS/SYNOP
        AVG(temp_air)    AS temp_air,
        AVG(rh)          AS rh,
        AVG(pressure)    AS pressure,
        AVG(ws)          AS ws,
        MAX(ws_max)      AS ws_max,
        SUM(rain)        AS rain_sum,
        -- DNI, DHI
        AVG(dni)         AS dni_h,
        AVG(dhi)         AS dhi_h,
        COUNT(*)         AS n_10min,
        -- Validitas
        BOOL_AND(ghi IS NOT NULL)              AS all_ghi_valid
    FROM solar_radiation_valid
    WHERE YEAR(timestamp_wib) BETWEEN 2022 AND 2025
    GROUP BY 1
    HAVING COUNT(*) >= 3   -- ≥30 menit per jam
),
-- Tambah SYNOP extra: visibility, pressure tendency, rainfall 6h
synop_extra AS (
    SELECT
        DATE_TRUNC('hour', waktu) AS hour_wib,
        AVG(visibility_km)         AS visibility_km,
        AVG(pressure_tend_3h_mb)   AS pressure_tend_3h,
        AVG(rainfall_6h_mm)        AS rain_6h,
        AVG(wind_speed_kt * 0.5144) AS ws_synop_full  -- convert kt→m/s
    FROM synop_jambi_combined
    WHERE YEAR(waktu) BETWEEN 2022 AND 2025
    GROUP BY 1
),
hourly AS (
    SELECT h.*, s.visibility_km, s.pressure_tend_3h, s.rain_6h
    FROM hourly_raw h
    LEFT JOIN synop_extra s ON s.hour_wib = h.hour_wib
),
-- Tambah lag dan lead menggunakan window functions
with_window AS (
    SELECT
        h.*,
        -- === LAG GHI ===
        LAG(h.ghi_h,     1) OVER w AS ghi_lag1,
        LAG(h.ghi_h,     2) OVER w AS ghi_lag2,
        LAG(h.ghi_h,     3) OVER w AS ghi_lag3,
        LAG(h.ghi_last,  1) OVER w AS ghi_last_lag1,
        LAG(h.ghi_std,   1) OVER w AS ghi_std_lag1,
        -- === LAG KT ===
        LAG(h.kt_h,      1) OVER w AS kt_lag1,
        LAG(h.kt_h,      2) OVER w AS kt_lag2,
        LAG(h.kt_h,      3) OVER w AS kt_lag3,
        -- === LAG CLOUD ===
        LAG(h.cloud_oktas, 1) OVER w AS cloud_oktas_lag1,
        LAG(h.cloud_oktas, 2) OVER w AS cloud_oktas_lag2,
        LAG(h.cloud_oktas, 3) OVER w AS cloud_oktas_lag3,
        -- === LAG METEO ===
        LAG(h.temp_air,  1) OVER w AS temp_lag1,
        LAG(h.rh,        1) OVER w AS rh_lag1,
        LAG(h.ws,        1) OVER w AS ws_lag1,
        LAG(h.rain_sum,  1) OVER w AS rain_lag1,
        LAG(h.rain_sum,  2) OVER w AS rain_lag2,
        LAG(h.visibility_km, 1) OVER w AS vis_lag1,
        -- === TARGET (LEAD) ===
        LEAD(h.ghi_h,       1) OVER w AS ghi_next,
        LEAD(h.clearsky_h,  1) OVER w AS clearsky_next,
        LEAD(h.sun_alt_h,   1) OVER w AS sun_alt_next,
        LEAD(h.sun_az_h,    1) OVER w AS sun_az_next,
        LEAD(h.kt_h,        1) OVER w AS kt_next,
        -- Jam berikutnya (untuk verifikasi)
        LEAD(h.hour_wib,    1) OVER w AS hour_next
    FROM hourly h
    WINDOW w AS (ORDER BY h.hour_wib)
)
SELECT *
FROM with_window
WHERE sun_alt_h    > 5          -- anchor siang
  AND sun_alt_next > 5          -- target siang
  AND ghi_next IS NOT NULL       -- target harus ada
  AND ghi_lag1 IS NOT NULL       -- minimal 1 lag harus ada
  -- Pastikan lead 1 jam = gap 1 jam (bukan loncat karena missing data)
  AND DATEDIFF('hour', hour_wib, hour_next) = 1
ORDER BY hour_wib
"""

df = con.execute(query).fetchdf()
print(f"Total windows valid: {len(df):,}")
print(f"  Kolom: {len(df.columns)}")
print(f"  Periode: {df.hour_wib.min()} s/d {df.hour_wib.max()}")
print(f"  Distribusi per tahun:")
for yr, g in df.groupby(df.hour_wib.dt.year):
    print(f"    {yr}: {len(g):,} windows")

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
print("\nFeature engineering ...")

# 1. Delta (perubahan 1 jam)
df["delta_ghi_1h"]   = df["ghi_h"]      - df["ghi_lag1"]
df["delta_kt_1h"]    = df["kt_h"]       - df["kt_lag1"]
df["delta_cloud_1h"] = df["cloud_oktas"]- df["cloud_oktas_lag1"]
df["delta_temp_1h"]  = df["temp_air"]   - df["temp_lag1"]
df["delta_rh_1h"]    = df["rh"]         - df["rh_lag1"]

# 2. GHI intra-jam: variabilitas dan fraction dari clearsky
df["kt_last"]   = np.where(
    df["clearsky_h"] > 10,
    df["ghi_last"] / df["clearsky_h"],
    np.nan
)
df["ghi_std_norm"] = np.where(
    df["ghi_h"] > 10,
    df["ghi_std"] / df["ghi_h"],
    0.0
)

# 3. Cyclical time features
df["month_sin"] = np.sin(2 * np.pi * df.hour_wib.dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.hour_wib.dt.month / 12)
df["hour_sin"]  = np.sin(2 * np.pi * df.hour_wib.dt.hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df.hour_wib.dt.hour / 24)
df["doy_sin"]   = np.sin(2 * np.pi * df.hour_wib.dt.dayofyear / 365)
df["doy_cos"]   = np.cos(2 * np.pi * df.hour_wib.dt.dayofyear / 365)

# 4. Clearsky GHI untuk jam target — hitung ulang dengan pvlib
print("  Menghitung clearsky target jam dengan pvlib ...")
loc = pvlib.location.Location(latitude=LAT, longitude=LON, altitude=ELEV, tz='Asia/Jakarta')

# Buat timestamps untuk jam berikutnya
ts_next = pd.DatetimeIndex(df["hour_next"])
if ts_next.tz is None:
    ts_next = ts_next.tz_localize("Asia/Jakarta")
else:
    ts_next = ts_next.tz_convert("Asia/Jakarta")

solar_pos_next = loc.get_solarposition(ts_next)
cs_next = loc.get_clearsky(ts_next, model="ineichen")

df["clearsky_pvlib_next"] = cs_next["ghi"].values
df["sun_alt_pvlib_next"]  = solar_pos_next["apparent_elevation"].values
df["sun_az_pvlib_next"]   = solar_pos_next["azimuth"].values

# Clearsky untuk anchor jam — pvlib
ts_now = pd.DatetimeIndex(df["hour_wib"])
if ts_now.tz is None:
    ts_now = ts_now.tz_localize("Asia/Jakarta")
else:
    ts_now = ts_now.tz_convert("Asia/Jakarta")

solar_pos_now = loc.get_solarposition(ts_now)
cs_now = loc.get_clearsky(ts_now, model="ineichen")
df["clearsky_pvlib_h"]  = cs_now["ghi"].values
df["sun_alt_pvlib_h"]   = solar_pos_now["apparent_elevation"].values

# 5. Clearness index target (ini adalah TARGET kt, bukan fitur!)
# Simpan sebagai info saja, jangan pakai sebagai fitur langsung
df["kt_next"] = np.where(
    df["clearsky_pvlib_next"] > 10,
    df["ghi_next"] / df["clearsky_pvlib_next"],
    np.nan
)

# 6. Cloud-clearsky interaction
df["cloud_clearsky_ratio"] = df["cloud_oktas"] / 8.0   # normalize 0-1
df["ghi_cloud_interact"]   = df["ghi_h"] * (1 - df["cloud_clearsky_ratio"])

# 7. Rain flag
df["rain_flag"] = (df["rain_sum"] > 0.1).astype(float)
df["rain_prev_flag"] = (df["rain_lag1"] > 0.1).astype(float)

# 8. Cloud type one-hot encoding (cloud_low_type paling informatif)
# Tipe utama: 0=no cloud, 1-5=Cu/Sc/St, 6=Cb, 7-9=Ns/fracto
df["is_cb"] = (df["cloud_low_type"].isin([3, 9])).astype(float)   # Cb
df["is_cu"] = (df["cloud_low_type"].isin([1, 2])).astype(float)   # Cu
df["is_sc"] = (df["cloud_low_type"].isin([4, 5, 8])).astype(float) # Sc

# Oktas bucket
df["sky_clear"]      = (df["cloud_oktas"] <= 2).astype(float)
df["sky_scattered"]  = ((df["cloud_oktas"] > 2) & (df["cloud_oktas"] <= 5)).astype(float)
df["sky_broken"]     = ((df["cloud_oktas"] > 5) & (df["cloud_oktas"] <= 7)).astype(float)
df["sky_overcast"]   = (df["cloud_oktas"] >= 8).astype(float)

# ─── DEFINISI FITUR ──────────────────────────────────────────────────────────
FEAT_COLS = [
    # Solar anchor
    "ghi_h", "ghi_last", "ghi_std", "ghi_std_norm",
    "clearsky_h", "clearsky_pvlib_h",
    "sun_alt_h", "sun_az_h", "sun_alt_pvlib_h",
    "kt_h", "kt_last",

    # GHI lags
    "ghi_lag1", "ghi_lag2", "ghi_lag3",
    "ghi_last_lag1", "ghi_std_lag1",

    # KT lags
    "kt_lag1", "kt_lag2", "kt_lag3",

    # Delta
    "delta_ghi_1h", "delta_kt_1h", "delta_cloud_1h",
    "delta_temp_1h", "delta_rh_1h",

    # Cloud SYNOP (current hour)
    "cloud_oktas", "cloud_base_m",
    "cloud_low_type", "cloud_med_type", "cloud_high_type",
    "present_weather",

    # Cloud lags
    "cloud_oktas_lag1", "cloud_oktas_lag2", "cloud_oktas_lag3",

    # Cloud derived
    "cloud_clearsky_ratio", "ghi_cloud_interact",
    "is_cb", "is_cu", "is_sc",
    "sky_clear", "sky_scattered", "sky_broken", "sky_overcast",

    # Meteo
    "temp_air", "rh", "pressure", "ws", "ws_max",
    "rain_sum", "rain_flag",
    "visibility_km", "pressure_tend_3h", "rain_6h",

    # Meteo lags
    "temp_lag1", "rh_lag1", "ws_lag1",
    "rain_lag1", "rain_lag2", "rain_prev_flag",
    "vis_lag1",

    # Future-known (solar geometri jam target)
    "clearsky_pvlib_next",
    "sun_alt_pvlib_next", "sun_az_pvlib_next",

    # Cyclical time
    "month_sin", "month_cos",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
]

TARGET_COL = "ghi_next"
ID_COLS    = ["hour_wib", "hour_next"]

# Filter kolom yang benar-benar ada
feat_available = [c for c in FEAT_COLS if c in df.columns]
missing_feat   = [c for c in FEAT_COLS if c not in df.columns]
if missing_feat:
    print(f"  ⚠  Fitur tidak ditemukan (skip): {missing_feat}")
print(f"  Fitur dipakai: {len(feat_available)}")

# ─── NULL RATE CHECK ─────────────────────────────────────────────────────────
print("\nNull rate fitur kunci:")
for col in ["cloud_oktas", "cloud_base_m", "visibility_km", "pressure_tend_3h",
            "ghi_lag1", "clearsky_pvlib_next"]:
    if col in df.columns:
        nr = df[col].isna().mean()
        flag = " ⚠" if nr > 0.1 else ""
        print(f"  {col:30s}: {nr:.1%}{flag}")

# ─── SPLIT DATASET ────────────────────────────────────────────────────────────
print("\nSplit dataset:")
yr = df.hour_wib.dt.year
df_train = df[yr.isin([2022, 2023])].copy()
df_val   = df[yr == 2024].copy()
df_test  = df[yr == 2025].copy()

for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(f"  {name:5s}: {len(d):5,} windows | "
          f"GHI target: mean={d[TARGET_COL].mean():.1f} std={d[TARGET_COL].std():.1f} W/m²")

# ─── SIMPAN PARQUET ──────────────────────────────────────────────────────────
print("\nMenyimpan parquet ...")
cols_out = ID_COLS + feat_available + [TARGET_COL]
# Tambah kolom info clearsky target untuk diagnostik
for c in ["clearsky_pvlib_next", "kt_next"]:
    if c not in cols_out and c in df.columns:
        cols_out.append(c)

for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    out_path = OUT_DIR / f"jambi_hourly_{name}.parquet"
    d[cols_out].to_parquet(out_path, index=False)
    print(f"  {out_path.name}: {len(d):,} rows, {len(cols_out)} kolom")

print(f"\nDataset hourly selesai → {OUT_DIR}")
