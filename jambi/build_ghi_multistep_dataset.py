#!/usr/bin/env python3
"""
build_ghi_multistep_dataset.py
================================
Dataset GHI perjam multi-horizon 6 langkah ke depan — Stasiun Jambi.

Desain:
  Anchor : jam t (GHI rata-rata, cloud SYNOP, meteo)
  Target : ghi_h1..ghi_h6 = rata-rata GHI jam t+1 sampai t+6
  History: 18 jam ke belakang (termasuk malam → cloud SYNOP tetap tersedia)
  Syarat : sun_alt > 5° di anchor DAN semua 6 target jam

Label musim Jambi:
  Musim Hujan  : Oktober–April   (bulan 10,11,12,1,2,3,4)
  Musim Kemarau: Mei–September   (bulan 5,6,7,8,9)

Output:
  dataset_multistep/
    jambi_ms_{train,val,test}.parquet
"""

import os
import duckdb
import numpy as np
import pandas as pd
import pvlib
from pathlib import Path

# ─── KONEKSI ─────────────────────────────────────────────────────────────────
MD_TOKEN = os.environ.get("MOTHERDUCK_TOKEN", "")
con = duckdb.connect(f"md:jambi?motherduck_token={MD_TOKEN}" if MD_TOKEN else "md:jambi")

LAT, LON, ELEV = -1.5833, 103.6667, 35.0
OUT_DIR = Path(__file__).parent / "dataset_multistep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON  = 6
LAG_HOURS = 18

# ─── GENERATE SQL LAGS DAN LEADS ─────────────────────────────────────────────
def make_lag_sql():
    """Generate SQL expressions untuk 18 lag dan 6 lead."""
    lines = []
    # GHI lags (akan NULL untuk jam malam → ditangani di Python)
    for k in range(1, LAG_HOURS + 1):
        lines.append(f"        LAG(m.ghi_h,    {k:2d}) OVER w AS ghi_lag{k}")
        lines.append(f"        LAG(m.ghi_last, {k:2d}) OVER w AS ghi_last_lag{k}")
    # KT lags 1-6
    for k in range(1, 7):
        lines.append(f"        LAG(m.kt_h, {k}) OVER w AS kt_lag{k}")
    # Cloud lags 18 jam (SYNOP tersedia malam, bagus untuk tren awan)
    for k in range(1, LAG_HOURS + 1):
        lines.append(f"        LAG(m.cloud_oktas, {k:2d}) OVER w AS cloud_lag{k}")
    # Meteo lags 1-6
    for k in range(1, 7):
        lines.append(f"        LAG(m.temp_air,  {k}) OVER w AS temp_lag{k}")
        lines.append(f"        LAG(m.rh,        {k}) OVER w AS rh_lag{k}")
    for k in range(1, 4):
        lines.append(f"        LAG(m.ws,        {k}) OVER w AS ws_lag{k}")
    for k in range(1, 7):
        lines.append(f"        LAG(m.rain_sum,  {k}) OVER w AS rain_lag{k}")
    # Lead targets
    for h in range(1, HORIZON + 1):
        lines.append(f"        LEAD(m.ghi_h,    {h}) OVER w AS ghi_h{h}")
        lines.append(f"        LEAD(m.sun_alt_h,{h}) OVER w AS sun_alt_lead{h}")
        lines.append(f"        LEAD(m.hour_wib, {h}) OVER w AS hour_h{h}")
    return ",\n".join(lines)


def make_filter_sql():
    """WHERE clause: semua 6 target harus siang & tidak ada gap."""
    conds = ["sun_alt_h > 5", "ghi_h1 IS NOT NULL", "ghi_h6 IS NOT NULL"]
    for h in range(1, HORIZON + 1):
        conds.append(f"sun_alt_lead{h} > 5")
    conds.append(f"DATEDIFF('hour', hour_wib, hour_h6) = {HORIZON}")
    return "\n  AND ".join(conds)


# ─── MAIN QUERY ───────────────────────────────────────────────────────────────
print("Mengambil data dari MotherDuck ...")

query = f"""
WITH hourly_solar AS (
    -- Agregasi solar_radiation_valid ke jam
    SELECT
        DATE_TRUNC('hour', timestamp_wib) AS hour_wib,
        AVG(ghi)                               AS ghi_h,
        LAST(ghi ORDER BY timestamp_wib)       AS ghi_last,
        STDDEV_SAMP(ghi)                       AS ghi_std,
        AVG(ghi_clearsky)                      AS clearsky_h,
        AVG(sun_altitude)                      AS sun_alt_h,
        AVG(sun_azimuth)                       AS sun_az_h,
        AVG(kt)                                AS kt_h,
        AVG(dni)                               AS dni_h,
        AVG(cloud_oktas)                       AS cloud_oktas_solar,
        AVG(cloud_base_m)                      AS cloud_base_m,
        FIRST(cloud_low_type  ORDER BY timestamp_wib) AS cloud_low_type,
        FIRST(cloud_med_type  ORDER BY timestamp_wib) AS cloud_med_type,
        FIRST(cloud_high_type ORDER BY timestamp_wib) AS cloud_high_type,
        FIRST(present_weather ORDER BY timestamp_wib) AS present_weather,
        AVG(temp_air)  AS temp_air_solar,
        AVG(rh)        AS rh_solar,
        AVG(pressure)  AS pressure_solar,
        AVG(ws)        AS ws_solar,
        SUM(rain)      AS rain_sum_solar,
        COUNT(*)       AS n_10min
    FROM solar_radiation_valid
    WHERE YEAR(timestamp_wib) BETWEEN 2022 AND 2025
    GROUP BY 1
    HAVING COUNT(*) >= 3
),
hourly_synop AS (
    -- SYNOP: ALL 24 jam (penting untuk lag malam)
    SELECT
        DATE_TRUNC('hour', waktu) AS hour_wib,
        AVG(cloud_cover_oktas)     AS synop_oktas,
        AVG(cloud_base_m)          AS synop_cloud_base,
        FIRST(cloud_low_type ORDER BY waktu)  AS synop_cloud_low,
        FIRST(present_weather ORDER BY waktu) AS synop_pw,
        AVG(temp_drybulb_c)        AS synop_temp,
        AVG(relative_humidity_pct) AS synop_rh,
        AVG(pressure_qff_mb)       AS synop_pressure,
        AVG(wind_speed_kt * 0.5144) AS synop_ws,
        AVG(visibility_km)         AS visibility_km,
        AVG(pressure_tend_3h_mb)   AS pressure_tend_3h,
        AVG(rainfall_6h_mm)        AS rain_6h
    FROM synop_jambi_combined
    WHERE YEAR(waktu) BETWEEN 2022 AND 2025
    GROUP BY 1
),
merged AS (
    -- Gabung solar (siang) + SYNOP (24 jam)
    -- Untuk jam siang: pakai solar, fallback ke SYNOP
    -- Untuk jam malam: hanya SYNOP (untuk lag fitur cloud)
    SELECT
        COALESCE(s.hour_wib, y.hour_wib)          AS hour_wib,
        s.ghi_h, s.ghi_last, s.ghi_std,
        s.clearsky_h, s.sun_alt_h, s.sun_az_h, s.kt_h, s.dni_h,
        -- Cloud: preferensi solar (lebih dekat stasiun)
        COALESCE(s.cloud_oktas_solar, y.synop_oktas)  AS cloud_oktas,
        COALESCE(s.cloud_base_m, y.synop_cloud_base)  AS cloud_base_m,
        COALESCE(s.cloud_low_type, y.synop_cloud_low)  AS cloud_low_type,
        COALESCE(s.cloud_med_type, NULL)               AS cloud_med_type,
        COALESCE(s.cloud_high_type, NULL)              AS cloud_high_type,
        COALESCE(s.present_weather, y.synop_pw)        AS present_weather,
        -- Meteo
        COALESCE(s.temp_air_solar, y.synop_temp)   AS temp_air,
        COALESCE(s.rh_solar, y.synop_rh)           AS rh,
        COALESCE(s.pressure_solar, y.synop_pressure) AS pressure,
        COALESCE(s.ws_solar, y.synop_ws)           AS ws,
        COALESCE(s.rain_sum_solar, 0)              AS rain_sum,
        y.visibility_km, y.pressure_tend_3h, y.rain_6h,
        s.n_10min
    FROM hourly_solar s
    FULL OUTER JOIN hourly_synop y ON y.hour_wib = s.hour_wib
),
with_window AS (
    SELECT
        m.*,
{make_lag_sql()}
    FROM merged m
    WINDOW w AS (ORDER BY m.hour_wib)
)
SELECT *
FROM with_window
WHERE {make_filter_sql()}
ORDER BY hour_wib
"""

df = con.execute(query).fetchdf()
print(f"Windows valid: {len(df):,}")
print(f"Distribusi per tahun:")
for yr, g in df.groupby(df.hour_wib.dt.year):
    print(f"  {yr}: {len(g):,} windows  (jam anchor: {g.hour_wib.dt.hour.min()}:00-{g.hour_wib.dt.hour.max()}:00)")

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
print("\nFeature engineering ...")

# 1. Delta (perubahan 1 jam)
df["delta_ghi_1h"]   = df["ghi_h"]      - df["ghi_lag1"]
df["delta_kt_1h"]    = df["kt_h"]       - df["kt_lag1"]
df["delta_cloud_1h"] = df["cloud_oktas"]- df["cloud_lag1"]
df["delta_temp_1h"]  = df["temp_air"]   - df["temp_lag1"]
df["delta_rh_1h"]    = df["rh"]         - df["rh_lag1"]

# Delta 3-jam (tren lebih panjang)
df["delta_ghi_3h"]   = df["ghi_h"]      - df["ghi_lag3"]
df["delta_cloud_3h"] = df["cloud_oktas"]- df["cloud_lag3"]

# 2. Variabilitas GHI intra-jam
df["ghi_std_norm"] = np.where(df["ghi_h"] > 10, df["ghi_std"] / df["ghi_h"], 0.0)
df["kt_last"]      = np.where(df["clearsky_h"] > 10, df["ghi_last"] / df["clearsky_h"], np.nan)

# 3. Rolling mean cloud 3h, 6h, 12h (overnight trend)
df_sorted = df.sort_values("hour_wib")
cloud_cols_6 = [f"cloud_lag{k}" for k in range(1, 7)   if f"cloud_lag{k}" in df.columns]
cloud_cols_12= [f"cloud_lag{k}" for k in range(1, 13)  if f"cloud_lag{k}" in df.columns]
cloud_cols_18= [f"cloud_lag{k}" for k in range(1, 19)  if f"cloud_lag{k}" in df.columns]
df["cloud_mean_6h"]  = df[cloud_cols_6].mean(axis=1)
df["cloud_mean_12h"] = df[cloud_cols_12].mean(axis=1)
df["cloud_mean_18h"] = df[cloud_cols_18].mean(axis=1)
df["cloud_trend_6h"] = df["cloud_oktas"] - df["cloud_mean_6h"]   # positive = cloud increasing

# Rolling mean GHI 3h, 6h (ingat banyak NULL malam → fillna 0)
ghi_lag_cols_3 = [f"ghi_lag{k}" for k in range(1, 4)]
ghi_lag_cols_6 = [f"ghi_lag{k}" for k in range(1, 7)]
df["ghi_mean_3h"] = df[ghi_lag_cols_3].fillna(0).mean(axis=1)
df["ghi_mean_6h"] = df[ghi_lag_cols_6].fillna(0).mean(axis=1)

# 4. Cyclical time
df["month_sin"] = np.sin(2 * np.pi * df.hour_wib.dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.hour_wib.dt.month / 12)
df["hour_sin"]  = np.sin(2 * np.pi * df.hour_wib.dt.hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df.hour_wib.dt.hour / 24)
df["doy_sin"]   = np.sin(2 * np.pi * df.hour_wib.dt.dayofyear / 365)
df["doy_cos"]   = np.cos(2 * np.pi * df.hour_wib.dt.dayofyear / 365)

# 5. Label musim
# Musim Hujan: Oktober-April | Musim Kemarau: Mei-September
df["is_wet_season"] = df.hour_wib.dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(float)
df["musim"] = df["is_wet_season"].map({1.0: "Hujan", 0.0: "Kemarau"})

# 6. Cloud regime dummies
df["sky_clear"]     = (df["cloud_oktas"] <= 2).astype(float)
df["sky_scattered"] = ((df["cloud_oktas"] > 2) & (df["cloud_oktas"] <= 5)).astype(float)
df["sky_broken"]    = ((df["cloud_oktas"] > 5) & (df["cloud_oktas"] <= 7)).astype(float)
df["sky_overcast"]  = (df["cloud_oktas"] >= 8).astype(float)
df["is_cb"]         = (df["cloud_low_type"].isin([3, 9])).astype(float)
df["rain_flag"]     = (df["rain_sum"] > 0.1).astype(float)

# 7. pvlib clearsky untuk 6 jam ke depan
print("  Menghitung clearsky pvlib untuk 6 horizon ...")
loc = pvlib.location.Location(latitude=LAT, longitude=LON, altitude=ELEV, tz="Asia/Jakarta")

# Clearsky anchor jam
ts_now = pd.DatetimeIndex(df["hour_wib"])
if ts_now.tz is None:
    ts_now = ts_now.tz_localize("Asia/Jakarta")
cs_now = loc.get_clearsky(ts_now, model="ineichen")
sp_now = loc.get_solarposition(ts_now)
df["clearsky_pvlib_h"]  = cs_now["ghi"].values
df["sun_alt_pvlib_h"]   = sp_now["apparent_elevation"].values

# Clearsky untuk setiap horizon t+1..t+6
for h in range(1, HORIZON + 1):
    ts_h = ts_now + pd.Timedelta(hours=h)
    cs_h = loc.get_clearsky(ts_h, model="ineichen")
    sp_h = loc.get_solarposition(ts_h)
    df[f"clearsky_pvlib_h{h}"] = cs_h["ghi"].values
    df[f"sun_alt_pvlib_h{h}"]  = sp_h["apparent_elevation"].values
    df[f"sun_az_pvlib_h{h}"]   = sp_h["azimuth"].values

# Fill NaN GHI lags malam dengan 0 (siang hari tidak ada GHI)
ghi_lag_cols = [f"ghi_lag{k}" for k in range(1, LAG_HOURS + 1)]
ghi_last_cols= [f"ghi_last_lag{k}" for k in range(1, LAG_HOURS + 1)]
df[ghi_lag_cols]  = df[ghi_lag_cols].fillna(0.0)
df[ghi_last_cols] = df[ghi_last_cols].fillna(0.0)

# ─── DEFINISI FITUR ──────────────────────────────────────────────────────────
FEAT_SOLAR = (
    ["ghi_h", "ghi_last", "ghi_std", "ghi_std_norm", "kt_h", "kt_last",
     "clearsky_h", "clearsky_pvlib_h", "sun_alt_h", "sun_az_h", "sun_alt_pvlib_h",
     "dni_h",
     "ghi_mean_3h", "ghi_mean_6h", "delta_ghi_1h", "delta_ghi_3h", "delta_kt_1h"]
    + [f"ghi_lag{k}"       for k in range(1, LAG_HOURS + 1)]
    + [f"ghi_last_lag{k}"  for k in range(1, 7)]
    + [f"kt_lag{k}"        for k in range(1, 7)]
    # Future solar geometry (no leakage)
    + [f"clearsky_pvlib_h{h}" for h in range(1, HORIZON + 1)]
    + [f"sun_alt_pvlib_h{h}"  for h in range(1, HORIZON + 1)]
    + [f"sun_az_pvlib_h{h}"   for h in range(1, HORIZON + 1)]
    # Time cyclical
    + ["month_sin", "month_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)

FEAT_CLOUD = (
    ["cloud_oktas", "cloud_base_m", "cloud_low_type", "cloud_med_type",
     "cloud_high_type", "present_weather",
     "cloud_mean_6h", "cloud_mean_12h", "cloud_mean_18h", "cloud_trend_6h",
     "delta_cloud_1h", "delta_cloud_3h",
     "sky_clear", "sky_scattered", "sky_broken", "sky_overcast", "is_cb"]
    + [f"cloud_lag{k}" for k in range(1, LAG_HOURS + 1)]
)

FEAT_METEO = (
    ["temp_air", "rh", "pressure", "ws", "rain_sum", "rain_flag",
     "visibility_km", "pressure_tend_3h", "rain_6h",
     "delta_temp_1h", "delta_rh_1h"]
    + [f"temp_lag{k}" for k in range(1, 7)]
    + [f"rh_lag{k}"   for k in range(1, 7)]
    + [f"ws_lag{k}"   for k in range(1, 4)]
    + [f"rain_lag{k}" for k in range(1, 7)]
)

FEAT_SEASON = ["is_wet_season"]

ALL_FEATS = FEAT_SOLAR + FEAT_CLOUD + FEAT_METEO + FEAT_SEASON

# Filter ke fitur yang ada
available = [c for c in ALL_FEATS if c in df.columns]
missing   = [c for c in ALL_FEATS if c not in df.columns]
if missing:
    print(f"  ⚠  Tidak ada: {missing[:5]}{'...' if len(missing)>5 else ''}")
print(f"  Fitur tersedia: {len(available)}")

# ─── NULL RATE CHECK ─────────────────────────────────────────────────────────
print("\nNull rate fitur kunci:")
check_cols = ["cloud_oktas", "cloud_lag1", "cloud_lag12", "cloud_lag18",
              "visibility_km", "pressure_tend_3h", "ghi_lag6", "ghi_lag12"]
for col in check_cols:
    if col in df.columns:
        nr = df[col].isna().mean()
        flag = " ⚠" if nr > 0.15 else ""
        print(f"  {col:25s}: {nr:.1%}{flag}")

# ─── SPLIT + MUSIM ───────────────────────────────────────────────────────────
yr = df.hour_wib.dt.year
df_train = df[yr.isin([2022, 2023])].copy()
df_val   = df[yr == 2024].copy()
df_test  = df[yr == 2025].copy()

print("\nSplit dataset:")
for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    n_wet = (d.musim == "Hujan").sum()
    n_dry = (d.musim == "Kemarau").sum()
    print(f"  {name:5s}: {len(d):5,} windows | "
          f"Hujan={n_wet:4,} ({n_wet/len(d):.0%}) | Kemarau={n_dry:4,} ({n_dry/len(d):.0%})")

# ─── SIMPAN ──────────────────────────────────────────────────────────────────
TARGET_COLS = [f"ghi_h{h}" for h in range(1, HORIZON + 1)]
ID_COLS     = ["hour_wib", "musim", "is_wet_season"]
COLS_OUT    = ID_COLS + available + TARGET_COLS

# Tambah kolom clearsky target untuk diagnostik
for h in range(1, HORIZON + 1):
    c = f"clearsky_pvlib_h{h}"
    if c not in COLS_OUT and c in df.columns:
        COLS_OUT.append(c)

COLS_OUT = list(dict.fromkeys(COLS_OUT))   # deduplicate

print("\nMenyimpan parquet ...")
for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    cols = [c for c in COLS_OUT if c in d.columns]
    out = OUT_DIR / f"jambi_ms_{name}.parquet"
    d[cols].to_parquet(out, index=False)
    print(f"  {out.name}: {len(d):,} rows × {len(cols)} kolom")

print(f"\nDataset multi-step selesai → {OUT_DIR}")
print(f"Lag history: {LAG_HOURS} jam | Horizons: {HORIZON}")
