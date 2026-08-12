"""
Preprocessing dataset training: prediksi GHI 1 jam ke depan (direct, single-step) — Kalbar.

Sumber tabel (kalbar_local.db):
  solar_kalbar_10m            radiasi + fill_tier (0% NULL, lihat note_07)
  meteorologi_kalbar_10m      suhu/RH/angin/hujan (0% NULL)
  clp_pontianak                cloud properties satelit Himawari (buffer 30km)
  arp_pontianak                 aerosol retrieval satelit Himawari (buffer 30km)
  aeronet_aod_min_pontianak     AOD ground-truth (coverage ~38%, sudah QC quality_pass_core)

Keputusan desain (terverifikasi lewat eksperimen sesi ini):
  1. Koordinat stasiun: lat=0.07489, lon=109.1905, alt=2m (Staklim Kalbar/Mempawah) —
     dikonfirmasi dari arp_pontianak.station_lat/lon. BUKAN koordinat Pontianak kota
     (-0.0236, 109.3294) yang salah dipakai training_enhanced_v2 generasi sebelumnya.
  2. Target utama = GHI mentah (W/m2) di T+60, bukan kt (clearsky index). kt menghilangkan
     komponen geometri matahari (yg 100% predictable) sehingga R2-nya tampak rendah meski
     metodologinya benar -- untuk kebutuhan praktis "prediksi GHI", target GHI mentah lebih
     representatif dan sebanding dengan riset Jambi/Bengkulu. kt tetap disimpan sbg kolom
     diagnostik & dipakai utk fitur lag/rolling (lebih stabil scale-nya utk lag features).
  3. Fitur satelit (CLOT/CLTT/CLTH/CLER, AOT/AE) di-shift +10 menit sebelum join -- cross-
     correlation menunjukkan CLOT(t) berkorelasi terkuat dgn kt(t+10), bukan kt(t).
  4. anchor_valid mensyaratkan: siang hari di T DAN T+60 (sun_altitude>5), tidak ada gap
     waktu, DAN target/anchor BUKAN TIER_5_CLOUD_ML_IMPUTED (aturan baku proyek -- TIER_5
     adalah estimasi model penuh, melatih model di atasnya = "menebak tebakan model lain").
  5. Fitur "future" (sun_altitude_future, ghi_clearsky_future, hour_sin/cos_future) bersifat
     deterministik (posisi matahari diketahui pasti di masa depan) -- bukan leakage,
     terbukti membantu di proyek Jambi (Hybrid B).
"""
import duckdb
import pandas as pd
import numpy as np
import pvlib

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
OUT_PARQUET = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_direct.parquet"

LAT, LON, ALT = 0.07489, 109.1905, 2.0
TZ = "Asia/Jakarta"
HORIZON_MIN = 60

# ---------------------------------------------------------------- 1. Load sumber
con = duckdb.connect(DB_PATH, read_only=True)

solar = con.execute("""
    SELECT timestamp_wib, ghi_final, dni_final, dhi_final, sun_altitude,
           sunshine_minutes, fill_tier, quality_score
    FROM solar_kalbar_10m ORDER BY timestamp_wib
""").df()

meteo = con.execute("""
    SELECT timestamp_wib, suhu_avg_c AS temp_air_c, rh_percent AS humidity_pct,
           wind_speed_ms, rainfall_mm
    FROM meteorologi_kalbar_10m
""").df()

clp = con.execute("""
    SELECT timestamp_wib, CLOT_mean, CLTT_mean, CLTH_mean, CLER_23_mean,
           cloud_present::INT AS clp_cloud_present_int
    FROM clp_pontianak
""").df()

arp = con.execute("SELECT timestamp_wib, AOT_mean, AE_mean FROM arp_pontianak").df()

aeronet = con.execute("""
    SELECT timestamp_wib, AOD_500nm, "440-870_Angstrom_Exponent" AS angstrom_440_870,
           "Precipitable_Water(cm)" AS precipitable_water_cm
    FROM aeronet_aod_min_pontianak
""").df()

con.close()

# ---------------------------------------------------------------- 2. Shift satelit +10 menit (causal)
clp["timestamp_wib"] = clp["timestamp_wib"] + pd.Timedelta(minutes=10)
arp["timestamp_wib"] = arp["timestamp_wib"] + pd.Timedelta(minutes=10)

df = (
    solar.merge(meteo, on="timestamp_wib", how="left")
         .merge(clp, on="timestamp_wib", how="left")
         .merge(arp, on="timestamp_wib", how="left")
         .merge(aeronet, on="timestamp_wib", how="left")
         .sort_values("timestamp_wib")
         .reset_index(drop=True)
)
df["ts"] = df["timestamp_wib"]

# ---------------------------------------------------------------- 3. Clearsky (pvlib Ineichen)
loc = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)

def clearsky_at(timestamps):
    idx = pd.DatetimeIndex(timestamps).tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    solpos = loc.get_solarposition(idx)
    cs = loc.get_clearsky(idx, model="ineichen", solar_position=solpos)
    return solpos["apparent_elevation"].values, cs["ghi"].values

_, df["ghi_clearsky"] = clearsky_at(df["timestamp_wib"])
df["kt"] = (df["ghi_final"] / df["ghi_clearsky"].clip(lower=1e-6)).clip(0, 1.5)

future_ts = df["timestamp_wib"] + pd.Timedelta(minutes=HORIZON_MIN)
df["sun_altitude_future"], df["ghi_clearsky_future"] = clearsky_at(future_ts)

# ---------------------------------------------------------------- 4. Target di T+60 (lookup langsung)
ghi_lookup = df.set_index("timestamp_wib")["ghi_final"]
kt_lookup = df.set_index("timestamp_wib")["kt"]
fill_lookup = df.set_index("timestamp_wib")["fill_tier"]

df["ghi_target_60m"] = future_ts.map(ghi_lookup).values   # TARGET UTAMA
df["kt_target_60m"] = future_ts.map(kt_lookup).values     # diagnostik
df["fill_tier_target"] = future_ts.map(fill_lookup).values

# ---------------------------------------------------------------- 5. Lag / rolling / dinamika
for lag, mnt in [(1, 10), (2, 20), (3, 30), (6, 60)]:
    gap_ok = df["ts"].diff(lag).dt.total_seconds().div(60).between(lag * 10 - 5, lag * 10 + 5)
    df[f"ghi_lag{mnt}m"] = np.where(gap_ok, df["ghi_final"].shift(lag), np.nan)
    df[f"kt_lag{mnt}m"] = np.where(gap_ok, df["kt"].shift(lag), np.nan)

df["kt_roll30m_mean"] = df["kt"].rolling(3, min_periods=2).mean().shift(1)
df["kt_roll60m_mean"] = df["kt"].rolling(6, min_periods=3).mean().shift(1)
df["kt_roll30m_std"] = df["kt"].rolling(3, min_periods=2).std().shift(1)
df["kt_roll60m_std"] = df["kt"].rolling(6, min_periods=3).std().shift(1)
gap_min = df["ts"].diff().dt.total_seconds() / 60
for c in ["kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std", "kt_roll60m_std"]:
    df.loc[gap_min > 15, c] = np.nan

df["delta_kt_10m"] = df["kt_lag10m"] - df["kt_lag20m"]
df["delta_kt_30m"] = df["kt_lag10m"] - df["kt_lag60m"]
clot_gap30_ok = df["ts"].diff(3).dt.total_seconds().div(60).between(25, 35)
df["clot_lag10m"] = np.where(gap_min.between(5, 15), df["CLOT_mean"].shift(1), np.nan)
df["clot_lag30m"] = np.where(clot_gap30_ok, df["CLOT_mean"].shift(3), np.nan)
df["delta_clot_30m"] = df["CLOT_mean"] - df["clot_lag30m"]
df["delta_ghi_30m"] = np.where(clot_gap30_ok, df["ghi_final"] - df["ghi_final"].shift(3), np.nan)

# ---------------------------------------------------------------- 6. Waktu siklik (T dan T+60)
hour = df["timestamp_wib"].dt.hour + df["timestamp_wib"].dt.minute / 60
doy = df["timestamp_wib"].dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
df["month"] = df["timestamp_wib"].dt.month

hour_f = future_ts.dt.hour + future_ts.dt.minute / 60
df["hour_sin_future"] = np.sin(2 * np.pi * hour_f / 24)
df["hour_cos_future"] = np.cos(2 * np.pi * hour_f / 24)

# ---------------------------------------------------------------- 7. Validitas & flag kualitas
step_gap_ok = (df["ts"].shift(-6) - df["ts"]).dt.total_seconds().eq(HORIZON_MIN * 60)

df["anchor_valid"] = (
    (df["sun_altitude"] > 5)
    & (df["sun_altitude_future"] > 5)
    & step_gap_ok.values
    & df["ghi_target_60m"].notna()
    & (df["fill_tier"] != "TIER_5_CLOUD_ML_IMPUTED")
    & (df["fill_tier_target"] != "TIER_5_CLOUD_ML_IMPUTED")
)
df["dhi_gt_ghi_flag"] = df["dhi_final"] > df["ghi_final"]

df = df.drop(columns=["ts"])

# ---------------------------------------------------------------- 8. Ringkasan & simpan
n_valid = int(df["anchor_valid"].sum())
print(f"Total baris grid 10-menit : {len(df):,}")
print(f"Baris anchor_valid        : {n_valid:,} ({n_valid/len(df)*100:.1f}%)")
print(f"dhi_gt_ghi_flag rate      : {df['dhi_gt_ghi_flag'].mean()*100:.2f}%")
print("\nDistribusi fill_tier (anchor_valid saja):")
print(df.loc[df["anchor_valid"], "fill_tier"].value_counts())
print("\nDeskripsi GHI target (anchor_valid saja):")
print(df.loc[df["anchor_valid"], "ghi_target_60m"].describe())
print("\nCakupan AERONET (anchor_valid saja):",
      f"{df.loc[df['anchor_valid'], 'AOD_500nm'].notna().mean()*100:.1f}%")
print("\nRentang tahun:", df["timestamp_wib"].dt.year.min(), "-", df["timestamp_wib"].dt.year.max())

df.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved: {OUT_PARQUET}  ({len(df.columns)} kolom)")
