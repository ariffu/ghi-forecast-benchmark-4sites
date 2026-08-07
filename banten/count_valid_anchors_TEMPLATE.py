#!/usr/bin/env python3
"""
PENGHITUNG ANCHOR VALID §2.3 (HOMOGEN 4 LOKASI) — untuk Tabel 1 paper.
Menerapkan filter benchmark IDENTIK di semua lokasi:
  - resolusi 10-menit
  - riwayat 3-jam kontinu (LAG 18 langkah tidak-null; ±grid kontinu)
  - matahari > 5 deg DI ANCHOR *dan* DI t+60 (akhir jendela target)   <- inti §2.3
  - GHI anchor & GHI(t+60) dalam [0, 1400]
Elevasi dihitung ASTRONOMIS (Cooper 1969) dari lat/lon — sama persis skrip R1/R8,
sehingga schema-agnostic (hanya butuh timestamp + GHI + lat/lon).

Cara pakai per lokasi: ganti blok CONFIG, jalankan:
  & "C:\\Program Files\\Python39\\python.exe" count_valid_anchors_TEMPLATE.py

Referensi terverifikasi BANTEN (basis ASTRONOMIS homogen — dipakai Tabel 1):
  total_anchor=90,384 (train 45,192 / val 22,660 / test 22,532).
  [Catatan: basis elevasi-tersimpan legacy = 90,488; selisih 104 baris (0,11%),
   R2 identik 0,680. Tabel 1 paper memakai 90,384 agar homogen dgn Bengkulu/Jambi.]
"""
import duckdb, numpy as np, pandas as pd

# ============ CONFIG PER-LOKASI (ganti bagian ini saja) ============
CONFIG = {
    "lokasi":   "Banten",
    "db_path":  "banten.duckdb",
    "sql":      "SELECT timestamp_wib AS ts, ghi AS ghi_now FROM solar_features_base ORDER BY timestamp_wib",
    "lat":      -6.26147,
    "lon":      106.7509,
    "meridian": 105.0,     # WIB
}
# Contoh konfig lokasi lain (isi sesuai skema masing-masing):
# Bengkulu: db "C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb",
#           sql "SELECT ts_wib AS ts, ghi_now FROM bengkulu_sch.<tabel_10min> ORDER BY ts_wib",
#           lat -3.865, lon 102.312
# Kalbar:   db "C:/Users/ariff/DuckDB_kalbar/kalbar.duckdb", lat 0.075, lon 109.191, meridian 105
# Jambi:    db "C:/Users/ariff/DuckDB_jambi/jambi.duckdb",   lat -1.583, lon 103.667, meridian 105
# ===================================================================

GRID_MIN = 10           # resolusi 10 menit
HIST_STEPS = 18         # riwayat 3 jam = 18 langkah
LEAD_STEPS = 6          # t+60 = 6 langkah
SUN_MIN_DEG = 5.0
GHI_LO, GHI_HI = 0.0, 1400.0

def solar_elev_deg(ts, lat, lon, meridian):
    idx = pd.DatetimeIndex(ts); doy = idx.dayofyear.values.astype(float)
    h = idx.hour.values.astype(float) + idx.minute.values.astype(float)/60.0
    decl = 23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0))
    ha = (h + 4.0*(lon-meridian)/60.0 - 12.0)*15.0
    se = (np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(decl))
          + np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(decl))*np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(se, -1, 1)))

def main():
    c = CONFIG
    print(f"=== HITUNG ANCHOR VALID §2.3 — {c['lokasi']} ===")
    con = duckdb.connect(c["db_path"], read_only=True)
    df = con.execute(c["sql"]).fetchdf(); con.close()
    df["ts"] = pd.to_datetime(df["ts"]); df = df.sort_values("ts").reset_index(drop=True)
    print(f"baris mentah 10-menit : {len(df):,}  ({df['ts'].min().date()}..{df['ts'].max().date()})")

    # riwayat 3-jam kontinu: LAG 18 ada DAN gap tepat 10 menit di seluruh 18 langkah
    dt = df["ts"].diff().dt.total_seconds().div(60.0)
    cont_hist = pd.Series(True, index=df.index)
    for k in range(1, HIST_STEPS+1):
        cont_hist &= (dt.shift(k-1) == GRID_MIN) if k > 1 else (dt == GRID_MIN)
    # GHI(t+60) via LEAD 6 pd grid kontinu
    lead_dt_ok = pd.Series(True, index=df.index)
    for k in range(1, LEAD_STEPS+1):
        lead_dt_ok &= (dt.shift(-k) == GRID_MIN)
    ghi_t60 = df["ghi_now"].shift(-LEAD_STEPS)

    elev_anchor = solar_elev_deg(df["ts"], c["lat"], c["lon"], c["meridian"])
    elev_t60    = solar_elev_deg(df["ts"] + pd.Timedelta(minutes=GRID_MIN*LEAD_STEPS), c["lat"], c["lon"], c["meridian"])

    valid = (
        cont_hist & lead_dt_ok
        & (elev_anchor > SUN_MIN_DEG) & (elev_t60 > SUN_MIN_DEG)
        & df["ghi_now"].between(GHI_LO, GHI_HI)
        & ghi_t60.between(GHI_LO, GHI_HI)
    )
    dfa = df[valid].copy()
    yr = dfa["ts"].dt.year
    tr = (dfa["ts"] < pd.Timestamp("2024-01-01")).sum()
    va = ((dfa["ts"] >= pd.Timestamp("2024-01-01")) & (dfa["ts"] < pd.Timestamp("2025-01-01"))).sum()
    te = (dfa["ts"] >= pd.Timestamp("2025-01-01")).sum()
    print(f"ANCHOR VALID §2.3     : {len(dfa):,}")
    print(f"  train (<2024)       : {tr:,}")
    print(f"  val   (2024)        : {va:,}")
    print(f"  test  (2025)        : {te:,}")
    print(f"per tahun: {dict(yr.value_counts().sort_index())}")

if __name__ == "__main__":
    main()
