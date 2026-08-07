#!/usr/bin/env python3
"""
Audit Tabel 1 — Bengkulu anchor §2.3
Verifikasi langsung ke database; JANGAN menyalin dari catatan lama.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH   = "C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb"
LAT       = -3.8607
LON       = 102.3381
MERIDIAN  = 105.0

# ── helpers ──────────────────────────────────────────────────────────────────

def astro_elev(ts_series: pd.Series) -> np.ndarray:
    """Elevasi matahari astronomis (Cooper 1969) — identik pipeline lain."""
    ts  = pd.DatetimeIndex(ts_series)
    doy = ts.dayofyear.values.astype(float)
    h   = ts.hour.values + ts.minute.values / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
    ha   = (h + 4 * (LON - MERIDIAN) / 60 - 12) * 15
    sin_e = (np.sin(np.deg2rad(LAT)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


# ── 1. Muat data mentah 2021–2025 ────────────────────────────────────────────

print("=== 1. MUAT DATA MENTAH 10-MENIT ===")
con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{DB_PATH}' AS bdb (READ_ONLY)")

raw = con.execute("""
    SELECT ts_wib, asrs_ghi_w_m2 AS ghi
    FROM bdb.bengkulu_sch.bengkulu_master_10min_quality_final
    WHERE YEAR(ts_wib) BETWEEN 2021 AND 2025
    ORDER BY ts_wib
""").fetchdf()

raw["ts_wib"] = pd.to_datetime(raw["ts_wib"])
raw["ghi"]    = pd.to_numeric(raw["ghi"], errors="coerce")

n_raw = len(raw)
print(f"  Raw record 2021–2025: {n_raw:,}")

# ── 2. Hitung elevasi astronomis: anchor & t+60 ───────────────────────────────

print("\n=== 2. HITUNG ELEVASI ASTRONOMIS ===")
raw["elev_anchor"] = astro_elev(raw["ts_wib"])
raw["elev_t60"]    = astro_elev(raw["ts_wib"] + pd.Timedelta(minutes=60))

# ── 3. Filter GHI bounds anchor ───────────────────────────────────────────────

print("\n=== 3. FILTER GHI [0,1400] & ELEV >5° anchor & t+60 ===")
# §2.3: "GHI(anchor) dan GHI(t+60) keduanya dalam [0, 1400]"
# Lookup GHI(t+60) = 6 langkah ke depan (index shift by 6)
raw["ghi_t60"] = raw["ghi"].shift(-6)

f_ghi     = raw["ghi"].between(0, 1400, inclusive="both")
f_ghi_t60 = raw["ghi_t60"].between(0, 1400, inclusive="both")
f_elev_a  = raw["elev_anchor"] > 5.0
f_elev_t60= raw["elev_t60"]    > 5.0
mask_sun  = f_ghi & f_ghi_t60 & f_elev_a & f_elev_t60

# ── 4. Filter riwayat 3-jam kontinu (18 langkah sebelumnya gap-free) ─────────

print("\n=== 4. FILTER RIWAYAT 3-JAM KONTINU ===")
# Untuk setiap baris i, periksa apakah ts_wib[i-k] == ts_wib[i] - 10k menit, k=1..18
# Cara efisien: bangun kolom diff bernomor, semua harus = 10 menit

raw_sorted = raw.reset_index(drop=True)
ts = raw_sorted["ts_wib"]

# diff berurutan (bernilai 10 menit jika kontinu)
step = pd.Timedelta(minutes=10)
diffs = ts.diff().fillna(pd.Timedelta(0))           # baris pertama = 0

# sliding window: 18 langkah ke belakang harus semua 10 menit
# Gunakan rolling minimum dari diffs — kalau ada gap dalam 18 langkah, min < 10 min ATAU > 10 min
# Konversi ke detik untuk rolling
diff_sec = diffs.dt.total_seconds()

# Window 18 (termasuk baris itu sendiri; kita lihat 18 step sebelumnya)
# rolling(19) karena mencakup row sekarang + 18 sebelumnya
win_min = diff_sec.rolling(window=19, min_periods=19).min().fillna(0)
win_max = diff_sec.rolling(window=19, min_periods=19).max().fillna(0)

# Semua diff dalam window harus persis 600 detik (kecuali baris paling awal window = NaN diganti 0)
# Baris 0-17 tidak punya history 18 langkah → otomatis gagal (win_min = 0 ≠ 600)
f_hist = (win_min == 600) & (win_max == 600)

# ── 5. Riwayat 3-jam: GHI anchor juga harus valid — periksa GHI dalam 18 langkah sebelumnya ──
# Ini sesuai 'has_continuous_3h_history' di pipeline yang mensyaratkan GHI tersedia
# Untuk §2.3 kita cukup syaratkan timestamp kontinu (tanpa syarat GHI valid di setiap step-nya)
# karena angka referensi (109,196) kemungkinan menggunakan definisi timestamp-only
# Kita akan laporkan keduanya.

mask_final_ts_only = mask_sun & f_hist

# Versi strict (GHI [0,1400] di semua 18 langkah sebelumnya — mengikuti pipeline):
# Gunakan rolling sum of valid-ghi flags
ghi_ok = raw_sorted["ghi"].between(0, 1400).astype(int)
win_ghi_ok = ghi_ok.rolling(window=19, min_periods=19).sum().fillna(0)
f_hist_ghi = f_hist & (win_ghi_ok == 19)   # semua 19 titik (incl. anchor) ghi valid
mask_final_strict = mask_sun & f_hist_ghi

n_anchor_ts   = mask_final_ts_only.sum()
n_anchor_strict = mask_final_strict.sum()

print(f"  Anchor §2.3 (timestamp-only 3h history)   : {n_anchor_ts:,}")
print(f"  Anchor §2.3 (strict: GHI valid 18-step)   : {n_anchor_strict:,}")
print(f"  Referensi angka Tabel 1                    : 109,196")

# ── 6. Split per tahun ────────────────────────────────────────────────────────

print("\n=== 5. SPLIT TRAIN/VAL/TEST & PER TAHUN ===")
df_anchor_ts  = raw_sorted[mask_final_ts_only].copy()
df_anchor_str = raw_sorted[mask_final_strict].copy()

for label, df in [("ts-only", df_anchor_ts), ("strict", df_anchor_str)]:
    year = df["ts_wib"].dt.year
    tr  = (df["ts_wib"] < pd.Timestamp("2024-01-01")).sum()
    va  = ((df["ts_wib"] >= pd.Timestamp("2024-01-01")) & (df["ts_wib"] < pd.Timestamp("2025-01-01"))).sum()
    te  = (df["ts_wib"] >= pd.Timestamp("2025-01-01")).sum()
    per_yr = year.value_counts().sort_index()
    print(f"\n  [{label}] total={len(df):,}  train={tr:,}  val={va:,}  test={te:,}")
    print(f"  per tahun: {per_yr.to_dict()}")

# ── 7. Uji grid 24-jam (ada baris malam?) ─────────────────────────────────────

print("\n=== 6. UJI GRID 24-JAM (BARIS MALAM JAM 0–5) ===")
hr_all   = raw_sorted["ts_wib"].dt.hour
hr_night = hr_all.between(0, 5)
n_night_raw  = hr_night.sum()
n_night_anch = df_anchor_ts["ts_wib"].dt.hour.between(0, 5).sum()
print(f"  Malam jam 0–5 di raw           : {n_night_raw:,}")
print(f"  Malam jam 0–5 di anchor §2.3   : {n_night_anch:,}")
print(f"  Referensi uji cepat grid       : ~65,829 (semua jam, BUKAN hanya malam)")

# Interpretasi: ~65,829 = anchor siang saja (elev>5° matahari, malam < 1%)
night_ratio = n_night_anch / max(len(df_anchor_ts), 1)
if night_ratio < 0.01:
    print("  ✓ Hampir nol baris malam di anchor — wajar (filter elev>5° membuang malam)")
else:
    print(f"  FLAG: {night_ratio:.1%} baris malam di anchor — periksa elev threshold")

# Koreksi interpretasi referensi:
# "uji cepat grid 24-jam: baris malam jam 0–5 harus ada; referensi ~65,829"
# Kemungkinan ~65,829 = jumlah anchor TRAIN saja (bukan grid 24-jam)
# karena train < 2024 = 3 tahun (2021-2023) ≈ 64,863 referensi.
# Kita laporkan apa adanya.

# ── 8. Konsistensi pipeline: view ghi_forecast_1h_train_3h_rollback_2021_2025 ─

print("\n=== 7. KONSISTENSI PIPELINE (VIEW TRAINING) ===")
try:
    # Query view langsung dengan filter yang dipakai R1/R8
    pipe_q = con.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN ts_wib  < '2024-01-01' THEN 1 ELSE 0 END) AS train,
            SUM(CASE WHEN ts_wib >= '2024-01-01' AND ts_wib < '2025-01-01' THEN 1 ELSE 0 END) AS val,
            SUM(CASE WHEN ts_wib >= '2025-01-01' THEN 1 ELSE 0 END) AS test
        FROM bdb.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
        WHERE is_model_ready = 1
          AND has_continuous_3h_history = 1
          AND ghi_now BETWEEN 0 AND 1400
    """).fetchdf()
    p = pipe_q.iloc[0]
    print(f"  Pipeline view (is_model_ready + 3h + GHI):")
    print(f"    total={int(p['total']):,}  train={int(p['train']):,}  val={int(p['val']):,}  test={int(p['test']):,}")

    # Sama + filter elev_t60 > 5° (persis kondisi skrip R1/R8 — sun_gt5_t60)
    # elev_t60 dihitung dari solar_elev_deg + 6 langkah ke depan → tidak tersimpan di view
    # Kita estimasi dari anchor: berapa % baris view yang memenuhi elev_t60 > 5°
    view_df = con.execute("""
        SELECT ts_wib, ghi_now, solar_elev_deg
        FROM bdb.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
        WHERE is_model_ready = 1
          AND has_continuous_3h_history = 1
          AND ghi_now BETWEEN 0 AND 1400
        ORDER BY ts_wib
    """).fetchdf()
    view_df["ts_wib"] = pd.to_datetime(view_df["ts_wib"])
    view_df["elev_t60_calc"] = astro_elev(view_df["ts_wib"] + pd.Timedelta(minutes=60))
    n_with_elt60 = (view_df["elev_t60_calc"] > 5.0).sum()
    n_no_elt60   = len(view_df) - n_with_elt60
    print(f"\n  Pipeline + filter elev_t60>5° (astronomis R1/R8 runtime):")
    print(f"    total={n_with_elt60:,}  (drop vs no-elev-filter: {n_no_elt60:,})")
    pipe_sun = view_df[view_df["elev_t60_calc"] > 5.0].copy()
    yr = pipe_sun["ts_wib"].dt.year
    tr_p = (pipe_sun["ts_wib"] < pd.Timestamp("2024-01-01")).sum()
    va_p = ((pipe_sun["ts_wib"] >= pd.Timestamp("2024-01-01")) & (pipe_sun["ts_wib"] < pd.Timestamp("2025-01-01"))).sum()
    te_p = (pipe_sun["ts_wib"] >= pd.Timestamp("2025-01-01")).sum()
    per_yr_p = yr.value_counts().sort_index().to_dict()
    print(f"    train={tr_p:,}  val={va_p:,}  test={te_p:,}")
    print(f"    per tahun: {per_yr_p}")

    # Perbedaan pipeline vs §2.3 ts-only
    diff_total = n_with_elt60 - n_anchor_ts
    print(f"\n  Selisih (pipeline+elev) vs §2.3 ts-only: {diff_total:+,}")
    if abs(diff_total) / max(n_anchor_ts, 1) > 0.01:
        print("  ⚠ Selisih >1% — ada perbedaan definisi filter, lihat catatan.")
    else:
        print("  ✓ Selisih <1% — pipeline dan §2.3 cukup setara.")

except Exception as e:
    print(f"  ERROR mengakses view: {e}")

# ── 9. Ringkasan ──────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RINGKASAN AUDIT TABEL 1 — BENGKULU")
print("="*60)
ref = 109196
best = n_anchor_ts  # gunakan ts-only karena referensi kemungkinan tidak mewajibkan GHI valid 18-step
diff_pct = (best - ref) / ref * 100
verdict = "MATCH" if abs(diff_pct) <= 1.0 else f"DEVIASI {diff_pct:+.2f}%"
print(f"  Raw 10-min (2021–2025)         : {n_raw:,}")
print(f"  Anchor §2.3 ts-only            : {n_anchor_ts:,}   <- {verdict} vs referensi")
print(f"  Anchor §2.3 strict (GHI 18-step): {n_anchor_strict:,}")
print(f"  Referensi Tabel 1              : {ref:,}")
print(f"  Deviasi                        : {diff_pct:+.2f}%")
print()
if abs(diff_pct) <= 1.0:
    print("  VERDICT: 109,196 VALID untuk Tabel 1 ✓")
else:
    print(f"  VERDICT: Perlu koreksi → angka final = {best:,}")

con.close()
