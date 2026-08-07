#!/usr/bin/env python3
"""
Audit Tabel 1 Anchor Basis §2.3 — Kalbar
Verifikasi homogenitas filter lintas-lokasi dan cek konsistensi training data.

Konteks:
  - Tabel 1 harus menggunakan "valid forecast anchors" §2.3 identik di semua lokasi
  - Di Banten, Kalbar dihitung ulang hasilnya: 90,579 anchor (vs angka sebelumnya?)
  - Perlu confirm di environment Kalbar + test sensitivitas koordinat

Run:
    & "C:\Program Files\Python39\python.exe" audit_anchor_kalbar_table1.py
"""

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path(r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db")
OUTPUT_DIR = Path(r"C:\Users\ariff\DuckDB_kalbar\audit_table1_kalbar")
OUTPUT_DIR.mkdir(exist_ok=True)

# Koordinat: ada dua versi
LAT_USED = -0.0356      # Yang dipakai di build-script Kalbar
LON_USED = 109.3384
LAT_ACTUAL = 0.07489    # Koordinat stasiun sebenarnya (Staklim Kalbar/Mempawah)
LON_ACTUAL = 109.1905
MERIDIAN = 105.0        # WIB

# Filter §2.3
YEARS = [2022, 2023, 2024, 2025]
HISTORY_STEPS = 18      # 3 jam = 18 × 10-min steps sebelumnya
ELEV_THRESHOLD = 5.0    # Matahari > 5° di anchor dan t+60
GHI_MIN, GHI_MAX = 0.0, 1400.0
T_PLUS_60_STEPS = 6     # 60 menit ke depan

# Referensi dari Banten (untuk dibandingkan)
REFERENCE_ANCHOR = 90579
REFERENCE_TRAIN = 45260
REFERENCE_VAL = 22692
REFERENCE_TEST = 22627
REFERENCE_PER_YEAR = {2022: 22630, 2023: 22630, 2024: 22692, 2025: 22627}

# ---------------------------------------------------------------------------
# Fungsi
# ---------------------------------------------------------------------------

def solar_elevation_deg(timestamps, lat, lon, meridian=105.0):
    """
    Hitung elevasi matahari dengan formula Cooper 1969.
    Semua input/output dalam derajat.
    """
    idx = pd.DatetimeIndex(timestamps)
    doy = idx.dayofyear.values.astype(float)
    h = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0

    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    ha = (h + 4.0 * (lon - meridian) / 60.0 - 12.0) * 15.0

    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl)) +
             np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))

    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def check_continuous_3h_history(ts_series, history_steps=18):
    """
    Verifikasi setiap baris punya 18 baris sebelumnya tanpa gap (10-menit intervals).
    Input: pandas Series dengan datetime values
    Return: boolean array (True = valid, False = missing history)

    Note: 18 steps = 17 intervals × 10 min = 170 minutes (dari step i-18 ke step i)
    """
    valid = np.zeros(len(ts_series), dtype=bool)

    for i in range(len(ts_series)):
        if i < history_steps:
            # Need at least 18 previous rows
            valid[i] = False
        else:
            # Check apakah gap antara i-18 dan i adalah ~170 menit (17 × 10min)
            # Using iloc to safely access by position
            time_diff = pd.Timestamp(ts_series.iloc[i]) - pd.Timestamp(ts_series.iloc[i-history_steps])
            time_diff_min = time_diff.total_seconds() / 60.0

            # Toleransi: 165-175 menit (17 intervals × 10 ± 5)
            if 165.0 <= time_diff_min <= 175.0:
                valid[i] = True
            else:
                valid[i] = False

    return valid


def apply_filter_section_2_3(df, lat, lon, history_steps=18, elev_thresh=5.0,
                               ghi_min=0.0, ghi_max=1400.0, t_plus_60_steps=6):
    """
    Terapkan filter §2.3 lengkap.
    Input: df dengan kolom timestamp_wib, ghi_final (sorted, gap-free dalam periode)
    Output: df yang sudah difilter
    """
    print(f"\n{'='*70}")
    print(f"Applying Section 2.3 Filter (lat={lat}, lon={lon})")
    print(f"{'='*70}")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp_wib'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    n_raw = len(df)
    print(f"Raw data: {n_raw:,} rows")

    # 1. Verifikasi riwayat 3-jam kontinu
    valid_history = check_continuous_3h_history(df['timestamp'], history_steps)
    df['has_history'] = valid_history
    n_after_history = valid_history.sum()
    print(f"  After 3-hour history filter: {n_after_history:,} rows")

    # 2. Hitung elevasi di anchor dan t+60
    df['elev_anchor'] = solar_elevation_deg(df['timestamp'], lat, lon, MERIDIAN)
    df['timestamp_t60'] = df['timestamp'] + pd.Timedelta(minutes=60)
    df['elev_t60'] = solar_elevation_deg(df['timestamp_t60'], lat, lon, MERIDIAN)

    # 3. Filter elevasi > threshold di anchor dan t+60
    valid_elev = (df['elev_anchor'] > elev_thresh) & (df['elev_t60'] > elev_thresh)
    print(f"  After solar elevation > {elev_thresh}° at anchor & t+60: {valid_elev.sum():,} rows")

    # 4. Filter GHI di anchor dalam range
    valid_ghi_anchor = (df['ghi_final'] >= ghi_min) & (df['ghi_final'] <= ghi_max)
    print(f"  After GHI(anchor) in [{ghi_min}, {ghi_max}]: {valid_ghi_anchor.sum():,} rows")

    # 5. Filter GHI di t+60 dalam range (perlu shift forward)
    df['ghi_t60'] = df['ghi_final'].shift(-t_plus_60_steps)  # 6 steps = 60 min
    valid_ghi_t60 = (df['ghi_t60'] >= ghi_min) & (df['ghi_t60'] <= ghi_max)
    print(f"  After GHI(t+60) in [{ghi_min}, {ghi_max}]: {valid_ghi_t60.sum():,} rows")

    # Kombinasi semua filter
    valid_all = valid_history & valid_elev & valid_ghi_anchor & valid_ghi_t60
    df_filtered = df[valid_all].copy()

    print(f"\nTotal after all §2.3 filters: {len(df_filtered):,} rows")

    return df_filtered


def compute_splits(df_filtered, train_end="2024-01-01", valid_end="2025-01-01"):
    """Pisah train/val/test berdasarkan timestamp."""
    df = df_filtered.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp_wib'])

    train_mask = df['timestamp'] < pd.Timestamp(train_end)
    valid_mask = (df['timestamp'] >= pd.Timestamp(train_end)) & (df['timestamp'] < pd.Timestamp(valid_end))
    test_mask = df['timestamp'] >= pd.Timestamp(valid_end)

    train = df[train_mask]
    valid = df[valid_mask]
    test = df[test_mask]

    return train, valid, test


def main():
    print("\n" + "="*70)
    print("AUDIT TABEL 1 ANCHOR BASIS §2.3 — KALBAR")
    print("="*70)

    # Load data
    print("\nLoading data from DuckDB Kalbar...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        df_raw = con.execute("""
            SELECT timestamp_wib, ghi_final
            FROM main.solar_kalbar_10m
            WHERE YEAR(timestamp_wib) >= 2022 AND YEAR(timestamp_wib) <= 2025
            ORDER BY timestamp_wib
        """).df()
    except Exception as e:
        print(f"Error loading solar_kalbar_10m: {e}")
        print("Trying alternative table name...")
        df_raw = con.execute("""
            SELECT * FROM information_schema.tables
            WHERE table_name LIKE '%kalbar%' AND table_schema = 'main'
        """).df()
        print("Available tables:")
        print(df_raw)
        con.close()
        return

    con.close()

    print(f"Raw data loaded: {len(df_raw):,} rows")

    # Apply filter §2.3 dengan koordinat yang dipakai
    print("\n" + "-"*70)
    print("TEST 1: Koordinat yang dipakai di Kalbar pipeline")
    print("-"*70)
    df_filtered_used = apply_filter_section_2_3(
        df_raw, LAT_USED, LON_USED,
        history_steps=HISTORY_STEPS, elev_thresh=ELEV_THRESHOLD,
        ghi_min=GHI_MIN, ghi_max=GHI_MAX, t_plus_60_steps=T_PLUS_60_STEPS
    )

    # Compute splits
    train_used, val_used, test_used = compute_splits(df_filtered_used)

    # Per tahun
    df_filtered_used['year'] = pd.to_datetime(df_filtered_used['timestamp_wib']).dt.year
    per_year_used = df_filtered_used.groupby('year').size().to_dict()

    print(f"\nSplits (koordinat used):")
    print(f"  Train (<2024): {len(train_used):,} rows")
    print(f"  Valid (2024):  {len(val_used):,} rows")
    print(f"  Test (2025):   {len(test_used):,} rows")
    print(f"  Total:         {len(df_filtered_used):,} rows")
    print(f"\nPer tahun: {per_year_used}")

    # Test sensitivitas koordinat
    print("\n" + "-"*70)
    print("TEST 2: Sensitivitas koordinat (actual stasiun)")
    print("-"*70)
    df_filtered_actual = apply_filter_section_2_3(
        df_raw, LAT_ACTUAL, LON_ACTUAL,
        history_steps=HISTORY_STEPS, elev_thresh=ELEV_THRESHOLD,
        ghi_min=GHI_MIN, ghi_max=GHI_MAX, t_plus_60_steps=T_PLUS_60_STEPS
    )

    train_actual, val_actual, test_actual = compute_splits(df_filtered_actual)

    delta_total = len(df_filtered_actual) - len(df_filtered_used)
    delta_train = len(train_actual) - len(train_used)
    delta_val = len(val_actual) - len(val_used)
    delta_test = len(test_actual) - len(test_used)

    print(f"\nDifference (actual vs used coords):")
    print(f"  Total: {delta_total:+d} rows")
    print(f"  Train: {delta_train:+d} rows")
    print(f"  Val:   {delta_val:+d} rows")
    print(f"  Test:  {delta_test:+d} rows")

    if abs(delta_total) < 100:
        print("  → Koordinat sensitivity SMALL (difference < 100 rows) ✓")
    else:
        print(f"  → Koordinat sensitivity SIGNIFICANT (difference >= 100 rows) ⚠")

    # Bandingkan dengan referensi
    print("\n" + "-"*70)
    print("TEST 3: Perbandingan dengan Referensi (dari Banten)")
    print("-"*70)

    results = pd.DataFrame([
        {
            "Metric": "Total Anchor §2.3",
            "Kalbar": len(df_filtered_used),
            "Referensi": REFERENCE_ANCHOR,
            "Diff": len(df_filtered_used) - REFERENCE_ANCHOR,
            "Match": "✓" if abs(len(df_filtered_used) - REFERENCE_ANCHOR) < 500 else "✗"
        },
        {
            "Metric": "Train (<2024)",
            "Kalbar": len(train_used),
            "Referensi": REFERENCE_TRAIN,
            "Diff": len(train_used) - REFERENCE_TRAIN,
            "Match": "✓" if abs(len(train_used) - REFERENCE_TRAIN) < 500 else "✗"
        },
        {
            "Metric": "Val (2024)",
            "Kalbar": len(val_used),
            "Referensi": REFERENCE_VAL,
            "Diff": len(val_used) - REFERENCE_VAL,
            "Match": "✓" if abs(len(val_used) - REFERENCE_VAL) < 500 else "✗"
        },
        {
            "Metric": "Test (2025)",
            "Kalbar": len(test_used),
            "Referensi": REFERENCE_TEST,
            "Diff": len(test_used) - REFERENCE_TEST,
            "Match": "✓" if abs(len(test_used) - REFERENCE_TEST) < 500 else "✗"
        },
    ])

    print("\n" + results.to_string(index=False))

    # Konsistensi dengan training data aktual
    print("\n" + "-"*70)
    print("TEST 4: Konsistensi dengan training data aktual di model")
    print("-"*70)
    print("(Checking training_ghi_1h_direct row count vs §2.3 anchor count)")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        n_training_table = con.execute("SELECT COUNT(*) FROM training_ghi_1h_direct").fetchone()[0]
        print(f"  Rows in training_ghi_1h_direct: {n_training_table:,}")
        print(f"  §2.3 anchor (this audit):       {len(df_filtered_used):,}")
        print(f"  Diff: {n_training_table - len(df_filtered_used):+d}")

        if n_training_table == len(df_filtered_used):
            print("  → Konsistensi PERFECT ✓")
        elif abs(n_training_table - len(df_filtered_used)) < 1000:
            print("  → Konsistensi ACCEPTABLE (diff < 1000 rows, likely from different filter version)")
        else:
            print("  → MISMATCH WARNING: Check if training_ghi_1h_direct uses different filter! ⚠")
    except Exception as e:
        print(f"  Could not access training_ghi_1h_direct: {e}")
    finally:
        con.close()

    # Grid 24-jam check (ada data malam hari)
    print("\n" + "-"*70)
    print("TEST 5: Grid 24-jam (ada data malam/dini hari?)")
    print("-"*70)

    df_filtered_used['hour'] = pd.to_datetime(df_filtered_used['timestamp_wib']).dt.hour
    hour_counts = df_filtered_used['hour'].value_counts().sort_index()

    print("\nRow count per jam UTC (setelah filter §2.3):")
    for hr in range(24):
        count = hour_counts.get(hr, 0)
        status = "OK" if count > 0 else "MISSING"
        print(f"  {hr:2d}:00 — {count:6,} rows  [{status}]")

    n_hours_with_data = (hour_counts > 0).sum()
    print(f"\nHours dengan data: {n_hours_with_data}/24")
    if n_hours_with_data == 24:
        print("  → 24-jam coverage COMPLETE ✓")
    elif n_hours_with_data >= 12:
        print(f"  → Partial coverage (expected: mostly siang due to elev>5° filter)")
    else:
        print(f"  → WARNING: Hanya {n_hours_with_data} jam dengan data — cek apakah filter terlalu ketat! ⚠")

    # Save results
    results.to_csv(OUTPUT_DIR / "comparison_with_reference.csv", index=False)
    df_filtered_used.to_csv(OUTPUT_DIR / "kalbar_anchor_filtered.csv", index=False)

    print("\n" + "="*70)
    print("AUDIT COMPLETE")
    print("="*70)
    print(f"Output files saved to: {OUTPUT_DIR}/")
    print(f"  - comparison_with_reference.csv")
    print(f"  - kalbar_anchor_filtered.csv")

    # Summary verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if abs(len(df_filtered_used) - REFERENCE_ANCHOR) < 500:
        print("✓ TABEL 1 ANCHOR UNTUK KALBAR: 90,579 adalah VALID (or close enough)")
        print("  Pergunakan untuk paper dengan confidence tinggi.")
    elif abs(len(df_filtered_used) - REFERENCE_ANCHOR) < 2000:
        print("⚠ TABEL 1 ANCHOR UNTUK KALBAR: Selisih ~500-2000 rows dari referensi")
        print(f"  Angka aktual Kalbar: {len(df_filtered_used):,}")
        print("  Kemungkinan: versi filter sedikit berbeda atau sumber data berbeda.")
        print("  Rekomendasi: Gunakan angka aktual Kalbar untuk paper, dokumentasikan perbedaan.")
    else:
        print("✗ TABEL 1 ANCHOR UNTUK KALBAR: SELISIH SIGNIFIKAN dari referensi!")
        print(f"  Angka aktual Kalbar: {len(df_filtered_used):,}")
        print(f"  Referensi:           {REFERENCE_ANCHOR:,}")
        print(f"  Selisih:             {len(df_filtered_used) - REFERENCE_ANCHOR:+d}")
        print("  ACTION REQUIRED: Investigate filter differences or data source mismatch!")


if __name__ == "__main__":
    main()
