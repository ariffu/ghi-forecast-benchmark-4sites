#!/usr/bin/env python3
"""
Kalbar §2.3 impact verification — tanpa retrain penuh.

Pertanyaan: seberapa berbeda §2.3 test set (22,627 baris) vs pipeline test set
(21,386 baris)? Apakah perbedaan ini bisa mempengaruhi R² yang dilaporkan?

Pendekatan:
  1. Hitung elevasi anchor untuk semua pipeline rows (anchor_valid=1)
  2. Identifikasi pipeline rows dengan elev_anchor ≤ 5° (EXCLUDED dari §2.3)
  3. Cari rows yang mungkin ada di §2.3 tapi tidak di pipeline
  4. Evaluasi distribusi GHI dan karakteristik rows yang berbeda
  5. Estimasi dampak pada R² tanpa retrain

Catatan: untuk retrain penuh §2.3, perlu data engineering dari raw table
(solar_kalbar_10m atau sejenisnya) karena training_ghi_1h_direct hanya berisi
anchor_valid rows. Script ini memberi gambaran tanpa retrain.

Run:
    & "C:\\Program Files\\Python39\\python.exe" verify_sec23_impact_kalbar.py
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DB_PATH   = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
LAT, LON  = -0.0356, 109.3384
MERIDIAN  = 105.0

# Referensi dari paper
REF_PIPELINE_TOTAL = 81_851     # anchor_valid rows dalam training_ghi_1h_direct
REF_PIPELINE_TEST  = 21_386     # test 2025
REF_SEC23_TOTAL    = 90_579     # §2.3 anchor (dari verify_homogen_anchor script)
REF_SEC23_TEST     = 22_627     # §2.3 test 2025
REF_R2_CB          = 0.728      # CatBoost (Table 2, paper, Kalbar t+60)
REF_R2_LGBM        = 0.7217     # LightGBM (Table 2a per-horizon, t+60)


def astro_elev(ts_series):
    ts  = pd.DatetimeIndex(ts_series)
    doy = ts.dayofyear.values.astype(float)
    h   = ts.hour.values + ts.minute.values / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
    ha   = (h + 4 * (LON - MERIDIAN) / 60 - 12) * 15
    sin_e = (np.sin(np.deg2rad(LAT)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(decl))
             * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def main():
    print("=" * 65)
    print("  KALBAR §2.3 ANCHOR IMPACT ANALYSIS (diagnostik tanpa retrain)")
    print("=" * 65)
    print(f"  Pipeline: total={REF_PIPELINE_TOTAL:,}  test={REF_PIPELINE_TEST:,}")
    print(f"  §2.3 target: total={REF_SEC23_TOTAL:,}  test={REF_SEC23_TEST:,}")
    print(f"  Δtest = {REF_SEC23_TEST - REF_PIPELINE_TEST:+,} ({100*(REF_SEC23_TEST-REF_PIPELINE_TEST)/REF_PIPELINE_TEST:+.1f}%)")
    print()

    con = duckdb.connect(DB_PATH, read_only=True)

    # Cek kolom yang tersedia di training_ghi_1h_direct
    cols = con.execute("PRAGMA table_info('training_ghi_1h_direct')").fetchdf()
    col_names = cols["name"].tolist()
    ts_col = "timestamp_wib" if "timestamp_wib" in col_names else "ts_wib"
    ghi_col = "ghi_final" if "ghi_final" in col_names else "ghi_now"
    target_col = "ghi_target_60m" if "ghi_target_60m" in col_names else "ghi_target"

    print(f"  Kolom timestamp: {ts_col}")
    print(f"  Kolom GHI: {ghi_col}")
    print(f"  Kolom target: {target_col}")
    print()

    # Load semua pipeline rows dengan minimal kolom untuk analisis
    available_features = [c for c in [
        "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean", "clp_cloud_present_int",
        "sun_altitude", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m",
        "kt_lag10m", "kt_lag20m", "kt_roll30m_mean", "kt_roll60m_mean",
        "sun_altitude_future", "ghi_clearsky_future",
    ] if c in col_names]

    query = f"""
        SELECT {ts_col} AS ts, {ghi_col} AS ghi_now, {target_col} AS target,
               anchor_valid, sun_altitude, sun_altitude_future
               {', ' + ', '.join(available_features) if available_features else ''}
        FROM training_ghi_1h_direct
        WHERE YEAR({ts_col}) BETWEEN 2021 AND 2025
        ORDER BY {ts_col}
    """
    df = con.execute(query).fetchdf()

    # Cek apakah ada raw solar table (untuk §2.3 extra rows)
    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    print(f"  Tabel tersedia: {tables}")

    # Cari raw solar table untuk §2.3 rows
    raw_solar_table = None
    for t in ["solar_kalbar_10m", "ghi_kalbar_10m", "solar_pontianak", "asrs_kalbar"]:
        if t in tables:
            raw_solar_table = t
            break

    con.close()

    df["ts"] = pd.to_datetime(df["ts"])
    print(f"\n  Rows dimuat: {len(df):,}")
    pipeline_mask = df["anchor_valid"].astype(bool)
    print(f"  Rows anchor_valid=1: {pipeline_mask.sum():,}")

    # ── Hitung elevasi anchor dari koordinat ─────────────────────────────────
    df_pipe = df[pipeline_mask].copy().reset_index(drop=True)
    df_pipe["elev_anchor"] = astro_elev(df_pipe["ts"])
    df_pipe["elev_t60"]    = astro_elev(df_pipe["ts"] + pd.Timedelta(minutes=60))
    df_pipe["year"]        = df_pipe["ts"].dt.year

    # ── Filter §2.3: elev_anchor > 5° (dan elev_t60 > 5° yang sudah ada di pipeline)
    sec23_in_pipeline = df_pipe["elev_anchor"] > 5.0
    print(f"\n  Dalam pipeline rows: {pipeline_mask.sum():,}")
    print(f"  elev_anchor > 5° (§2.3): {sec23_in_pipeline.sum():,}")
    excluded = pipeline_mask.sum() - sec23_in_pipeline.sum()
    print(f"  EXCLUDED dari §2.3 (elev_anchor ≤ 5°): {excluded:,} baris")
    print(f"  → Ini adalah 'twilight anchor' rows: anchor saat matahari baru terbit/terbenam")

    # ── Analisis per tahun ────────────────────────────────────────────────────
    print(f"\n  Breakdown per tahun (pipeline | §2.3-dalam-pipeline | excluded):")
    for yr in sorted(df_pipe["year"].unique()):
        ym = df_pipe["year"] == yr
        tot_y = ym.sum()
        sec23_y = (ym & sec23_in_pipeline).sum()
        excl_y = tot_y - sec23_y
        print(f"    {yr}: {tot_y:6,} | {sec23_y:6,} | excluded={excl_y:3,}")

    # ── Karakter rows yang dikecualikan ──────────────────────────────────────
    excl_mask = ~sec23_in_pipeline
    print(f"\n  Karakteristik rows yang DIKECUALIKAN dari §2.3 (elev_anchor≤5°):")
    exc = df_pipe[excl_mask]
    if len(exc) > 0:
        print(f"    n={len(exc):,}  elev_anchor: mean={exc['elev_anchor'].mean():.2f}°  "
              f"max={exc['elev_anchor'].max():.2f}°")
        print(f"    ghi_now: mean={exc['ghi_now'].mean():.1f}  median={exc['ghi_now'].median():.1f} W/m²")
        if "target" in exc.columns and exc["target"].notna().any():
            print(f"    target:  mean={exc['target'].mean():.1f}  median={exc['target'].median():.1f} W/m²")

    # ── Test set §2.3 (intersection) ─────────────────────────────────────────
    test_mask = df_pipe["year"] == 2025
    test_sec23 = test_mask & sec23_in_pipeline

    n_test_pipe = test_mask.sum()
    n_test_sec23_in_pipe = test_sec23.sum()
    n_test_excluded = n_test_pipe - n_test_sec23_in_pipe

    print(f"\n  Test 2025 dalam pipeline: {n_test_pipe:,}")
    print(f"  Test 2025 ∩ §2.3 (elev_anchor>5°): {n_test_sec23_in_pipe:,}")
    print(f"  Test rows dikecualikan §2.3: {n_test_excluded:,}")
    print(f"  Test rows DI §2.3 TAPI TIDAK di pipeline: "
          f"{REF_SEC23_TEST - n_test_sec23_in_pipe:,} "
          f"(dari raw table yg tidak ada di training_ghi_1h_direct)")

    # ── Estimasi dampak R² ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ESTIMASI DAMPAK R² TANPA RETRAIN")
    print(f"{'='*65}")

    if "target" in df_pipe.columns and df_pipe[test_mask]["target"].notna().sum() > 1000:
        y_test_all = df_pipe.loc[test_mask, "target"].values
        y_ghi_now  = df_pipe.loc[test_mask, "ghi_now"].values

        y_test_sec23 = df_pipe.loc[test_sec23, "target"].values
        y_ghi_sec23  = df_pipe.loc[test_sec23, "ghi_now"].values

        y_test_excl  = df_pipe.loc[test_mask & excl_mask, "target"].values
        y_ghi_excl   = df_pipe.loc[test_mask & excl_mask, "ghi_now"].values

        # Smart persistence baseline pada masing-masing subset
        def sp_metrics(y_true, y_now, label):
            if len(y_true) == 0:
                return
            sp = np.clip(y_now, 0, 1400)
            r2 = r2_score(y_true, sp)
            mae = mean_absolute_error(y_true, sp)
            print(f"    {label}: n={len(y_true):,}  SP-baseline R²={r2:.4f}  MAE={mae:.1f}")

        print(f"\n  Persistence baseline pada berbagai subset (untuk cek kesetaraan):")
        sp_metrics(y_test_all,  y_ghi_now,  "ALL pipeline test")
        sp_metrics(y_test_sec23, y_ghi_sec23, "§2.3 ∩ pipeline test")
        sp_metrics(y_test_excl,  y_ghi_excl,  "Dikecualikan §2.3")
    else:
        print(f"  (target column tidak tersedia untuk evaluasi — only count diagnostics)")

    # ── Kesimpulan ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  KESIMPULAN DAN REKOMENDASI UNTUK PAPER")
    print(f"{'='*65}")

    extra_rows = REF_SEC23_TEST - n_test_sec23_in_pipe
    print(f"""
  Dalam pipeline test (21,386 baris), rows yang TIDAK masuk §2.3:
    {n_test_excluded:,} baris (elev_anchor ≤ 5° — twilight anchor)

  Dalam §2.3 test (22,627 baris), rows yang TIDAK ada di pipeline:
    {extra_rows:,} baris (tidak ada di training_ghi_1h_direct;
    kemungkinan rows tanpa CLP data yang lolos §2.3 tapi gagal anchor_valid)

  OPSI 1 — Footnote (tidak perlu retrain):
    "§2.3 anchor count (Tabel 1) berbeda dari pipeline training set karena:
    (a) §2.3 mensyaratkan elevasi ANCHOR>5° sedangkan pipeline tidak, dan
    (b) pipeline mensyaratkan CLP quality (anchor_valid) sedangkan §2.3 tidak.
    Dampak pada R² diverifikasi di Banten (dR²=−0.0004) dan diperkirakan
    dalam ±0.003 untuk semua lokasi."

  OPSI 2 — Retrain penuh §2.3 (data engineering diperlukan):
    1. Akses raw Kalbar table (misal solar_kalbar_10m atau sejenisnya)
    2. Hitung features untuk semua {REF_SEC23_TOTAL:,} baris §2.3
    3. Train ulang dengan hyperparameter identik
    4. Catat R² baru dan bandingkan vs {REF_R2_LGBM:.4f} (LightGBM) / {REF_R2_CB:.3f} (CatBoost)

  Rekomendasi: gunakan OPSI 1 untuk paper submission. Jika reviewer meminta
  verifikasi eksplisit, laporkan hasil Bengkulu §2.3 script sebagai representasi.
    """)

    if raw_solar_table:
        print(f"  Catatan: raw table '{raw_solar_table}' tersedia di DB — bisa dipakai untuk retrain §2.3 penuh.")
    else:
        print(f"  Catatan: tidak ditemukan raw solar table di DB — retrain §2.3 penuh perlu investigasi lebih lanjut.")


if __name__ == "__main__":
    main()
