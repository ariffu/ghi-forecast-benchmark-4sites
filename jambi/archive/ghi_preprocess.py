"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         GHI Forecasting — Preprocessing Module                             ║
║         Dataset: jambi.jambi_sch.jambi_obs_combined                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Data   : 2022–2025 (2021 dikecualikan — 64.6% ptm_gap_filled)             ║
║  Split  : Train 2022–2023 | Val 2024 | Test 2025                           ║
║                                                                             ║
║  Masalah yang ditangani (berdasarkan eksplorasi data):                      ║
║                                                                             ║
║  [M1] Quality weight GHI (2022–2025)                                       ║
║       cloud_enhancement=1.0, capped_kt=0.5, ptm_gap_filled=0.3            ║
║                                                                             ║
║  [M2] AOD: Missing Not At Random (MNAR)                                     ║
║       → Saat AOD hilang: COT 14 vs 5, Kt 0.64 vs 0.82, GHI 310 vs 498    ║
║       → ffill sederhana BIAS. Strategi: flag biner + median by sun_alt bin ║
║                                                                             ║
║  [M3] Rainfall: ffill maks 3 step (30 menit), sisa fill 0                  ║
║       (masalah 2021 sudah tidak relevan)                                    ║
║                                                                             ║
║  [M4] kt > 1.5: clip (cloud enhancement ekstrem / noise sensor)            ║
║                                                                             ║
║  [M5] Gap 11–30 menit: threshold diperketat ke 12 menit                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Output: DataFrame bersih yang siap dipakai oleh ghi_forecast_pipeline.py
"""

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
PREP_CFG = {
    "year_start":           2022,   # exclude 2021 (64.6% ptm_gap_filled)
    "min_sun_alt":          3.0,    # elevasi matahari minimum
    "max_gap_min":         12,      # gap lebih dari ini → segmen baru (lebih ketat dari 15)
    "kt_clip_max":          1.5,    # kt > ini → clip (cloud enhancement ekstrem)
    "kt_clip_min":          0.0,    # kt negatif → clip ke 0
    "ghi_clip_min":         0.0,    # GHI negatif → clip ke 0
    "rainfall_ffill_max":   3,      # max langkah ffill untuk rainfall (30 menit)
    "aod_sun_alt_bins":     [3, 15, 30, 45, 60, 90],  # bin untuk median imputation AOD
    "train_years":          [2022, 2023],
    "val_years":            [2024],
    "test_years":           [2025],
    "quality_weight": {
        "good":             1.0,
        "cloud_enhancement": 1.0,   # tetap valid, bukan error sensor
        "capped_kt_gt_1.5": 0.5,    # kurang dipercaya
        "ptm_gap_filled":   0.3,    # interpolasi (sedikit di 2022–2025)
        "clamped_negative": 0.2,
        "fallback":         0.1,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_raw(token: str = None) -> pd.DataFrame:
    """Load data mentah dari MotherDuck."""
    if token:
        con = duckdb.connect(f"md:?motherduck_token={token}")
    else:
        con = duckdb.connect("md:")

    sql = f"""
    SELECT
        timestamp_wib,
        sun_altitude, sun_azimuth, optical_air_mass,
        ghi_consolidated, ghi_clearsky, kt_consolidated,
        dhi_consolidated, dni_consolidated,
        ghi_quality_flag,
        temp_air_c, dewpoint_c, rh_pct, vapour_pressure_hpa,
        pressure_hpa, wind_speed_ms, wind_dir_deg,
        rainfall_mm, cloud_cover_oktas, cloud_cover_fraction,
        AOD_440nm, AOD_500nm, AOD_675nm, AOD_870nm,
        angstrom_exp_440_870, precipitable_water_cm,
        beam_transmittance_500nm, aod_best, aod_550nm,
        fine_mode_aot_proxy, coarse_mode_aot_proxy,
        sat_cloud_present, cloud_optical_thickness, clot_std,
        cloud_top_temp_k, cloud_top_height_m, cloud_eff_radius_um
    FROM jambi.jambi_sch.jambi_obs_combined
    WHERE sun_altitude >= 3.0
      AND YEAR(timestamp_wib) >= {PREP_CFG['year_start']}
    ORDER BY timestamp_wib
    """

    df = con.execute(sql).df()
    con.close()
    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    df = df.sort_values("timestamp_wib").reset_index(drop=True)
    print(f"[RAW] {len(df):,} baris | {df['timestamp_wib'].min().date()} → {df['timestamp_wib'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. [M1] BOBOT KUALITAS GHI
# ══════════════════════════════════════════════════════════════════════════════
def add_quality_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah kolom 'sample_weight' berdasarkan ghi_quality_flag.
    Digunakan oleh LGBM (sample_weight) dan torch (weighted sampler opsional).
    Data 2022–2025: ptm_gap_filled sangat sedikit (<1.1%), tidak perlu perlakuan khusus.
    """
    df = df.copy()
    weight_map = PREP_CFG["quality_weight"]
    df["sample_weight"] = df["ghi_quality_flag"].map(weight_map).fillna(0.1)

    breakdown = df.groupby("ghi_quality_flag")["sample_weight"].count()
    print(f"  [M1] Quality flags: {breakdown.to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. [M2] IMPUTASI AOD (MNAR — Missing Not At Random)
# ══════════════════════════════════════════════════════════════════════════════
def impute_aod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategi imputasi AOD yang sadar bias MNAR:

    1. Tambah binary flag 'aod_is_available' (0/1) — model bisa belajar konteks ini
    2. Imputasi menggunakan median per (bulan, bin_sun_altitude) dari training set
       (bukan ffill — karena nilai terakhir AOD kemungkinan dari kondisi cerah,
        sedangkan AOD hilang justru saat awan tebal)
    3. COT, CTT, CER: ffill pendek (≤3 step = 30 menit) karena data satelit
       memiliki resolusi temporal lebih rendah — masih valid untuk interpolasi pendek

    Catatan: scaler harus fit HANYA pada training data. Flag ini harus dibuat
    SEBELUM split, tapi median reference dihitung dari training years.
    """
    df = df.copy()

    # ── Flag biner ketersediaan AOD ───────────────────────
    aod_cols = ["aod_best", "aod_550nm", "AOD_500nm", "AOD_440nm",
                "AOD_675nm", "AOD_870nm", "angstrom_exp_440_870",
                "precipitable_water_cm", "beam_transmittance_500nm",
                "fine_mode_aot_proxy", "coarse_mode_aot_proxy"]
    df["aod_is_available"]  = (~df["aod_best"].isna()).astype(float)

    # Flag terpisah untuk satellite cloud properties
    df["clp_is_available"]  = (~df["cloud_optical_thickness"].isna()).astype(float)

    # ── Bin sun_altitude untuk median reference ───────────
    bins = PREP_CFG["aod_sun_alt_bins"]
    df["sun_alt_bin"] = pd.cut(df["sun_altitude"], bins=bins, labels=False)

    # ── Hitung median AOD dari TRAIN YEARS (2022-2023) ────
    # Saat digunakan di test, referensi tetap dari train — tidak ada leakage
    train_mask = df["timestamp_wib"].dt.year.isin(PREP_CFG["train_years"])
    df["month"] = df["timestamp_wib"].dt.month

    train_ref = (df[train_mask & df["aod_best"].notna()]
                 .groupby(["month", "sun_alt_bin"])["aod_best"]
                 .median()
                 .reset_index()
                 .rename(columns={"aod_best": "aod_median_ref"}))

    df = df.merge(train_ref, on=["month", "sun_alt_bin"], how="left")

    # ── Imputasi: pakai median_ref; jika tidak ada, pakai median global train ──
    global_median_aod = df.loc[train_mask & df["aod_best"].notna(), "aod_best"].median()
    df["aod_best_clean"] = df["aod_best"].fillna(
        df["aod_median_ref"].fillna(global_median_aod)
    )

    # Derivatif AOD: isi dengan rasio dari aod_best_clean
    ratio_500_ref = (df["AOD_500nm"] / df["aod_best"]).median()
    ratio_ang_ref = df["angstrom_exp_440_870"].median()
    ratio_pwat_ref = df["precipitable_water_cm"].median()
    ratio_beam_ref = df["beam_transmittance_500nm"].median()
    ratio_fine_ref = (df["fine_mode_aot_proxy"] / df["aod_best"].replace(0, np.nan)).median()
    ratio_coarse_ref = (df["coarse_mode_aot_proxy"] / df["aod_best"].replace(0, np.nan)).median()

    df["aod_550nm_clean"]  = df["aod_550nm"].fillna(df["aod_best_clean"])
    df["aod_500nm_clean"]  = df["AOD_500nm"].fillna(df["aod_best_clean"] * ratio_500_ref)
    df["angstrom_clean"]   = df["angstrom_exp_440_870"].fillna(ratio_ang_ref)
    df["pwat_clean"]       = df["precipitable_water_cm"].fillna(ratio_pwat_ref)
    df["beam_clean"]       = df["beam_transmittance_500nm"].fillna(ratio_beam_ref)
    df["fine_aot_clean"]   = df["fine_mode_aot_proxy"].fillna(
                                 df["aod_best_clean"] * ratio_fine_ref)
    df["coarse_aot_clean"] = df["coarse_mode_aot_proxy"].fillna(
                                 df["aod_best_clean"] * ratio_coarse_ref)

    # ── Cloud Properties: ffill pendek ────────────────────
    # Awan bergerak perlahan → ffill ≤3 step masih reasonable
    clp_cols = ["cloud_optical_thickness", "cloud_top_temp_k",
                "cloud_top_height_m", "cloud_eff_radius_um", "clot_std"]
    for col in clp_cols:
        df[f"{col}_clean"] = (df[col]
                               .ffill(limit=3)    # maks 30 menit
                               .bfill(limit=1))   # 1 step ke belakang untuk awal hari

    # sat_cloud_present: ffill + default False (tidak ada awan terdeteksi)
    df["sat_cloud_present_clean"] = (df["sat_cloud_present"]
                                     .ffill(limit=3)
                                     .fillna(False)
                                     .astype(float))

    # Bersihkan kolom bantu
    df = df.drop(columns=["aod_median_ref", "sun_alt_bin", "month"], errors="ignore")

    n_aod_imputed = df["aod_best"].isna().sum()
    print(f"  [M2] AOD imputasi: {n_aod_imputed:,} baris  "
          f"(median_ref by month×sun_alt_bin) + flag 'aod_is_available'")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. [M3] IMPUTASI RAINFALL
# ══════════════════════════════════════════════════════════════════════════════
def impute_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data 2022–2025: rainfall hanya <6% missing (2022) dan hampir 0% sisanya.
    Strategi: ffill maks 3 step (30 menit) lalu fill 0.
    Tambah flag 'rain_is_available' untuk info kualitas ke model.
    """
    df = df.copy()
    df["rain_is_available"] = (~df["rainfall_mm"].isna()).astype(float)

    df["rainfall_mm_clean"] = (
        df["rainfall_mm"]
        .ffill(limit=PREP_CFG["rainfall_ffill_max"])
        .fillna(0.0)
    )

    n_filled = df["rainfall_mm"].isna().sum()
    print(f"  [M3] Rainfall imputasi: {n_filled:,} baris "
          f"(ffill {PREP_CFG['rainfall_ffill_max']}-step + fill 0) + flag 'rain_is_available'")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. [M4] CLIPPING GHI & Kt
# ══════════════════════════════════════════════════════════════════════════════
def clip_solar(df: pd.DataFrame) -> pd.DataFrame:
    """
    GHI negatif → 0 (fisika tidak mungkin).
    kt > 1.5 → clip (cloud enhancement SANGAT ekstrem, kemungkinan noise sensor).
    kt antara 1.0–1.5 → biarkan (cloud enhancement wajar, ≈14% kasus).
    """
    df = df.copy()
    n_neg_ghi = (df["ghi_consolidated"] < 0).sum()
    n_kt_clip = (df["kt_consolidated"] > PREP_CFG["kt_clip_max"]).sum()

    df["ghi_consolidated"] = df["ghi_consolidated"].clip(lower=0)
    df["kt_consolidated"]  = df["kt_consolidated"].clip(
        lower=PREP_CFG["kt_clip_min"],
        upper=PREP_CFG["kt_clip_max"]
    )
    # DHI & DNI juga clip negatif
    for col in ["dhi_consolidated", "dni_consolidated"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    print(f"  [M4] GHI negatif di-clip: {n_neg_ghi:,} | "
          f"kt > {PREP_CFG['kt_clip_max']} di-clip: {n_kt_clip:,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. [M5] FITUR WAKTU & SEGMENTASI
# ══════════════════════════════════════════════════════════════════════════════
def add_features_and_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah fitur waktu siklik dan segmentasi berdasarkan gap waktu.
    Gap threshold: 12 menit (lebih ketat — tangkap gap 11-30 mnt yang ada 270 kasus).
    """
    df = df.copy()

    # ── Fitur waktu ───────────────────────────────────────
    hr   = df["timestamp_wib"].dt.hour + df["timestamp_wib"].dt.minute / 60.0
    doy  = df["timestamp_wib"].dt.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * hr / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hr / 24)
    df["sin_doy"]  = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"]  = np.cos(2 * np.pi * doy / 365.25)
    df["year"]     = df["timestamp_wib"].dt.year

    # Arah angin → siklik
    wdir_rad       = np.deg2rad(df["wind_dir_deg"].fillna(0))
    df["sin_wdir"] = np.sin(wdir_rad)
    df["cos_wdir"] = np.cos(wdir_rad)

    # ── Segmentasi ─────────────────────────────────────────
    df["gap_min"]  = df["timestamp_wib"].diff().dt.total_seconds().div(60).fillna(0)
    # Potong jika: gap > 12 mnt ATAU beda hari
    df["is_break"] = (
        (df["gap_min"] > PREP_CFG["max_gap_min"]) |
        (df["timestamp_wib"].dt.date != df["timestamp_wib"].shift().dt.date)
    )
    df["seg_id"]   = df["is_break"].cumsum()

    n_segs = df["seg_id"].nunique()
    n_breaks = df["is_break"].sum()
    print(f"  [M5] Segmentasi: {n_breaks:,} potongan → {n_segs:,} segmen kontinu")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. IMPUTASI METEOROLOGI UMUM (per segmen)
# ══════════════════════════════════════════════════════════════════════════════
def impute_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk fitur meteo dengan missing < 5% (rh, pressure, temp, dll):
    ffill → bfill dalam segmen → median global.
    """
    df = df.copy()
    meteo_cols = [
        "temp_air_c", "dewpoint_c", "rh_pct", "vapour_pressure_hpa",
        "pressure_hpa", "wind_speed_ms", "cloud_cover_oktas",
        "cloud_cover_fraction",
    ]
    before = {c: df[c].isna().sum() for c in meteo_cols}

    # Per segmen: ffill → bfill
    for col in meteo_cols:
        if col in df.columns:
            df[col] = (df.groupby("seg_id")[col]
                         .transform(lambda s: s.ffill().bfill()))
            # Sisa: median global (edge case di awal/akhir dataset)
            global_med = df[col].median()
            df[col] = df[col].fillna(global_med)

    after = {c: df[c].isna().sum() for c in meteo_cols}
    filled = sum(before[c] - after.get(c, 0) for c in meteo_cols)
    print(f"  [Meteo] {filled:,} sel diimputasi (ffill per segmen + global median)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. DEFINISI KOLOM FITUR BERSIH (untuk pipeline training)
# ══════════════════════════════════════════════════════════════════════════════
#
# Gunakan kolom '_clean' untuk fitur yang sudah diimputasi
# Ganti referensi di ghi_forecast_pipeline.py dengan kolom di bawah ini.

CLEAN_FEAT_PHASE1 = [
    "ghi_consolidated",     # target & lag — sudah di-clip
    "kt_consolidated",      # sudah di-clip
    "ghi_clearsky",
    "dhi_consolidated",
    "dni_consolidated",
    "sun_altitude",
    "sun_azimuth",
    "optical_air_mass",
    "sin_hour", "cos_hour",
    "sin_doy",  "cos_doy",
]

CLEAN_FEAT_PHASE2 = CLEAN_FEAT_PHASE1 + [
    "temp_air_c",
    "dewpoint_c",
    "rh_pct",
    "vapour_pressure_hpa",
    "pressure_hpa",
    "wind_speed_ms",
    "sin_wdir", "cos_wdir",
    "rainfall_mm_clean",    # ← pakai versi bersih
    "rain_is_available",    # ← flag ketersediaan sensor hujan (info 2021)
    "cloud_cover_oktas",
    "cloud_cover_fraction",
]

CLEAN_FEAT_PHASE3 = CLEAN_FEAT_PHASE2 + [
    "aod_best_clean",       # ← median-imputed
    "aod_550nm_clean",
    "aod_500nm_clean",
    "angstrom_clean",
    "pwat_clean",
    "beam_clean",
    "fine_aot_clean",
    "coarse_aot_clean",
    "aod_is_available",     # ← flag MNAR (KRITIS untuk Phase 3)
    "cloud_optical_thickness_clean",
    "cloud_top_temp_k_clean",
    "cloud_top_height_m_clean",
    "cloud_eff_radius_um_clean",
    "sat_cloud_present_clean",
    "clp_is_available",     # ← flag ketersediaan cloud properties
]


# ══════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE UTAMA
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(token: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Jalankan semua tahap preprocessing. Return DataFrame siap training.
    Simpan juga ke parquet untuk menghindari reload dari MotherDuck.
    """
    print("═" * 60)
    print("  GHI PREPROCESSING PIPELINE  (2022–2025)")
    print("═" * 60)

    df = load_raw(token=token)

    print("\n[STEP 1/5] Quality weight (M1)...")
    df = add_quality_weight(df)

    print("\n[STEP 2/5] Clip GHI & kt (M4)...")
    df = clip_solar(df)

    print("\n[STEP 3/5] Feature engineering & segmentasi (M5)...")
    df = add_features_and_segments(df)

    print("\n[STEP 4/5] Imputasi meteorologi + rainfall (M3)...")
    df = impute_meteo(df)
    df = impute_rainfall(df)

    print("\n[STEP 5/5] Imputasi AOD — MNAR-aware (M2)...")
    df = impute_aod(df)

    # ── Ringkasan null setelah preprocessing ──────────────
    if verbose:
        print("\n── Ringkasan null setelah preprocessing ──")
        key_cols = (CLEAN_FEAT_PHASE1 + ["rainfall_mm_clean", "rain_is_available",
                    "aod_best_clean", "aod_is_available",
                    "cloud_optical_thickness_clean", "sat_cloud_present_clean"])
        key_cols = [c for c in key_cols if c in df.columns]
        null_summary = pd.DataFrame({
            "col": key_cols,
            "n_null": [df[c].isna().sum() for c in key_cols],
            "pct_null": [round(100 * df[c].isna().sum() / len(df), 2) for c in key_cols],
        })
        remaining = null_summary[null_summary["n_null"] > 0]
        if len(remaining) == 0:
            print("  ✓ Semua kolom utama sudah bersih (0 null)")
        else:
            print(remaining.to_string(index=False))

    print(f"\n✓ Preprocessing selesai. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 10. SIMPAN & VISUALISASI QC
# ══════════════════════════════════════════════════════════════════════════════
def save_preprocessed(df: pd.DataFrame, output_dir: str = "./ghi_output"):
    """Simpan ke parquet — jauh lebih cepat untuk re-load."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "jambi_obs_preprocessed.parquet"
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  Saved → {path}  ({size_mb:.1f} MB)")
    return path


def plot_qc_report(df: pd.DataFrame, output_dir: str = "./ghi_output"):
    """
    Buat laporan QC visual:
    - Distribusi GHI per tahun + quality flag
    - Coverage AOD per bulan
    - Distribusi Kt sebelum/sesudah clip
    - Sample time series 3 hari
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Distribusi GHI per tahun ─────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]["ghi_consolidated"]
        sub[sub > 0].hist(ax=ax1, bins=50, alpha=0.5, label=str(yr), density=True)
    ax1.set_xlabel("GHI (W/m²)")
    ax1.set_title("Distribusi GHI per Tahun")
    ax1.legend(fontsize=7)

    # ── Panel 2: Quality flag breakdown per tahun ──────────
    ax2 = fig.add_subplot(gs[0, 1])
    flag_yr = (df.groupby(["year", "ghi_quality_flag"])
                 .size().unstack(fill_value=0))
    flag_yr_pct = flag_yr.div(flag_yr.sum(axis=1), axis=0) * 100
    flag_yr_pct.plot(kind="bar", stacked=True, ax=ax2,
                     colormap="tab10", width=0.8)
    ax2.set_xlabel("Tahun")
    ax2.set_ylabel("%")
    ax2.set_title("Komposisi Quality Flag per Tahun")
    ax2.legend(fontsize=6, loc="lower right")
    ax2.tick_params(axis="x", rotation=0)

    # ── Panel 3: Sample weight distribution ───────────────
    ax3 = fig.add_subplot(gs[0, 2])
    df["sample_weight"].value_counts().sort_index().plot(
        kind="bar", ax=ax3, color="steelblue")
    ax3.set_xlabel("Sample Weight")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribusi Sample Weight")
    ax3.tick_params(axis="x", rotation=0)

    # ── Panel 4: Coverage AOD per bulan ───────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    df["month"] = df["timestamp_wib"].dt.month
    aod_cov = df.groupby(["year", "month"])["aod_is_available"].mean() * 100
    aod_piv = aod_cov.unstack(level=0)
    aod_piv.plot(ax=ax4, marker="o", linewidth=1.5)
    ax4.set_xlabel("Bulan")
    ax4.set_ylabel("Coverage AOD (%)")
    ax4.set_title("Coverage AOD Ground per Bulan")
    ax4.set_xticks(range(1, 13))
    ax4.legend(fontsize=7, title="Tahun")
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Kt distribution sebelum & sesudah clip ───
    ax5 = fig.add_subplot(gs[1, 1])
    # Setelah clip (dari df)
    df["kt_consolidated"].clip(0, 2).hist(ax=ax5, bins=60, alpha=0.7,
                                           label="Setelah clip (≤1.5)",
                                           color="steelblue")
    ax5.axvline(1.0, color="orange", linestyle="--", label="kt=1.0")
    ax5.axvline(1.5, color="red",    linestyle="--", label="kt=1.5 (clip)")
    ax5.set_xlabel("kt (clearness index)")
    ax5.set_title("Distribusi Kt (Clearness Index)")
    ax5.legend(fontsize=8)

    # ── Panel 6: AOD before/after imputation ──────────────
    ax6 = fig.add_subplot(gs[1, 2])
    df["aod_best"].dropna().hist(ax=ax6, bins=50, alpha=0.6,
                                  label="AOD asli", color="steelblue")
    df["aod_best_clean"].hist(ax=ax6, bins=50, alpha=0.4,
                               label="AOD imputasi", color="orange")
    ax6.set_xlabel("AOD")
    ax6.set_title("AOD Asli vs Setelah Imputasi")
    ax6.legend(fontsize=8)

    # ── Panel 7: GHI rata-rata bulanan per tahun (2022–2025) ──
    ax7 = fig.add_subplot(gs[2, 0])
    df["month"] = df["timestamp_wib"].dt.month
    ghi_monthly = (df[df["ghi_consolidated"] > 0]
                   .groupby(["year", "month"])["ghi_consolidated"]
                   .mean().unstack(level=0))
    ghi_monthly.plot(ax=ax7, marker="o", linewidth=1.5)
    ax7.set_xlabel("Bulan")
    ax7.set_ylabel("GHI rata-rata (W/m²)")
    ax7.set_title("GHI Rata-rata Bulanan 2022–2025")
    ax7.set_xticks(range(1, 13))
    ax7.legend(fontsize=7, title="Tahun")
    ax7.grid(True, alpha=0.3)

    # ── Panel 8: Time series sample 3 hari ────────────────
    ax8 = fig.add_subplot(gs[2, 1:])
    # Ambil 3 hari dari 2023 (data terbaik)
    sample = df[(df["year"] == 2023) &
                (df["timestamp_wib"].dt.month == 6) &
                (df["timestamp_wib"].dt.day.isin([1, 2, 3]))]
    ax8.plot(sample["timestamp_wib"], sample["ghi_consolidated"],
             "b-", label="GHI obs", linewidth=1.5)
    ax8.plot(sample["timestamp_wib"], sample["ghi_clearsky"],
             "r--", label="GHI clearsky", linewidth=1)
    ax8.fill_between(sample["timestamp_wib"],
                     sample["ghi_consolidated"],
                     sample["ghi_clearsky"],
                     alpha=0.15, color="gray")
    ax8_r = ax8.twinx()
    ax8_r.bar(sample["timestamp_wib"],
              sample["aod_is_available"] * sample["aod_best_clean"],
              alpha=0.3, color="orange", label="AOD", width=0.006)
    ax8.set_ylabel("GHI (W/m²)")
    ax8_r.set_ylabel("AOD")
    ax8.set_title("Contoh Time Series: 1-3 Juni 2023")
    ax8.legend(fontsize=8, loc="upper left")
    ax8.grid(True, alpha=0.3)

    plt.suptitle("QC Report — Preprocessing GHI Forecasting Jambi", fontsize=13)
    save_path = out / "qc_preprocessing_report.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  QC report → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. INTEGRASI DENGAN ghi_forecast_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════
def load_preprocessed(parquet_path: str) -> pd.DataFrame:
    """
    Load data yang sudah dipreprocess dari parquet.
    Gunakan ini di ghi_forecast_pipeline.py sebagai pengganti load_data().

    Contoh penggunaan di pipeline:
        from ghi_preprocess import load_preprocessed, CLEAN_FEAT_PHASE1, ...
        df = load_preprocessed('./ghi_output/jambi_obs_preprocessed.parquet')
        phase_results = run_phase('phase1', df, CLEAN_FEAT_PHASE1, out_dir)
    """
    df = pd.read_parquet(parquet_path)
    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    print(f"[LOAD] Preprocessed: {len(df):,} baris | kolom: {df.shape[1]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GHI Preprocessing")
    parser.add_argument("--token",      type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./ghi_output")
    parser.add_argument("--plot",       action="store_true",
                        help="Buat QC report visual")
    args = parser.parse_args()

    # Jalankan preprocessing
    df_clean = preprocess(token=args.token)

    # Simpan ke parquet
    save_preprocessed(df_clean, args.output_dir)

    # QC report (opsional)
    if args.plot:
        print("\nMembuat QC report...")
        plot_qc_report(df_clean, args.output_dir)

    # Tampilkan ringkasan
    print("\n── Ringkasan kolom baru yang ditambahkan ──")
    new_cols = [c for c in df_clean.columns
                if c.endswith("_clean") or c.endswith("_available")
                or c in ["sample_weight", "seg_id", "gap_min"]]
    print(pd.DataFrame({
        "kolom_baru": new_cols,
        "dtype": [str(df_clean[c].dtype) for c in new_cols],
        "n_null": [df_clean[c].isna().sum() for c in new_cols],
    }).to_string(index=False))

    print(f"\n✓ Selesai. Data siap training di: {args.output_dir}/jambi_obs_preprocessed.parquet")
    print("\nLangkah berikutnya:")
    print("  python ghi_forecast_pipeline.py \\")
    print("    --preprocess_mode parquet \\")
    print("    --parquet_path ./ghi_output/jambi_obs_preprocessed.parquet \\")
    print("    --phases phase1 phase2 phase3")
