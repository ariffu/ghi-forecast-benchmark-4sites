"""
sync_jambi_local.py
-------------------
Sync tabel-tabel utama dari MotherDuck (md:jambi / schema jambi_sch)
ke DuckDB lokal (C:/Users/ariff/DuckDB_jambi/jambi.duckdb).

VIEW (jambi_obs_combined, cloud_features_onehot) tidak di-sync —
cukup dibuat ulang secara lokal dari tabel base-nya.

Cara pakai:
  python sync_jambi_local.py              # sync semua tabel (full replace)
  python sync_jambi_local.py --incremental # sync inkremental (hanya baris baru)
  python sync_jambi_local.py --tables solar_radiation_valid aod_consolidated
"""

import duckdb
import argparse
import time
import os
from datetime import datetime

# ── Konfigurasi ──────────────────────────────────────────────────────────────
LOCAL_DB  = r"C:\Users\ariff\DuckDB_jambi\jambi.duckdb"
CLOUD_DB  = "md:jambi"
SCHEMA    = "jambi_sch"

# Tabel yang di-sync beserta kolom timestamp untuk mode inkremental.
# None = tidak ada kolom timestamp → selalu full replace.
#
# Kelompok A: Tabel sumber / raw
# Kelompok B: Tabel konsolidasi (hasil pipeline)
TABLES = {
    # ── A. Radiasi Matahari ───────────────────────────────────────────────────
    "solar_radiation_valid":    "timestamp_wib",   # tabel utama konsolidasi GHI/DHI/DNI

    # ── B. Aerosol ────────────────────────────────────────────────────────────
    "aeronet_jambi_aerosol":    "datetime_wib",    # AERONET Level 2.0 (TIMESTAMP WITH TIME ZONE)
    "aod_directsun_clean":      "timestamp_wib",   # AERONET direct-sun Level 2.0
    "aod_ground_obs":           "obs_ts",          # gabungan directsun + aeronet (intermediate)
    "aod_metadata":             None,              # summary 3 baris, full replace
    "arp_jambi":                "timestamp",       # Himawari satellite AOT (10-min)
    "aod_consolidated":         "timestamp_wib",   # aerosol terkonsolidasi (10-min grid)

    # ── C. Meteorologi & Awan ─────────────────────────────────────────────────
    "meteo_obs_10min":          "timestamp_wib",   # observasi AWS 10-menit
    "synop_jambi_combined":     "waktu",           # SYNOP jam-an (WIB)
    "clp_jambi":                "timestamp",       # Cloud Layer Properties raw Himawari
    "jambi_clp_combined":       "timestamp",       # CLP gabungan H08+H09
    "meteo_consolidated":       "timestamp_wib",   # meteorologi terkonsolidasi (10-min grid)
}

# VIEW tidak perlu di-sync (dibuat ulang dari base tables)
VIEWS_LOCAL = [
    "jambi_obs_combined",       # JOIN solar + aerosol + meteo
    "cloud_features_onehot",    # fitur cloud one-hot encoding
]
# ─────────────────────────────────────────────────────────────────────────────

VIEW_DEFINITIONS = {
    "jambi_obs_combined": """
CREATE OR REPLACE VIEW {schema}.jambi_obs_combined AS
SELECT
  sr.timestamp_wib, sr.sun_altitude, sr.sun_azimuth, sr.ghi_clearsky, sr.n_minutes,
  sr.ghi_consolidated, sr.ghi_cons_source, sr.ghi_quality_flag, sr.kt_consolidated,
  sr.ghi_consolidated_min, sr.ghi_consolidated_max,
  sr.dhi_consolidated, sr.dhi_consolidated_min, sr.dhi_consolidated_max, sr.dhi_quality_flag,
  sr.dni_consolidated, sr.dni_consolidated_min, sr.dni_consolidated_max, sr.dni_quality_flag,
  aod.optical_air_mass,
  aod.AOD_440nm, aod.AOD_500nm, aod.AOD_675nm, aod.AOD_870nm,
  aod.angstrom_exp_440_870, aod.angstrom_exp_440_675, aod.angstrom_exp_500_870,
  aod.precipitable_water_cm, aod.ground_solar_zenith, aod.ground_triplet_var_500,
  aod.ground_source AS aod_ground_source, aod.ground_time_gap_min AS aod_ground_gap_min,
  aod.ground_quality_flag AS aod_ground_quality,
  aod.sat_aot, aod.sat_ae, aod.sat_aot_class, aod.sat_aot_quality,
  aod.sat_retrieval_valid, aod.sat_platform AS aod_sat_platform,
  aod.fine_mode_aot_proxy, aod.coarse_mode_aot_proxy,
  aod.aod_best, aod.aod_best_source, aod.aod_550nm, aod.beam_transmittance_500nm,
  met.temp_air_c, met.dewpoint_c, met.rh_pct, met.vapour_pressure_hpa,
  met.pressure_hpa, met.wind_speed_ms, met.wind_speed_max_ms, met.wind_dir_deg,
  met.rainfall_mm, met.cloud_cover_oktas, met.cloud_cover_fraction, met.cloud_base_m,
  met.cloud_low_type, met.cloud_med_type, met.cloud_high_type,
  met.visibility_km, met.present_weather,
  met.rh_iklim_4m, met.rh_iklim_7m, met.rh_iklim_10m,
  met.sat_cloud_present, met.sat_cloud_class,
  met.cloud_optical_thickness, met.clot_std,
  met.cloud_top_temp_k, met.cloud_top_temp_c, met.cloud_top_height_m,
  met.cloud_eff_radius_um, met.cler_valid, met.clp_satellite, met.clp_quality,
  met.meteo_source, met.meteo_gap_min, met.clp_source
FROM {schema}.solar_radiation_valid  sr
JOIN {schema}.aod_consolidated       aod USING (timestamp_wib)
JOIN {schema}.meteo_consolidated     met USING (timestamp_wib)
""",
}


def fmt_num(n):
    return f"{n:,}"


def sync_table(con, table, ts_col, incremental):
    """Sync satu tabel dari cloud (jambi_sch) ke lokal (jambi_sch)."""
    t0 = time.time()
    src = f"cloud.{SCHEMA}.{table}"
    dst = f"{SCHEMA}.{table}"

    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema=? AND table_name=?", [SCHEMA, table]
    ).fetchone()[0] > 0

    if incremental and exists and ts_col:
        last_ts = con.execute(
            f'SELECT MAX("{ts_col}") FROM {dst}'
        ).fetchone()[0]

        if last_ts is None:
            mode = "full (tabel kosong)"
        else:
            mode = f"inkremental (sejak {last_ts})"
            n_new = con.execute(
                f'SELECT COUNT(*) FROM {src} WHERE "{ts_col}" > ?', [last_ts]
            ).fetchone()[0]

            if n_new == 0:
                elapsed = time.time() - t0
                print(f"  ✓ {table}: tidak ada data baru [{elapsed:.1f}s]")
                return

            con.execute(
                f'INSERT INTO {dst} SELECT * FROM {src} WHERE "{ts_col}" > ?',
                [last_ts]
            )
            n_after = con.execute(f"SELECT COUNT(*) FROM {dst}").fetchone()[0]
            elapsed = time.time() - t0
            print(f"  ✓ {table}: +{fmt_num(n_new)} baris baru "
                  f"(total {fmt_num(n_after)}) [{elapsed:.1f}s] [{mode}]")
            return
    else:
        mode = "full replace"

    con.execute(f"CREATE OR REPLACE TABLE {dst} AS SELECT * FROM {src}")
    n = con.execute(f"SELECT COUNT(*) FROM {dst}").fetchone()[0]
    elapsed = time.time() - t0
    print(f"  ✓ {table}: {fmt_num(n)} baris [{elapsed:.1f}s] [{mode}]")


def recreate_views(con):
    """Buat ulang VIEW lokal dari definisi yang tersimpan."""
    for view_name, ddl in VIEW_DEFINITIONS.items():
        t0 = time.time()
        try:
            con.execute(ddl.format(schema=SCHEMA))
            elapsed = time.time() - t0
            print(f"  ✓ VIEW {view_name} dibuat ulang [{elapsed:.1f}s]")
        except Exception as e:
            print(f"  ✗ VIEW {view_name}: ERROR — {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync jambi MotherDuck (jambi_sch) → DuckDB lokal"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Hanya ambil baris baru (berdasarkan kolom timestamp)"
    )
    parser.add_argument(
        "--tables", nargs="+", metavar="TABLE",
        help="Pilih tabel tertentu (default: semua)"
    )
    parser.add_argument(
        "--skip-views", action="store_true",
        help="Lewati pembuatan ulang VIEW lokal"
    )
    args = parser.parse_args()

    tables_to_sync = args.tables if args.tables else list(TABLES.keys())

    unknown = [t for t in tables_to_sync if t not in TABLES]
    if unknown:
        print(f"ERROR: Tabel tidak dikenal: {unknown}")
        print(f"Tabel tersedia: {list(TABLES.keys())}")
        return

    os.makedirs(os.path.dirname(LOCAL_DB), exist_ok=True)

    print("=" * 62)
    print("Sync jambi MotherDuck → DuckDB lokal")
    print(f"Mode     : {'inkremental' if args.incremental else 'full replace'}")
    print(f"Schema   : {SCHEMA}")
    print(f"Tabel    : {tables_to_sync}")
    print(f"Target   : {LOCAL_DB}")
    print(f"Waktu    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    t_total = time.time()

    con = duckdb.connect(LOCAL_DB)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"ATTACH '{CLOUD_DB}' AS cloud (READ_ONLY)")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    # ── Sync tabel ────────────────────────────────────────────────────────────
    for table in tables_to_sync:
        ts_col = TABLES[table]
        print(f"\n→ {table} ...")
        try:
            sync_table(con, table, ts_col, args.incremental)
        except Exception as e:
            print(f"  ✗ {table}: ERROR — {e}")

    # ── Buat ulang VIEW lokal ─────────────────────────────────────────────────
    if not args.skip_views and not args.tables:
        print(f"\n── VIEW lokal ──")
        recreate_views(con)

    con.close()

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 62}")
    print(f"Selesai dalam {elapsed_total:.1f} detik")
    print(f"File lokal: {LOCAL_DB}")
    size_mb = os.path.getsize(LOCAL_DB) / (1024 * 1024)
    print(f"Ukuran   : {size_mb:.1f} MB")
    print("=" * 62)


if __name__ == "__main__":
    main()
