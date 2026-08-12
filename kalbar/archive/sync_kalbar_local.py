"""
sync_kalbar_local.py
--------------------
Sync tabel-tabel utama dari MotherDuck (md:kalbar) ke DuckDB lokal (kalbar_local.db).

Cara pakai:
  python sync_kalbar_local.py              # sync semua tabel (full replace)
  python sync_kalbar_local.py --incremental # sync inkremental (hanya baris baru)
  python sync_kalbar_local.py --tables solar_radiation_valid synop_unified
"""

import duckdb
import argparse
import time
import os
from datetime import datetime

# ── Konfigurasi ──────────────────────────────────────────────────────────────
LOCAL_DB   = os.path.join(os.path.dirname(__file__), "kalbar_local.db")
CLOUD_DB   = "md:kalbar"

# Tabel yang di-sync beserta kolom timestamp untuk mode inkremental
# None = tidak ada kolom timestamp (selalu full replace)
TABLES = {
    "solar_radiation_valid":      "timestamp_wib",
    "synop_unified":              "timestamp_wib",
    "radiationweather":           "timestamp_wib",
    "aws_sta2117_clean":          "waktu_wib",
    "aeronetpontianak":           "timestamp_wib",
    "stationcrossvalidation":     "timestamp_wib",
    "unifiedradiationcompletegrid": "timestamp_wib",
    "asrskalbar":                 None,   # large, full replace saja
    "ghi_valid":                  "timestamp_wib",
    "arp_pontianak":              "timestamp_wib",   # aerosol Himawari ARP
    "clp_pontianak":              "timestamp_wib",   # cloud properties Himawari CLP
    "aeronet_aod_min_pontianak":  "timestamp_wib",   # diturunkan dari datetime_10min_utc + 7 jam
}

# Tabel yang kolom waktunya di cloud tidak bernama timestamp_wib -- turunkan saat sync
# supaya skema lokal konsisten dengan tabel lain (key timestamp_wib, WIB = UTC+7).
DERIVED_TS = {
    "aeronet_aod_min_pontianak": "datetime_10min_utc + INTERVAL '7 HOUR'",
}
# ─────────────────────────────────────────────────────────────────────────────


def cloud_source(table):
    """SQL sumber dari cloud, menambahkan timestamp_wib turunan bila perlu."""
    if table in DERIVED_TS:
        return f"(SELECT *, {DERIVED_TS[table]} AS timestamp_wib FROM cloud.main.{table})"
    return f"cloud.main.{table}"


def fmt_num(n):
    return f"{n:,}"

def sync_table(con, table, ts_col, incremental):
    """Sync satu tabel dari cloud ke lokal."""
    t0 = time.time()
    src = cloud_source(table)

    # Cek apakah tabel sudah ada di lokal
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name=?", [table]
    ).fetchone()[0] > 0

    if incremental and exists and ts_col:
        # Ambil timestamp terakhir di lokal
        last_ts = con.execute(
            f"SELECT MAX({ts_col}) FROM main.{table}"
        ).fetchone()[0]

        if last_ts is None:
            # Tabel kosong, lakukan full sync
            mode = "full (tabel kosong)"
            n_before = 0
        else:
            mode = f"inkremental (sejak {last_ts})"
            n_before = con.execute(
                f"SELECT COUNT(*) FROM main.{table}"
            ).fetchone()[0]

            n_new = con.execute(
                f"SELECT COUNT(*) FROM {src} "
                f"WHERE {ts_col} > ?", [last_ts]
            ).fetchone()[0]

            if n_new == 0:
                elapsed = time.time() - t0
                print(f"  ✓ {table}: tidak ada data baru [{elapsed:.1f}s]")
                return

            con.execute(
                f"INSERT INTO main.{table} "
                f"SELECT * FROM {src} WHERE {ts_col} > ?",
                [last_ts]
            )
            n_after = con.execute(
                f"SELECT COUNT(*) FROM main.{table}"
            ).fetchone()[0]
            elapsed = time.time() - t0
            print(f"  ✓ {table}: +{fmt_num(n_new)} baris baru "
                  f"(total {fmt_num(n_after)}) [{elapsed:.1f}s] [{mode}]")
            return
    else:
        mode = "full replace"

    # Full replace
    con.execute(f"CREATE OR REPLACE TABLE main.{table} AS "
                f"SELECT * FROM {src}")
    n = con.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
    elapsed = time.time() - t0
    print(f"  ✓ {table}: {fmt_num(n)} baris [{elapsed:.1f}s] [{mode}]")


def main():
    parser = argparse.ArgumentParser(description="Sync kalbar MotherDuck → lokal DuckDB")
    parser.add_argument("--incremental", action="store_true",
                        help="Hanya ambil baris baru (berdasarkan kolom timestamp)")
    parser.add_argument("--tables", nargs="+", metavar="TABLE",
                        help="Pilih tabel tertentu (default: semua)")
    args = parser.parse_args()

    tables_to_sync = args.tables if args.tables else list(TABLES.keys())

    # Validasi nama tabel
    unknown = [t for t in tables_to_sync if t not in TABLES]
    if unknown:
        print(f"ERROR: Tabel tidak dikenal: {unknown}")
        print(f"Tabel tersedia: {list(TABLES.keys())}")
        return

    print("=" * 60)
    print(f"Sync kalbar MotherDuck → DuckDB lokal")
    print(f"Mode     : {'inkremental' if args.incremental else 'full replace'}")
    print(f"Tabel    : {tables_to_sync}")
    print(f"Target   : {LOCAL_DB}")
    print(f"Waktu    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t_total = time.time()

    # Buka koneksi ke lokal, attach cloud
    con = duckdb.connect(LOCAL_DB)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"ATTACH '{CLOUD_DB}' AS cloud (READ_ONLY)")
    # File lokal sudah ter-attach otomatis sebagai 'main' saat connect — tidak perlu ATTACH lagi

    for table in tables_to_sync:
        ts_col = TABLES[table]
        print(f"\n→ {table} ...")
        try:
            sync_table(con, table, ts_col, args.incremental)
        except Exception as e:
            print(f"  ✗ {table}: ERROR — {e}")

    con.close()

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Selesai dalam {elapsed_total:.1f} detik")
    print(f"File lokal: {LOCAL_DB}")

    # Tampilkan ukuran file
    size_mb = os.path.getsize(LOCAL_DB) / (1024 * 1024)
    print(f"Ukuran   : {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
