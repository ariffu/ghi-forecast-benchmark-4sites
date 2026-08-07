#!/usr/bin/env python3
"""
Sync derived views + training tables from MotherDuck (md:bengkulu) into the local
bengkulu.duckdb file, so v1-v8 training scripts can run fully offline.

The 4 raw base tables (asrs_bengkulu_combined, aws_bengkulu, clp_bengkulu_combined,
synop_bengkulu) were already verified identical (row count + max ts_wib) between
local and remote, so they are NOT re-copied here.

What this script does:
  1. Recreate the 4 "*_quality_final" views + bengkulu_master_10min_quality_final
     view locally, rewritten to reference local.bengkulu_sch.* instead of
     remote.bengkulu_sch.* (same SQL logic, pulled live via duckdb_views()).
  2. Physically copy the 2 derived training tables (ghi_forecast_1h_train_3h_rollback,
     ghi_forecast_1h_train_3h_rollback_2021_2025) since they are base tables, not views.
"""
import duckdb

LOCAL_PATH = "C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb"

VIEWS_IN_ORDER = [
    "asrs_bengkulu_quality_final",
    "aws_bengkulu_quality_final",
    "clp_bengkulu_quality_final",
    "synop_bengkulu_quality_final",
    "bengkulu_master_10min_quality_final",  # depends on the 4 views above
]

TABLES_TO_COPY = [
    "ghi_forecast_1h_train_3h_rollback",
    "ghi_forecast_1h_train_3h_rollback_2021_2025",
]


def main():
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'md:bengkulu' AS remote")
    con.execute(f"ATTACH '{LOCAL_PATH}' AS local")

    print("=== 1. Recreate views locally ===")
    view_sql = dict(con.execute("""
        SELECT view_name, sql FROM duckdb_views()
        WHERE database_name = 'remote' AND view_name IN ({})
    """.format(",".join("'" + v + "'" for v in VIEWS_IN_ORDER))).fetchall())

    for view_name in VIEWS_IN_ORDER:
        sql = view_sql[view_name]
        # Strip catalog qualification entirely (just "bengkulu_sch.table") so the
        # view resolves correctly regardless of what alias this file is ATTACHed
        # under later (e.g. "bengkulu_db" in the training scripts vs "local" here).
        local_sql = sql.replace("bengkulu_db.bengkulu_sch.", "bengkulu_sch.")
        local_sql = local_sql.replace("remote.bengkulu_sch.", "bengkulu_sch.")
        local_sql = local_sql.replace(
            "CREATE VIEW bengkulu_sch." + view_name,
            "CREATE OR REPLACE VIEW local.bengkulu_sch." + view_name,
        )
        con.execute(local_sql)
        n = con.execute(f"SELECT COUNT(*) FROM local.bengkulu_sch.{view_name}").fetchone()[0]
        print(f"  OK  {view_name:42s} n={n}")

    print()
    print("=== 2. Copy derived training tables ===")
    for t in TABLES_TO_COPY:
        con.execute(f"CREATE OR REPLACE TABLE local.bengkulu_sch.{t} AS SELECT * FROM remote.bengkulu_sch.{t}")
        n_local = con.execute(f"SELECT COUNT(*) FROM local.bengkulu_sch.{t}").fetchone()[0]
        n_remote = con.execute(f"SELECT COUNT(*) FROM remote.bengkulu_sch.{t}").fetchone()[0]
        print(f"  OK  {t:42s} local={n_local} remote={n_remote} match={n_local == n_remote}")

    print()
    print("=== 3. Verifikasi akhir: daftar tabel/view di local.bengkulu_sch ===")
    rows = con.execute("""
        SELECT table_name, table_type FROM information_schema.tables
        WHERE table_catalog = 'local' ORDER BY table_name
    """).fetchall()
    for name, ttype in rows:
        print(f"  {name:48s} {ttype}")

    con.close()
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
