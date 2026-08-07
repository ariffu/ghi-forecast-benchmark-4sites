import duckdb

con = duckdb.connect()
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS db (READ_ONLY)")

# dedup: one row per ts_wib, priority asrs_corrected > gapfills > raw21_26 > 21_23 > sta6009 > asrs_bengkulu(dup of corrected)
con.execute("""
CREATE OR REPLACE TEMP TABLE radiation_dedup AS
SELECT * FROM (
  SELECT
    ts_wib,
    ghi_w_m2_corrected AS ghi, dhi_w_m2_corrected AS dhi, dni_w_m2_corrected AS dni,
    reflected_w_m2_corrected AS reflected, solar_elev_deg, source_table,
    ROW_NUMBER() OVER (
      PARTITION BY ts_wib
      ORDER BY CASE source_table
        WHEN 'asrs_corrected' THEN 1
        WHEN 'gapfill_ml' THEN 2
        WHEN 'gapfill_night' THEN 2
        WHEN 'gapfill_short_night' THEN 2
        WHEN 'gapfill_clp' THEN 2
        WHEN 'asrs_raw_bengkulu21_26' THEN 3
        WHEN 'asrs_bengkulu_21_23' THEN 4
        WHEN 'asrs_bengkulu_sta6009' THEN 5
        WHEN 'asrs_bengkulu' THEN 6
        ELSE 9
      END
    ) AS rn
  FROM db.bengkulu_sch.asrs_bengkulu_combined
  WHERE ghi_w_m2_corrected IS NOT NULL
) WHERE rn = 1
""")

n = con.execute("SELECT count(*) FROM radiation_dedup").fetchone()[0]
print("deduped rows:", n)

rng = con.execute("SELECT min(ts_wib), max(ts_wib) FROM radiation_dedup").fetchone()
print("range:", rng)

# sanity: any timestamp still duplicated after dedup (shouldn't be possible given rn=1, just confirm)
dup = con.execute("SELECT count(*) FROM (SELECT ts_wib, count(*) c FROM radiation_dedup GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
print("residual dup ts:", dup)

# distribution of which source actually won
print(con.execute("SELECT source_table, count(*) c FROM radiation_dedup GROUP BY 1 ORDER BY c DESC").df())

con.execute("COPY radiation_dedup TO 'C:/Users/ariff/bengkulu_ghi_forecast/radiation_dedup.parquet' (FORMAT PARQUET)")
