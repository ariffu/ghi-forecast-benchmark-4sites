import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()
con.execute("CREATE OR REPLACE TABLE radiation_dedup AS SELECT * FROM read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/radiation_dedup.parquet')")
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS db (READ_ONLY)")

# 10-minute bin-averaged radiation (slightly denoised vs Kalbar's raw instant, since our native cadence is finer)
con.execute("""
CREATE OR REPLACE TABLE rad_10min AS
SELECT
  date_trunc('hour', ts_wib) + INTERVAL (10*(minute(ts_wib)/10)) MINUTE AS ts_bin,
  avg(ghi) AS ghi_wm2, avg(dni) AS dni_wm2, avg(dhi) AS dhi_wm2,
  avg(solar_elev_deg) AS sun_altitude
FROM radiation_dedup
GROUP BY 1
""")
print("rad_10min:", con.execute("SELECT count(*) FROM rad_10min").fetchone()[0])

# CLP nearest-match within +/-5min tolerance to the 10-min grid (sparse satellite passes)
con.execute("""
CREATE OR REPLACE TABLE clp_raw AS
SELECT ts_wib, cot_consolidated AS cot, cth_m_consolidated AS cth, ctt_k_consolidated AS ctt,
       cer_consolidated AS cer, CAST(cloud_present_corrected AS INT) AS cloud_present
FROM db.bengkulu_sch.clp_bengkulu_combined WHERE source_table='clp_bengkulu'
""")

con.execute("""
CREATE OR REPLACE TABLE base_10min AS
SELECT r.*,
  c.cot, c.cth, c.ctt, c.cer, c.cloud_present
FROM rad_10min r
LEFT JOIN clp_raw c ON c.ts_wib = r.ts_bin
ORDER BY r.ts_bin
""")
n = con.execute("SELECT count(*) FROM base_10min").fetchone()[0]
n_clp = con.execute("SELECT count(*) FROM base_10min WHERE cot IS NOT NULL").fetchone()[0]
print(f"base_10min: {n} rows, CLP exact-match coverage: {n_clp} ({100*n_clp/n:.1f}%)")

con.execute("COPY base_10min TO 'C:/Users/ariff/bengkulu_ghi_forecast/base_10min.parquet' (FORMAT PARQUET)")
