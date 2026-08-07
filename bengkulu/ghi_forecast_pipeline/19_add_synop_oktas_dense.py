import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS db (READ_ONLY)")
con.execute("CREATE OR REPLACE TABLE base AS SELECT * FROM read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features.parquet')")
con.execute("""
CREATE OR REPLACE TABLE synop_oktas AS
SELECT ts_wib, cloud_cover_oktas_m FROM db.bengkulu_sch.synop_bengkulu WHERE cloud_cover_oktas_m IS NOT NULL
""")

# forward-fill within 65min tolerance (SYNOP hourly, so last reading stays "fresh" for up to ~60min)
con.execute("""
CREATE OR REPLACE TABLE base_with_oktas AS
SELECT b.*, s.cloud_cover_oktas_m AS oktas_raw,
  date_diff('minute', s.ts_wib, b.ts_bin) AS oktas_age_min
FROM base b
LEFT JOIN synop_oktas s
  ON s.ts_wib <= b.ts_bin AND s.ts_wib > b.ts_bin - INTERVAL 65 MINUTE
QUALIFY row_number() OVER (PARTITION BY b.ts_bin ORDER BY s.ts_wib DESC) = 1
""")
n = con.execute("SELECT count(*) FROM base_with_oktas").fetchone()[0]
n_valid = con.execute("SELECT count(*) FROM base_with_oktas WHERE oktas_raw IS NOT NULL").fetchone()[0]
print(f"rows: {n}, oktas coverage (ffill <=65min): {n_valid} ({100*n_valid/n:.1f}%)")

con.execute("""
CREATE OR REPLACE TABLE final AS
SELECT *,
  COALESCE(oktas_raw, -1) AS oktas_ffill,
  CASE WHEN oktas_raw IS NOT NULL THEN 1 ELSE 0 END AS oktas_valid_bin
FROM base_with_oktas
ORDER BY ts_bin
""")
con.execute("COPY final TO 'C:/Users/ariff/bengkulu_ghi_forecast/dense_features_oktas.parquet' (FORMAT PARQUET)")
print("saved. sample:")
print(con.execute("SELECT ts_bin, oktas_raw, oktas_age_min, oktas_ffill, oktas_valid_bin FROM final WHERE oktas_valid_bin=1 LIMIT 5").df())
