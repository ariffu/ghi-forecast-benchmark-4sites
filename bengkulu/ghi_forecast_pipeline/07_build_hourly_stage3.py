import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS db (READ_ONLY)")

con.execute("""
CREATE OR REPLACE TABLE hourly_clp AS
SELECT
  date_trunc('hour', ts_wib) + INTERVAL 1 HOUR AS ts_hour,
  count(*) AS n_obs_clp,
  avg(cot_consolidated) AS cot_mean, stddev(cot_consolidated) AS cot_std,
  avg(cth_m_consolidated) AS cth_mean,
  avg(ctt_k_consolidated) AS ctt_mean,
  avg(cer_consolidated) AS cer_mean,
  avg(CAST(cloud_present_corrected AS INT)) AS cloud_frac
FROM db.bengkulu_sch.clp_bengkulu_combined
WHERE source_table = 'clp_bengkulu'
GROUP BY 1
""")
print("hourly_clp:", con.execute("SELECT count(*) FROM hourly_clp").fetchone()[0])
con.execute("COPY hourly_clp TO 'C:/Users/ariff/bengkulu_ghi_forecast/hourly_clp.parquet' (FORMAT PARQUET)")

stage2 = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage2_dataset.parquet')
clp = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/hourly_clp.parquet')

con2 = duckdb.connect()
con2.register('stage2', stage2)
con2.register('clp', clp)
merged = con2.execute("""
WITH clp_lag AS (
  SELECT *,
    lag(cot_mean,1) OVER w AS cot_mean_lag1, lag(cot_mean,2) OVER w AS cot_mean_lag2, lag(cot_mean,3) OVER w AS cot_mean_lag3,
    lag(cot_mean,4) OVER w AS cot_mean_lag4, lag(cot_mean,5) OVER w AS cot_mean_lag5, lag(cot_mean,6) OVER w AS cot_mean_lag6,
    lag(cth_mean,1) OVER w AS cth_mean_lag1, lag(cth_mean,2) OVER w AS cth_mean_lag2, lag(cth_mean,3) OVER w AS cth_mean_lag3,
    lag(ctt_mean,1) OVER w AS ctt_mean_lag1, lag(ctt_mean,2) OVER w AS ctt_mean_lag2, lag(ctt_mean,3) OVER w AS ctt_mean_lag3,
    lag(cloud_frac,1) OVER w AS cloud_frac_lag1, lag(cloud_frac,2) OVER w AS cloud_frac_lag2, lag(cloud_frac,3) OVER w AS cloud_frac_lag3,
    lag(cloud_frac,4) OVER w AS cloud_frac_lag4, lag(cloud_frac,5) OVER w AS cloud_frac_lag5, lag(cloud_frac,6) OVER w AS cloud_frac_lag6
  FROM clp
  WINDOW w AS (ORDER BY ts_hour)
)
SELECT s.*, c.* EXCLUDE (ts_hour, n_obs_clp)
FROM stage2 s LEFT JOIN clp_lag c ON c.ts_hour = s.ts_hour
""").df()

print("merged shape:", merged.shape)
print("missing clp current-hour rows:", merged['cot_mean'].isna().sum(), "/", len(merged))
merged.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage3_dataset.parquet')
