import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS db (READ_ONLY)")

con.execute("""
CREATE OR REPLACE TABLE hourly_aws AS
SELECT
  date_trunc('hour', ts_wib) + INTERVAL 1 HOUR AS ts_hour,
  count(*) AS n_obs_aws,
  avg(tt_air_avg_filled) AS temp_mean, stddev(tt_air_avg_filled) AS temp_std,
  min(tt_air_min_filled) AS temp_min, max(tt_air_max_filled) AS temp_max,
  avg(rh_avg_filled) AS rh_mean, stddev(rh_avg_filled) AS rh_std, min(rh_avg_filled) AS rh_min, max(rh_avg_filled) AS rh_max,
  avg(pp_air_filled) AS pressure_mean, stddev(pp_air_filled) AS pressure_std,
  avg(ws_avg_filled) AS wind_speed_mean, max(ws_max_filled) AS wind_speed_max,
  avg(wd_avg_filled) AS wind_dir_mean,
  sum(rr_filled) AS rainfall_sum
FROM db.bengkulu_sch.aws_bengkulu
GROUP BY 1
""")
print("hourly_aws:", con.execute("SELECT count(*) FROM hourly_aws").fetchone()[0])
con.execute("COPY hourly_aws TO 'C:/Users/ariff/bengkulu_ghi_forecast/hourly_aws.parquet' (FORMAT PARQUET)")

# merge with stage1 hourly_features (need to rebuild that full grid here too, reuse logic)
stage1 = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage1_dataset.parquet')
aws = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/hourly_aws.parquet')

# merge current-hour meteo + lag1-6 of meteo onto stage1 rows
con2 = duckdb.connect()
con2.register('stage1', stage1)
con2.register('aws', aws)
merged = con2.execute("""
WITH aws_lag AS (
  SELECT *,
    lag(temp_mean,1) OVER w AS temp_mean_lag1, lag(temp_mean,2) OVER w AS temp_mean_lag2, lag(temp_mean,3) OVER w AS temp_mean_lag3,
    lag(temp_mean,4) OVER w AS temp_mean_lag4, lag(temp_mean,5) OVER w AS temp_mean_lag5, lag(temp_mean,6) OVER w AS temp_mean_lag6,
    lag(rh_mean,1) OVER w AS rh_mean_lag1, lag(rh_mean,2) OVER w AS rh_mean_lag2, lag(rh_mean,3) OVER w AS rh_mean_lag3,
    lag(rh_mean,4) OVER w AS rh_mean_lag4, lag(rh_mean,5) OVER w AS rh_mean_lag5, lag(rh_mean,6) OVER w AS rh_mean_lag6,
    lag(pressure_mean,1) OVER w AS pressure_mean_lag1, lag(pressure_mean,2) OVER w AS pressure_mean_lag2, lag(pressure_mean,3) OVER w AS pressure_mean_lag3,
    lag(wind_speed_mean,1) OVER w AS wind_speed_mean_lag1, lag(wind_speed_mean,2) OVER w AS wind_speed_mean_lag2, lag(wind_speed_mean,3) OVER w AS wind_speed_mean_lag3,
    lag(rainfall_sum,1) OVER w AS rainfall_sum_lag1, lag(rainfall_sum,2) OVER w AS rainfall_sum_lag2, lag(rainfall_sum,3) OVER w AS rainfall_sum_lag3,
    lag(rainfall_sum,4) OVER w AS rainfall_sum_lag4, lag(rainfall_sum,5) OVER w AS rainfall_sum_lag5, lag(rainfall_sum,6) OVER w AS rainfall_sum_lag6
  FROM aws
  WINDOW w AS (ORDER BY ts_hour)
)
SELECT s.*, a.* EXCLUDE (ts_hour, n_obs_aws)
FROM stage1 s LEFT JOIN aws_lag a ON a.ts_hour = s.ts_hour
""").df()

print("merged shape:", merged.shape)
print("missing meteo rows:", merged['temp_mean'].isna().sum(), "/", len(merged))
merged.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage2_dataset.parquet')
