import duckdb

con = duckdb.connect()
con.execute("CREATE OR REPLACE TABLE radiation_dedup AS SELECT * FROM read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/radiation_dedup.parquet')")

con.execute("""
CREATE OR REPLACE TEMP TABLE rad_pts AS
SELECT
  ts_wib,
  date_trunc('hour', ts_wib) + INTERVAL 1 HOUR AS hour_bucket,
  ghi, dhi, dni, reflected, solar_elev_deg,
  1366.1*(1+0.033*cos(2*pi()*extract(doy FROM ts_wib)/365.0)) * sin(radians(solar_elev_deg)) AS toa_ghi,
  CASE WHEN solar_elev_deg > 5 THEN ghi / NULLIF(1366.1*(1+0.033*cos(2*pi()*extract(doy FROM ts_wib)/365.0)) * sin(radians(solar_elev_deg)),0) END AS kt
FROM radiation_dedup
""")

con.execute("""
CREATE OR REPLACE TABLE hourly_rad AS
SELECT
  hour_bucket AS ts_hour,
  count(*) AS n_obs,
  avg(toa_ghi) AS ghi_clearsky_mean, min(toa_ghi) AS ghi_clearsky_min, max(toa_ghi) AS ghi_clearsky_max,
  avg(kt) AS kt_mean, stddev(kt) AS kt_std, min(kt) AS kt_min, max(kt) AS kt_max,
  avg(ghi) AS ghi_mean, stddev(ghi) AS ghi_std, min(ghi) AS ghi_min, max(ghi) AS ghi_max,
  avg(dni) AS dni_mean, stddev(dni) AS dni_std, min(dni) AS dni_min, max(dni) AS dni_max,
  avg(dhi) AS dhi_mean, stddev(dhi) AS dhi_std, min(dhi) AS dhi_min, max(dhi) AS dhi_max,
  avg(reflected) AS reflected_mean,
  arg_max(solar_elev_deg, ts_wib) AS sun_altitude_at_h
FROM rad_pts
GROUP BY 1
HAVING count(*) >= 5
""")
n = con.execute("SELECT count(*) FROM hourly_rad").fetchone()[0]
print("hourly_rad (quality filtered):", n)

# full continuous hourly grid (so lag features are correctly NULL across real gaps, not silently shifted)
con.execute("""
CREATE OR REPLACE TABLE hourly_full AS
SELECT g.ts_hour, h.* EXCLUDE (ts_hour)
FROM (SELECT range AS ts_hour FROM range(
    (SELECT min(ts_hour) FROM hourly_rad), (SELECT max(ts_hour) FROM hourly_rad) + INTERVAL 1 HOUR, INTERVAL 1 HOUR)) g
LEFT JOIN hourly_rad h ON h.ts_hour = g.ts_hour
ORDER BY g.ts_hour
""")
print("hourly_full (continuous grid):", con.execute("SELECT count(*) FROM hourly_full").fetchone()[0])

# daytime-only filter for target validity (clearsky present)
con.execute("""
CREATE OR REPLACE TABLE hourly_features AS
SELECT
  *,
  lag(kt_mean,1) OVER w AS kt_mean_lag1, lag(kt_mean,2) OVER w AS kt_mean_lag2, lag(kt_mean,3) OVER w AS kt_mean_lag3,
  lag(kt_mean,4) OVER w AS kt_mean_lag4, lag(kt_mean,5) OVER w AS kt_mean_lag5, lag(kt_mean,6) OVER w AS kt_mean_lag6,
  lag(kt_min,1) OVER w AS kt_min_lag1, lag(kt_min,2) OVER w AS kt_min_lag2, lag(kt_min,3) OVER w AS kt_min_lag3,
  lag(kt_min,4) OVER w AS kt_min_lag4, lag(kt_min,5) OVER w AS kt_min_lag5, lag(kt_min,6) OVER w AS kt_min_lag6,
  lag(kt_max,1) OVER w AS kt_max_lag1, lag(kt_max,2) OVER w AS kt_max_lag2, lag(kt_max,3) OVER w AS kt_max_lag3,
  lag(kt_max,4) OVER w AS kt_max_lag4, lag(kt_max,5) OVER w AS kt_max_lag5, lag(kt_max,6) OVER w AS kt_max_lag6,
  lag(ghi_clearsky_mean,1) OVER w AS ghi_clearsky_mean_lag1, lag(ghi_clearsky_mean,2) OVER w AS ghi_clearsky_mean_lag2,
  lag(ghi_clearsky_mean,3) OVER w AS ghi_clearsky_mean_lag3, lag(ghi_clearsky_mean,6) OVER w AS ghi_clearsky_mean_lag6,
  lag(dhi_std,1) OVER w AS dhi_std_lag1, lag(dhi_std,2) OVER w AS dhi_std_lag2, lag(dhi_std,3) OVER w AS dhi_std_lag3,
  -- target: next hour's mean kt + its clearsky (to know if next hour is daytime/valid) + n_obs (quality)
  lead(kt_mean,1) OVER w AS y_kt_next,
  lead(ghi_clearsky_mean,1) OVER w AS y_clearsky_next,
  lead(n_obs,1) OVER w AS y_n_obs_next,
  lead(ghi_mean,1) OVER w AS y_ghi_next
FROM hourly_full
WINDOW w AS (ORDER BY ts_hour)
""")

# final modeling dataset: current hour must be valid daytime, target hour must be valid daytime too
con.execute("""
COPY (
  SELECT * FROM hourly_features
  WHERE ghi_clearsky_mean > 5 AND n_obs >= 5
    AND y_clearsky_next > 5 AND y_n_obs_next >= 5
    AND y_kt_next IS NOT NULL
) TO 'C:/Users/ariff/bengkulu_ghi_forecast/stage1_dataset.parquet' (FORMAT PARQUET)
""")

import pandas as pd
df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage1_dataset.parquet')
print("final stage1 dataset shape:", df.shape)
print("date range:", df['ts_hour'].min(), "to", df['ts_hour'].max())
print("y_kt_next stats:", df['y_kt_next'].describe())
