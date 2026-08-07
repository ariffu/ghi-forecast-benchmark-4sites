import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS db (READ_ONLY)")
con.execute("CREATE OR REPLACE TABLE radiation_dedup AS SELECT * FROM read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/radiation_dedup.parquet')")

# anchors = exact SYNOP observation instants (genuine hourly, 07-19 WIB)
con.execute("""
CREATE OR REPLACE TABLE synop_anchors AS
SELECT ts_wib AS ts_hour,
  cloud_cover_oktas_m, temp_drybulb_c_corrected, temp_dewpoint_c_corrected,
  relative_humidity_pc_corrected, wind_speed_ff_corrected, wind_dir_deg_corrected,
  visibility_corrected, sunshine_h_corrected
FROM db.bengkulu_sch.synop_bengkulu
""")
print("synop anchors:", con.execute("SELECT count(*) FROM synop_anchors").fetchone()[0])

# radiation aggregated over (H-60,H] for EACH SYNOP anchor hour
con.execute("""
CREATE OR REPLACE TEMP TABLE rad_pts AS
SELECT ts_wib, ghi, dhi, dni, reflected, solar_elev_deg,
  1366.1*(1+0.033*cos(2*pi()*extract(doy FROM ts_wib)/365.0)) * sin(radians(solar_elev_deg)) AS toa_ghi,
  CASE WHEN solar_elev_deg > 5 THEN ghi / NULLIF(1366.1*(1+0.033*cos(2*pi()*extract(doy FROM ts_wib)/365.0)) * sin(radians(solar_elev_deg)),0) END AS kt
FROM radiation_dedup
""")

con.execute("""
CREATE OR REPLACE TABLE hourly_rad_synop AS
SELECT a.ts_hour,
  count(r.ts_wib) AS n_obs,
  avg(r.toa_ghi) AS ghi_clearsky_mean, min(r.toa_ghi) AS ghi_clearsky_min, max(r.toa_ghi) AS ghi_clearsky_max,
  avg(r.kt) AS kt_mean, stddev(r.kt) AS kt_std, min(r.kt) AS kt_min, max(r.kt) AS kt_max,
  avg(r.ghi) AS ghi_mean, stddev(r.ghi) AS ghi_std, min(r.ghi) AS ghi_min, max(r.ghi) AS ghi_max,
  avg(r.dni) AS dni_mean, stddev(r.dni) AS dni_std,
  avg(r.dhi) AS dhi_mean, stddev(r.dhi) AS dhi_std,
  avg(r.reflected) AS reflected_mean,
  arg_max(r.solar_elev_deg, r.ts_wib) AS sun_altitude_at_h
FROM synop_anchors a
LEFT JOIN rad_pts r ON r.ts_wib > a.ts_hour - INTERVAL 60 MINUTE AND r.ts_wib <= a.ts_hour
GROUP BY a.ts_hour
""")
print("hourly_rad_synop:", con.execute("SELECT count(*) FROM hourly_rad_synop").fetchone()[0])

# CLP aggregated over same window
con.execute("""
CREATE OR REPLACE TABLE hourly_clp_synop AS
SELECT a.ts_hour,
  avg(c.cot_consolidated) AS cot_mean, stddev(c.cot_consolidated) AS cot_std,
  avg(c.cth_m_consolidated) AS cth_mean, avg(c.ctt_k_consolidated) AS ctt_mean, avg(c.cer_consolidated) AS cer_mean,
  avg(CAST(c.cloud_present_corrected AS INT)) AS cloud_frac
FROM synop_anchors a
LEFT JOIN db.bengkulu_sch.clp_bengkulu_combined c
  ON c.source_table='clp_bengkulu' AND c.ts_wib > a.ts_hour - INTERVAL 60 MINUTE AND c.ts_wib <= a.ts_hour
GROUP BY a.ts_hour
""")

# target: next-hour radiation aggregate (independent grid, not restricted to synop anchors)
con.execute("""
CREATE OR REPLACE TABLE hourly_rad_anyhour AS
SELECT date_trunc('hour', ts_wib) + INTERVAL 1 HOUR AS ts_hour,
  count(*) AS n_obs,
  avg(1366.1*(1+0.033*cos(2*pi()*extract(doy FROM ts_wib)/365.0)) * sin(radians(solar_elev_deg))) AS ghi_clearsky_mean,
  avg(CASE WHEN solar_elev_deg > 5 THEN ghi / NULLIF(1366.1*(1+0.033*cos(2*pi()*extract(doy FROM ts_wib)/365.0)) * sin(radians(solar_elev_deg)),0) END) AS kt_mean,
  avg(ghi) AS ghi_mean
FROM radiation_dedup
GROUP BY 1
""")

con.execute("""
CREATE OR REPLACE TABLE full_dataset AS
SELECT r.*, c.* EXCLUDE (ts_hour), a.* EXCLUDE (ts_hour),
  nxt.kt_mean AS y_kt_next, nxt.ghi_clearsky_mean AS y_clearsky_next, nxt.n_obs AS y_n_obs_next, nxt.ghi_mean AS y_ghi_next
FROM hourly_rad_synop r
JOIN hourly_clp_synop c ON c.ts_hour = r.ts_hour
JOIN synop_anchors a ON a.ts_hour = r.ts_hour
LEFT JOIN hourly_rad_anyhour nxt ON nxt.ts_hour = r.ts_hour + INTERVAL 1 HOUR
ORDER BY r.ts_hour
""")
n = con.execute("SELECT count(*) FROM full_dataset").fetchone()[0]
print("full_dataset (pre-filter):", n)
con.execute("COPY full_dataset TO 'C:/Users/ariff/bengkulu_ghi_forecast/synop_hourly_full.parquet' (FORMAT PARQUET)")
