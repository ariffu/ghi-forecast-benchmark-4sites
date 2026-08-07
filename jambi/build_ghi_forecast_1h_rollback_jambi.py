#!/usr/bin/env python3
"""
Build ghi_forecast_1h_train_3h_rollback_2021_2025 (Jambi) -- replika skema
tabel Bengkulu bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025 (102 kolom),
untuk memperbaiki bug pemangkasan-ke-siang-hari yang ditemukan di pipeline R1
Jambi lama (lihat Restrukturisasi/09_Audit_Volume_Data_Jambi.md).

PENTING -- kenapa output-nya file .duckdb TERPISAH, bukan ditulis ke jambi.duckdb:
  jambi.duckdb sudah punya file .wal tersisa dari percobaan sebelumnya yang tidak
  bisa dihapus (keterbatasan izin file di lingkungan sandbox saat dokumen ini
  dibuat). DuckDB tidak bisa membuka koneksi tulis baru ke file yang .wal-nya
  tidak bisa dibersihkan. Solusinya: tulis ke file BARU (aman, tidak menyentuh
  jambi.duckdb sama sekali -- hanya dibaca via ATTACH READ_ONLY), lalu satukan
  manual nanti (mis. via DuckDB Anda sendiri di komputer lokal, yang tidak
  punya keterbatasan izin ini):
      ATTACH 'jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb' AS newdb;
      CREATE TABLE jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025 AS
          SELECT * FROM newdb.jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025;

Perbedaan kunci vs pipeline lama (lihat 09_Audit_Volume_Data_Jambi.md):
- Sumber radiasi: asrs_jambi_menit_rev (per-menit, 24 JAM PENUH, 2021-2026)
  diagregasi ke grid 10-menit LENGKAP -- bukan jambi_obs_combined/
  dfm_with_clp_stats.parquet yang sudah dipangkas ke 07:00-17:40.
- Grid waktu dibuat eksplisit (generate_series) sehingga TIDAK ADA celah
  timestamp tersembunyi -- fitur lag/rolling dihitung via window function
  SQL (LAG/AVG/MIN/MAX/STDDEV OVER ORDER BY ts_wib), bukan pandas .shift()
  yang rentan salah hitung kalau ada baris hilang.
- has_continuous_3h_history dihitung dari keberadaan data ASRS di 18 slot
  SEBELUM baris ini (termasuk dini hari/malam, karena grid sekarang 24 jam)
  -- ini memperbaiki bug "3 jam pertama tiap hari otomatis gagal" di pipeline lama.

CATATAN JUJUR (baca sebelum dipakai untuk paper final):
- qc_status/master_qc_status di sini HEURISTIK SEDERHANA (ada/tidak data per
  sumber), BUKAN replikasi audit kualitas penuh (closure check, deteksi
  sentinel, dsb) seperti solar_radiation_valid Jambi atau asrs_bengkulu_combined.
  Lihat 08_Standardisasi_Data_Mentah.md Prioritas A untuk audit lanjutan.
- SYNOP di-ASOF-join (observasi terakhir yang tersedia, toleransi maks 6 jam).
- CLP tetap kosong (NULL) di malam hari -- ini WAJAR (satelit tidak retrieve
  cloud properties malam hari), BUKAN bug seperti masalah asli yang diperbaiki.

Run (satu proses, satu koneksi -- JANGAN dipecah jadi beberapa proses karena
akan menabrak masalah .wal yang sama):
    python build_ghi_forecast_1h_rollback_jambi.py
"""
import duckdb
import time

SRC_DB   = "jambi.duckdb"
OUT_DB   = "jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb"
SCHEMA   = "jambi_sch"
TABLE    = "ghi_forecast_1h_train_3h_rollback_2021_2025"

GRID_START = "2021-01-01 00:00:00"
GRID_END   = "2025-12-31 23:50:00"


def main():
    t0 = time.time()
    con = duckdb.connect(OUT_DB, read_only=False)
    con.execute(f"ATTACH '{SRC_DB}' AS src (READ_ONLY)")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    print(f"[0] connected + attached  t={time.time()-t0:.1f}s")

    # 1. Grid waktu 10-menit LENGKAP, 24 jam penuh
    con.execute(f"""
        CREATE OR REPLACE TABLE {SCHEMA}.stg_grid AS
        SELECT unnest(generate_series(
            TIMESTAMP '{GRID_START}', TIMESTAMP '{GRID_END}', INTERVAL 10 MINUTE
        )) AS ts_wib
    """)
    print(f"[1] grid  t={time.time()-t0:.1f}s  n={con.execute(f'SELECT count(*) FROM {SCHEMA}.stg_grid').fetchone()[0]:,}")

    # 2. Radiasi ASRS per-menit -> agregat 10-menit (24-jam-penuh)
    con.execute(f"""
        CREATE OR REPLACE TABLE {SCHEMA}.stg_asrs_10min AS
        SELECT
            time_bucket(INTERVAL 10 MINUTE, "Tanggal_WIB") AS ts_wib,
            avg(global_avg)      AS ghi_now,
            avg(diffuse_avg)     AS dhi_now,
            avg(dni_avg)         AS dni_now,
            avg(reflected_avg)   AS reflected_now,
            avg(net_avg)         AS nett_rad_now,
            avg(solar_elevation) AS solar_elev_deg,
            count(*)             AS asrs_n_obs_1min,
            sum(CASE WHEN data_flag = 0 THEN 1 ELSE 0 END) AS asrs_ok_obs
        FROM src.jambi_sch.asrs_jambi_menit_rev
        GROUP BY 1
    """)
    print(f"[2] asrs  t={time.time()-t0:.1f}s  n={con.execute(f'SELECT count(*) FROM {SCHEMA}.stg_asrs_10min').fetchone()[0]:,}")

    # 3. CLP satelit -> agregat 10-menit (siang hari saja -- wajar)
    con.execute(f"""
        CREATE OR REPLACE TABLE {SCHEMA}.stg_clp_10min AS
        SELECT
            time_bucket(INTERVAL 10 MINUTE, timestamp) AS ts_wib,
            avg(CLOT_mean) AS clp_cot,
            avg(CLTH_mean) AS clp_cth_m,
            avg(CLTT_mean) AS clp_ctt_k,
            avg(CLER_23_mean) AS clp_cer,
            max(CASE WHEN cloud_present THEN 1 ELSE 0 END) AS clp_cloud_present,
            max(CASE WHEN cloud_class IN ('clear_or_no_retrieval','clear_or_very_thin') THEN 1 ELSE 0 END) AS clp_clear_flag,
            max(CASE WHEN cloud_class = 'thin_cloud' THEN 1 ELSE 0 END) AS clp_thin_cloud_flag,
            max(CASE WHEN cloud_class = 'moderate_cloud' THEN 1 ELSE 0 END) AS clp_moderate_cloud_flag,
            max(CASE WHEN cloud_class = 'thick_cloud' THEN 1 ELSE 0 END) AS clp_thick_cloud_flag
        FROM src.jambi_sch.jambi_clp_combined
        GROUP BY 1
    """)
    print(f"[3] clp   t={time.time()-t0:.1f}s  n={con.execute(f'SELECT count(*) FROM {SCHEMA}.stg_clp_10min').fetchone()[0]:,}")

    # 4. Meteo permukaan (AWS-equivalent) -> agregat 10-menit (24 jam)
    con.execute(f"""
        CREATE OR REPLACE TABLE {SCHEMA}.stg_meteo_10min AS
        SELECT
            time_bucket(INTERVAL 10 MINUTE, timestamp_wib) AS ts_wib,
            avg(temp_air) AS aws_temp_c,
            min(temp_air) AS aws_temp_min_c,
            max(temp_air) AS aws_temp_max_c,
            avg(rh) AS aws_rh_pct,
            avg(pressure) AS aws_pressure_hpa,
            avg(ws) AS aws_ws_avg,
            max(ws_max) AS aws_ws_max,
            avg(wd) AS aws_wd_deg,
            sum(rain) AS aws_rain_mm,
            avg(sr_aws) AS aws_sr_avg_w_m2
        FROM src.jambi_sch.meteo_obs_10min
        GROUP BY 1
    """)
    print(f"[4] meteo t={time.time()-t0:.1f}s  n={con.execute(f'SELECT count(*) FROM {SCHEMA}.stg_meteo_10min').fetchone()[0]:,}")

    # 5. SYNOP ASOF-join ke grid (toleransi 6 jam)
    con.execute(f"""
        CREATE OR REPLACE TABLE {SCHEMA}.stg_synop_10min AS
        SELECT ts_wib,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_temp_c END AS synop_temp_c,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_dewpoint_c END AS synop_dewpoint_c,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_rh_pct END AS synop_rh_pct,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_wind_speed END AS synop_wind_speed,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_wind_dir_deg END AS synop_wind_dir_deg,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_visibility END AS synop_visibility,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_rainfall_24h_mm END AS synop_rainfall_24h_mm,
               CASE WHEN waktu IS NULL OR ts_wib - waktu > INTERVAL 6 HOUR THEN NULL ELSE synop_solar_rad_24h END AS synop_solar_rad_24h
        FROM (
            SELECT g2.ts_wib, syn.waktu,
                   syn.temp_drybulb_c AS synop_temp_c,
                   syn.temp_dewpoint_c AS synop_dewpoint_c,
                   syn.relative_humidity_pct AS synop_rh_pct,
                   syn.wind_speed_kt AS synop_wind_speed,
                   syn.wind_dir_deg AS synop_wind_dir_deg,
                   syn.visibility_km AS synop_visibility,
                   syn.rainfall_24h_mm AS synop_rainfall_24h_mm,
                   syn.solar_rad_24h_jcm2 AS synop_solar_rad_24h
            FROM {SCHEMA}.stg_grid g2
            ASOF LEFT JOIN src.jambi_sch.synop_jambi_combined syn
                ON g2.ts_wib >= syn.waktu
        )
    """)
    print(f"[5] synop t={time.time()-t0:.1f}s  n={con.execute(f'SELECT count(*) FROM {SCHEMA}.stg_synop_10min').fetchone()[0]:,}")

    # 6. Gabung + hitung fitur waktu, lag, rolling, target via window function
    con.execute(f"""
        CREATE OR REPLACE TABLE {SCHEMA}.{TABLE} AS
        WITH base AS (
            SELECT
                g.ts_wib,
                g.ts_wib + INTERVAL 60 MINUTE AS target_ts_wib,
                extract(year  FROM g.ts_wib)::BIGINT AS year,
                extract(month FROM g.ts_wib)::BIGINT AS month,
                extract(day   FROM g.ts_wib)::BIGINT AS day,
                extract(hour  FROM g.ts_wib)::BIGINT AS hour,
                extract(minute FROM g.ts_wib)::BIGINT AS minute,
                sin(2*pi()*(extract(hour FROM g.ts_wib)+extract(minute FROM g.ts_wib)/60.0)/24.0) AS hour_sin,
                cos(2*pi()*(extract(hour FROM g.ts_wib)+extract(minute FROM g.ts_wib)/60.0)/24.0) AS hour_cos,
                sin(2*pi()*extract(month FROM g.ts_wib)/12.0) AS month_sin,
                cos(2*pi()*extract(month FROM g.ts_wib)/12.0) AS month_cos,
                a.ghi_now, a.dhi_now, a.dni_now, a.reflected_now, a.nett_rad_now, a.solar_elev_deg,
                a.asrs_n_obs_1min, a.asrs_ok_obs,
                m.aws_temp_c, m.aws_temp_min_c, m.aws_temp_max_c, m.aws_rh_pct, m.aws_pressure_hpa,
                m.aws_ws_avg, m.aws_ws_max, m.aws_wd_deg, m.aws_rain_mm, m.aws_sr_avg_w_m2,
                c.clp_cot, c.clp_cth_m, c.clp_ctt_k, c.clp_cer, c.clp_cloud_present,
                c.clp_clear_flag, c.clp_thin_cloud_flag, c.clp_moderate_cloud_flag, c.clp_thick_cloud_flag,
                s.synop_temp_c, s.synop_dewpoint_c, s.synop_rh_pct, s.synop_wind_speed,
                s.synop_wind_dir_deg, s.synop_visibility, s.synop_rainfall_24h_mm, s.synop_solar_rad_24h,
                (a.ghi_now IS NOT NULL)      AS has_asrs,
                (m.aws_temp_c IS NOT NULL)   AS has_aws,
                (c.clp_cot IS NOT NULL)      AS has_clp,
                (s.synop_temp_c IS NOT NULL) AS has_synop
            FROM {SCHEMA}.stg_grid g
            LEFT JOIN {SCHEMA}.stg_asrs_10min a  ON a.ts_wib = g.ts_wib
            LEFT JOIN {SCHEMA}.stg_meteo_10min m ON m.ts_wib = g.ts_wib
            LEFT JOIN {SCHEMA}.stg_clp_10min c   ON c.ts_wib = g.ts_wib
            LEFT JOIN {SCHEMA}.stg_synop_10min s ON s.ts_wib = g.ts_wib
        ),
        feat AS (
            SELECT
                base.*,
                LAG(ts_wib,1)  OVER w AS lag_10m_ts,
                LAG(ts_wib,3)  OVER w AS lag_30m_ts,
                LAG(ts_wib,6)  OVER w AS lag_60m_ts,
                LAG(ts_wib,12) OVER w AS lag_120m_ts,
                LAG(ts_wib,18) OVER w AS lag_180m_ts,
                LAG(ghi_now,1)  OVER w AS ghi_lag_10m,
                LAG(ghi_now,3)  OVER w AS ghi_lag_30m,
                LAG(ghi_now,6)  OVER w AS ghi_lag_60m,
                LAG(ghi_now,12) OVER w AS ghi_lag_120m,
                LAG(ghi_now,18) OVER w AS ghi_lag_180m,
                LAG(dhi_now,6) OVER w AS dhi_lag_60m,
                LAG(dni_now,6) OVER w AS dni_lag_60m,
                LAG(aws_temp_c,6) OVER w AS aws_temp_lag_60m,
                LAG(aws_rh_pct,6) OVER w AS aws_rh_lag_60m,
                LAG(aws_pressure_hpa,6) OVER w AS aws_pressure_lag_60m,
                LAG(clp_cot,6) OVER w AS clp_cot_lag_60m,
                LAG(clp_cth_m,6) OVER w AS clp_cth_lag_60m,

                AVG(ghi_now)         OVER w30  AS ghi_roll_30m_mean,
                MIN(ghi_now)         OVER w30  AS ghi_roll_30m_min,
                MAX(ghi_now)         OVER w30  AS ghi_roll_30m_max,
                STDDEV_SAMP(ghi_now) OVER w30  AS ghi_roll_30m_std,
                AVG(ghi_now)         OVER w60  AS ghi_roll_60m_mean,
                MIN(ghi_now)         OVER w60  AS ghi_roll_60m_min,
                MAX(ghi_now)         OVER w60  AS ghi_roll_60m_max,
                STDDEV_SAMP(ghi_now) OVER w60  AS ghi_roll_60m_std,
                AVG(ghi_now)         OVER w180 AS ghi_roll_180m_mean,
                MIN(ghi_now)         OVER w180 AS ghi_roll_180m_min,
                MAX(ghi_now)         OVER w180 AS ghi_roll_180m_max,
                STDDEV_SAMP(ghi_now) OVER w180 AS ghi_roll_180m_std,
                AVG(dhi_now) OVER w180 AS dhi_roll_180m_mean,
                AVG(dni_now) OVER w180 AS dni_roll_180m_mean,
                AVG(aws_temp_c) OVER w180 AS aws_temp_roll_180m_mean,
                AVG(aws_rh_pct) OVER w180 AS aws_rh_roll_180m_mean,
                AVG(aws_ws_avg) OVER w180 AS aws_ws_roll_180m_mean,
                SUM(aws_rain_mm) OVER w180 AS aws_rain_sum_180m,
                AVG(clp_cot) OVER w180 AS clp_cot_roll_180m_mean,
                AVG(clp_cth_m) OVER w180 AS clp_cth_roll_180m_mean,

                LEAD(ghi_now,6) OVER w AS target_ghi_1h_ahead,
                LEAD(ts_wib,6)  OVER w AS observed_target_ts_wib,

                SUM(CASE WHEN has_asrs THEN 1 ELSE 0 END) OVER wprev18 AS n_asrs_valid_prev18
            FROM base
            WINDOW
                w      AS (ORDER BY ts_wib),
                w30    AS (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
                w60    AS (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW),
                w180   AS (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW),
                wprev18 AS (ORDER BY ts_wib ROWS BETWEEN 18 PRECEDING AND 1 PRECEDING)
        )
        SELECT
            ts_wib, target_ts_wib, year, month, day, hour, minute,
            hour_sin, hour_cos, month_sin, month_cos,
            CASE WHEN solar_elev_deg > 0 THEN 1 ELSE 0 END AS daylight_flag,
            CASE WHEN solar_elev_deg > 5 THEN 1 ELSE 0 END AS sun_above_5deg_flag,
            ghi_now, dhi_now, dni_now, reflected_now, nett_rad_now, solar_elev_deg,
            asrs_n_obs_1min, asrs_ok_obs,
            aws_temp_c, aws_temp_min_c, aws_temp_max_c, aws_rh_pct, aws_pressure_hpa,
            aws_ws_avg, aws_ws_max, aws_wd_deg, aws_rain_mm, aws_sr_avg_w_m2,
            clp_cot, clp_cth_m, clp_ctt_k, clp_cer, clp_cloud_present,
            clp_clear_flag, clp_thin_cloud_flag, clp_moderate_cloud_flag, clp_thick_cloud_flag,
            synop_temp_c, synop_dewpoint_c, synop_rh_pct, synop_wind_speed, synop_wind_dir_deg,
            synop_visibility, synop_rainfall_24h_mm, synop_solar_rad_24h,
            has_asrs, has_aws, has_clp, has_synop,
            CASE WHEN has_asrs AND has_clp THEN 'ok'
                 WHEN has_asrs THEN 'partial_no_clp'
                 ELSE 'missing_radiation' END AS master_qc_status,
            CASE WHEN has_asrs THEN 'ok' ELSE 'missing' END AS asrs_qc_status,
            CASE WHEN has_aws  THEN 'ok' ELSE 'missing' END AS aws_qc_status,
            CASE WHEN has_clp  THEN 'ok' ELSE 'missing' END AS clp_qc_status,
            CASE WHEN has_synop THEN 'ok' ELSE 'missing' END AS synop_qc_status,
            observed_target_ts_wib, target_ghi_1h_ahead,
            lag_10m_ts, lag_30m_ts, lag_60m_ts, lag_120m_ts, lag_180m_ts,
            ghi_lag_10m, ghi_lag_30m, ghi_lag_60m, ghi_lag_120m, ghi_lag_180m,
            dhi_lag_60m, dni_lag_60m, aws_temp_lag_60m, aws_rh_lag_60m, aws_pressure_lag_60m,
            clp_cot_lag_60m, clp_cth_lag_60m,
            ghi_roll_30m_mean, ghi_roll_30m_min, ghi_roll_30m_max, ghi_roll_30m_std,
            ghi_roll_60m_mean, ghi_roll_60m_min, ghi_roll_60m_max, ghi_roll_60m_std,
            ghi_roll_180m_mean, ghi_roll_180m_min, ghi_roll_180m_max, ghi_roll_180m_std,
            dhi_roll_180m_mean, dni_roll_180m_mean,
            aws_temp_roll_180m_mean, aws_rh_roll_180m_mean, aws_ws_roll_180m_mean, aws_rain_sum_180m,
            clp_cot_roll_180m_mean, clp_cth_roll_180m_mean,
            (ghi_now - ghi_lag_10m) AS ghi_delta_10m,
            (ghi_now - ghi_lag_60m) AS ghi_delta_60m,
            (aws_temp_c - aws_temp_lag_60m) AS aws_temp_delta_60m,
            (aws_rh_pct - aws_rh_lag_60m)   AS aws_rh_delta_60m,
            CASE WHEN n_asrs_valid_prev18 = 18 THEN 1 ELSE 0 END AS has_continuous_3h_history,
            CASE WHEN n_asrs_valid_prev18 = 18 AND has_asrs AND target_ghi_1h_ahead IS NOT NULL
                 THEN 1 ELSE 0 END AS is_model_ready
        FROM feat
        ORDER BY ts_wib
    """)
    print(f"[6] final table  t={time.time()-t0:.1f}s  n={con.execute(f'SELECT count(*) FROM {SCHEMA}.{TABLE}').fetchone()[0]:,}")

    # 7. Verifikasi
    ncols = con.execute(f"""
        SELECT count(*) FROM information_schema.columns
        WHERE table_schema='{SCHEMA}' AND table_name='{TABLE}'
    """).fetchone()[0]
    rng = con.execute(f"SELECT min(ts_wib), max(ts_wib) FROM {SCHEMA}.{TABLE}").fetchone()
    n_ready = con.execute(f"SELECT count(*) FROM {SCHEMA}.{TABLE} WHERE is_model_ready=1").fetchone()[0]
    n_ready_sun5 = con.execute(
        f"SELECT count(*) FROM {SCHEMA}.{TABLE} WHERE is_model_ready=1 AND sun_above_5deg_flag=1"
    ).fetchone()[0]
    hrs = con.execute(f"""
        SELECT hour, count(*) FROM {SCHEMA}.{TABLE} WHERE is_model_ready=1 AND sun_above_5deg_flag=1
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print(f"[7] VERIFIKASI: kolom={ncols}  rentang={rng}")
    print(f"    is_model_ready=1               -> {n_ready:,} baris")
    print(f"    is_model_ready=1 & sun>5deg     -> {n_ready_sun5:,} baris")
    print(f"    distribusi jam (is_model_ready & sun>5deg):")
    print(hrs.to_string(index=False))

    # bersihkan tabel staging (drop, bukan hapus file -- aman)
    for stg in ["stg_grid", "stg_asrs_10min", "stg_clp_10min", "stg_meteo_10min", "stg_synop_10min"]:
        con.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{stg}")

    con.close()
    print(f"\nSELESAI  t={time.time()-t0:.1f}s")
    print(f"File output: {OUT_DB}")
    print(f"Tabel: {SCHEMA}.{TABLE}")


if __name__ == "__main__":
    main()
