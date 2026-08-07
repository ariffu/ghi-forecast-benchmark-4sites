import duckdb, pandas as pd
con = duckdb.connect(":memory:")
con.execute("ATTACH 'C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb' AS bdb (READ_ONLY)")

df = con.execute("""
    SELECT DATE_TRUNC('month', ts_wib) AS month, COUNT(*) AS n_raw,
           SUM(CASE WHEN asrs_ghi_w_m2 IS NULL THEN 1 ELSE 0 END) AS n_null_ghi,
           SUM(CASE WHEN asrs_ghi_w_m2 BETWEEN 0 AND 1400 THEN 1 ELSE 0 END) AS n_ghi_ok
    FROM bdb.bengkulu_sch.bengkulu_master_10min_quality_final
    WHERE YEAR(ts_wib) = 2021
    GROUP BY 1 ORDER BY 1
""").fetchdf()
print("2021 per-bulan:")
print(df.to_string(index=False))

yr = con.execute("""
    SELECT YEAR(ts_wib) AS yr, COUNT(*) AS n_raw,
           SUM(CASE WHEN asrs_ghi_w_m2 BETWEEN 0 AND 1400 THEN 1 ELSE 0 END) AS n_ghi_ok
    FROM bdb.bengkulu_sch.bengkulu_master_10min_quality_final
    WHERE YEAR(ts_wib) BETWEEN 2021 AND 2025
    GROUP BY 1 ORDER BY 1
""").fetchdf()
print("\nPer tahun raw vs GHI-ok:")
print(yr.to_string(index=False))
con.close()
