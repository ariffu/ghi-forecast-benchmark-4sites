import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect('jambi.duckdb', read_only=True)

# Check timestamp format and sample data in CLP
q = """
SELECT
    timestamp,
    cloud_present, cloud_class_code,
    CLOT_mean, CLOT_std, CLOT_median, CLOT_coverage,
    CLER_23_coverage, CLER_23_std, CLER_23_mean,
    CLTH_mean, CLTH_std, CLTH_median,
    CLTT_mean, CLTT_std, CLTT_median,
    quality_flag
FROM jambi_sch.jambi_clp_combined
ORDER BY timestamp
LIMIT 5
"""
print("Sample CLP rows:")
print(con.execute(q).df().to_string())
print()

# Check date range and coverage
q2 = """
SELECT
    MIN(timestamp) as ts_min,
    MAX(timestamp) as ts_max,
    COUNT(*) as n_total,
    COUNT(CLOT_std) as n_clot_std,
    COUNT(CLER_23_coverage) as n_cler_cov,
    COUNT(CLOT_median) as n_clot_med
FROM jambi_sch.jambi_clp_combined
"""
print("CLP coverage:")
print(con.execute(q2).df().to_string())
print()

# Export all needed CLP stats
q3 = """
SELECT
    timestamp,
    cloud_present, cloud_class_code,
    CLOT_mean, CLOT_std, CLOT_median, CLOT_coverage,
    CLER_23_coverage, CLER_23_std, CLER_23_mean,
    CLTH_mean, CLTH_std, CLTH_median,
    CLTT_mean, CLTT_std, CLTT_median
FROM jambi_sch.jambi_clp_combined
ORDER BY timestamp
"""
df_clp = con.execute(q3).df()
con.close()

print(f"CLP rows exported: {len(df_clp)}")
print(f"Date range: {df_clp['timestamp'].min()} to {df_clp['timestamp'].max()}")
print()
print("Null rates:")
for c in df_clp.columns:
    if c != 'timestamp':
        print(f"  {c}: {df_clp[c].isna().mean()*100:.1f}%")

# Save
df_clp.to_parquet('clp_stats_jambi.parquet', index=False)
print("\nSaved clp_stats_jambi.parquet")
