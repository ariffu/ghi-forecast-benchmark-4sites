"""
Perbaikan tuntas pelanggaran DHI>GHI di solar_kalbar_10m.
Metode: konsisten dengan note_07 (klimatologi kd = DHI/GHI per bulan x bucket sun_altitude),
diterapkan ke SEMUA baris yang masih melanggar (termasuk 2.979 baris yang lolos dari
perbaikan sebelumnya -- kemungkinan karena insiden sync/recovery yang menimpa ulang tabel).
"""
import duckdb
import pandas as pd
import numpy as np

DB_PATH = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"

con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT timestamp_wib, ghi_final, dni_final, dhi_final, dni_corrected,
           reflected_rad, net_rad, sunshine_minutes, sun_altitude, fill_tier, quality_score
    FROM solar_kalbar_10m ORDER BY timestamp_wib
""").df()
con.close()

df["dhi_final_original"] = df["dhi_final"]
df["month"] = df["timestamp_wib"].dt.month
df["alt_bucket"] = (df["sun_altitude"] // 10 * 10).clip(lower=0)

violating = (df["dhi_final"] > df["ghi_final"]) & (df["ghi_final"] > 0)
clean = (~violating) & (df["ghi_final"] > 0) & (df["sun_altitude"] > 5)

df["kd"] = np.where(df["ghi_final"] > 0, df["dhi_final"] / df["ghi_final"], np.nan)
clim = (
    df.loc[clean].groupby(["month", "alt_bucket"])["kd"]
    .median()
    .rename("kd_climatology")
    .reset_index()
)
print(f"Climatology cells terbentuk: {len(clim)}")

df = df.merge(clim, on=["month", "alt_bucket"], how="left")
global_median_kd = df.loc[clean, "kd"].median()
df["kd_climatology"] = df["kd_climatology"].fillna(global_median_kd)

n_violating_before = int(violating.sum())
df["dhi_was_corrected_v2"] = violating
df.loc[violating, "dhi_final"] = (df.loc[violating, "kd_climatology"] * df.loc[violating, "ghi_final"]).clip(
    upper=df.loc[violating, "ghi_final"]
)

zero_ghi_dhi_issue = (df["ghi_final"] == 0) & (df["dhi_final_original"] > 0)
df.loc[zero_ghi_dhi_issue, "dhi_final"] = 0.0
df["dhi_was_corrected_v2"] = df["dhi_was_corrected_v2"] | zero_ghi_dhi_issue

n_violating_after = int((df["dhi_final"] > df["ghi_final"]).sum())
print(f"Baris melanggar sebelum perbaikan : {n_violating_before + int(zero_ghi_dhi_issue.sum())}")
print(f"Baris melanggar setelah perbaikan : {n_violating_after}")
print(f"Baris dikoreksi (v2)              : {int(df['dhi_was_corrected_v2'].sum())}")
print(f"Global median kd (fallback)       : {global_median_kd:.4f}")

df = df.drop(columns=["month", "alt_bucket", "kd", "kd_climatology"])
df.to_parquet(r"C:\Users\ariff\DuckDB_kalbar\solar_kalbar_10m_dhi_fixed.parquet", index=False)
print("Saved: solar_kalbar_10m_dhi_fixed.parquet")
