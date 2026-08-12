"""
Sync training_ghi_1h_direct ke MotherDuck — via query lokal + insert cloud
1. Query ALL data dari lokal
2. DROP table di cloud
3. CREATE & INSERT di cloud
"""

import sys
import duckdb
sys.stdout.reconfigure(encoding="utf-8")

LOCAL_DB = r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
CLOUD_DB = "md:kalbar"

print("="*70)
print("SYNC training_ghi_1h_direct: LOCAL → MOTHERDUCK")
print("="*70)

# Step 1: Query data lokal
print("\n[1] Query data dari lokal...")
con_local = duckdb.connect(LOCAL_DB, read_only=True)
df = con_local.execute("""
    SELECT * FROM training_ghi_1h_direct
    ORDER BY timestamp_wib
""").df()
con_local.close()

n_rows, n_cols = df.shape
print(f"  ✓ {n_rows:,} baris × {n_cols} kolom")
print(f"  Kolom: {', '.join(df.columns[:5])}... (dst {n_cols-5})")

# Step 2: Connect ke cloud, DROP & CREATE
print("\n[2] Koneksi ke MotherDuck & persiapkan table...")
con_cloud = duckdb.connect(CLOUD_DB)

# Drop jika ada
try:
    con_cloud.execute("DROP TABLE IF EXISTS kalbar.main.training_ghi_1h_direct")
    print("  ✓ Table lama (jika ada) di-drop")
except Exception as e:
    print(f"  ! Drop: {e}")

# Step 3: INSERT data ke cloud
print("\n[3] Insert data ke cloud kalbar.main.training_ghi_1h_direct...")
try:
    con_cloud.register("temp_df", df)
    con_cloud.execute("""
        CREATE TABLE kalbar.main.training_ghi_1h_direct AS
        SELECT * FROM temp_df
    """)
    print(f"  ✓ Inserted {n_rows:,} baris")
except Exception as e:
    print(f"  ! Insert error: {e}")
    sys.exit(1)

# Step 4: Verifikasi
print("\n[4] Verifikasi di cloud...")
result = con_cloud.execute("""
    SELECT
        COUNT(*) AS n_baris,
        COUNT(DISTINCT timestamp_wib) AS n_unique_ts,
        MIN(timestamp_wib) AS earliest,
        MAX(timestamp_wib) AS latest
    FROM kalbar.main.training_ghi_1h_direct
""").fetchall()

r = result[0]
print(f"  Baris: {r[0]:,}")
print(f"  Unique timestamp: {r[1]:,}")
print(f"  Range: {r[2]} s/d {r[3]}")

# Cek kolom
cols = con_cloud.execute("""
    SELECT column_name, data_type
    FROM kalbar.main.information_schema.columns
    WHERE table_name = 'training_ghi_1h_direct'
    ORDER BY ordinal_position
""").fetchall()
print(f"\n  Kolom (total {len(cols)}):")
for i, (col, dtype) in enumerate(cols, 1):
    print(f"    {i:2d}. {col:<30} {dtype}")

con_cloud.close()

print("\n" + "="*70)
print("✓ SYNC BERHASIL")
print("="*70)
