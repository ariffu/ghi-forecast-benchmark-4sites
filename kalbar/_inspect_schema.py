import duckdb
con = duckdb.connect("kalbar_local.db", read_only=True)
schema = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='training_ghi_1h_direct' ORDER BY ordinal_position").fetchall()
print("Kolom Kalbar database (66 total):")
for i, (col, dtype) in enumerate(schema, 1):
    print(f"  {i:2d}. {col:<30} {dtype}")
con.close()
