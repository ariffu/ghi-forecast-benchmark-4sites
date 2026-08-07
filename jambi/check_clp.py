import duckdb
con = duckdb.connect('jambi.duckdb', read_only=True)

q1 = "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema='jambi_sch'"
print('Tables:')
print(con.execute(q1).df().to_string())
print()

for tbl in ['jambi_clp_combined','clp_jambi']:
    q = "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='jambi_sch' AND table_name='" + tbl + "'"
    df = con.execute(q).df()
    if len(df):
        print('Columns in ' + tbl + ':')
        print(df.to_string(index=False))
        print()
        # Sample row count and non-null of key columns
        q2 = "SELECT COUNT(*) as n FROM jambi_sch." + tbl
        print('Rows:', con.execute(q2).fetchone()[0])

con.close()
