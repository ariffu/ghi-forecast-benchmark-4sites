#!/usr/bin/env python
"""
Audit Tabel 1 — Verify anchor_valid counts across all 4 lokasi
"""

import sys
import os

# Try to import duckdb
try:
    import duckdb
except ImportError:
    print("[ERROR] DuckDB not installed. Installing...")
    os.system("pip install duckdb -q")
    import duckdb

LOKASI_DBS = {
    "Kalbar": r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db",
    "Jambi": r"C:\Users\ariff\DuckDB_jambi\jambi.duckdb",
    "Banten": r"C:\Users\ariff\Duckdb_Banten\banten.duckdb",
    # Bengkulu is on MotherDuck
}

QUERY = """
SELECT
  COUNT(*) as total_anchors,
  COUNT(CASE WHEN timestamp_wib < '2024-01-01' THEN 1 END) as train_anchors,
  COUNT(CASE WHEN timestamp_wib >= '2024-01-01' AND timestamp_wib < '2025-01-01' THEN 1 END) as val_anchors,
  COUNT(CASE WHEN timestamp_wib >= '2025-01-01' THEN 1 END) as test_anchors
FROM training_ghi_1h_direct
WHERE anchor_valid = TRUE;
"""

def query_lokasi(lokasi, db_path):
    """Query anchor counts for one lokasi."""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(QUERY).fetchall()
        conn.close()

        if result:
            total, train, val, test = result[0]
            return {
                "total": total,
                "train": train,
                "val": val,
                "test": test,
                "status": "OK"
            }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def main():
    print("="*80)
    print("TABEL 1 AUDIT — All 4 Lokasi Anchor Counts")
    print("="*80)

    results = {}

    # Query local databases
    for lokasi, db_path in LOKASI_DBS.items():
        print(f"\n[{lokasi}] Querying {db_path}...")
        results[lokasi] = query_lokasi(lokasi, db_path)

        if results[lokasi]["status"] == "OK":
            r = results[lokasi]
            print(f"  Total:  {r['total']:,}")
            print(f"  Train:  {r['train']:,}")
            print(f"  Val:    {r['val']:,}")
            print(f"  Test:   {r['test']:,}")
        else:
            print(f"  ERROR: {results[lokasi].get('error', 'Unknown error')}")

    # Try MotherDuck for Bengkulu
    print(f"\n[Bengkulu] Querying MotherDuck (bengkulu_db)...")
    try:
        conn = duckdb.connect("md:kalbar")  # Connect to MotherDuck
        result = conn.execute(f"SELECT * FROM bengkulu.main.training_ghi_1h_direct LIMIT 1").fetchall()
        conn.close()

        # Query Bengkulu
        conn = duckdb.connect("md:kalbar")
        result = conn.execute("""
            SELECT
              COUNT(*) as total_anchors,
              COUNT(CASE WHEN timestamp_wib < '2024-01-01' THEN 1 END) as train_anchors,
              COUNT(CASE WHEN timestamp_wib >= '2024-01-01' AND timestamp_wib < '2025-01-01' THEN 1 END) as val_anchors,
              COUNT(CASE WHEN timestamp_wib >= '2025-01-01' THEN 1 END) as test_anchors
            FROM bengkulu.main.training_ghi_1h_direct
            WHERE anchor_valid = TRUE;
        """).fetchall()
        conn.close()

        if result:
            total, train, val, test = result[0]
            results["Bengkulu"] = {
                "total": total,
                "train": train,
                "val": val,
                "test": test,
                "status": "OK"
            }
            print(f"  Total:  {total:,}")
            print(f"  Train:  {train:,}")
            print(f"  Val:    {val:,}")
            print(f"  Test:   {test:,}")
    except Exception as e:
        print(f"  MotherDuck connection failed: {e}")
        results["Bengkulu"] = {"status": "ERROR", "error": str(e)}

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Lokasi':<15} {'Total':>12} {'Train':>12} {'Val':>12} {'Test':>12} {'Status':<10}")
    print("-" * 80)

    for lokasi in ["Kalbar", "Jambi", "Banten", "Bengkulu"]:
        if lokasi in results:
            r = results[lokasi]
            if r["status"] == "OK":
                print(f"{lokasi:<15} {r['total']:>12,} {r['train']:>12,} {r['val']:>12,} {r['test']:>12,} {r['status']:<10}")
            else:
                print(f"{lokasi:<15} {'ERROR':<12} {'—':>12} {'—':>12} {'—':>12} {r.get('error',''):<10}")

    # Comparison with Kalbar
    print("\n" + "="*80)
    print("COMPARISON vs KALBAR (reference)")
    print("="*80)

    if "Kalbar" in results and results["Kalbar"]["status"] == "OK":
        kalbar_total = results["Kalbar"]["total"]
        print(f"\n{'Lokasi':<15} {'Total':>12} {'Delta':>12} {'Delta%':>12}")
        print("-" * 80)

        for lokasi in ["Kalbar", "Jambi", "Banten", "Bengkulu"]:
            if lokasi in results and results[lokasi]["status"] == "OK":
                total = results[lokasi]["total"]
                delta = total - kalbar_total
                delta_pct = (delta / kalbar_total) * 100 if kalbar_total > 0 else 0

                symbol = "=" if delta == 0 else ("+" if delta > 0 else "")
                print(f"{lokasi:<15} {total:>12,} {symbol}{delta:>11,} {delta_pct:>11.1f}%")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
