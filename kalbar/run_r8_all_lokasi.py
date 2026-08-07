#!/usr/bin/env python3
"""
Universal R8 Batch Runner — Execute R8 (Arm A/B/C) on all 4 lokasi
Adapts automatically to each lokasi's database path & configuration
"""

import subprocess
from pathlib import Path

LOKASI_CONFIG = {
    "Kalbar": {
        "script": r"C:\Users\ariff\DuckDB_kalbar\train_ghi_1h_kalbar_R8_comprehensive.py",
        "cwd": r"C:\Users\ariff\DuckDB_kalbar",
        "db": r"C:\Users\ariff\DuckDB_kalbar\kalbar_local.db",
    },
    "Bengkulu": {
        "script": r"C:\Users\ariff\bengkulu_ghi_julius\train_ghi_1h_bengkulu_R8_comprehensive.py",
        "cwd": r"C:\Users\ariff\bengkulu_ghi_julius",
        "db": "bengkulu_db",  # MotherDuck
    },
    "Jambi": {
        "script": r"C:\Users\ariff\DuckDB_jambi\train_ghi_1h_jambi_R8_comprehensive.py",
        "cwd": r"C:\Users\ariff\DuckDB_jambi",
        "db": "jambi.duckdb",
    },
    "Banten": {
        "script": r"C:\Users\ariff\Duckdb_Banten\train_ghi_1h_banten_R8_comprehensive.py",
        "cwd": r"C:\Users\ariff\Duckdb_Banten",
        "db": "banten.duckdb",
    },
}

PYTHON_EXE = r"C:\Program Files\Python39\python.exe"

def run_r8(lokasi):
    """Run R8 for one lokasi."""
    config = LOKASI_CONFIG[lokasi]
    script = Path(config["script"])

    if not script.exists():
        print(f"[{lokasi}] Script not found: {script}")
        return False

    print(f"\n{'='*70}")
    print(f"[{lokasi}] Running R8 Comprehensive Benchmark")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [PYTHON_EXE, str(script)],
            cwd=config["cwd"],
            capture_output=False,
            timeout=3600  # 1 hour max
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[{lokasi}] Error: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("R8 BATCH RUNNER — All Lokasi")
    print("="*70)

    # Run all lokasi sequentially
    results = {}
    for lokasi in ["Kalbar", "Bengkulu", "Jambi", "Banten"]:
        results[lokasi] = run_r8(lokasi)

    # Summary
    print("\n" + "="*70)
    print("BATCH RUN SUMMARY")
    print("="*70)
    for lokasi, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {lokasi:12}: {status}")

    # Aggregate
    all_success = all(results.values())
    if all_success:
        print("\nAll lokasi complete! Running compile_r8_results.py...")
        result = subprocess.run(
            [PYTHON_EXE, r"C:\Users\ariff\DuckDB_kalbar\compile_r8_results.py"],
            cwd=r"C:\Users\ariff\DuckDB_kalbar"
        )
        if result.returncode == 0:
            print("\nR8 BATCH PIPELINE COMPLETE!")
            print("Outputs: r8_compiled/ (Tabel 2a_v2, 2c, 2d)")
    else:
        print("\nSome lokasi failed. Fix & retry.")
