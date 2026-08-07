#!/usr/bin/env python3
"""
Compile R1 results dari 4 lokasi (Bengkulu, Jambi, Banten, Kalbar) ke Tabel 2a/2b
"""
import pandas as pd
from pathlib import Path

# Paths untuk outputs R1 setiap lokasi
BENGKULU_DIR = Path(r"C:\Users\ariff\bengkulu_ghi_julius\outputs_R1_bengkulu")
JAMBI_DIR = Path(r"C:\Users\ariff\DuckDB_jambi\outputs_R1_jambi")
BANTEN_DIR = Path(r"C:\Users\ariff\Duckdb_Banten\outputs_R1_banten")
KALBAR_DIR = Path(r"C:\Users\ariff\DuckDB_kalbar\outputs_R1_kalbar")

OUTPUT_DIR = Path(r"C:\Users\ariff\DuckDB_kalbar\r1_compiled")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("COMPILE R1 RESULTS — 4 LOKASI")
print("="*70)

# ── Load data ────────────────────────────────────────────────────────────
data = {}
for name, path in [("Bengkulu", BENGKULU_DIR), ("Jambi", JAMBI_DIR),
                    ("Banten", BANTEN_DIR), ("Kalbar", KALBAR_DIR)]:
    results_file = path / "ghi_1h_R1_results.csv"
    wf_file = path / "ghi_1h_R1_wf_folds.csv"

    if not results_file.exists():
        print(f"  ! {name}: ghi_1h_R1_results.csv tidak ditemukan")
        continue

    df_results = pd.read_csv(results_file)
    df_results["location"] = name

    print(f"  OK {name}: {len(df_results)} rows dari results")

    if wf_file.exists():
        df_wf = pd.read_csv(wf_file)
        df_wf["location"] = name
        data[name] = {"results": df_results, "wf": df_wf}
    else:
        data[name] = {"results": df_results, "wf": None}

print(f"\n  Loaded: {', '.join(data.keys())}")

# ── TABEL 2a: Test 2025 Results (headline) ───────────────────────────────
print("\n" + "="*70)
print("TABEL 2a: Test 2025 Results (point & avg targets)")
print("="*70)

# Combine all results
all_results = pd.concat([data[loc]["results"] for loc in data.keys()], ignore_index=True)

# Pivot untuk tabel yang rapi: rows=location×target, cols=model metrics
tbl2a_rows = []
for location in ["Bengkulu", "Jambi", "Banten", "Kalbar"]:
    if location not in data:
        continue

    df_loc = data[location]["results"]

    # point_t60
    row_pt = df_loc[df_loc["target"] == "point_t60"]
    if len(row_pt) > 0:
        for _, r in row_pt.iterrows():
            tbl2a_rows.append({
                "Location": location,
                "Target": "point_t60",
                "Model": r["model"],
                "n": r["n"],
                "R²": r["r2"],
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "Skill vs SP": r["skill_vs_sp"],
            })

    # avg_t10_t60
    row_av = df_loc[df_loc["target"] == "avg_t10_t60"]
    if len(row_av) > 0:
        for _, r in row_av.iterrows():
            tbl2a_rows.append({
                "Location": location,
                "Target": "avg_t10_t60",
                "Model": r["model"],
                "n": r["n"],
                "R²": r["r2"],
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "Skill vs SP": r["skill_vs_sp"],
            })

tbl2a = pd.DataFrame(tbl2a_rows)
tbl2a = tbl2a.sort_values(["Target", "Location", "Model"])

print("\nTabel 2a: Test 2025 Results")
print(tbl2a.to_string(index=False))

# Save
tbl2a.to_csv(OUTPUT_DIR / "TABLE_2a_test_2025_results.csv", index=False)
print(f"\n-> Saved: {OUTPUT_DIR / 'TABLE_2a_test_2025_results.csv'}")

# ── TABEL 2b: Walk-Forward Summary (LGBM residual, point target) ─────────
print("\n" + "="*70)
print("TABEL 2b: Walk-Forward Summary (LGBM residual × point_t60)")
print("="*70)

tbl2b_rows = []
for location in ["Bengkulu", "Jambi", "Banten", "Kalbar"]:
    if location not in data or data[location]["wf"] is None:
        print(f"  ! {location}: walk-forward data tidak ada")
        continue

    df_wf = data[location]["wf"]

    # Summary stats
    r2_mean = df_wf["r2"].mean()
    r2_std = df_wf["r2"].std()
    mae_mean = df_wf["mae"].mean()
    mae_std = df_wf["mae"].std()
    rmse_mean = df_wf["rmse"].mean()
    rmse_std = df_wf["rmse"].std()
    skill_mean = df_wf["skill_vs_sp"].mean()
    skill_std = df_wf["skill_vs_sp"].std()

    tbl2b_rows.append({
        "Location": location,
        "Metric": "R²",
        "Mean": f"{r2_mean:.4f}",
        "Std": f"{r2_std:.4f}",
    })
    tbl2b_rows.append({
        "Location": location,
        "Metric": "MAE",
        "Mean": f"{mae_mean:.2f}",
        "Std": f"{mae_std:.2f}",
    })
    tbl2b_rows.append({
        "Location": location,
        "Metric": "RMSE",
        "Mean": f"{rmse_mean:.2f}",
        "Std": f"{rmse_std:.2f}",
    })
    tbl2b_rows.append({
        "Location": location,
        "Metric": "Skill vs SP",
        "Mean": f"{skill_mean:.4f}",
        "Std": f"{skill_std:.4f}",
    })

tbl2b = pd.DataFrame(tbl2b_rows)

print("\nTabel 2b: Walk-Forward Summary (5-fold)")
print(tbl2b.to_string(index=False))

# Save
tbl2b.to_csv(OUTPUT_DIR / "TABLE_2b_walkforward_summary.csv", index=False)
print(f"\n-> Saved: {OUTPUT_DIR / 'TABLE_2b_walkforward_summary.csv'}")

# ── Detailed WF results ──────────────────────────────────────────────────
print("\n" + "="*70)
print("Detailed Walk-Forward Results (fold-by-fold)")
print("="*70)

all_wf = []
for location in ["Bengkulu", "Jambi", "Banten", "Kalbar"]:
    if location in data and data[location]["wf"] is not None:
        df = data[location]["wf"].copy()
        df["location"] = location
        all_wf.append(df)

if all_wf:
    tbl2b_detail = pd.concat(all_wf, ignore_index=True)
    tbl2b_detail = tbl2b_detail[["location", "fold", "period", "n_train_eff", "n_test", "best_iter", "r2", "mae", "rmse", "skill_vs_sp"]]
    tbl2b_detail = tbl2b_detail.sort_values(["location", "fold"])

    print(tbl2b_detail.to_string(index=False, max_rows=30))
    tbl2b_detail.to_csv(OUTPUT_DIR / "TABLE_2b_walkforward_detail.csv", index=False)
    print(f"\n-> Saved: {OUTPUT_DIR / 'TABLE_2b_walkforward_detail.csv'}")

print("\n" + "="*70)
print(f"OK Compilation done. Output -> {OUTPUT_DIR}/")
print("="*70)
