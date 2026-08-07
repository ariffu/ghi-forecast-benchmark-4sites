#!/usr/bin/env python3
"""
Compile R8 results dari 4 lokasi -> Tabel 2a/2b/2c/2d updates
Simplified version: Arm A (meteo) + Arm B (GBM only)
"""

import pandas as pd
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS (Auto-detect outputs_R8_* in each lokasi)
# ─────────────────────────────────────────────────────────────────────────────

LOCATIONS = {
    "Bengkulu": Path(r"C:\Users\ariff\bengkulu_ghi_julius\outputs_R8_Bengkulu"),
    "Jambi": Path(r"C:\Users\ariff\DuckDB_jambi\outputs_R8_Jambi"),
    "Banten": Path(r"C:\Users\ariff\Duckdb_Banten\outputs_R8_Banten"),
    "Kalbar": Path(r"C:\Users\ariff\DuckDB_kalbar\outputs_R8_Kalbar"),
}

COMPILE_DIR = Path(r"C:\Users\ariff\DuckDB_kalbar\r8_compiled")
COMPILE_DIR.mkdir(exist_ok=True)

print("="*70)
print("COMPILE R8 RESULTS — 4 LOKASI")
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARM A/B/C FROM ALL LOKASI
# ─────────────────────────────────────────────────────────────────────────────

all_arm_a = []
all_arm_b = []
all_arm_b_summary = []
all_arm_c = []

for loc_name, loc_dir in LOCATIONS.items():
    print(f"\n{loc_name}:")

    # Arm A
    try:
        df = pd.read_csv(loc_dir / "arm_A_results.csv")
        df["location"] = loc_name
        all_arm_a.append(df)
        print(f"  OK Arm A: {len(df)} rows")
    except Exception as e:
        print(f"  ! Arm A: {e}")

    # Arm B
    try:
        df = pd.read_csv(loc_dir / "arm_B_results.csv")
        df["location"] = loc_name
        all_arm_b.append(df)
        print(f"  OK Arm B: {len(df)} rows")
    except Exception as e:
        print(f"  ! Arm B: {e}")

    # Arm B Summary
    try:
        df = pd.read_csv(loc_dir / "arm_B_summary.csv", index_col=0)
        df["location"] = loc_name
        all_arm_b_summary.append(df)
        print(f"  OK Arm B Summary")
    except Exception as e:
        print(f"  ! Arm B Summary: {e}")

    # Arm C
    try:
        df = pd.read_csv(loc_dir / "arm_C_features.csv")
        df["location"] = loc_name
        all_arm_c.append(df)
        print(f"  OK Arm C: {len(df)} rows")
    except Exception as e:
        print(f"  ! Arm C: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TABEL 2a_v2: FEATURE ENGINEERING (F1 vs F2)
# ─────────────────────────────────────────────────────────────────────────────

if all_arm_a:
    print("\n" + "="*70)
    print("TABEL 2a_v2: FEATURE ENGINEERING IMPACT (F1 vs F2)")
    print("="*70)

    tbl2a_v2 = pd.concat(all_arm_a, ignore_index=True)

    # Pivot: ΔR² = R²_F2 - R²_F1
    delta_r2 = []
    for loc in LOCATIONS.keys():
        for target in ["point_t60", "avg_t10_t60"]:
            f1_r2 = tbl2a_v2[(tbl2a_v2["location"] == loc) & (tbl2a_v2["target"] == target) &
                             (tbl2a_v2["features"] == "F1")]["r2"].values
            f2_r2 = tbl2a_v2[(tbl2a_v2["location"] == loc) & (tbl2a_v2["target"] == target) &
                             (tbl2a_v2["features"] == "F2")]["r2"].values

            if len(f1_r2) > 0 and len(f2_r2) > 0:
                delta_r2.append({
                    "Location": loc,
                    "Target": target,
                    "R2_F1": round(float(f1_r2[0]), 4),
                    "R2_F2": round(float(f2_r2[0]), 4),
                    "Delta_R2": round(float(f2_r2[0]) - float(f1_r2[0]), 4),
                    "Meteo_Value": "high" if (float(f2_r2[0]) - float(f1_r2[0])) > 0.010 else
                                  ("medium" if (float(f2_r2[0]) - float(f1_r2[0])) > 0.005 else "low"),
                })

    tbl2a_v2_pivot = pd.DataFrame(delta_r2)
    print("\nMeteo Feature Contribution (ΔR² = R²_F2 - R²_F1):")
    print(tbl2a_v2_pivot.to_string(index=False))
    tbl2a_v2_pivot.to_csv(COMPILE_DIR / "TABLE_2a_v2_feature_engineering.csv", index=False)

    # Interpretation
    print("\nInterpretation:")
    high_meteo = tbl2a_v2_pivot[tbl2a_v2_pivot["Meteo_Value"] == "high"]
    if len(high_meteo) > 0:
        print(f"  High meteo value (ΔR² > 0.010): {len(high_meteo)} cases")
        print("  → AWS meteo NOT redundant; valuable for {these cases}")
    else:
        print("  No high meteo value cases → AWS meteo redundant across all lokasi")

    print(f"\n→ Saved: TABLE_2a_v2_feature_engineering.csv")

# ─────────────────────────────────────────────────────────────────────────────
# TABEL 2c: MODEL ARCHITECTURE COMPARISON (5-seed mean ± std)
# ─────────────────────────────────────────────────────────────────────────────

if all_arm_b:
    print("\n" + "="*70)
    print("TABEL 2c: MODEL ARCHITECTURE COMPARISON (5-seed DL mean ± std)")
    print("="*70)

    tbl_arm_b = pd.concat(all_arm_b, ignore_index=True)

    # Summary by location & model
    summary_rows = []
    for loc in LOCATIONS.keys():
        df_loc = tbl_arm_b[tbl_arm_b["location"] == loc]

        for model in df_loc["model"].unique():
            df_model = df_loc[df_loc["model"] == model]

            if model in ["catboost", "lgbm"]:  # Single seed (0)
                df_model = df_model[df_model["seed"] == 0]

            r2_mean = df_model["r2"].mean()
            r2_std = df_model["r2"].std()
            mae_mean = df_model["mae"].mean()
            mae_std = df_model["mae"].std()
            rmse_mean = df_model["rmse"].mean()
            rmse_std = df_model["rmse"].std()

            summary_rows.append({
                "Location": loc,
                "Model": model,
                "R2_Mean": round(r2_mean, 4),
                "R2_Std": round(r2_std, 4),
                "MAE_Mean": round(mae_mean, 1),
                "MAE_Std": round(mae_std, 1),
                "RMSE_Mean": round(rmse_mean, 1),
                "RMSE_Std": round(rmse_std, 1),
            })

    tbl2c = pd.DataFrame(summary_rows)
    print("\nModel Comparison (5-seed mean ± std for DL, single seed for GBM):")
    print(tbl2c.to_string(index=False))
    tbl2c.to_csv(COMPILE_DIR / "TABLE_2c_model_architecture.csv", index=False)

    # Interpretation
    print("\nInterpretation:")
    gbm_best = tbl2c[tbl2c["Model"].isin(["catboost", "lgbm"])].groupby("Location")["R2_Mean"].max()
    dl_best = tbl2c[~tbl2c["Model"].isin(["catboost", "lgbm"])].groupby("Location")["R2_Mean"].max()

    for loc in LOCATIONS.keys():
        if loc in gbm_best.index and loc in dl_best.index:
            gap = gbm_best[loc] - dl_best[loc]
            print(f"  {loc}: GBM-DL gap = {gap:+.4f} (GBM better" if gap > 0 else f"  {loc}: DL-GBM gap = {-gap:+.4f} (DL better")

    print(f"\n→ Saved: TABLE_2c_model_architecture.csv")

# ─────────────────────────────────────────────────────────────────────────────
# TABEL 2d: FEATURE PRUNING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

if all_arm_c:
    print("\n" + "="*70)
    print("TABEL 2d: FEATURE PRUNING (Arm C summary)")
    print("="*70)

    # Aggregate pruning results
    pruning_summary = []
    for loc in LOCATIONS.keys():
        df_loc = pd.concat([df[df["location"] == loc] for df in all_arm_c if loc in df["location"].values], ignore_index=True)

        if len(df_loc) > 0:
            n_selected = len(df_loc[df_loc["selected"] == True])
            selected_features = df_loc[df_loc["selected"] == True]["feature"].tolist()

            pruning_summary.append({
                "Location": loc,
                "N_Pruned": n_selected,
                "N_Total": len(df_loc),
                "Reduction_Pct": round(100.0 * (1 - n_selected / len(df_loc)), 1),
                "Top_Features": ", ".join(selected_features[:5]),
            })

    if pruning_summary:
        tbl2d = pd.DataFrame(pruning_summary)
        print("\nFeature Pruning Summary:")
        print(tbl2d.to_string(index=False))
        tbl2d.to_csv(COMPILE_DIR / "TABLE_2d_pruning_summary.csv", index=False)
        print(f"\n→ Saved: TABLE_2d_pruning_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("R8 COMPILATION COMPLETE")
print("="*70)
print(f"Outputs → {COMPILE_DIR}/")
print("\nNew tables generated:")
print("  - TABLE_2a_v2_feature_engineering.csv (meteo contribution)")
print("  - TABLE_2c_model_architecture.csv (GBM vs DL comparison)")
print("  - TABLE_2d_pruning_summary.csv (feature reduction)")
