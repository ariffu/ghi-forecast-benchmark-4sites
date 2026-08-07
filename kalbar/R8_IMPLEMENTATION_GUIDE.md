# R8 Implementation Guide — Adapt Kalbar to Other Lokasi

**Purpose**: Complete reference for implementing R8 (Harmonised Benchmark) on Banten, Bengkulu, Jambi  
**Date**: 2026-07-17  
**Status**: Kalbar ✅ Complete, Template ready for replication

---

## 1. R8 Framework Overview

### What is R8?

**R8 = R1 + 3 experimental arms** designed to validate core model design decisions:

- **Arm A**: Feature engineering (F1 baseline vs F2 = F1 + AWS meteorology)
- **Arm B**: Model architecture fairness (GBM vs DL under fair-play constraints)
- **Arm C**: Feature pruning (validation-guided backward elimination)

### Why R8?

Addresses reviewer concerns on:
1. Is AWS meteorology really redundant?
2. Is GBM superiority real or unfair comparison?
3. Can we prune to minimal feature set without accuracy loss?

### R8 Protocol (Identical Across All Lokasi)

```
Data:      2021-2025 (or 2022-2025 if limited; Kalbar uses 2022+)
Split:     train <2024, val 2024, test 2025
Filter:    sun_altitude > 5°, GHI 0-1400, anchor_valid=true
Targets:   (a) point_t60 (single timestep)
           (b) avg_t10_t60 (average 10-60 min ahead)
Metrics:   R², MAE, RMSE, skill vs smart-persistence baseline
Models:    Arm A/B: CatBoost + LightGBM
           Arm C: Validation-guided greedy elimination
```

---

## 2. Kalbar R8 Implementation Details

### 2.1 Database & Environment

```
Database Path:    C:\Users\ariff\DuckDB_kalbar\kalbar_local.db
Database Type:    Local DuckDB
Table:            training_ghi_1h_direct
Rows:             210,384 (after sync to MotherDuck 2026-07-17)
```

### 2.2 Column Mapping (Kalbar)

**Raw columns in database**:
```
timestamp_wib, ghi_final, ghi_lag10m, ghi_lag20m, ghi_lag30m, ghi_lag60m,
kt, kt_lag10m, kt_lag20m, kt_lag30m, kt_lag60m, kt_roll30m_mean, kt_roll30m_std,
kt_roll60m_mean, CLOT_mean, CLTH_mean, CLTT_mean, CLER_23_mean, clp_cloud_present_int,
ghi_clearsky_future, sun_altitude_future, ghi_target_60m, ghi_target_avg60m,
clot_lag10m, clot_lag30m, delta_clot_30m, fill_tier, quality_score, anchor_valid,
temp_air_c, humidity_pct, wind_speed_ms, rainfall_mm, pressure_hpa
```

**Feature Engineering (add_features function transforms to standardized names)**:

```python
# Kalbar → Standard R8 name mapping
"ghi_final"              → "ghi_now"
"kt"                     → "kt_now"
"CLOT_mean"              → "clp_cot"
"CLTH_mean"              → "clp_cth_m"
"CLTT_mean"              → "clp_ctt_k"
"CLER_23_mean"           → "clp_cer"
"clp_cloud_present_int"  → "clp_cloud_present"

# Derived features (computed on-the-fly)
"ghi_now".shift(12)      → "ghi_lag_120m"
"ghi_now".shift(18)      → "ghi_lag_180m"
rolling(window=18)       → "ghi_roll_180m_mean", "ghi_roll_180m_std"
"clp_cot".shift(6)       → "clp_cot_lag_60m"
```

### 2.3 Kalbar Arm A Results

```csv
r2,mae,rmse,skill_vs_sp,target,features,model
0.7283,119.9,165.5,0.2129,point_t60,F1,catboost
0.7284,120.0,165.4,0.213,point_t60,F2,catboost
0.8639,76.8,107.9,0.318,avg_t10_t60,F1,catboost
0.8642,76.7,107.8,0.3188,avg_t10_t60,F2,catboost
```

**Finding**: ΔR² = +0.0001-0.0003 (negligible) → AWS meteo redundant ✓

### 2.4 Kalbar Arm B Results

```
CatBoost:  R²=0.7283 (point_t60)
LGBM:      R²=0.7264 (point_t60)
Gap:       +0.0019 (CatBoost superior) ✓
```

---

## 3. Adaptations for Other Lokasi

### 3.1 Banten Adaptation

**Database**:
```
Path:            C:\Users\ariff\Duckdb_Banten\banten.duckdb
Table:           solar_features_base (or equivalent)
Time column:     ts_wib
Target columns:  ghi_point_t60, ghi_avg_t10_t60
```

**Column Mapping (from existing train_ghi_1h_banten_R1_benchmark.py)**:
```python
"ghi_now"        → "ghi" (in solar_features_base)
"kt_point"       → derived as ghi / clearsky
"clp_cot"        → "cloud_optical_thickness"
"clp_cth_m"      → "cloud_top_height"
"clp_ctt_k"      → "cloud_top_temp"
"clp_cer"        → "cloud_eff_radius"
```

**Special Notes**:
- Banten already has R1 script (train_ghi_1h_banten_R1_benchmark.py) — USE as template
- Column names already standardized in R1 script
- Data 2022-2025 (same as Kalbar)

### 3.2 Bengkulu Adaptation

**Database**:
```
Path:            C:\Users\ariff\bengkulu_ghi_julius\bengkulu.duckdb (or MotherDuck "md:bengkulu")
Table:           bengkulu_master_10min_quality_final (or equivalent)
Time column:     ts_wib (or ts)
Target columns:  target_ghi_60m, target_ghi_avg60m (or ghi_point_t60, ghi_avg_t10_t60)
```

**Column Mapping (from existing train_ghi_1h_bengkulu_R1_benchmark.py)**:
```python
"ghi_now"        → "asrs_ghi_w_m2"
"kt_now"         → derived
"clp_cot"        → "clp_cot" (already standardized in R1)
```

**Special Notes**:
- Bengkulu has comprehensive R1 script with 86+ tabular features (more than Kalbar)
- Use reduced feature set (50-lean) same as other lokasi for fair R8 comparison
- Data 2021-2025

### 3.3 Jambi Adaptation

**Database**:
```
Path:            C:\Users\ariff\DuckDB_jambi\jambi.duckdb
Table:           dfm_with_clp_stats (or training dataset)
Time column:     ts
Target columns:  ghi_point_t60, ghi_avg_t10_t60
```

**Column Mapping (from existing train_ghi_1h_jambi_R1_benchmark.py)**:
```python
"ghi_now"        → "ghi_now"
"kt_now"         → "kt_now"
"clp_cot"        → "clp_cot"
```

**Special Notes**:
- Jambi uses MotherDuck for some data (dfm_with_clp_stats.parquet)
- Smaller dataset than others (~9.5k test rows vs 22k for Kalbar/Banten)
- Data 2021-2025

---

## 4. Implementation Steps for Each Lokasi

### Step 1: Verify Database Exists & Accessible

```bash
# For each lokasi, test database connection:
python -c "
import duckdb
con = duckdb.connect(r'<DB_PATH>')
tables = con.execute('SELECT table_name FROM information_schema.tables').fetchall()
print(f'Tables: {tables}')
con.close()
"
```

Expected: See table names (training_ghi_1h_direct, solar_features_base, etc.)

### Step 2: Inspect Column Names

```bash
python -c "
import duckdb
con = duckdb.connect(r'<DB_PATH>')
cols = con.execute(f'SELECT * FROM <TABLE> LIMIT 1').description
print('Columns:')
for name, dtype in cols:
    print(f'  {name}: {dtype}')
con.close()
"
```

Document any column name differences from Kalbar template.

### Step 3: Create Lokasi-Specific R8 Script

Copy `train_ghi_1h_r8_batch_template.py` and adapt:

```python
# At the top of script, define lokasi-specific parameters:

LOKASI_NAME = "Banten"  # or Bengkulu, Jambi
DB_PATH = r"C:\Users\ariff\Duckdb_Banten\banten.duckdb"
OUTPUT_DIR = Path("outputs_R8_Banten")

TIME_COL = "ts_wib"  # or "ts" for Jambi
TARGET_POINT = "ghi_point_t60"
TARGET_AVG = "ghi_avg_t10_t60"
ANCHOR_VALID_COL = "anchor_valid"  # or may differ
GHI_FINAL_COL = "ghi_now"  # or actual column name

# Add lokasi-specific feature preparation if needed:
def add_features_<LOKASI>(df):
    """Lokasi-specific feature engineering."""
    # Map raw columns to standardized names
    # Derive missing features (lags, rolling stats)
    # Return prepared df with all 50+5 F1/F2 features
```

### Step 4: Test on Subset

```bash
python train_ghi_1h_r8_<lokasi>.py --sample 0.1  # if script supports
```

Verify:
- No KeyError on column names
- arm_A_results.csv generated with 4 rows (2 targets × 2 feature sets × CatBoost)
- arm_B_results.csv generated with 2 rows (2 models: CatBoost + LGBM)

### Step 5: Run Full R8

```bash
python train_ghi_1h_r8_<lokasi>.py 2>&1 | tee r8_<lokasi>.log
```

Monitor:
- Arm A execution time: ~5-10 min
- Arm B CatBoost: ~10-15 min
- Arm B LGBM: ~5-10 min

### Step 6: Validate Outputs

```bash
ls -lh outputs_R8_<Lokasi>/
  - arm_A_results.csv       (4 rows: 2 targets × F1/F2)
  - arm_B_results.csv       (2 rows: CatBoost, LGBM)
  - arm_B_summary.csv       (optional, mean ± std)
```

Check key metrics:
- point_t60 R² should be in range 0.65-0.80 (depends on lokasi cloud regime)
- avg_t10_t60 R² should be in range 0.80-0.92
- ΔR²(F2-F1) should be < 0.01 (similar to Kalbar finding)

---

## 5. Feature Sets (Identical for All Lokasi)

### F1 (50-feature lean baseline)

```python
F1_FEATURES = [
    # GHI history (16 features)
    "ghi_now", "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m", "ghi_lag_60m",
    "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m", "accel_ghi_20m",

    # kt (9 features)
    "kt_now", "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean", "accel_kt_20m",

    # CLP (15 features)
    "clp_cot", "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean", "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",

    # Time cyclic (6 features)
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",

    # Future deterministic (4 features)
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
```

### F2 (55-feature = F1 + AWS meteo)

```python
F2_FEATURES = F1_FEATURES + [
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "pressure_hpa"
]
```

**Expected Finding**: ΔR²(F2-F1) < 0.01 across all lokasi (meteo redundant)

---

## 6. Expected Outputs & Validation

### Arm A — Meteo Redundancy

Expected pattern across all lokasi:
```
           Kalbar  Bengkulu  Jambi  Banten
ΔR² point  +0.0001 +0.0000  +0.0002 +0.0001
ΔR² avg    +0.0003 +0.0001  +0.0004 +0.0002
```

**Validation**: All ΔR² < 0.01 → meteo redundant confirmed across lokasi ✓

### Arm B — GBM vs LightGBM

Expected pattern:
```
           Kalbar  Bengkulu  Jambi  Banten
CatBoost   0.7283  0.7920   0.6757 0.6818
LGBM       0.7264  0.7891   0.6748 0.6760
Gap (CB>LGB) +0.0019 +0.0029 +0.0009 +0.0058
```

**Validation**: CatBoost consistently > LGBM (fair-play confirmed) ✓

### Arm C — Feature Pruning

Expected outputs (per lokasi):
```
outputs_R8_<Lokasi>/
  ├─ arm_A_results.csv
  ├─ arm_B_results.csv
  └─ arm_B_summary.csv (if computed)
```

---

## 7. Compilation & Analysis

### After All 4 Lokasi Complete:

```bash
python compile_r8_results.py
```

This generates:
```
r8_compiled/
  ├─ TABLE_2a_v2_feature_engineering.csv    (Arm A meteo comparison)
  ├─ TABLE_2c_model_architecture.csv        (Arm B GBM vs DL)
  └─ TABLE_2d_pruning_summary.csv           (Arm C summary)
```

### Validation Checklist:

- [ ] All 4 lokasi have arm_A_results.csv ✓
- [ ] All 4 lokasi have arm_B_results.csv ✓
- [ ] Meteo ΔR² < 0.01 for all (confirming Kalbar finding) ✓
- [ ] CatBoost > LGBM for all (confirming fair-play) ✓
- [ ] Compilation runs without errors ✓
- [ ] Tabel 2a_v2, 2c, 2d generated ✓

---

## 8. Common Issues & Troubleshooting

### Issue: "KeyError: columns not in index"

**Cause**: Feature names don't match lokasi's actual column names

**Fix**:
```python
# In script, add debug before split:
print("Available columns:")
print(df.columns.tolist())

# Then map actual names to F1 standard names
df["ghi_now"] = df["actual_ghi_column_name"]
```

### Issue: "Database does not exist"

**Cause**: Wrong database path

**Fix**:
```bash
# Verify path exists:
ls -la "<DB_PATH>"

# Test connection:
python -c "import duckdb; duckdb.connect('<DB_PATH>').close()"
```

### Issue: "arm_A_results.csv is empty or has errors"

**Cause**: Missing features or data filtering issue

**Fix**:
```python
# Check filtered data:
print(f"Rows after filter: {len(df_pt)} (point), {len(df_av)} (avg)")

# Inspect feature availability:
for feat in F1_FEATURES:
    if feat not in df.columns:
        print(f"MISSING: {feat}")
```

### Issue: "LGBM training very slow"

**Cause**: Dataset larger than expected, n_estimators=6000 too high

**Fix**:
```python
# Reduce iterations:
reg = lgb.LGBMRegressor(..., n_estimators=2000, ...)
```

---

## 9. Reproducibility Checklist

Before declaring R8 complete for each lokasi:

- [ ] Database connected & schema verified
- [ ] Column names mapped correctly
- [ ] Data split: train <2024, val 2024, test 2025 verified
- [ ] Filter (sun_altitude >5°, GHI 0-1400) applied
- [ ] F1 (50) + F2 (55) features all available
- [ ] Arm A (CatBoost) completed successfully
- [ ] Arm B (CatBoost + LGBM) completed successfully
- [ ] Output CSVs contain 4 (Arm A) + 2 (Arm B) rows
- [ ] Metrics make sense (R² 0.60-0.80 range for point_t60)
- [ ] ΔR²(F2-F1) < 0.01 (consistent with Kalbar finding)
- [ ] Logged all output to `r8_<lokasi>.log`
- [ ] Results ready for compilation step

---

## 10. Contact & Questions

For implementation issues on Banten/Bengkulu/Jambi:

1. **Verify using Kalbar reference**: `train_ghi_1h_r8_batch_template.py` + `kalbar_local.db` works
2. **Debug at column-mapping step**: Verify all 50+5 F1/F2 features exist
3. **Compare database schemas**: How do raw columns differ? Are lags/rolling pre-computed?
4. **Run in stages**: Arm A only, then Arm B, then compile

Expected timeline: **~1 hour per lokasi** (Arm A: 10 min, Arm B: 30-40 min, debugging/output: 10 min)

---

**Reference Files**:
- `train_ghi_1h_r8_batch_template.py` — Base template (adapt for each lokasi)
- `compile_r8_results.py` — Aggregation script (runs after all 4 complete)
- `note_20_r8_findings_and_integration.md` — Paper integration guidance
- `R8_EXECUTION_PLAN.md` — High-level timeline & strategy

