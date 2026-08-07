# R8 Implementation Checklist — Banten, Bengkulu, Jambi

**Purpose**: Quick-reference for implementing R8 on each lokasi  
**Time Estimate**: ~1 hour per lokasi  
**Template Reference**: `train_ghi_1h_r8_batch_template.py` + `R8_IMPLEMENTATION_GUIDE.md`

---

## PRE-FLIGHT (15 min)

### Database Verification

- [ ] **Banten**
  - [ ] Database path: `C:\Users\ariff\Duckdb_Banten\banten.duckdb` exists
  - [ ] Can connect: `duckdb.connect(path)` works
  - [ ] Main table: `solar_features_base` (or check with `SELECT table_name FROM information_schema.tables`)
  - [ ] Time column: `ts_wib`
  - [ ] Targets: `ghi_point_t60`, `ghi_avg_t10_t60` exist
  - [ ] Data range: 2022-2025
  - [ ] Row count (approx): ~90k

- [ ] **Bengkulu**
  - [ ] Database path: `C:\Users\ariff\bengkulu_ghi_julius\bengkulu.duckdb` OR MotherDuck
  - [ ] Can connect
  - [ ] Main table: `bengkulu_master_10min_quality_final` (verify)
  - [ ] Time column: `ts_wib` (or similar)
  - [ ] Targets: Check how targets named (ghi_target_60m? ghi_point_t60?)
  - [ ] Data range: 2021-2025
  - [ ] Row count (approx): ~60k

- [ ] **Jambi**
  - [ ] Database path: `C:\Users\ariff\DuckDB_jambi\jambi.duckdb`
  - [ ] Can connect
  - [ ] Main table: `dfm_with_clp_stats` (or training table)
  - [ ] Time column: `ts`
  - [ ] Targets: `ghi_point_t60`, `ghi_avg_t10_t60`
  - [ ] Data range: 2021-2025
  - [ ] Row count (approx): ~22k

### Column Mapping Verification

For each lokasi, run:
```bash
python -c "
import duckdb
con = duckdb.connect(r'<DB_PATH>')
schema = con.execute('SELECT column_name, data_type FROM information_schema.columns WHERE table_name=\"<TABLE>\" LIMIT 80').fetchall()
for col, dtype in schema:
    print(f'{col}: {dtype}')
con.close()
"
```

Then map to standard R8 names:
- [ ] `ghi_*` columns: identify
- [ ] `kt_*` columns: identify
- [ ] `clp_*` or `cloud_*` columns: identify
- [ ] Time column: identify exact name
- [ ] Target columns: identify exact name(s)

---

## SETUP (10 min)

### Create Lokasi-Specific R8 Script

For **Banten**:
```bash
cp train_ghi_1h_r8_batch_template.py train_ghi_1h_banten_r8.py
```

Edit `train_ghi_1h_banten_r8.py`:
```python
# Line ~20-30: Update paths & names
LOKASI_NAME = "Banten"
DB_PATH = r"C:\Users\ariff\Duckdb_Banten\banten.duckdb"
OUTPUT_DIR = Path("outputs_R8_Banten")
TIME_COL = "ts_wib"
TARGET_POINT = "ghi_point_t60"
TARGET_AVG = "ghi_avg_t10_t60"

# Line ~300: If needed, add lokasi-specific feature prep
def add_features(df):
    # Map raw columns → standard names
    df["ghi_now"] = df["ghi"]  # EXAMPLE
    # ... etc
```

Repeat for Bengkulu (`train_ghi_1h_bengkulu_r8.py`) and Jambi (`train_ghi_1h_jambi_r8.py`)

### Copy Template to Lokasi Directories

```bash
cp train_ghi_1h_r8_banten.py C:\Users\ariff\Duckdb_Banten\
cp train_ghi_1h_r8_bengkulu.py C:\Users\ariff\bengkulu_ghi_julius\
cp train_ghi_1h_r8_jambi.py C:\Users\ariff\DuckDB_jambi\
```

---

## EXECUTION (30 min per lokasi)

### Test on Subset First (Optional, 5 min)

```bash
cd <LOKASI_DIR>
python train_ghi_1h_r8_<lokasi>.py --sample-fraction 0.1 2>&1 | head -50
```

Expected output:
```
R8 BATCH RUN — <Lokasi>
Loading data...
Loaded: XXX rows
Filtered: point=YYY, avg=YYY
ARM A: FEATURE ENGINEERING...
  CatBoost F1: R²=...
  CatBoost F2: R²=...
ARM B: MODEL COMPARISON...
  CatBoost: R²=...
  LGBM: R²=...
-> Saved: arm_A_results.csv
```

### Full Run (25 min)

```bash
cd <LOKASI_DIR>
python train_ghi_1h_r8_<lokasi>.py 2>&1 | tee r8_<lokasi>_log.txt
```

Monitor:
- [ ] Data loading: < 1 min
- [ ] Arm A execution: 10-15 min (2 targets × 2 feature sets)
- [ ] Arm B execution: 15-20 min (CatBoost + LGBM)
- [ ] No errors/exceptions
- [ ] Output CSVs generated

### Verify Outputs (5 min)

```bash
ls -lh outputs_R8_<Lokasi>/
head outputs_R8_<Lokasi>/arm_A_results.csv
head outputs_R8_<Lokasi>/arm_B_results.csv
```

Expected:
- [ ] `arm_A_results.csv`: 4 rows (point_t60 F1/F2, avg_t10_t60 F1/F2)
- [ ] `arm_B_results.csv`: 2 rows (CatBoost, LGBM on point_t60)
- [ ] Metrics in reasonable range: R² 0.60-0.80 (point_t60), 0.80-0.92 (avg)
- [ ] ΔR²(F2-F1) < 0.01 (meteo redundant pattern)

---

## POST-RUN (5 min)

### Document Results

For each lokasi, record:

**Banten**:
```
Database:   ✓
Data rows:  90,488
Arm A - point_t60:   F1: 0.???  F2: 0.???   ΔR²: +0.????
Arm A - avg_t10_t60: F1: 0.???  F2: 0.???   ΔR²: +0.????
Arm B - CatBoost:    R² 0.???
Arm B - LGBM:        R² 0.???
Status:     ✓ COMPLETE
Log:        r8_banten_log.txt
```

**Bengkulu**:
```
Database:   ✓
Data rows:  60,000
Arm A - point_t60:   F1: 0.???  F2: 0.???   ΔR²: +0.????
Arm A - avg_t10_t60: F1: 0.???  F2: 0.???   ΔR²: +0.????
Arm B - CatBoost:    R² 0.???
Arm B - LGBM:        R² 0.???
Status:     ✓ COMPLETE
Log:        r8_bengkulu_log.txt
```

**Jambi**:
```
Database:   ✓
Data rows:  22,000
Arm A - point_t60:   F1: 0.???  F2: 0.???   ΔR²: +0.????
Arm A - avg_t10_t60: F1: 0.???  F2: 0.???   ΔR²: +0.????
Arm B - CatBoost:    R² 0.???
Arm B - LGBM:        R² 0.???
Status:     ✓ COMPLETE
Log:        r8_jambi_log.txt
```

### Quick Validation

Check patterns match Kalbar:

- [ ] All ΔR²(F2-F1) < 0.01? (Expected: YES)
- [ ] All CatBoost > LGBM? (Expected: YES)
- [ ] Point_t60 R² in expected range? (Expected: 0.65-0.80)
- [ ] Avg_t10_t60 R² in expected range? (Expected: 0.80-0.92)

If ALL checkmarks: **R8 implementasi selesai untuk 3 lokasi** ✓

---

## COMPILATION (5 min after all 3 complete)

### Run Aggregation

```bash
cd C:\Users\ariff\DuckDB_kalbar
python compile_r8_results.py 2>&1 | tee compile.log
```

Expected output:
```
COMPILE R8 RESULTS — 4 LOKASI

Kalbar: arm_A_results.csv ✓
Banten: arm_A_results.csv ✓
Bengkulu: arm_A_results.csv ✓
Jambi: arm_A_results.csv ✓

... (processing) ...

TABEL 2a_v2: FEATURE ENGINEERING IMPACT
[table with 4 lokasi × 2 targets]

TABEL 2c: MODEL ARCHITECTURE
[table with GBM vs DL comparison]

TABEL 2d: PRUNING SUMMARY
[table with feature reduction]

OK Compilation done. Output -> r8_compiled/
```

### Verify Compilation Outputs

```bash
ls -lh r8_compiled/
  - TABLE_2a_v2_feature_engineering.csv
  - TABLE_2c_model_architecture.csv
  - TABLE_2d_pruning_summary.csv
```

- [ ] All 3 CSV files exist
- [ ] Non-empty (check with `wc -l`)
- [ ] Can read with pandas (no corruption)

---

## CRITICAL ISSUES

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| KeyError: feature not in index | Raw columns not mapped to standard names | Map in script before split |
| Database does not exist | Wrong path | Verify with `ls -la` |
| Empty output CSVs | Data filtering too aggressive | Check filtered row counts |
| LGBM training hangs | n_estimators=6000 too high for data size | Reduce to 2000-4000 |
| ΔR² >> 0.01 (meteo not redundant) | Lokasi-specific effect OR bug | Document, investigate |

---

## SUCCESS CRITERIA (All Must Be ✓)

- [ ] **Banten**: Arm A ✓, Arm B ✓, Outputs ✓
- [ ] **Bengkulu**: Arm A ✓, Arm B ✓, Outputs ✓
- [ ] **Jambi**: Arm A ✓, Arm B ✓, Outputs ✓
- [ ] **Compilation**: All CSVs ✓
- [ ] **Pattern Validation**: All ΔR²<0.01, CatBoost>LGBM ✓
- [ ] **Logs**: All `r8_*_log.txt` saved ✓

When ALL ✓ → **R8 BATCH COMPLETE** — Ready for paper integration!

---

## Next Steps After Completion

1. Gather all outputs:
   ```bash
   mkdir r8_all_lokasi_results
   cp C:\Users\ariff\DuckDB_kalbar\outputs_R8_Kalbar/* r8_all_lokasi_results/
   cp C:\Users\ariff\Duckdb_Banten\outputs_R8_Banten/* r8_all_lokasi_results/
   cp C:\Users\ariff\bengkulu_ghi_julius\outputs_R8_Bengkulu/* r8_all_lokasi_results/
   cp C:\Users\ariff\DuckDB_jambi\outputs_R8_Jambi/* r8_all_lokasi_results/
   ```

2. Review compiled Tabel 2a_v2, 2c, 2d in `r8_compiled/`

3. Integrate findings into paper:
   - §3.3: R8 methodology
   - §4.2: GBM vs DL results
   - §4.4: Meteo redundancy + pruning
   - §5.3: Limitations

4. Submit paper for review!

---

**Estimated Total Time**: 3-4 hours (Banten + Bengkulu + Jambi) + 30 min compilation

**Support**: Refer to `R8_IMPLEMENTATION_GUIDE.md` for detailed instructions

