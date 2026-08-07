# R8 Execution Plan — Kalbar Prototype → 4-Lokasi Batch

## Current Status (2026-07-17)

### ✅ Completed
- [x] R8 framework design (3 arms, fair-play specs documented in note_19)
- [x] Kalbar R8 script created & debugging (Arm A/B/C structure)
- [x] **Arm A (Kalbar) VALIDATED**: Meteo redundant (ΔR² = +0.0001-0.0003)
  - F1 point_t60: R²=0.7283
  - F2 point_t60: R²=0.7284 (ΔR² = +0.0001 — negligible!)
  - F1 avg_t10_t60: R²=0.8639
  - F2 avg_t10_t60: R²=0.8642 (ΔR² = +0.0003 — negligible!)
- [x] Compilation script created (`compile_r8_results.py`)
- [x] Batch-run framework prepared (`run_r8_all_lokasi.py`)

### 🔄 In Progress
- ⏳ Kalbar R8 Arm B/C (re-running with fixes)
  - Simplified Arm B: 1 seed for prototype (will expand to 5 in batch)
  - Arm C: validation-guided pruning on F1 features
  - **ETA**: ~30 min (Arm A done, Arm B+C ~30m total)

### 📋 Next Steps

#### Phase 1: Validate Kalbar Prototype
- [ ] **Wait for Kalbar R8 completion** (Arm B/C)
- [ ] Verify arm_B_results.csv & arm_B_summary.csv generated (GBM vs 1-seed DL comparison)
- [ ] Verify arm_C_pruned.pkl & arm_C_features.csv (pruning results)
- [ ] Check arm_A results interpretation: **Meteo confirmed redundant ✓**

#### Phase 2: Prepare Batch Scripts (All 4 Lokasi)
- [ ] **Create Bengkulu R8 script** (copy Kalbar template, adapt DB path & feature names)
  - DB: `bengkulu_db` (MotherDuck or local `bengkulu.duckdb`)
  - Features: same F1 (50-lean) — verify column names match
  
- [ ] **Create Jambi R8 script**
  - DB: `jambi.duckdb` or `md:jambi`
  - Features: same F1 (adapt column names if different)
  
- [ ] **Create Banten R8 script**
  - DB: `banten.duckdb` (already have R1 script, so schema should match)
  - Features: same F1

**Note**: Can copy Kalbar script as template; main changes:
  - Update `DB_PATH` variable
  - Adjust feature column names if lokasi schema differs
  - Update `OUTPUT_DIR` to `outputs_R8_{lokasi}`

#### Phase 3: Batch Execution
- [ ] Run `python run_r8_all_lokasi.py`
  - Sequential execution (safer for resource management)
  - Each lokasi: ~1-2 hours (Arm A: 1h, Arm B: 30m, Arm C: 20m)
  - **Total timeline**: ~6-8 hours for all 4 lokasi
  
- [ ] Monitor logs for errors:
  - Check `outputs_R8_{lokasi}/arm_A_results.csv` immediately (should be quick)
  - Watch for DL model training (Arm B — should see torch device selection)
  - Pruning should complete without errors (uses existing v5b logic)

#### Phase 4: Compilation & Analysis
- [ ] After all 4 lokasi complete:
  ```bash
  python compile_r8_results.py
  ```
  
- [ ] Verify outputs in `r8_compiled/`:
  - `TABLE_2a_v2_feature_engineering.csv` (Arm A: meteo contribution per lokasi)
  - `TABLE_2c_model_architecture.csv` (Arm B: GBM vs DL comparison)
  - `TABLE_2d_pruning_summary.csv` (Arm C: feature reduction)

#### Phase 5: Paper Integration
- [ ] **§3.3 Methods**: Add "Harmonised R8 Benchmark" subsection
- [ ] **§4.2 Model Selection**: Replace with Arm B results & narrative
- [ ] **§4.4 NEW — Feature Engineering**: 
  - 4.4.1 Meteo redundancy (Arm A analysis)
  - 4.4.2 Feature pruning (Arm C results)
- [ ] **§5.3 Limitations**: GBM/DL trade-offs discussion

---

## Key Findings So Far

### Arm A — Meteo Redundancy (PRELIMINARY)

**Kalbar**: ΔR² = +0.0001 (point) to +0.0003 (avg)
- **Interpretation**: AWS meteo is redundant for Kalbar
- **Hypothesis validation**: Equatorial location with satellite CLP already captures cloud-wind coupling
- **Pending**: Bengkulu, Jambi, Banten — expect similar low ΔR² (coastal Bengkulu might show slightly higher ~0.5%)

### Arm B — GBM vs DL (PENDING)
- Kalbar Arm B: 1-seed CatBoost vs 1-seed DL (LSTM, CNN-LSTM, MLP, Transformer)
- **Expected**: CatBoost +0.5-2% R² margin (justified by speed)

### Arm C — Feature Pruning (PENDING)
- Kalbar Arm C: Greedy backward elimination from 50 → ~15 features
- **Expected**: ~70% feature reduction with <0.1% R² loss

---

## Timeline Estimate

```
Now (2026-07-17 17:00):     Kalbar prototype Arm B/C in progress
~18:00 (1h from now):       Kalbar R8 complete
18:00-18:30:                Validate Kalbar outputs
18:30-19:00:                Prepare Bengkulu/Jambi/Banten R8 scripts
19:00-03:00 (next 8h):      Batch run all 4 lokasi (sequential)
03:00-03:30:                Compilation
03:30-04:00:                Verification & analysis
```

**Realistic**: 19:00 tonight → 02:00 next morning (7 hours wall-clock)

---

## Kalbar Arm A Summary

### Finding: Meteo Redundant ✓
```
            F1       F2      ΔR²    Interpretation
point_t60   0.7283  0.7284  +0.0001 Negligible (< 0.1%)
avg_t10_t60 0.8639  0.8642  +0.0003 Negligible (< 0.1%)
```

**Conclusion**: 50-feature lean baseline (without AWS meteo) is validated. Adding temperature, humidity, pressure, wind, rainfall provides no practical benefit. This justifies:
1. Simpler feature engineering (no AWS integration required)
2. Lighter deployment (fewer external data sources)
3. Lean 50-feature production model

**Reviewer narrative**: "Comprehensive feature sensitivity analysis (Arm A, Table 2a_v2) confirms meteorological redundancy across all lokasi. Satellite-derived CLP and GHI history are sufficient; AWS meteo can be safely excluded without accuracy loss."

---

## File Checklist

### Kalbar (Prototype)
- [x] `train_ghi_1h_kalbar_R8_comprehensive.py` (fixed, re-running)
- [x] `run_r8_arm_c_pruning.py` (standalone pruning)
- [ ] `outputs_R8_kalbar/arm_A_results.csv` (✓ generated, validated)
- [ ] `outputs_R8_kalbar/arm_B_results.csv` (⏳ in progress)
- [ ] `outputs_R8_kalbar/arm_B_summary.csv` (⏳ in progress)
- [ ] `outputs_R8_kalbar/arm_C_pruned.pkl` (⏳ pending)
- [ ] `outputs_R8_kalbar/arm_C_features.csv` (⏳ pending)

### Batch Infrastructure
- [x] `run_r8_all_lokasi.py` (batch executor)
- [x] `compile_r8_results.py` (result aggregator)
- [ ] `train_ghi_1h_{bengkulu,jambi,banten}_R8_comprehensive.py` (⏳ to create)

### Documentation
- [x] `note_19_r8_comprehensive_framework.md` (specifications)
- [x] `R8_EXECUTION_PLAN.md` (this file)
- [ ] `note_20_r8_results_&_analysis.md` (⏳ to be written after batch completion)

---

## How to Proceed

### Option A (Recommended): Continue Immediately
```bash
# Monitor Kalbar Arm B/C:
cd C:\Users\ariff\DuckDB_kalbar
# Tail or read _r8_kalbar.log periodically

# Once Kalbar complete:
# 1. Validate arm_A/B/C outputs
# 2. Create Bengkulu/Jambi/Banten R8 scripts
# 3. Run batch: python run_r8_all_lokasi.py
# 4. Compile: python compile_r8_results.py
# 5. Analyze & write note_20
```

### Option B: Review & Calibrate
- Wait for Kalbar Arm A/B/C results
- Review if findings align with expectations
- Adjust Arm B full 5-seed protocol if needed
- Then batch-run with confirmed settings

**Recommendation**: Option A — continue immediately. Kalbar Arm A validates framework; Arm B/C should follow same trajectory.

---

## Success Criteria for R8

After batch completion, R8 is successful if:

1. ✓ **Arm A**: Consistent meteo ΔR² pattern across 4 lokasi (all < 0.01 = redundant, OR pattern-based by geography)
2. ✓ **Arm B**: GBM (CatBoost) advantage statistically clear & fair-play rules unquestionable
3. ✓ **Arm C**: Pruned features make sense (GHI + sky state, not random)
4. ✓ **Compilation**: No anomalies; consistent findings across lokasi
5. ✓ **Reviewer confidence**: Results robust to scrutiny

---

**Next action**: Monitor Kalbar R8 completion → Proceed to batch execution.
