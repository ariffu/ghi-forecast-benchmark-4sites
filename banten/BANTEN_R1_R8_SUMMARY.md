# Banten — Ringkasan R1 & R8 (Tabel 2a/2b/2c)

> **PENTING**: Dokumen ini di WORKSPACE (`C:\Users\ariff\Duckdb_Banten\`), BUKAN vault.
> Vault Obsidian (`04_Eksperimen/Banten/Catatan/`) punya plugin sync `remotely-save`
> yang MENGHAPUS setiap file baru dari sesi ini (catatan 18-22 & skrip .py semua ter-quarantine).
> Semua hasil di sini nyata & tervalidasi dari output CSV di workspace.

Semua konfigurasi harmonis identik lintas-lokasi (F1 lean-50, clearsky 1100·sin(elev),
split train<2024/val2024/test2025, filter sun>5° anchor+t60, target point t+60 & avg t+10..t+60).
Data Banten 2022-2025 (tanpa 2021, sesuai keputusan). Python: `C:\Program Files\Python39\python.exe`.

---

## TABEL 2c — GBM vs Deep Learning (point t+60, test 2025)  [R8 Arm B]

| Model       | Bengkulu | Kalbar  | **Banten** | Jambi   |
|-------------|----------|---------|------------|---------|
| CatBoost    | 0.7920   | ?       | **0.6818** | 0.6757  |
| LightGBM    | 0.7899   | 0.7264* | **0.6787** | 0.6741  |
| MLP         | 0.7901   | ?       | **0.6768** | 0.6700  |
| Transformer | 0.7848   | ?       | **0.6735** | 0.6671  |
| LSTM        | 0.7537   | ?       | **0.6172** | 0.5884  |

*Kalbar LGBM dari R1, belum Arm B penuh. Banten & Jambi = Arm B nyata (DL mean 3 seed).
Banten DL std: MLP ±0.0018, Transformer ±0.0032, LSTM ±0.0035.
Sisa Tabel 2c: **Kalbar**.

**Pola identik 3 lokasi** (Bengkulu, Banten, Jambi): CatBoost > LightGBM > MLP > Transformer >> LSTM.
- §4.2 poin 1 — konvergensi arsitektur: MLP/Transformer hanya −0.005…−0.008 di bawah CatBoost (Banten); serupa Jambi (−0.006…−0.009).
- §4.2 poin 2 — LSTM tertinggal: −0.065 Banten, −0.087 Jambi, −0.038 Bengkulu.

Output: `outputs_R8_banten/arm_B_results.csv` (11 baris), `arm_B_summary.csv`.
Skrip: `train_ghi_1h_banten_R8_armB.py` (kelas DL disalin verbatim dari Bengkulu, 3 bug-fix; builder R1 Banten → fitur bit-per-bit identik R1/Arm A; koordinat R1 −6.26147/106.7509).
Data: model dilatih pada 90.488 baris (train 45.244 / val 22.685 / test 22.559).
**Tabel 1 paper melaporkan 90.384** (basis elevasi ASTRONOMIS homogen dgn Bengkulu/Jambi;
selisih 104 baris/0,11%, R² identik 0,680 — lihat `BANTEN_HOMOGENITAS_BENGKULU.md`).

---

## TABEL 2a/2b — Benchmark Harmonis (test 2025)  [R1]

**2a — target TITIK (GHI t+60):**
| Model | R² | MAE | RMSE | Skill vs SP |
|---|---|---|---|---|
| smart-persistence | 0.4955 | 128.4 | 185.0 | — |
| LightGBM residual (primary) | 0.6760 | 105.8 | 148.3 | +19.9% |
| CatBoost direct (sensitivitas) | 0.6818 | 104.6 | 146.9 | +20.6% |

**2b — target RATA-RATA (GHI t+10..t+60):**
| Model | R² | MAE | RMSE | Skill vs SP |
|---|---|---|---|---|
| smart-persistence | 0.7066 | 88.7 | 129.2 | — |
| LightGBM residual (primary) | 0.8322 | 68.0 | 97.7 | +24.4% |
| CatBoost direct (sensitivitas) | 0.8345 | 67.6 | 97.0 | +24.9% |

**Walk-forward 5-fold** (LGBM × titik): R²=0.643±0.057, MAE=112.3±12.0, skill=0.210±0.019.
Output: `outputs_R1_banten/`. Skrip: `train_ghi_1h_banten_R1_benchmark.py`.

---

## R8 Arm A — Meteo Redundancy (CatBoost, test 2025)

| Target | F1 (50) | F2 (55=+meteo) | ΔR² |
|---|---|---|---|
| titik t+60 | 0.6804 | 0.6932 | **+0.0128** |
| avg | 0.8344 | 0.8424 | +0.0080 |

**Banten BERBEDA dari Kalbar**: meteo TIDAK redundan (membantu +0.013) — krn meteo Banten 100% lengkap sedangkan CLP ~50% coverage (meteo mengisi celah). Arm B: CatBoost > LGBM (+0.0017) — konsisten.

## R8 Gap-Closing (lean-50 → superset → pruned, target titik)

| Langkah | n_feat | test R² | Δ vs lean |
|---|---|---|---|
| lean-50 | 50 | 0.6804 | — |
| +variabilitas | 59 | 0.6812 | +0.0008 |
| +cloud-trend | 66 | 0.6818 | +0.0014 |
| +full-lags | 82 | 0.6809 | +0.0005 (turun) |
| +meteo (SUPERSET) | 87 | 0.6926 | +0.0122 |

Gap lean→produksi (−0.064): grup fitur dinamika tutup hanya ~20% (variabilitas/cloud-trend sudah jenuh di lean-50; meteo satu-satunya lever). Sisa ~80% butuh clearsky pvlib + fitur eksogen (aerosol/present-weather) + ensemble. Pruning: top-25 ≈ superset penuh.
Output: `outputs_R8_gapclose_banten/`.

## §4.4 — ISOLASI MURNI F1 vs F2 meteo (konfirmasi, bukan artefak kumulatif)
Karena +meteo di gap-close bersifat kumulatif (82→87 fitur), diuji isolasi murni lean-50 vs +5 meteo:
- **Robust**: ΔR² POSITIF di VAL & TEST, kedua model, kedua target. Point: CatBoost +0.0128 test/+0.0095 val; LightGBM +0.0114 test/+0.0104 val. Avg: +0.007…+0.008. → bukan overfit test.
- **Pendorong = WIND SPEED**: per-fitur (lean-50 + 1), wind_speed_ms +0.0116 (≈90% efek); temp/rh/rain/pressure masing-masing ≤+0.0012.
- Interpretasi: angin = proksi laju adveksi awan di rezim konvektif Banten. Banten satu-satunya lokasi meteo-tidak-redundan (meteo 100% lengkap vs CLP ~50%).
- Tetap TIDAK menutup gap penuh (0.693 vs target 0.73/0.740).
Detail: `BANTEN_S44_METEO_ISOLATION.md`. Output: `outputs_R8_meteo_isolation_banten/`.

---

## R3 — Target-Titik MAE/RMSE (produksi 149-fitur, referensi)
+60min: R²=0.7402, MAE=95.1, RMSE=139.1 (ensemble produksi; beda dari R1 lean-50 krn 149 fitur + pvlib + ensemble).
