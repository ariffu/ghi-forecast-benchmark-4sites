# Jambi — Ringkasan R1 & R8 (setara `BANTEN_R1_R8_SUMMARY.md` / `note_18-20` Kalbar)

> **Dibuat 2026-07-25** untuk mengisi gap yang dicatat di `Restrukturisasi/07_Status_dan_Rencana_Selanjutnya.md` Prioritas C: hasil Jambi sebelumnya tersebar di banyak CSV (`outputs_R1_jambi/`, `outputs_R1_jambi_v2/`, `outputs_R1_jambi_v2_medium/`, `outputs_R8_Jambi/`, `outputs_v10_jambi/`) tanpa narasi terpadu. Dokumen ini konsolidasi semuanya, dan **secara eksplisit menandai mana yang v1 (pipeline bug, sudah usang) vs v2 (pipeline diperbaiki, dipakai sekarang)** — perbedaan ini penting karena Jambi satu-satunya lokasi yang datasetnya diperbaiki di tengah proyek (lihat `Restrukturisasi/09_Audit_Volume_Data_Jambi.md`).

**Sumber data v2**: tabel `ghi_forecast_1h_train_3h_rollback_2021_2025` di file terpisah `jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb` (skema `jambi_sch`) — dibangun 2026-07-24 dari sumber 24-jam-penuh (`asrs_jambi_menit_rev`), meniru skema 102-kolom Bengkulu 1:1. **Belum digabung ke `jambi.duckdb` utama** (perlu manual oleh pengguna, SQL di `09_Audit_Volume_Data_Jambi.md` §5).

**Split & konfigurasi**: identik dengan 3 lokasi lain — train<2024, val 2024, test 2025, F1 lean-50 fitur, LightGBM residual (primer) + CatBoost direct (sensitivitas), clearsky 1100·sin(elev), hyperparameter penuh (n_estimators=6000, learning_rate=0,02, early-stopping 150).

---

## Tabel 2a — R1 Harmonised Benchmark (v2, test 2025)

| Model | Target | n | R² | MAE | RMSE | Skill vs SP |
|---|---|---|---|---|---|---|
| Smart-persistence | point_t60 | 21.129 | 0,4266 | 142,0 | 205,2 | 0,0 |
| **LightGBM residual** | point_t60 | 21.129 | **0,6931** | 108,9 | 150,1 | 0,2683 |
| CatBoost direct | point_t60 | 21.129 | 0,6932 | 108,8 | 150,1 | 0,2685 |
| Smart-persistence | avg_t10_t60 (lampiran, bukan hasil utama) | 21.098 | 0,6495 | 99,8 | 147,0 | 0,0 |
| LightGBM residual | avg_t10_t60 | 21.098 | 0,8541 | 66,9 | 94,9 | 0,3548 |
| CatBoost direct | avg_t10_t60 | 21.098 | **0,8561** | 66,7 | 94,2 | 0,3592 |

Sumber: `outputs_R1_jambi_v2_medium/ghi_1h_R1_results.csv` (hyperparameter penuh, menggantikan versi cepat awal di `outputs_R1_jambi_v2/`).

**Catatan naik dari v1**: R² titik naik 0,676→0,693, R² rata-rata naik 0,831→0,856, test n naik 9.511→21.129 (+122%). Jambi sekarang mengungguli Banten (0,676), bukan lagi peringkat terakhir.

---

## Tabel 1 — Point Forecast per Horizon (HASIL UTAMA, v2, hyperparameter penuh)

| Horizon | n test | R² Smart-Persistence | R² LightGBM | Skill vs SP |
|---|---|---|---|---|
| 10m | 22.241 | 0,7964 | 0,8474 | 0,1342 |
| 20m | 22.184 | 0,6805 | 0,7875 | 0,1845 |
| 30m | 22.014 | 0,6110 | 0,7585 | 0,2121 |
| 40m | 21.766 | 0,5424 | 0,7347 | 0,2386 |
| 50m | 21.454 | 0,4833 | 0,7142 | 0,2563 |
| 60m | 21.129 | 0,4266 | 0,6926 | 0,2678 |

Sumber: `outputs_R1_jambi_v2_medium/per_horizon_results_full.csv`. Ini yang dipakai di `Restrukturisasi/06_Perbandingan_4_Lokasi.md` §1 (Tabel 1 4-lokasi).

## Tabel 2b — Walk-Forward 5-Fold per Horizon (v2, hyperparameter penuh)

| Horizon | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean±Std |
|---|---|---|---|---|---|---|
| 10m | 0,8270 | 0,8296 | 0,8156 | 0,8482 | 0,8480 | 0,8337±0,0142 |
| 20m | 0,7625 | 0,7647 | 0,7335 | 0,7965 | 0,7884 | 0,7691±0,0248 |
| 30m | 0,7322 | 0,7339 | 0,6955 | 0,7782 | 0,7594 | 0,7398±0,0313 |
| 40m | 0,7095 | 0,7100 | 0,6659 | 0,7649 | 0,7356 | 0,7172±0,0366 |
| 50m | 0,6883 | 0,6904 | 0,6397 | 0,7526 | 0,7160 | 0,6974±0,0414 |
| 60m | 0,6691 | 0,6677 | 0,6216 | 0,7368 | 0,6947 | 0,6780±0,0421 |

Sumber: `outputs_R1_jambi_v2_medium/wf_per_horizon.csv`. Fold periode: 2023H1, 2023H2, 2024H1, 2024H2, 2025 (test 2025 = fold 5, konsisten dengan Tabel 2a/2b di atas).

---

## Resep v10 (43-fitur accel-lean, single LightGBM residual) — Prioritas B, 2026-07-25

| Target | R² test | MAE | Skill vs persistence |
|---|---|---|---|
| Titik t+60 | 0,7232 | 103,9 | 0,3311 |
| Rata-rata t10-t60 (v11-style, supplementary) | 0,8620 | 65,0 | 0,3550 |

Sumber: `outputs_v10_jambi/ghi_1h_v10_jambi_metrics.csv`. Naik +0,0306 dari baseline F1 (0,6926→0,7232) — konsisten dengan 3 lokasi lain, lihat `06_Perbandingan_4_Lokasi.md` §10.

---

## Arm A / Arm B / Arm C — v2 (Prioritas D, selesai 2026-07-25)

**Sumber data (dipastikan)**: seluruh rerun ini memakai `jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb` (`jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025`) — bukan file/tabel v1 (`dfm_with_clp_stats.parquet` atau `jambi.duckdb` lama). Setiap skrip mem-print baris "DB reference" saat membangun cache data sebagai verifikasi. train=52.108, val=18.172, test=21.129 (identik split R1 v2).

Ini menggantikan angka v1 yang usang (`outputs_R8_Jambi/*.csv`, tanggal file 17-18 Juli, n=9.511, R²≈0,675, dari SEBELUM perbaikan pipeline 24 Juli).

**Tabel Arm A (redundansi meteo F1 vs F2)** — sumber: `outputs_R8_jambi_v2/arm_A_results_v2.csv`

| Target | Fitur | R² | MAE | best_iter |
|---|---|---|---|---|
| point_t60 | F1 (50) | 0,6947 | 108,3 | 802 |
| point_t60 | F2 (55, +meteo) | 0,6925 | 108,8 | 969 |
| avg_t10_t60 | F1 (50) | 0,8560 | 66,5 | 1075 |
| avg_t10_t60 | F2 (55, +meteo) | 0,8544 | 67,0 | 1001 |

Kesimpulan: ΔR² F2−F1 = −0,0022 (point) dan −0,0016 (avg) — meteo permukaan tetap redundan setelah kontrol fitur cuaca, konsisten dengan Bengkulu/Kalbar/Banten.

**Tabel Arm B (GBM vs DL) — LENGKAP (update 2026-07-25 sore)** — sumber: `outputs_R8_jambi_v2/arm_B_results_v2_full.csv` + `arm_B_summary_v2_full.csv`

| Model | Tipe | R² | MAE | Δ vs CatBoost |
|---|---|---|---|---|
| LightGBM | GBM | 0,6937 | 108,8 | +0,0009 |
| **CatBoost** | GBM | **0,6928** | 108,7 | — |
| Transformer (3 seed) | DL | 0,6808 ± 0,0002 | — | −0,0120 |
| LSTM (3 seed) | DL | 0,6501 ± 0,0010 | — | −0,0427 |
| MLP (3 seed) | DL | 0,6092 ± 0,0006 | — | −0,0836 |

**Update status (2026-07-25 sore):** DL yang sebelumnya ditandai "NOT EXECUTED" karena PyTorch tak terpasang di sandbox sesi sebelumnya, **berhasil dijalankan di sesi lanjutan** (environment Python 3.9 lokal `C:\Program Files\Python39`, torch 2.8.0+cpu tersedia — constraint sandbox tidak berlaku di sini). GBM dijalankan ulang sebagai verifikasi (data dibangun ulang dari SQL karena cache pickle sesi sebelumnya numpy-incompatible) — **hasil identik persis** (CatBoost 0,6928, LightGBM 0,6937), memvalidasi rebuild data. Script: `train_ghi_1h_jambi_R8_armB_v2_DL.py`.

**Temuan menarik — urutan DL Jambi v2 BERBEDA dari 3 lokasi lain:** Bengkulu/Kalbar/Banten (dan Jambi v1 lama) semua berpola CatBoost > LGBM > **MLP** > Transformer >> LSTM. Jambi v2 berpola CatBoost > LGBM > **Transformer** > LSTM > MLP — MLP justru model DL terlemah, bukan kedua terbaik. Kemungkinan penyebab: dataset v2 jauh lebih besar (train 52.108 vs 22.166 v1 / vs ~40-90k di 3 lokasi lain), MLP flat (256-256) kurang cocok menangani skala baru dibanding arsitektur dengan mekanisme atensi/rekursi. Bukan bug — kelas model identik (bug-fixed) dan GBM tervalidasi identik dgn run sebelumnya. Perlu dicatat di §4.2/§5.3 sebagai variasi antar-lokasi, bukan disamaratakan dengan pola 3 lokasi lain.

Paper §5.3 Keterbatasan **tidak lagi perlu** mencatat pengecualian Jambi — 4/4 lokasi kini punya perbandingan DL lengkap.

**Tabel Arm C (feature pruning, backward-elimination)** — sumber: `outputs_R8_jambi_v2/arm_C_v2_summary.csv`

| Lokasi | Fitur awal→akhir | Reduksi | R² test baseline | R² test pruned | Δ |
|---|---|---|---|---|---|
| Jambi | 50→20 | 60,0% | 0,6928 | 0,6920 | −0,0008 |

Metodologi identik Bengkulu/Kalbar/Banten (greedy validation-guided backward elimination, ε=0,001, PRUNE config lalu FINAL config untuk evaluasi). 20 fitur terpilih: `ghi_roll_30m_mean, ghi_roll_60m_std, ghi_roll_180m_mean, ghi_roll_180m_std, clp_cot, clp_cot_lag_10m, clp_cot_delta_10m, clp_cot_delta_30m, clp_cot_delta_180m, clp_cot_roll_180m_mean, clp_cth_m, clp_ctt_k, clp_cer, hour_sin, hour_cos, doy_sin, doy_cos, ghi_cs_t60, elev_sin_t60, smart_persist`. Reduksi 60% dengan kehilangan R² dapat diabaikan (−0,0008) — pola sama seperti Bengkulu (62% reduksi, Δ=−0,0001) dan Banten (52%, Δ=−0,0003).

---

## Audit Konsistensi Fisis (Prioritas A2, 2026-07-25)

| Metrik | Nilai |
|---|---|
| Closure violation (>15%, standar baru) | 1,92% (**terbaik dari 4 lokasi**) |
| Baris valid (standar baru) | 98,08% (**terbaik dari 4 lokasi**) |
| Rasio volatilitas kt/CLOT | 3,84× (tertinggi dari 4 lokasi) |
| Cross-corr CLOT vs kt: lag puncak | +30 menit |
| Kontradiksi CLOT vs kt | 2,67% |
| Sentinel value (9999 dsb) | ~0% di semua kolom radiasi |
| Stuck-value (siang, non-nol) | <0,01% GHI/DHI, **2,26% DNI** (dicurigai artefak derivasi model Erbs — 21,5% kolom DNI Jambi historis bukan observasi langsung) |

Sumber: `audit_data_quality_jambi.py` → `outputs_audit_konsistensi_fisis/audit_konsistensi_fisis_jambi.json`. Detail interpretasi lintas-lokasi di `06_Perbandingan_4_Lokasi.md` §7.1.

---

## Ringkasan Status Jambi (per 2026-07-25)

| Komponen | Status |
|---|---|
| R1 (titik + rata-rata + per-horizon + walk-forward) | ✅ v2, lengkap, hyperparameter penuh |
| Resep v10 (43-fitur) | ✅ v2, titik + avg (v11-style) |
| R8 Arm A (redundansi meteo) | ✅ v2, lengkap |
| R8 Arm B (GBM vs DL) | ✅ v2 LENGKAP (GBM+DL) — DL dijalankan 2026-07-25 sore di environment lokal (torch tersedia) |
| R8 Arm C (feature pruning) | ✅ v2, backward-elimination, 50→20 fitur |
| Audit konsistensi fisis | ✅ v2, lengkap |
| Penyatuan tabel v2 ke `jambi.duckdb` utama | ❌ Belum (manual oleh pengguna) |
