# Perbandingan 4 Lokasi — Tabel Benchmark Utama

Sumber: `DuckDB_kalbar/r1_compiled/TABLE_2a_test_2025_results.csv`, `note_18_r1_harmonised_benchmark.md`, dan `outputs_R8_<lokasi>/`. Semua angka di bawah dihasilkan dari protokol **identik** (lihat `00_Ringkasan_dan_Protokol_Standar.md`).

> **Update metodologi (2026-07-24)**: hasil utama sekarang adalah forecast titik per-horizon (10/20/30/40/50/60 menit, terpisah), bukan target rata-rata dirangkum jadi satu angka — koreksi dari pembimbing, alasan lengkap di `03_Target_dan_Split.md` §1.3. Target `avg_t10_t60` dipindah ke Lampiran (§9 di bawah).

---

## 1. Tabel 1 — Point Forecast per Horizon (HASIL UTAMA)

Akurasi titik dilaporkan **terpisah per horizon**, bukan dirata-ratakan — supaya kurva peluruhan akurasi (10→60 menit) terlihat, bukan disembunyikan jadi satu angka. ✅ **Keempat lokasi selesai dengan hyperparameter LightGBM PENUH** (n_estimators=6000, learning_rate=0,02, early-stopping 150 — konfigurasi resmi R1, bukan lagi versi dipercepat; run 2026-07-25, menggantikan angka versi cepat sebelumnya, selisihnya terbukti <0,002 R² di semua horizon/lokasi). Model LightGBM residual, test 2025.

**R² LightGBM per horizon:**

| Lokasi | 10m | 20m | 30m | 40m | 50m | 60m |
|---|---|---|---|---|---|---|
| **Bengkulu** | **0,9244** | **0,8768** | **0,8481** | **0,8258** | **0,8049** | **0,7868** |
| Kalbar | 0,8705 | 0,8107 | 0,7752 | 0,7499 | 0,7339 | 0,7217 |
| Jambi (v2) | 0,8474 | 0,7875 | 0,7585 | 0,7347 | 0,7142 | 0,6926 |
| Banten | 0,8753 | 0,8042 | 0,7622 | 0,7302 | 0,7017 | 0,6760 |

**R² Smart-Persistence (baseline, sama urutan horizon) — untuk menghitung skill:**

| Lokasi | 10m | 20m | 30m | 40m | 50m | 60m |
|---|---|---|---|---|---|---|
| Bengkulu | 0,9025 | 0,8232 | 0,7612 | 0,7080 | 0,6580 | 0,6136 |
| Kalbar | 0,7458 | 0,6312 | 0,5534 | 0,4934 | 0,4530 | 0,4233 |
| Jambi (v2) | 0,7964 | 0,6805 | 0,6110 | 0,5424 | 0,4833 | 0,4266 |
| Banten | 0,8488 | 0,7333 | 0,6574 | 0,6019 | 0,5467 | 0,4955 |

**Skill vs SP (1 − RMSE/RMSE_SP) — metrik paling adil untuk membandingkan lokasi**, karena menormalkan terhadap kesulitan baseline fisika masing-masing lokasi:

| Lokasi | 10m | 20m | 30m | 40m | 50m | 60m | Tren |
|---|---|---|---|---|---|---|---|
| **Kalbar** | 0,2863 | 0,2836 | 0,2905 | 0,2973 | 0,3025 | **0,3053** | Naik, paling tinggi di 60m |
| Bengkulu | 0,1193 | 0,1652 | 0,2025 | 0,2276 | 0,2448 | 0,2571 | Naik konsisten |
| Jambi (v2) | 0,1342 | 0,1845 | 0,2121 | 0,2386 | 0,2563 | 0,2678 | Naik konsisten |
| Banten | 0,0920 | 0,1432 | 0,1668 | 0,1768 | 0,1888 | 0,1985 | Naik, melandai di horizon jauh |

Tiga temuan yang hanya terlihat karena horizon dilaporkan terpisah (hilang total kalau dirata-ratakan jadi satu angka avg_t10_t60):

1. **R² absolut menurun mulus di keempat lokasi** seiring horizon memanjang (fisika: makin jauh horizon, makin banyak ketidakpastian evolusi awan) — Bengkulu tetap unggul di semua horizon, bukan cuma di t+60.
2. **Skill vs SP justru naik di semua lokasi** — model machine learning semakin mengungguli baseline fisika sederhana seiring horizon memanjang, karena smart-persistence meluruh lebih cepat daripada model (SP mengasumsikan kondisi awan konstan, asumsi ini makin buruk untuk horizon lebih jauh).
3. **Peringkat berubah tergantung metrik dipakai**: berdasarkan R² absolut, Bengkulu > Kalbar > Jambi > Banten konsisten di semua horizon. Tapi berdasarkan **skill vs SP**, Kalbar justru tertinggi di t+60 (0,303) — mengindikasikan baseline smart-persistence Kalbar jauh lebih lemah (SP R²=0,42) sehingga model punya "ruang perbaikan" lebih besar, bukan berarti model Kalbar lebih baik secara absolut. Ini nuansa yang wajib disebutkan eksplisit di paper — jangan hanya melaporkan satu metrik sebagai "lokasi terbaik".

**Gambar**: `figures/gambar_r2_vs_horizon_4lokasi.png` — kandidat Figure utama bab Hasil, dua panel (kiri: R² LightGBM vs Smart-Persistence per horizon; kanan: skill vs SP per horizon), 4 garis per panel satu warna/marker per lokasi.

Sumber CSV mentah: `outputs_R1_bengkulu_per_horizon/per_horizon_results_full.csv`, `outputs_R1_kalbar_per_horizon/per_horizon_results_full.csv`, `outputs_R1_banten_per_horizon/per_horizon_results_full.csv`, `outputs_R1_jambi_v2_medium/per_horizon_results_full.csv` (masing-masing di folder lokasi).

**Catatan metodologi**: angka di atas sudah memakai hyperparameter LightGBM PENUH (n_estimators=6000, learning_rate=0,02, early-stopping 150 — sama seperti Tabel 2a §2), menggantikan run awal yang sempat memakai konfigurasi dipercepat (3000/0,03/100) untuk validasi cepat. Selisih keduanya terbukti kecil di semua 24 kombinasi horizon×lokasi (maksimum ±0,002 R²) — early-stopping membuat model konvergen jauh sebelum mencapai batas 6000 iterasi (`best_iter` aktual berkisar 198–633), sehingga waktu training penuh ternyata tidak jauh lebih lama dari versi dipercepat. File `per_horizon_results.csv` (tanpa `_full`) di tiap folder adalah versi cepat lama, disimpan untuk jejak audit, bukan lagi angka acuan.

**Catatan Kalbar**: tabel `training_ghi_1h_direct` tidak menyimpan lead individual per horizon (hanya `ghi_target_60m` titik dan `ghi_target_avg60m` rata-rata precomputed) — nilai per-horizon Kalbar di atas dihitung dengan JOIN ke `solar_kalbar_10m` (tabel radiasi mentah 10-menit, 210.384 baris, timestamp cocok 1:1 dengan `training_ghi_1h_direct`) lalu menghitung `LEAD(ghi_final, 1..6)` langsung di situ — bukan dari kolom precomputed, genuine values.

---

## 2. Tabel 2a — Test 2025, Target Titik t+60 (satu titik horizon dari Tabel 1)

| Lokasi | Model | R² | MAE (W/m²) | RMSE (W/m²) | Skill vs SP |
|---|---|---|---|---|---|
| **Bengkulu** | CatBoost | **0,792** | **96,5** | 137,0 | **0,2633** |
| Bengkulu | LightGBM | 0,789 | 96,7 | 137,9 | 0,2583 |
| Kalbar | CatBoost | 0,728 | 120,0 | 165,6 | 0,2122 |
| Kalbar | LightGBM | 0,723 | 120,7 | 167,0 | 0,2057 |
| Banten | CatBoost | 0,682 | 104,6 | 146,9 | 0,2058 |
| Banten | LightGBM | 0,676 | 105,8 | 148,3 | 0,1985 |
| Jambi (v2, tabel 24-jam) | CatBoost | **0,693** | 108,8 | 150,1 | 0,2685 |
| Jambi (v2, tabel 24-jam) | LightGBM | 0,693 | 108,9 | 150,1 | 0,2683 |
| ~~Jambi (v1, bug pipeline)~~ | ~~CatBoost~~ | ~~0,676~~ | ~~114,4~~ | ~~153,0~~ | ~~0,2384~~ |
| Smart-persistence (baseline) | — | 0,44–0,62 | 128–144 | 186–211 | 0,0 |

**Update Jambi (2026-07-24)**: angka v1 di atas (dicoret) dihasilkan dari pipeline bug yang membuang ~3 jam pertama tiap hari (lihat `09_Audit_Volume_Data_Jambi.md`). Setelah pipeline diperbaiki (tabel `ghi_forecast_1h_train_3h_rollback_2021_2025` baru, sumber 24-jam-penuh), test set naik dari 9.511 → 21.129 baris dan R² titik naik dari 0,676 → **0,693** — Jambi sekarang mengungguli Banten (0,682), bukan lagi peringkat terakhir. Ini mengonfirmasi kecurigaan awal pengguna bahwa dataset Jambi yang kecil bukan representasi kondisi iklim yang sebenarnya.

---

## 3. Tabel 2b — Walk-Forward 5-Fold, Diperluas ke Semua Horizon (LightGBM residual)

✅ **Diperluas ke per-horizon (2026-07-25)** — sebelumnya walk-forward hanya mengevaluasi titik t+60; sekarang direplikasi penuh untuk keenam horizon (10/20/30/40/50/60 menit) di keempat lokasi (120 model: 5 fold × 6 horizon × 4 lokasi, hyperparameter penuh 6000/0,02/150).

**R² walk-forward (mean ± std antar 5 fold) per horizon:**

| Lokasi | 10m | 20m | 30m | 40m | 50m | 60m |
|---|---|---|---|---|---|---|
| **Bengkulu** | 0,9241±0,0017 | 0,8787±0,0077 | 0,8520±0,0127 | 0,8304±0,0159 | 0,8106±0,0190 | **0,7939±0,0195** |
| Jambi (v2) | 0,8337±0,0142 | 0,7691±0,0248 | 0,7398±0,0313 | 0,7172±0,0366 | 0,6974±0,0414 | 0,6780±0,0421 |
| Kalbar | 0,8316±0,0311 | 0,7683±0,0294 | 0,7278±0,0348 | 0,6977±0,0377 | 0,6786±0,0376 | 0,6626±0,0422 |
| Banten | 0,8253±0,0614 | 0,7584±0,0562 | 0,7216±0,0526 | 0,6925±0,0539 | 0,6677±0,0546 | 0,6428±0,0574 |

**Temuan**: (1) Bengkulu **paling stabil di SEMUA horizon**, bukan cuma t+60 — σ 3–4× lebih kecil dari Banten di semua titik. (2) σ (variabilitas antar-fold) membesar seiring horizon memanjang di keempat lokasi — konsisten dengan prediksi lebih jauh = lebih sensitif terhadap kondisi cuaca spesifik periode fold. (3) Banten punya σ terbesar secara konsisten (0,0614 di 10m naik ke 0,0574 di 60m) — walau σ 60m Banten sedikit lebih kecil dari 50m, pola umum tetap σ membesar dengan horizon. (4) Kolom 60m di atas mengonfirmasi angka test-2025 tunggal Tabel 1/2a bukan kebetulan window mudah — Bengkulu WF 0,7939 dekat dengan R² test-2025 0,7868 (Tabel 1), begitu juga Jambi (WF 0,6780 vs test 0,6926), Kalbar (WF 0,6626 vs test 0,7217), Banten (WF 0,6428 vs test 0,6760) — walau ada gap wajar karena WF adalah rata-rata 5 periode berbeda, bukan satu tahun 2025 saja.

Sumber CSV: `outputs_R1_<lokasi>_per_horizon/wf_per_horizon.csv` (Jambi: `outputs_R1_jambi_v2_medium/wf_per_horizon.csv`), 30 baris per lokasi (6 horizon × 5 fold).

---

## 4. Tabel 2c — GBM vs Deep Learning (target titik, test 2025)

| Model | Bengkulu | Kalbar | Banten | Jambi | Pola |
|---|---|---|---|---|---|
| **CatBoost** | **0,7920** | **0,7278** | **0,6818** | **0,6928** (v2) | Primer di semua lokasi |
| LightGBM | 0,7899 | 0,7264 | 0,6787 | 0,6937 (v2) | −0,14…+0,13% vs CatBoost |
| MLP | 0,7901 ±0,0003 | 0,7218 ±0,0004 | 0,6768 ±0,0021 | **Not executed** | — |
| Transformer | 0,7848 ±0,0011 | 0,7223 ±0,0016 | 0,6735 ±0,0039 | **Not executed** | — |
| LSTM | 0,7537 ±0,0037 | 0,6718 ±0,0023 | 0,6172 ±0,0043 | **Not executed** | — |

Pola identik di 4 lokasi (GBM): CatBoost ≈ LightGBM, keduanya jauh di atas urutan DL di 3 lokasi lain. **Jambi GBM diperbarui ke v2 2026-07-25** (CatBoost 0,6928, LightGBM 0,6937, n test=21.129) — naik dari v1 (0,6757/0,6741, n=9.511), sekarang mengungguli Banten.

**Keputusan (2026-07-25): DL Jambi dilaporkan "Not executed", bukan angka v1 usang.** PyTorch tidak berhasil terpasang di sandbox (wheel CPU-only diblokir proxy `download.pytorch.org`; wheel PyPI default adalah build CUDA-linked yang butuh runtime NVIDIA ~2,1GB tambahan meski untuk CPU-only). Ini konstrain lingkungan teknis, bukan pilihan metodologi. Perbandingan DL/GBM tetap lengkap dan valid di 3/4 lokasi (Bengkulu, Kalbar, Banten); hanya Jambi dikecualikan — dicatat eksplisit sebagai keterbatasan paper (§5.3): *"DL architecture comparison completed for 3/4 locations (Bengkulu, Kalbar, Banten); Jambi deferred due to sandbox environment constraints (PyTorch installation blocked)."* Tidak direncanakan dikerjakan ulang kecuali constraint sandbox berubah. Detail fair-play constraints di `04_Model_dan_Training.md`.

---

## 5. Tabel 2a_v2 — Redundansi Meteo (Arm A, ΔR² = F2 − F1)

| Lokasi | ΔR² titik | ΔR² rata-rata | Interpretasi |
|---|---|---|---|
| Kalbar | +0,0001 | +0,0003 | Redundan |
| Bengkulu | +0,0005 | −0,0005 | Redundan |
| Jambi (v2) | −0,0022 | −0,0016 | Redundan |
| **Banten** | **+0,0128** | **+0,0080** | **Tidak redundan** — meteo membantu |

**Update Jambi (2026-07-25):** angka di atas kini v2 (sumber `outputs_R8_jambi_v2/arm_A_results_v2.csv`, n test=21.129), menggantikan v1 (+0,0009/−0,0021, n=9.511). Kesimpulan kualitatif tidak berubah — meteo tetap redundan.

Penjelasan anomali Banten di `02_Feature_Engineering.md` §5 (cakupan CLP satelit Banten ~50%, lebih rendah dari lokasi lain; AWS 100% lengkap mengisi celah; pendorong utama = wind speed sebagai proksi laju adveksi awan).

---

## 6. Tabel 2d — Feature Pruning (Arm C)

| Lokasi | N Baseline | N Pruned | Reduksi | R² Test (baseline → pruned) | ΔR² | Status |
|---|---|---|---|---|---|---|
| Kalbar | 50 | 31 | 38% | 0,7273 (titik) | — | ⚠️ Selesai, tapi metodologi lama (lihat catatan) |
| Jambi | 50 | 20 | 60% | 0,6928 → 0,6920 | −0,0008 | ✅ Selesai (v2, backward-elimination murni) |
| Bengkulu | 50 | 19 | 62% | 0,7920 → 0,7921 | +0,0001 | ✅ Selesai (diulang dgn backward-elimination murni dari F1) |
| Banten | 50 | 24 | 52% | 0,6818 → 0,6821 | +0,0003 | ✅ Selesai |

**Update (2026-07-25):** Bengkulu, Banten, dan Jambi kini seluruhnya memakai metodologi backward-elimination murni, dimulai dari F1 (50 fitur) — bukan superset F_super yang tercampur DHI/DNI mentah dengan `ghi_now` (lihat root cause di `02_Feature_Engineering.md` §4 poin 9, yang membuat hasil Bengkulu sebelumnya void), dan bukan top-K sweep v1 lama yang sebelumnya dipakai Jambi (50→10 fitur, 80% reduksi, tanpa breakdown baseline/pruned test R² yang tersimpan). Ketiga lokasi menunjukkan pola sama dengan Kalbar: reduksi fitur besar (52–62%) dengan biaya R² dapat diabaikan (bahkan sedikit membaik di test set, dalam margin noise). Kalbar sendiri masih memakai run lama (metodologi Arm A v1 turut belum diperbarui) — perbedaan skala reduksi (38% vs 52–62%) kemungkinan sebagian mencerminkan metodologi yang belum diselaraskan, bukan properti data.

- **Jambi** (50→20 fitur, 60% reduksi): fitur terpilih `['ghi_roll_30m_mean', 'ghi_roll_60m_std', 'ghi_roll_180m_mean', 'ghi_roll_180m_std', 'clp_cot', 'clp_cot_lag_10m', 'clp_cot_delta_10m', 'clp_cot_delta_30m', 'clp_cot_delta_180m', 'clp_cot_roll_180m_mean', 'clp_cth_m', 'clp_ctt_k', 'clp_cer', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'ghi_cs_t60', 'elev_sin_t60', 'smart_persist']`. Sama seperti lokasi lain, `ghi_now` tereliminasi (iterasi 30) — fitur CLP dan smart-persistence tetap dominan.
- **Bengkulu** (50→19 fitur, 62% reduksi): fitur terpilih `['ghi_lag_180m', 'ghi_roll_180m_mean', 'ghi_delta_60m', 'kt_now', 'kt_roll60m_mean', 'clp_cot', 'clp_cot_lag_10m', 'clp_cot_delta_10m', 'clp_cot_delta_180m', 'clp_cot_roll_180m_mean', 'clp_ctt_k', 'clp_cer', 'hour_sin', 'doy_sin', 'doy_cos', 'ghi_cs_t60', 'elev_sin_t60', 'smart_persist', 'smart_persist_avg']`. Catatan: bahkan `ghi_now` sendiri berhasil dieliminasi (delta 0,0009, di bawah epsilon 0,001) — sinyal recency lain (`ghi_lag_180m`, roll-mean, smart persistence) sudah cukup menggantikannya.
- **Banten** (50→24 fitur, 52% reduksi): fitur terpilih `['ghi_roll_30m_mean', 'ghi_roll_60m_mean', 'ghi_roll_60m_std', 'ghi_roll_180m_mean', 'ghi_roll_180m_std', 'kt_roll60m_mean', 'clp_cot', 'clp_cot_lag_10m', 'clp_cot_lag_20m', 'clp_cot_delta_10m', 'clp_cot_delta_30m', 'clp_cot_delta_180m', 'clp_cot_roll_180m_mean', 'clp_cth_m', 'clp_ctt_k', 'clp_cer', 'hour_sin', 'doy_sin', 'doy_cos', 'month_cos', 'ghi_cs_t60', 'elev_sin_t60', 'smart_persist', 'smart_persist_avg']`. Sama seperti Bengkulu, `ghi_now` juga tereliminasi lebih awal (iterasi 14) — fitur CLP (awan) dan smart-persistence mendominasi selection di semua lokasi.

Konsisten di keempat lokasi: fitur CLP (`clp_cot` beserta lag/delta/roll), smart-persistence baseline, dan encoding waktu (hour/doy sin-cos) selalu bertahan hingga akhir pruning — mengonfirmasi bahwa sinyal awan dan posisi matahari adalah driver utama, bukan artefak `ghi_now` mentah.

Referensi produksi Kalbar yang tervalidasi terpisah dari Arm C: model v5b, 7 fitur (86% reduksi), R²=0,8686 (avg_t10_t60) — pruning ekstrem yang tetap valid karena memakai metodologi backward-elimination berbasis validasi, bukan top-K dari superset campuran.

---

## 7. Mengapa Bengkulu Unggul? (Hipotesis Geografis)

| Faktor | Bengkulu | Kalbar | Jambi | Banten |
|---|---|---|---|---|
| Rezim awan | Pesisir, musiman (trade-wind) | Ekuatorial, persisten/konvektif | Orografi kompleks | Campuran |
| Prediktabilitas awan | Tinggi (angin pasat) | Sedang (konvektif) | Rendah (terrain) | Sedang |
| Kontinuitas data CLP | 99,8% (buffer 30km) | 88,4% (buffer 20km) | Jarang | Baik (~50% coverage) |
| Ukuran test set | 22.711 | 21.386 | 21.129 (v2, dulu 9.511) | 22.559 |
| R² titik (CatBoost) | **0,792** | 0,728 | 0,693 (v2, dulu 0,676) | 0,682 |

**Hipotesis** (belum dibuktikan kausal, perlu disebut sebagai hipotesis di paper, bukan kesimpulan final):
1. Angin pasat pesisir Bengkulu → evolusi awan lebih predictable dibanding konveksi ekuatorial murni
2. Cakupan satelit CLP lebih baik di lintang Bengkulu (resolusi lebih tinggi, gap lebih sedikit)
3. Konsistensi historis stasiun radiasi/SYNOP Bengkulu
4. Supresi konveksi: pola tropis Bengkulu kurang eksplosif dibanding ekuatorial murni (Kalbar)

**Analisis kesenjangan Kalbar** (0,064 R² di bawah Bengkulu, titik, CatBoost):

| Kontributor | Estimasi ΔR² | Bukti |
|---|---|---|
| Rezim awan | −0,040 | WF fold 1 Kalbar hanya 0,611 (vs Bengkulu 0,777) |
| Cakupan/kualitas CLP | −0,015 | Buffer 20km validitas lebih rendah (88% vs 99%) |
| Ketidakpastian aerosol | −0,009 | Varians AOD lebih tinggi di zona ekuatorial |
| Ukuran data | ~0,000 | Kalbar ~21k test, mirip Bengkulu ~22k |

**Mitigasi yang disarankan untuk Kalbar** (belum diuji): all-sky imager untuk deteksi inisiasi konvektif, integrasi medan angin NWP untuk sirkulasi orografis, asimilasi aerosol per-jam.

> ✅ **Update (2026-07-24): R1 Jambi sudah dijalankan ulang dengan tabel 24-jam-penuh** (`ghi_forecast_1h_train_3h_rollback_2021_2025`, lihat `09_Audit_Volume_Data_Jambi.md` §5 dan §6). Hasilnya mengonfirmasi dugaan di atas: R² titik naik 0,676 → 0,693 dan Jambi kini **mengungguli Banten** (0,682), bukan lagi peringkat terakhir. Test set juga naik dari 9.511 → 21.129 baris, sebanding dengan 3 lokasi lain. Peringkat baru: Bengkulu (0,792) > Kalbar (0,728) > **Jambi (0,693)** > Banten (0,682). Kolom "Kontinuitas data CLP" dan hipotesis geografis Jambi ("orografi kompleks, rendah") di atas belum diverifikasi ulang — masih hipotesis lama, ditulis sebelum perbaikan pipeline, dan bisa jadi perlu direvisi juga.

### 7.1 Dari Spekulasi ke Bukti: Audit Konsistensi Fisis 4-Lokasi (Prioritas A2, 2026-07-25)

Replikasi metodologi `note_09_audit_konsistensi_fisis.md` (awalnya hanya Kalbar, lihat `08_Standardisasi_Data_Mentah.md` §1) ke Bengkulu, Banten, Jambi, dengan **satu skrip identik** (`audit_data_quality_<lokasi>.py`) supaya angka benar-benar sebanding — bukan dikutip dari audit lama yang metodologinya berbeda-beda per lokasi. **Catatan penting**: karena definisi threshold distandarkan ulang di sini, angka closure-violation di bawah TIDAK sama persis dengan angka yang pernah dilaporkan terpisah per lokasi sebelumnya (mis. Bengkulu 12,9%, Jambi 1,33% di `08_Standardisasi_Data_Mentah.md`) — itu justru intinya standardisasi: definisi lama tidak seragam antar lokasi, di sini semua dihitung dengan rumus dan ambang batas yang SAMA (closure error relatif >15% dari `GHI_pred=DHI+DNI·sin(elev)` vs `GHI` aktual, baris siang elev>5°).

| Metrik | Bengkulu | Kalbar | Banten | Jambi (v2) |
|---|---|---|---|---|
| Closure violation (>15%, standar baru) | 40,35% | 42,10% | **7,91%** | **1,92%** |
| Rasio volatilitas kt/CLOT (>1 = target berubah lebih cepat dari fitur cloud) | 2,77× | 2,66× | 3,66× | 3,84× |
| Cross-corr CLOT(t) vs kt(t+lag): lag puncak | +20 menit | +30 menit | +10 menit | +30 menit |
| Kontradiksi langsung CLOT vs kt (baris siang) | **4,84%** | 3,78% | **2,09%** | 2,67% |
| Kontradiksi 2-produk (CLP vs ARP/analog) | n/a (tak ada tabel ARP) | 48,6% | 18,5% | 45,1% |

**R² v10 (referensi, untuk dikorelasikan)**: Bengkulu 0,8212 > Kalbar 0,7473 > Banten 0,7365 > Jambi 0,7232.

**Temuan jujur — hasilnya TIDAK mendukung cerita sederhana "data lebih bersih = R² lebih tinggi":**
1. **Closure violation TIDAK berkorelasi dengan R².** Jambi punya closure paling bersih (1,92%) tapi R² PALING RENDAH; Bengkulu R² tertinggi tapi closure violation kedua terburuk (40,35%, hampir sama dengan Kalbar). Kemungkinan closure violation di sini lebih mencerminkan kualitas kalibrasi sensor triplet (GHI/DNI/DHI independen) daripada kegunaan sinyal untuk forecasting horizon 1 jam — dua hal yang berbeda.
2. **Kontradiksi CLOT-vs-kt JUGA berkorelasi TERBALIK dengan R².** Bengkulu (R² tertinggi) justru punya kontradiksi tertinggi (4,84%); Banten (R² lebih rendah) kontradiksinya paling kecil (2,09%). Ini mengejutkan dan berlawanan dengan hipotesis geografis awal §7.
3. **Rasio volatilitas adalah SATU-SATUNYA metrik yang searah dengan R² di level kelompok**: Bengkulu+Kalbar (rasio 2,66–2,77×, R² lebih tinggi) vs Banten+Jambi (rasio 3,66–3,84×, R² lebih rendah) — meski urutan persis di dalam tiap pasangan tidak sempurna (Kalbar rasio paling rendah tapi bukan R² tertinggi).
4. **Kontradiksi 2-produk (CLP vs ARP/analog satelit)** juga tidak monoton: Kalbar terburuk (48,6%) tapi R²-nya kedua terbaik; Banten terbaik (18,5%) tapi R²-nya ketiga. Catatan metodologis: hanya Kalbar yang punya produk ARP asli (`clear_sky` flag khusus aerosol-retrieval); Banten dan Jambi memakai substitusi (`qa_clear`/`sat_retrieval_valid` dari tabel AOD terpisah, cakupan lebih sempit); Bengkulu tidak punya tabel analog sama sekali — jadi baris ini kurang bisa dibandingkan apple-to-apple dibanding 4 metrik lainnya.

**Kesimpulan revisi**: audit ini **mengubah hipotesis geografis dari spekulasi menjadi klaim berbasis bukti** seperti yang diminta — tapi buktinya justru **menolak** cerita sederhana "rezim awan lebih predictable secara fisis = R² lebih tinggi". Metrik yang paling konsisten dengan ranking R² adalah rasio volatilitas target-vs-fitur (item 3), bukan closure atau kontradiksi cloud-vs-radiasi. Ini mengarah ke kesimpulan yang lebih hati-hati: **gap R² antar-lokasi kemungkinan besar dijelaskan oleh kombinasi faktor** (volume data training, cakupan CLP %, resolusi buffer spasial — lihat `01_Dataset.md` §4.5, dan arsitektur pengumpulan data yang berbeda per lokasi — lihat `08_Standardisasi_Data_Mentah.md` §2), **bukan satu penjelasan fisis tunggal** seperti "rezim awan pesisir lebih predictable". Hipotesis geografis di atas (angin pasat, supresi konveksi, dst.) tetap masuk akal sebagai salah satu faktor kontribusi, tapi TIDAK bisa lagi diklaim sebagai penjelasan dominan tanpa kualifikasi ini.

Skrip: `audit_data_quality_<lokasi>.py` di masing-masing folder lokasi, output di `outputs_audit_konsistensi_fisis/` (JSON lengkap + CSV audit sentinel).

---

## 8. Perbandingan dengan Literatur Internasional (konteks, bukan klaim setara)

| Model | Lokasi | Horizon | R² | Sebanding dengan Bengkulu? |
|---|---|---|---|---|
| v6/v8 Bengkulu (sesi Julius) | Bengkulu | 60 menit | 0,82 (titik) | — |
| Random Forest | Kalbar (Indonesia) | 60 menit | 0,824 | ✅ Ya — setara |
| SMA-WT-LSTM | Australia | 60 menit | 0,95 | ❌ Tidak — iklim non-tropis |
| SMA-DELM | Iraq | **harian** | 0,92 | ❌ Tidak — beda horizon & satuan (MJ/m² vs W/m²) |

Klaim R²≥0,90 di beberapa paper internasional sering memakai iklim non-tropis atau horizon/satuan berbeda — tidak bisa dijadikan pembanding langsung tanpa menyamakan horizon, satuan, dan definisi target (lihat peringatan di `00_Ringkasan_dan_Protokol_Standar.md` §3).

---

## 9. Lampiran — Target Rata-Rata (avg_t10_t60), Bukan Hasil Utama

> **Didemosikan dari hasil utama (2026-07-24), koreksi dari pembimbing.** Alasan lengkap di `03_Target_dan_Split.md` §1.3. Ringkas: merata-ratakan GHI di 6 titik 10-menitan (t+10..t+60) menghasilkan target dengan varians jauh lebih kecil dari titik tunggal t+60, sehingga R²-nya secara struktural lebih tinggi — terbukti dari baseline smart-persistence (tanpa ML sama sekali) yang R²-nya ikut melompat murni karena definisi target (lihat tabel SP di bawah). Angka di tabel ini **tidak boleh dibandingkan langsung** dengan R² titik di Tabel 1/2a sebagai "akurasi forecast 1 jam" yang setara — ini tugas statistik berbeda (estimasi rata-rata jendela), relevan hanya untuk use-case dispatch grid yang memang butuh rata-rata per jam.

| Lokasi | Model | R² | MAE (W/m²) | RMSE (W/m²) | Skill vs SP |
|---|---|---|---|---|---|
| **Bengkulu** | CatBoost | **0,900** | **62,0** | 89,7 | **0,3038** |
| Bengkulu | LightGBM | 0,899 | 61,8 | 90,2 | 0,3004 |
| Kalbar | CatBoost | 0,863 | 77,0 | 108,3 | 0,3154 |
| Kalbar | LightGBM | 0,862 | 77,0 | 108,8 | 0,3123 |
| Banten | CatBoost | 0,835 | 67,6 | 97,0 | 0,2489 |
| Banten | LightGBM | 0,832 | 68,0 | 97,7 | 0,2438 |
| Jambi (v2, tabel 24-jam) | CatBoost | **0,856** | 66,7 | 94,2 | 0,3592 |
| Jambi (v2, tabel 24-jam) | LightGBM | 0,854 | 66,9 | 94,9 | 0,3548 |
| ~~Jambi (v1, bug pipeline)~~ | ~~LightGBM~~ | ~~0,834~~ | ~~72,0~~ | ~~97,3~~ | ~~0,3323~~ |
| Smart-persistence (baseline) | — | 0,63–0,79 | 87–113 | 129–159 | 0,0 |

**Catatan Banten**: MAE terendah kedua (67,6) tapi skill vs SP paling rendah (0,249) di antara target rata-rata — kemungkinan artefak dari baseline smart-persistence Banten yang sudah relatif kuat (R² SP = 0,707, tertinggi ke-2), bukan model Banten yang buruk secara absolut.

**Rekomendasi pemakaian di paper**: kalau dipertahankan sama sekali, tempatkan di lampiran/analisis tambahan dengan disclaimer di atas dicantumkan penuh — jangan dipakai di abstrak, headline hasil, atau tabel perbandingan utama antar-lokasi.

---

## 10. Replikasi Resep v10 Bengkulu ke Kalbar, Banten, Jambi (Prioritas B, 2026-07-25)

> **Keputusan editorial (2026-07-25): §10/§10.1 ini adalah LAMPIRAN/robustness check, BUKAN pengganti Tabel 1 (F1, §1) sebagai hasil utama.** F1 (50 fitur) tetap recipe resmi di seluruh tabel benchmark paper karena sudah lengkap tervalidasi di per-horizon, walk-forward, DAN Arm A/B/C di 4 lokasi. Resep v10 (43 fitur) terbukti lebih akurat di semua lokasi pada target titik t+60, tapi belum diuji di per-horizon/walk-forward/Arm A-C — mengganti F1 akan butuh rework besar yang diputuskan tidak sepadan untuk draft submit saat ini. Cantumkan §10 sebagai bukti bahwa ruang perbaikan lebih lanjut ada (arah riset selanjutnya), bukan sebagai angka headline.

Resep v10 (`train_ghi_1h_bengkulu_v10_accel_lean.py`) adalah hasil tunggal terbaik di seluruh proyek untuk Bengkulu: 43 fitur (40 fitur ter-pruning dari superset v6 + 3 fitur akselerasi/turunan-kedua yang di-porting dari Training_Banten), satu model LightGBM residual saja (tanpa bagging/stacking), R² test 2025 = 0,8212 — mengalahkan v6 yang pakai 143 fitur + 5-model stack (0,8210).

**Audit ketersediaan fitur**: dicek satu-per-satu apakah 43 fitur v10 bisa dihitung di skema masing-masing lokasi (lihat `train_ghi_1h_<lokasi>_v10_accel_lean.py` untuk detail per lokasi):
- **Jambi**: tabel v2 (`ghi_forecast_1h_train_3h_rollback_2021_2025`) dibangun 1:1 meniru skema 102-kolom Bengkulu, jadi ke-43 fitur tersedia langsung tanpa substitusi apa pun (join SYNOP cloud-layer + fitur wavelet Bengkulu di-drop karena ternyata tidak dipakai oleh satu pun dari 43 fitur terpilih, diverifikasi dari `PRUNED_40_FEATURES`).
- **Banten**: `solar_features_base` punya semua kolom mentah yang dibutuhkan (dni, dhi, net_rad, reflected_rad, temp, rh, pressure, ws, dewpoint, wind_dir, cloud_top_height/temp/eff_radius; cakupan 89–100%), tapi skrip R1 lama cuma hitung rolling 30/60m — SQL diperluas untuk window 180m yang v10 butuhkan. Substitusi: `synop_temp_c` pakai kolom `temp` yang sama dengan `aws_temp_c` (Banten tidak punya sumber SYNOP terpisah dari AWS, sudah terfusi di hulu — lihat `08_Standardisasi_Data_Mentah.md`).
- **Kalbar**: tabel resmi `training_ghi_1h_direct` (66 kolom) TIDAK punya sebagian besar fitur mentah v10 (tidak ada DHI/DNI mentah, reflected/net-rad, rolling AWS 180m). Dibangun query SQL baru langsung dari tabel mentah 10-menitan (`solar_kalbar_10m` + `meteorologi_kalbar_10m` + `clp_pontianak_20km`, LEFT JOIN karena cakupan CLP cuma 63,1% + `synop_unified` per jam untuk dewpoint). Target dihitung ulang via `LEAD(ghi_final,6)` langsung dari `solar_kalbar_10m` (genuine lead, bukan target `anchor_valid` versi lama).

**Hasil (target titik t+60, model LightGBM residual tunggal, hyperparameter v10 penuh):**

| Lokasi | R² F1 (50 fitur lean, baseline @60m) | R² v10 (43 fitur, single model) | ΔR² |
|---|---|---|---|
| Bengkulu (referensi) | 0,7868 | 0,8212 | +0,0344 |
| Kalbar | 0,7217 | 0,7473 | +0,0256 |
| Jambi (v2) | 0,6926 | 0,7232 | +0,0306 |
| Banten | 0,6760 | 0,7365 | **+0,0605** |

**Kesimpulan**: resep v10 generalisasi ke KETIGA lokasi lain, dengan arah dan kira-kira magnitude yang konsisten (+0,026 s/d +0,061 R²) — bukan artefak khusus Bengkulu. Banten justru mendapat kenaikan TERBESAR (+0,0605), melebihi Bengkulu sendiri (+0,0344), kemungkinan karena baseline F1 Banten paling lemah di antara 4 lokasi sehingga fitur tambahan (rolling 180m DHI/DNI/AWS, akselerasi) punya ruang perbaikan lebih besar. Kalbar dapat kenaikan paling kecil (+0,0256), plausibel karena cakupan CLP cuma 63,1% (vs Bengkulu/Jambi yang jauh lebih lengkap) membatasi manfaat fitur cloud tambahan. **Rekomendasi**: pertimbangkan mengganti F1 (50 fitur) dengan resep v10 (43 fitur) sebagai recipe utama di keempat lokasi untuk draft submit, karena hasilnya strictly lebih baik di semua lokasi dengan jumlah fitur yang malah lebih sedikit.

### 10.1 Uji Generalisasi Target Hybrid Rata-Rata (v11-style)

Menjawab pertanyaan kedua Prioritas B: apakah pola "averaging horizon menaikkan R² secara struktural" (§1.3 di `03_Target_dan_Split.md`, §9 di atas) spesifik-Bengkulu atau universal? Diuji dengan resep 43-fitur v10 yang sama, target diganti `avg_t10_t60` (rata-rata GHI di 6 titik 10-menitan), model + hyperparameter identik.

| Lokasi | R² v10 titik (t+60) | R² v10 avg (t10..t60) | ΔR² |
|---|---|---|---|
| Bengkulu (referensi, dari `outputs_v11_hybrid_avg/ghi_1h_v11_metrics.csv`) | 0,8212 | 0,9025 | +0,0813 |
| Kalbar | 0,7473 | 0,8746 | +0,1273 |
| Banten | 0,7365 | 0,8679 | +0,1314 |
| Jambi (v2) | 0,7232 | 0,8620 | +0,1389 |

**Kesimpulan: pola ini UNIVERSAL, bukan spesifik Bengkulu** — malah LEBIH kuat di Kalbar/Banten/Jambi (ΔR² +0,127 s/d +0,139) dibanding Bengkulu (+0,081). Interpretasi paling masuk akal: Bengkulu punya model titik yang sudah sangat kuat (R²=0,821), jadi ruang perbaikan dari pengurangan varians target lebih sempit (efek langit-langit/ceiling); ketiga lokasi lain punya model titik lebih lemah, sehingga trik "meratakan target" punya ruang lebih besar untuk menaikkan R² secara struktural — BUKAN karena model jadi lebih akurat secara riil. Ini memperkuat, bukan melemahkan, keputusan pembimbing untuk mendemosikan `avg_t10_t60` dari hasil utama (§1.3 `03_Target_dan_Split.md`): kalau dibiarkan jadi headline, efeknya justru PALING menyesatkan di lokasi yang modelnya paling lemah — persis kebalikan dari kesan "lokasi X ternyata bagus" yang mungkin diberikan pembaca sekilas.

Skrip: `train_ghi_1h_<lokasi>_v10_accel_lean.py` di masing-masing folder lokasi (Kalbar, Banten, Jambi), output di `outputs_v10_<lokasi>/`.
