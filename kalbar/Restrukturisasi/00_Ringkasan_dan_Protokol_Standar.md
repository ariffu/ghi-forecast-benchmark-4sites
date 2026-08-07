# Restrukturisasi Penelitian GHI Nowcasting/Forecasting — Protokol Standar (Basis Bengkulu)

**Tujuan dokumen**: menyatukan penentuan data, feature engineering, arsitektur model, training, dan evaluasi ke dalam **satu protokol identik** yang berlaku untuk keempat lokasi (Bengkulu, Kalbar, Jambi, Banten), sebagai dasar bab Metodologi paper *benchmarking*. Bengkulu dipakai sebagai **standar rujukan** karena, di bawah protokol yang sama persis, Bengkulu konsisten mencapai akurasi tertinggi.

Status: draft restrukturisasi, disusun dari catatan eksperimen yang sudah ada (vault `Balai/04_Eksperimen`, `Balai/02_Metodologi`) dan hasil aktual di folder kerja tiap lokasi (`outputs_R1_*`, `outputs_R8_*`). Sebagian besar protokol ini **sudah pernah dijalankan** (disebut proyek "R1 Harmonised Benchmark" dan "R8 Comprehensive Framework") — dokumen ini merapikannya menjadi satu alur baku, bukan memulai dari nol.

---

## 1. Mengapa Bengkulu jadi standar?

Di bawah konfigurasi **identik** (fitur, split, filter, model) — bukan konfigurasi terbaik masing-masing lokasi — Bengkulu unggul di kedua definisi target:

| Lokasi | R² titik (t+60m) | R² rata-rata (t+10..t+60) | Walk-forward R² (±std) |
|---|---|---|---|
| **Bengkulu** | **0.792** | **0.900** | **0.7944 ± 0.0197** |
| Kalbar | 0.728 | 0.863 | 0.6628 ± 0.0424 |
| Banten | 0.682 | 0.835 | 0.6429 ± 0.0574 |
| Jambi | 0.676 | 0.831 | 0.6249 ± 0.0533 |

(Sumber: `DuckDB_kalbar/r1_compiled/TABLE_2a_test_2025_results.csv` dan `note_18_r1_harmonised_benchmark.md`, model CatBoost direct, test 2025.)

Bengkulu juga paling **stabil** (walk-forward σ terkecil, 2–3× lebih kecil dari lokasi lain) — bukan cuma rata-rata tertinggi tapi juga paling konsisten antar periode. Hipotesis penjelas (belum dibuktikan kausal, lihat `06_Perbandingan_4_Lokasi.md`): rezim awan pesisir/trade-wind Bengkulu lebih predictable dibanding rezim konvektif ekuatorial (Kalbar) atau orografis (Jambi), plus kontinuitas data CLP satelit Bengkulu yang lebih tinggi (99.8%).

**Konsekuensi metodologis**: karena Bengkulu unggul di bawah protokol yang sama (bukan karena resep fitur/model berbeda), maka wajar dijadikan **rujukan desain protokol** — tapi bukan berarti hasil numeriknya (R²) bisa langsung "dipinjam" ke lokasi lain. Setiap lokasi tetap dilatih dan dievaluasi sendiri dengan protokol yang sama; yang diseragamkan adalah *cara*-nya, bukan angkanya.

---

## 2. Definisi Protokol Standar (ringkas — detail per bagian di file 01–04)

```
Horizon         : 1 jam ke depan (t+60 menit)
Target          : DUA definisi, dilaporkan berdampingan (lihat 03_Target_dan_Split.md)
                  (a) point_t60      — GHI titik pada t+60m (tugas presisi-menit)
                  (b) avg_t10_t60    — rata-rata GHI dari t+10m s.d. t+60m (tugas dispatch/operasional)
Resolusi input  : 10 menit (dense), fitur lag/rolling dihitung di resolusi native
Fitur           : F1 = 50 fitur lean (GHI history 16 + kt 9 + CLP 15 + waktu siklik 6 + future-deterministic 4)
                  F2 = F1 + 5 fitur AWS meteo (untuk uji redundansi, lihat Arm A)
Split           : kronologis — train < 2024-01-01 | validasi 2024 | test 2025
                  (Bengkulu/Jambi mulai 2021; Kalbar/Banten mulai 2022 — lihat 01_Dataset.md)
Filter          : sun_altitude > 5° (di titik anchor dan di t+60) | GHI dalam 0–1400 W/m² |
                  anchor_valid = true | gap kontinuitas time-series ±30 detik
Model           : LightGBM residual (PRIMER, produksi) + CatBoost direct (SENSITIVITAS/pembanding)
                  Baseline wajib: smart-persistence (bukan persistence naif)
Validasi        : walk-forward 5-fold kronologis (LightGBM residual × target titik) untuk cek stabilitas,
                  di atas evaluasi test-2025 tunggal
Metrik          : R², MAE, RMSE (W/m²), skill = 1 − RMSE/RMSE_smart-persistence
Leakage guard   : future regressors hanya yang deterministik (I_clr, cos θ) atau proyeksi dari data lampau
                  (EMA/smart-persistence); tidak ada nilai aktual masa depan di fitur manapun
```

Catatan skrip acuan (sudah ada, tinggal direplikasi/diverifikasi per lokasi): `train_ghi_1h_<lokasi>_R1_benchmark.py` dan `train_ghi_1h_<lokasi>_R8_armA/B/C.py` di masing-masing folder lokasi. Lihat `07_Status_dan_Rencana_Selanjutnya.md` untuk status kelengkapan tiap skrip.

---

## 3. Peringatan Metodologis Paling Penting

> **R²=0,90 Bengkulu bukan angka yang sebanding langsung dengan R² titik (~0,68–0,79) di laporan lain.**

Angka 0,90 dicapai pada target **rata-rata GHI 1 jam ke depan** (avg_t10_t60), bukan nilai instan pada menit ke-60 (point_t60). Merata-ratakan 6 titik 10-menitan secara inheren mengurangi varians target — ini tugas statistik yang lebih mudah, **bukan** model yang "lebih pintar", dan **bukan** kebocoran data (tidak ada nilai aktual masa depan yang bocor ke fitur — forecast tetap genuine 1 jam ke depan). Ini juga **berbeda** dari kesalahan evaluasi yang pernah ditemukan di Kalbar (nowcasting 10-menit dirata-rata ke jam lalu diklaim sebagai forecast 1 jam — itu trik evaluasi yang tidak valid dan sudah dikoreksi, lihat `Balai/04_Eksperimen/Bengkulu/Catatan/note_02_pipeline_hourly_agregat_dan_klarifikasi_kalbar.md`).

**Implikasi untuk paper**: laporkan **kedua target berdampingan** untuk semua lokasi (sudah tersedia di `r1_compiled/TABLE_2a_test_2025_results.csv`), dan jelaskan eksplisit di Metodologi bahwa keduanya adalah tugas prediksi yang berbeda kegunaan (point = kontrol real-time/menit; avg = dispatch grid/perencanaan operasi). Jangan membandingkan R² titik satu lokasi dengan R² rata-rata lokasi lain seolah setara.

---

## 4. Isi Folder Ini

| File | Isi |
|---|---|
| `01_Dataset.md` | Sumber data, path database, jumlah baris train/val/test per lokasi, pemetaan nama kolom ke skema standar |
| `02_Feature_Engineering.md` | Definisi lengkap F1/F2, filter kualitas, aturan anti-leakage, daftar teknik yang terbukti GAGAL (jangan diulang) |
| `03_Target_dan_Split.md` | Definisi target titik vs rata-rata, strategi split kronologis, walk-forward |
| `04_Model_dan_Training.md` | Arsitektur model standar, hyperparameter, aturan ensemble, baseline smart-persistence |
| `05_Hasil_Referensi_Bengkulu.md` | Hasil Bengkulu sebagai rujukan tertinggi (R1 harmonis + eksplorasi lanjutan v10/v11) |
| `06_Perbandingan_4_Lokasi.md` | Tabel benchmark lengkap 4 lokasi, analisis kesenjangan geografis |
| `07_Status_dan_Rencana_Selanjutnya.md` | Checklist status implementasi R1/R8 per lokasi, langkah yang masih kurang, pemetaan ke bab paper |
| `08_Standardisasi_Data_Mentah.md` | **Axis kedua restrukturisasi**: ceiling R² dipengaruhi kualitas/struktur data mentah, bukan cuma protokol training — bukti audit per lokasi, apa yang sudah vs belum diseragamkan di level data, rekomendasi bertahap |
| `09_Audit_Volume_Data_Jambi.md` | **Temuan konkret**: dataset Jambi (22.314 baris) jauh lebih kecil dari 3 lokasi lain (58k–90k) bukan karena data mentahnya lebih sedikit, tapi bug pipeline — filter kontinuitas yang di-porting dari Bengkulu diterapkan ke sumber data yang sudah dipangkas ke siang hari, membuang ~3 jam pertama tiap hari secara sistematis. Perbaikan disarankan sebelum angka Jambi dipakai final. |

> **Catatan penting**: dokumen 01–07 menyeragamkan *cara mengevaluasi* (protokol). Dokumen 08 menyoroti bahwa *data yang dievaluasi* juga perlu diseragamkan/diaudit — dua axis yang berbeda dan **sama-sama membatasi ceiling R²**. Audit Kalbar (`note_09_audit_konsistensi_fisis.md`) membuktikan kuantitatif bahwa sebagian ceiling berasal dari noise struktural di input (mismatch spasial satelit-vs-titik, kontradiksi antar-produk cloud), bukan keterbatasan model atau training. Baca `08_Standardisasi_Data_Mentah.md` sebelum menyimpulkan bahwa kesenjangan R² antar-lokasi murni soal rezim iklim.

---

## 5. Sumber Utama yang Dirujuk

- Vault Obsidian `Balai/04_Eksperimen/Bengkulu/` (14 catatan sesi, `note_00`–`note_14`)
- Vault Obsidian `Balai/02_Metodologi/01_Setup_Data.md` s.d. `07_Teknik_Tidak_Transfer.md` (sintesis lintas-4-lokasi, berbasis presentasi Jitkomut Songsiri 2025 + validasi empiris Indonesia)
- `DuckDB_kalbar/note_18_r1_harmonised_benchmark.md`, `R8_IMPLEMENTATION_GUIDE.md`, `R8_EXECUTION_PLAN.md`, `note_20_r8_findings_and_integration.md`
- `Duckdb_Banten/BANTEN_R1_R8_SUMMARY.md`, `BANTEN_S44_METEO_ISOLATION.md`
- `bengkulu_ghi_julius/Catatan/note_13_ringkasan_dan_rekomendasi_fase_julius.md`, `note_14_teknik_banten_dan_target_hybrid_rata_rata_jam.md`
- Output CSV aktual: `outputs_R1_*/ghi_1h_R1_results.csv`, `outputs_R8_*/arm_{A,B,C}_results.csv` di keempat folder lokasi, dan `DuckDB_kalbar/r1_compiled/*.csv`
