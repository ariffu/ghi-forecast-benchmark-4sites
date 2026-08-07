# Audit: Kenapa Dataset Jambi Jauh Lebih Kecil — Bukan (Hanya) Soal Ketersediaan Data

**Pemicu**: dataset R1 Jambi (22.314 baris total: train 8.701 + val 4.102 + test 9.511) jauh lebih kecil dari 3 lokasi lain (Bengkulu 58.732, Kalbar 81.851, Banten 90.488) — padahal Jambi mencakup periode 2021–2025, **sama panjang dengan Bengkulu** dan **lebih panjang** dari Kalbar/Banten (2022–2025). Dugaan awal (data historis Jambi memang lebih sedikit) **terbukti tidak sepenuhnya benar** setelah diaudit langsung ke database — sebagian besar penyusutan berasal dari **cara pipeline Jambi dibangun**, bukan dari ketersediaan sensor.

---

## 1. Rantai Penyusutan Data (Diverifikasi Langsung ke Database)

| Tahap | Baris | Catatan |
|---|---|---|
| `jambi_sch.jambi_obs_combined` / `solar_radiation_valid` (tabel gabungan hasil audit kualitas) | 123.093 | Rentang 2021-01-01 07:00 s.d. 2025-12-31 17:40 — **sudah dipangkas ke jam siang** (07:00–17:40) di level view |
| `dfm_with_clp_stats.parquet` (sumber aktual yang dibaca skrip `train_ghi_1h_jambi_R1_benchmark.py`) | **87.008** | Turun 29% dari 123.093 — **BUKAN akibat JOIN CLP** (dugaan awal di dokumen ini, sudah dikoreksi, lihat §2.4) |
| Setelah filter kontinuitas `has_continuous_3h_history` & `has_continuous_1h_forward` (18 langkah 10-menit berturutan ±30 detik, dihitung dari nol di Python) | **38.706** | **Hanya 44,5% dari 87.008 lolos** — lihat akar masalah di §2 |
| Setelah filter target valid + `sun_gt5_t60` + split train/val/test | 22.314 | Angka final yang dilaporkan di R1 |

**Titik kritis ada di baris ketiga**: dari 87.008 baris sumber, filter kontinuitas sendirian membuang 55,5% — jauh lebih besar dari yang wajar untuk data yang secara individual 96,3% "sehat" (lihat §2).

---

## 2. Akar Masalah: Filter Kontinuitas Salah Pasang untuk Sumber Data yang Sudah Dipangkas ke Siang Hari

### 2.1 Bukti kuantitatif

Diaudit langsung dari `dfm_with_clp_stats.parquet` (87.008 baris):
- **96,3%** dari seluruh langkah antar-baris berjarak tepat 600 detik ±30 detik (artinya secara individual, hampir semua data "sehat" / tidak ada gap).
- Tapi filter kontinuitas mensyaratkan **18 langkah berturutan tanpa cela** (3 jam riwayat) **ditambah** 6 langkah maju tanpa cela (1 jam ke depan) — total jendela 4 jam yang harus 100% mulus.
- Karena syaratnya bersifat majemuk (compounding), 3,7% langkah yang "cacat" cukup untuk membuat **55,5% baris gagal** — bukan 3,7%.

### 2.2 Kenapa gagalnya sistemik, bukan acak

Distribusi ukuran gap menunjukkan pola yang sangat mencurigakan: gap sebesar **48.600–52.800 detik (≈13,5–14,7 jam)** muncul >1.300 kali — ini **persis** selisih antara jam terakhir data siang satu hari (≈17:40) dan jam pertama data siang hari berikutnya (≈06:50–07:00). Artinya sebagian besar "gap" yang terdeteksi bukan data hilang secara acak, melainkan **batas hari yang memang sengaja tidak ada** karena sumber datanya sudah dipangkas ke jam siang saja.

**Konsekuensi struktural**: karena tidak ada data sebelum matahari terbit, **setiap baris dalam ~3 jam pertama tiap hari (kira-kira pukul 07:00–10:00) otomatis gagal syarat "3 jam riwayat kontinu"** — bukan karena datanya hilang/rusak, tapi karena riwayat 3 jam ke belakang itu **memang tidak mungkin ada** dalam tabel yang sudah dipangkas ke siang hari. Ini menghapus kira-kira 25–30% baris siang hari **di setiap hari, secara sistematis**, terlepas dari kualitas data itu sendiri.

### 2.3 Perbandingan dengan 3 Lokasi Lain — Bukti Bahwa Ini Bug Implementasi, Bukan Karakteristik Data Jambi

Diverifikasi langsung dari database masing-masing lokasi:

| Lokasi | Tabel sumber skrip R1 | Total baris | Cakupan jam | Metode filter |
|---|---|---|---|---|
| **Bengkulu** | `ghi_forecast_1h_train_3h_rollback_2021_2025` (view) | 259.534 | **24 jam penuh** (≈10.800/jam merata) | Flag pre-computed `is_model_ready` + `has_continuous_3h_history` di level SQL/view — 117.559 baris lolos (45,3%), dihitung dari basis 24-jam sehingga riwayat dini hari tersedia |
| **Kalbar** | `training_ghi_1h_direct` (tabel) | 210.384 | **24 jam penuh** (8.766/jam persis merata) | Flag tunggal pre-computed `anchor_valid` — 81.851 baris (38,9%) |
| **Banten** | `solar_features_base` (tabel) | 210.241 | **24 jam penuh** (≈8.760/jam merata) | SQL native: `ghi_lag_180m IS NOT NULL` (cukup ada nilai lag, bukan syarat 18-langkah-berturutan keras) |
| **Jambi** | `dfm_with_clp_stats.parquet` | 87.008 | **Hanya siang** (07:00–17:40) | Python from-scratch: 18 langkah berturutan ±30 detik, dihitung ulang tanpa memperhitungkan bahwa sumbernya sudah dipangkas |

**Tiga lokasi lain semuanya bersumber dari tabel/view 24-jam penuh** dengan flag kualitas yang sudah dihitung dengan benar di level data (bukan di-mask ulang secara naif di Python). **Hanya Jambi** yang bersumber dari parquet siang-hari-saja lalu menerapkan syarat kontinuitas yang didesain untuk sumber 24-jam ("diporting 1:1 dari Bengkulu" — lihat komentar di kode `train_ghi_1h_jambi_R1_benchmark.py` baris 4). Ini **bug harmonisasi implementasi**, bukan bukti bahwa data mentah Jambi lebih buruk dari 3 lokasi lain.

### 2.4 Koreksi (2026-07-25): Penyusutan 123.093→87.008 BUKAN dari JOIN CLP — Root Cause Sebenarnya Ditemukan

Rekomendasi Prioritas A0 butir 3 di dokumen ini meminta audit ulang penyusutan 123.093→87.008. Sudah diaudit langsung ke `jambi.duckdb`, dan **dugaan awal di §1 salah**:

**Uji langsung**: `INNER JOIN solar_radiation_valid × jambi_clp_combined` pada `timestamp_wib` menghasilkan **120.892 baris cocok** dari 123.093 (98,2% match rate, hanya 2.201 baris timestamp yang benar-benar tidak ada padanan CLP-nya) — jauh dari 87.008. **JOIN dengan CLP bukan penyebabnya.**

**Root cause sebenarnya**: file `dfm_with_clp_stats.parquet` yang dibaca skrip R1 v1 ternyata dibangun dari `dfm_banten_features.parquet` (87.008 baris — persis sama, karena `merge_clp_and_ablate.py` baris 44 melakukan `LEFT JOIN` pada CLP yang **tidak pernah mengurangi baris**, terbukti 87.008→87.008 sebelum/sesudah merge). Jadi drop ke 87.008 sudah terjadi **sebelum** tahap CLP sama sekali — di skrip pembangun `dfm_banten_features.parquet` sendiri, yang sudah tidak ada lagi di folder kerja (tidak bisa direkonstruksi persis).

**Nama file yang menyesatkan**: `dfm_banten_features.parquet` memuat data Jambi asli (diverifikasi silang: nilai `ghi_consolidated` di file ini cocok persis dengan `jambi_sch.solar_radiation_valid` untuk timestamp yang sama) — nama "banten" kemungkinan besar sisa penamaan skrip/template yang di-copy dari pipeline Banten lalu dipakai ulang untuk Jambi tanpa di-rename. Bukan indikasi data lokasi tertukar, tapi tetap contoh lain dari pola "logika/script di-porting antar-lokasi tanpa adaptasi penuh" yang sudah beberapa kali ditemukan di proyek ini (lihat juga temuan `clp_cot_lag_20m` interpolasi di Kalbar, `02_Feature_Engineering.md` §6).

**Dua faktor yang terbukti berkontribusi ke penyusutan 123.093→87.008**:
1. **Filter `ghi_quality_flag='good'`**: memotong 123.093 → 98.897 (−19,6%). Baris ber-flag `ptm_gap_filled`, `cloud_enhancement`, `capped_kt_gt_1.5`, dll (24.196 baris) dibuang — filter kualitas yang sah, tapi jauh lebih ketat daripada apa yang diterapkan di 3 lokasi lain (belum ada threshold closure-violation yang seragam, lihat `08_Standardisasi_Data_Mentah.md` §3).
2. **Filter tambahan tak-teridentifikasi**: 98.897 → 87.008 (−11.889 baris, −12%) tidak bisa direproduksi dari kolom yang tersedia (bukan filter tanggal mulai 2021-06-01 saja — semua baris "good" memang sudah berada di rentang itu; bukan filter AOD/aerosol non-null; bukan filter elevasi matahari). Skrip pembangun `dfm_banten_features.parquet` sudah tidak ada di folder kerja, jadi kriteria pastinya tidak bisa dipastikan.

**Implikasi**: temuan §2.1–2.3 (filter kontinuitas 3-jam yang salah pasang untuk sumber siang-hari-saja) **tetap berlaku sebagai penyebab utama** hilangnya baris (87.008→38.706, −55,5%) — itu penyebab dominan dan sudah diperbaiki di tabel v2 (§5). Penyusutan 123.093→87.008 di tahap sebelumnya **juga nyata** (bukan artefak JOIN), tapi skalanya lebih kecil dan sebagian besar (19,6% dari 29% total) berasal dari filter kualitas yang sah (`ghi_quality_flag='good'`), bukan bug. Karena tabel v2 dibangun ulang dari sumber mentah (`asrs_jambi_menit_rev`) dengan heuristik QC sendiri (bukan mewarisi `dfm_with_clp_stats.parquet` sama sekali), masalah ini **tidak terbawa ke v2** — tapi perlu diingat tabel v2 belum melalui audit kualitas closure/sentinel yang setara (sudah dicatat sebagai keterbatasan di §5).

---

## 3. Implikasi untuk Tabel Benchmark yang Sudah Ada

- Angka R1 Jambi saat ini (R²=0,676 titik / 0,831 rata-rata, test n=9.511) dihitung dari subset data yang **secara sistematis kehilangan jam-jam awal pagi setiap hari** — bukan sampel acak dari kondisi siang hari penuh. Kalau kondisi radiasi pagi (07:00–10:00) punya karakteristik berbeda secara sistematis dari siang/sore (mis. lebih variabel karena awan konvektif pagi, atau justru lebih stabil), R² Jambi yang dilaporkan **bisa bias** — arah biasnya belum diketahui tanpa menjalankan ulang dengan data yang benar.
- Jumlah baris Jambi yang jauh lebih kecil (22.314 vs 58k–90k) **memperbesar varians estimasi metrik** dibanding 3 lokasi lain — sudah dicatat sebagai keterbatasan di `01_Dataset.md` dan `07_Status_dan_Rencana_Selanjutnya.md`, tapi sekarang jelas sebagian besar penyebabnya bisa diperbaiki (bukan batasan ketersediaan data yang inheren).
- **Peringkat performa Jambi (terendah/kedua-terendah dari 4 lokasi, lihat `06_Perbandingan_4_Lokasi.md`) perlu dipertimbangkan ulang** setelah pipeline diperbaiki — belum tentu Jambi benar-benar "lokasi paling sulit", bisa jadi sebagian gap R²-nya berasal dari kehilangan sepertiga data siang hari yang justru lebih predictable (pagi/sore biasanya NRMSE lebih rendah menurut pola umum di `Balai/02_Metodologi/05_Evaluasi_Metrik.md` §4 — kalau jam-jam "mudah" ini yang justru terbuang, R² Jambi mungkin under-estimated).

---

## 4. Rekomendasi Perbaikan

### Prioritas tertinggi — sebelum angka Jambi dipakai final di paper

1. **Cari/bangun sumber data Jambi 24-jam-penuh** setara `ghi_forecast_1h_train_3h_rollback_2021_2025` (Bengkulu) atau `training_ghi_1h_direct` (Kalbar). Kandidat yang perlu dicek: `jambi_sch.asrs_jambi_awscenter` (272.091 baris, rentang 2022-01-18 s.d. 2025-12-31, kolom radiasi mentah `global_rad_round` dll — perlu diverifikasi apakah mencakup malam hari) atau `asrs_jambi_menit_rev` (2.846.880 baris, resolusi per-menit — kemungkinan sumber paling mentah dan paling lengkap, belum diagregasi ke 10 menit).
2. **Ganti logika kontinuitas Python Jambi** dengan pendekatan yang konsisten dengan Banten (`ghi_lag_180m IS NOT NULL`, longgar) atau Bengkulu/Kalbar (flag pre-computed di level SQL, bukan mask keras 18-langkah dihitung ulang tiap kali) — supaya jam-jam awal pagi tidak otomatis terbuang hanya karena sumbernya dipangkas ke siang hari.
3. **Jalankan ulang R1 Jambi** setelah perbaikan, bandingkan n baris dan R² baru dengan angka lama di dokumen ini dan di `06_Perbandingan_4_Lokasi.md` — update tabel benchmark utama begitu tersedia.
4. **Audit ulang CLP-join loss** (123.093 → 87.008, −29%) — cek apakah ini disebabkan gap riil di `jambi_clp_combined`/`clp_jambi`, atau JOIN key yang tidak match sempurna (mis. rounding timestamp) yang sebenarnya bisa diperbaiki.

### Prioritas sedang — untuk kelengkapan `08_Standardisasi_Data_Mentah.md`

5. Setelah pipeline Jambi diperbaiki, jalankan audit konsistensi fisis ala Kalbar (`note_09_audit_konsistensi_fisis.md`) pada dataset Jambi yang sudah direvisi, supaya perbandingan kualitas data 4 lokasi (§5 di `08_Standardisasi_Data_Mentah.md`) memakai basis yang benar-benar sebanding.
6. Cek juga apakah Banten dan Kalbar (yang filternya lebih longgar dari desain awal Jambi) mungkin **kurang ketat** dibanding yang seharusnya — tujuannya bukan menyeragamkan ke yang paling longgar begitu saja, tapi memastikan *definisi kontinuitas yang sama* diterapkan dengan *cara perhitungan yang benar* di keempat lokasi.

---

## 5. Update — Tabel Perbaikan Sudah Dibangun dan Diverifikasi

Tabel baru `ghi_forecast_1h_train_3h_rollback_2021_2025` untuk Jambi (skema 102 kolom identik dengan Bengkulu) sudah dibangun dari `asrs_jambi_menit_rev` (per-menit, 24-jam-penuh) + `meteo_obs_10min` + `jambi_clp_combined` + `synop_jambi_combined`, memakai grid waktu 10-menit lengkap dan fitur lag/rolling via window function SQL (bukan pandas shift). Skrip: `DuckDB_jambi/build_ghi_forecast_1h_rollback_jambi.py`.

**Hasil verifikasi**:

| Metrik | Pipeline lama | Tabel baru |
|---|---|---|
| Baris `is_model_ready & sun>5°` (2021–2025) | 22.314 | **96.576** (+333%) |
| Distribusi per jam (07:00–16:00) | Timpang — jam pagi (07–09) jauh lebih sedikit dari siang/sore | **Merata** — semua jam 07:00–16:00 berkisar 8.476–8.596 baris |
| Baris per tahun | tidak dirinci | 2021: 7.405 (parsial, wajar — tahun pertama data) · 2022: 22.782 · 2023: 24.621 · 2024: 19.052 · 2025: 22.716 |

Sampel jam 05:00–08:00 (2023-06-01) dicek manual: nilai GHI malam mendekati nol/negatif kecil (wajar, noise sensor), naik mulus melewati fajar, `has_continuous_3h_history=1` dan `is_model_ready=1` sudah benar sejak jam 05:00–07:00 (dulu pasti gagal di pipeline lama), dan `target_ghi_1h_ahead` tervalidasi cocok persis dengan `ghi_now` baris +60 menit.

**Lokasi file**: `C:\Users\ariff\DuckDB_jambi\jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb`, tabel `jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025`.

> ⚠️ **Kenapa file terpisah, bukan langsung ditambahkan ke `jambi.duckdb`**: `jambi.duckdb` di lingkungan sandbox saat ini punya file `.wal` tersisa dari percobaan awal yang gagal, dan sandbox tidak mengizinkan operasi hapus file — jadi tidak bisa dibuka untuk ditulis lagi dari sesi ini. Tabel baru dibangun di file terpisah (hanya *membaca* `jambi.duckdb`, tidak pernah menulis ke situ, jadi `jambi.duckdb` asli tidak tersentuh/rusak). Untuk menyatukan ke `jambi.duckdb`, jalankan ini di DuckDB Anda sendiri (di komputer, bukan di sandbox ini, yang tidak punya keterbatasan izin ini):
> ```sql
> ATTACH 'jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb' AS newdb;
> CREATE TABLE jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025 AS
>     SELECT * FROM newdb.jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025;
> ```

**Keterbatasan yang jujur perlu disebut**: kolom `*_qc_status`/`master_qc_status` di tabel baru ini adalah heuristik sederhana (ada/tidak data per sumber), BUKAN audit kualitas penuh (closure-check GHI=DHI+DNI·sinθ, deteksi sentinel, dsb) seperti yang sudah dilakukan untuk `solar_radiation_valid` Jambi lama atau `asrs_bengkulu_combined`. Sebelum dipakai sebagai sumber final R1, jalankan audit kualitas yang sepadan (lihat Prioritas A `08_Standardisasi_Data_Mentah.md`) pada tabel baru ini.

### Langkah selanjutnya
1. Satukan tabel ke `jambi.duckdb` (lihat perintah di atas) — **masih perlu dilakukan pengguna secara manual**, belum dilakukan.
2. ~~Jalankan ulang `train_ghi_1h_jambi_R1_benchmark.py` dengan sumber baru ini~~ — **selesai, lihat §6 di bawah.**
3. ~~Update `06_Perbandingan_4_Lokasi.md` dan `01_Dataset.md`~~ — **selesai.**

---

## 6. Update — R1 Jambi Dijalankan Ulang dengan Tabel Baru (2026-07-24)

Skrip baru `DuckDB_jambi/train_ghi_1h_jambi_R1_benchmark_v2.py` mem-port `train_ghi_1h_bengkulu_R1_benchmark.py` 1:1 (fitur F1 50-kolom, split kronologis train<2024/val 2024/test 2025, filter `is_model_ready=1 AND has_continuous_3h_history=1 AND ghi_now BETWEEN 0 AND 1400`, model LightGBM residual + CatBoost direct, baseline smart-persistence) — hanya path database dan koordinat stasiun (`-1.5833, 103.6667`) yang diganti. Dijalankan dua kali sebagai pengecekan konvergensi: mode cepat (n_estimators 800, lr 0,05) dan mode medium (n_estimators 3000, lr 0,03) — hasil keduanya sangat dekat (Δ R² < 0,002), mengonfirmasi angka sudah konvergen, bukan artefak under-training.

**Hasil (test 2025, mode medium, `outputs_R1_jambi_v2_medium/ghi_1h_R1_results.csv`)**:

| Model | Target | n test | R² | MAE (W/m²) | RMSE (W/m²) | Skill vs SP |
|---|---|---|---|---|---|---|
| smart_persistence | point_t60 | 21.129 | 0,4266 | 142,0 | 205,2 | 0,0 |
| lgbm_residual | point_t60 | 21.129 | 0,6931 | 108,9 | 150,1 | 0,2683 |
| **catboost_direct** | **point_t60** | 21.129 | **0,6932** | 108,8 | 150,1 | 0,2685 |
| smart_persistence | avg_t10_t60 | 21.098 | 0,6495 | 99,8 | 147,0 | 0,0 |
| lgbm_residual | avg_t10_t60 | 21.098 | 0,8541 | 66,9 | 94,9 | 0,3548 |
| **catboost_direct** | **avg_t10_t60** | 21.098 | **0,8561** | 66,7 | 94,2 | 0,3592 |

**Perbandingan lama vs baru**:

| | R² titik (CatBoost) | R² rata-rata (CatBoost) | Test n |
|---|---|---|---|
| v1 (bug pipeline, siang-hari-saja) | 0,676 | 0,831 | 9.511 |
| v2 (tabel 24-jam-penuh) | **0,693** | **0,856** | 21.129 |
| Δ | **+0,017** | **+0,025** | **+11.618 (+122%)** |

**Implikasi**: dugaan di §3/§5 terbukti — R² Jambi memang under-estimated di v1. Peringkat 4-lokasi berubah: Jambi naik dari peringkat terakhir menjadi **mengungguli Banten** (0,682 titik / 0,835 rata-rata) di kedua target. Peringkat baru (titik): Bengkulu (0,792) > Kalbar (0,728) > **Jambi (0,693)** > Banten (0,682). `06_Perbandingan_4_Lokasi.md` dan `01_Dataset.md` sudah diperbarui dengan angka ini.

**Yang BELUM dikerjakan** (di luar cakupan sesi ini, lihat `07_Status_dan_Rencana_Selanjutnya.md`):
- Walk-forward 5-fold Jambi v2 (Tabel 2b di `06_Perbandingan_4_Lokasi.md` masih pakai angka v1 lama, R²=0,6249 — belum direplikasi dengan tabel baru).
- R8 Arm A/B/C Jambi belum dijalankan ulang dengan tabel v2 (angka existing di `06_Perbandingan_4_Lokasi.md` Tabel 2c/2a_v2/2d masih dari pipeline v1).
- Audit kualitas fisis penuh (closure-check, sentinel value) pada tabel v2 — lihat catatan keterbatasan di §5 di atas.
- Menyatukan tabel v2 ke `jambi.duckdb` utama (langkah #1 di atas, manual oleh pengguna).

---

## 7. Catatan Metodologis

Temuan ini adalah contoh konkret dari poin di `08_Standardisasi_Data_Mentah.md`: menyeragamkan *resep fitur dan protokol evaluasi* (F1/F2, split, model) tidak otomatis membuat *pipeline pembangunan datanya* benar-benar identik. "Porting 1:1" logika dari satu lokasi ke lokasi lain bisa gagal diam-diam kalau asumsi struktural sumber datanya berbeda (di sini: 24-jam-penuh vs siang-hari-saja) — dan kegagalannya tidak memunculkan error, hanya diam-diam membuang banyak data yang terlihat seperti "keterbatasan data lokasi" padahal sebenarnya bug pipeline.
