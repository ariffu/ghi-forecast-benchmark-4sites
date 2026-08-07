# Standardisasi Data Mentah — Ceiling R² Ditentukan di Input, Bukan Hanya di Training

**Premis dokumen ini**: dokumen 00–07 menyeragamkan *protokol evaluasi* (fitur, split, model, metrik) — dan itu perlu, supaya angka antar-lokasi bisa dibandingkan secara adil. Tapi protokol yang seragam **tidak menjamin data mentahnya sebanding**. Audit yang sudah pernah dilakukan (terutama di Kalbar) membuktikan secara kuantitatif bahwa sebagian ceiling R² berasal dari **kualitas dan struktur data input** — sesuatu yang tidak bisa diperbaiki dengan mengganti model atau menambah fitur, karena noise-nya sudah ada sebelum tahap training dimulai.

---

## 1. Bukti Paling Kuat: Audit Konsistensi Fisis Kalbar

`Balai/04_Eksperimen/Kalbar/Catatan/note_09_audit_konsistensi_fisis.md` — disusun setelah enam sudut eksplorasi teknik (algoritma, window, rezim, PCA, PSO, deep learning) semua mentok di ceiling yang sama, lalu dicari bukti di level data:

| Temuan | Angka | Implikasi |
|---|---|---|
| **Mismatch spasial struktural** | CLP/ARP = rata-rata buffer radius **30 km** (grid 10×11 = 110 piksel Himawari); radiasi = **titik tunggal** pyranometer | Awan konvektif tropis (diameter beberapa km) bisa menutupi/tidak menutupi titik stasiun independen dari kondisi rata-rata 60 km sekitarnya — akar dari semua temuan di bawah |
| **Dua produk satelit kontradiktif** | `cloud_present` (CLP) vs `clear_sky` (ARP) bertentangan **64,9%** waktu | 37,3% dari seluruh observasi siang: CLP bilang "ada awan" tapi radiasi aktual hampir clear-sky (kt=0,878) — false positive terhadap dampak radiasi riil |
| **`clear_sky` bukan berarti "tidak berawan"** | Hanya 1 dari 65.034 baris `clear_sky=TRUE` yang benar-benar `clear_or_no_retrieval` menurut CLP | Nama variabel menyesatkan — artinya "retrieval aerosol valid", bukan indikator visual awan |
| **Volatilitas GHI 3× lebih cepat dari fitur satelit** | rasio Δ/std: kt=0,328 vs CLOT=0,107 | Resolusi temporal 10 menit Himawari terlalu kasar untuk dinamika awan konvektif |
| **Offset sinkronisasi waktu** | Korelasi CLOT(t) vs kt(t+lag) puncak di **lag +10 menit**, bukan lag=0 | Indikasi pergeseran assignment waktu scan satelit vs dampak riil di permukaan |
| **Kontradiksi langsung CLOT vs GHI** | 1,8% baris siang: CLOT tebal tapi GHI tinggi, atau sebaliknya | Noise struktural langsung di pasangan fitur-target |

**Kesimpulan audit ini (dikutip langsung karena penting)**: *"Ceiling R² ≈ 0,73 (nilai sesaat) / 0,86 (rata-rata jam) — tidak bisa ditembus dengan teknik ML apa pun karena noise-nya ada DI INPUT, bukan di model."*

**Ini investigasi paling rigorous yang ada di keempat lokasi.** Belum direplikasi dengan metodologi yang sama persis di Bengkulu, Banten, atau Jambi — lihat §5.

---

## 2. Bukti dari Lokasi Lain: Kualitas Data Tidak Seragam

### 2.1 Bengkulu — Konsolidasi 9 Sumber, Closure Violation 12,9%

- `asrs_bengkulu_combined` adalah gabungan **9 sumber historis** berbeda (bukan 1 sensor kontinu) via `UNION BY NAME` + `COALESCE` — penamaan kolom antar-sumber tidak konsisten, perlu harmonisasi manual.
- Audit closure (`GHI = DHI + DNI·sin(elev)`) menemukan **12,9% baris mismatch**, tersebar 5,9–16,7% di semua sumber — didiagnosis sebagai ketidakpastian inheren antar-sensor independen (umum di jaringan radiasi 3-komponen), **bukan** dikoreksi, hanya diberi flag (`closure_qc_flag`).
- Anomali **DHI > GHI** ditemukan di 101.025 baris siang hari (8,2% dari total) — didiagnosis sebagai **shadow-band misalignment** (piringan peneduh sensor DHI tidak memblokir sinar langsung dengan benar saat elevasi matahari rendah pagi/sore), dikonfirmasi lewat pola bimodal 06:00–09:00 & 14:00–17:00 dengan minimum di tengah hari. 84.016 baris dikoreksi via closure equation.
- Bug polaritas terbalik pada `cloud_present` CLP (correlate dengan kt sebelum diperbaiki menunjukkan arah salah) — sudah diperbaiki, tapi menunjukkan QC awal tidak otomatis benar.
- Dua sumber CLP kualitas rendah (`clp_bengkulu_10m`, `clp_bengkulu_min_tz`) **di-drop total** — COT 100% sentinel di satu sumber.
- Sensor tekanan AWS gagal **17 bulan** — dikalibrasi ulang terhadap SYNOP.
- Gap-filling GHI historis pakai **LightGBM tervalidasi random-split** (bukan time-split) karena ditemukan drift temporal yang membuat time-split menyesatkan (R² negatif di time-split vs R²=0,71 di random-split) — pilihan metodologis yang **berbeda** dari cara split evaluasi model produksi (kronologis).

### 2.2 Banten — Cakupan CLP Rendah, Multi-Stasiun Tidak Seragam Kualitasnya

- 4 stasiun CLP (BSD, Golf Modern, TMII, UI) masing-masing **per-stasiun titik**, bukan buffer spasial seperti Kalbar/Jambi — arsitektur data yang **berbeda secara struktural** dari 2 lokasi lain.
- Kolom `is_daytime` NULL untuk 2023–2025 (hanya terisi 2022) — harus di-rederivasi dari formula elevasi matahari.
- Data 2022 **tidak representatif**: hanya 175 hari (bukan 365), menyebabkan KT tahunan anomali 2–3× lebih rendah dari 2023–2025 di 2 dari 4 stasiun (BSD, TMII).
- Kategori `clear_or_no_retrieval` mencampur dua kondisi berbeda (langit benar-benar cerah **dan** retrieval gagal karena awan terlalu tebal) — ambiguitas struktural yang sama persis dengan temuan Kalbar §1 (retrieval gagal disalahartikan sebagai "cerah").
- Korelasi CLOT vs KT **sangat lemah** (Pearson r = −0,09 s/d +0,04) — jauh lebih lemah dari yang diharapkan, kemungkinan besar karena mismatch kategori di atas + confounding musiman (hubungan CLOT-KT jelas hanya di musim kemarau Jul–Okt, delta KT +0,05 s/d +0,12; nyaris tak bermakna di musim hujan Jan–Apr, kadang berlawanan arah).
- Kualitas antar-stasiun **tidak seragam**: Golf Modern (gap 0,39%, hierarki KT paling bersih) jauh lebih baik dari AWS UI (gap 1,48%, `ws_max`/`sr_max` 22% null karena sensor baru aktif 2023, nilai sentinel −9999, radiasi negatif siang hari, spike radiasi jam 18:00 setelah matahari terbenam).
- Ini **menjelaskan** kenapa Banten satu-satunya lokasi di mana AWS meteo TIDAK redundan (`02_Feature_Engineering.md` §5) — CLP hanya mencakup ~15% baris valid (baris berawan saja, ~4.600–4.800 dari 30.267 baris hourly per stasiun), sehingga meteo (gap <2%) secara struktural mengisi celah yang jauh lebih lebar dibanding di lokasi lain.

### 2.3 Jambi — DHI/DNI 21,5% Bukan Observasi Langsung, Hanya 79,3% Baris Valid

- **21,5% nilai historis DHI dan DNI** yang dipakai sebagai fitur sebenarnya **diturunkan dari model** (Erbs model untuk DHI, closure equation untuk DNI), bukan observasi independen — informatif sebagai fitur riwayat, tapi perlu disebutkan eksplisit saat menafsirkan korelasi.
- Closure violation >15%: **1,33%** baris (1.639 dari 123.093) — dibuang dari target valid.
- Setelah *strict cleaning* (hanya `ghi_quality_flag == 'good'`, closure valid, kt di-clip ke [0, 1,3]): tersisa **79,3%** baris sebagai target prediksi valid (97.659 dari 123.093).
- AOD/aerosol missingness **58–83%** — dikeluarkan dari fitur utama karena tidak bisa diimputasi andal.
- `jambi_clp_combined` sama seperti Kalbar/Kalbar: hanya **statistik spasial agregat** (buffer 30 km, grid 11×11), bukan grid piksel mentah — batasan struktural yang sama, potensi mismatch spasial yang sama seperti Temuan #1 Kalbar, **belum diuji secara eksplisit** di Jambi.

---

## 3. Sintesis: Apa yang SUDAH Diseragamkan vs BELUM

| Aspek | Sudah diseragamkan? | Detail |
|---|---|---|
| Resep fitur (F1/F2) | ✅ Ya | `02_Feature_Engineering.md` |
| Definisi target & split | ✅ Ya | `03_Target_dan_Split.md` |
| Model & metrik evaluasi | ✅ Ya | `04_Model_dan_Training.md` |
| **Buffer/resolusi spasial CLP** | ✅ **Terdokumentasi (2026-07-25)** | Kalbar & Jambi: buffer 30km area-average; Banten: per-stasiun titik (4 stasiun terpisah); Bengkulu: ~22×22km diinferensi dari `CLOT_count` (bukan metadata eksplisit, tapi sudah didokumentasikan sebagai estimasi) — lihat `01_Dataset.md` §4.5 |
| **Metodologi gap-filling** | ❌ Tidak konsisten (belum dikerjakan — Prioritas B, butuh keputusan desain) | Bengkulu: interpolasi + LightGBM (validasi random-split); Jambi: Erbs model + closure derivation (~21,5% kolom); Banten: sebagian besar dibiarkan raw dengan flag anomali; Kalbar: belum terdokumentasi di audit yang dibaca untuk dokumen ini |
| **Threshold & penanganan closure violation** | ✅ **Distandarkan (2026-07-25)** | Satu threshold (>15% closure error relatif) dihitung identik di 4 lokasi: Jambi 1,92%, Banten 7,91%, Bengkulu 40,35%, Kalbar 42,10% — lihat `06_Perbandingan_4_Lokasi.md` §7.1. Angka Bengkulu/Kalbar jauh lebih tinggi dari laporan lama karena definisi lama tidak seragam (bukan kontradiksi data) |
| **Definisi "baris valid" untuk target** | ✅ **Distandarkan (2026-07-25)** | "Baris valid" = closure error relatif ≤15%, dihitung sama di 4 lokasi: Jambi 98,1%, Banten 92,1%, Bengkulu 59,7%, Kalbar 57,9% |
| **Penanganan nilai sentinel** (9999/-9999/dsb) | ✅ **Diaudit sistematis (2026-07-25)** | Skrip identik (`audit_data_quality_<lokasi>.py`) dijalankan di 4 lokasi: sentinel value (9999/-9999/999/-999) hampir nol di semua lokasi (<0,02%) pada tabel training final (kemungkinan besar sudah dibersihkan di tahap upstream); stuck-value run (≥6 pembacaan 10-menit identik non-nol, non-malam) juga rendah (<0,3%) di semua lokasi KECUALI Jambi DNI (2,26% — dicurigai artefak derivasi model Erbs, ~21,5% kolom DNI Jambi memang bukan observasi langsung, lihat §2.3) |
| **Audit kontradiksi antar-produk satelit** (spasial mismatch, seperti Temuan Kalbar §1–2) | ✅ **Direplikasi (2026-07-25)** | Dijalankan di 4 lokasi dengan skrip identik — lihat tabel lengkap + interpretasi di `06_Perbandingan_4_Lokasi.md` §7.1. Temuan utama: closure violation dan kontradiksi CLOT-vs-kt TIDAK berkorelasi searah dengan gap R² antar-lokasi (bertentangan dengan hipotesis geografis awal) — hanya rasio volatilitas target-vs-fitur yang konsisten di level kelompok |

**Kesimpulan langsung menjawab observasi Anda**: benar, ceiling R² per lokasi (Bengkulu 0,79–0,90, Kalbar 0,73–0,86, Banten 0,68–0,84, Jambi 0,68–0,83 — lihat `06_Perbandingan_4_Lokasi.md`) kemungkinan besar sebagian dijelaskan oleh **kesenjangan kualitas data mentah** di atas, bukan cuma perbedaan protokol training (yang sudah diseragamkan). Banten misalnya punya alasan struktural konkret (cakupan CLP rendah + heterogenitas kualitas antar-stasiun) untuk R²-nya yang lebih rendah — bukan cuma "rezim awan lebih sulit" seperti hipotesis geografis di `06_Perbandingan_4_Lokasi.md` §7.

> ✅ **Update (2026-07-25)**: audit terukur sudah dijalankan (§7.1 `06_Perbandingan_4_Lokasi.md`) — hasilnya **lebih rumit** dari kesimpulan di atas. Closure violation dan kontradiksi CLOT-vs-kt (dua metrik "kualitas data" paling intuitif) ternyata TIDAK berkorelasi searah dengan R²; hanya rasio volatilitas target-vs-fitur yang konsisten di level kelompok. Kesimpulan yang lebih akurat: gap R² kemungkinan dijelaskan oleh KOMBINASI faktor (volume data, cakupan CLP, arsitektur buffer, dan mungkin juga rezim iklim) — bukan satu metrik kualitas data tunggal yang bisa dijadikan penjelasan dominan.

---

## 4. Rekomendasi Standardisasi Data (Bertahap)

### Prioritas A — Bisa dilakukan tanpa data baru (murni harmonisasi metodologi) — ✅ SELESAI (2026-07-25)

1. ✅ **Audit konsistensi fisis ala Kalbar di 3 lokasi lain** — direplikasi dengan skrip identik (`audit_data_quality_<lokasi>.py`), hasil di `06_Perbandingan_4_Lokasi.md` §7.1. Hipotesis geografis sekarang berbasis bukti (meski buktinya menolak cerita sederhana yang diharapkan — lihat temuan di §7.1).
2. ✅ **Standardisasi threshold closure-violation dan definisi "baris valid"** — satu ambang (15%) dan satu rumus dipakai di 4 lokasi, dilaporkan di `06_Perbandingan_4_Lokasi.md` §7.1 dan `01_Dataset.md`.
3. ✅ **Audit sentinel value sistematis** — skrip identik dijalankan di 4 database, hasil hampir seragam bersih (<0,3%) kecuali Jambi DNI (2,26% stuck-run, dicurigai artefak Erbs-model derivation).
4. ✅ **Dokumentasikan buffer/resolusi spasial CLP secara eksplisit per lokasi** — ditambahkan di `01_Dataset.md` §4.5, termasuk estimasi Bengkulu (~22×22km, diinferensi dari `CLOT_count`, karena metadata buffer eksplisit tidak tersedia di skema database Bengkulu).

### Prioritas B — Butuh keputusan desain / effort tambahan

5. **Pertimbangkan menyeragamkan metodologi gap-filling** (pilih satu: interpolasi+model, atau closure-derivation) dengan validasi yang konsisten (bukan campuran random-split di satu lokasi dan time-split di lokasi lain untuk tahap yang secara konseptual sama).
6. **Kalau anggaran/waktu memungkinkan**: cari produk satelit dengan buffer lebih kecil (<10km) atau eksplorasi all-sky imager di 1 stasiun sebagai studi kasus — ini rekomendasi langsung dari audit Kalbar untuk mengurangi mismatch spasial Temuan #1, dan akan jadi kontribusi metodologis kuat untuk paper kalau sempat diuji di skala kecil.

### Prioritas C — Untuk paper (tidak butuh kerja teknis baru)

7. **Tulis eksplisit di bagian Data/Metodologi** bahwa keempat lokasi memakai infrastruktur pengumpulan data yang tidak identik (jumlah sumber, arsitektur buffer-vs-titik, kualitas sensor) — ini bagian dari kontribusi paper (benchmarking di kondisi dunia nyata yang heterogen), bukan kelemahan yang perlu disembunyikan. Kombinasikan dengan temuan Prioritas A untuk argumen: *"sebagian varians R² antar-lokasi dijelaskan oleh kualitas data terukur (Tabel X), bukan hanya perbedaan intrinsik rezim iklim."*

---

## 5. Status Audit Data per Lokasi (untuk pelacakan)

| Lokasi | Audit konsistensi fisis (ala Kalbar) | Audit closure/sentinel | Dokumentasi buffer spasial |
|---|---|---|---|
| **Kalbar** | ✅ Selesai, sangat rinci (`note_09`), direcompute ulang 2026-07-25 dengan skrip standar | ✅ Standar baru: closure 42,1%, sentinel/stuck <0,3% | ✅ 30km, grid 10×11 |
| Bengkulu | ✅ **Selesai 2026-07-25** (`audit_data_quality_bengkulu.py`) | ✅ Standar baru: closure 40,35%, sentinel/stuck <0,1% | ✅ **Diperiksa 2026-07-25**: ~22×22km (diinferensi dari `CLOT_count` maks=121), lihat `01_Dataset.md` §4.5 |
| Banten | ✅ **Selesai 2026-07-25** (`audit_data_quality_banten.py`) | ✅ Standar baru: closure 7,91% (**terbaik ke-2**), sentinel/stuck <0,3% | ✅ Per-stasiun titik (4 stasiun, bukan buffer) |
| Jambi | ✅ **Selesai 2026-07-25** (`audit_data_quality_jambi.py`) | ✅ Standar baru: closure 1,92% (**terbaik**), sentinel/stuck <0,1% kecuali DNI stuck-run 2,26% (dicurigai artefak derivasi model Erbs, lihat §2.3) | ✅ 30km, grid 11×11 |

**✅ SELESAI (2026-07-25).** Keempat lokasi sekarang punya angka konsistensi fisis yang benar-benar sebanding (metodologi identik: closure error relatif >15% pada `GHI=DHI+DNI·sin(elev)`, rasio volatilitas kt-vs-CLOT, cross-correlation lag, kontradiksi CLOT-vs-kt, dan — kalau produk analog tersedia — kontradiksi dua-produk cloud). Hasil lengkap + interpretasi (termasuk temuan mengejutkan: closure violation dan kontradiksi CLOT-vs-kt TIDAK berkorelasi searah dengan R², berlawanan dengan hipotesis geografis awal) ada di `06_Perbandingan_4_Lokasi.md` §7.1. **Catatan penting**: closure violation versi standar baru (Bengkulu 40,4%, Kalbar 42,1%) jauh lebih tinggi dari angka yang pernah dilaporkan terpisah sebelumnya (Bengkulu 12,9%, Jambi 1,33% masih cocok/dekat) — ini BUKAN kontradiksi, melainkan bukti bahwa definisi/threshold sebelumnya memang tidak seragam antar lokasi (persis masalah yang mau diperbaiki Prioritas A2 butir 2). Detail per-baris (termasuk breakdown per `fill_tier` Kalbar yang menunjukkan closure violation jauh lebih tinggi di baris hasil imputasi ML) ada di output JSON masing-masing skrip (`outputs_audit_konsistensi_fisis/audit_konsistensi_fisis_<lokasi>.json`).
