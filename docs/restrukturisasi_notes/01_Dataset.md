# Dataset — Sumber Data dan Struktur per Lokasi

Mengikuti protokol standar di `00_Ringkasan_dan_Protokol_Standar.md`. Bagian ini mendokumentasikan **dari mana data berasal** dan **bagaimana skema tiap lokasi dipetakan** ke satu skema fitur standar (F1/F2), supaya training bisa dijalankan dengan kode yang (hampir) identik di keempat lokasi.

---

## 1. Bengkulu (rujukan standar)

**Database**: `C:\Users\ariff\DuckDB_bengkulu\bengkulu.duckdb` (lokal, sinkron dari MotherDuck `md:bengkulu` via `sync_motherduck_to_local.py`), fallback ke MotherDuck `bengkulu.bengkulu_sch`.

**Tabel sumber mentah**:
| Tabel | Baris | Isi |
|---|---|---|
| `asrs_bengkulu_combined` | 8.375.946 | Radiasi (GHI/DNI/DHI), gabungan 9 sumber historis, sudah QC + gap-filled |
| `aws_bengkulu` | 279.665 | Meteorologi permukaan (suhu, RH, angin, tekanan, hujan), 100% lengkap |
| `clp_bengkulu_combined` | 143.785 | Sifat awan satelit (Himawari CLP: COT, CTH, CTT, CER) |
| `synop_bengkulu` | 25.289 | Kode SYNOP, cloud layer/oktas, native hourly 07:00–19:00 WIB |

**Tabel training (harmonised R1/R8)**: `bengkulu_master_10min_quality_final` (view hasil JOIN keempat tabel di atas + `ghi_forecast_1h_train_3h_rollback_2021_2025`).

**Folder kerja & skrip**:
- `C:\Users\ariff\bengkulu_ghi_julius\` — skrip R1/R8 harmonis + seluruh eksplorasi v1–v11 (lihat `05_Hasil_Referensi_Bengkulu.md`)
- `C:\Users\ariff\bengkulu_ghi_forecast\` — pipeline awal (Fase 1, sebelum harmonisasi), model produksi lama `model_production_bengkulu_ghi_1h.pkl`

**Periode & split (R1)**: 2021-12 s.d. 2025-12 | Train 30.087 | Val 5.934 | Test 22.711 | Total 58.732

---

## 2. Kalbar

**Database**: `C:\Users\ariff\DuckDB_kalbar\kalbar_local.db` (lokal DuckDB, sinkron dari MotherDuck).
**Tabel training**: `training_ghi_1h_direct` (210.384 baris setelah sync 2026-07-17).
**Periode & split (R1)**: 2022-01 s.d. 2025-12 (Kalbar **tidak** memakai 2021 — beda dari lokasi lain) | Train 39.759 | Val 20.706 | Test 21.386 | Total 81.851.

**Kolom mentah → nama standar** (fungsi `add_features`):
```
ghi_final              → ghi_now
kt                      → kt_now
CLOT_mean               → clp_cot
CLTH_mean               → clp_cth_m
CLTT_mean               → clp_ctt_k
CLER_23_mean            → clp_cer
clp_cloud_present_int   → clp_cloud_present
```
Fitur turunan (lag/rolling) dihitung on-the-fly dari kolom-kolom ini.

---

## 3. Banten

**Database**: `C:\Users\ariff\Duckdb_Banten\banten.duckdb`.
**Tabel training**: `solar_features_base` (atau padanan).
**Kolom waktu**: `ts_wib`. **Target**: `ghi_point_t60`, `ghi_avg_t10_t60` (sudah dalam skema standar sejak skrip R1 dibuat).
**Periode & split (R1)**: 2022-01 s.d. 2025-12 (tanpa 2021, sama seperti Kalbar) | Train 45.244 | Val 22.685 | Test 22.559 | Total 90.488.

**Kolom mentah → nama standar**:
```
ghi                        → ghi_now
kt_point (diturunkan)      → ghi / clearsky
cloud_optical_thickness    → clp_cot
cloud_top_height           → clp_cth_m
cloud_top_temp             → clp_ctt_k
cloud_eff_radius           → clp_cer
```
**Catatan khusus**: koordinat referensi Banten −6.26147/106.7509. Vault Obsidian Banten (`04_Eksperimen/Banten/Catatan/`) memakai plugin sync `remotely-save` yang pernah **menghapus** file catatan sesi baru (catatan 18–22 & skrip .py sempat ter-quarantine) — dokumentasi definitif untuk Banten disimpan di workspace (`Duckdb_Banten/*.md`), bukan vault, sampai masalah sync ini diperbaiki.

---

## 4. Jambi

**Database**: `C:\Users\ariff\DuckDB_jambi\jambi.duckdb` (+ sebagian data via MotherDuck/parquet).
**Tabel training (v1, usang)**: `dfm_with_clp_stats` — dipangkas ke jam siang saja, lihat peringatan di bawah.
**Tabel training (v2, dipakai sekarang)**: `ghi_forecast_1h_train_3h_rollback_2021_2025`, dibangun ulang 2026-07-24 di file terpisah `jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb` (skema `jambi_sch`), mengikuti skema 102-kolom Bengkulu 1:1. Skrip: `DuckDB_jambi/build_ghi_forecast_1h_rollback_jambi.py`.
**Kolom waktu**: `ts_wib`. **Target**: `ghi_point_t60`, `ghi_avg_t10_t60`.
**Periode & split (R1, v2)**: 2021-01 s.d. 2025-12 | Train 52.108 | Val 18.172 | Test 21.129 (target titik) | Total baris siap-model 96.576 — **sebanding dengan 3 lokasi lain**, bukan lagi dataset terkecil.

> ✅ **Sudah diperbaiki (2026-07-24).** Root cause: skrip R1 Jambi v1 membaca dari sumber yang sudah dipangkas ke jam siang saja (07:00–17:40), lalu menerapkan filter kontinuitas "3 jam riwayat berturutan" yang di-porting dari Bengkulu tanpa penyesuaian — akibatnya ~3 jam pertama setiap hari otomatis gagal syarat kontinuitas. Tabel v2 dibangun dari sumber 24-jam-penuh (`asrs_jambi_menit_rev`, `jambi_clp_combined`, `meteo_obs_10min`, `synop_jambi_combined`), meniru skema Bengkulu persis. Hasil: baris siap-model naik 22.314 → 96.576 (+333%), R1 dijalankan ulang (`train_ghi_1h_jambi_R1_benchmark_v2.py`) dan R² titik naik 0,676 → 0,693. Detail lengkap di `09_Audit_Volume_Data_Jambi.md`. **Catatan**: tabel v2 baru ada di file `.duckdb` terpisah — belum digabung ke `jambi.duckdb` utama (perlu dilakukan manual oleh pengguna, lihat SQL di `09_Audit_Volume_Data_Jambi.md` §5).

**Kolom mentah → nama standar**: sudah relatif standar (`ghi_now`, `kt_now`, `clp_cot` langsung terpakai tanpa rename besar).

---

## 4.5 Buffer/Resolusi Spasial CLP per Lokasi (Prioritas A2, 2026-07-25)

Variabel struktural yang terbukti mempengaruhi ceiling R² (Temuan #1 audit Kalbar, `08_Standardisasi_Data_Mentah.md` §1: mismatch antara radiasi titik-tunggal pyranometer vs cloud-product area-average) — didokumentasikan eksplisit di sini untuk keempat lokasi supaya tidak lagi jadi "belum terdokumentasi" seperti sebelumnya.

| Lokasi | Arsitektur | Buffer/Grid | Sumber satelit | Catatan |
|---|---|---|---|---|
| Kalbar | Area-average | 30 km, grid 10×11 (110 piksel) | Himawari L2CLP, `clp_pontianak_20km` | Metadata buffer eksplisit di kolom `buffer_km`/`grid_size` tabel |
| Jambi | Area-average | 30 km, grid 11×11 (121 piksel) | Himawari L2CLP, `jambi_clp_combined` | Metadata buffer eksplisit (dikutip dari `08_Standardisasi_Data_Mentah.md` §2.3) |
| **Bengkulu** | Area-average (diinferensi) | **~22×22 km, grid ≤11×11 (maks 121 piksel, `CLOT_count` maksimum = 121)** | Himawari L2CLP full-disk (`NC_H09_*_FLDK.02401_02401.nc`), diagregasi ke `clp_bengkulu_quality_final` | **Baru diperiksa 2026-07-25**: tabel Bengkulu TIDAK menyimpan kolom `buffer_km`/`grid_size` eksplisit seperti Kalbar/Jambi — ukuran buffer diinferensi dari `CLOT_count` maksimum (121 = 11×11), dikombinasikan dengan resolusi native Himawari L2CLP (~2 km/piksel) → estimasi ~22×22 km. Ini **estimasi**, bukan angka yang eksplisit didokumentasikan di skrip build asli (skrip build CLP Bengkulu tidak ada di folder kerja sandbox ini). Arsitekturnya tetap **area-average**, sebanding dengan Kalbar/Jambi, BUKAN per-stasiun titik seperti Banten. |
| Banten | Per-stasiun titik | Tidak ada buffer (titik tunggal per stasiun) | 4 stasiun CLP terpisah (BSD Serpong, Golf Modern, TMII, UI) | Arsitektur **berbeda secara struktural** dari 3 lokasi lain — konsisten dengan temuan `08_Standardisasi_Data_Mentah.md` §2.2 |

**Implikasi**: tiga dari empat lokasi (Bengkulu, Kalbar, Jambi) memakai arsitektur area-average yang sebanding (~22–30 km), jadi mismatch spasial Temuan #1 Kalbar kemungkinan berlaku serupa di ketiganya. Banten satu-satunya dengan arsitektur titik — ini bagian dari penjelasan kenapa Banten satu-satunya lokasi di mana AWS meteo tidak redundan terhadap CLP (`02_Feature_Engineering.md` §5) dan kenapa korelasi CLOT-KT di Banten jauh lebih lemah (`08_Standardisasi_Data_Mentah.md` §2.2).

---

## 5. Ringkasan Perbandingan Cepat

| Lokasi | DB Path | Baris Total | Test 2025 | Mulai Data |
|---|---|---|---|---|
| **Bengkulu** | `DuckDB_bengkulu/bengkulu.duckdb` | 58.732 | 22.711 | 2021 |
| Kalbar | `DuckDB_kalbar/kalbar_local.db` | 81.851 | 21.386 | 2022 |
| Banten | `Duckdb_Banten/banten.duckdb` | 90.488 | 22.559 | 2022 |
| Jambi (v2, 24-jam-penuh) | `DuckDB_jambi/jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb` | 96.576 | 21.129 | 2021 |
| ~~Jambi (v1, bug pipeline)~~ | ~~`DuckDB_jambi/jambi.duckdb`~~ | ~~22.314~~ | ~~9.511~~ | ~~2021~~ |

**Perhatian untuk paper**: Kalbar dan Banten memakai rentang data 2022–2025 (tanpa 2021), sementara Bengkulu dan Jambi memakai 2021–2025. Ini perlu disebutkan eksplisit sebagai batasan harmonisasi (bukan cacat, tapi ketersediaan data historis yang berbeda per lokasi) — lihat `07_Status_dan_Rencana_Selanjutnya.md` §keterbatasan.
