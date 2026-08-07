# Prompt Investigasi: Hipotesis SYNOP Okta sebagai Faktor Keunggulan R² Bengkulu

**Tanggal**: 2026-08-02  
**Konteks**: Paper GHI 1-jam 4 lokasi (Bengkulu, Kalbar, Jambi, Banten)

---

## Latar Belakang dan Hipotesis

Model R1 (benchmark LightGBM + CatBoost, fitur 50 kolom F1) menghasilkan R² yang berbeda signifikan antar lokasi:

| Lokasi | R² CB | R² LGBM | Data mulai |
|--------|-------|---------|-----------|
| **Bengkulu** | **0.792** | **0.789** | 2021 |
| Kalbar | 0.728 | 0.7217 | 2022 |
| Banten | 0.6818 | — | 2022 |
| Jambi | 0.693 | 0.676 | 2021 |

**Hipotesis yang perlu diinvestigasi**: Apakah ketersediaan data SYNOP cloud layer (okta CL/CM/CH) yang lebih lengkap di Bengkulu berkontribusi pada R²-nya yang lebih tinggi dibanding tiga lokasi lain?

---

## Fakta Ketersediaan Data SYNOP per Lokasi (TERVERIFIKASI — jangan diubah)

| Lokasi | Apa yang tersedia | Catatan kritis |
|--------|-------------------|----------------|
| **Bengkulu** | CL + CM + CH oktas **lengkap sepanjang rekaman** (2021–2025) | Referensi: `synop_bengkulu` 25.289 baris; `cloud_low_base_1` terisi 99.72%, `cloud_med_base_1` terisi 98.6%, `cloud_high_base_1` terisi 93.79% |
| **Kalbar** | Okta total + CM/CH oktas **hanya s/d 2024** — data 2025 tidak tersedia | Tabel SYNOP Kalbar: `synop_radiasi_jam`; missing 2025 berarti test set 2025 tidak bisa menggunakan fitur SYNOP penuh |
| **Jambi** | Okta total + **kode type L/M/H** (BUKAN okta CM/CH terpisah) | `synop_jambi_combined`; tidak ada kolom okta per lapisan terpisah — hanya total oktas dan kode type cuaca |
| **Banten** | **CL oktas saja**, dan **hanya mulai pertengahan 2024** (2021–2023 kosong) | Data SYNOP Banten sangat terbatas; cakupan efektif hanya ~1.5 tahun dari 4 tahun periode model |

**Catatan penting — Banten cross-check**:  
Validasi silang CLP-Himawari vs SYNOP di Banten hanya bisa dilakukan untuk periode **2024–2025**, bukan penuh 2022–2025. Ini berarti dampak SYNOP di Banten tidak bisa diukur pada train set (2022–2023) — hanya pada val/test (2024–2025).

---

## Apa yang SUDAH Diketahui (Jangan Ulang Riset Ini)

### Fitur F1 (50 kolom, dipakai di semua R1/R8 di keempat lokasi)
F1 adalah fitur STANDAR yang **sama persis** di keempat lokasi. F1 **TIDAK memuat** SYNOP cloud oktas sama sekali. F1 terdiri dari:
- GHI dynamics: `ghi_now`, lag ghi 10/20/30/40/50/60 menit, rolling stats
- kt dynamics: `kt_now`, lag kt, rolling kt
- CLP (Himawari): `clp_cot`, `clp_cth_m`, `clp_ctt_k`, `clp_cer`, `clp_cloud_present`
- Time cyclic: `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`, `month_sin`, `month_cos`
- Future deterministic: `solar_elev_t60`, `clearsky_t60`, `hour_sin_t60`, `hour_cos_t60`

### Arm A — AWS Meteorologi (SUDAH diuji, hasilnya redundan)
Arm A menguji AWS surface meteorology (T, RH, tekanan, angin, hujan) sebagai tambahan ke F1. Hasilnya: **redundan** di 3 dari 4 lokasi — tidak menambah R² signifikan karena CLP sudah mengkodekan kondisi atmosfer. SYNOP oktas **berbeda** dari AWS dan **belum diuji** dalam framework R8.

### Eksperimen SYNOP Kalbar yang Sudah Ada
File: `C:\Users\ariff\DuckDB_kalbar\experiment_synop_weather_hourly.py`  
Eksperimen ini menguji SYNOP pada model **hourly-averaged** (baseline R²=0.8341) — **bukan pada pipeline R1 10-menit**. Fitur yang dicoba: `cloud_cover_oktas`, `present_weather`, `past_weather_w1`, `cloud_low_type`, `cloud_layer2_type`, `visibility_km`. Ini adalah total oktas, BUKAN oktas per lapisan CL/CM/CH terpisah. Hasil eksperimen ini belum terdokumentasikan di Obsidian dan perlu ditelusuri outputnya.

### Eksperimen SYNOP Bengkulu (v10 kandidat)
File: `C:\Users\ariff\bengkulu_ghi_julius\train_ghi_1h_bengkulu_v10_accel_lean.py`  
Mengandung daftar `SYNOP_CLOUD_FEATURES` sebagai kandidat: `syn_cloud_cover_oktas_m`, `syn_cloud_low_cover_oktas`, `syn_cloud_med_cover_oktas`, per-layer height dan amount oktas, `syn_present_weather`, dsb. Namun ini adalah eksperimen v10 (model percobaan), bukan bagian dari F1 standar yang dipakai di paper.

---

## Pertanyaan Investigasi

### Pertanyaan Utama
1. **Apakah model Bengkulu yang sudah ada (R1/R8) memang menggunakan fitur SYNOP?**  
   Periksa SQL query di `train_ghi_1h_bengkulu_v10_accel_lean.py` — apakah SYNOP features ada di feature set yang dipakai untuk menghasilkan R²=0.792, atau hanya ada di eksperimen v10 yang tidak dilaporkan di paper?

2. **Berapa R² model Kalbar jika SYNOP 2021–2024 ditambahkan sebagai fitur tambahan ke F1?**  
   Kalbar punya SYNOP s/d 2024. Train (2022–2023) + Val (2024) punya SYNOP; Test (2025) tidak. Ini berarti kalau SYNOP dipakai sebagai fitur, model perlu strategi fallback untuk test set 2025 (imputation atau switch ke F1-only mode di 2025). Apakah keuntungan di val set cukup untuk membenarkan kompleksitas ini?

3. **Apakah CL oktas Banten (hanya 2024–pertengahan 2025) cukup untuk dievaluasi dampaknya?**  
   Dengan hanya ~1.5 tahun data, dan ketersediaan SYNOP Banten yang sangat terbatas, apakah investigasi ini feasible secara statistik untuk Banten?

4. **Berapa fraksi jam siang di setiap lokasi yang memiliki SYNOP tercatat?**  
   Jam operasional SYNOP biasanya 07:00–19:00 WIB (setiap 3 jam = 5 pengamatan/hari). Ini tidak sinkron sempurna dengan resolusi 10-menit pipeline. Berapa coverage efektif setelah JOIN dengan training data?

### Pertanyaan Sekunder
5. Apakah okta CL/CM/CH memberikan informasi TAMBAHAN di atas CLP (Himawari COT/CTH/CTT/CER), atau keduanya mengukur hal yang sama dari perspektif berbeda (satelit vs pengamat darat)?

6. Jika SYNOP memang berkontribusi ke R² Bengkulu, apakah ini berarti keunggulan Bengkulu sebagian bersifat "data artifact" (lebih banyak fitur tersedia) daripada murni kondisi iklim/geografis yang lebih mudah diprediksi?

---

## Rencana Investigasi per Lokasi

### LANGKAH 0 — Konfirmasi Baseline (WAJIB pertama)
Periksa skrip R1 standar Bengkulu (`train_ghi_1h_bengkulu_R1_benchmark.py` atau ekuivalennya di `bengkulu_ghi_julius/`) dan konfirmasi: apakah 50 fitur F1 yang menghasilkan R²=0.792 **SUDAH mengandung** SYNOP oktas atau tidak? Kalau sudah, hipotesis ini langsung terkonfirmasi sebagian. Kalau belum, lanjutkan ke Langkah 1.

### LANGKAH 1 — Bengkulu: Tambah SYNOP ke F1 (Arm D)
- Lokasi skrip: `C:\Users\ariff\bengkulu_ghi_julius\`
- DB: `C:\Users\ariff\DuckDB_bengkulu\bengkulu.duckdb`
- Tabel SYNOP: `synop_bengkulu` (JOIN via `ts_wib`, resolusi 1-jam, perlu forward-fill ke 10-menit)
- Fitur SYNOP kandidat (berdasarkan ketersediaan, lihat `synop_null_audit.csv`):
  - `cloud_low_cover_oktas` (~99.72% filled → sangat reliable)
  - `cloud_med_cover_oktas` (~98.6% filled → sangat reliable)
  - `cloud_high_cover_oktas` (~93.79% filled → reliable)
  - `cloud_layer_1_amt_oktas_ns` (35.57% → partial)
  - Total oktas (jika tersedia sebagai kolom terpisah)
- Join strategy: JOIN ke window satu jam sebelumnya, forward-fill untuk 10-menit resolution
- **Target**: Apakah F1 + SYNOP CL/CM/CH > 0.792?

### LANGKAH 2 — Kalbar: SYNOP terbatas 2022–2024
- DB: `C:\Users\ariff\DuckDB_kalbar\kalbar_local.db`
- Tabel SYNOP: `synop_radiasi_jam` (dari eksperimen hourly yang sudah ada)
- Fitur tersedia: `cloud_cover_oktas` (total), `cloud_low_type`, `cloud_layer2_type` — BUKAN okta CM/CH terpisah
- **Caveat kritis**: SYNOP 2025 tidak tersedia → test set 2025 harus menggunakan imputed value (misalnya mean dari 2022–2024) atau diuji terpisah sebagai "tanpa SYNOP"
- Strategi yang direkomendasikan: train dengan SYNOP (2022–2023), eval val 2024 dengan SYNOP, eval test 2025 dengan imputation → catat selisih R² val (dgn SYNOP) vs test (tanpa SYNOP). Kalau gap besar, SYNOP tidak bisa dipakai di production untuk Kalbar.
- **Target**: Berapa R² val 2024 dengan SYNOP total oktas? Bandingkan vs baseline 0.728.

### LANGKAH 3 — Jambi: Hanya okta total + type code
- DB: `C:\Users\ariff\DuckDB_jambi\jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb`
- Tabel SYNOP: `synop_jambi_combined`
- Fitur tersedia: total cloud oktas + kode type L/M/H (bukan okta per lapisan terpisah)
- Jambi punya iklim berbeda (lebih pegunungan, konveksi orografis Gunung Kerinci)
- Periksa coverage SYNOP: berapa persen baris training yang dapat JOIN ke SYNOP?
- **Target**: Apakah total oktas + type code saja sudah memberikan peningkatan R² ≥ 0.003 di atas 0.693?

### LANGKAH 4 — Banten: SYNOP sangat terbatas (skip jika infeasible)
- DB: `C:\Users\ariff\Duckdb_Banten\banten.duckdb`
- SYNOP: CL oktas saja, tersedia hanya mulai pertengahan 2024
- Coverage efektif: hanya Val (2024) dan Test (2025) — Train (2022–2023) tidak punya SYNOP sama sekali
- **Caveat**: cross-check CLP-Himawari vs SYNOP di Banten hanya bisa untuk 2024–2025. Evaluasi Banten SYNOP berbeda dari 3 lokasi lain karena arsitektur CLP Banten berbeda (per-stasiun titik, 4 stasiun, bukan area-average).
- **Rekomendasi**: Lakukan investigasi Banten hanya sebagai ablation study pada Val+Test — bukan sebagai training feature, mengingat Train 2022–2023 tidak punya data.

---

## Format Output yang Diharapkan

Untuk setiap lokasi, laporkan:
1. **Coverage SYNOP** dalam training data: berapa % baris dapat JOIN ke SYNOP non-null
2. **R² baseline** (F1 only): konfirmasi sama dengan angka referensi di atas
3. **R² dengan SYNOP** (F1 + oktas tersedia per lokasi)
4. **Delta R²** = R²(F1+SYNOP) − R²(F1-only)
5. **Kesimpulan feasibility**: apakah SYNOP bisa dipakai di production (tersedia saat inference)?

### Tabel Ringkasan Target

| Lokasi | SYNOP tersedia | Coverage di train | R² baseline | R² + SYNOP | Delta R² | Feasible di production? |
|--------|----------------|-------------------|-------------|-----------|---------|------------------------|
| Bengkulu | CL/CM/CH lengkap | ~99%? | 0.792 | ? | ? | Ya (lengkap) |
| Kalbar | Total+CM/CH s/d 2024 | ~?% | 0.728 | ? (val only) | ? | Tidak untuk test 2025 |
| Jambi | Total+type L/M/H | ~?% | 0.693 | ? | ? | Ya jika coverage cukup |
| Banten | CL saja, ab mid-2024 | ~0% di train | 0.6818 | ? (val/test) | ? | Tidak untuk train |

---

## Implikasi untuk Paper

Jika Delta R² Bengkulu ≥ 0.003 dan lokasi lain Delta R² ≈ 0:
→ **Temuan penting**: Bengkulu R² tinggi sebagian karena kelengkapan data SYNOP. Ini perlu disebutkan di paper sebagai konfaunding faktor, bukan hanya kondisi iklim/geografis.

Jika Delta R² semua lokasi kecil (< 0.003):
→ SYNOP tidak berkontribusi signifikan ke R² dalam framework R1/F1. Keunggulan Bengkulu murni kondisi iklim (konsistensi radiasi, konveksi lokal lebih terprediksi). Hasil negatif ini juga penting — memvalidasi bahwa F1 (berbasis CLP satelit) sudah cukup mengkode informasi awan.

Jika Delta R² Bengkulu < 0.003 bahkan dengan SYNOP lengkap:
→ Perlu cek lebih dalam di level ablation: matikan CLP dan nyalakan SYNOP — apakah SYNOP bisa *menggantikan* CLP? Ini relevan untuk lokasi yang CLP-nya bermasalah.

---

## Constraints Teknis

- Python executable: `& "C:\Program Files\Python39\python.exe"` (Python 3.14 di `C:\Python314\` tidak punya ML packages)
- MOTHERDUCK_TOKEN: JANGAN echo/print token — hanya `bool(os.environ.get('MOTHERDUCK_TOKEN'))` yang boleh
- Join SYNOP ke pipeline: SYNOP native per-jam (atau per 3-jam) → perlu forward-fill atau asof join ke resolusi 10-menit pipeline
- Prioritas: mulai dari Bengkulu (paling lengkap datanya) untuk mengkonfirmasi baseline sebelum ke lokasi lain
