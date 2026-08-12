# Feature Engineering — Resep Standar (F1/F2)

Resep fitur ini **identik untuk keempat lokasi** — bagian yang berbeda hanya pemetaan nama kolom mentah (lihat `01_Dataset.md`). Berbasis resep yang sudah divalidasi di Bengkulu (v8 pruned, 40 fitur → v10 accel-lean, 43 fitur) dan diharmoniskan ke 50 fitur (F1) untuk R1/R8 di semua lokasi.

---

## 1. F1 — 50 Fitur Lean (baseline wajib)

```python
F1_FEATURES = [
    # GHI history (16 fitur)
    "ghi_now", "ghi_lag_10m", "ghi_lag_20m", "ghi_lag_30m", "ghi_lag_60m",
    "ghi_lag_120m", "ghi_lag_180m",
    "ghi_roll_30m_mean", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_std",
    "ghi_delta_10m", "ghi_delta_60m", "accel_ghi_20m",

    # Clearness index kt (9 fitur)
    "kt_now", "kt_lag_10m", "kt_lag_20m", "kt_lag_30m", "kt_lag_60m",
    "kt_roll30m_mean", "kt_roll30m_std", "kt_roll60m_mean", "accel_kt_20m",

    # Awan satelit CLP (15 fitur)
    "clp_cot", "clp_cot_lag_10m", "clp_cot_lag_20m", "clp_cot_lag_30m", "clp_cot_lag_60m",
    "clp_cot_delta_10m", "clp_cot_delta_30m", "clp_cot_delta_60m", "clp_cot_delta_180m",
    "clp_cot_roll_180m_mean", "accel_clp_cot_20m",
    "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",

    # Waktu siklik (6 fitur)
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",

    # Future deterministic (4 fitur)
    "ghi_cs_t60", "elev_sin_t60", "smart_persist", "smart_persist_avg",
]
```

`accel_*` = turunan kedua (percepatan) dari sinyal GHI/kt/CLP-COT dalam jendela 20 menit — teknik yang diadopsi dari Banten dan terbukti generalisasi ke Bengkulu (+0,0014 R² pada target titik).

## 2. F2 — F1 + AWS Meteo (untuk uji redundansi, Arm A)

```python
F2_FEATURES = F1_FEATURES + [
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "pressure_hpa"
]
```

**Tidak dipakai untuk produksi secara default** — F2 hanya untuk menguji apakah meteo permukaan menambah nilai (Arm A). Hasilnya **berbeda per lokasi**, lihat §5.

---

## 3. Filter Kualitas & Anti-Leakage

Checklist wajib sebelum training, identik di semua lokasi:

- [ ] `sun_altitude > 5°` di titik anchor **dan** di t+60 (buang data dekat matahari terbit/terbenam)
- [ ] `GHI` dalam rentang 0–1400 W/m² (buang outlier sensor)
- [ ] `anchor_valid = true`
- [ ] Gap kontinuitas time-series ±30 detik (buang baris yang timestamp-nya melompat)
- [ ] Future regressors **hanya** yang deterministik (`I_clr(t+k)`, `cos θ(t+k)`) atau proyeksi dari data lampau (`smart_persist`, EMA) — **tidak ada** nilai GHI/CI aktual masa depan di kolom fitur manapun
- [ ] Normalisasi/scaling (jika dipakai model yang butuh): fit di train saja, terapkan ke val/test
- [ ] Jika ada dekomposisi sinyal (wavelet/VMD/EMD): **wajib setelah split**, bukan sebelum (lihat §4)

---

## 4. Teknik yang SUDAH Terbukti Gagal — Jangan Diulang

Divalidasi empiris di 4 proyek (bukan asumsi dari literatur). Semua entri berikut sudah diuji dan **ditolak**:

| # | Teknik | Hasil | Bukti (lokasi) |
|---|---|---|---|
| 1 | **PCA** untuk reduksi dimensi | Merugikan −0,02 s/d −0,04 R² | Banten (0,872→0,832), Bengkulu (0,780→0,760) |
| 2 | **Model per-regime** (cuaca cerah/berawan/hujan terpisah) | Lebih buruk dari model global, −0,04 R² | Kalbar (0,750→0,710), Bengkulu (0,780→0,740) |
| 3 | **LSTM/CNN-LSTM standalone** (bukan bagian ensemble) | Kalah dari gradient boosting, −0,05 s/d −0,08 R² | Jambi, Kalbar, Bengkulu (5 eksperimen independen) |
| 4 | **Ensemble lintas-keluarga dengan anggota lemah** (mis. MLP jauh di bawah tree) | Merugikan jika gap R² > 0,02 antar anggota | Bengkulu (0,780→0,750, MLP menarik turun) |
| 5 | **Smart-persistence sebagai fitur eksplisit** (`kt_now × clearsky_target`) | Redundan dengan `kt_now`+`clear_sky_ghi_now` yang sudah ada, R² flat/turun | Bengkulu (importance rank #2 tapi kontribusi bersih nol) |
| 6 | **Fitur musim eksplisit** (`is_kemarau`, label diskret) | Redundan dengan encoding siklik kontinu | Jambi, Bengkulu (Δ ±0,01, siklik lebih baik) |
| 7 | **VMD/EMD/wavelet SEBELUM split temporal** | Look-ahead bias, R² inflated | Kalbar (0,920 inflated → 0,750 genuine, Δ −0,17) |
| 8 | **Data stasiun tetangga untuk adveksi spasial** | Redundan di tropis konvektif (awan tumbuh/buyar lokal) | Banten (Δ −0,002, negligible) |
| 9 | **Fitur radiasi mentah DHI/DNI digabung dengan `ghi_now`** (F_super) | Collinearity collapse — `ghi_now` kalah rank, R² turun −0,03 | Bengkulu (Arm C investigasi mendalam, root cause: identitas linier GHI=DHI+DNI·cosθ) |

**Aturan turunan dari #9**: kalau ingin melakukan feature pruning berbasis importance sweep, JANGAN mulai dari superset yang mengandung DHI/DNI mentah bersama `ghi_now`. Gunakan **forward selection dari F1** (tambah fitur satu per satu) atau greedy backward elimination dari F1, bukan top-K dari pool superset campuran.

---

## 5. Anomali Penting: Meteo Permukaan TIDAK Selalu Redundan

Temuan umum (Kalbar, Jambi, Bengkulu, 4 wilayah konsisten): AWS meteo (`T, RH, WS, tekanan, hujan`) redundan begitu CLP satelit tersedia (ΔR² < 0,01).

**Banten adalah pengecualian**: ΔR² = **+0,0128** (titik) dan **+0,008** (rata-rata) — meteo justru **membantu** di Banten. Penyebab: cakupan CLP satelit Banten hanya ~50% (lebih rendah dari lokasi lain), sehingga data AWS (100% lengkap) mengisi celah yang ditinggalkan CLP. Analisis isolasi murni (bukan artefak kumulatif) mengonfirmasi pendorong utamanya adalah **wind speed** (≈90% dari efek +0,0116 R²) — kemungkinan proksi laju adveksi awan di rezim konvektif Banten.

**Implikasi**: F2 (dengan meteo) sebaiknya tetap dilaporkan sebagai sensitivity check per lokasi (Arm A), bukan diasumsikan hasilnya seragam. Untuk produksi, default tetap F1 (lean) kecuali di lokasi dengan cakupan CLP rendah seperti Banten, di mana F2 bisa dipertimbangkan.

---

## 6. Audit Kesamaan Skema — Tabel Fisik Beda, F1 Genuine (dicek langsung ke 4 skrip R1, 2026-07-24)

Pertanyaan yang wajar muncul setelah Jambi diseragamkan ke skema Bengkulu: apakah Kalbar dan Banten juga memakai skema tabel yang sama? **Jawabannya tidak — dan memang tidak perlu.** Yang harus identik untuk keadilan benchmarking bukan *skema tabel fisik* (nama tabel, nama kolom mentah), tapi **fitur F1 yang benar-benar masuk ke model** (nama, definisi, dan cara hitungnya). Dicek langsung ke `FEATURES` list dan `add_features()`/`build_sql()` di keempat skrip R1:

| Lokasi | Tabel sumber | Nama 50 fitur F1 | Cara hitung lag/rolling |
|---|---|---|---|
| Bengkulu | `ghi_forecast_1h_train_3h_rollback_2021_2025` (102 kolom, precomputed) | ✅ Identik | Genuine — SQL window function (LAG/AVG/STDDEV OVER) di level pembuatan tabel |
| Jambi (v2) | `ghi_forecast_1h_train_3h_rollback_2021_2025` (meniru skema Bengkulu 1:1) | ✅ Identik | Genuine — sama persis, SQL window function |
| Banten | `solar_features_base` (tabel minimal, ~7 kolom mentah) | ✅ Identik | Genuine — **semua** lag/rolling/delta dihitung di `build_sql()` skrip R1 lewat SQL window function (LAG/LEAD/AVG/STDDEV_SAMP OVER), bukan dari kolom precomputed |
| Kalbar | `training_ghi_1h_direct` (66 kolom, sebagian precomputed) | ✅ Identik (nama) | ⚠️ **Satu shortcut**: `clp_cot_lag_20m` dihitung via interpolasi linear `clot_lag10m×0,67 + clp_cot×0,33` (`train_ghi_1h_kalbar_R1_benchmark.py` baris 146) — **bukan** lag genap 20 menit sungguhan seperti 3 lokasi lain, karena tabel sumber Kalbar hanya menyimpan CLP lag 10m/30m native, tidak ada 20m. Fitur lain (termasuk `ghi_lag_120m`/`180m` via `.shift(12)`/`.shift(18)`) genuine. |

**Temuan kedua — definisi "baris valid" (continuity filter) juga tidak seragam ketat**, konsisten dengan yang sudah dicatat di `08_Standardisasi_Data_Mentah.md` §3, sekarang diverifikasi persis:
- Bengkulu & Jambi (v2): `has_continuous_3h_history=1` — **strict**, mensyaratkan 18 langkah 10-menit berturutan penuh ke belakang.
- Banten: `ghi_lag_180m IS NOT NULL` — **longgar**, cuma butuh satu nilai lag terjauh ada, tidak mengecek semua langkah di antaranya berturutan.
- Kalbar: `anchor_valid` (`preprocess_ghi_1h_dataset.py` baris 148–155) — mengecek `sun_altitude>5°` di anchor & t+60, gap tunggal T→T+60 tepat 60 menit, target tidak null, dan bukan tier ML-imputed — **tidak** mengecek kontinuitas penuh 3 jam ke belakang seperti Bengkulu/Jambi.

**Dampak untuk paper**: dampaknya kecil (satu dari 50 fitur di satu lokasi untuk temuan pertama) tapi nyata — sebaiknya disebutkan eksplisit di bagian keterbatasan/Metodologi, bersama poin longgarnya filter kontinuitas Banten/Kalbar vs Bengkulu/Jambi (kandidat penjelas kecil untuk sebagian gap R², di luar hipotesis rezim iklim di `06_Perbandingan_4_Lokasi.md` §7). Perbaikan yang disarankan (opsional, Prioritas B — bukan blocker karena dampaknya kecil): (1) ganti `clp_cot_lag_20m` Kalbar dengan lag genuine (butuh menambah kolom CLP lag-20m ke `training_ghi_1h_direct` atau hitung ulang dari sumber CLP mentah), (2) samakan continuity filter Kalbar & Banten ke standar strict Bengkulu/Jambi (18-langkah), lalu jalankan ulang R1 keduanya untuk melihat apakah R² berubah material.
