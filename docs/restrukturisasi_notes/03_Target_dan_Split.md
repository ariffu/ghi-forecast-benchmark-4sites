# Definisi Target dan Strategi Split

## 1. Target Utama: Point Forecast per Horizon (10/20/30/40/50/60 menit, dilaporkan TERPISAH)

> **Update metodologi (2026-07-24), koreksi dari pembimbing.** Versi sebelumnya dokumen ini menjadikan `avg_t10_t60` (rata-rata GHI t+10..t+60) sebagai target berdampingan yang setara dengan `point_t60`, dengan alasan "keduanya forecast genuine, tidak ada leakage". Itu benar secara teknis (tidak ada informasi masa depan yang bocor ke fitur), **tapi keliru secara praktik pelaporan untuk paper benchmarking** — lihat §1.3. Protokol yang benar: **laporkan akurasi titik di tiap horizon (10, 20, 30, 40, 50, 60 menit) secara terpisah** sebagai hasil utama. `avg_t10_t60` didemosikan ke analisis tambahan (§1.4), bukan dihapus.

### 1.1 Definisi
Untuk tiap horizon $h \in \{10,20,30,40,50,60\}$ menit, target adalah nilai GHI instan pada $t+h$ (`ghi_lead_{h}m`, dihitung via `LEAD(ghi_now, h/10)` di SQL). Fitur input **selalu** memakai data hingga waktu anchor $t$ saja (resolusi 10 menit, lag/rolling/delta/akselerasi seperti biasa) — model dilatih ulang per horizon (6 model per lokasi per algoritma), bukan satu model yang dipakai untuk semua horizon.

### 1.2 Hasil pilot — Jambi (LightGBM residual, test 2025, `outputs_R1_jambi_v2_medium/per_horizon_results.csv`)

| Horizon | n test | R² Smart-Persistence | R² LightGBM | MAE (W/m²) | Skill vs SP |
|---|---|---|---|---|---|
| 10m | 22.241 | 0,7964 | **0,8469** | 66,2 | 0,1330 |
| 20m | 22.184 | 0,6805 | **0,7863** | 84,2 | 0,1821 |
| 30m | 22.014 | 0,6110 | **0,7580** | 92,3 | 0,2113 |
| 40m | 21.766 | 0,5424 | **0,7342** | 98,8 | 0,2378 |
| 50m | 21.454 | 0,4833 | **0,7121** | 104,1 | 0,2536 |
| 60m | 21.129 | 0,4266 | **0,6924** | 109,0 | 0,2676 |

Pola yang persis seperti yang diminta pembimbing terlihat jelas: R² LightGBM menurun mulus 0,847 → 0,692 seiring horizon bertambah (peluruhan akurasi, bukan angka tunggal yang menyembunyikannya), sementara **skill relatif terhadap smart-persistence justru naik** (0,133 → 0,268) — model semakin unggul dibanding baseline fisika sederhana seiring horizon memanjang, karena SP meluruh lebih cepat (0,796 → 0,427) daripada model. Kedua pola ini adalah insight yang hilang total kalau horizon-horizon ini dirata-ratakan jadi satu angka.

**Status per lokasi**: ✅ **Keempat lokasi selesai** (2026-07-24) — Jambi (tabel di atas), Bengkulu, Kalbar, Banten. Tabel lengkap perbandingan 4-lokasi ada di `06_Perbandingan_4_Lokasi.md` §1 (R² LightGBM, R² SP, dan skill vs SP per horizon, plus tiga temuan lintas-lokasi yang baru terlihat karena horizon dipisah).

### 1.3 Kenapa merata-ratakan lintas horizon bermasalah (bukan soal leakage)

Tiga alasan, bukan cuma preferensi pembimbing:

1. **Artefak variansi, bukan skill genuine.** Merata-ratakan 6 titik 10-menitan menghasilkan target dengan varians jauh lebih kecil daripada titik tunggal t+60 — noise transient awan saling meniadakan, tren yang mudah ditebak (posisi matahari) tetap utuh. Karena R² = 1 − MSE/Var(target), target yang variansnya lebih kecil otomatis mendorong R² naik, terlepas dari skill model. Bukti langsung: baseline smart-persistence — **tanpa machine learning sama sekali** — R²-nya ikut melompat murni karena definisi target (Jambi: 0,427 di t+60 → 0,650 di avg_t10_t60; pola sama di 3 lokasi lain, lihat §1.4 tabel SP). Kalau baseline "bodoh" mendapat lompatan sebesar itu tanpa belajar apa pun, sebagian besar kenaikan R² avg bukan bukti model lebih baik meramal.
2. **Menghapus kurva yang justru jadi inti kontribusi paper benchmarking.** Tujuan benchmarking horizon forecasting adalah menunjukkan *di menit keberapa* model kehabisan tenaga prediksi — itu yang informatif secara operasional (kontrol real-time vs dispatch) dan saintifik (batas prediktabilitas fisis). Satu angka rata-rata menyembunyikan persis kurva ini.
3. **Preseden risiko interpretasi.** Insiden Kalbar sebelumnya (nowcasting 10-menit dirata-rata lalu diklaim forecast 1 jam) memang berbeda secara teknis dari `avg_t10_t60` (yang satu ini genuinely tidak bocor), tapi bentuknya cukup mirip (perata-rataan lintas horizon) untuk berisiko disalahpahami pembaca/reviewer sebagai hal yang sama, kecuali dijelaskan dengan sangat eksplisit.

### 1.4 `avg_t10_t60` — didemosikan ke analisis tambahan, bukan dihapus

Tetap berguna untuk use-case dispatch grid / perencanaan operasi PLTS yang secara operasional memang butuh rata-rata per jam, bukan nilai instan. Tapi **tidak** dibandingkan head-to-head dengan R² titik di tabel utama paper — ditempatkan di bagian analisis tambahan/lampiran dengan disclaimer eksplisit: *ini tugas statistik yang berbeda (estimasi rata-rata jendela 1 jam), variansnya inheren lebih kecil dari target titik, sehingga R²-nya tidak boleh dibaca sebagai "akurasi forecast 1 jam" dalam pengertian titik.*

Baseline smart-persistence titik-vs-rata-rata (bukti struktural §1.3 poin 1), test 2025:

| Lokasi | SP R² titik (t+60) | SP R² rata-rata (avg_t10_t60) | Δ (murni dari definisi target) |
|---|---|---|---|
| Bengkulu | 0,617 | 0,795 | +0,178 |
| Banten | 0,496 | 0,707 | +0,211 |
| Jambi | 0,427 | 0,650 | +0,223 |
| Kalbar | — (lihat r1_compiled) | 0,707 | — |

Tabel lengkap hasil model untuk `avg_t10_t60` (4 lokasi) dipindah ke `06_Perbandingan_4_Lokasi.md` §"Lampiran".

---

## 2. Strategi Split — Kronologis Wajib

```
Train      : semua data sebelum 2024-01-01
Validasi   : tahun 2024 penuh
Test       : tahun 2025 penuh (holdout, hanya dievaluasi sekali di akhir)
```

**Kenapa bukan random split atau stratified-by-sky-condition?** Percobaan awal (v4, Bengkulu) menunjukkan random/stratified split rentan menyembunyikan **temporal leakage** — data masa depan ikut masuk ke train set melalui kedekatan waktu dengan test set. Split kronologis murni adalah yang paling ketat dan yang dipakai konsisten di seluruh eksperimen v3–v8 Bengkulu serta R1/R8 harmonis.

Fit scaler/normalizer (jika dipakai) **hanya di train**; terapkan ke val dan test tanpa refit.

---

## 3. Walk-Forward 5-Fold — Uji Stabilitas Tambahan

Di atas evaluasi test-2025 tunggal, jalankan walk-forward 5-fold kronologis (LightGBM residual × target titik) untuk menguji **stabilitas** model antar periode 6-bulanan — satu angka R² dari satu holdout bisa menyesatkan kalau window-nya kebetulan jatuh di periode sulit (musim tidak lengkap, anomali iklim).

Hasil walk-forward yang sudah ada (lihat `06_Perbandingan_4_Lokasi.md` untuk detail per fold):

| Lokasi | R² rata-rata | σ (std antar fold) | Interpretasi |
|---|---|---|---|
| **Bengkulu** | **0,7944** | **±0,0197** | Paling stabil |
| Kalbar | 0,6628 | ±0,0424 | Fold awal lemah, membaik seiring waktu (+0,115 dari fold 1→5) |
| Banten | 0,6429 | ±0,0574 | Variabilitas tertinggi |
| Jambi | 0,6249 | ±0,0533 | Test set terkecil per fold (n≈9,5k) |

**Catatan penting**: R² walk-forward Bengkulu (0,7944) lebih rendah dari R² test-2025 tunggal (0,792 CatBoost / R1 point_t60) — keduanya konsisten dalam rentang yang sama, mengonfirmasi angka test-2025 bukan kebetulan window yang mudah.

---

## 4. Baseline Wajib: Smart-Persistence

Bukan persistence naif (`Î(t+k) = I(t)`), melainkan:

$$\hat{I}(t+k) = k_t(t) \times I_{clr}(t+k)$$

di mana $k_t(t) = I(t)/I_{clr}(t)$ adalah clearness index saat ini, dan $I_{clr}(t+k)$ dihitung murni dari astronomi (elevasi matahari di waktu target) — bukan leakage. Semua model harus di-*outperform* baseline ini; skill score dilaporkan relatif terhadap RMSE baseline ini (lihat `04_Model_dan_Training.md` §metrik).

Tabel baseline SP titik-vs-rata-rata per lokasi ada di §1.4 (Jambi sudah pakai angka v2/tabel 24-jam; Bengkulu, Kalbar, Banten masih angka lama, menunggu rerun per-horizon).
