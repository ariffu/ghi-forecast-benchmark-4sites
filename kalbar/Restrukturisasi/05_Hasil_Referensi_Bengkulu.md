# Hasil Referensi Bengkulu — Standar Tertinggi

Dua lapis hasil Bengkulu perlu dibedakan dengan jelas di paper:

1. **Hasil di bawah protokol harmonis R1/R8** (sebanding langsung dengan 3 lokasi lain, dipakai untuk tabel benchmarking utama)
2. **Hasil eksplorasi lanjutan** (v9–v11, di luar protokol harmonis, dipakai untuk menunjukkan *ceiling* yang bisa dicapai dan arah pengembangan produksi Bengkulu — bukan untuk dibandingkan apel-ke-apel dengan lokasi lain sampai direplikasi ke lokasi lain)

---

## 1. Hasil Harmonis (R1, sebanding lintas-lokasi)

**Test 2025** (`outputs_R1_bengkulu/ghi_1h_R1_results.csv`):

| Model | Target | R² | MAE | RMSE | Skill vs SP |
|---|---|---|---|---|---|
| smart-persistence | titik | 0,6167 | 127,6 | 186,0 | 0,0 |
| LightGBM residual | titik | 0,7891 | 96,7 | 137,9 | 0,2583 |
| **CatBoost direct** | titik | **0,7920** | **96,5** | **137,0** | **0,2633** |
| smart-persistence | rata-rata | 0,7945 | 87,2 | 128,9 | 0,0 |
| LightGBM residual | rata-rata | 0,8994 | 61,8 | 90,2 | 0,3004 |
| **CatBoost direct** | rata-rata | **0,9004** | **62,0** | 89,7 | **0,3038** |

**Walk-forward 5-fold** (LightGBM residual × titik): R²=0,7944 ± 0,0197 — paling stabil dari 4 lokasi (lihat `03_Target_dan_Split.md` §3).

**R8 Arm A** (uji redundansi meteo, CatBoost): F1=0,792 → F2=0,7925 (titik), F1=0,9004 → F2=0,8999 (rata-rata) — ΔR² < 0,001, meteo **redundan** di Bengkulu (konsisten dengan Kalbar/Jambi, berbeda dari Banten).

**R8 Arm C** (feature pruning): **JANGAN dipakai apa adanya.** Investigasi mendalam (`diagnosa_armC_anomali.py`) menemukan cacat desain — superset fitur (F_super, 84 fitur) yang mencampur `ghi_now` dengan DHI/DNI mentah menyebabkan collinearity collapse (R² turun ke 0,763, di bawah F1 baseline 0,792). Root cause & mitigasi didokumentasikan di `02_Feature_Engineering.md` §4 poin 9. Kalau pruning Bengkulu ingin dilaporkan di paper, pakai metodologi forward-selection dari F1 (belum dijalankan ulang dengan metode yang benar — lihat `07_Status_dan_Rencana_Selanjutnya.md`).

---

## 2. Eksplorasi Lanjutan (di luar protokol harmonis — folder `bengkulu_ghi_julius`)

Bagian ini menjawab pertanyaan "bagaimana Bengkulu bisa sampai R²=0,90+" secara teknis, tapi **hasilnya belum diulang di 3 lokasi lain** dengan resep fitur persis sama (43 fitur v10, bukan 50 fitur F1 standar) — jadi dipakai sebagai **arah pengembangan**, bukan bagian dari tabel benchmark utama.

### v10 — Rekor Target Titik (43 fitur, single model)
40 fitur v8 + 3 fitur akselerasi (2nd-diff GHI/kT/cloud-optical-thickness, jendela 20 menit), satu model LightGBM residual (tanpa ensemble/bagging/stacking):

| Model | Fitur | R² | MAE | RMSE |
|---|---|---|---|---|
| v6 (stacked ensemble penuh) | 143 | 0,8210 | 94,1 | 134,6 |
| v8 (pruned, lama) | 40 | 0,8198 | 95,7 | 135,1 |
| **v10 (akselerasi + single model)** | **43** | **0,8212** | **94,7** | **134,6** |

v10 mengungguli stacking 5-model dengan 70% lebih sedikit fitur — **direkomendasikan sebagai resep produksi target-titik**.

### v11 — Menembus R²=0,90 (target rata-rata jam)
Target `target_ghi_1h_avg` = rata-rata `LEAD(ghi_now, 1..6)`, fitur tetap 43 (resolusi 10-menit penuh):

| Konfigurasi | Target | Fitur | R² | MAE | RMSE |
|---|---|---|---|---|---|
| (A) referensi | titik (t+60) | 43 | 0,8212 | 94,7 | 134,6 |
| **(B) hybrid Banten** | rata-rata (t+10..t+60) | 43 (resolusi 10-menit) | **0,9025** | **65,9** | **94,5** |
| (C) kontrol kasar | rata-rata (t+10..t+60) | 37 (fitur <60m dibuang) | 0,9008 | 66,3 | 95,2 |

**(B) vs (C) adalah perbandingan yang genuinely informatif** (target sama, baris sama, hanya resolusi fitur input beda): mempertahankan variabilitas input 10-menitan memberi +0,0016 R² dibanding fitur kasar — kecil tapi konsisten di 25.336 baris test.

**(A) vs (B) BUKAN perbandingan setara** — lihat penjelasan lengkap di `03_Target_dan_Split.md` §1. Ini alasan kenapa v11-B (R²=0,9025) tidak langsung dipakai sebagai "hasil resmi Bengkulu" di tabel benchmark utama — R²=0,9004 dari protokol R1 harmonis (§1 di atas, konfigurasi 50-fitur F1 standar) sudah menembus 0,90 dengan cara yang **fully comparable** dengan 3 lokasi lain.

### Breakdown R² per Kondisi Awan (v6, temuan paling penting untuk narasi ceiling)

| Kondisi awan (`clp_cot`) | n baris | MAE | R² |
|---|---|---|---|
| Tidak ada data CLP (proxy cerah) | 3.340 | 44,1 | **0,893** |
| Awan tipis | 12.308 | 100,1 | 0,766 |
| Awan sedang | 7.185 | 121,7 | 0,599 |
| Awan tebal | 2.503 | 48,0 | 0,521 |

R² agregat 0,82 (target titik) adalah rata-rata tertimbang kondisi mudah (cerah, R²=0,89) dan sulit (awan dinamis, R²=0,50–0,60). Bottleneck akurasi terkonsentrasi di kondisi awan sedang-dinamis, bukan kegagalan model di semua kondisi — argumen kuat untuk narasi *ceiling* fisis di paper (§`06_Perbandingan_4_Lokasi.md`).

### Kenapa Ceiling ≈0,82 (Target Titik) Sulit Ditembus

Sudah dicoba dan **tidak membantu**: HistGradientBoosting, LSTM, Transformer, ensemble stacking, segmentasi per-regime, PCA, PSO, dekomposisi wavelet, lag halus, kt-lag/roll, resolusi per-jam, pruning fitur ekstrem.

| Sumber ketidakpastian | Status |
|---|---|
| `clp_cot` cuma satu titik ukur, tanpa info arah/kecepatan gerak awan | Bottleneck utama |
| Resolusi satelit CLP 10 menit, tidak menangkap awan konvektif <5 menit | Kontribusi |
| Data ARP/aerosol tidak tersedia di database | Belum diuji |
| Single-station, tanpa jaringan untuk tracking spasial awan | Bottleneck struktural |

**Target realistis per pendekatan** (estimasi, sebagian belum diuji):
| Pendekatan | R² titik achievable |
|---|---|
| Single-station + CLP satelit (kondisi saat ini) | 0,78–0,82 ✅ tercapai |
| + Data ARP/aerosol lengkap | 0,83–0,85 (estimasi) |
| + Cloud motion vector / all-sky imager | 0,85–0,90 (estimasi) |
| + Jaringan stasiun + NWP | ≥0,90 (estimasi) |

---

## 3. Ringkasan untuk Bab Metodologi Paper

- **Angka resmi yang sebanding lintas-lokasi**: R1 harmonis, R²=0,792 (titik) / 0,900 (rata-rata) — masuk tabel benchmark utama.
- **Angka v10/v11 (0,8212 / 0,9025)**: dilaporkan terpisah sebagai *upper-bound exploration* khusus Bengkulu, dengan catatan metodologis eksplisit bahwa resep fiturnya (43, bukan 50 standar) belum direplikasi ke lokasi lain.
- **Rekomendasi konkret**: replikasi resep v10 (43-fitur accel-lean + single LightGBM residual) ke Kalbar/Banten/Jambi sebagai langkah lanjutan restrukturisasi — lihat `07_Status_dan_Rencana_Selanjutnya.md`.
