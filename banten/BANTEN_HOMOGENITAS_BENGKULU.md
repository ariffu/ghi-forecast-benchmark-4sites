# Banten ↔ Bengkulu — Bukti Homogenitas Data & Pipeline

> **PENTING**: Dokumen di WORKSPACE (`C:\Users\ariff\Duckdb_Banten\`), BUKAN vault
> (vault sync `remotely-save` menghapus file baru). Semua angka tervalidasi dari
> `confirm_homogeneous_anchor.py` (dijalankan Python39).

Menjawab: *"bagaimana caranya supaya data yang digunakan homogen dengan Bengkulu?"*
**Jawaban singkat: sudah homogen.** Benchmark Banten (R1/R8) memakai konsep IDENTIK
dengan Bengkulu di semua dimensi metodologis; satu-satunya perbedaan (periode awal 2021)
adalah ketersediaan data, bukan pilihan metode.

---

## 1. Konsep anchor Bengkulu (sumber kebenaran)

Dari `bengkulu_ghi_julius/*.py` (mis. `diagnosa_armC_anomali.py`, `train_ghi_1h_bengkulu_R1_benchmark.py`),
membaca view `ghi_forecast_1h_train_3h_rollback_2021_2025` dengan filter:

```
WHERE is_model_ready = 1              -- matahari di anchor cukup tinggi + GHI valid + bukan gap
  AND has_continuous_3h_history = 1   -- 18 langkah 10-menit sebelumnya lengkap tanpa gap
  AND ghi_now BETWEEN 0 AND 1400
```
lalu di Python ditambah:
```
sun_gt5_t60 = elev_sin_t60 > sin(5°)          # matahari > 5° di t+60
df = df[df.target.between(0,1400) & df.sun_gt5_t60]
```
clearsky = `1100·max(sin(elev),0)`, kt = `ghi/max(1100·max(sin(elev),0.02),20)`.

## 2. Konsep anchor Banten (R1/R8)

```
WHERE ghi_now BETWEEN 0 AND 1400
  AND ghi_lag_180m IS NOT NULL        -- riwayat 3-jam lengkap (grid 10-mnt kontinu)
  AND solar_elev_deg > 5              -- matahari > 5° di anchor
  AND sun_gt5_t60                     -- matahari > 5° di t+60
  AND target BETWEEN 0 AND 1400
```
clearsky & kt IDENTIK dengan Bengkulu.

## 3. Peta padanan (dimensi demi dimensi)

| Dimensi | Bengkulu | Banten | Status |
|---|---|---|---|
| Fitur | lean-50 (F1) | lean-50 (disalin **verbatim**) | ✅ identik |
| Clearsky | 1100·sin(elev) | 1100·sin(elev) | ✅ identik |
| kt | ghi/max(1100·sin,20) | idem | ✅ identik |
| Riwayat 3-jam kontinu | `has_continuous_3h_history=1` | `ghi_lag_180m NOT NULL` (grid kontinu) | ✅ setara |
| Matahari di anchor | `is_model_ready=1` (elev cukup) | `solar_elev_deg > 5` | ✅ setara |
| Matahari di t+60 | `sun_gt5_t60` (>5°) | `sun_gt5_t60` (>5°) | ✅ identik |
| GHI valid anchor & target | [0,1400] | [0,1400] | ✅ identik |
| Target | point t+60 & avg t+10..t+60 | idem | ✅ identik |
| Split | train<2024 / val2024 / test2025 | idem | ✅ identik |
| **Periode** | **2021–2025** | **2022–2025** | ⚠️ beda ketersediaan data |

## 4. Satu-satunya beda teknis: sumber elevasi anchor (dan dampaknya = nol)

Bengkulu menghitung elevasi surya **astronomis** (`solar_elev_deg`); Banten R1/R8 memakai
`elevation_deg` **tersimpan** (legacy). Uji langsung (`confirm_homogeneous_anchor.py`, point t+60):

| Definisi anchor | n | train/val/test | R² test | MAE |
|---|---|---|---|---|
| A) elevasi tersimpan (R1/R8 kini) | 90.488 | 45.244/22.685/22.559 | 0.6804 | 104.8 |
| B) elevasi astronomis (**persis Bengkulu**) | 90.384 | 45.192/22.660/22.532 | 0.6800 | 104.8 |

→ selisih 104 baris (**0,11%**), ΔR² = **−0,0004** (noise), MAE identik.
**Kesimpulan: pilihan sumber elevasi tidak berpengaruh; kedua set homogen.**

## 5. Periode 2021 (Banten tak punya) — bukan masalah metodologi

- Radiasi Banten hanya tersedia mulai 2022 (2021 tidak ada/terlalu jarang di sumber).
- **Learning-curve** membuktikan 2021 pun tak akan mengubah hasil: penambahan
  data lama (recent-subsample) hanya +0,003 R² dan sudah mendekati plateau; volume
  BUKAN faktor pembatas Banten. Gap R² Banten↔Bengkulu (0,68 vs 0,79) bersifat
  **site-intrinsic** (rezim konvektif/sea-breeze), bukan kuantitas data.
- Dilaporkan transparan di Tabel 1 sebagai rentang per-lokasi (Banten/Kalbar 2022–2025;
  Bengkulu/Jambi 2021–2025).

## 5b. Uji-ketahanan anchor/basis-data (point t+60) — untuk antisipasi reviewer

Diuji 3 kombinasi definisi (`confirm_hybrid_dbBengkulu_anchorBanten.py`):

| Konfigurasi | n | test | R² | MAE |
|---|---|---|---|---|
| A) Banten murni (elev tersimpan, kontinuitas longgar) | 90.488 | 22.559 | 0.6804 | 104.8 |
| B) Bengkulu murni (astronomis + kontinuitas ketat) | 90.384 | 22.532 | 0.6800 | 104.8 |
| C) Hibrida (pool DB Bengkulu × anchor Banten) | 89.960 | 22.427 | 0.6795 | 104.9 |

Rentang ΔR² = 0,0009 (< noise walk-forward ±0,003). **Hasil tak sensitif** terhadap
definisi anchor/basis-data → gap Banten↔Bengkulu (≈0,11) site-intrinsic, bukan artefak data.
**Tabel 1 mengadopsi basis B (90.384)** karena homogen dengan Bengkulu/Jambi.

## 6. Cara pelaporan Tabel 1 (agar tidak jadi pertanyaan)

Gunakan **basis anchor §2.3 yang sama** untuk keempat lokasi (jangan campur raw-record vs anchor):
jalankan `count_valid_anchors_TEMPLATE.py` (filter identik, elevasi astronomis) di tiap DB.
- Banten (terverifikasi) = **90.384** anchor valid §2.3.
- Kolom periode terpisah menjelaskan 2022– vs 2021– secara jujur.

## 6b. Anchor §2.3 HOMOGEN — 4 lokasi (Tabel 1 final) — `count_anchors_ALL4_homogen.py`

Satu filter identik (elevasi astronomis Cooper, kontinuitas 3-jam ketat, matahari>5°
di anchor & t+60, GHI∈[0,1400]):

| Lokasi | Periode | Raw 10-min | Anchor §2.3 | train | val | test | Koordinat |
|---|---|---|---|---|---|---|---|
| Banten | 2022–2025 | 210.241 | 90.384 | 45.192 | 22.660 | 22.532 | −6,26147/106,7509 |
| Bengkulu | 2021–2025 | 263.448 | 109.196 | 64.863 | 22.359 | 21.974 | −3,8607/102,3381 |
| Kalbar | 2022–2025 | 210.342 | 90.579 | 45.260 | 22.692 | 22.627 | −0,0356/109,3384 |
| Jambi | 2021–2025 | 262.944 | 88.462 | 50.390 | 17.533 | 20.539 | −1,5833/103,6667 |

**Keputusan Tabel 1 paper (hibrida aman, 2026-08-01):** Bengkulu & Kalbar diganti
RAW→anchor §2.3 (109.196 / 90.579); Banten 90.384; **Jambi dipertahankan 96.576**
(anchor pipeline v2, terikat n-test Results §4) — kolom jadi sejenis tanpa mengubah
angka Section 4. Basis §2.3 murni Jambi = 88.462 (dicatat, tidak dipakai di Tabel 1).
Sumber tabel radiasi mentah: Banten `solar_features_base`; Bengkulu
`bengkulu_master_10min_quality_final`; Kalbar `solar_kalbar_10m`;
Jambi `ghi_forecast_1h_train_3h_rollback_2021_2025` (grid 24-jam v2).

## 7. Catatan tentang vault 04_Eksperimen/Banten

Vault Banten (catatan 00–17) mendokumentasikan **model PRODUKSI** (149 fitur, target hybrid,
ensemble, R²≈0,740) — varian site-specific, bukan benchmark lintas-lokasi. Yang homogen dengan
Bengkulu adalah **benchmark harmonis lean-50** (R1/R8). Dokumentasi benchmark ini (catatan 18–22)
sempat ter-quarantine plugin sync; ringkasannya ada di `BANTEN_R1_R8_SUMMARY.md` (workspace).
Untuk paper: Tabel 2 (lintas-lokasi) memakai lean-50 homogen; produksi 149-fitur dilaporkan
terpisah sebagai sensitivitas site-specific.
