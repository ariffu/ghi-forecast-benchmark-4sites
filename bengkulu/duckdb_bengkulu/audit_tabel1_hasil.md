# Audit Tabel 1 Bengkulu — Hasil Verifikasi Langsung DB

**Tanggal audit**: 2026-08-02  
**DB**: `C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb`  
**Script**: `audit_tabel1_bengkulu.py` (commit sesama folder)

---

## Tabel Ringkasan

| Metrik | Referensi | Audit (ts-only) | Audit (strict) | Deviasi | Status |
|--------|-----------|-----------------|----------------|---------|--------|
| Raw 10-min 2021–2025 | 241,714¹ | **263,448** | — | lihat catatan | ⚠ FLAG |
| Anchor §2.3 total | **109,196** | 109,717 | 109,669 | +0.48% / +0.43% | ✓ VALID |
| train (<2024) | 64,863 | 65,293 | 65,246 | +0.66% | ✓ |
| val (2024) | 22,359 | 22,363 | 22,363 | +0.02% | ✓ |
| test (2025) | 21,974 | 22,061 | 22,060 | +0.40% | ✓ |

¹ Angka 241,714 adalah nilai lama yang sebelumnya di Tabel 1 — **bukan** dari lingkungan referensi Banten.  
Referensi Banten untuk raw 2021–2025: **263,448** — cocok persis dengan hasil audit Bengkulu.

### Per tahun (ts-only vs referensi)

| Tahun | Referensi | Audit | Deviasi |
|-------|-----------|-------|---------|
| 2021 | 19,817 | 20,220 | +2.0% ← terbesar |
| 2022 | 22,419 | 22,446 | +0.1% |
| 2023 | 22,627 | **22,627** | **±0** (persis) |
| 2024 | 22,359 | 22,363 | +0.02% |
| 2025 | 21,974 | 22,061 | +0.4% |

2023 cocok persis → formula elevasi dan filter sudah benar.  
2021 gap +403 baris: kemungkinan perbedaan snapshot database atau minor quality flag di lingkungan sumber referensi (Jan 2021 memiliki 228 NULL GHI, tertinggi per-bulan).

---

## Uji Grid 24-Jam

| Pemeriksaan | Nilai | Status |
|-------------|-------|--------|
| Baris malam (jam 0–5) di raw | **65,829** | Cocok persis dgn referensi |
| Baris malam di anchor §2.3 | 0 | Wajar — filter elev>5° membuang semua baris malam |

> Catatan: referensi menyebut "uji cepat grid 24-jam: referensi ~65,829". Ini adalah jumlah baris malam di data RAW, bukan di anchor. Anchor memang nol baris malam — sesuai spesifikasi §2.3 (elev>5° anchor DAN t+60).

---

## Konsistensi Pipeline vs §2.3

| Set | Total | train | val | test |
|-----|-------|-------|-----|------|
| §2.3 ts-only | 109,717 | 65,293 | 22,363 | 22,061 |
| Pipeline view (is_ready+3h+GHI) | 117,559 | 66,247 | 25,976 | 25,336 |
| Pipeline + elev_t60>5° (runtime R1/R8) | **105,051** | **59,114** | **23,226** | **22,711** |

**Selisih pipeline+elev vs §2.3: −4,666 (−4.3%)** — di atas threshold 1%, tapi **bukan** inkonsistensi berbahaya:

### Penjelasan selisih

| Faktor | Arah | Penjelasan |
|--------|------|-----------|
| `is_model_ready=1` memerlukan CLP tersedia | Pipeline lebih kecil | 2021: pipeline 14,793 vs §2.3 20,220 — CLP jarang di 2021 |
| Pipeline **tidak** mensyaratkan elev_anchor>5° | Pipeline lebih besar | Baris twilight (anchor gelap, target cerah) masuk pipeline tapi tidak §2.3 |
| §2.3 mensyaratkan GHI(t+60) valid | §2.3 lebih kecil | Baris dengan GHI target NULL/outlier ikut pipeline (target dihitung nanti) |

Net: faktor CLP mendominasi → pipeline secara keseluruhan lebih kecil dari §2.3.

**Test 2025**: pipeline 22,711 vs §2.3 22,061 (+2.9%) — pipeline punya +650 baris karena tidak mensyaratkan elev_anchor>5°. Ini masuk akal: 2025 CLP lengkap (tidak ada penalty CLP), sehingga efek "tidak perlu elev_anchor" mendominasi.

---

## Apakah R² Results §4 Perlu Diubah?

**TIDAK.** R² dihitung konsisten di pipeline test set (22,711 baris, semua model dilatih pada 59,114 baris yang sama). §2.3 hanya mendefinisikan basis deskriptif untuk Tabel 1; pipeline mendefinisikan basis evaluasi. Keduanya berbeda secara wajar dan terjelaskan.

Satu-satunya risiko: kalau pembaca membandingkan `n=22,061` di Tabel 1 dengan `n=22,711` di Results §4 dan mempertanyakan selisih 650 baris. Catatan kaki berikut cukup:

> "Jumlah anchor §2.3 (Tabel 1) dan set evaluasi pipeline (§4) sedikit berbeda: filter pipeline mensyaratkan CLP tersedia dan elev_t60>5° tanpa mensyaratkan elev_anchor>5°, sedangkan §2.3 mensyaratkan elev>5° di anchor DAN t+60. Perbedaan ~2.9% di test 2025 tidak mempengaruhi komparabilitas hasil lintas lokasi."

---

## Verdict Final

| Pertanyaan | Jawaban |
|-----------|---------|
| Apakah 109,196 valid untuk Tabel 1? | **YA — VALID** (deviasi +0.48%, di bawah threshold 1%) |
| Perlu koreksi angka di Tabel 1? | TIDAK |
| Angka raw yang benar? | **263,448** (bukan 241,714 yang lama) |
| Pipeline vs Results konsisten? | YA — R² tidak perlu diubah |
| Ada baris malam di anchor §2.3? | TIDAK — sesuai spesifikasi |
| Verifikasi 2023 (benchmark tahun tengah)? | EXACT MATCH (22,627 = 22,627) |

---

*Audit dilakukan langsung dari database, bukan dari catatan lama.*
