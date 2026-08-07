# Tabel 1 — Audit Kolom SYNOP (4 lokasi) — 2026-08-02

> Di WORKSPACE (`C:\Users\ariff\Duckdb_Banten\`), BUKAN vault (plugin sync meng-quarantine
> file baru). Semua angka dari query DuckDB langsung ke DB tiap situs.

## Keputusan pelaporan
Jumlah SYNOP dilaporkan **per periode radiasi masing-masing situs** dan **kadens seragam
07–19 LT (13 jam/hari)**:
- Banten & Kalbar → 2022–2025 (4 tahun)
- Bengkulu & Jambi → 2021–2025 (5 tahun)

## Hasil final (masuk Tabel 1)

| Situs | Periode | Thn | **SYNOP rec.** | Sumber tabel | Basis |
|---|---|---|---|---|---|
| Banten | 2022–2025 | 4 | **18.956** | main.synopcompletefilledv | record_status='ORIGINAL', 07–19 LT |
| Kalbar | 2022–2025 | 4 | **18.993** | main.synop_unified | record in-period, 07–19 LT |
| Bengkulu | 2021–2025 | 5 | **23.326** | bengkulu_sch.synop_bengkulu_quality_final | QC-final in-period, 07–19 LT |
| Jambi | 2021–2025 | 5 | **23.738** | jambi_sch.synop_jambi_combined | dibatasi 07–19 LT (sumber 24-jam) |

Pola logis: 4-tahun ~18,9k; 5-tahun ~23,3–23,7k. Bengkulu sedikit < Jambi krn gap observasi
2023–2024 (bukan pemotongan).

## Jejak koreksi dari angka lama
| Situs | Lama | Baru | Alasan |
|---|---|---|---|
| Banten | 23.700 | 18.956 | angka lama termasuk 2021 (+4.745, di luar periode radiasi 2022–2025) |
| Bengkulu | 25.289 | 23.326 | angka lama termasuk baris 2026 (+1.963) |
| Jambi | 43.824 | 23.738 | angka lama = 24-jam penuh + 2026; dibatasi ke 07–19 LT |
| Kalbar | 25.000 | 18.993 | angka lama placeholder; buang 2021 (+2.782) |

## Detail verifikasi Banten (asal angka 23.700 → 18.956)
- main.synopcompletefilledv: 23.738 baris total, 2021-01..2025-12, per-jam 07–19 LT.
- record_status: ORIGINAL=23.700, FILLED=37, RECOVERED=1 → angka lama 23.700 = ORIGINAL semua tahun.
- ORIGINAL periode 2022–2025 = **18.956** (2021 menyumbang 4.745).
- Bukan digelembungkan interpolasi; observasi asli 2 format sumber (OLD ME45/ME48 + NEW ME48_STD).

## Kolom AWS & Himawari CLP Banten (2026-08-02) — beri angka seperti situs lain
Baris Banten sebelumnya deskriptif tanpa angka. Diisi periode-konsisten 2022–2025:

| Kolom | Banten (2022–2025) | Full 2021–2025 | Sumber | Situs lain (pembanding) |
|---|---|---|---|---|
| AWS | **210.244** | 262.804 | main.aws_banten_sta2062 (STA2062, 10-mnt) | Bkl 279.665 / Jmb 211.103 / Kbr 196.043 |
| CLP | **106.653** | 133.298 | main.clp_banten (30-km buffer, 10-mnt siang) | Bkl 143.785 / Jmb 140.278 / Kbr 142.428 |

- Per tahun AWS: 2021=52.560, 2022=52.561, 2023=52.561, 2024=52.705, 2025=52.417.
- Per tahun CLP: 2021=26.645, 2022=26.645, 2023=26.645, 2024=26.718, 2025=26.645 (siang 06–18).
- Model produksi pakai clp_banten (CLP non-null di solar_features_base = 90.415).
- 4 seri CLP tetangga (BSD/Golf/TMII/UI) resolusi lebih rendah (per-jam siang, ~17.456/seri
  2022–2025) → auxiliary/eksploratif, ditulis "+ 4-station series" di sel. Magnitudo Banten
  kini setara situs lain.

Sel Tabel 1 Banten (final):
- AWS: "✓ (210,244 rec.; + 4 neighbour AWS, exploratory)"
- CLP: "✓ (106,653 rec., 30-km buffer; + 4-station series)"

## Okta awan SYNOP per situs (2026-08-02) — §2.2 dikualifikasi & DISELESAIKAN
Non-null okta/tahun (query DB langsung):

| Situs | CL | CM | CH | Ringkas |
|---|---|---|---|---|
| Bengkulu | ✅ 2021–25 | ✅ 2021–25 | ✅ 2021–25 | CL/CM/CH okta penuh (cloud_low/med/high_cover_oktas) |
| Kalbar | total ✅ | ✅ 2022–24 | ✅ 2022–24 | CM/CH (cloud_med/high_oktas) NOL di 2025 |
| Jambi | okta total ✅ | type ✅ | type ~86–99% | 1 okta total + kode *type* L/M/H; bukan okta CM/CH terpisah |
| Banten | ✅ 2024–25 | ❌ | ❌ | cloud_low_cover_oktas saja, mulai pertengahan 2024 (~7.170) |

**Tindakan (SELESAI):** kalimat §2.2 di draft diganti agar mencerminkan heterogenitas ini
(bukan klaim polos "CL/CM/CH di 4 situs"). Ditambah caveat: cross-check CLP-vs-SYNOP di
Banten hanya 2024–2025.

**MASIH PERLU DICEK (file lain):** `02_Draft_Sec4_Results` §4.2 — kalau Banten disertakan
dalam validasi silang CLP-vs-SYNOP tanpa catatan periode, tambahkan caveat 2024–2025.

## Referensi
Draft: `06_Disertasi/01_Draft_Sec2_Data_Sec3_Methods.md` — Tabel 1 + `[KOREKSI 2026-08-02 (kolom SYNOP)]`.
Terkait: `BANTEN_HOMOGENITAS_BENGKULU.md` (kolom radiasi/anchor), `count_anchors_ALL4_homogen.py`.
