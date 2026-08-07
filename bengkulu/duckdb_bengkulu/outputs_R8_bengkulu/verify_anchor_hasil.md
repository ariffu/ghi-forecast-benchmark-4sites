# Hasil Verifikasi Anchor Homogen §2.3 — Bengkulu & Kalbar

**Script**: `verify_homogen_anchor_bengkulu_kalbar.py`  
**Tanggal**: 2026-08-02

---

## Bengkulu

| | Nilai | vs Referensi |
|---|---|---|
| §2.3 anchor (hitung ulang) | **109,855** | +0.60% vs ref 109,196 ✓ VALID |
| §2.3 split | train=65,391 / val=22,376 / **test=22,088** | |
| Pipeline R1/R8 (Results §4) | train=59,114 / val=23,226 / **test=22,711** | |
| Selisih test §2.3 vs pipeline | **−2.74%** (623 baris lebih sedikit di §2.3) | ⚠ perlu cek |

**Catatan**: §2.3 test LEBIH KECIL dari pipeline. Alasan: §2.3 mensyaratkan `elev_anchor>5°`, pipeline tidak — rows twilight (anchor gelap, target terang) masuk pipeline tapi tidak §2.3.

---

## Kalbar

| | Nilai | vs Referensi |
|---|---|---|
| §2.3 anchor (hitung ulang) | **90,579** | **+0.00%** vs ref 90,579 ✓ EXACT MATCH |
| §2.3 split | train=45,260 / val=22,692 / **test=22,627** | |
| Pipeline R1/R8 (Results §4) | train=39,759 / val=20,706 / **test=21,386** | |
| Selisih test §2.3 vs pipeline | **+5.80%** (1,241 baris lebih banyak di §2.3) | ⚠ perlu cek |

Per-tahun §2.3: {2022: 22,630 / 2023: 22,630 / 2024: 22,692 / 2025: 22,627} — sangat merata, tidak ada anomali.

**Catatan**: §2.3 test LEBIH BESAR dari pipeline. Pipeline `anchor_valid` lebih ketat (kemungkinan mensyaratkan CLP tersedia), §2.3 tidak mensyaratkan CLP. Ini kebalikan dari Bengkulu.

---

## Perbandingan §2.3 Lintas 4 Lokasi (untuk Tabel 1 Paper)

| Lokasi | §2.3 anchor | Referensi Tabel 1 | Deviasi | Status |
|--------|-------------|-------------------|---------|--------|
| Kalbar | **90,579** | 90,579 | 0.00% | ✓ EXACT |
| Jambi v2 | 91,409 | 91,409 | 0.00% | ✓ EXACT |
| Banten | ~90,488 | 90,488 | ~0% | ✓ (verify Banten done sebelumnya, dR²=-0.0004) |
| Bengkulu | 109,855 | 109,196 | +0.60% | ✓ VALID |

Urutan terkecil ke terbesar (§2.3): Kalbar ≈ Banten < Jambi < Bengkulu

---

## Apakah Perlu Re-evaluasi R² di §2.3 Test Set?

| Lokasi | Selisih test | Preseden Banten | Estimasi dampak |
|--------|-------------|-----------------|-----------------|
| Bengkulu | −2.74% (623 baris kurang) | dR²=−0.0004 di 0.1% | Kemungkinan kecil (perlu verifikasi) |
| Kalbar | +5.80% (1,241 baris lebih) | — | Lebih besar dari Banten, perlu cek |

**Yang perlu dilakukan** (BUKAN retrain penuh, tapi re-evaluasi model yang sudah ada):
1. Bengkulu: evaluasi model CatBoost R1/R8 yang sudah ada di §2.3 test set (22,088 baris) → catat R²
2. Kalbar: evaluasi model CatBoost R1/R8 yang sudah ada di §2.3 test set (22,627 baris) → catat R²
3. Bandingkan vs R² yang sudah di paper (Bengkulu 0.792, Kalbar 0.7217)
4. Kalau selisih <±0.003: update Tabel 1 saja, Results §4 tidak perlu diubah
5. Kalau selisih >±0.003: update Results §4 juga

**Jalan pintas yang defensible**: karena Tabel 1 adalah tabel DESKRIPTIF (jumlah anchor, bukan evaluasi model), dan Results §4 secara eksplisit menggunakan pipeline filter yang berbeda, cukup tambahkan footnote:
> "Jumlah 'valid forecast anchor' di Tabel 1 mengikuti definisi §2.3 homogen. R² di §4.x dihitung pada set evaluasi pipeline masing-masing lokasi (lihat §2.3.x untuk definisi per-lokasi) yang sedikit berbeda karena perbedaan filter CLP dan elevasi anchor."
