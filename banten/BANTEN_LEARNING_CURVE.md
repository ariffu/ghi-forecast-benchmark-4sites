# Banten — Learning Curve: Apakah Volume Data Membatasi R²?

> Workspace (vault quarantine). Skrip: `train_ghi_1h_banten_learning_curve.py`.
> Output: `outputs_learning_curve_banten/`. Konfig harmonis R1/R8 (lean-50, CatBoost, point t+60).

## Pertanyaan
R² Bengkulu (~0,79) > Banten (~0,68). Bengkulu punya data lebih besar (2021-2025).
Apakah gap ini karena VOLUME data, atau beda SITUS (Bengkulu lebih mudah)?

## Hasil — dua mode subsample training (eval test 2025 tetap)

**ACAK (isolasi murni volume, periode 2 tahun konstan):**
| Fraksi | n_train | test R² |
|---|---|---|
| 10% | 4.524 | 0,6713 |
| 25% | 11.311 | 0,6777 |
| 50% | 22.622 | 0,6802 |
| 75% | 33.933 | 0,6807 |
| 100% | 45.244 | 0,6804 |
-> PLATEAU: naik cepat 10->50%, lalu DATAR. delta(75->100%)=-0,0003.

**RECENT (N terbaru — menambah cakupan musim):**
| Fraksi | n_train | test R² |
|---|---|---|
| 10% (~1,5 bln) | 4.524 | 0,6360 |
| 25% | 11.311 | 0,6533 |
| 50% (~1 thn) | 22.622 | 0,6704 |
| 75% | 33.933 | 0,6775 |
| 100% (2 thn) | 45.244 | 0,6804 |
-> MASIH NAIK: delta(75->100%)=+0,0029.

## Interpretasi (jawaban ilmiah)
1. **Volume BUKAN penghambat** (mode acak plateau di ~50%/22k sampel). Menambah data
   sejenis (distribusi sama) TIDAK menaikkan R² -> ceiling ~0,68 diset oleh
   stokastisitas konvektif, bukan jumlah sampel.
2. **Yang membantu = cakupan MUSIM/rezim, bukan volume** (mode recent masih naik:
   1 thn -> 2 thn +0,010 krn menangkap kedua fase monsun). Tapi gain 75->100% cuma
   +0,0029 -> ekstrapolasi +1 tahun (2021) paling banter +0,003-0,005.
3. **Gap Bengkulu-Banten (0,79 vs 0,68) BUKAN efek volume.** Andai Banten punya volume
   data Bengkulu pun, mode-acak membuktikan ia tetap plateau ~0,68. Gap = beda SITUS
   (Banten sea-breeze/urban-konvektif vs Bengkulu pesisir lebih mudah).

## Kesimpulan
Melatih ulang Banten dgn data "homogen/lebih besar seperti Bengkulu" **TIDAK akan**
menaikkan R² secara berarti (paling banter +0,003-0,005 dari cakupan musim ekstra).
Memvalidasi keputusan membatalkan ekstensi 2021: gainnya negligible.
Perbedaan Banten-vs-Bengkulu adalah karakter iklim situs, bukan kuantitas data.
