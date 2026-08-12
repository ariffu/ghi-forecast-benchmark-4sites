# Model dan Training — Arsitektur Standar

## 1. Model Wajib (identik di semua lokasi)

| Peran | Model | Alasan |
|---|---|---|
| **Primer / produksi** | LightGBM, mode **residual** (target = future − now) | Cepat, ringan, performa kompetitif; residual learning terbukti jadi pembeda terbesar antar fase di Bengkulu (+0,07 R² dari direct ke residual+ensemble) |
| **Sensitivitas / pembanding** | CatBoost, mode **direct** | Konsisten sedikit unggul dari LightGBM (+0,002–0,006 R²) di keempat lokasi — dipakai untuk validasi silang, bukan produksi utama karena training lebih lambat |
| **Baseline wajib** | Smart-persistence | Lihat `03_Target_dan_Split.md` §4 |

**Tidak dipakai sebagai model utama**: LSTM/CNN-LSTM standalone (lihat daftar teknik gagal di `02_Feature_Engineering.md` §4). MLP dan Transformer kompetitif (gap −0,2% s/d −0,8% dari CatBoost) dan boleh dilaporkan sebagai pembanding arsitektur (Arm B), tapi bukan kandidat produksi.

## 2. Prinsip Desain: Satu Model Baik Mengalahkan Ensemble Besar yang Timpang

Temuan kunci dari eksplorasi lanjutan Bengkulu (v6 → v10): model tunggal LightGBM residual dengan fitur yang tepat (43 fitur, v10) **mengungguli** stacked ensemble 5-model penuh (143 fitur, v6) — R²=0,8212 vs 0,8210, dengan fitur jauh lebih sedikit dan tanpa kompleksitas bagging/stacking.

Ensemble lintas-keluarga (LightGBM+CatBoost+XGBoost+MLP, rata-rata sederhana) **merugikan** ketika salah satu anggota jauh lebih lemah (gap R² > 0,02) — di Bengkulu, MLP sederhana (R²=0,765) menarik rata-rata ensemble turun di bawah LightGBM tunggal (R²=0,819→0,818). Aturan: **hanya ensemble kalau semua anggota kompetitif** (selisih R² < 0,02); kalau tidak, pakai model tunggal terbaik.

## 3. Hyperparameter Default (titik awal, boleh tuning ringan)

### LightGBM (residual)
```
num_leaves       : 31 (range tuning: 15–255)
learning_rate    : 0.05–0.1
n_estimators     : 300–1000 (early stopping berbasis validasi)
max_depth        : -1 (unlimited, biarkan num_leaves yang kontrol)
min_child_samples: 20
subsample        : 0.8
colsample_bytree : 0.8
reg_alpha/lambda : mulai 0, naikkan kalau overfit
early_stopping   : berbasis metric validasi (bukan test), patience wajar (~150 round GBM)
```

### CatBoost (direct)
Hyperparameter default library + early stopping di validation set. `best_iter` dilaporkan per run (lihat hasil aktual di §5).

**Catatan tuning**: berdasarkan temuan Banten, tuning hyperparameter ekstensif hanya menambah ~0,005 R² — prioritaskan kualitas fitur dan data dulu sebelum tuning intensif.

## 4. Fair-Play Constraints (untuk perbandingan arsitektur / Arm B)

Kalau ingin membandingkan GBM vs Deep Learning (LSTM/MLP/Transformer), wajib pakai aturan berikut supaya perbandingan tidak bias:

- Fitur **identik** untuk semua model (F1, 50 fitur)
- `seq_len=1` untuk DL kalau fiturnya sudah tabular ter-engineer (lag/rolling eksplisit) — supaya setara dengan input GBM
- Early stopping **hanya** berbasis validation set, tidak ada kebocoran test
- DL: minimal **3 seed**, laporkan rata-rata ± std
- Semua model dievaluasi pada baris test yang sama persis

**Peringatan interpretasi LSTM**: dengan `seq_len=1`, LSTM kehilangan konteks sekuensial yang jadi kekuatan utamanya — hasil underperform LSTM di eksperimen ini (Δ −3,8% s/d −8,7% dari CatBoost) mencerminkan keterbatasan setup fair-play tabular, bukan kapasitas DL secara umum. Jangan generalisasi ke "LSTM selalu buruk untuk solar forecasting" tanpa catatan ini.

## 5. Hasil Arm B Aktual (test 2025, target titik) — 4 Lokasi

| Model | Bengkulu | Kalbar | Banten | Jambi | Tipe |
|---|---|---|---|---|---|
| **CatBoost** | **0,7920** | **0,7278–0,7283** | **0,6818** | **0,6757** | GBM (primer sensitivitas) |
| LightGBM | 0,7899 | 0,7264 | 0,6787 | 0,6741 | GBM (primer produksi) |
| MLP | 0,7901 ±0,0003 | 0,7218 ±0,0004 | 0,6768 ±0,0021 | 0,6700 ±0,0008 | DL |
| Transformer | 0,7848 ±0,0011 | 0,7223 ±0,0016 | 0,6735 ±0,0039 | 0,6671 ±0,0041 | DL |
| LSTM | 0,7537 ±0,0037 | 0,6718–0,672 ±0,0023 | 0,6172 ±0,0043 | 0,5884 ±0,0014 | DL |

**Pola konsisten di 4 lokasi**: CatBoost > LightGBM > MLP ≈ Transformer >> LSTM. Gap CatBoost vs LightGBM hanya 0,14–0,31% (bukan 0,5–2% seperti asumsi umum di literatur) — GBM-vs-GBM lebih dekat daripada GBM-vs-DL.

Sumber: `outputs_R8_<lokasi>/arm_B_results.csv` dan `arm_B_summary.csv` di masing-masing folder lokasi.

## 6. Metrik Wajib

```
R²    : koefisien determinasi, metrik utama untuk perbandingan lintas-lokasi
MAE   : W/m², lebih robust terhadap outlier, mudah diinterpretasi
RMSE  : W/m², dipakai untuk hitung skill score
Skill : 1 − RMSE_model / RMSE_smart-persistence  (di atas 0 = mengalahkan baseline)
```

Loss function saat training **boleh berbeda** dari metrik evaluasi (mis. train dengan MSE/residual, evaluasi dilaporkan dengan R²/MAE/RMSE) — jangan dicampur dalam penulisan laporan.
