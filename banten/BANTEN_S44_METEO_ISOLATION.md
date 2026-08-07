# Banten §4.4 — Isolasi Meteo (F1 vs F2 murni)

> Di WORKSPACE (vault meng-quarantine file sesi ini). Angka nyata dari
> `outputs_R8_meteo_isolation_banten/`. Skrip: `train_ghi_1h_banten_R8_meteo_isolation.py`.

## Motivasi
Gap-close menunjukkan +meteo (+0,012) sbg kontributor terbesar, TAPI dari langkah **kumulatif**
(82→87 fitur), jadi bisa jadi artefak interaksi dgn full-lags/variabilitas. Run ini
**mengisolasi murni**: lean-50 (F1) vs lean-50 + 5 meteo (F2), tanpa tambahan lain.

## Hasil 1 — Isolasi F1 vs F2 (robust: 2 model × 2 target × VAL+TEST)

| Target | Model | F1 test | F2 test | ΔR² test | ΔR² **val** |
|---|---|---|---|---|---|
| point t+60 | CatBoost | 0,6804 | 0,6932 | **+0,0128** | +0,0095 |
| point t+60 | LightGBM | 0,6787 | 0,6901 | **+0,0114** | +0,0104 |
| avg t+10..t+60 | CatBoost | 0,8344 | 0,8424 | +0,0081 | +0,0061 |
| avg t+10..t+60 | LightGBM | 0,8329 | 0,8404 | +0,0075 | +0,0065 |

**KONFIRMASI**: meteo membantu **positif di VAL DAN TEST**, di **kedua model**, **kedua target**.
Bukan overfit test-split (val gains +0,006…+0,010 konsisten positif). ΔMAE test −1,4…−2,1 W/m².
→ Temuan meteo-tidak-redundan di Banten **valid & robust** untuk §4.4.

## Hasil 2 — Pendorong utama: WIND SPEED (per-fitur, point, CatBoost)

| lean-50 + 1 fitur | test R² | ΔR² |
|---|---|---|
| (base lean-50) | 0,6804 | — |
| + temp_air_c | 0,6816 | +0,0012 |
| + humidity_pct | 0,6811 | +0,0007 |
| **+ wind_speed_ms** | **0,6919** | **+0,0116** |
| + rainfall_mm | 0,6808 | +0,0004 |
| + pressure_hpa | 0,6810 | +0,0006 |

**SMOKING GUN**: seluruh efek meteo digerakkan **wind_speed** (+0,0116 dari total +0,0128 utk 5 fitur).
Suhu/RH/hujan/tekanan masing-masing ~nol. Wind = ~90% efek.

## Interpretasi Fisis (§4.4)
- "Meteo membantu di Banten" sebenarnya = "**kecepatan angin membantu di Banten**".
- Angin = proksi **laju adveksi awan** / pencampuran lapisan batas. Di rezim konvektif Banten,
  kecepatan angin menandai seberapa cepat awan bergerak melintas → memengaruhi GHI 1 jam ke depan.
- Menyambung temuan tetangga (adveksi): angin membawa awan; wind_speed lokal = proksi laju adveksi
  tanpa perlu stasiun tetangga.
- Konsisten: di superset gap-close, `wind_speed_ms` rank #6 importance.

## Batasan (jujur utk paper)
- Meteo (wind) menutup hanya +0,013 → capai 0,693; **tidak** menutup gap penuh ke 0,73/0,740.
  Sisa −0,037…−0,047 tetap butuh clearsky pvlib + fitur eksogen (aerosol/present-weather) + ensemble.
- Banten = **satu-satunya** dari 4 lokasi di mana meteo tidak redundan — kemungkinan krn meteo
  Banten 100% lengkap (konsolidasi SYNOP) & CLP hanya ~50% coverage (angin isi celah adveksi).

## Kalimat §4.4 (draf)
> "At Banten — uniquely among the four sites — surface meteorology is not redundant with the
> lean-50 baseline, improving point-t+60 R² by +0.011–0.013 (robust across CatBoost/LightGBM and
> val/test). Per-feature isolation attributes essentially the entire gain to **wind speed**
> (+0.0116 of +0.0128), a proxy for cloud-advection rate in Banten's convective regime; the other
> four surface variables are individually negligible (≤+0.0012). This does not close the full
> site-specific gap (−0.047 to production), which requires physical clearsky + exogenous aerosol/
> weather features + ensembling."
