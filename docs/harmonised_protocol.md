# Harmonised Protocol

This document specifies, in full, the single reference protocol applied
identically at all four sites (Banten, Bengkulu, Jambi, West Kalimantan)
for the headline benchmark results reported in the accompanying paper.
Site-specific production variants (curated feature sets, ensembles,
stacking, pvlib clear-sky) exist in each site's own scripts but are
reported separately as sensitivity/supplementary analyses — they are
not part of this harmonised configuration.

## 1. Forecast task

- **Horizon:** 1 hour ahead (t+60 min), forecasts issued every 10 min.
- **Two target definitions**, evaluated separately (not pooled as one headline number):
  - **Instantaneous target:** `GHI(t+60min)` — irradiance at a single future 10-min timestamp. Evaluated and reported at each of six lead times (10, 20, 30, 40, 50, 60 min) as the primary result.
  - **Hourly-mean target:** `ghi_next_1h_mean = mean{GHI(t+10), …, GHI(t+60)}` — relevant for PV dispatch/energy planning. Reported as a supplementary result only; temporal averaging mechanically reduces target variance, so the two target definitions are **not directly comparable**.

## 2. Data resolution and split

- Common 10-min grid, local time (WIB), at all four sites.
- Chronological split: training 2021/2022–2023, validation 2024, held-out test 2025.
- Complemented by 5-fold chronological walk-forward validation with semi-annual test blocks (2023–2025), reported as the fold mean.
- Analysis period: 2021–2025 at Bengkulu and Jambi; 2022–2025 at Banten and West Kalimantan (2021 radiation archives at the latter two are too sparse for modelling).

## 3. Quality control and valid-anchor definition (§2.3)

A forecast sample ("anchor") is valid under the harmonised filter if, identically across all four sites:
- Continuous 10-min history with no gaps inside the lookback or target window (±30 s tolerance), spanning a strict 3-hour lookback;
- Solar elevation > 5° at **both** the issue time and the end of the target window, computed astronomically (Cooper 1969, from station lat/lon — not a stored/legacy elevation column);
- `GHI ∈ [0, 1400] W/m²`.

QC applied before this filter: de-duplication across overlapping source files; physical-consistency screening (e.g. `DHI ≤ GHI`, corrected at West Kalimantan using monthly × solar-altitude diffuse-fraction climatology); timestamp and station-metadata audits including verification of station coordinates; gap flagging; daylight filtering. Night-time data are excluded from both training and evaluation.

Note: the per-site pipeline filter used to generate the results in the paper's results section additionally requires a cloud-product quality flag not available at every site, differing from this pure §2.3 anchor count by ≤5.8% in test-set size (see the paper's Data and Study Area section for the measured impact, which is within the ±0.003 noise floor from seed variation).

## 4. Feature engineering

Compact harmonised recipe: **50 features** total —
GHI/kt history (16) + satellite cloud (CLP) history (15) + cyclic time encodings (6) + deterministic future regressors (4), with a with-meteorology arm (+5 AWS features: temperature, humidity, pressure, wind, rainfall) evaluated separately to test redundancy rather than assumed.

- **Radiation history:** lags of GHI and kt over a 3-hour lookback (18 × 10-min steps); rolling mean/std/range at 1–3 h scales; first differences (trend) and second differences (acceleration) of GHI and kt.
- **Satellite cloud history:** lags of COT/CLOT and companion Himawari CLP variables; cloud dynamics features (ΔCOT, rolling σ of COT, 30/60-min cloud trend); at Jambi, additional CLP spatial statistics around the station.
- **Deterministic future regressors:** clear-sky irradiance `I_clr(t+k)` and solar-geometry terms for the target window — legitimate because they depend only on deterministic astronomy, known perfectly in advance (Eqs. 1–3 below).
- **Cyclic time encodings:** hour-of-day and day-of-year sine/cosine pairs.
- Doubling the lookback window to 6 h yielded no measurable gain over 3 h.

Site-specific extras validated by ablation (not part of the 50-feature harmonised recipe): smart-persistence projection (Banten); SYNOP layered oktas at native hourly resolution (Bengkulu hourly pipeline); spectral AOD (Jambi, West Kalimantan).

Leakage guards: no actual future sensor or satellite values enter the feature set; scalers are fitted on training data only; signal decompositions (VMD/EMD/wavelet) are excluded from all reported models because they leak future information when applied before the temporal split.

### Physical basis (equations)

For a station at latitude φ, longitude λ (local meridian λ₀), solar declination δ on day-of-year n (Cooper 1969):

```
δ = 23.45° · sin(360° · (284 + n) / 365)                         (1)
```

Local solar time `t_s = t_clock + 4(λ − λ₀)/60` (minutes), hour angle `H = 15°(t_s − 12)`, solar elevation α:

```
sin(α) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(H)                        (2)
```

Harmonised clear-sky reference (identical form across all four sites, no site-specific turbidity/aerosol correction):

```
I_clr(t) = 1100 · max(sin(α(t)), 0)   [W/m²]                      (3)
```

Clearness index (input feature only, never the forecast target):

```
k_t(t) = GHI(t) / max(I_clr(t), 20 W/m²)                          (4)
```

Acceleration feature (20-min second difference) for X ∈ {GHI, kt, COT}:

```
accel_X(t) = X(t) − 2·X(t−10min) + X(t−20min)                     (5)
```

Mandatory smart-persistence baseline (Section 6 below):

```
Î_SP(t+k) = k_t(t) · I_clr(t+k)                                   (6)
```

## 5. Models

- **Reference (primary) model:** LightGBM in residual form (predicts Δ = target − GHI_now), trained identically at every site for both target definitions, with identical hyperparameters and early stopping.
- **Sensitivity check:** CatBoost in direct form (differences ≤ 0.006 R² vs. LightGBM in every cell).
- **Deep-learning comparators:** LSTM, CNN-LSTM, MLP, Transformer/PatchTST, trained on identical samples with early stopping on the validation year, comparable capacity, and multi-seed averaging. Three composition strategies evaluated as supplementary analyses: cross-family ensemble averaging (Banten), stacked direct+residual with a meta-learner (Bengkulu), regime-conditional models switched by cloud regime (West Kalimantan, Bengkulu).
- Hyperparameters tuned with Optuna within compute budgets; tuning contributed < 0.001 R² everywhere, so defaults with early stopping are used unless stated otherwise.
- Site-specific best-performing variants (curated feature sets beyond the 50-feature recipe, ensembles, stacking, pvlib/Ineichen clear-sky) are reported separately and never mixed into the cross-site harmonised benchmark.

## 6. Baseline and evaluation

Mandatory baseline — smart-persistence (Eq. 6), strictly stronger than naive persistence since it combines the current clearness index with the deterministic clear-sky curve at the target lead time. Every reported model is evaluated against it via the skill score:

```
SS = 1 − RMSE_model / RMSE_SP                                     (7)
```

Metrics: R², MAE, RMSE (W/m²) on daylight samples, plus SS (Eq. 7), evaluated on the same held-out samples as the model it scores. Results are additionally stratified by cloud regime (clear: COT < 1; partly cloudy: 1 ≤ COT < 8; overcast: COT ≥ 8) and by hour of day.

## 7. Reference scripts per site

Each site directory in this repository follows the naming convention:

```
train_ghi_1h_<site>_R1_benchmark.py    # harmonised reference-configuration benchmark
train_ghi_1h_<site>_R8_armA.py         # feature/meteorology redundancy arm
train_ghi_1h_<site>_R8_armB.py         # architecture comparison arm
train_ghi_1h_<site>_R8_armC.py         # feature-pruning arm
<site>_armB_bootstrap.py               # paired block-bootstrap significance test (arm B)
```

## Source

Consolidated from the project's internal methodology notes
(`04_Eksperimen/Restrukturisasi/00_Ringkasan_dan_Protokol_Standar.md` and
`06_Disertasi/01_Draft_Sec2_Data_Sec3_Methods.md`, §2.3 and §3.1–3.4) and
verified against the production feature/training scripts referenced
therein. Numeric per-site results (R², MAE, RMSE, feature counts, split
sizes) are reported in the paper itself and in each site's `outputs_*`
folders in this repository, not repeated here.
