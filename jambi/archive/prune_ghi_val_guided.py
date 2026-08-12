"""
Val-guided feature pruning utk model GHI 1h (96 fitur) — protokol Kalbar note_17:
1. Baseline 96 fitur (VAL R²)
2. Sweep top-K importance (keputusan di VAL saja)
3. Greedy backward elimination dari top-K* (di VAL)
4. TEST dievaluasi SEKALI di akhir

Split 3-way: train=65%, val=15%, test=20% (kronologis) — identik prune_val_guided.py (kt).
Params pruning: 600it/lr0.05/d6 (cepat; keputusan relatif). Kandidat final diverifikasi
ulang dengan params tuned produksi (4000it/lr0.015/d8) di TEST.

Konteks: pruning kt-v4 (72 fitur) TIDAK membantu (pola Banten). Model GHI 96 fitur
punya blok baru (future_cs 14 fitur yang saling redundan — cs_t1..t6 hampir kolinear)
sehingga ada peluang pruning lebih besar; tapi ekspektasi tetap: netral.
Nilai utamanya kalau berhasil: model produksi lebih ramping & inference lebih cepat,
BUKAN kenaikan akurasi.
"""
import pandas as pd
import numpy as np
import warnings, json, time
import pvlib
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

t0 = time.time()
LAT, LON, ELEV = -1.5833, 103.6667, 35.0

df = pd.read_parquet('dfm_with_clp_stats.parquet')
df = df.sort_values('ts').reset_index(drop=True)
ts_arr = (df['ts'].values.astype('int64') / 1_000_000).astype(np.int64)

def add_delta_fast(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90):
    val = df[col].values.astype(float)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n - n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lag_arr = np.full(n, np.nan)
    lag_arr[n_steps:][valid] = val[:n - n_steps][valid]
    df[col + '_delta' + str(n_steps * 10) + 'm'] = val - lag_arr
    return df

def add_lag_fast(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90, name=None):
    val = df[col].values.astype(float)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n - n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lag_arr = np.full(n, np.nan)
    lag_arr[n_steps:][valid] = val[:n - n_steps][valid]
    df[name or (col + '_lag' + str(n_steps * 10) + 'm')] = lag_arr
    return df

for var in ['cloud_top_height_m', 'cloud_top_temp_c', 'cloud_eff_radius_um']:
    df = add_delta_fast(df, var, 3, ts_arr)
    df = add_delta_fast(df, var, 6, ts_arr)
df['dni_ratio'] = df['dni_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df['dhi_fraction'] = df['dhi_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df.loc[df['ghi_consolidated'] < 10, ['dni_ratio', 'dhi_fraction']] = np.nan
df['clot_std_roll30m'] = df['clot_std'].rolling(4, min_periods=2).mean()
df = add_delta_fast(df, 'clot_std', 3, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 3, ts_arr)

log("pvlib future clearsky...")
loc = pvlib.location.Location(latitude=LAT, longitude=LON, altitude=ELEV, tz='Asia/Jakarta')
ts_base = pd.DatetimeIndex(df['ts']).tz_localize('Asia/Jakarta')
for k in range(1, 7):
    ts_f = ts_base + pd.Timedelta(minutes=10 * k)
    sp = loc.get_solarposition(ts_f)
    cs = loc.get_clearsky(ts_f, model='ineichen')
    df[f'ghi_cs_t{k}'] = cs['ghi'].values
    df[f'sun_alt_t{k}'] = sp['apparent_elevation'].values
df['ghi_cs_next_mean'] = df[[f'ghi_cs_t{k}' for k in range(1, 7)]].mean(axis=1)
df['ghi_persist_next'] = df['kt'] * df['ghi_cs_next_mean']

for n in [1, 2, 3]:
    df = add_lag_fast(df, 'clot_median', n, ts_arr)
df = add_delta_fast(df, 'clot_median', 1, ts_arr)
df = add_delta_fast(df, 'clot_median', 3, ts_arr)
df['clot_median_roll30m'] = df['clot_median'].rolling(4, min_periods=2).mean()
for n in [1, 2]:
    df = add_lag_fast(df, 'cler23_coverage', n, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 1, ts_arr)

FEATS_V2 = [
    'sun_altitude', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'month',
    'kt', 'kt_lag10m', 'kt_lag20m', 'kt_lag30m', 'kt_lag60m',
    'kt_roll30m_mean', 'kt_roll60m_mean', 'kt_roll30m_std',
    'kt_short_lag1m', 'kt_short_lag2m', 'kt_short_lag3m', 'kt_short_lag5m',
    'kt_short_roll5m_mean', 'kt_short_roll5m_std',
    'kt_delta_1m', 'kt_delta_5m', 'kt_delta_10m', 'kt_slope_5m',
    'cloud_optical_thickness', 'cloud_top_temp_c', 'cloud_top_height_m',
    'cloud_eff_radius_um', 'sat_cloud_present',
    'cloud_cover_oktas', 'cloud_low_type', 'cloud_med_type', 'cloud_high_type',
    'cloud_optical_thickness_lag10m', 'cloud_optical_thickness_delta10m',
    'cloud_cover_oktas_lag10m', 'cloud_cover_oktas_delta10m',
    'cloud_optical_thickness_roll60m_mean', 'cloud_optical_thickness_roll60m_std',
    'cloud_cover_oktas_roll60m_mean', 'cloud_cover_oktas_roll60m_std',
    'angstrom_exp_440_870', 'precipitable_water_cm', 'AOD_planck_avg',
    'cloud_optical_thickness_delta30m', 'cloud_optical_thickness_delta60m',
    'cloud_cover_oktas_delta30m', 'cloud_cover_oktas_delta60m',
]
V3_NEW = [
    'dni_ratio', 'dhi_fraction',
    'cloud_top_height_m_delta30m', 'cloud_top_temp_c_delta30m', 'cloud_eff_radius_um_delta30m',
    'cloud_top_height_m_delta60m', 'cloud_top_temp_c_delta60m', 'cloud_eff_radius_um_delta60m',
]
V4_CLP = [
    'clot_std', 'clot_median', 'clot_coverage',
    'cler23_coverage', 'cler23_std', 'cler23_mean',
    'clth_mean', 'clth_std', 'clth_median',
    'cltt_mean', 'cltt_std', 'cltt_median',
    'cloud_class_code_new',
    'cler23_coverage_delta30m', 'clot_std_delta30m', 'clot_std_roll30m',
]
FUTURE_CS = ([f'ghi_cs_t{k}' for k in range(1, 7)] +
             [f'sun_alt_t{k}' for k in range(1, 7)] +
             ['ghi_cs_next_mean', 'ghi_persist_next'])
CLP_LAGS = [
    'clot_median_lag10m', 'clot_median_lag20m', 'clot_median_lag30m',
    'clot_median_delta10m', 'clot_median_delta30m', 'clot_median_roll30m',
    'cler23_coverage_lag10m', 'cler23_coverage_lag20m', 'cler23_coverage_delta10m',
]
FEATS96 = FEATS_V2 + ['ghi_clearsky'] + V3_NEW + V4_CLP + FUTURE_CS + CLP_LAGS

TARGET = 'ghi_next_1h_mean'

# 3-way split identik prune_val_guided.py
n = len(df)
n_tr = int(n * 0.65)
n_va = int(n * 0.80)
df_tr = df.iloc[:n_tr].copy()
df_va = df.iloc[n_tr:n_va].copy()
df_te = df.iloc[n_va:].copy()
log(f"Split: train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")

mask_tr = df_tr[TARGET].notna() & (df_tr['sun_altitude'] > 5)
mask_va = df_va[TARGET].notna() & (df_va['sun_altitude'] > 5)
mask_te = df_te[TARGET].notna() & (df_te['sun_altitude'] > 5)
y_va = df_va.loc[mask_va, TARGET].astype(float)
y_te = df_te.loc[mask_te, TARGET].astype(float)
X_va = df_va.loc[mask_va]
X_te = df_te.loc[mask_te]

CAT_FAST = dict(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                min_data_in_leaf=20, loss_function='Quantile:alpha=0.5',
                verbose=0, random_seed=42, allow_writing_files=False)
CAT_TUNED = dict(iterations=4000, learning_rate=0.015, depth=8, l2_leaf_reg=5,
                 min_data_in_leaf=20, loss_function='Quantile:alpha=0.5',
                 verbose=0, random_seed=42, allow_writing_files=False)

def fit_eval(feats, params=None):
    m = CatBoostRegressor(**(params or CAT_FAST))
    m.fit(df_tr.loc[mask_tr, feats].astype(float),
          df_tr.loc[mask_tr, TARGET].astype(float))
    return m, r2_score(y_va, m.predict(X_va[feats].astype(float)))

# PHASE 1: baseline
log("\nPhase 1: Baseline 96 fitur...")
m_base, r2_val_base = fit_eval(FEATS96)
log(f"  VAL R2 = {r2_val_base:.4f}")
fi = pd.Series(m_base.get_feature_importance(), index=FEATS96).sort_values(ascending=False)
log("\nTop 20 importance:")
for feat, imp in fi.head(20).items():
    log(f"  {imp:6.2f}  {feat}")

# PHASE 2: sweep top-K
log("\nPhase 2: Sweep top-K (di VAL)...")
K_values = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
sweep = {}
for K in K_values:
    feats_k = fi.head(K).index.tolist()
    _, r2_k = fit_eval(feats_k)
    sweep[K] = r2_k
    log(f"  K={K:3d}: VAL R2 = {r2_k:.4f}  (delta {r2_k - r2_val_base:+.4f})")

tolerance = 0.0005
k_star = None
for K in K_values:
    if sweep[K] >= r2_val_base - tolerance:
        k_star = K
        break
if k_star is None:
    k_star = max(sweep, key=sweep.get)
log(f"\nK* = {k_star} (VAL = {sweep[k_star]:.4f})")

# PHASE 3: greedy backward elimination
log(f"\nPhase 3: Greedy backward dari top-{k_star}...")
current = fi.head(k_star).index.tolist()
_, r2_cur = fit_eval(current)
tol_greedy = 0.0003
while len(current) > 3:
    best_drop, best_r2 = None, r2_cur - tol_greedy - 1e-9
    for feat in current:
        cand = [f for f in current if f != feat]
        _, r2_c = fit_eval(cand)
        if r2_c >= best_r2:
            best_r2, best_drop = r2_c, feat
    if best_drop is None:
        log(f"  Stop: tidak ada fitur yang bisa dibuang (tol {tol_greedy})")
        break
    current.remove(best_drop)
    r2_cur = best_r2
    log(f"  [{time.time()-t0:5.0f}s] Drop '{best_drop}': {len(current)} fitur, VAL = {r2_cur:.4f}")

pruned = current
log(f"\nFinal pruned: {len(pruned)} fitur")

# PHASE 4: TEST sekali — dengan params FAST dan TUNED
log("\nPhase 4: TEST (sekali), verifikasi dgn params tuned produksi...")
candidates = {
    'baseline_96': FEATS96,
    f'top_{k_star}': fi.head(k_star).index.tolist(),
    f'pruned_{len(pruned)}': pruned,
}
final = {}
for name, feats in candidates.items():
    m_f, r2_v = fit_eval(feats)
    r2_t_fast = r2_score(y_te, m_f.predict(X_te[feats].astype(float)))
    m_t = CatBoostRegressor(**CAT_TUNED)
    m_t.fit(df_tr.loc[mask_tr, feats].astype(float),
            df_tr.loc[mask_tr, TARGET].astype(float))
    r2_t_tuned = r2_score(y_te, m_t.predict(X_te[feats].astype(float)))
    final[name] = {'n': len(feats), 'val': r2_v, 'test_fast': r2_t_fast, 'test_tuned': r2_t_tuned}
    log(f"  {name:16s}: n={len(feats):3d}  VAL={r2_v:.4f}  TEST(fast)={r2_t_fast:.4f}  TEST(tuned)={r2_t_tuned:.4f}")

json.dump({'baseline_val': r2_val_base, 'sweep': sweep, 'k_star': k_star,
           'pruned_feats': pruned,
           'final': final},
          open('ghi_pruning_results.json', 'w'), indent=2)
log(f"\nSelesai {time.time()-t0:.0f}s — hasil di ghi_pruning_results.json")
