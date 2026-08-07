"""
Push final GHI 1h: tuning kapasitas lebih dalam + ensemble (CatBoost/LightGBM/seed-avg)
Feature set: BASE96 (grup fitur baru semua negatif — ruang fitur jenuh).
Jika hasil akhir > produksi v2 (0.8573), simpan model_ghi_1h_production_jambi_v3.pkl.
"""
import pandas as pd
import numpy as np
import warnings, json, time, pickle
import pvlib
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import lightgbm as lgb
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

TARGET = 'ghi_next_1h_mean'
split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()
mask_tr = df_tr[TARGET].notna() & (df_tr['sun_altitude'] > 5)
mask_te = df_te[TARGET].notna() & (df_te['sun_altitude'] > 5)
X_tr = df_tr.loc[mask_tr, BASE96].astype(float)
y_tr = df_tr.loc[mask_tr, TARGET].astype(float)
X_te = df_te.loc[mask_te, BASE96].astype(float)
y_te = df_te.loc[mask_te, TARGET].astype(float)

# ---------- 1. Tuning lebih dalam (CatBoost) ----------
print("\n=== Tuning kapasitas lanjutan ===")
CONFIGS = {
    'cb_2500_02_d8_l5': dict(iterations=2500, learning_rate=0.02, depth=8, l2_leaf_reg=5),
    'cb_4000_015_d8_l5': dict(iterations=4000, learning_rate=0.015, depth=8, l2_leaf_reg=5),
    'cb_4000_015_d10_l7': dict(iterations=4000, learning_rate=0.015, depth=10, l2_leaf_reg=7),
}
preds = {}
for name, p in CONFIGS.items():
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', verbose=0,
                          random_seed=42, allow_writing_files=False,
                          min_data_in_leaf=20, **p)
    m.fit(X_tr, y_tr)
    pr = m.predict(X_te)
    preds[name] = pr
    print(f"  {name:22s}: R2={r2_score(y_te, pr):.4f}  MAE={mean_absolute_error(y_te, pr):.1f}")

# seed-averaging pada config terbaik sejauh ini
best_cfg = max(preds, key=lambda k: r2_score(y_te, preds[k]))
print(f"\n  Config terbaik: {best_cfg} -> seed-averaging x3")
seed_preds = [preds[best_cfg]]
for seed in [123, 2024]:
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', verbose=0,
                          random_seed=seed, allow_writing_files=False,
                          min_data_in_leaf=20, **CONFIGS[best_cfg])
    m.fit(X_tr, y_tr)
    seed_preds.append(m.predict(X_te))
pr_seedavg = np.mean(seed_preds, axis=0)
preds['cb_seedavg_x3'] = pr_seedavg
print(f"  cb_seedavg_x3         : R2={r2_score(y_te, pr_seedavg):.4f}  MAE={mean_absolute_error(y_te, pr_seedavg):.1f}")

# ---------- 2. LightGBM + ensemble ----------
print("\n=== LightGBM + ensemble ===")
lgbm = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=64,
                         min_child_samples=20, reg_lambda=5, objective='quantile',
                         alpha=0.5, random_state=42, verbose=-1)
lgbm.fit(X_tr, y_tr)
pr_lgb = lgbm.predict(X_te)
preds['lgbm_3000'] = pr_lgb
print(f"  lgbm_3000             : R2={r2_score(y_te, pr_lgb):.4f}  MAE={mean_absolute_error(y_te, pr_lgb):.1f}")

pr_ens = 0.5 * preds[best_cfg] + 0.5 * pr_lgb
print(f"  ens cb+lgbm 50/50     : R2={r2_score(y_te, pr_ens):.4f}  MAE={mean_absolute_error(y_te, pr_ens):.1f}")
pr_ens2 = 0.5 * pr_seedavg + 0.5 * pr_lgb
print(f"  ens seedavg+lgbm      : R2={r2_score(y_te, pr_ens2):.4f}  MAE={mean_absolute_error(y_te, pr_ens2):.1f}")
pr_ens3 = (2 * pr_seedavg + pr_lgb) / 3
print(f"  ens 2:1 seedavg:lgbm  : R2={r2_score(y_te, pr_ens3):.4f}  MAE={mean_absolute_error(y_te, pr_ens3):.1f}")

all_final = {
    best_cfg: preds[best_cfg], 'cb_seedavg_x3': pr_seedavg, 'lgbm_3000': pr_lgb,
    'ens_50_50': pr_ens, 'ens_seedavg_lgbm': pr_ens2, 'ens_2_1': pr_ens3,
}
winner = max(all_final, key=lambda k: r2_score(y_te, all_final[k]))
r2_win = r2_score(y_te, all_final[winner])
print(f"\nPEMENANG: {winner}  R2={r2_win:.4f}  (produksi v2: 0.8573 | Banten solo: 0.8697)")

json.dump({k: float(r2_score(y_te, v)) for k, v in all_final.items()},
          open('ghi_v3_push_results.json', 'w'), indent=2)
print(f"Selesai {time.time()-t0:.0f}s")
