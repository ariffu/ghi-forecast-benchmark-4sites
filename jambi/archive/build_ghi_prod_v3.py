"""
Produksi GHI 1h v3:
- P50 = seed-average 3x CatBoost tuned (4000it, lr0.015, depth8, l2=5)
- P10/P90 = CatBoost tuned single seed
- RCC rolling backtest + buffer awal, simpan model_ghi_1h_production_jambi_v3.pkl
"""
import pandas as pd
import numpy as np
import warnings, time, pickle, sys
import pvlib
from sklearn.metrics import r2_score, mean_absolute_error
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
FEATS = FEATS_V2 + ['ghi_clearsky'] + V3_NEW + V4_CLP + FUTURE_CS + CLP_LAGS

TARGET = 'ghi_next_1h_mean'
split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()
cal_size = int(len(df_tr) * 0.2)
df_cal = df_tr.iloc[-cal_size:].copy()
mask_tr = df_tr[TARGET].notna() & (df_tr['sun_altitude'] > 5)
mask_cal = df_cal[TARGET].notna() & (df_cal['sun_altitude'] > 5)
mask_te = df_te[TARGET].notna() & (df_te['sun_altitude'] > 5)
X_tr = df_tr.loc[mask_tr, FEATS].astype(float)
y_tr = df_tr.loc[mask_tr, TARGET].astype(float)

TUNED = dict(iterations=4000, learning_rate=0.015, depth=8, l2_leaf_reg=5,
             min_data_in_leaf=20, verbose=0, allow_writing_files=False)

log(f"Training P50 seed-avg x3 + P10/P90 ({len(FEATS)} fitur)...")
p50_models = []
for seed in [42, 123, 2024]:
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', random_seed=seed, **TUNED)
    m.fit(X_tr, y_tr)
    p50_models.append(m)
    log(f"  P50 seed {seed} done ({time.time()-t0:.0f}s)")
m_p10 = CatBoostRegressor(loss_function='Quantile:alpha=0.1', random_seed=42, **TUNED)
m_p10.fit(X_tr, y_tr); log(f"  P10 done ({time.time()-t0:.0f}s)")
m_p90 = CatBoostRegressor(loss_function='Quantile:alpha=0.9', random_seed=42, **TUNED)
m_p90.fit(X_tr, y_tr); log(f"  P90 done ({time.time()-t0:.0f}s)")

def p50_predict(X):
    return np.mean([m.predict(X) for m in p50_models], axis=0)

# rolling backtest with RCC
d = df_te[mask_te]
X_te = d[FEATS].astype(float)
y_te = d[TARGET].values
ts_te = d['ts'].values
ts_pd = pd.to_datetime(ts_te)
p_lo, p_hi, p_med = m_p10.predict(X_te), m_p90.predict(X_te), p50_predict(X_te)
months = ts_pd.to_period('M').unique()
corr = 0.0
adj_lo, adj_hi = p_lo.copy(), p_hi.copy()
buf_lo, buf_hi, buf_y, buf_ts = [], [], [], []
for idx, month in enumerate(months):
    m_mask = ts_pd.to_period('M') == month
    if idx > 0:
        prev = ts_pd.to_period('M') == months[idx - 1]
        buf_lo += p_lo[prev].tolist(); buf_hi += p_hi[prev].tolist()
        buf_y += y_te[prev].tolist(); buf_ts += ts_te[prev].tolist()
        cutoff = ts_te[m_mask][0] - np.timedelta64(60 * 86400, 's')
        keep = [i for i, t in enumerate(buf_ts) if t >= cutoff]
        buf_lo = [buf_lo[i] for i in keep]; buf_hi = [buf_hi[i] for i in keep]
        buf_y = [buf_y[i] for i in keep]; buf_ts = [buf_ts[i] for i in keep]
        if len(buf_y) >= 30:
            scores = [max(lo - yy, yy - hi) for lo, hi, yy in zip(buf_lo, buf_hi, buf_y)]
            n = len(scores)
            corr = float(np.quantile(scores, min(np.ceil((n + 1) * 0.8) / n, 1.0)))
    adj_lo[m_mask] = p_lo[m_mask] - corr
    adj_hi[m_mask] = p_hi[m_mask] + corr

r2 = r2_score(y_te, p_med)
mae = mean_absolute_error(y_te, p_med)
rmse = float(np.sqrt(((y_te - p_med) ** 2).mean()))
cov = float(np.mean((y_te >= adj_lo) & (y_te <= adj_hi)))
iw = float(np.mean(adj_hi - adj_lo))

sp = d['ghi_persist_next'].astype(float)
ok = sp.notna().values
mse_sp = ((y_te[ok] - sp.values[ok]) ** 2).mean()
skill = 1 - ((y_te - p_med) ** 2).mean() / mse_sp

log("")
log(f"R2={r2:.4f}  MAE={mae:.1f}  RMSE={rmse:.1f}  Cov80={cov:.3f}  IW={iw:.1f}  skill_vs_SP={skill:+.3f}")
log(f"Referensi: Kalbar avg60m ensemble 0.8606 | Banten hybrid 0.86 | Banten ensemble-149f 0.8716")

# RCC initial buffer from cal window
X_cal = df_cal.loc[mask_cal, FEATS].astype(float)
p_lo_c, p_hi_c = m_p10.predict(X_cal), m_p90.predict(X_cal)
y_cal = df_cal.loc[mask_cal, TARGET].values
ts_cal = df_cal.loc[mask_cal, 'ts']
keep = ts_cal >= (ts_cal.max() - pd.Timedelta(days=60))
scores = np.maximum(p_lo_c[keep.values] - y_cal[keep.values], y_cal[keep.values] - p_hi_c[keep.values])
n = len(scores)
correction = float(np.quantile(scores, min(np.ceil((n + 1) * 0.8) / n, 1.0)))

prod = {
    'models': {0.1: m_p10, 0.5: p50_models, 0.9: m_p90},
    'p50_is_seed_ensemble': True,
    'features': FEATS, 'correction': correction,
    'buffer_ts': ts_cal[keep].tolist(),
    'buffer_p_lo': p_lo_c[keep.values].tolist(),
    'buffer_p_hi': p_hi_c[keep.values].tolist(),
    'buffer_y': y_cal[keep.values].tolist(),
    'target_definition': 'ghi_next_1h_mean = mean(GHI T+10m..T+60m), W/m2',
    'notes': ('v3: 96 fitur (v4-kt + ghi_clearsky + future cs/sun_alt + CLP lags), '
              'CatBoost tuned 4000it/lr0.015/d8/l2=5, P50 = seed-average 3x. '
              'P50 inference: mean([m.predict(X) for m in models[0.5]])'),
    'metric_r2': r2, 'metric_mae_wm2': mae, 'metric_rmse_wm2': rmse,
    'metric_coverage80': cov, 'metric_skill_vs_smartpersist': skill,
}
with open('model_ghi_1h_production_jambi_v3.pkl', 'wb') as f:
    pickle.dump(prod, f)
log(f"\nSaved model_ghi_1h_production_jambi_v3.pkl  (total {time.time()-t0:.0f}s)")
