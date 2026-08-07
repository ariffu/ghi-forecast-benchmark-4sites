"""
Finalisasi model GHI langsung v2:
1. Multi-seed check: B+future_cs vs B+future+clp_lags (apakah CLP lags robust?)
2. Bangun model produksi P10/P50/P90 + RCC, rolling backtest coverage
3. Bandingkan vs produksi lama (model_ghi_1h_production_jambi.pkl) di protokol sama
"""
import pandas as pd
import numpy as np
import warnings, json, time, pickle
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

def add_lag_fast(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90):
    val = df[col].values.astype(float)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n - n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lag_arr = np.full(n, np.nan)
    lag_arr[n_steps:][valid] = val[:n - n_steps][valid]
    df[col + '_lag' + str(n_steps * 10) + 'm'] = lag_arr
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

B_V4 = FEATS_V2 + ['ghi_clearsky'] + V3_NEW + V4_CLP
CAND_FUT = B_V4 + FUTURE_CS
CAND_FULL = B_V4 + FUTURE_CS + CLP_LAGS

TARGET = 'ghi_next_1h_mean'
split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()
cal_size = int(len(df_tr) * 0.2)
df_cal = df_tr.iloc[-cal_size:].copy()
mask_tr = df_tr[TARGET].notna() & (df_tr['sun_altitude'] > 5)
mask_te = df_te[TARGET].notna() & (df_te['sun_altitude'] > 5)
y_te = df_te.loc[mask_te, TARGET].astype(float)

# ---------- 1. Multi-seed robustness: apakah CLP lags nyata? ----------
print("\n=== Multi-seed check (3 seeds) ===")
for name, feats in [('B+future_cs (87)', CAND_FUT), ('B+future+clp_lags (96)', CAND_FULL)]:
    r2s = []
    for seed in [42, 123, 2024]:
        m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6,
                              l2_leaf_reg=3, min_data_in_leaf=20,
                              loss_function='Quantile:alpha=0.5',
                              verbose=0, random_seed=seed, allow_writing_files=False)
        m.fit(df_tr.loc[mask_tr, feats].astype(float),
              df_tr.loc[mask_tr, TARGET].astype(float))
        r2s.append(r2_score(y_te, m.predict(df_te.loc[mask_te, feats].astype(float))))
    print(f"  {name:28s}: R2 = {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}  ({['%.4f'%r for r in r2s]})")

# ---------- 2. Model produksi kandidat final + RCC rolling backtest ----------
def make_cat(alpha, seed=42):
    return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6,
                             l2_leaf_reg=3, min_data_in_leaf=20,
                             loss_function=f'Quantile:alpha={alpha}',
                             verbose=0, random_seed=seed, allow_writing_files=False)

def rolling_eval(df_test, models, feats, target, alpha=0.2, window_days=60):
    mask = df_test[target].notna() & (df_test['sun_altitude'] > 5)
    d = df_test[mask]
    X = d[feats].astype(float)
    y = d[target].values
    ts = d['ts'].values
    ts_pd = pd.to_datetime(ts)
    p_lo, p_hi, p_med = models[0.1].predict(X), models[0.9].predict(X), models[0.5].predict(X)
    months = ts_pd.to_period('M').unique()
    corr = 0.0
    adj_lo, adj_hi = p_lo.copy(), p_hi.copy()
    buf_lo, buf_hi, buf_y, buf_ts = [], [], [], []
    for idx, month in enumerate(months):
        m_mask = ts_pd.to_period('M') == month
        if idx > 0:
            prev = ts_pd.to_period('M') == months[idx - 1]
            buf_lo += p_lo[prev].tolist(); buf_hi += p_hi[prev].tolist()
            buf_y += y[prev].tolist(); buf_ts += ts[prev].tolist()
            cutoff = ts[m_mask][0] - np.timedelta64(window_days * 86400, 's')
            keep = [i for i, t in enumerate(buf_ts) if t >= cutoff]
            buf_lo = [buf_lo[i] for i in keep]; buf_hi = [buf_hi[i] for i in keep]
            buf_y = [buf_y[i] for i in keep]; buf_ts = [buf_ts[i] for i in keep]
            if len(buf_y) >= 30:
                scores = [max(lo - yy, yy - hi) for lo, hi, yy in zip(buf_lo, buf_hi, buf_y)]
                n = len(scores)
                corr = float(np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))
        adj_lo[m_mask] = p_lo[m_mask] - corr
        adj_hi[m_mask] = p_hi[m_mask] + corr
    cov = float(np.mean((y >= adj_lo) & (y <= adj_hi)))
    return (r2_score(y, p_med), cov, mean_absolute_error(y, p_med),
            float(np.sqrt(((y - p_med) ** 2).mean())), float(np.mean(adj_hi - adj_lo)))

FINAL_FEATS = CAND_FULL
print(f"\n=== Training produksi GHI v2 ({len(FINAL_FEATS)} fitur, P10/P50/P90) ===")
mask_cal = df_cal[TARGET].notna() & (df_cal['sun_altitude'] > 5)
models = {}
for a in [0.1, 0.5, 0.9]:
    models[a] = make_cat(a)
    models[a].fit(df_tr.loc[mask_tr, FINAL_FEATS].astype(float),
                  df_tr.loc[mask_tr, TARGET].astype(float))

r2, cov, mae, rmse, iw = rolling_eval(df_te, models, FINAL_FEATS, TARGET)

# produksi lama di protokol yang sama
with open('model_ghi_1h_production_jambi.pkl', 'rb') as f:
    old = pickle.load(f)
r2o, covo, maeo, rmseo, iwo = rolling_eval(df_te, old['models'], old['features'], TARGET)

# smart-persistence
mask_sp = mask_te & df_te['ghi_persist_next'].notna()
sp = df_te.loc[mask_sp, 'ghi_persist_next'].astype(float)
ysp = df_te.loc[mask_sp, TARGET].astype(float)
r2_sp = r2_score(ysp, sp)
mse_sp = ((ysp - sp) ** 2).mean()

d = df_te[mask_te]
pred_new = models[0.5].predict(d[FINAL_FEATS].astype(float))
skill_new = 1 - ((d[TARGET].values - pred_new) ** 2).mean() / mse_sp
pred_old = old['models'][0.5].predict(d[old['features']].astype(float))
skill_old = 1 - ((d[TARGET].values - pred_old) ** 2).mean() / mse_sp

print("\n%-24s %8s %8s %8s %8s %10s %10s" % ('Model', 'R2', 'MAE', 'RMSE', 'Cov80', 'IntWidth', 'SkillSP'))
print("-" * 84)
print("%-24s %8.4f %8.1f %8.1f %8.3f %10.1f %+10.3f" % ('produksi lama (49f)', r2o, maeo, rmseo, covo, iwo, skill_old))
print("%-24s %8.4f %8.1f %8.1f %8.3f %10.1f %+10.3f" % ('produksi baru (96f)', r2, mae, rmse, cov, iw, skill_new))
print("%-24s %8.4f" % ('smart-persistence', r2_sp))

# ---------- 3. RCC initial buffer + save ----------
X_cal = df_cal.loc[mask_cal, FINAL_FEATS].astype(float)
p_lo_c = models[0.1].predict(X_cal)
p_hi_c = models[0.9].predict(X_cal)
y_cal = df_cal.loc[mask_cal, TARGET].values
ts_cal = df_cal.loc[mask_cal, 'ts']
cut = ts_cal.max() - pd.Timedelta(days=60)
keep = ts_cal >= cut
scores = np.maximum(p_lo_c[keep.values] - y_cal[keep.values], y_cal[keep.values] - p_hi_c[keep.values])
n = len(scores)
correction = float(np.quantile(scores, min(np.ceil((n + 1) * 0.8) / n, 1.0)))

prod = {
    'models': models, 'features': FINAL_FEATS, 'correction': correction,
    'buffer_ts': ts_cal[keep].tolist(),
    'buffer_p_lo': p_lo_c[keep.values].tolist(),
    'buffer_p_hi': p_hi_c[keep.values].tolist(),
    'buffer_y': y_cal[keep.values].tolist(),
    'target_definition': 'ghi_next_1h_mean = mean(GHI T+10m..T+60m), W/m2',
    'notes': 'v2: fitur v4-kt + ghi_clearsky + future cs/sun_alt T+10..T+60 (konvensi Banten/Kalbar) + CLP temporal lags',
    'metric_r2': r2, 'metric_mae_wm2': mae, 'metric_rmse_wm2': rmse,
    'metric_coverage80': cov, 'metric_skill_vs_smartpersist': skill_new,
}
with open('model_ghi_1h_production_jambi_v2.pkl', 'wb') as f:
    pickle.dump(prod, f)
print(f"\nSaved model_ghi_1h_production_jambi_v2.pkl ({len(FINAL_FEATS)} fitur, correction={correction:.1f} W/m2)")

fi = pd.Series(models[0.5].get_feature_importance(), index=FINAL_FEATS).sort_values(ascending=False)
print("\nTop 15 importance (P50):")
for feat, imp in fi.head(15).items():
    print(f"  {imp:6.2f}  {feat}")

print(f"\nTotal {time.time()-t0:.0f}s")
