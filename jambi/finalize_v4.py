"""Finalize and evaluate model v4 with rolling backtest + coverage."""
import pandas as pd, numpy as np, warnings, pickle, json
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

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

for var in ['cloud_top_height_m', 'cloud_top_temp_c', 'cloud_eff_radius_um']:
    df = add_delta_fast(df, var, 3, ts_arr)
    df = add_delta_fast(df, var, 6, ts_arr)
df['dni_ratio'] = df['dni_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df['dhi_fraction'] = df['dhi_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df.loc[df['ghi_consolidated'] < 10, ['dni_ratio', 'dhi_fraction']] = np.nan
df['clot_std_roll30m'] = df['clot_std'].rolling(4, min_periods=2).mean()
df = add_delta_fast(df, 'clot_std', 3, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 3, ts_arr)

FEATS_V4 = [
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
    'dni_ratio', 'dhi_fraction',
    'cloud_top_height_m_delta30m', 'cloud_top_temp_c_delta30m', 'cloud_eff_radius_um_delta30m',
    'cloud_top_height_m_delta60m', 'cloud_top_temp_c_delta60m', 'cloud_eff_radius_um_delta60m',
    'clot_std', 'clot_median', 'clot_coverage',
    'cler23_coverage', 'cler23_std', 'cler23_mean',
    'clth_mean', 'clth_std', 'clth_median',
    'cltt_mean', 'cltt_std', 'cltt_median',
    'cloud_class_code_new',
    'cler23_coverage_delta30m', 'clot_std_delta30m', 'clot_std_roll30m',
]

class RCC:
    def __init__(self, window_days=60, alpha=0.2):
        self.window_days = window_days; self.alpha = alpha
        self.buf_ts = []; self.buf_lo = []; self.buf_hi = []; self.buf_y = []
        self.correction = 0.0
    def update(self, ts, lo, hi, y):
        self.buf_ts.extend(ts); self.buf_lo.extend(lo)
        self.buf_hi.extend(hi); self.buf_y.extend(y)
        if not self.buf_ts: return
        cutoff = max(self.buf_ts) - pd.Timedelta(days=self.window_days)
        keep = [i for i, t in enumerate(self.buf_ts) if t >= cutoff]
        self.buf_ts = [self.buf_ts[i] for i in keep]
        self.buf_lo = [self.buf_lo[i] for i in keep]
        self.buf_hi = [self.buf_hi[i] for i in keep]
        self.buf_y = [self.buf_y[i] for i in keep]
    def recalibrate(self):
        if len(self.buf_y) < 30: return self.correction
        scores = [max(lo - y, y - hi) for lo, hi, y in zip(self.buf_lo, self.buf_hi, self.buf_y)]
        n = len(scores)
        q = min(np.ceil((n+1) * 0.8) / n, 1.0)
        self.correction = float(np.quantile(scores, q))
        return self.correction
    def apply(self, lo, hi):
        return np.asarray(lo) - self.correction, np.asarray(hi) + self.correction

split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()
cal_size = int(len(df_tr) * 0.2)
df_cal = df_tr.iloc[-cal_size:].copy()

def make_cat(alpha):
    return CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        min_data_in_leaf=20, loss_function=f'Quantile:alpha={alpha}',
        verbose=0, random_seed=42, allow_writing_files=False
    )

def rolling_coverage(df_test, models, feats, horizon, alpha=0.2, window_days=60):
    mask = df_test[horizon].notna() & (df_test['sun_altitude'] > 5)
    df_h = df_test[mask].copy()
    X = df_h[feats].astype(float)
    y = df_h[horizon].values
    ts = df_h['ts'].values
    ts_pd = pd.to_datetime(ts)
    p_lo = models[0.1].predict(X)
    p_hi = models[0.9].predict(X)
    p_med = models[0.5].predict(X)
    months = ts_pd.to_period('M').unique()
    corr = 0.0
    adj_lo, adj_hi = p_lo.copy(), p_hi.copy()
    buf_lo, buf_hi, buf_y, buf_ts = [], [], [], []
    for idx, month in enumerate(months):
        m_mask = ts_pd.to_period('M') == month
        if idx > 0:
            prev = ts_pd.to_period('M') == months[idx-1]
            buf_lo.extend(p_lo[prev].tolist()); buf_hi.extend(p_hi[prev].tolist())
            buf_y.extend(y[prev].tolist()); buf_ts.extend(ts[prev].tolist())
            if buf_ts:
                cutoff = ts[m_mask][0] - np.timedelta64(window_days * 86400, 's')
                keep = [i for i, t in enumerate(buf_ts) if t >= cutoff]
                buf_lo = [buf_lo[i] for i in keep]; buf_hi = [buf_hi[i] for i in keep]
                buf_y = [buf_y[i] for i in keep]; buf_ts = [buf_ts[i] for i in keep]
            if len(buf_y) >= 30:
                scores = [max(lo-yy, yy-hi) for lo, hi, yy in zip(buf_lo, buf_hi, buf_y)]
                n = len(scores)
                corr = float(np.quantile(scores, min(np.ceil((n+1)*0.8)/n, 1.0)))
        adj_lo[m_mask] = p_lo[m_mask] - corr
        adj_hi[m_mask] = p_hi[m_mask] + corr
    cov = np.mean((y >= adj_lo) & (y <= adj_hi))
    r2 = r2_score(y, p_med)
    mae = mean_absolute_error(y, p_med)
    iw = np.mean(adj_hi - adj_lo)
    return r2, cov, mae, iw

# Train v4
print("Training model v4 (72 features, 3 quantiles, 3 horizons)...")
all_results = {}
for horizon in ['kt_next_1h_mean', 'kt_next_2h_mean', 'kt_next_3h_mean']:
    print(f"  {horizon}...", end=' ', flush=True)
    mask_tr = df_tr[horizon].notna() & (df_tr['sun_altitude'] > 5)
    mask_cal = df_cal[horizon].notna() & (df_cal['sun_altitude'] > 5)
    models = {a: make_cat(a) for a in [0.1, 0.5, 0.9]}
    for a, m in models.items():
        m.fit(df_tr.loc[mask_tr, FEATS_V4].astype(float), df_tr.loc[mask_tr, horizon].astype(float))
    # calibration
    X_cal = df_cal.loc[mask_cal, FEATS_V4].astype(float)
    cal = RCC()
    cal.update(df_cal.loc[mask_cal, 'ts'].tolist(),
               list(models[0.1].predict(X_cal)), list(models[0.9].predict(X_cal)),
               list(df_cal.loc[mask_cal, horizon].values))
    cal.recalibrate()
    all_results[horizon] = {
        'models': models, 'features': FEATS_V4, 'correction': cal.correction,
        'buffer_ts': cal.buf_ts, 'buffer_p_lo': cal.buf_lo,
        'buffer_p_hi': cal.buf_hi, 'buffer_y': cal.buf_y,
    }
    print("done.")

# Load v2, v3 for comparison
with open('model_hourly_production_jambi_v2.pkl', 'rb') as f: v2 = pickle.load(f)
with open('model_hourly_production_jambi_v3.pkl', 'rb') as f: v3 = pickle.load(f)

print()
print("=== Full Rolling Backtest: v2 vs v3 vs v4 ===")
print("%-12s  %8s %8s  %8s %8s  %8s %8s  %8s %8s" % (
    'Horizon', 'v2 R2', 'v2 Cov', 'v3 R2', 'v3 Cov', 'v4 R2', 'v4 Cov', 'v4-v2', 'v4-v3'))
print("-" * 100)

metrics = {}
for horizon in ['kt_next_1h_mean', 'kt_next_2h_mean', 'kt_next_3h_mean']:
    r2v2, cv2, _, _ = rolling_coverage(df_te, v2[horizon]['models'], v2[horizon]['features'], horizon)
    r2v3, cv3, _, _ = rolling_coverage(df_te, v3[horizon]['models'], v3[horizon]['features'], horizon)
    r2v4, cv4, mae4, iw4 = rolling_coverage(df_te, all_results[horizon]['models'], FEATS_V4, horizon)
    h_l = horizon.replace('kt_next_', 'jam ke-').replace('_mean', '')
    print("%-12s  %8.4f %8.3f  %8.4f %8.3f  %8.4f %8.3f  %+8.4f %+8.4f" % (
        h_l, r2v2, cv2, r2v3, cv3, r2v4, cv4, r2v4 - r2v2, r2v4 - r2v3))
    metrics[horizon] = {'r2v4': r2v4, 'cov_80': cv4, 'mae': mae4, 'iw': iw4,
                        'r2v3': r2v3, 'r2v2': r2v2}

# Save
with open('model_hourly_production_jambi_v4.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Save metrics
with open('model_v4_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model_hourly_production_jambi_v4.pkl")
print(f"Features: {len(FEATS_V4)}")
print("\nFeature highlights (v4 new features vs v3):")
new_in_v4 = ['clot_std','clot_median','clot_coverage',
             'cler23_coverage','cler23_std','cler23_mean',
             'clth_mean','clth_std','clth_median',
             'cltt_mean','cltt_std','cltt_median',
             'cloud_class_code_new',
             'cler23_coverage_delta30m','clot_std_delta30m','clot_std_roll30m']
print(f"  New CLP stats: {len(new_in_v4)} features")
print(f"  Total v3: 56 features")
print(f"  Total v4: {len(FEATS_V4)} features (= v3 + {len(new_in_v4)} CLP stats)")
