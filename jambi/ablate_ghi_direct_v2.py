"""
Ablasi model GHI langsung (ghi_next_1h_mean, W/m2):
A. baseline produksi saat ini  : v2 (48) + ghi_clearsky = 49 fitur
B. + upgrade v3+v4             : beam_diffuse, cloud_top_trend, CLP stats
C. + future deterministic      : ghi_cs_t1..t6, sun_alt_t1..t6 (konvensi Banten/Kalbar)
D. + CLP temporal lags         : clot_median_lag10m dkk (gap-validated)
Grup diuji terpisah dan gabungan di atas baseline B.
"""
import pandas as pd
import numpy as np
import warnings, json, time
import pvlib
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

t0 = time.time()
LAT, LON, ELEV = -1.5833, 103.6667, 35.0   # Stasiun Klimatologi Jambi

df = pd.read_parquet('dfm_with_clp_stats.parquet')
df = df.sort_values('ts').reset_index(drop=True)
ts_arr = (df['ts'].values.astype('int64') / 1_000_000).astype(np.int64)
print(f"Loaded: {len(df)} rows")

# ---------- derived features v3/v4 (identik dgn build_model_v4.py) ----------
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

# ---------- C: future deterministic (konvensi Banten/Kalbar) ----------
print("Menghitung clearsky & sun_alt masa depan T+10..T+60 (pvlib Ineichen)...")
loc = pvlib.location.Location(latitude=LAT, longitude=LON, altitude=ELEV, tz='Asia/Jakarta')
ts_base = pd.DatetimeIndex(df['ts']).tz_localize('Asia/Jakarta')
for k in range(1, 7):
    ts_f = ts_base + pd.Timedelta(minutes=10 * k)
    sp = loc.get_solarposition(ts_f)
    cs = loc.get_clearsky(ts_f, model='ineichen')
    df[f'ghi_cs_t{k}'] = cs['ghi'].values
    df[f'sun_alt_t{k}'] = sp['apparent_elevation'].values
df['ghi_cs_next_mean'] = df[[f'ghi_cs_t{k}' for k in range(1, 7)]].mean(axis=1)
# smart-persistence eksplisit sbg fitur: kt_now x clearsky masa depan
df['ghi_persist_next'] = df['kt'] * df['ghi_cs_next_mean']

# ---------- D: CLP temporal lags (gap-validated) ----------
for n in [1, 2, 3]:
    df = add_lag_fast(df, 'clot_median', n, ts_arr)      # lag10m/20m/30m
df = add_delta_fast(df, 'clot_median', 1, ts_arr)        # delta10m
df = add_delta_fast(df, 'clot_median', 3, ts_arr)        # delta30m
df['clot_median_roll30m'] = df['clot_median'].rolling(4, min_periods=2).mean()
for n in [1, 2]:
    df = add_lag_fast(df, 'cler23_coverage', n, ts_arr)  # lag10m/20m
df = add_delta_fast(df, 'cler23_coverage', 1, ts_arr)    # delta10m

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

A_BASELINE = FEATS_V2 + ['ghi_clearsky']                      # 49 (= produksi saat ini)
B_V4       = A_BASELINE + V3_NEW + V4_CLP                     # 73

CANDIDATES = {
    'A_prod_now (49)':        A_BASELINE,
    'B_+v3v4 (73)':           B_V4,
    'B+future_cs (87)':       B_V4 + FUTURE_CS,
    'B+clp_lags (82)':        B_V4 + CLP_LAGS,
    'B+future+clp_lags (96)': B_V4 + FUTURE_CS + CLP_LAGS,
}

TARGET = 'ghi_next_1h_mean'
CAT_PARAMS = dict(
    iterations=600, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, min_data_in_leaf=20,
    loss_function='Quantile:alpha=0.5',
    verbose=0, random_seed=42, allow_writing_files=False
)

split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()
mask_tr = df_tr[TARGET].notna() & (df_tr['sun_altitude'] > 5)
mask_te = df_te[TARGET].notna() & (df_te['sun_altitude'] > 5)
y_te = df_te.loc[mask_te, TARGET].astype(float)
print(f"Train n={mask_tr.sum()}, test n={mask_te.sum()} (test mulai {df_te['ts'].min()})\n")

# smart-persistence baseline
sp_pred = (df_te.loc[mask_te, 'kt'] * df_te.loc[mask_te, 'ghi_cs_next_mean']).astype(float)
sp_ok = sp_pred.notna()
r2_sp = r2_score(y_te[sp_ok], sp_pred[sp_ok])
print(f"Smart-persistence (kt_now x cs_next_mean): R2 = {r2_sp:.4f}\n")

print("=== Ablasi GHI langsung 1 jam ke depan ===")
results = {}
best_name, best_r2, best_model, best_feats = None, -9, None, None
for name, feats in CANDIDATES.items():
    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(df_tr.loc[mask_tr, feats].astype(float),
          df_tr.loc[mask_tr, TARGET].astype(float))
    pred = m.predict(df_te.loc[mask_te, feats].astype(float))
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    skill = 1 - ((y_te - pred) ** 2).mean() / ((y_te[sp_ok] - sp_pred[sp_ok]) ** 2).mean()
    results[name] = {'n_feats': len(feats), 'r2': r2, 'mae': mae, 'skill_vs_sp': skill}
    print(f"  {name:28s}: R2={r2:.4f}  MAE={mae:5.1f} W/m2  skill_vs_SP={skill:+.3f}")
    if r2 > best_r2:
        best_name, best_r2, best_model, best_feats = name, r2, m, feats

print(f"\nTerbaik: {best_name} (R2={best_r2:.4f})")

# feature importance kandidat terbaik
fi = pd.Series(best_model.get_feature_importance(), index=best_feats).sort_values(ascending=False)
print(f"\nTop 25 feature importance ({best_name}):")
newset = set(FUTURE_CS + CLP_LAGS + V4_CLP + V3_NEW)
for feat, imp in fi.head(25).items():
    tag = '  <-- baru' if feat in newset else ''
    print(f"  {imp:6.2f}  {feat}{tag}")

with open('ghi_direct_ablation_results.json', 'w') as f:
    json.dump({'smart_persistence_r2': r2_sp, 'results': results,
               'best': best_name}, f, indent=2)
print(f"\nSelesai {time.time()-t0:.0f}s — hasil di ghi_direct_ablation_results.json")
