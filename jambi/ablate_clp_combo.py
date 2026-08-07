import pandas as pd
import numpy as np
import warnings, pickle
from sklearn.metrics import r2_score
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
df = add_delta_fast(df, 'cltt_mean', 3, ts_arr)

FEATS_V3 = [
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
]

COMBO_GROUPS = {
    'cler23 + clot_median': ['cler23_coverage', 'clot_median'],
    'cler23 + clot_median + cltt': ['cler23_coverage', 'clot_median', 'cltt_mean', 'cltt_std'],
    'cler23 + clot_median + cltt + derived': [
        'cler23_coverage', 'clot_median', 'cltt_mean', 'cltt_std',
        'cler23_coverage_delta30m', 'cltt_mean_delta30m'
    ],
    'cler23 + cltt + clth_stats': [
        'cler23_coverage', 'cltt_mean', 'cltt_std',
        'clth_mean', 'clth_std', 'clth_median'
    ],
    'all_new_clp_curated': [
        'cler23_coverage', 'clot_median', 'clot_coverage',
        'cltt_mean', 'cltt_std', 'cltt_median',
        'clth_mean', 'clth_std',
        'cler23_coverage_delta30m', 'cltt_mean_delta30m'
    ],
    'all_new_clp': [
        'clot_std', 'clot_median', 'clot_coverage',
        'cler23_coverage', 'cler23_std', 'cler23_mean',
        'clth_mean', 'clth_std', 'clth_median',
        'cltt_mean', 'cltt_std', 'cltt_median',
        'cloud_class_code_new',
        'cler23_coverage_delta30m', 'clot_std_delta30m', 'clot_std_roll30m'
    ],
}

CAT_PARAMS = dict(
    iterations=600, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, min_data_in_leaf=20,
    loss_function='Quantile:alpha=0.5',
    verbose=0, random_seed=42,
    allow_writing_files=False
)

split = int(len(df) * 0.8)
df_tr = df.iloc[:split].copy()
df_te = df.iloc[split:].copy()

print("=== Combination Ablation: v3 + New CLP Features ===\n")
baselines = {}

for horizon in ['kt_next_1h_mean', 'kt_next_2h_mean', 'kt_next_3h_mean']:
    mask_tr = df_tr[horizon].notna() & (df_tr['sun_altitude'] > 5)
    mask_te = df_te[horizon].notna() & (df_te['sun_altitude'] > 5)
    y_te = df_te.loc[mask_te, horizon].astype(float)

    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(df_tr.loc[mask_tr, FEATS_V3].astype(float),
          df_tr.loc[mask_tr, horizon].astype(float))
    r2_base = r2_score(y_te, m.predict(df_te.loc[mask_te, FEATS_V3].astype(float)))
    baselines[horizon] = r2_base
    print(f'{horizon} BASELINE (v3): {r2_base:.4f}')

    for gname, extra in COMBO_GROUPS.items():
        feats_plus = FEATS_V3 + extra
        m2 = CatBoostRegressor(**CAT_PARAMS)
        m2.fit(df_tr.loc[mask_tr, feats_plus].astype(float),
               df_tr.loc[mask_tr, horizon].astype(float))
        r2_p = r2_score(y_te, m2.predict(df_te.loc[mask_te, feats_plus].astype(float)))
        print(f'  + {gname}: {r2_p:.4f}  ({r2_p - r2_base:+.4f})')
    print()

# Find best combination
print("=== Summary: best combination per horizon ===")
for h, r2_b in baselines.items():
    print(f'{h}: baseline={r2_b:.4f}')
