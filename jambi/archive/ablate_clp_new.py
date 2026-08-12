import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

# Load pre-merged dataset
df = pd.read_parquet('dfm_with_clp_stats.parquet')
df = df.sort_values('ts').reset_index(drop=True)
ts_arr = (df['ts'].values.astype('int64') / 1_000_000).astype(np.int64)
print(f"Loaded: {len(df)} rows, {len(df.columns)} cols")

def add_lag_delta(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90):
    val = df[col].values.astype(float)
    lag_vals = np.full(len(df), np.nan)
    for i in range(n_steps, len(df)):
        gap = ts_arr[i] - ts_arr[i - n_steps]
        expected = n_steps * step_sec
        if abs(gap - expected) <= tol_sec:
            lag_vals[i] = val[i - n_steps]
    suffix = str(n_steps * 10) + 'm'
    df[col + '_lag' + suffix] = lag_vals
    df[col + '_delta' + suffix] = val - lag_vals
    return df

# Build cloud top trends (30m and 60m) using proper vectorized gap check
def add_delta_fast(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90):
    val = df[col].values.astype(float)
    lag_vals = np.full(len(df), np.nan)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n-n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lag_arr = np.full(n, np.nan)
    lag_arr[n_steps:][valid] = val[:n-n_steps][valid]
    suffix = str(n_steps * 10) + 'm'
    new_col = col + '_delta' + suffix
    df[new_col] = val - lag_arr
    return df

# Build all needed derived features
print("Building derived features...")

# cloud_top trends 30m/60m (needed for v3 baseline)
for var in ['cloud_top_height_m', 'cloud_top_temp_c', 'cloud_eff_radius_um']:
    df = add_delta_fast(df, var, 3, ts_arr)  # -> _delta30m
    df = add_delta_fast(df, var, 6, ts_arr)  # -> _delta60m

# beam_diffuse (needed for v3 baseline)
df['dni_ratio'] = df['dni_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df['dhi_fraction'] = df['dhi_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df.loc[df['ghi_consolidated'] < 10, ['dni_ratio', 'dhi_fraction']] = np.nan

# New CLP derived
df['clot_std_roll30m'] = df['clot_std'].rolling(4, min_periods=2).mean()
df = add_delta_fast(df, 'clot_std', 3, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 3, ts_arr)

# Verify columns exist
needed = ['cloud_top_height_m_delta30m', 'cloud_top_height_m_delta60m',
          'cloud_top_temp_c_delta30m', 'cloud_top_temp_c_delta60m',
          'cloud_eff_radius_um_delta30m', 'cloud_eff_radius_um_delta60m',
          'dni_ratio', 'dhi_fraction',
          'clot_std', 'cler23_coverage', 'clot_median', 'clth_mean', 'clth_std', 'cltt_mean', 'cltt_std']
missing = [c for c in needed if c not in df.columns]
if missing:
    print("MISSING COLUMNS:", missing)
else:
    print("All required columns present.")

# Feature sets
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
    # v3 new
    'dni_ratio', 'dhi_fraction',
    'cloud_top_height_m_delta30m', 'cloud_top_temp_c_delta30m', 'cloud_eff_radius_um_delta30m',
    'cloud_top_height_m_delta60m', 'cloud_top_temp_c_delta60m', 'cloud_eff_radius_um_delta60m',
]

NEW_GROUPS = {
    'cler23_coverage': ['cler23_coverage'],
    'clot_std': ['clot_std'],
    'clot_median': ['clot_median'],
    'clth_mean': ['clth_mean'],
    'clth_stats': ['clth_mean', 'clth_std', 'clth_median'],
    'cltt_stats': ['cltt_mean', 'cltt_std'],
    'cler23_derived': ['cler23_coverage', 'cler23_coverage_delta30m'],
    'clot_derived': ['clot_std', 'clot_std_delta30m', 'clot_std_roll30m'],
    'top_candidates': ['clot_std', 'cler23_coverage', 'clot_median',
                       'cler23_coverage_delta30m', 'clot_std_delta30m'],
    'all_new_clp': ['clot_std', 'clot_median', 'clot_coverage',
                    'cler23_coverage', 'cler23_std', 'cler23_mean',
                    'clth_mean', 'clth_std', 'clth_median',
                    'cltt_mean', 'cltt_std', 'cltt_median',
                    'cloud_class_code_new',
                    'cler23_coverage_delta30m', 'clot_std_delta30m', 'clot_std_roll30m'],
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

print()
print("=== Ablation Results ===")

import json
results = {}

for horizon in ['kt_next_1h_mean', 'kt_next_2h_mean', 'kt_next_3h_mean']:
    results[horizon] = {}
    mask_tr = df_tr[horizon].notna() & (df_tr['sun_altitude'] > 5)
    mask_te = df_te[horizon].notna() & (df_te['sun_altitude'] > 5)
    y_te = df_te.loc[mask_te, horizon].astype(float)

    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(df_tr.loc[mask_tr, FEATS_V3].astype(float),
          df_tr.loc[mask_tr, horizon].astype(float))
    r2_base = r2_score(y_te, m.predict(df_te.loc[mask_te, FEATS_V3].astype(float)))
    results[horizon]['baseline_v3'] = r2_base
    print(f'{horizon} BASELINE v3: {r2_base:.4f}  (n_tr={mask_tr.sum()}, n_te={mask_te.sum()})')

    for gname, extra in NEW_GROUPS.items():
        feats_plus = FEATS_V3 + extra
        m2 = CatBoostRegressor(**CAT_PARAMS)
        m2.fit(df_tr.loc[mask_tr, feats_plus].astype(float),
               df_tr.loc[mask_tr, horizon].astype(float))
        r2_p = r2_score(y_te, m2.predict(df_te.loc[mask_te, feats_plus].astype(float)))
        results[horizon][gname] = r2_p
        print(f'  + {gname}: {r2_p:.4f}  ({r2_p - r2_base:+.4f})')
    print()

with open('clp_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to clp_ablation_results.json")
