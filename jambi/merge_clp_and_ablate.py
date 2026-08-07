import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

print("=== PHASE 1: Merge CLP stats into training data ===")

df_train = pd.read_parquet('dfm_banten_features.parquet')
df_train = df_train.sort_values('ts').reset_index(drop=True)

df_clp = pd.read_parquet('clp_stats_jambi.parquet')
df_clp['ts'] = pd.to_datetime(df_clp['timestamp'])
df_clp = df_clp.drop(columns=['timestamp']).sort_values('ts').reset_index(drop=True)

# Rename to avoid collision with existing cols
rename_map = {
    'CLOT_std': 'clot_std',
    'CLOT_median': 'clot_median',
    'CLOT_coverage': 'clot_coverage',
    'CLER_23_coverage': 'cler23_coverage',
    'CLER_23_std': 'cler23_std',
    'CLER_23_mean': 'cler23_mean',
    'CLTH_std': 'clth_std',
    'CLTH_median': 'clth_median',
    'CLTT_mean': 'cltt_mean',
    'CLTT_std': 'cltt_std',
    'CLTT_median': 'cltt_median',
    'CLOT_mean': 'clot_mean_check',
    'CLTH_mean': 'clth_mean',
    'cloud_class_code': 'cloud_class_code_new',
    'cloud_present': 'cloud_present_new',
}
df_clp = df_clp.rename(columns=rename_map)

# Merge on ts
new_cols = ['ts','clot_std','clot_median','clot_coverage',
            'cler23_coverage','cler23_std','cler23_mean',
            'clth_mean','clth_std','clth_median',
            'cltt_mean','cltt_std','cltt_median',
            'cloud_class_code_new']
df_merged = df_train.merge(df_clp[new_cols], on='ts', how='left')
print(f"Training rows: {len(df_train)}, after merge: {len(df_merged)}")

# Check null rates after merge
for c in ['clot_std','clot_median','clot_coverage','cler23_coverage','cler23_std','clth_mean','clth_std','cltt_mean','cltt_std']:
    print(f"  {c}: null={df_merged[c].isna().mean()*100:.1f}%")

# Check correlation with targets
print()
print("Correlation with kt_next_1h_mean:")
for c in ['clot_std','clot_median','clot_coverage','cler23_coverage','cler23_std','clth_mean','clth_std','cltt_mean','cltt_std']:
    mask = df_merged[c].notna() & df_merged['kt_next_1h_mean'].notna() & (df_merged['sun_altitude'] > 5)
    if mask.sum() > 100:
        r = df_merged.loc[mask, c].corr(df_merged.loc[mask, 'kt_next_1h_mean'])
        print(f"  {c}: r={r:.3f} (n={mask.sum()})")

print()
print("Correlation with kt_next_3h_mean:")
for c in ['clot_std','clot_median','clot_coverage','cler23_coverage','cler23_std','clth_mean','clth_std','cltt_mean','cltt_std']:
    mask = df_merged[c].notna() & df_merged['kt_next_3h_mean'].notna() & (df_merged['sun_altitude'] > 5)
    if mask.sum() > 100:
        r = df_merged.loc[mask, c].corr(df_merged.loc[mask, 'kt_next_3h_mean'])
        print(f"  {c}: r={r:.3f} (n={mask.sum()})")

# Compute derived features
print()
print("=== PHASE 2: Build derived features ===")

ts_arr = (df_merged['ts'].values.astype('int64') / 1_000_000).astype(np.int64)

def add_delta(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90):
    val = df[col].values.astype(float)
    lag_vals = np.full(len(df), np.nan)
    valid_gap = np.abs((ts_arr[n_steps:] - ts_arr[:-n_steps]) - n_steps * step_sec) <= tol_sec
    lag_vals[n_steps:][valid_gap] = val[:-n_steps][valid_gap]
    df[col + '_delta30m'] = val - lag_vals
    return df

# Cloud top trends from new CLP stats
for var in ['clot_std','cler23_coverage']:
    df_merged = add_delta(df_merged, var, 3, ts_arr)  # delta 30m

# clot_std rolling 30m (std of std - captures volatility spikes)
df_merged['clot_std_roll30m'] = df_merged['clot_std'].rolling(4, min_periods=2).mean()

# Also build beam_diffuse and cloud_top_trend (v3 features - needed for baseline)
for var in ['cloud_top_height_m','cloud_top_temp_c','cloud_eff_radius_um']:
    df_merged = add_delta(df_merged, var, 3, ts_arr)
    col_d = var + '_delta30m'
    df_merged = add_delta(df_merged, var, 6, ts_arr)

df_merged['dni_ratio'] = df_merged['dni_consolidated'] / df_merged['ghi_consolidated'].clip(lower=1)
df_merged['dhi_fraction'] = df_merged['dhi_consolidated'] / df_merged['ghi_consolidated'].clip(lower=1)
df_merged.loc[df_merged['ghi_consolidated'] < 10, ['dni_ratio','dhi_fraction']] = np.nan

print("New derived features built.")
for c in ['clot_std_delta30m','cler23_coverage_delta30m','clot_std_roll30m']:
    print(f"  {c}: null={df_merged[c].isna().mean()*100:.1f}%")

# Save enhanced dataset
df_merged.to_parquet('dfm_with_clp_stats.parquet', index=False)
print(f"Saved dfm_with_clp_stats.parquet ({len(df_merged)} rows, {len(df_merged.columns)} cols)")

print()
print("=== PHASE 3: Ablation test ===")

# Feature sets
FEATS_V3 = ['sun_altitude','hour_sin','hour_cos','doy_sin','doy_cos','month',
    'kt','kt_lag10m','kt_lag20m','kt_lag30m','kt_lag60m',
    'kt_roll30m_mean','kt_roll60m_mean','kt_roll30m_std',
    'kt_short_lag1m','kt_short_lag2m','kt_short_lag3m','kt_short_lag5m',
    'kt_short_roll5m_mean','kt_short_roll5m_std',
    'kt_delta_1m','kt_delta_5m','kt_delta_10m','kt_slope_5m',
    'cloud_optical_thickness','cloud_top_temp_c','cloud_top_height_m',
    'cloud_eff_radius_um','sat_cloud_present',
    'cloud_cover_oktas','cloud_low_type','cloud_med_type','cloud_high_type',
    'cloud_optical_thickness_lag10m','cloud_optical_thickness_delta10m',
    'cloud_cover_oktas_lag10m','cloud_cover_oktas_delta10m',
    'cloud_optical_thickness_roll60m_mean','cloud_optical_thickness_roll60m_std',
    'cloud_cover_oktas_roll60m_mean','cloud_cover_oktas_roll60m_std',
    'angstrom_exp_440_870','precipitable_water_cm','AOD_planck_avg',
    'cloud_optical_thickness_delta30m','cloud_optical_thickness_delta60m',
    'cloud_cover_oktas_delta30m','cloud_cover_oktas_delta60m',
    'dni_ratio','dhi_fraction',
    'cloud_top_height_m_delta30m','cloud_top_temp_c_delta30m','cloud_eff_radius_um_delta30m',
    'cloud_top_height_m_delta60m','cloud_top_temp_c_delta60m','cloud_eff_radius_um_delta60m',
]

NEW_GROUPS = {
    'cler23_coverage': ['cler23_coverage'],
    'clot_std': ['clot_std'],
    'clot_median': ['clot_median'],
    'clot_coverage': ['clot_coverage'],
    'clth_stats': ['clth_mean','clth_std','clth_median'],
    'cltt_stats': ['cltt_mean','cltt_std','cltt_median'],
    'cler23_full': ['cler23_coverage','cler23_std','cler23_mean'],
    'clot_derived': ['clot_std_delta30m','clot_std_roll30m','cler23_coverage_delta30m'],
    'all_clp_new': ['clot_std','clot_median','clot_coverage',
                    'cler23_coverage','cler23_std','cler23_mean',
                    'clth_mean','clth_std','clth_median',
                    'cltt_mean','cltt_std','cltt_median',
                    'cloud_class_code_new'],
    'top_candidates': ['clot_std','cler23_coverage',
                       'clot_std_delta30m','cler23_coverage_delta30m',
                       'clot_std_roll30m'],
}

CAT_PARAMS = dict(
    iterations=600, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, min_data_in_leaf=20,
    loss_function='Quantile:alpha=0.5',
    verbose=0, random_seed=42,
    allow_writing_files=False
)

split = int(len(df_merged)*0.8)
df_tr = df_merged.iloc[:split].copy()
df_te = df_merged.iloc[split:].copy()

for horizon in ['kt_next_1h_mean','kt_next_2h_mean','kt_next_3h_mean']:
    mask_tr = df_tr[horizon].notna() & (df_tr['sun_altitude'] > 5)
    mask_te = df_te[horizon].notna() & (df_te['sun_altitude'] > 5)
    y_te = df_te.loc[mask_te, horizon].astype(float)

    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(df_tr.loc[mask_tr, FEATS_V3].astype(float), df_tr.loc[mask_tr, horizon].astype(float))
    r2_base = r2_score(y_te, m.predict(df_te.loc[mask_te, FEATS_V3].astype(float)))
    print(f'{horizon} BASELINE (v3): {r2_base:.4f}')

    for gname, extra in NEW_GROUPS.items():
        feats_plus = FEATS_V3 + extra
        m2 = CatBoostRegressor(**CAT_PARAMS)
        m2.fit(df_tr.loc[mask_tr, feats_plus].astype(float), df_tr.loc[mask_tr, horizon].astype(float))
        r2_p = r2_score(y_te, m2.predict(df_te.loc[mask_te, feats_plus].astype(float)))
        print(f'  + {gname}: {r2_p:.4f} ({r2_p-r2_base:+.4f})')
    print()
