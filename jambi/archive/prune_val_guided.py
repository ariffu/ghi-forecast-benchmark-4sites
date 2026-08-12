"""
Val-guided feature pruning — protokol Kalbar note_17:
1. Baseline 72 fitur (VAL + TEST R²)
2. Sweep top-K (di VAL saja), K = {5,8,10,15,20,25,30,40}
3. Greedy backward elimination dari top-K* (di VAL)
4. TEST dievaluasi SEKALI di akhir untuk kandidat final

Split: train/val/test (bukan train/test 80/20)
Tujuan: menyederhanakan model tanpa menurunkan akurasi (seperti Kalbar 49→7 fitur, +0.004 R²)
"""
import pandas as pd
import numpy as np
import warnings, json, time
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

print("=== Val-Guided Feature Pruning (Kalbar protocol) ===\n")
t_start = time.time()

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

# 3-way split: train=65%, val=15%, test=20%
n = len(df)
n_tr = int(n * 0.65)
n_va = int(n * 0.80)
df_tr = df.iloc[:n_tr].copy()
df_va = df.iloc[n_tr:n_va].copy()
df_te = df.iloc[n_va:].copy()
print(f"Split: train={len(df_tr)} ({df_tr['ts'].max()}), "
      f"val={len(df_va)} ({df_va['ts'].min()} to {df_va['ts'].max()}), "
      f"test={len(df_te)} ({df_te['ts'].min()} to {df_te['ts'].max()})")

CAT_PARAMS = dict(
    iterations=500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, min_data_in_leaf=20,
    loss_function='Quantile:alpha=0.5',
    verbose=0, random_seed=42, allow_writing_files=False
)

# Pruning is per-horizon — do kt_next_1h_mean first (primary metric)
# Then check consistency across horizons

HORIZON = 'kt_next_1h_mean'
print(f"\nPruning target: {HORIZON}")

mask_tr = df_tr[HORIZON].notna() & (df_tr['sun_altitude'] > 5)
mask_va = df_va[HORIZON].notna() & (df_va['sun_altitude'] > 5)
mask_te = df_te[HORIZON].notna() & (df_te['sun_altitude'] > 5)
y_va = df_va.loc[mask_va, HORIZON].astype(float)
y_te = df_te.loc[mask_te, HORIZON].astype(float)

def fit_eval(feats, mask_tr, X_va, y_va):
    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(df_tr.loc[mask_tr, feats].astype(float),
          df_tr.loc[mask_tr, HORIZON].astype(float))
    return m, r2_score(y_va, m.predict(X_va[feats].astype(float)))

X_va = df_va.loc[mask_va]
X_te = df_te.loc[mask_te]

# PHASE 1: Baseline
print("\nPhase 1: Baseline (72 features)...")
m_base, r2_val_base = fit_eval(FEATS_V4, mask_tr, X_va, y_va)
print(f"  VAL R² = {r2_val_base:.4f}")

# Get feature importances from baseline model
fi = pd.Series(m_base.get_feature_importance(), index=FEATS_V4).sort_values(ascending=False)
print(f"\nTop 20 features by importance:")
for feat, imp in fi.head(20).items():
    print(f"  {imp:6.2f}  {feat}")

# PHASE 2: Sweep top-K (val-guided)
print("\nPhase 2: Sweep top-K...")
K_values = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
sweep_results = {}
for K in K_values:
    feats_k = fi.head(K).index.tolist()
    _, r2_k = fit_eval(feats_k, mask_tr, X_va, y_va)
    sweep_results[K] = r2_k
    print(f"  K={K:3d}: VAL R² = {r2_k:.4f}  (Δ vs baseline: {r2_k - r2_val_base:+.4f})")

# Find optimal K: smallest K within tolerance 0.0005 of baseline
tolerance = 0.0005
k_star = None
for K in K_values:
    if sweep_results[K] >= r2_val_base - tolerance:
        k_star = K
        break
if k_star is None:
    k_star = min(sweep_results, key=lambda k: abs(sweep_results[k] - r2_val_base))
print(f"\nK* = {k_star} (VAL R² = {sweep_results[k_star]:.4f})")

# PHASE 3: Greedy backward elimination from top-K*
print(f"\nPhase 3: Greedy backward elimination from top-{k_star}...")
current_feats = fi.head(k_star).index.tolist()
_, r2_current = fit_eval(current_feats, mask_tr, X_va, y_va)
print(f"  Start: {len(current_feats)} feats, VAL R² = {r2_current:.4f}")

tol_greedy = 0.0003
history = [(list(current_feats), r2_current)]

while len(current_feats) > 3:
    best_drop = None
    best_r2 = r2_current - tol_greedy - 1e-9  # must improve or stay flat
    for feat in current_feats:
        candidate = [f for f in current_feats if f != feat]
        _, r2_c = fit_eval(candidate, mask_tr, X_va, y_va)
        if r2_c >= best_r2:
            best_r2 = r2_c
            best_drop = feat
    if best_drop is None:
        print(f"  Stop: dropping any feature hurts VAL more than tolerance ({tol_greedy})")
        break
    current_feats.remove(best_drop)
    r2_current = best_r2
    history.append((list(current_feats), r2_current))
    elapsed = time.time() - t_start
    print(f"  [{elapsed:5.0f}s] Dropped '{best_drop}': {len(current_feats)} feats, VAL R² = {r2_current:.4f}")

pruned_feats = current_feats
print(f"\nFinal pruned set: {len(pruned_feats)} features")
print("Pruned features:", pruned_feats)

# PHASE 4: Evaluate on TEST (once)
print("\nPhase 4: TEST evaluation (evaluated ONCE now)...")
candidates = {
    'baseline_72': FEATS_V4,
    f'top_{k_star}': fi.head(k_star).index.tolist(),
    f'pruned_{len(pruned_feats)}': pruned_feats,
}

final_results = {}
for name, feats in candidates.items():
    m, r2_v = fit_eval(feats, mask_tr, X_va, y_va)
    r2_t = r2_score(y_te, m.predict(X_te[feats].astype(float)))
    final_results[name] = {'feats': feats, 'val_r2': r2_v, 'test_r2': r2_t}
    print(f"  {name:25s}: VAL={r2_v:.4f}, TEST={r2_t:.4f}")

# Check consistency on other horizons for best pruned
print(f"\nConsistency check on all horizons for pruned_{len(pruned_feats)} vs baseline:")
for h in ['kt_next_1h_mean', 'kt_next_2h_mean', 'kt_next_3h_mean']:
    mask_tr_h = df_tr[h].notna() & (df_tr['sun_altitude'] > 5)
    mask_te_h = df_te[h].notna() & (df_te['sun_altitude'] > 5)
    y_te_h = df_te.loc[mask_te_h, h].astype(float)

    m_b = CatBoostRegressor(**CAT_PARAMS)
    m_b.fit(df_tr.loc[mask_tr_h, FEATS_V4].astype(float), df_tr.loc[mask_tr_h, h].astype(float))
    r2_b = r2_score(y_te_h, m_b.predict(df_te.loc[mask_te_h, FEATS_V4].astype(float)))

    m_p = CatBoostRegressor(**CAT_PARAMS)
    m_p.fit(df_tr.loc[mask_tr_h, pruned_feats].astype(float), df_tr.loc[mask_tr_h, h].astype(float))
    r2_p = r2_score(y_te_h, m_p.predict(df_te.loc[mask_te_h, pruned_feats].astype(float)))

    h_label = h.replace('kt_next_', 'jam ke-').replace('_mean', '')
    print(f"  {h_label}: baseline={r2_b:.4f}, pruned={r2_p:.4f} ({r2_p-r2_b:+.4f})")

# Save results
out = {
    'baseline_feats': FEATS_V4,
    'pruned_feats': pruned_feats,
    'k_star': k_star,
    'sweep': sweep_results,
    'val_r2_baseline': r2_val_base,
    'final_results': {k: {'n_feats': len(v['feats']), 'val_r2': v['val_r2'], 'test_r2': v['test_r2']}
                      for k, v in final_results.items()},
}
with open('pruning_results.json', 'w') as f:
    json.dump(out, f, indent=2)

total = time.time() - t_start
print(f"\nTotal time: {total:.0f}s ({total/60:.1f} min)")
print("Results saved to pruning_results.json")
