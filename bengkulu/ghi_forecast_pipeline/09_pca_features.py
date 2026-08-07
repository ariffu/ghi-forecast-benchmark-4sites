import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage3_dataset.parquet')
df = df.sort_values('ts_hour').reset_index(drop=True)

rad_base = ['ghi_clearsky_mean','ghi_clearsky_min','ghi_clearsky_max',
            'kt_mean','kt_std','kt_min','kt_max',
            'ghi_mean','ghi_std','ghi_min','ghi_max',
            'dni_mean','dni_std','dni_min','dni_max',
            'dhi_mean','dhi_std','dhi_min','dhi_max',
            'reflected_mean','sun_altitude_at_h']
meteo_base = ['temp_mean','temp_std','temp_min','temp_max','rh_mean','rh_std','rh_min','rh_max',
              'pressure_mean','pressure_std','wind_speed_mean','wind_speed_max','wind_dir_mean','rainfall_sum']
clp_base = ['cot_mean','cot_std','cth_mean','ctt_mean','cer_mean','cloud_frac']

suf6 = tuple(f'lag{i}' for i in range(1,7))
lag_cols = [c for c in df.columns if c.endswith(suf6)]
all_feats = [f for f in dict.fromkeys(rad_base+meteo_base+clp_base+lag_cols) if f in df.columns]
print("Total candidate features:", len(all_feats))

X = df[all_feats].fillna(df[all_feats].median())
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

pca_full = PCA().fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)
for thresh in [0.90, 0.95, 0.975, 0.99]:
    k = np.argmax(cum >= thresh) + 1
    print(f"  PCs needed for {thresh*100:.1f}% variance: {k} / {len(all_feats)}")

print("\nExplained variance ratio, first 20 PCs:")
print(np.round(pca_full.explained_variance_ratio_[:20], 4))
print("\nCumulative, first 20 PCs:")
print(np.round(cum[:20], 4))

# per-block PCA (mirroring Jambi methodology: blocks tested separately too)
print("\n=== Per-block redundancy check ===")
for name, block in [('radiation (t=0)', rad_base), ('meteo (t=0)', meteo_base), ('CLP (t=0)', clp_base),
                     ('radiation lags', [c for c in lag_cols if any(c.startswith(b+'_lag') for b in rad_base)]),
                     ('meteo lags', [c for c in lag_cols if any(c.startswith(b+'_lag') for b in meteo_base)]),
                     ('CLP lags', [c for c in lag_cols if any(c.startswith(b+'_lag') for b in clp_base)])]:
    block = [b for b in block if b in df.columns]
    if len(block) < 2: continue
    Xb = StandardScaler().fit_transform(df[block].fillna(df[block].median()))
    p = PCA().fit(Xb)
    c = np.cumsum(p.explained_variance_ratio_)
    k95 = np.argmax(c >= 0.95) + 1
    print(f"  {name}: {len(block)} cols -> {k95} PCs for 95% variance  (top3 EVR: {np.round(p.explained_variance_ratio_[:3],3)})")

# save full-PCA transform artifacts for reuse
import joblib
joblib.dump({'scaler': scaler, 'pca': pca_full, 'feature_names': all_feats}, 'C:/Users/ariff/bengkulu_ghi_forecast/pca_full_artifact.pkl')
