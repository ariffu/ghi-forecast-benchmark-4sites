import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

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

# radiation+CLP only (best raw baseline from before, no meteo)
radclp_feats = [f for f in all_feats if not any(f==b or f.startswith(b+'_lag') for b in meteo_base)]

n = len(df)
n_folds = 5
fold_size = n // (n_folds + 1)
y = df['y_kt_next']

def walk_forward_raw(feats, label):
    rows = []
    for k in range(1, n_folds+1):
        tr_end = fold_size*k; te_end = min(fold_size*(k+1), n)
        if te_end <= tr_end: break
        Xtr = df[feats].fillna(-999).iloc[:tr_end]; ytr = y.iloc[:tr_end]
        Xte = df[feats].fillna(-999).iloc[tr_end:te_end]; yte = y.iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
        m.fit(Xtr, ytr)
        r2 = r2_score(yte, m.predict(Xte))
        rows.append(r2)
    print(f"{label}: mean R2={np.mean(rows):.4f}  (folds: {np.round(rows,4)})")
    return rows

def walk_forward_pca(feats, n_components, label):
    rows = []
    for k in range(1, n_folds+1):
        tr_end = fold_size*k; te_end = min(fold_size*(k+1), n)
        if te_end <= tr_end: break
        Xraw_tr = df[feats].fillna(df[feats].median()).iloc[:tr_end]
        Xraw_te = df[feats].fillna(df[feats].median()).iloc[tr_end:te_end]
        scaler = StandardScaler().fit(Xraw_tr)
        pca = PCA(n_components=n_components).fit(scaler.transform(Xraw_tr))
        Xtr = pca.transform(scaler.transform(Xraw_tr))
        Xte = pca.transform(scaler.transform(Xraw_te))
        ytr = y.iloc[:tr_end]; yte = y.iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
        m.fit(Xtr, ytr)
        r2 = r2_score(yte, m.predict(Xte))
        rows.append(r2)
    print(f"{label}: mean R2={np.mean(rows):.4f}  (folds: {np.round(rows,4)})")
    return rows

print("--- Full feature set (108 features) ---")
r_raw_full = walk_forward_raw(all_feats, "Raw (108 features, incl meteo)        ")
r_pca_full = walk_forward_pca(all_feats, 40, "PCA (40 PCs, 95% var, incl meteo)     ")

print("\n--- Radiation+CLP only (no meteo, prior best) ---")
r_raw_radclp = walk_forward_raw(radclp_feats, "Raw radiation+CLP ("+str(len(radclp_feats))+" features)   ")
n_pc_radclp = 40 - 8  # rough proportional cut; will just refit with explicit threshold instead below
