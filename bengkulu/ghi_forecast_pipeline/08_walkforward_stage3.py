import pandas as pd
import numpy as np
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

def get_feats(lookback, groups):
    suf = tuple(f'lag{i}' for i in range(1, lookback+1))
    lag_cols = [c for c in df.columns if c.endswith(suf)]
    feats = list(rad_base)
    if 'meteo' in groups: feats += meteo_base
    if 'clp' in groups: feats += clp_base
    bases = rad_base + (meteo_base if 'meteo' in groups else []) + (clp_base if 'clp' in groups else [])
    feats += [c for c in lag_cols if any(c.startswith(b+'_lag') for b in bases)]
    return [f for f in dict.fromkeys(feats) if f in df.columns]

n = len(df)
n_folds = 5
fold_size = n // (n_folds + 1)

def walk_forward(feats, label):
    rows = []
    last_m = None
    for k in range(1, n_folds+1):
        tr_end = fold_size * k
        te_end = min(fold_size * (k+1), n)
        if te_end <= tr_end: break
        Xtr = df[feats].fillna(-999).iloc[:tr_end]; ytr = df['y_kt_next'].iloc[:tr_end]
        Xte = df[feats].fillna(-999).iloc[tr_end:te_end]; yte = df['y_kt_next'].iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        r2 = r2_score(yte, pred); rmse = np.sqrt(mean_squared_error(yte, pred))
        rows.append((k, r2, rmse))
        last_m = m
    res = pd.DataFrame(rows, columns=['fold','r2','rmse'])
    print(f"{label}: mean R2={res['r2'].mean():.4f}  mean RMSE={res['rmse'].mean():.4f}")
    return res, last_m, feats

res1, _, _ = walk_forward(get_feats(6, []), "Stage 1 (radiation only)        ")
res2, _, _ = walk_forward(get_feats(6, ['meteo']), "Stage 2 (+meteo)                ")
res3, m3, f3 = walk_forward(get_feats(6, ['clp']), "Stage 3a (radiation+CLP, no meteo)")
res4, m4, f4 = walk_forward(get_feats(6, ['meteo','clp']), "Stage 3b (radiation+meteo+CLP)   ")

print("\nDelta R2 vs Stage1:")
print("  +meteo only      :", res2['r2'].mean()-res1['r2'].mean())
print("  +CLP only        :", res3['r2'].mean()-res1['r2'].mean())
print("  +meteo+CLP (all) :", res4['r2'].mean()-res1['r2'].mean())

imp = pd.Series(m4.feature_importances_, index=f4).sort_values(ascending=False)
print("\nTop 15 feature importance (last fold, full model):")
print(imp.head(15).to_string())
