import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage2_dataset.parquet')
df = df.sort_values('ts_hour').reset_index(drop=True)

rad_base = ['ghi_clearsky_mean','ghi_clearsky_min','ghi_clearsky_max',
            'kt_mean','kt_std','kt_min','kt_max',
            'ghi_mean','ghi_std','ghi_min','ghi_max',
            'dni_mean','dni_std','dni_min','dni_max',
            'dhi_mean','dhi_std','dhi_min','dhi_max',
            'reflected_mean','sun_altitude_at_h']
meteo_base = ['temp_mean','temp_std','temp_min','temp_max','rh_mean','rh_std','rh_min','rh_max',
              'pressure_mean','pressure_std','wind_speed_mean','wind_speed_max','wind_dir_mean','rainfall_sum']

def get_feats(lookback, include_meteo):
    suf = tuple(f'lag{i}' for i in range(1, lookback+1))
    lag_cols = [c for c in df.columns if c.endswith(suf)]
    feats = rad_base + [c for c in lag_cols if any(c.startswith(b+'_lag') or c.startswith(b) for b in rad_base)]
    if include_meteo:
        feats += meteo_base + [c for c in lag_cols if any(c.startswith(b+'_lag') for b in meteo_base)]
    return [f for f in dict.fromkeys(feats) if f in df.columns]

n = len(df)
n_folds = 5
fold_size = n // (n_folds + 1)

def walk_forward(feats, label):
    rows = []
    for k in range(1, n_folds+1):
        tr_end = fold_size * k
        te_end = min(fold_size * (k+1), n)
        if te_end <= tr_end: break
        Xtr = df[feats].fillna(-999).iloc[:tr_end]; ytr = df['y_kt_next'].iloc[:tr_end]
        Xte = df[feats].fillna(-999).iloc[tr_end:te_end]; yte = df['y_kt_next'].iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        r2 = r2_score(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        rows.append((k, r2, rmse))
    res = pd.DataFrame(rows, columns=['fold','r2','rmse'])
    print(f"{label}: mean R2={res['r2'].mean():.4f}  mean RMSE={res['rmse'].mean():.4f}")
    return res, m

res_rad6, _ = walk_forward(get_feats(6, False), "Stage 1 (radiation only, lag1-6)  ")
res_full6, m_full6 = walk_forward(get_feats(6, True), "Stage 2 (radiation+meteo, lag1-6) ")

print("\nDelta R2 (stage2 - stage1):", res_full6['r2'].mean() - res_rad6['r2'].mean())

imp = pd.Series(m_full6.feature_importances_, index=get_feats(6, True)).sort_values(ascending=False)
print("\nTop 15 feature importance (last fold, Stage 2):")
print(imp.head(15).to_string())
