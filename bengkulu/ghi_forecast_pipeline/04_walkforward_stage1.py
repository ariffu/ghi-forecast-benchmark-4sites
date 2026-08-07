import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage1_dataset.parquet')
df = df.sort_values('ts_hour').reset_index(drop=True)

base_feats = ['ghi_clearsky_mean','ghi_clearsky_min','ghi_clearsky_max',
              'kt_mean','kt_std','kt_min','kt_max',
              'ghi_mean','ghi_std','ghi_min','ghi_max',
              'dni_mean','dni_std','dni_min','dni_max',
              'dhi_mean','dhi_std','dhi_min','dhi_max',
              'reflected_mean','sun_altitude_at_h']

def get_feats(lookback):
    suf = tuple(f'lag{i}' for i in range(1, lookback+1))
    lag_cols = [col for col in df.columns if col.endswith(suf)]
    return [f for f in base_feats + lag_cols if f in df.columns]

n = len(df)
n_folds = 5
fold_size = n // (n_folds + 1)  # first fold_size always train-only seed

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
        pers_r2 = r2_score(yte, Xte['kt_mean'])
        rows.append((k, df['ts_hour'].iloc[tr_end], df['ts_hour'].iloc[te_end-1], len(Xte), r2, rmse, pers_r2))
    res = pd.DataFrame(rows, columns=['fold','test_start','test_end','n_test','r2','rmse','persistence_r2'])
    print(f"\n=== {label} ===")
    print(res.to_string(index=False))
    print(f"Mean R2 = {res['r2'].mean():.4f}  (persistence mean = {res['persistence_r2'].mean():.4f})")
    return res

res3 = walk_forward(get_feats(3), "Stage 1 walk-forward - lag 1-3h")
res6 = walk_forward(get_feats(6), "Stage 1 walk-forward - lag 1-6h")
