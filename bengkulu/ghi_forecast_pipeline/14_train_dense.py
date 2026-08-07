import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def walk_forward(path, label):
    df = pd.read_parquet(path).sort_values('anchor_ts').reset_index(drop=True)
    feats = [c for c in df.columns if c not in ('anchor_ts','y_ghi_t60')]
    y = df['y_ghi_t60']
    n = len(df)
    n_folds = 5
    fold_size = n // (n_folds+1)
    rows = []
    for k in range(1, n_folds+1):
        tr_end = fold_size*k; te_end = min(fold_size*(k+1), n)
        if te_end <= tr_end: break
        Xtr, ytr = df[feats].iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = df[feats].iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=127,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte).clip(0)
        r2 = r2_score(yte, pred); mae = mean_absolute_error(yte, pred); rmse = np.sqrt(mean_squared_error(yte, pred))
        # persistence baseline: ghi_wm2_lag0 (current GHI) = prediction for t+60min
        pers = df['ghi_wm2_lag0'].iloc[tr_end:te_end]
        pers_r2 = r2_score(yte, pers)
        rows.append((k, r2, mae, rmse, pers_r2))
    res = pd.DataFrame(rows, columns=['fold','r2','mae','rmse','persistence_r2'])
    print(f"\n=== {label} ===")
    print(res.to_string(index=False))
    print(f"Mean R2={res['r2'].mean():.4f}  Mean MAE={res['mae'].mean():.2f} W/m2  (persistence mean R2={res['persistence_r2'].mean():.4f})")
    return res, m, feats

res18, m18, f18 = walk_forward('C:/Users/ariff/bengkulu_ghi_forecast/dense_dataset_w18.parquet', "Lookback 3h (18 steps x 10min), target=GHI(t+60min)")
res36, m36, f36 = walk_forward('C:/Users/ariff/bengkulu_ghi_forecast/dense_dataset_w36.parquet', "Lookback 6h (36 steps x 10min), target=GHI(t+60min)")

best_m, best_f = (m18, f18) if res18['r2'].mean() >= res36['r2'].mean() else (m36, f36)
imp = pd.Series(best_m.feature_importances_, index=best_f).sort_values(ascending=False)
print("\nTop 15 feature importance (best lookback, last fold):")
print(imp.head(15).to_string())
