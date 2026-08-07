import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_dataset_w18.parquet').sort_values('anchor_ts').reset_index(drop=True)
feats = [c for c in df.columns if c not in ('anchor_ts','y_ghi_t60')]
y = df['y_ghi_t60']
n = len(df)

# final holdout: last 10% chronological, purely for an honest final report number (not used in training)
i_final = int(n*0.90)
Xtr, ytr = df[feats].iloc[:i_final], y.iloc[:i_final]
Xte, yte = df[feats].iloc[i_final:], y.iloc[i_final:]

model = LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=127,
                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1)
model.fit(Xtr, ytr)
pred = model.predict(Xte).clip(0)
r2 = r2_score(yte, pred); mae = mean_absolute_error(yte, pred); rmse = np.sqrt(mean_squared_error(yte, pred))
pers_r2 = r2_score(yte, Xte['ghi_wm2_lag0'])
print(f"Final holdout (last 10%, {df['anchor_ts'].iloc[i_final]} to {df['anchor_ts'].iloc[-1]}):")
print(f"  Model R2={r2:.4f}  MAE={mae:.2f} W/m2  RMSE={rmse:.2f} W/m2")
print(f"  Persistence R2={pers_r2:.4f}")

# retrain on FULL data for production deployment (maximize data usage)
model_full = LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=127,
                            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1)
model_full.fit(df[feats], y)

artifact = {
    'model': model_full,
    'feature_names': feats,
    'window_steps': 18,
    'window_minutes': 180,
    'horizon_minutes': 60,
    'target': 'ghi_wm2 at t+60min (point value, not hourly mean)',
    'walk_forward_r2': 0.754,
    'holdout_r2': r2,
    'holdout_mae_wm2': mae,
    'holdout_rmse_wm2': rmse,
    'persistence_r2': pers_r2,
    'training_period': f"{df['anchor_ts'].min()} to {df['anchor_ts'].max()}",
    'n_train_rows': n,
    'methodology': 'Native 10-min resolution dense lag (Kalbar-style), LightGBM, single-horizon point target. '
                    'Tested and rejected: hourly-aggregate target, AWS meteo features, PCA, explicit season features, SYNOP oktas at 10-min resolution.',
}
joblib.dump(artifact, 'C:/Users/ariff/bengkulu_ghi_forecast/model_production_bengkulu_ghi_1h.pkl')
print("\nSaved: model_production_bengkulu_ghi_1h.pkl")
print(f"  Features: {len(feats)}")
print(f"  Trained on full data: {n:,} rows")
