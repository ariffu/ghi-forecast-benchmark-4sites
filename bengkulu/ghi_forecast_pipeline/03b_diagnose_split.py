import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage1_dataset.parquet')
df = df.sort_values('ts_hour').reset_index(drop=True)

n = len(df)
i_tr, i_va = int(n*0.70), int(n*0.85)
print("train range:", df['ts_hour'].iloc[0], "to", df['ts_hour'].iloc[i_tr-1])
print("val   range:", df['ts_hour'].iloc[i_tr], "to", df['ts_hour'].iloc[i_va-1])
print("test  range:", df['ts_hour'].iloc[i_va], "to", df['ts_hour'].iloc[-1])

print("\nrows per year:")
print(df['ts_hour'].dt.year.value_counts().sort_index())

print("\nkt_mean (current hour) stats by split:")
for name, sl in [('train', slice(0,i_tr)), ('val', slice(i_tr,i_va)), ('test', slice(i_va,n))]:
    print(name, df['kt_mean'].iloc[sl].describe()[['mean','std']].to_dict())

base_feats = ['ghi_clearsky_mean','ghi_clearsky_min','ghi_clearsky_max',
              'kt_mean','kt_std','kt_min','kt_max',
              'ghi_mean','ghi_std','ghi_min','ghi_max',
              'dni_mean','dni_std','dni_min','dni_max',
              'dhi_mean','dhi_std','dhi_min','dhi_max',
              'reflected_mean','sun_altitude_at_h']
lag6_feats = base_feats + [c for c in df.columns if c.endswith(('lag1','lag2','lag3','lag4','lag5','lag6'))]
feats = [f for f in lag6_feats if f in df.columns]

X = df[feats].fillna(-999)
y = df['y_kt_next']

# RANDOM split diagnostic
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42)
m = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
m.fit(Xtr, ytr)
pred = m.predict(Xte)
print("\n=== RANDOM split diagnostic ===")
print("R2:", r2_score(yte, pred), "RMSE:", np.sqrt(mean_squared_error(yte, pred)))
pers_r2 = r2_score(yte, Xte['kt_mean'])
print("Persistence R2 (random split):", pers_r2)
