import pandas as pd
import numpy as np

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features.parquet').reset_index(drop=True)
ts = pd.to_datetime(df['ts_bin'])
df['is_kemarau'] = ts.dt.month.isin([4,5,6,7,8,9]).astype(np.float32)
df['month_sin'] = np.sin(2*np.pi*ts.dt.month/12).astype(np.float32)
df['month_cos'] = np.cos(2*np.pi*ts.dt.month/12).astype(np.float32)
df.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features_season.parquet')
print("added: is_kemarau, month_sin, month_cos")
print(df[['is_kemarau','month_sin','month_cos']].describe())
