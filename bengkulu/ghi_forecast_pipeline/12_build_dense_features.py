import pandas as pd
import numpy as np

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/base_10min.parquet').sort_values('ts_bin').reset_index(drop=True)
print("rows:", len(df), "range:", df['ts_bin'].min(), df['ts_bin'].max())

# fill CLP gaps the way Kalbar handled AOT: 0 + valid flag (not forward-fill)
df['cloud_valid_bin'] = df['cot'].notna().astype(int)
for c in ['cot','cth','ctt','cer','cloud_present']:
    df[c] = df[c].fillna(0.0)

sin_alt = np.sin(np.radians(df['sun_altitude'].clip(lower=0))).clip(lower=0)
df['ghi_clearsky_rough'] = (950.0 * sin_alt**1.1).clip(lower=1.0)
df['kt_approx'] = np.where(df['sun_altitude']>0, (df['ghi_wm2']/df['ghi_clearsky_rough']).clip(0,1.3), 0.0).astype(np.float32)

g = df['ghi_wm2'].values.astype(np.float32)
df['delta_ghi_1'] = np.concatenate([[0], g[1:]-g[:-1]]).astype(np.float32)
df['delta_ghi_3'] = np.concatenate([[0,0,0], g[3:]-g[:-3]]).astype(np.float32)
df['delta_ghi_6'] = np.concatenate([[0]*6, g[6:]-g[:-6]]).astype(np.float32)
df['ghi_std6'] = pd.Series(g).rolling(6, min_periods=1).std().fillna(0).values.astype(np.float32)

c = df['cot'].values.astype(np.float32)
df['delta_cot_1'] = np.concatenate([[0], c[1:]-c[:-1]]).astype(np.float32)
df['cot_std6'] = pd.Series(c).rolling(6, min_periods=1).std().fillna(0).values.astype(np.float32)

df.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features.parquet')
print("saved. columns:", list(df.columns))
print("gap analysis (minutes between consecutive bins):")
gaps = df['ts_bin'].diff().dt.total_seconds().div(60)
print(gaps.value_counts().head(5))
