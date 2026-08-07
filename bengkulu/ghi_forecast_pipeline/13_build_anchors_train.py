import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features.parquet').reset_index(drop=True)
MIN_ALT = 5.0
MAX_GAP_MIN = 10
HORIZON = 6  # 6 x 10min = 60 min ahead

FEATURES = ['ghi_wm2','dni_wm2','dhi_wm2','sun_altitude','kt_approx',
            'cloud_present','cot','cth','ctt','cer','cloud_valid_bin',
            'delta_ghi_1','delta_ghi_3','delta_ghi_6','ghi_std6','delta_cot_1','cot_std6']

n = len(df)
alt = df['sun_altitude'].values
ts = df['ts_bin'].values
diffs = pd.Series(ts).diff().dt.total_seconds().div(60).fillna(0)
breaks = (diffs > MAX_GAP_MIN).astype(int).values
cum_b = np.cumsum(breaks)

ts_pd = pd.to_datetime(ts)
hour_frac = ts_pd.hour + ts_pd.minute/60.0
doy = ts_pd.dayofyear.values
hs = np.sin(2*np.pi*hour_frac/24).astype(np.float32); hc = np.cos(2*np.pi*hour_frac/24).astype(np.float32)
ds = np.sin(2*np.pi*doy/365).astype(np.float32); dc = np.cos(2*np.pi*doy/365).astype(np.float32)

feat_np = df[FEATURES].values.astype(np.float32)
ghi_np = df['ghi_wm2'].values.astype(np.float32)
n_feats = len(FEATURES)

def build_for_window(WINDOW, label):
    valid = []
    for i in range(WINDOW-1, n-HORIZON):
        if cum_b[i+HORIZON] - cum_b[i-WINDOW+1] > 0: continue
        if alt[i] <= MIN_ALT or alt[i+HORIZON] <= MIN_ALT: continue
        valid.append(i)
    valid = np.array(valid, dtype=np.int64)
    print(f"{label}: WINDOW={WINDOW} ({WINDOW*10}min) -> {len(valid):,} valid anchors")

    n_samp = len(valid)
    n_cols = n_feats*WINDOW + 4 + 1
    out = np.empty((n_samp, n_cols), dtype=np.float32)
    for j,i in enumerate(valid):
        ptr = 0
        for lag in range(WINDOW):
            out[j, ptr:ptr+n_feats] = feat_np[i-lag]; ptr += n_feats
        out[j, ptr:ptr+4] = [hs[i], hc[i], ds[i], dc[i]]; ptr += 4
        out[j, ptr] = ghi_np[i+HORIZON]

    col_names = []
    for lag in range(WINDOW):
        for f in FEATURES: col_names.append(f"{f}_lag{lag}")
    col_names += ['hour_sin','hour_cos','doy_sin','doy_cos','y_ghi_t60']
    result = pd.DataFrame(out, columns=col_names)
    result.insert(0, 'anchor_ts', ts[valid])
    return result

ds18 = build_for_window(18, "Lookback 3h")
ds36 = build_for_window(36, "Lookback 6h")
ds18.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_dataset_w18.parquet')
ds36.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_dataset_w36.parquet')
