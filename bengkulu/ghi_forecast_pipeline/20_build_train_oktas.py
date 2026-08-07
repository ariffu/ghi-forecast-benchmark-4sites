import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features_oktas.parquet').reset_index(drop=True)
MIN_ALT, MAX_GAP_MIN, HORIZON, WINDOW = 5.0, 10, 6, 18

FEATURES = ['ghi_wm2','dni_wm2','dhi_wm2','sun_altitude','kt_approx',
            'cloud_present','cot','cth','ctt','cer','cloud_valid_bin',
            'delta_ghi_1','delta_ghi_3','delta_ghi_6','ghi_std6','delta_cot_1','cot_std6',
            'oktas_ffill','oktas_age_min','oktas_valid_bin']
FEATURES_BASELINE = [f for f in FEATURES if not f.startswith('oktas')]

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
ghi_np = df['ghi_wm2'].values.astype(np.float32)

valid = []
for i in range(WINDOW-1, n-HORIZON):
    if cum_b[i+HORIZON] - cum_b[i-WINDOW+1] > 0: continue
    if alt[i] <= MIN_ALT or alt[i+HORIZON] <= MIN_ALT: continue
    valid.append(i)
valid = np.array(valid, dtype=np.int64)
print(f"valid anchors: {len(valid):,}")

def build(features):
    n_feats = len(features)
    feat_np = df[features].values.astype(np.float32)
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
        for f in features: col_names.append(f"{f}_lag{lag}")
    col_names += ['hour_sin','hour_cos','doy_sin','doy_cos','y_ghi_t60']
    result = pd.DataFrame(out, columns=col_names)
    result.insert(0, 'anchor_ts', ts[valid])
    return result

def wf(result, label):
    feats = [c for c in result.columns if c not in ('anchor_ts','y_ghi_t60')]
    y = result['y_ghi_t60']
    n_tot = len(result); n_folds=5; fold_size = n_tot//(n_folds+1)
    rows = []
    for k in range(1,n_folds+1):
        tr_end = fold_size*k; te_end = min(fold_size*(k+1), n_tot)
        if te_end <= tr_end: break
        Xtr,ytr = result[feats].iloc[:tr_end], y.iloc[:tr_end]
        Xte,yte = result[feats].iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=127,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1)
        m.fit(Xtr,ytr)
        pred = m.predict(Xte).clip(0)
        rows.append(r2_score(yte,pred))
    print(f"{label}: mean R2={np.mean(rows):.4f}  folds={np.round(rows,4)}")
    return rows, m, feats

ds_base = build(FEATURES_BASELINE)
ds_oktas = build(FEATURES)
r_base,_,_ = wf(ds_base, "Baseline (tanpa oktas SYNOP)        ")
r_oktas,m_oktas,f_oktas = wf(ds_oktas, "Dengan oktas SYNOP (ffill 65min)   ")
print(f"\nDelta R2: {np.mean(r_oktas)-np.mean(r_base):+.4f}")

imp = pd.Series(m_oktas.feature_importances_, index=f_oktas).sort_values(ascending=False)
print("\noktas feature ranks:")
for f in [c for c in imp.index if c.startswith('oktas')][:6]:
    print(f"  {f}: rank {list(imp.index).index(f)+1}/{len(imp)}  importance={imp[f]}")
print("\nTop 10 overall:")
print(imp.head(10).to_string())
