import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_features_season.parquet').reset_index(drop=True)
MIN_ALT, MAX_GAP_MIN, HORIZON, WINDOW = 5.0, 10, 6, 18

FEATURES = ['ghi_wm2','dni_wm2','dhi_wm2','sun_altitude','kt_approx',
            'cloud_present','cot','cth','ctt','cer','cloud_valid_bin',
            'delta_ghi_1','delta_ghi_3','delta_ghi_6','ghi_std6','delta_cot_1','cot_std6']
# season features are SLOW-scale (constant within the lookback window) so attach once per anchor, not per lag
SEASON_FEATS = ['is_kemarau','month_sin','month_cos']

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
season_np = df[SEASON_FEATS].values.astype(np.float32)
ghi_np = df['ghi_wm2'].values.astype(np.float32)
n_feats = len(FEATURES); n_season = len(SEASON_FEATS)

valid = []
for i in range(WINDOW-1, n-HORIZON):
    if cum_b[i+HORIZON] - cum_b[i-WINDOW+1] > 0: continue
    if alt[i] <= MIN_ALT or alt[i+HORIZON] <= MIN_ALT: continue
    valid.append(i)
valid = np.array(valid, dtype=np.int64)
print(f"valid anchors: {len(valid):,}")

n_samp = len(valid)
n_cols = n_feats*WINDOW + n_season + 4 + 1
out = np.empty((n_samp, n_cols), dtype=np.float32)
for j,i in enumerate(valid):
    ptr = 0
    for lag in range(WINDOW):
        out[j, ptr:ptr+n_feats] = feat_np[i-lag]; ptr += n_feats
    out[j, ptr:ptr+n_season] = season_np[i]; ptr += n_season
    out[j, ptr:ptr+4] = [hs[i], hc[i], ds[i], dc[i]]; ptr += 4
    out[j, ptr] = ghi_np[i+HORIZON]

col_names = []
for lag in range(WINDOW):
    for f in FEATURES: col_names.append(f"{f}_lag{lag}")
col_names += SEASON_FEATS + ['hour_sin','hour_cos','doy_sin','doy_cos','y_ghi_t60']
result = pd.DataFrame(out, columns=col_names)
result.insert(0, 'anchor_ts', ts[valid])
result.to_parquet('C:/Users/ariff/bengkulu_ghi_forecast/dense_dataset_w18_season.parquet')

# walk-forward: with season feats vs without (ablation, isolate marginal contribution)
feats_with = [c for c in result.columns if c not in ('anchor_ts','y_ghi_t60')]
feats_without = [c for c in feats_with if c not in SEASON_FEATS]
y = result['y_ghi_t60']
n_tot = len(result); n_folds = 5; fold_size = n_tot//(n_folds+1)

def wf(feats, label):
    rows = []
    for k in range(1, n_folds+1):
        tr_end = fold_size*k; te_end = min(fold_size*(k+1), n_tot)
        if te_end <= tr_end: break
        Xtr, ytr = result[feats].iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = result[feats].iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=127,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte).clip(0)
        rows.append(r2_score(yte, pred))
    print(f"{label}: mean R2={np.mean(rows):.4f}  folds={np.round(rows,4)}")
    return rows, m

r_without, _ = wf(feats_without, "Tanpa fitur musim (baseline)        ")
r_with, m_with = wf(feats_with, "Dengan is_kemarau+month_sin/cos     ")
print(f"\nDelta R2 (dengan - tanpa): {np.mean(r_with)-np.mean(r_without):+.4f}")

imp = pd.Series(m_with.feature_importances_, index=feats_with).sort_values(ascending=False)
print("\nRanking fitur musim vs lainnya (last fold):")
print(imp.head(20).to_string())
print("\nis_kemarau rank:", list(imp.index).index('is_kemarau')+1, "/", len(imp))
print("month_sin rank:", list(imp.index).index('month_sin')+1)
print("month_cos rank:", list(imp.index).index('month_cos')+1)
