import duckdb, warnings, numpy as np, pandas as pd, json, pickle
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

DB = '/sessions/hopeful-bold-meitner/mnt/DuckDB/kalbar_local.db'
OUT = '/sessions/hopeful-bold-meitner/mnt/DuckDB/'

print("Loading data...")
con = duckdb.connect(DB, read_only=True)
df = con.execute("SELECT * FROM training_enhanced_v2").df()
con.close()

# Encode booleans
for col in ['cloud_present','aerosol_retrieval_valid','clear_sky','clp_cloud_present']:
    if col in df.columns:
        df[col] = df[col].astype(float)

df = df.sort_values('timestamp_wib').reset_index(drop=True)

# Impute NaN dengan sentinel values (tidak dropna agresif)
for c in ['AOT_mean','AOT_std','AE_mean','fine_mode_aot_proxy','coarse_mode_aot_proxy','AOT_uncertainty_mean']:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)
df['CLER_23_mean'] = df['CLER_23_mean'].fillna(-1.0)
df['rainfall_mm']  = df['rainfall_mm'].fillna(0.0)
df['sunshine_min'] = df['sunshine_min'].fillna(0.0)
for c in ['AOD_500nm','angstrom_440_870','precipitable_water_cm']:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

print(f"Dataset: {len(df):,} baris, periode {df.timestamp_wib.min()} s/d {df.timestamp_wib.max()}")

def eval_model(name, model, X_tr, y_tr, X_te, y_te, w=None, ts_te=None):
    kw = {'sample_weight': w} if w is not None else {}
    model.fit(X_tr, y_tr, **kw)
    p = model.predict(X_te)
    r2   = r2_score(y_te, p)
    mae  = mean_absolute_error(y_te, p)
    rmse = np.sqrt(mean_squared_error(y_te, p))
    r2h  = None
    if ts_te is not None:
        tmp = pd.DataFrame({'p':p,'a':y_te.values,'ts':pd.to_datetime(ts_te)})
        hly = tmp.groupby(tmp['ts'].dt.floor('h'))[['p','a']].mean()
        r2h = r2_score(hly['a'], hly['p'])
    bias = float(np.mean(p - y_te.values))
    print(f"  [{name}]  R²={r2:.4f}  R²_jam={r2h:.4f}  MAE={mae:.4f}  bias={bias:+.4f}  n_tr={len(y_tr):,}  n_te={len(y_te):,}")
    return {'name':name,'r2':r2,'r2h':r2h,'mae':mae,'rmse':rmse,'bias':bias}

results = {}

# ── A. BASELINE ────────────────────────────────────────────────────────
print("\n=== A. BASELINE (cloud_optical_thick asli, no lag-fix) ===")
fa = [f for f in ['cloud_optical_thick','cloud_top_temp_k','cloud_top_height_m',
      'cloud_eff_radius_um','cloud_present','temp_air_c','humidity_pct',
      'wind_speed_ms','rainfall_mm','sun_altitude',
      'hour_sin','hour_cos','doy_sin','doy_cos','month'] if f in df.columns]
da = df[fa+['kt','timestamp_wib']].dropna(subset=fa+['kt'])
sp = int(len(da)*0.80)
results['A'] = eval_model("A-Baseline",
    lgb.LGBMRegressor(n_estimators=400,num_leaves=63,learning_rate=0.05,
                      subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1),
    da[fa].iloc[:sp], da['kt'].iloc[:sp],
    da[fa].iloc[sp:], da['kt'].iloc[sp:],
    ts_te=da['timestamp_wib'].iloc[sp:].values)

# ── B. ENHANCED ────────────────────────────────────────────────────────
print("\n=== B. ENHANCED (lag-fix + aerosol_retrieval_valid + impute + weight) ===")
fb = [f for f in [
    'CLOT_mean','CLOT_std','CLTT_mean','CLTH_mean','CLER_23_mean',
    'AOT_mean','AOT_std','AE_mean','fine_mode_aot_proxy','coarse_mode_aot_proxy',
    'aerosol_retrieval_valid','clear_sky',
    'temp_air_c','humidity_pct','pressure_qff_mb','wind_speed_ms','rainfall_mm','sunshine_min',
    'sun_altitude','hour_sin','hour_cos','doy_sin','doy_cos','month'
] if f in df.columns]

db_c = df[~df['flag_extreme_contradict']].copy()
db   = db_c[fb+['kt','sample_weight','timestamp_wib']].dropna(subset=['kt','CLOT_mean'])
sp   = int(len(db)*0.80)
Xtr_b,ytr_b,wtr_b = db[fb].iloc[:sp], db['kt'].iloc[:sp], db['sample_weight'].iloc[:sp]
Xte_b,yte_b        = db[fb].iloc[sp:],  db['kt'].iloc[sp:]
ts_b               = db['timestamp_wib'].iloc[sp:].values

lgb_b = lgb.LGBMRegressor(n_estimators=400,num_leaves=63,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
results['B_w']  = eval_model("B-Enhanced+Weight", lgb_b, Xtr_b, ytr_b, Xte_b, yte_b, w=wtr_b.values, ts_te=ts_b)

lgb_b2 = lgb.LGBMRegressor(n_estimators=400,num_leaves=63,learning_rate=0.05,
                            subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
results['B_nw'] = eval_model("B-Enhanced-noWeight", lgb_b2, Xtr_b, ytr_b, Xte_b, yte_b, ts_te=ts_b)

# ── C. ENHANCED + AERONET ──────────────────────────────────────────────
print("\n=== C. ENHANCED + AERONET ===")
fc = fb + [f for f in ['AOD_500nm','angstrom_440_870','precipitable_water_cm'] if f in df.columns]
dc = db_c[fc+['kt','sample_weight','timestamp_wib']].dropna(subset=['kt','CLOT_mean'])
sp = int(len(dc)*0.80)
lgb_c = lgb.LGBMRegressor(n_estimators=400,num_leaves=63,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
results['C'] = eval_model("C-Enhanced+AERONET", lgb_c,
    dc[fc].iloc[:sp], dc['kt'].iloc[:sp],
    dc[fc].iloc[sp:], dc['kt'].iloc[sp:],
    w=dc['sample_weight'].iloc[:sp].values, ts_te=dc['timestamp_wib'].iloc[sp:].values)

# ── D. RANDOM FOREST ───────────────────────────────────────────────────
print("\n=== D. RandomForest ===")
rf = RandomForestRegressor(n_estimators=200,min_samples_leaf=10,
                           max_features=0.7,n_jobs=-1,random_state=42)
results['D_rf'] = eval_model("D-RandomForest", rf, Xtr_b, ytr_b, Xte_b, yte_b, w=wtr_b.values, ts_te=ts_b)

# ── E. ENSEMBLE ────────────────────────────────────────────────────────
print("\n=== E. Ensemble LGB+RF ===")
p_lgb = lgb_b.predict(Xte_b)
p_rf  = rf.predict(Xte_b)
for wl in [0.5, 0.6, 0.7]:
    p_ens = wl*p_lgb + (1-wl)*p_rf
    r2e = r2_score(yte_b, p_ens)
    tmp = pd.DataFrame({'p':p_ens,'a':yte_b.values,'ts':pd.to_datetime(ts_b)})
    r2he = r2_score(*[tmp.groupby(tmp['ts'].dt.floor('h'))[x].mean() for x in ['a','p']])
    mae_e = mean_absolute_error(yte_b, p_ens)
    print(f"  [Ensemble LGB{int(wl*100)}+RF{int((1-wl)*100)}]  R²={r2e:.4f}  R²_jam={r2he:.4f}  MAE={mae_e:.4f}")
    if wl == 0.6:
        results['E'] = {'name':'E-Ensemble-LGB60+RF40','r2':r2e,'r2h':r2he,'mae':mae_e,
                        'rmse':float(np.sqrt(mean_squared_error(yte_b,p_ens))),'bias':float(np.mean(p_ens-yte_b.values))}

# ── Feature importance ─────────────────────────────────────────────────
print("\n=== Feature Importance Top-15 (B-Enhanced LightGBM) ===")
fi = pd.Series(lgb_b.feature_importances_, index=fb).sort_values(ascending=False)
for feat, imp in fi.head(15).items():
    bar = '█' * int(imp/fi.max()*28)
    print(f"  {feat:<32} {imp:5.0f}  {bar}")

# ── Error per regime ───────────────────────────────────────────────────
print("\n=== Error per regime kt (B-Enhanced) ===")
te_df = pd.DataFrame({'pred': lgb_b.predict(Xte_b), 'actual': yte_b.values})
te_df['bin'] = pd.cut(te_df['actual'], bins=[0,.2,.4,.6,.8,1.0,1.5],
                       labels=['0.0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0','>1.0'])
for lbl, g in te_df.groupby('bin', observed=True):
    if len(g) > 5:
        r2b  = r2_score(g['actual'], g['pred'])
        mae_b = (g['pred']-g['actual']).abs().mean()
        bias_b = (g['pred']-g['actual']).mean()
        print(f"  kt={lbl}: n={len(g):5,}  R²={r2b:.3f}  MAE={mae_b:.3f}  bias={bias_b:+.3f}")

# ── Ringkasan ──────────────────────────────────────────────────────────
print("\n" + "="*75)
print("RINGKASAN AKHIR")
print(f"  {'Model':<38} {'R²(10min)':>10} {'R²(jam)':>9} {'MAE':>7} {'bias':>7}")
print("-"*75)
for k,r in results.items():
    r2h_str = f"{r['r2h']:.4f}" if r.get('r2h') else "-"
    print(f"  {r['name']:<38} {r['r2']:>10.4f} {r2h_str:>9} {r['mae']:>7.4f} {r.get('bias',0):>+7.4f}")
print("="*75)

b_r2  = results['B_w']['r2'];  a_r2  = results['A']['r2']
b_r2h = results['B_w']['r2h']; a_r2h = results['A']['r2h']
print(f"\n  ΔR²(10min)  Baseline→Enhanced : {b_r2 - a_r2:+.4f}")
print(f"  ΔR²(jam)    Baseline→Enhanced : {b_r2h - a_r2h:+.4f}")
print(f"  ΔR²(10min)  Enhanced→+AERONET : {results['C']['r2'] - b_r2:+.4f}")

# Save
with open(OUT+'training_results_v2.json','w') as f:
    json.dump(results, f, indent=2, default=str)
with open(OUT+'model_lgbm_enhanced_v2.pkl','wb') as f:
    pickle.dump({'model':lgb_b,'features':fb,'target':'kt','rf':rf,'lgb_c':lgb_c,'fc':fc}, f)
print("\nSaved: training_results_v2.json + model_lgbm_enhanced_v2.pkl")
