#!/usr/bin/env python3
"""
R8 Arm B — BANTEN: GBM vs Deep Learning Fair-Play Comparison (Tabel 2c).
Kelas DL (LSTMModel/MLPModel/TransformerModel) & loop training DISALIN PERSIS dari
train_ghi_1h_bengkulu_R8_armB.py (3 bug-fix dipertahankan). Yang beda hanya:
  - DB/tabel Banten (solar_features_base) + SQL + add_features (identik R1 Banten)
  - Station lat/lon Banten
  - Output dir

Fair-play: F1 lean-50, split train<2024/val2024/test2025, ES di val, scaler fit-train,
DL 3 seeds (mean±std), seq_len=1, target point t+60.
Run: & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_banten_R8_armB.py
"""
import warnings
from pathlib import Path
import duckdb, lightgbm as lgb, numpy as np, pandas as pd, torch, torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

# ---- Config (BANTEN) ----
DB_PATH="banten.duckdb"; OUTPUT_DIR=Path("outputs_R8_banten"); OUTPUT_DIR.mkdir(exist_ok=True)
STATION_LAT_DEG, STATION_LON_DEG, WIB_MERIDIAN_DEG = -6.26147, 106.7509, 105.0
TIME_COL="ts_wib"; TRAIN_END,VALID_END="2024-01-01","2025-01-01"
PRED_MIN,PRED_MAX=0.0,1400.0; RANDOM_STATE=42; DL_SEEDS=[0,1,2]
TARGET_POINT="ghi_point_t60"

F1_FEATURES=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m","kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
    "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m","clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
    "clp_cot_lag_30m","clp_cot_lag_60m","clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m",
    "clp_cot_roll_180m_mean","accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present",
    "hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]
assert len(F1_FEATURES)==50

# ============ DL Architectures (COPY PERSIS dari Bengkulu — 3 bug-fix) ============
class LSTMModel(nn.Module):
    """2-layer LSTM (128->64). Bug fix: pass lstm1 OUTPUT ke lstm2, bukan hidden state."""
    def __init__(self, n_features):
        super().__init__()
        self.lstm1=nn.LSTM(n_features,128,batch_first=True,dropout=0.2)
        self.lstm2=nn.LSTM(128,64,batch_first=True)
        self.dropout=nn.Dropout(0.2); self.fc=nn.Linear(64,1)
    def forward(self,x):
        out1,_=self.lstm1(x)
        _,(h2,_)=self.lstm2(out1)
        return self.fc(self.dropout(h2[-1]))

class MLPModel(nn.Module):
    """3-layer MLP. Bug fix: flat_size=n_features (seq_len=1)."""
    def __init__(self, n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,256),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,1))
    def forward(self,x): return self.net(x[:,-1])

class TransformerModel(nn.Module):
    """Transformer encoder (8 heads, 4 layers, d_model=64). Mean-pool over seq."""
    def __init__(self, n_features, d_model=64, nhead=8, nlayers=4):
        super().__init__()
        self.embed=nn.Linear(n_features,d_model)
        enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=256,dropout=0.2,batch_first=True)
        self.transformer=nn.TransformerEncoder(enc,nlayers); self.fc=nn.Linear(d_model,1)
    def forward(self,x):
        x=self.embed(x); x=self.transformer(x); x=x.mean(dim=1)
        return self.fc(x)

# ============ SQL (Banten — identik R1, dari solar_features_base) ============
def build_sql():
    return """
    WITH base AS (
        SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m, cloud_top_temp AS clp_ctt_k,
               cloud_eff_radius AS clp_cer, CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present
        FROM solar_features_base
    ), with_kt AS (SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base),
    w AS (
        SELECT *,
          LAG(ghi_now,1) OVER o AS ghi_lag_10m, LAG(ghi_now,2) OVER o AS ghi_lag_20m, LAG(ghi_now,3) OVER o AS ghi_lag_30m,
          LAG(ghi_now,6) OVER o AS ghi_lag_60m, LAG(ghi_now,12) OVER o AS ghi_lag_120m, LAG(ghi_now,18) OVER o AS ghi_lag_180m,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
          LAG(kt_point,1) OVER o AS kt_lag_10m, LAG(kt_point,2) OVER o AS kt_lag_20m, LAG(kt_point,3) OVER o AS kt_lag_30m, LAG(kt_point,6) OVER o AS kt_lag_60m,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
          LAG(clp_cot,1) OVER o AS clp_cot_lag_10m, LAG(clp_cot,2) OVER o AS clp_cot_lag_20m, LAG(clp_cot,3) OVER o AS clp_cot_lag_30m, LAG(clp_cot,6) OVER o AS clp_cot_lag_60m,
          AVG(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
          LEAD(ghi_now,6) OVER o AS ghi_lead_60m
        FROM with_kt WINDOW o AS (ORDER BY ts_wib))
    SELECT * FROM w WHERE solar_elev_deg>5 AND ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib
    """

# ============ Feature engineering (identik R1 Banten) ============
def solar_elevation_deg(timestamps):
    idx=pd.DatetimeIndex(timestamps); doy=idx.dayofyear.values.astype(float)
    h=idx.hour.values.astype(float)+idx.minute.values.astype(float)/60.0
    decl=23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0))
    ha=(h+4.0*(STATION_LON_DEG-WIB_MERIDIAN_DEG)/60.0-12.0)*15.0
    sin_e=(np.sin(np.deg2rad(STATION_LAT_DEG))*np.sin(np.deg2rad(decl))
           +np.cos(np.deg2rad(STATION_LAT_DEG))*np.cos(np.deg2rad(decl))*np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e,-1.0,1.0)))
def clearsky(e): return 1100.0*np.maximum(np.sin(np.deg2rad(e)),0.0)

def add_features(df):
    out=df.copy(); ts=pd.DatetimeIndex(out[TIME_COL])
    out["kt_now"]=out["ghi_now"].values/np.maximum(clearsky(out["solar_elev_deg"].values.astype(float)),20.0)
    out["clp_cot_delta_10m"]=out["clp_cot"]-out["clp_cot_lag_10m"]; out["clp_cot_delta_30m"]=out["clp_cot"]-out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]=out["clp_cot"]-out["clp_cot_lag_60m"]; out["clp_cot_delta_180m"]=out["clp_cot"]-out["clp_cot_roll_180m_mean"]
    out["accel_ghi_20m"]=out["ghi_now"]-2.0*out["ghi_lag_10m"]+out["ghi_lag_20m"]
    out["accel_kt_20m"]=out["kt_now"]-2.0*out["kt_lag_10m"]+out["kt_lag_20m"]
    out["accel_clp_cot_20m"]=out["clp_cot"]-2.0*out["clp_cot_lag_10m"]+out["clp_cot_lag_20m"]
    out["ghi_delta_10m"]=out["ghi_now"]-out["ghi_lag_10m"]; out["ghi_delta_60m"]=out["ghi_now"]-out["ghi_lag_60m"]
    hh=ts.hour.values.astype(float)+ts.minute.values.astype(float)/60.0; doy=ts.dayofyear.values.astype(float); mo=ts.month.values.astype(float)
    out["hour_sin"]=np.sin(2*np.pi*hh/24); out["hour_cos"]=np.cos(2*np.pi*hh/24)
    out["doy_sin"]=np.sin(2*np.pi*doy/365.25); out["doy_cos"]=np.cos(2*np.pi*doy/365.25)
    out["month_sin"]=np.sin(2*np.pi*mo/12); out["month_cos"]=np.cos(2*np.pi*mo/12)
    et=solar_elevation_deg(out[TIME_COL]+pd.Timedelta(minutes=60))
    out["elev_sin_t60"]=np.maximum(np.sin(np.deg2rad(et)),0.0); out["ghi_cs_t60"]=clearsky(et)
    cs_steps=[clearsky(solar_elevation_deg(out[TIME_COL]+pd.Timedelta(minutes=s*10))) for s in range(1,7)]
    out["smart_persist"]=out["kt_now"]*out["ghi_cs_t60"]; out["smart_persist_avg"]=out["kt_now"]*np.column_stack(cs_steps).mean(axis=1)
    out[TARGET_POINT]=out["ghi_lead_60m"].copy()
    out["sun_gt5_t60"]=out["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    return out

def split_masks(df):
    ts=df[TIME_COL]
    return (ts<pd.Timestamp(TRAIN_END),(ts>=pd.Timestamp(TRAIN_END))&(ts<pd.Timestamp(VALID_END)),ts>=pd.Timestamp(VALID_END))

# ============ GBM (COPY PERSIS) ============
def train_catboost(x_tr,y_tr,x_va,y_va):
    m=CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",
        random_seed=RANDOM_STATE,verbose=False,thread_count=-1,allow_writing_files=False)
    m.fit(x_tr.astype(float).values,y_tr.astype(float).values,
          eval_set=(x_va.astype(float).values,y_va.astype(float).values),early_stopping_rounds=150)
    return m
def train_lgbm(x_tr,y_tr,x_va,y_va):
    pipe=Pipeline([("imp",SimpleImputer(strategy="median",keep_empty_features=True)),
        ("m",lgb.LGBMRegressor(objective="regression",n_estimators=6000,learning_rate=0.02,num_leaves=39,
            min_child_samples=70,reg_alpha=0.2,reg_lambda=2.5,colsample_bytree=0.82,subsample=0.85,subsample_freq=1,
            random_state=RANDOM_STATE,n_jobs=-1,force_col_wise=True,verbosity=-1))])
    pipe.fit(x_tr,y_tr,m__eval_set=[(pipe.named_steps["imp"].fit_transform(x_va),y_va)],
             m__eval_metric="rmse",m__callbacks=[lgb.early_stopping(150,verbose=False)])
    return pipe

# ============ DL trainer (COPY PERSIS) ============
def train_dl(model,train_loader,val_loader,device,patience=30,max_epochs=150):
    opt=Adam(model.parameters(),lr=0.001); criterion=nn.MSELoss()
    best_loss=float("inf"); wait=0; best_state=None
    for _ in range(max_epochs):
        model.train()
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=criterion(model(xb).squeeze(),yb); loss.backward(); opt.step()
        model.eval(); val_loss=0.0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb=xb.to(device),yb.to(device); val_loss+=criterion(model(xb).squeeze(),yb).item()
        val_loss/=max(len(val_loader),1)
        if val_loss<best_loss-1e-6: best_loss=val_loss; wait=0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model
def predict_dl(model,loader,device):
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in loader: preds.append(model(xb.to(device)).squeeze().cpu().numpy())
    return np.clip(np.concatenate(preds),PRED_MIN,PRED_MAX)

def metrics(y_true,y_pred,y_sp):
    rmse=float(np.sqrt(mean_squared_error(y_true,y_pred))); rmse_sp=float(np.sqrt(mean_squared_error(y_true,y_sp)))
    return {"r2":round(float(r2_score(y_true,y_pred)),4),"mae":round(float(mean_absolute_error(y_true,y_pred)),1),
            "rmse":round(rmse,1),"skill_vs_sp":round(1.0-rmse/rmse_sp if rmse_sp>0 else 0.0,4)}

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; print(f"Device: {device}")
    con=duckdb.connect(DB_PATH,read_only=True); print("Loading Banten data...")
    df=con.execute(build_sql()).fetchdf(); con.close()
    df[TIME_COL]=pd.to_datetime(df[TIME_COL]); df=add_features(df)
    du=df[df[TARGET_POINT].between(0,1400)&df["sun_gt5_t60"]].copy(); print(f"Rows: {len(du):,}")
    tr_m,va_m,te_m=split_masks(du); print(f"train={tr_m.sum():,} val={va_m.sum():,} test={te_m.sum():,}")
    f1=[f for f in F1_FEATURES if f in du.columns]
    x_tr,x_va,x_te=du.loc[tr_m,f1],du.loc[va_m,f1],du.loc[te_m,f1]
    y_tr,y_va,y_te=du.loc[tr_m,TARGET_POINT],du.loc[va_m,TARGET_POINT],du.loc[te_m,TARGET_POINT]
    y_sp=np.clip(du.loc[te_m,"smart_persist"].values,PRED_MIN,PRED_MAX)
    results=[]

    print(f"\n{'='*60}\nGBM MODELS\n{'='*60}")
    print("  Training CatBoost...")
    cb=train_catboost(x_tr,y_tr,x_va,y_va)
    pred=np.clip(cb.predict(x_te.astype(float).values),PRED_MIN,PRED_MAX); m=metrics(y_te,pred,y_sp)
    m.update({"model":"catboost","seed":0,"type":"GBM"}); results.append(m)
    print(f"  CatBoost  R2={m['r2']:.4f} MAE={m['mae']:.1f} RMSE={m['rmse']:.1f} iter={cb.get_best_iteration()}")
    print("  Training LightGBM...")
    lp=train_lgbm(x_tr,y_tr,x_va,y_va)
    pred=np.clip(lp.predict(x_te),PRED_MIN,PRED_MAX); m=metrics(y_te,pred,y_sp)
    m.update({"model":"lgbm","seed":0,"type":"GBM"}); results.append(m)
    print(f"  LightGBM  R2={m['r2']:.4f} MAE={m['mae']:.1f} RMSE={m['rmse']:.1f}")

    print(f"\n{'='*60}\nDL MODELS ({len(DL_SEEDS)} seeds)\n{'='*60}")
    imp=SimpleImputer(strategy="median"); x_tr_i=imp.fit_transform(x_tr); x_va_i=imp.transform(x_va); x_te_i=imp.transform(x_te)
    sc=StandardScaler(); x_tr_s=sc.fit_transform(x_tr_i); x_va_s=sc.transform(x_va_i); x_te_s=sc.transform(x_te_i)
    n_feat=x_tr_s.shape[1]
    to_t=lambda a: torch.from_numpy(a.astype(np.float32)).unsqueeze(1)
    xt_t,xv_t,xe_t=to_t(x_tr_s),to_t(x_va_s),to_t(x_te_s)
    yt_t=torch.from_numpy(y_tr.values.astype(np.float32)); yv_t=torch.from_numpy(y_va.values.astype(np.float32)); ye_t=torch.zeros(len(y_te))
    train_loader=DataLoader(TensorDataset(xt_t,yt_t),batch_size=128,shuffle=True)
    val_loader=DataLoader(TensorDataset(xv_t,yv_t),batch_size=256)
    test_loader=DataLoader(TensorDataset(xe_t,ye_t),batch_size=256)
    for mn,Cls in [("lstm",LSTMModel),("mlp",MLPModel),("transformer",TransformerModel)]:
        sr=[]
        for seed in DL_SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            model=Cls(n_feat).to(device); model=train_dl(model,train_loader,val_loader,device)
            pred=predict_dl(model,test_loader,device); m=metrics(y_te,pred,y_sp)
            m.update({"model":mn,"seed":seed,"type":"DL"}); results.append(m); sr.append(m["r2"])
        print(f"  {mn:<12s} R2={np.mean(sr):.4f} +/- {np.std(sr):.4f}  (seeds: {[f'{r:.4f}' for r in sr]})")

    dfr=pd.DataFrame(results); dfr.to_csv(OUTPUT_DIR/"arm_B_results.csv",index=False)
    srows=[]
    for mname,grp in dfr.groupby("model"):
        srows.append({"model":mname,"type":grp["type"].iloc[0],"r2_mean":round(grp["r2"].mean(),4),
                      "r2_std":round(grp["r2"].std(),4),"mae_mean":round(grp["mae"].mean(),1),
                      "rmse_mean":round(grp["rmse"].mean(),1),"n_seeds":len(grp)})
    pd.DataFrame(srows).sort_values("r2_mean",ascending=False).to_csv(OUTPUT_DIR/"arm_B_summary.csv",index=False)

    print(f"\n{'='*60}\nSUMMARY — R8 Arm B BANTEN (point t+60, test 2025)\n{'='*60}")
    cb_r2=dfr.loc[dfr["model"]=="catboost","r2"].values[0]; lgb_r2=dfr.loc[dfr["model"]=="lgbm","r2"].values[0]
    print(f"  CatBoost R2={cb_r2:.4f}"); print(f"  LightGBM R2={lgb_r2:.4f} (delta vs CB: {lgb_r2-cb_r2:+.4f})")
    for mname in ["lstm","mlp","transformer"]:
        grp=dfr[dfr["model"]==mname]; print(f"  DL {mname:<12s} R2={grp['r2'].mean():.4f} +/- {grp['r2'].std():.4f} (delta vs CB: {grp['r2'].mean()-cb_r2:+.4f})")
    print(f"\n  -> outputs: {OUTPUT_DIR}/arm_B_results.csv, arm_B_summary.csv (angka nyata Tabel 2c)")

if __name__=="__main__": main()
