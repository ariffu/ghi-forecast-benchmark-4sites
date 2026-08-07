#!/usr/bin/env python3
"""
R1 Harmonised Benchmark — BANTEN GHI 1-hour-ahead forecasting.
Replikasi PERSIS konfigurasi R1 Bengkulu untuk Banten dari solar_features_base (banten.duckdb).
Model: LightGBM residual (PRIMARY) | CatBoost direct (SENSITIVITY)
Features: §3.2 lean 50 fitur | Targets: point t+60 & avg t+10..t+60
Data 2022-2025 | Split train<2024/val2024/test2025 | filter sun>5 anchor+t60 | clearsky 1100*sin(elev)
Run: & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_banten_R1_benchmark.py
CATATAN: skrip disimpan di WORKSPACE (C:\\Users\\ariff\\Duckdb_Banten) — vault Obsidian
         meng-quarantine .py via plugin remotely-save.
"""
import warnings
from pathlib import Path
import duckdb, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

DB_PATH="banten.duckdb"; OUTPUT_DIR=Path("outputs_R1_banten"); OUTPUT_DIR.mkdir(exist_ok=True)
STATION_LAT_DEG,STATION_LON_DEG,WIB_MERIDIAN_DEG=-6.26147,106.7509,105.0
TIME_COL="ts_wib"; TRAIN_END,VALID_END="2024-01-01","2025-01-01"
PRED_MIN,PRED_MAX=0.0,1400.0; RANDOM_STATE=42
FOLDS=[("2023-01-01","2023-07-01"),("2023-07-01","2024-01-01"),("2024-01-01","2024-07-01"),
       ("2024-07-01","2025-01-01"),("2025-01-01",None)]; ES_MONTHS=3
FEATURES_GHI=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m"]
FEATURES_KT=["kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m","kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m"]
FEATURES_CLP=["clp_cot","clp_cot_lag_10m","clp_cot_lag_20m","clp_cot_lag_30m","clp_cot_lag_60m",
    "clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m","clp_cot_roll_180m_mean",
    "accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present"]
FEATURES_TIME=["hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos"]
FEATURES_FUTURE=["ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]
FEATURES=FEATURES_GHI+FEATURES_KT+FEATURES_CLP+FEATURES_TIME+FEATURES_FUTURE
assert len(FEATURES)==50,len(FEATURES)
TARGET_POINT,TARGET_AVG,DELTA_POINT,DELTA_AVG="ghi_point_t60","ghi_avg_t10_t60","delta_point","delta_avg"

def build_sql():
    return """
    WITH base AS (
        SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m,
               cloud_top_temp AS clp_ctt_k, cloud_eff_radius AS clp_cer,
               CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present
        FROM solar_features_base
    ), with_kt AS (
        SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base
    ), w AS (
        SELECT *,
          LAG(ghi_now,1) OVER o AS ghi_lag_10m, LAG(ghi_now,2) OVER o AS ghi_lag_20m,
          LAG(ghi_now,3) OVER o AS ghi_lag_30m, LAG(ghi_now,6) OVER o AS ghi_lag_60m,
          LAG(ghi_now,12) OVER o AS ghi_lag_120m, LAG(ghi_now,18) OVER o AS ghi_lag_180m,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
          LAG(kt_point,1) OVER o AS kt_lag_10m, LAG(kt_point,2) OVER o AS kt_lag_20m,
          LAG(kt_point,3) OVER o AS kt_lag_30m, LAG(kt_point,6) OVER o AS kt_lag_60m,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
          LAG(clp_cot,1) OVER o AS clp_cot_lag_10m, LAG(clp_cot,2) OVER o AS clp_cot_lag_20m,
          LAG(clp_cot,3) OVER o AS clp_cot_lag_30m, LAG(clp_cot,6) OVER o AS clp_cot_lag_60m,
          AVG(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
          LEAD(ghi_now,1) OVER o AS ghi_lead_10m, LEAD(ghi_now,2) OVER o AS ghi_lead_20m,
          LEAD(ghi_now,3) OVER o AS ghi_lead_30m, LEAD(ghi_now,4) OVER o AS ghi_lead_40m,
          LEAD(ghi_now,5) OVER o AS ghi_lead_50m, LEAD(ghi_now,6) OVER o AS ghi_lead_60m
        FROM with_kt WINDOW o AS (ORDER BY ts_wib)
    )
    SELECT * FROM w WHERE solar_elev_deg>5 AND ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib
    """

def solar_elevation_deg(ts,lat=STATION_LAT_DEG,lon=STATION_LON_DEG,meridian=WIB_MERIDIAN_DEG):
    idx=pd.DatetimeIndex(ts); doy=idx.dayofyear.values.astype(float)
    h=idx.hour.values.astype(float)+idx.minute.values.astype(float)/60.0
    decl=23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0))
    ha=((h+4.0*(lon-meridian)/60.0)-12.0)*15.0
    sin_e=(np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(decl))+np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(decl))*np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e,-1,1)))
def clearsky_simple(e): return 1100.0*np.maximum(np.sin(np.deg2rad(e)),0.0)

def add_features(df):
    out=df.copy(); ts=pd.DatetimeIndex(out[TIME_COL])
    out["kt_now"]=out["ghi_now"].values/np.maximum(clearsky_simple(out["solar_elev_deg"].values.astype(float)),20.0)
    out["clp_cot_delta_10m"]=out["clp_cot"]-out["clp_cot_lag_10m"]; out["clp_cot_delta_30m"]=out["clp_cot"]-out["clp_cot_lag_30m"]
    out["clp_cot_delta_60m"]=out["clp_cot"]-out["clp_cot_lag_60m"]; out["clp_cot_delta_180m"]=out["clp_cot"]-out["clp_cot_roll_180m_mean"]
    out["accel_ghi_20m"]=out["ghi_now"]-2*out["ghi_lag_10m"]+out["ghi_lag_20m"]
    out["accel_kt_20m"]=out["kt_now"]-2*out["kt_lag_10m"]+out["kt_lag_20m"]
    out["accel_clp_cot_20m"]=out["clp_cot"]-2*out["clp_cot_lag_10m"]+out["clp_cot_lag_20m"]
    out["ghi_delta_10m"]=out["ghi_now"]-out["ghi_lag_10m"]; out["ghi_delta_60m"]=out["ghi_now"]-out["ghi_lag_60m"]
    hh=ts.hour.values.astype(float)+ts.minute.values.astype(float)/60.0; doy=ts.dayofyear.values.astype(float); mo=ts.month.values.astype(float)
    out["hour_sin"]=np.sin(2*np.pi*hh/24); out["hour_cos"]=np.cos(2*np.pi*hh/24)
    out["doy_sin"]=np.sin(2*np.pi*doy/365.25); out["doy_cos"]=np.cos(2*np.pi*doy/365.25)
    out["month_sin"]=np.sin(2*np.pi*mo/12); out["month_cos"]=np.cos(2*np.pi*mo/12)
    elev_t60=solar_elevation_deg(out[TIME_COL]+pd.Timedelta(minutes=60))
    out["elev_sin_t60"]=np.maximum(np.sin(np.deg2rad(elev_t60)),0.0); out["ghi_cs_t60"]=clearsky_simple(elev_t60)
    cs=[clearsky_simple(solar_elevation_deg(out[TIME_COL]+pd.Timedelta(minutes=s*10))) for s in range(1,7)]
    out["ghi_cs_avg_t10_t60"]=np.column_stack(cs).mean(axis=1)
    out["smart_persist"]=out["kt_now"]*out["ghi_cs_t60"]; out["smart_persist_avg"]=out["kt_now"]*out["ghi_cs_avg_t10_t60"]
    out[TARGET_POINT]=out["ghi_lead_60m"].copy()
    lc=["ghi_lead_10m","ghi_lead_20m","ghi_lead_30m","ghi_lead_40m","ghi_lead_50m","ghi_lead_60m"]; leads=out[lc]
    av=leads.notna().all(axis=1)&leads.apply(lambda c:c.between(0,1400)).all(axis=1)
    out[TARGET_AVG]=np.where(av,leads.mean(axis=1),np.nan)
    out["sun_gt5_t60"]=out["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    out[DELTA_POINT]=out[TARGET_POINT]-out["ghi_now"]; out[DELTA_AVG]=out[TARGET_AVG]-out["ghi_now"]
    return out

def split_masks(df):
    ts=df[TIME_COL]
    return (ts<pd.Timestamp(TRAIN_END),(ts>=pd.Timestamp(TRAIN_END))&(ts<pd.Timestamp(VALID_END)),ts>=pd.Timestamp(VALID_END))
def lgbm_pipe(seed=RANDOM_STATE):
    reg=lgb.LGBMRegressor(objective="regression",n_estimators=6000,learning_rate=0.02,num_leaves=39,min_child_samples=70,
        reg_alpha=0.2,reg_lambda=2.5,colsample_bytree=0.82,subsample=0.85,subsample_freq=1,random_state=seed,
        n_jobs=-1,force_col_wise=True,verbosity=-1)
    return Pipeline([("imp",SimpleImputer(strategy="median",keep_empty_features=True)),("m",reg)])
def fit_lgbm(pipe,xt,yt,xe,ye):
    pipe.fit(xt,yt,m__eval_set=[(xe,ye)],m__eval_metric="rmse",m__callbacks=[lgb.early_stopping(150,verbose=False)]); return pipe
def catboost_model(seed=RANDOM_STATE):
    return CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",
        random_seed=seed,verbose=False,thread_count=-1,allow_writing_files=False)
def fit_catboost(m,xt,yt,xe,ye):
    m.fit(xt.astype(float).values,yt.astype(float).values,eval_set=(xe.astype(float).values,ye.astype(float).values),early_stopping_rounds=150); return m
def metr(y,p,sp,model,target):
    rmse=float(np.sqrt(mean_squared_error(y,p))); spr=float(np.sqrt(mean_squared_error(y,sp)))
    return {"model":model,"target":target,"n":len(y),"r2":round(float(r2_score(y,p)),4),
            "mae":round(float(mean_absolute_error(y,p)),1),"rmse":round(rmse,1),
            "skill_vs_sp":round(1.0-rmse/spr if spr>0 else float("nan"),4)}

def main():
    con=duckdb.connect(DB_PATH,read_only=True); print("Loading Banten data...")
    df=con.execute(build_sql()).fetchdf(); con.close()
    df[TIME_COL]=pd.to_datetime(df[TIME_COL]); df=add_features(df)
    print(f"Total rows: {len(df):,} | {df[TIME_COL].min().date()}..{df[TIME_COL].max().date()}")
    df_pt=df[df[TARGET_POINT].between(0,1400)&df["sun_gt5_t60"]].copy()
    df_av=df[df[TARGET_AVG].notna()&df["sun_gt5_t60"]].copy()
    print(f"Rows point={len(df_pt):,} avg={len(df_av):,}")
    results=[]
    for du,tc,dc,spc,tg in [(df_pt,TARGET_POINT,DELTA_POINT,"smart_persist","point_t60"),
                            (df_av,TARGET_AVG,DELTA_AVG,"smart_persist_avg","avg_t10_t60")]:
        print(f"\n{'='*56}\nTARGET: {tg}\n{'='*56}")
        tm,vm,em=split_masks(du); print(f"  train={tm.sum():,} val={vm.sum():,} test={em.sum():,}")
        xt,xv,xe=du.loc[tm,FEATURES],du.loc[vm,FEATURES],du.loc[em,FEATURES]
        yt,yv,ye=du.loc[tm,tc],du.loc[vm,tc],du.loc[em,tc]
        ydt,ydv=du.loc[tm,dc],du.loc[vm,dc]; gne=du.loc[em,"ghi_now"].values
        spe=np.clip(du.loc[em,spc].values,PRED_MIN,PRED_MAX)
        results.append(metr(ye,spe,spe,"smart_persistence",tg)); print(f"  SP R2={results[-1]['r2']:.4f} MAE={results[-1]['mae']:.1f}")
        lg=lgbm_pipe(); fit_lgbm(lg,xt,ydt,xv,ydv); bi=lg.named_steps["m"].best_iteration_
        pr=np.clip(gne+lg.predict(xe),PRED_MIN,PRED_MAX); r=metr(ye,pr,spe,"lgbm_residual",tg); r["best_iter"]=bi; results.append(r)
        print(f"  LGBM iter={bi} R2={r['r2']:.4f} MAE={r['mae']:.1f} skill={r['skill_vs_sp']:.4f}")
        cb=catboost_model(); fit_catboost(cb,xt,yt,xv,yv); bic=cb.get_best_iteration()
        cp=np.clip(cb.predict(xe.astype(float).values),PRED_MIN,PRED_MAX); r=metr(ye,cp,spe,"catboost_direct",tg); r["best_iter"]=bic; results.append(r)
        print(f"  CatB iter={bic} R2={r['r2']:.4f} MAE={r['mae']:.1f} skill={r['skill_vs_sp']:.4f}")
    pd.DataFrame(results).to_csv(OUTPUT_DIR/"ghi_1h_R1_results.csv",index=False)
    print(f"\n{'='*56}\nWALK-FORWARD 5-FOLD (lgbm_residual x point)\n{'='*56}"); wf=[]
    for fi,(t0,t1) in enumerate(FOLDS,1):
        s=pd.Timestamp(t0); e=pd.Timestamp(t1) if t1 else pd.Timestamp("2099-01-01")
        c=df_pt[TIME_COL]; tra=df_pt[c<s]; te=df_pt[(c>=s)&(c<e)]
        if len(tra)<5000 or len(te)<100: print(f"  Fold {fi}: skip"); continue
        cut=s-pd.DateOffset(months=ES_MONTHS); tre=tra[tra[TIME_COL]<cut]; tes=tra[tra[TIME_COL]>=cut]
        p=lgbm_pipe(); fit_lgbm(p,tre[FEATURES],tre[DELTA_POINT],tes[FEATURES],tes[DELTA_POINT]); bi=p.named_steps["m"].best_iteration_
        pr=np.clip(te["ghi_now"].values+p.predict(te[FEATURES]),PRED_MIN,PRED_MAX)
        yv=te[TARGET_POINT].values; sp=np.clip(te["smart_persist"].values,PRED_MIN,PRED_MAX)
        rmse=float(np.sqrt(mean_squared_error(yv,pr))); spr=float(np.sqrt(mean_squared_error(yv,sp)))
        per=t0[:7]+".."+(t1[:7] if t1 else "end")
        wf.append({"fold":fi,"period":per,"n_train_eff":len(tre),"n_test":len(te),"best_iter":bi,
                   "r2":float(r2_score(yv,pr)),"mae":float(mean_absolute_error(yv,pr)),"rmse":rmse,
                   "skill_vs_sp":1.0-rmse/spr if spr>0 else float("nan")})
        print(f"  Fold {fi} [{per}] n_tr={len(tre):,} n_te={len(te):,} iter={bi} R2={wf[-1]['r2']:.4f} MAE={wf[-1]['mae']:.1f} skill={wf[-1]['skill_vs_sp']:.4f}")
    wfdf=pd.DataFrame(wf); wfdf.to_csv(OUTPUT_DIR/"ghi_1h_R1_wf_folds.csv",index=False)
    print("\n--- WF summary ---")
    for c in ["r2","mae","rmse","skill_vs_sp"]: print(f"  {c:<12}: {wfdf[c].mean():.4f} +/- {wfdf[c].std():.4f}")
    print("\n--- HEADLINE (test 2025) ---")
    print(pd.DataFrame(results)[["model","target","r2","mae","rmse","skill_vs_sp"]].to_string(index=False))
    print(f"\nOutputs -> {OUTPUT_DIR}/")

if __name__=="__main__": main()
