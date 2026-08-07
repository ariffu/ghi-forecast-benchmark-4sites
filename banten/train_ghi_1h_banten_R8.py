#!/usr/bin/env python3
"""
R8 BANTEN — Arm A (meteo redundancy) + Arm B (GBM fair-play).
Adaptasi konfig R8 batch Kalbar (train_ghi_1h_r8_batch.py) ke Banten,
dibangun dari solar_features_base (F1 = 50 fitur R1 identik; F2 = F1 + 5 meteo).

  Arm A : F1 (50) vs F2 (55 = F1 + temp/rh/ws/rain/pressure), CatBoost, 2 target
          -> apakah surface-met menambah akurasi? (Kalbar: redundan, dR2 ~ 0)
  Arm B : CatBoost vs LightGBM di F1, target titik
          -> GBM fair-play (Kalbar: CatBoost > LGBM)
  Model/split/clearsky identik R1. Test 2025.
Run: & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_banten_R8.py
"""
import warnings
from pathlib import Path
import duckdb, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

DB_PATH="banten.duckdb"; OUTPUT_DIR=Path("outputs_R8_Banten"); OUTPUT_DIR.mkdir(exist_ok=True)
LAT,LON,MER=-6.26147,106.7509,105.0; TIME_COL="ts_wib"; RS=42
F1=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m","kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
    "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m","clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
    "clp_cot_lag_30m","clp_cot_lag_60m","clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m",
    "clp_cot_roll_180m_mean","accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present",
    "hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]
METEO=["temp_air_c","humidity_pct","wind_speed_ms","rainfall_mm","pressure_hpa"]
F2=F1+METEO
TP,TA="ghi_point_t60","ghi_avg_t10_t60"

def build_sql():
    return """
    WITH base AS (
        SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m, cloud_top_temp AS clp_ctt_k,
               cloud_eff_radius AS clp_cer, CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present,
               temp AS temp_air_c, rh AS humidity_pct, ws AS wind_speed_ms, rain AS rainfall_mm, pressure AS pressure_hpa
        FROM solar_features_base
    ), wk AS (SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base
    ), w AS (
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
          LEAD(ghi_now,1) OVER o AS ghi_lead_10m, LEAD(ghi_now,2) OVER o AS ghi_lead_20m, LEAD(ghi_now,3) OVER o AS ghi_lead_30m,
          LEAD(ghi_now,4) OVER o AS ghi_lead_40m, LEAD(ghi_now,5) OVER o AS ghi_lead_50m, LEAD(ghi_now,6) OVER o AS ghi_lead_60m
        FROM wk WINDOW o AS (ORDER BY ts_wib))
    SELECT * FROM w WHERE solar_elev_deg>5 AND ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib
    """

def elev(ts,lat=LAT,lon=LON,mer=MER):
    idx=pd.DatetimeIndex(ts); doy=idx.dayofyear.values.astype(float)
    h=idx.hour.values.astype(float)+idx.minute.values.astype(float)/60.0
    d=23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0)); ha=((h+4.0*(lon-mer)/60.0)-12.0)*15.0
    se=np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(d))+np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(d))*np.cos(np.deg2rad(ha))
    return np.degrees(np.arcsin(np.clip(se,-1,1)))
def cs(e): return 1100.0*np.maximum(np.sin(np.deg2rad(e)),0.0)

def add_feat(df):
    o=df.copy(); ts=pd.DatetimeIndex(o[TIME_COL])
    o["kt_now"]=o["ghi_now"].values/np.maximum(cs(o["solar_elev_deg"].values.astype(float)),20.0)
    o["clp_cot_delta_10m"]=o["clp_cot"]-o["clp_cot_lag_10m"]; o["clp_cot_delta_30m"]=o["clp_cot"]-o["clp_cot_lag_30m"]
    o["clp_cot_delta_60m"]=o["clp_cot"]-o["clp_cot_lag_60m"]; o["clp_cot_delta_180m"]=o["clp_cot"]-o["clp_cot_roll_180m_mean"]
    o["accel_ghi_20m"]=o["ghi_now"]-2*o["ghi_lag_10m"]+o["ghi_lag_20m"]; o["accel_kt_20m"]=o["kt_now"]-2*o["kt_lag_10m"]+o["kt_lag_20m"]
    o["accel_clp_cot_20m"]=o["clp_cot"]-2*o["clp_cot_lag_10m"]+o["clp_cot_lag_20m"]
    o["ghi_delta_10m"]=o["ghi_now"]-o["ghi_lag_10m"]; o["ghi_delta_60m"]=o["ghi_now"]-o["ghi_lag_60m"]
    hh=ts.hour.values.astype(float)+ts.minute.values.astype(float)/60.0; doy=ts.dayofyear.values.astype(float); mo=ts.month.values.astype(float)
    o["hour_sin"]=np.sin(2*np.pi*hh/24); o["hour_cos"]=np.cos(2*np.pi*hh/24)
    o["doy_sin"]=np.sin(2*np.pi*doy/365.25); o["doy_cos"]=np.cos(2*np.pi*doy/365.25)
    o["month_sin"]=np.sin(2*np.pi*mo/12); o["month_cos"]=np.cos(2*np.pi*mo/12)
    et=elev(o[TIME_COL]+pd.Timedelta(minutes=60)); o["elev_sin_t60"]=np.maximum(np.sin(np.deg2rad(et)),0.0); o["ghi_cs_t60"]=cs(et)
    csa=[cs(elev(o[TIME_COL]+pd.Timedelta(minutes=s*10))) for s in range(1,7)]; o["ghi_cs_avg"]=np.column_stack(csa).mean(axis=1)
    o["smart_persist"]=o["kt_now"]*o["ghi_cs_t60"]; o["smart_persist_avg"]=o["kt_now"]*o["ghi_cs_avg"]
    o[TP]=o["ghi_lead_60m"].copy()
    lc=["ghi_lead_10m","ghi_lead_20m","ghi_lead_30m","ghi_lead_40m","ghi_lead_50m","ghi_lead_60m"]; L=o[lc]
    av=L.notna().all(axis=1)&L.apply(lambda c:c.between(0,1400)).all(axis=1); o[TA]=np.where(av,L.mean(axis=1),np.nan)
    o["sun_gt5_t60"]=o["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    return o

def split_masks(df):
    ts=df[TIME_COL]
    return (ts<pd.Timestamp("2024-01-01"),(ts>=pd.Timestamp("2024-01-01"))&(ts<pd.Timestamp("2025-01-01")),ts>=pd.Timestamp("2025-01-01"))
def lgbm_pipe():
    reg=lgb.LGBMRegressor(objective="regression",n_estimators=6000,learning_rate=0.02,num_leaves=39,min_child_samples=70,
        reg_alpha=0.2,reg_lambda=2.5,colsample_bytree=0.82,subsample=0.85,subsample_freq=1,random_state=RS,n_jobs=-1,force_col_wise=True,verbosity=-1)
    return Pipeline([("imp",SimpleImputer(strategy="median",keep_empty_features=True)),("m",reg)])
def cbm():
    return CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",random_seed=RS,verbose=False,thread_count=-1,allow_writing_files=False)
def ev(y,p,sp):
    rmse=float(np.sqrt(mean_squared_error(y,p))); spr=float(np.sqrt(mean_squared_error(y,sp)))
    return {"r2":round(float(r2_score(y,p)),4),"mae":round(float(mean_absolute_error(y,p)),1),
            "rmse":round(rmse,1),"skill_vs_sp":round(1.0-rmse/spr if spr>0 else 0.0,4)}

def main():
    con=duckdb.connect(DB_PATH,read_only=True); print("Loading Banten...")
    df=con.execute(build_sql()).fetchdf(); con.close()
    df[TIME_COL]=pd.to_datetime(df[TIME_COL]); df=add_feat(df)
    dpt=df[df[TP].between(0,1400)&df["sun_gt5_t60"]].copy(); dav=df[df[TA].notna()&df["sun_gt5_t60"]].copy()
    print(f"Rows point={len(dpt):,} avg={len(dav):,}")

    print("\n"+"="*60+"\nARM A: METEO REDUNDANCY (F1 vs F2), CatBoost\n"+"="*60)
    A=[]
    for tn,tc,du in [("point_t60",TP,dpt),("avg_t10_t60",TA,dav)]:
        tm,vm,em=split_masks(du); spc="smart_persist" if "point" in tn else "smart_persist_avg"
        spe=np.clip(du.loc[em,spc].values,0,1400)
        for fn,fs in [("F1",F1),("F2",F2)]:
            xt=du.loc[tm,fs].fillna(du.loc[tm,fs].median()); xv=du.loc[vm,fs].fillna(du.loc[tm,fs].median()); xe=du.loc[em,fs].fillna(du.loc[tm,fs].median())
            m=cbm(); m.fit(xt,du.loc[tm,tc],eval_set=(xv,du.loc[vm,tc]),early_stopping_rounds=150,verbose=False)
            r=ev(du.loc[em,tc].values,np.clip(m.predict(xe),0,1400),spe); r.update({"target":tn,"features":fn,"n_feat":len(fs),"model":"catboost"}); A.append(r)
            print(f"  {tn} x {fn} ({len(fs)}f): R2={r['r2']:.4f} MAE={r['mae']:.1f}")
        d=A[-1]["r2"]-A[-2]["r2"]; print(f"    -> dR2(F2-F1) = {d:+.4f}  ({'meteo REDUNDAN' if abs(d)<0.005 else 'meteo membantu'})")
    pd.DataFrame(A).to_csv(OUTPUT_DIR/"arm_A_results.csv",index=False)

    print("\n"+"="*60+"\nARM B: GBM FAIR-PLAY (CatBoost vs LGBM), F1, titik\n"+"="*60)
    B=[]; tm,vm,em=split_masks(dpt); spe=np.clip(dpt.loc[em,"smart_persist"].values,0,1400)
    xt=dpt.loc[tm,F1].fillna(dpt.loc[tm,F1].median()); xv=dpt.loc[vm,F1].fillna(dpt.loc[tm,F1].median()); xe=dpt.loc[em,F1].fillna(dpt.loc[tm,F1].median())
    yt,yv,ye=dpt.loc[tm,TP],dpt.loc[vm,TP],dpt.loc[em,TP].values
    cb=cbm(); cb.fit(xt,yt,eval_set=(xv,yv),early_stopping_rounds=150,verbose=False)
    r=ev(ye,np.clip(cb.predict(xe),0,1400),spe); r.update({"model":"catboost"}); B.append(r); print(f"  catboost: R2={r['r2']:.4f} MAE={r['mae']:.1f}")
    lg=lgbm_pipe(); lg.fit(xt,yt,m__eval_set=[(xv,yv)],m__eval_metric="rmse",m__callbacks=[lgb.early_stopping(150,verbose=False)])
    r=ev(ye,np.clip(lg.predict(xe),0,1400),spe); r.update({"model":"lgbm"}); B.append(r); print(f"  lgbm    : R2={r['r2']:.4f} MAE={r['mae']:.1f}")
    dB=B[0]["r2"]-B[1]["r2"]; print(f"    -> dR2(CatBoost-LGBM) = {dB:+.4f}  ({'CatBoost unggul' if dB>0 else 'LGBM unggul'})")
    pd.DataFrame(B).to_csv(OUTPUT_DIR/"arm_B_results.csv",index=False)

    print("\n"+"="*60+"\nOK BANTEN R8 COMPLETE\n"+"="*60)
    print(f"Outputs -> {OUTPUT_DIR}/")

if __name__=="__main__": main()
