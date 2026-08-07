#!/usr/bin/env python3
"""
R8 §4.4 — ISOLASI MURNI F1 vs F2 (meteo) untuk BANTEN.
Menjawab: apakah +meteo pd gap-close (kumulatif 82->87) benar kontribusi meteo,
atau artefak interaksi dgn full-lags/variabilitas? -> uji ISOLASI: lean-50 vs lean-50+5 meteo saja.

Rigor utk paper:
  - Dua model: CatBoost + LightGBM (robust lintas-algoritma)
  - Dua target: point t+60 & avg t+10..t+60
  - Lapor VAL 2024 & TEST 2025 (F2>F1 di keduanya = bukan overfit test)
  - Ablation per-fitur: lean-50 + SATU fitur meteo (temp/rh/ws/rain/pressure) -> pendorong utama
Basis harmonis identik R1. Data 2022-2025.
Run: & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_banten_R8_meteo_isolation.py
"""
import warnings
from pathlib import Path
import duckdb, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

DB="banten.duckdb"; OUT=Path("outputs_R8_meteo_isolation_banten"); OUT.mkdir(exist_ok=True)
LAT,LON,MER=-6.26147,106.7509,105.0; TIME="ts_wib"; RS=42
TP,TA="ghi_point_t60","ghi_avg_t10_t60"
F1=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m","kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
    "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m","clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
    "clp_cot_lag_30m","clp_cot_lag_60m","clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m",
    "clp_cot_roll_180m_mean","accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present",
    "hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]
METEO=["temp_air_c","humidity_pct","wind_speed_ms","rainfall_mm","pressure_hpa"]

def build_sql():
    return """
    WITH base AS (
        SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m, cloud_top_temp AS clp_ctt_k,
               cloud_eff_radius AS clp_cer, CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present,
               temp AS temp_air_c, rh AS humidity_pct, ws AS wind_speed_ms, rain AS rainfall_mm, pressure AS pressure_hpa
        FROM solar_features_base
    ), wk AS (SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base),
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
          LEAD(ghi_now,1) OVER o AS ghi_lead_10m, LEAD(ghi_now,2) OVER o AS ghi_lead_20m, LEAD(ghi_now,3) OVER o AS ghi_lead_30m,
          LEAD(ghi_now,4) OVER o AS ghi_lead_40m, LEAD(ghi_now,5) OVER o AS ghi_lead_50m, LEAD(ghi_now,6) OVER o AS ghi_lead_60m
        FROM wk WINDOW o AS (ORDER BY ts_wib))
    SELECT * FROM w WHERE solar_elev_deg>5 AND ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib
    """

def elev(ts):
    idx=pd.DatetimeIndex(ts); doy=idx.dayofyear.values.astype(float)
    h=idx.hour.values.astype(float)+idx.minute.values.astype(float)/60.0
    d=23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0)); ha=(h+4.0*(LON-MER)/60.0-12.0)*15.0
    se=np.sin(np.deg2rad(LAT))*np.sin(np.deg2rad(d))+np.cos(np.deg2rad(LAT))*np.cos(np.deg2rad(d))*np.cos(np.deg2rad(ha))
    return np.degrees(np.arcsin(np.clip(se,-1,1)))
def cs(e): return 1100.0*np.maximum(np.sin(np.deg2rad(e)),0.0)

def add_feat(df):
    o=df.copy(); ts=pd.DatetimeIndex(o[TIME])
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
    et=elev(o[TIME]+pd.Timedelta(minutes=60)); o["elev_sin_t60"]=np.maximum(np.sin(np.deg2rad(et)),0.0); o["ghi_cs_t60"]=cs(et)
    csa=[cs(elev(o[TIME]+pd.Timedelta(minutes=s*10))) for s in range(1,7)]
    o["smart_persist"]=o["kt_now"]*o["ghi_cs_t60"]; o["smart_persist_avg"]=o["kt_now"]*np.column_stack(csa).mean(axis=1)
    o[TP]=o["ghi_lead_60m"].copy()
    lc=["ghi_lead_10m","ghi_lead_20m","ghi_lead_30m","ghi_lead_40m","ghi_lead_50m","ghi_lead_60m"]; L=o[lc]
    av=L.notna().all(axis=1)&L.apply(lambda c:c.between(0,1400)).all(axis=1); o[TA]=np.where(av,L.mean(axis=1),np.nan)
    o["sun_gt5_t60"]=o["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    return o

def split_masks(df):
    ts=df[TIME]
    return (ts<pd.Timestamp("2024-01-01"),(ts>=pd.Timestamp("2024-01-01"))&(ts<pd.Timestamp("2025-01-01")),ts>=pd.Timestamp("2025-01-01"))
def cbm():
    return CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",random_seed=RS,verbose=False,thread_count=-1,allow_writing_files=False)
def lgbmp():
    return Pipeline([("imp",SimpleImputer(strategy="median",keep_empty_features=True)),
        ("m",lgb.LGBMRegressor(objective="regression",n_estimators=6000,learning_rate=0.02,num_leaves=39,min_child_samples=70,
            reg_alpha=0.2,reg_lambda=2.5,colsample_bytree=0.82,subsample=0.85,subsample_freq=1,random_state=RS,n_jobs=-1,force_col_wise=True,verbosity=-1))])
def fit_eval(feats,du,tr,va,te,tgt,algo):
    xt=du.loc[tr,feats]; xv=du.loc[va,feats]; xe=du.loc[te,feats]; yt=du.loc[tr,tgt]
    if algo=="catboost":
        m=cbm(); m.fit(xt.fillna(xt.median()),yt,eval_set=(xv.fillna(xt.median()),du.loc[va,tgt]),early_stopping_rounds=150,verbose=False)
        pv=np.clip(m.predict(xv.fillna(xt.median())),0,1400); pe=np.clip(m.predict(xe.fillna(xt.median())),0,1400)
    else:
        m=lgbmp(); m.fit(xt,yt,m__eval_set=[(m.named_steps["imp"].fit_transform(xv),du.loc[va,tgt])],m__eval_metric="rmse",m__callbacks=[lgb.early_stopping(150,verbose=False)])
        pv=np.clip(m.predict(xv),0,1400); pe=np.clip(m.predict(xe),0,1400)
    return (r2_score(du.loc[va,tgt],pv), r2_score(du.loc[te,tgt],pe),
            mean_absolute_error(du.loc[te,tgt],pe), np.sqrt(mean_squared_error(du.loc[te,tgt],pe)))

def main():
    con=duckdb.connect(DB,read_only=True); print("Loading..."); df=con.execute(build_sql()).fetchdf(); con.close()
    df[TIME]=pd.to_datetime(df[TIME]); df=add_feat(df)
    F2=F1+METEO
    rows=[]
    for tgt,tname in [(TP,"point_t60"),(TA,"avg_t10_t60")]:
        du=df[df[tgt].between(0,1400)&df["sun_gt5_t60"]].copy() if tgt==TP else df[df[tgt].notna()&df["sun_gt5_t60"]].copy()
        tr,va,te=split_masks(du)
        print(f"\n{'='*64}\nISOLASI F1 vs F2 — {tname}  (n={len(du):,})\n{'='*64}")
        for algo in ["catboost","lgbm"]:
            v1,t1,mae1,_=fit_eval(F1,du,tr,va,te,tgt,algo)
            v2,t2,mae2,_=fit_eval(F2,du,tr,va,te,tgt,algo)
            rows.append({"target":tname,"algo":algo,"F1_val":round(v1,4),"F1_test":round(t1,4),
                         "F2_val":round(v2,4),"F2_test":round(t2,4),"dR2_val":round(v2-v1,4),"dR2_test":round(t2-t1,4),
                         "dMAE_test":round(mae2-mae1,1)})
            print(f"  {algo:<9} F1: val={v1:.4f} test={t1:.4f} | F2: val={v2:.4f} test={t2:.4f} | dR2 val={v2-v1:+.4f} test={t2-t1:+.4f}")
    pd.DataFrame(rows).to_csv(OUT/"meteo_isolation_F1vsF2.csv",index=False)

    # ablation per-fitur meteo (target titik, CatBoost) — pendorong utama
    print(f"\n{'='*64}\nABLATION PER-FITUR METEO (lean-50 + 1 fitur, point, CatBoost)\n{'='*64}")
    du=df[df[TP].between(0,1400)&df["sun_gt5_t60"]].copy(); tr,va,te=split_masks(du)
    _,t_base,_,_=fit_eval(F1,du,tr,va,te,TP,"catboost")
    prows=[{"feature":"(lean-50 base)","test_r2":round(t_base,4),"dR2":0.0}]
    print(f"  base lean-50: test={t_base:.4f}")
    for f in METEO:
        _,t,_,_=fit_eval(F1+[f],du,tr,va,te,TP,"catboost")
        prows.append({"feature":f,"test_r2":round(t,4),"dR2":round(t-t_base,4)})
        print(f"  +{f:<16} test={t:.4f}  dR2={t-t_base:+.4f}")
    pd.DataFrame(prows).to_csv(OUT/"meteo_per_feature.csv",index=False)
    print(f"\nOutputs -> {OUT}/")

if __name__=="__main__": main()
