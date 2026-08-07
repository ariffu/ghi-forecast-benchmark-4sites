#!/usr/bin/env python3
"""
R8 GAP-CLOSING — BANTEN (tujuan inti R8).
Menutup gap lean-50 (R1 benchmark) -> potensi penuh, lewat penambahan bertahap
grup fitur (akselerasi[ada di lean] + VARIABILITAS + CLOUD-TREND + FULL-LAGS + METEO),
lalu PRUNING val-guided ke set lean-optimal. Target: TITIK t+60 (gap = -0.064).

Basis harmonis IDENTIK R1 (clearsky sederhana 1100*sin, split train<2024/val2024/test2025,
filter sun>5 anchor+t60) — agar penutupan gap ter-atribusi murni ke FITUR, bukan clearsky.

Referensi: lean-50 point R2=0.676 (R1) ; produksi Banten point R2=0.740 -> gap -0.064.
Run: & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_banten_R8_gapclose.py
"""
import warnings
from pathlib import Path
import duckdb, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")

DB_PATH="banten.duckdb"; OUT=Path("outputs_R8_gapclose_banten"); OUT.mkdir(exist_ok=True)
LAT,LON,MER=-6.26147,106.7509,105.0; TIME="ts_wib"; RS=42
TP="ghi_point_t60"

# ---- lean-50 (R1) ----
LEAN=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m","kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
    "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m","clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
    "clp_cot_lag_30m","clp_cot_lag_60m","clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m",
    "clp_cot_roll_180m_mean","accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present",
    "hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]
# ---- grup tambahan (superset) ----
G_VAR=["ghi_std_1h","ghi_std_2h","ghi_std_3h","ghi_range_1h","ghi_range_2h","ghi_range_3h","kt_std_1h","kt_std_2h","kt_std_3h"]
G_CLOUD=["clp_cot_lag_90m","clp_cot_lag_120m","clp_cot_lag_150m","clp_cot_roll_60m_mean","clp_cot_trend_1h","clp_cot_std_1h","clp_cot_range_2h"]
G_LAGS=[f"ghi_lag_{m}m" for m in (40,50,70,80,90,100,110,130,140,150,160,170)]+[f"kt_lag_{m}m" for m in (40,50,90,120)]
G_METEO=["temp_air_c","humidity_pct","wind_speed_ms","rainfall_mm","pressure_hpa"]

def build_sql():
    lag=lambda c,n: f"LAG({c},{n}) OVER o AS {c}_L{n}"
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
          -- GHI lags full 1..18
          LAG(ghi_now,1) OVER o AS ghi_lag_10m, LAG(ghi_now,2) OVER o AS ghi_lag_20m, LAG(ghi_now,3) OVER o AS ghi_lag_30m,
          LAG(ghi_now,4) OVER o AS ghi_lag_40m, LAG(ghi_now,5) OVER o AS ghi_lag_50m, LAG(ghi_now,6) OVER o AS ghi_lag_60m,
          LAG(ghi_now,7) OVER o AS ghi_lag_70m, LAG(ghi_now,8) OVER o AS ghi_lag_80m, LAG(ghi_now,9) OVER o AS ghi_lag_90m,
          LAG(ghi_now,10) OVER o AS ghi_lag_100m, LAG(ghi_now,11) OVER o AS ghi_lag_110m, LAG(ghi_now,12) OVER o AS ghi_lag_120m,
          LAG(ghi_now,13) OVER o AS ghi_lag_130m, LAG(ghi_now,14) OVER o AS ghi_lag_140m, LAG(ghi_now,15) OVER o AS ghi_lag_150m,
          LAG(ghi_now,16) OVER o AS ghi_lag_160m, LAG(ghi_now,17) OVER o AS ghi_lag_170m, LAG(ghi_now,18) OVER o AS ghi_lag_180m,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_std,
          AVG(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
          -- variabilitas (std & range) 1h/2h/3h
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5  PRECEDING AND CURRENT ROW) AS ghi_std_1h,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS ghi_std_2h,
          STDDEV_SAMP(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_std_3h,
          (MAX(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5  PRECEDING AND CURRENT ROW)-MIN(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 5  PRECEDING AND CURRENT ROW)) AS ghi_range_1h,
          (MAX(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)-MIN(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)) AS ghi_range_2h,
          (MAX(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW)-MIN(ghi_now) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW)) AS ghi_range_3h,
          -- kt lags & rolls & variability
          LAG(kt_point,1) OVER o AS kt_lag_10m, LAG(kt_point,2) OVER o AS kt_lag_20m, LAG(kt_point,3) OVER o AS kt_lag_30m,
          LAG(kt_point,4) OVER o AS kt_lag_40m, LAG(kt_point,5) OVER o AS kt_lag_50m, LAG(kt_point,6) OVER o AS kt_lag_60m,
          LAG(kt_point,9) OVER o AS kt_lag_90m, LAG(kt_point,12) OVER o AS kt_lag_120m,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_mean,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS kt_roll30m_std,
          AVG(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS kt_roll60m_mean,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 5  PRECEDING AND CURRENT ROW) AS kt_std_1h,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS kt_std_2h,
          STDDEV_SAMP(kt_point) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS kt_std_3h,
          -- CLP lags/rolls/trend/var
          LAG(clp_cot,1) OVER o AS clp_cot_lag_10m, LAG(clp_cot,2) OVER o AS clp_cot_lag_20m, LAG(clp_cot,3) OVER o AS clp_cot_lag_30m,
          LAG(clp_cot,6) OVER o AS clp_cot_lag_60m, LAG(clp_cot,9) OVER o AS clp_cot_lag_90m, LAG(clp_cot,12) OVER o AS clp_cot_lag_120m,
          LAG(clp_cot,15) OVER o AS clp_cot_lag_150m,
          AVG(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 5  PRECEDING AND CURRENT ROW) AS clp_cot_roll_60m_mean,
          AVG(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
          STDDEV_SAMP(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS clp_cot_std_1h,
          (MAX(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)-MIN(clp_cot) OVER (ORDER BY ts_wib ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)) AS clp_cot_range_2h,
          LEAD(ghi_now,6) OVER o AS ghi_lead_60m
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
    o=df.copy(); ts=pd.DatetimeIndex(o[TIME])
    o["kt_now"]=o["ghi_now"].values/np.maximum(cs(o["solar_elev_deg"].values.astype(float)),20.0)
    o["clp_cot_delta_10m"]=o["clp_cot"]-o["clp_cot_lag_10m"]; o["clp_cot_delta_30m"]=o["clp_cot"]-o["clp_cot_lag_30m"]
    o["clp_cot_delta_60m"]=o["clp_cot"]-o["clp_cot_lag_60m"]; o["clp_cot_delta_180m"]=o["clp_cot"]-o["clp_cot_roll_180m_mean"]
    o["clp_cot_trend_1h"]=o["clp_cot"]-o["clp_cot_lag_60m"]
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
    o["sun_gt5_t60"]=o["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    # rename LAG cols already aliased in SQL to feature names used in groups (ghi_lag_40m etc already correct)
    return o

def split_masks(df):
    ts=df[TIME]
    return (ts<pd.Timestamp("2024-01-01"),(ts>=pd.Timestamp("2024-01-01"))&(ts<pd.Timestamp("2025-01-01")),ts>=pd.Timestamp("2025-01-01"))
def cbm():
    return CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",random_seed=RS,verbose=False,thread_count=-1,allow_writing_files=False)
def fit_eval(feats,du,tr,va,te,tgt):
    xt=du.loc[tr,feats].fillna(du.loc[tr,feats].median()); xv=du.loc[va,feats].fillna(du.loc[tr,feats].median()); xe=du.loc[te,feats].fillna(du.loc[tr,feats].median())
    m=cbm(); m.fit(xt,du.loc[tr,tgt],eval_set=(xv,du.loc[va,tgt]),early_stopping_rounds=150,verbose=False)
    pe=np.clip(m.predict(xe),0,1400); pv=np.clip(m.predict(xv),0,1400)
    return (r2_score(du.loc[te,tgt],pe),mean_absolute_error(du.loc[te,tgt],pe),np.sqrt(mean_squared_error(du.loc[te,tgt],pe)),
            r2_score(du.loc[va,tgt],pv),m)

def main():
    con=duckdb.connect(DB_PATH,read_only=True); print("Loading..."); df=con.execute(build_sql()).fetchdf(); con.close()
    df[TIME]=pd.to_datetime(df[TIME]); df=add_feat(df)
    du=df[df[TP].between(0,1400)&df["sun_gt5_t60"]].copy()
    tr,va,te=split_masks(du); print(f"rows={len(du):,} train={tr.sum():,} val={va.sum():,} test={te.sum():,}")

    print("\n"+"="*64+"\nGAP-CLOSING INKREMENTAL (target titik, CatBoost)\n"+"="*64)
    steps=[("lean-50",LEAN),("+variabilitas",LEAN+G_VAR),("+cloud-trend",LEAN+G_VAR+G_CLOUD),
           ("+full-lags",LEAN+G_VAR+G_CLOUD+G_LAGS),("+meteo (SUPERSET)",LEAN+G_VAR+G_CLOUD+G_LAGS+G_METEO)]
    rows=[]; r2_lean=None
    for name,feats in steps:
        feats=[f for f in feats if f in du.columns]
        r2t,mae,rmse,r2v,m=fit_eval(feats,du,tr,va,te,TP)
        if r2_lean is None: r2_lean=r2t
        rows.append({"step":name,"n_feat":len(feats),"val_r2":round(r2v,4),"test_r2":round(r2t,4),
                     "mae":round(mae,1),"rmse":round(rmse,1),"gap_closed_vs_lean":round(r2t-r2_lean,4)})
        print(f"  {name:<20} nfeat={len(feats):3d}  VAL={r2v:.4f}  TEST={r2t:.4f}  MAE={mae:.1f}  (+{r2t-r2_lean:.4f} vs lean)")
    superset=[f for f in (LEAN+G_VAR+G_CLOUD+G_LAGS+G_METEO) if f in du.columns]
    pd.DataFrame(rows).to_csv(OUT/"gapclose_incremental.csv",index=False)

    print("\n"+"="*64+"\nPRUNING SUPERSET (val-guided top-K)\n"+"="*64)
    _,_,_,_,mfull=fit_eval(superset,du,tr,va,te,TP)
    imp=pd.Series(mfull.get_feature_importance(),index=superset).sort_values(ascending=False)
    prune=[]
    for K in [10,15,20,25,30,40,len(superset)]:
        fk=list(imp.index[:K]); r2t,mae,rmse,r2v,_=fit_eval(fk,du,tr,va,te,TP)
        prune.append({"top_K":K,"val_r2":round(r2v,4),"test_r2":round(r2t,4),"mae":round(mae,1)})
        print(f"  top-{K:<3} VAL={r2v:.4f} TEST={r2t:.4f} MAE={mae:.1f}")
    pd.DataFrame(prune).to_csv(OUT/"gapclose_pruning.csv",index=False)
    print("\n15 fitur terpenting superset:"); print(imp.head(15).round(2).to_string())
    print(f"\nOutputs -> {OUT}/")

if __name__=="__main__": main()
