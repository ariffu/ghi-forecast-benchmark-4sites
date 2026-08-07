#!/usr/bin/env python3
"""
LEARNING CURVE Banten — apakah VOLUME DATA membatasi R²?
Uji: latih pd fraksi training membesar (10..100%), eval test 2025 tetap.
Konfig harmonis identik R1/R8 (lean-50, CatBoost, point t+60).
  - Subsample ACAK (seed tetap) -> isolasi murni efek volume (periode konstan)
  - Subsample RECENT (N terbaru) -> realistis temporal
Jika R2 plateau sebelum 100% -> volume BUKAN penghambat (gap ke Bengkulu = beda situs).
Run: & "C:\\Program Files\\Python39\\python.exe" train_ghi_1h_banten_learning_curve.py
"""
import warnings
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")

DB="banten.duckdb"; OUT=Path("outputs_learning_curve_banten"); OUT.mkdir(exist_ok=True)
LAT,LON,MER=-6.26147,106.7509,105.0; TIME="ts_wib"; RS=42; TP="ghi_point_t60"
F1=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m","kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
    "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m","clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
    "clp_cot_lag_30m","clp_cot_lag_60m","clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m",
    "clp_cot_roll_180m_mean","accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present",
    "hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]

def build_sql():
    return """
    WITH base AS (SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
               cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m, cloud_top_temp AS clp_ctt_k,
               cloud_eff_radius AS clp_cer, CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present FROM solar_features_base),
    wk AS (SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base),
    w AS (SELECT *,
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
    o[TP]=o["ghi_lead_60m"].copy(); o["sun_gt5_t60"]=o["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    return o
def cbm():
    return CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",random_seed=RS,verbose=False,thread_count=-1,allow_writing_files=False)

def main():
    con=duckdb.connect(DB,read_only=True); print("Loading..."); df=con.execute(build_sql()).fetchdf(); con.close()
    df[TIME]=pd.to_datetime(df[TIME]); df=add_feat(df)
    du=df[df[TP].between(0,1400)&df["sun_gt5_t60"]].copy()
    tr_m=du[TIME]<pd.Timestamp("2024-01-01"); va_m=(du[TIME]>=pd.Timestamp("2024-01-01"))&(du[TIME]<pd.Timestamp("2025-01-01")); te_m=du[TIME]>=pd.Timestamp("2025-01-01")
    Xtr,ytr=du.loc[tr_m,F1].fillna(du.loc[tr_m,F1].median()),du.loc[tr_m,TP]
    Xva,yva=du.loc[va_m,F1].fillna(du.loc[tr_m,F1].median()),du.loc[va_m,TP]
    Xte,yte=du.loc[te_m,F1].fillna(du.loc[tr_m,F1].median()),du.loc[te_m,TP].values
    sp=np.clip(du.loc[te_m,"smart_persist"].values,0,1400)
    n_full=len(Xtr); print(f"train penuh={n_full:,}  val={va_m.sum():,}  test={te_m.sum():,}")

    fracs=[0.10,0.25,0.50,0.75,1.00]
    for mode in ["acak","recent"]:
        print(f"\n{'='*60}\nLEARNING CURVE — subsample {mode.upper()}\n{'='*60}")
        rows=[]
        for f in fracs:
            n=int(round(n_full*f))
            if mode=="acak":
                idx=np.random.RandomState(RS).choice(n_full,n,replace=False); idx.sort()
                xt=Xtr.iloc[idx]; yt=ytr.iloc[idx]
            else:  # recent N (paling akhir sebelum val) — realistis temporal
                xt=Xtr.iloc[-n:]; yt=ytr.iloc[-n:]
            m=cbm(); m.fit(xt,yt,eval_set=(Xva,yva),early_stopping_rounds=150,verbose=False)
            pe=np.clip(m.predict(Xte),0,1400)
            r2=r2_score(yte,pe); mae=mean_absolute_error(yte,pe); rmse=np.sqrt(mean_squared_error(yte,pe))
            rows.append({"mode":mode,"frac":f,"n_train":n,"test_r2":round(r2,4),"mae":round(mae,1),"rmse":round(rmse,1)})
            print(f"  {int(f*100):>3}% (n={n:>6,}): test R2={r2:.4f}  MAE={mae:.1f}")
        pd.DataFrame(rows).to_csv(OUT/f"learning_curve_{mode}.csv",index=False)
        d=rows[-1]["test_r2"]-rows[-2]["test_r2"]
        print(f"  -> delta R2 dari 75%->100%: {d:+.4f}  ({'MASIH NAIK (volume membantu)' if d>0.002 else 'PLATEAU (volume bukan penghambat)'})")
    print(f"\nOutputs -> {OUT}/")

if __name__=="__main__": main()
