#!/usr/bin/env python3
"""
Konfirmasi HOMOGENITAS eksak dgn Bengkulu: filter anchor pakai elevasi ASTRONOMIS
(persis metode Bengkulu `solar_elev_deg`), bukan `elevation_deg` legacy Banten.
Uji: apakah anchor count & R² point t+60 berubah?  (harusnya ~sama).
"""
import warnings, duckdb, numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
LAT,LON,MER=-6.26147,106.7509,105.0; TIME="ts_wib"; RS=42; TP="ghi_point_t60"
F1=["ghi_now","ghi_lag_10m","ghi_lag_20m","ghi_lag_30m","ghi_lag_60m","ghi_lag_120m","ghi_lag_180m",
    "ghi_roll_30m_mean","ghi_roll_30m_std","ghi_roll_60m_mean","ghi_roll_60m_std","ghi_roll_180m_mean","ghi_roll_180m_std",
    "ghi_delta_10m","ghi_delta_60m","accel_ghi_20m","kt_now","kt_lag_10m","kt_lag_20m","kt_lag_30m","kt_lag_60m",
    "kt_roll30m_mean","kt_roll30m_std","kt_roll60m_mean","accel_kt_20m","clp_cot","clp_cot_lag_10m","clp_cot_lag_20m",
    "clp_cot_lag_30m","clp_cot_lag_60m","clp_cot_delta_10m","clp_cot_delta_30m","clp_cot_delta_60m","clp_cot_delta_180m",
    "clp_cot_roll_180m_mean","accel_clp_cot_20m","clp_cth_m","clp_ctt_k","clp_cer","clp_cloud_present",
    "hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","ghi_cs_t60","elev_sin_t60","smart_persist","smart_persist_avg"]
SQL="""WITH base AS (SELECT timestamp_wib AS ts_wib, ghi AS ghi_now, elevation_deg AS solar_elev_deg,
 cloud_optical_thickness AS clp_cot, cloud_top_height AS clp_cth_m, cloud_top_temp AS clp_ctt_k,
 cloud_eff_radius AS clp_cer, CASE WHEN cloud_present THEN 1 ELSE 0 END AS clp_cloud_present FROM solar_features_base),
wk AS (SELECT *, ghi_now/GREATEST(1100.0*GREATEST(SIN(RADIANS(solar_elev_deg)),0.02),20.0) AS kt_point FROM base),
w AS (SELECT *, LAG(ghi_now,1) OVER o AS ghi_lag_10m, LAG(ghi_now,2) OVER o AS ghi_lag_20m, LAG(ghi_now,3) OVER o AS ghi_lag_30m,
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
 LEAD(ghi_now,6) OVER o AS ghi_lead_60m FROM wk WINDOW o AS (ORDER BY ts_wib))
SELECT * FROM w WHERE ghi_now BETWEEN 0 AND 1400 AND ghi_lag_180m IS NOT NULL ORDER BY ts_wib"""
def astro_elev(ts):
    idx=pd.DatetimeIndex(ts); doy=idx.dayofyear.values.astype(float)
    h=idx.hour.values.astype(float)+idx.minute.values.astype(float)/60.0
    d=23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0)); ha=(h+4.0*(LON-MER)/60.0-12.0)*15.0
    se=np.sin(np.deg2rad(LAT))*np.sin(np.deg2rad(d))+np.cos(np.deg2rad(LAT))*np.cos(np.deg2rad(d))*np.cos(np.deg2rad(ha))
    return np.degrees(np.arcsin(np.clip(se,-1,1)))
def cs(e): return 1100.0*np.maximum(np.sin(np.deg2rad(e)),0.0)
def add_feat(df):
    o=df.copy()
    o["kt_now"]=o["ghi_now"].values/np.maximum(cs(o["solar_elev_deg"].values.astype(float)),20.0)
    o["clp_cot_delta_10m"]=o["clp_cot"]-o["clp_cot_lag_10m"]; o["clp_cot_delta_30m"]=o["clp_cot"]-o["clp_cot_lag_30m"]
    o["clp_cot_delta_60m"]=o["clp_cot"]-o["clp_cot_lag_60m"]; o["clp_cot_delta_180m"]=o["clp_cot"]-o["clp_cot_roll_180m_mean"]
    o["accel_ghi_20m"]=o["ghi_now"]-2*o["ghi_lag_10m"]+o["ghi_lag_20m"]; o["accel_kt_20m"]=o["kt_now"]-2*o["kt_lag_10m"]+o["kt_lag_20m"]
    o["accel_clp_cot_20m"]=o["clp_cot"]-2*o["clp_cot_lag_10m"]+o["clp_cot_lag_20m"]
    o["ghi_delta_10m"]=o["ghi_now"]-o["ghi_lag_10m"]; o["ghi_delta_60m"]=o["ghi_now"]-o["ghi_lag_60m"]
    ts=pd.DatetimeIndex(o[TIME]); hh=ts.hour.values.astype(float)+ts.minute.values.astype(float)/60.0
    doy=ts.dayofyear.values.astype(float); mo=ts.month.values.astype(float)
    o["hour_sin"]=np.sin(2*np.pi*hh/24); o["hour_cos"]=np.cos(2*np.pi*hh/24)
    o["doy_sin"]=np.sin(2*np.pi*doy/365.25); o["doy_cos"]=np.cos(2*np.pi*doy/365.25)
    o["month_sin"]=np.sin(2*np.pi*mo/12); o["month_cos"]=np.cos(2*np.pi*mo/12)
    et=astro_elev(o[TIME]+pd.Timedelta(minutes=60)); o["elev_sin_t60"]=np.maximum(np.sin(np.deg2rad(et)),0.0); o["ghi_cs_t60"]=cs(et)
    csa=[cs(astro_elev(o[TIME]+pd.Timedelta(minutes=s*10))) for s in range(1,7)]
    o["smart_persist"]=o["kt_now"]*o["ghi_cs_t60"]; o["smart_persist_avg"]=o["kt_now"]*np.column_stack(csa).mean(axis=1)
    o[TP]=o["ghi_lead_60m"].copy(); o["sun_gt5_t60"]=o["elev_sin_t60"]>np.sin(np.deg2rad(5.0))
    o["elev_astro_anchor"]=astro_elev(o[TIME])   # <- elevasi anchor ASTRONOMIS (metode Bengkulu)
    return o
def run(du,label):
    tr=du[TIME]<pd.Timestamp("2024-01-01"); va=(du[TIME]>=pd.Timestamp("2024-01-01"))&(du[TIME]<pd.Timestamp("2025-01-01")); te=du[TIME]>=pd.Timestamp("2025-01-01")
    xt=du.loc[tr,F1].fillna(du.loc[tr,F1].median()); xv=du.loc[va,F1].fillna(du.loc[tr,F1].median()); xe=du.loc[te,F1].fillna(du.loc[tr,F1].median())
    m=CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",random_seed=RS,verbose=False,thread_count=-1,allow_writing_files=False)
    m.fit(xt,du.loc[tr,TP],eval_set=(xv,du.loc[va,TP]),early_stopping_rounds=150,verbose=False)
    pe=np.clip(m.predict(xe),0,1400)
    print(f"  {label:<38} n={len(du):>6,} (tr {tr.sum():,}/va {va.sum():,}/te {te.sum():,})  R2={r2_score(du.loc[te,TP],pe):.4f}  MAE={mean_absolute_error(du.loc[te,TP],pe):.1f}")

def main():
    con=duckdb.connect("banten.duckdb",read_only=True); df=con.execute(SQL).fetchdf(); con.close()
    df[TIME]=pd.to_datetime(df[TIME]); df=add_feat(df)
    print("=== KONFIRMASI HOMOGENITAS ANCHOR (Banten point t+60) ===")
    # A) filter LAMA: elevasi tersimpan (elevation_deg / solar_elev_deg) > 5
    a=df[(df["solar_elev_deg"]>5)&df[TP].between(0,1400)&df["sun_gt5_t60"]].copy()
    run(a,"A) anchor=elevasi TERSIMPAN (R1/R8 lama)")
    # B) filter HOMOGEN: elevasi ASTRONOMIS anchor > 5 (metode Bengkulu)
    b=df[(df["elev_astro_anchor"]>5)&df[TP].between(0,1400)&df["sun_gt5_t60"]].copy()
    run(b,"B) anchor=elevasi ASTRONOMIS (homogen Bengkulu)")

if __name__=="__main__": main()
