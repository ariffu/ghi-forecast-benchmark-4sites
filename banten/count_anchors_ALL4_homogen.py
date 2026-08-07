#!/usr/bin/env python3
"""
PENGHITUNG ANCHOR §2.3 HOMOGEN — 4 LOKASI (Tabel 1 paper).
SATU filter identik di semua lokasi (schema-agnostic, elevasi ASTRONOMIS Cooper 1969):
  - grid 10-menit, riwayat 3-jam kontinu (18 langkah gap-free)
  - matahari > 5 deg di ANCHOR *dan* di t+60 (astronomis)
  - GHI anchor & GHI(t+60) dalam [0,1400]
Split: train<2024 / val 2024 / test 2025.
Jalankan: & "C:\\Program Files\\Python39\\python.exe" count_anchors_ALL4_homogen.py
"""
import duckdb, numpy as np, pandas as pd

SITES = [
 dict(loc="Banten",   db="C:/Users/ariff/Duckdb_Banten/banten.duckdb",
      sql="SELECT timestamp_wib AS ts, ghi AS ghi_now FROM solar_features_base ORDER BY 1",
      lat=-6.26147, lon=106.7509, period=(2022,2025)),
 dict(loc="Bengkulu", db="C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb",
      sql="SELECT ts_wib AS ts, asrs_ghi_w_m2 AS ghi_now FROM bengkulu_sch.bengkulu_master_10min_quality_final WHERE year BETWEEN 2021 AND 2025 ORDER BY 1",
      lat=-3.8607, lon=102.3381, period=(2021,2025)),
 dict(loc="Kalbar",   db="C:/Users/ariff/DuckDB_kalbar/kalbar_local.db",
      sql="SELECT timestamp_wib AS ts, ghi_final AS ghi_now FROM main.solar_kalbar_10m ORDER BY 1",
      lat=-0.0356, lon=109.3384, period=(2022,2025)),
 dict(loc="Jambi",    db="C:/Users/ariff/DuckDB_jambi/jambi_ghi_forecast_1h_train_3h_rollback_2021_2025.duckdb",
      sql="SELECT ts_wib AS ts, ghi_now FROM jambi_sch.ghi_forecast_1h_train_3h_rollback_2021_2025 ORDER BY 1",
      lat=-1.5833, lon=103.6667, period=(2021,2025)),
]
MER=105.0; GRID=10; HIST=18; LEAD=6; SUN=5.0; LO,HI=0.0,1400.0

def elev(ts,lat,lon):
    idx=pd.DatetimeIndex(ts); doy=idx.dayofyear.values.astype(float)
    h=idx.hour.values.astype(float)+idx.minute.values.astype(float)/60.0
    d=23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0)); ha=(h+4.0*(lon-MER)/60.0-12.0)*15.0
    se=np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(d))+np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(d))*np.cos(np.deg2rad(ha))
    return np.degrees(np.arcsin(np.clip(se,-1,1)))

def count(s):
    con=duckdb.connect(s["db"],read_only=True); df=con.execute(s["sql"]).fetchdf(); con.close()
    df["ts"]=pd.to_datetime(df["ts"]); df=df.sort_values("ts").reset_index(drop=True)
    y0,y1=s["period"]; df=df[(df["ts"].dt.year>=y0)&(df["ts"].dt.year<=y1)].reset_index(drop=True)
    dt=df["ts"].diff().dt.total_seconds().div(60.0)
    cont=pd.Series(True,index=df.index)
    for k in range(1,HIST+1): cont &= (dt.shift(k-1)==GRID) if k>1 else (dt==GRID)
    lead_ok=pd.Series(True,index=df.index)
    for k in range(1,LEAD+1): lead_ok &= (dt.shift(-k)==GRID)
    g60=df["ghi_now"].shift(-LEAD)
    ea=elev(df["ts"],s["lat"],s["lon"]); e60=elev(df["ts"]+pd.Timedelta(minutes=GRID*LEAD),s["lat"],s["lon"])
    ok=(cont&lead_ok&(ea>SUN)&(e60>SUN)&df["ghi_now"].between(LO,HI)&g60.between(LO,HI))
    a=df[ok]; t=a["ts"]
    tr=(t<pd.Timestamp("2024-01-01")).sum(); va=((t>=pd.Timestamp("2024-01-01"))&(t<pd.Timestamp("2025-01-01"))).sum(); te=(t>=pd.Timestamp("2025-01-01")).sum()
    return dict(loc=s["loc"],raw=len(df),anchor=len(a),tr=tr,va=va,te=te,period=f"{y0}-{y1}",
                yrs=dict(t.dt.year.value_counts().sort_index()))

if __name__=="__main__":
    print("=== ANCHOR VALID §2.3 HOMOGEN (metode identik, elevasi astronomis) ===\n")
    print(f"{'Lokasi':<9}{'Periode':<11}{'Raw10min':>10}{'Anchor§2.3':>12}{'train':>9}{'val':>8}{'test':>8}")
    rows=[]
    for s in SITES:
        r=count(s); rows.append(r)
        print(f"{r['loc']:<9}{r['period']:<11}{r['raw']:>10,}{r['anchor']:>12,}{r['tr']:>9,}{r['va']:>8,}{r['te']:>8,}")
    print("\nper-tahun:")
    for r in rows: print(f"  {r['loc']:<9}{ {int(k):int(v) for k,v in r['yrs'].items()} }")
