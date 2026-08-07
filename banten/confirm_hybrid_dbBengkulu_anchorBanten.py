#!/usr/bin/env python3
"""
HIBRIDA yang diminta user:
  - BASIS DATA (pool kandidat) = homogen Bengkulu:
        kontinuitas 3-jam KETAT (18 langkah 10-mnt gap-free)
        + siang ASTRONOMIS (anchor>5 DAN t+60>5) + GHI anchor & target in [0,1400]
  - ANCHOR TRAINING = versi Banten:
        filter final baris latih pakai elevasi TERSIMPAN solar_elev_deg>5
Bandingkan 3 kasus (point t+60):
  A  = Banten murni (R1/R8 kini)         : anchor tersimpan, kontinuitas longgar
  B  = Bengkulu murni (astronomis+ketat) : homogen penuh
  C  = HIBRIDA (pool Bengkulu + anchor Banten)  <- yang ditanyakan
"""
import warnings, duckdb, numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
LAT,LON,MER=-6.26147,106.7509,105.0; TIME="ts_wib"; RS=42; TP="ghi_point_t60"
from confirm_homogeneous_anchor import F1, SQL, astro_elev, cs, add_feat

def run(du,label):
    tr=du[TIME]<pd.Timestamp("2024-01-01"); va=(du[TIME]>=pd.Timestamp("2024-01-01"))&(du[TIME]<pd.Timestamp("2025-01-01")); te=du[TIME]>=pd.Timestamp("2025-01-01")
    med=du.loc[tr,F1].median()
    xt=du.loc[tr,F1].fillna(med); xv=du.loc[va,F1].fillna(med); xe=du.loc[te,F1].fillna(med)
    m=CatBoostRegressor(iterations=4000,learning_rate=0.02,depth=8,l2_leaf_reg=3.0,loss_function="RMSE",random_seed=RS,verbose=False,thread_count=-1,allow_writing_files=False)
    m.fit(xt,du.loc[tr,TP],eval_set=(xv,du.loc[va,TP]),early_stopping_rounds=150,verbose=False)
    pe=np.clip(m.predict(xe),0,1400)
    print(f"  {label:<44} n={len(du):>6,} (te {te.sum():,})  R2={r2_score(du.loc[te,TP],pe):.4f}  MAE={mean_absolute_error(du.loc[te,TP],pe):.1f}")

def main():
    con=duckdb.connect("banten.duckdb",read_only=True); df=con.execute(SQL).fetchdf(); con.close()
    df[TIME]=pd.to_datetime(df[TIME]); df=df.sort_values(TIME).reset_index(drop=True); df=add_feat(df)
    # kontinuitas 3-jam KETAT (semua 18 langkah = 10 mnt persis)
    dt=df[TIME].diff().dt.total_seconds().div(60.0); strict=pd.Series(True,index=df.index)
    for k in range(1,19): strict &= (dt.shift(k-1)==10) if k>1 else (dt==10)
    astA=df["elev_astro_anchor"]; base_valid=df[TP].between(0,1400)&df["sun_gt5_t60"]&df["ghi_now"].between(0,1400)
    pool_bengkulu = strict & (astA>5) & base_valid          # basis data homogen Bengkulu
    anchor_banten = (df["solar_elev_deg"]>5)                 # anchor training versi Banten (elev tersimpan)

    print("=== HIBRIDA: pool DB homogen-Bengkulu + anchor training versi-Banten (point t+60) ===")
    run(df[(df["solar_elev_deg"]>5)&base_valid].copy(),                 "A) Banten murni (R1/R8 kini)")
    run(df[pool_bengkulu].copy(),                                       "B) Bengkulu murni (astronomis+ketat)")
    run(df[pool_bengkulu & anchor_banten].copy(),                      "C) HIBRIDA pool-Bengkulu x anchor-Banten")

if __name__=="__main__": main()
