import duckdb
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

con = duckdb.connect()
con.execute("CREATE OR REPLACE TABLE full_dataset AS SELECT * FROM read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/synop_hourly_full.parquet')")

# lag features computed over the SYNOP-anchored sequence (note: gaps exist since SYNOP only 07-19 WIB,
# so "lag1" = previous SYNOP reading, which for hour=07 is previous day's 19:00 -- mark with hour-gap flag
base_cols = ['kt_mean','kt_std','kt_min','kt_max','ghi_clearsky_mean','ghi_mean','ghi_std',
             'dni_mean','dni_std','dhi_mean','dhi_std','reflected_mean','sun_altitude_at_h',
             'cot_mean','cot_std','cth_mean','ctt_mean','cer_mean','cloud_frac',
             'cloud_cover_oktas_m','temp_drybulb_c_corrected','temp_dewpoint_c_corrected',
             'relative_humidity_pc_corrected','wind_speed_ff_corrected','wind_dir_deg_corrected','visibility_corrected']

lag_exprs = []
for c in base_cols:
    for lag in range(1,7):
        lag_exprs.append(f"lag({c},{lag}) OVER w AS {c}_lag{lag}")

con.execute(f"""
CREATE OR REPLACE TABLE with_lags AS
SELECT *, {', '.join(lag_exprs)},
  date_diff('minute', lag(ts_hour) OVER w, ts_hour) AS gap_min_prev
FROM full_dataset
WINDOW w AS (ORDER BY ts_hour)
""")

con.execute("""
COPY (
  SELECT * FROM with_lags
  WHERE ghi_clearsky_mean > 5 AND n_obs >= 5 AND y_clearsky_next > 5 AND y_n_obs_next >= 5 AND y_kt_next IS NOT NULL
) TO 'C:/Users/ariff/bengkulu_ghi_forecast/synop_hourly_dataset.parquet' (FORMAT PARQUET)
""")

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/synop_hourly_dataset.parquet').sort_values('ts_hour').reset_index(drop=True)
print("final dataset:", df.shape)
print("gap_min_prev distribution (should mostly be 60, with ~720 for overnight jumps):")
print(df['gap_min_prev'].value_counts().head(5))

# walk-forward 5-fold, compare: radiation+CLP only (lag1-6) vs +SYNOP-native features
rad_clp_base = [c for c in base_cols if c not in ['cloud_cover_oktas_m','temp_drybulb_c_corrected','temp_dewpoint_c_corrected',
             'relative_humidity_pc_corrected','wind_speed_ff_corrected','wind_dir_deg_corrected','visibility_corrected']]
synop_base = ['cloud_cover_oktas_m','temp_drybulb_c_corrected','temp_dewpoint_c_corrected',
             'relative_humidity_pc_corrected','wind_speed_ff_corrected','wind_dir_deg_corrected','visibility_corrected']

def feats_for(bases):
    out = list(bases)
    for c in bases:
        for lag in range(1,7):
            out.append(f"{c}_lag{lag}")
    return [f for f in out if f in df.columns]

n = len(df); n_folds = 5; fold_size = n//(n_folds+1)
y = df['y_kt_next']

def wf(feats, label):
    rows = []
    for k in range(1,n_folds+1):
        tr_end = fold_size*k; te_end = min(fold_size*(k+1), n)
        if te_end <= tr_end: break
        Xtr, ytr = df[feats].fillna(-999).iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = df[feats].fillna(-999).iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        m = LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
        m.fit(Xtr, ytr)
        rows.append(r2_score(yte, m.predict(Xte)))
    print(f"{label}: mean R2={np.mean(rows):.4f}  folds={np.round(rows,4)}")
    return rows, m

r1,_ = wf(feats_for(rad_clp_base), "Radiasi+CLP only (anchor=jam SYNOP)      ")
r2,m2 = wf(feats_for(rad_clp_base+synop_base), "Radiasi+CLP+SYNOP native (cloud/temp/RH/dst)")
print(f"\nDelta R2 (+SYNOP): {np.mean(r2)-np.mean(r1):+.4f}")

imp = pd.Series(m2.feature_importances_, index=feats_for(rad_clp_base+synop_base)).sort_values(ascending=False)
print("\nTop 15 importance:")
print(imp.head(15).to_string())
