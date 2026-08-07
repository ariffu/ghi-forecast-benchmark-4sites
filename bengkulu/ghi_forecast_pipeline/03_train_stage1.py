import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_parquet('C:/Users/ariff/bengkulu_ghi_forecast/stage1_dataset.parquet')
df = df.sort_values('ts_hour').reset_index(drop=True)

base_feats = ['ghi_clearsky_mean','ghi_clearsky_min','ghi_clearsky_max',
              'kt_mean','kt_std','kt_min','kt_max',
              'ghi_mean','ghi_std','ghi_min','ghi_max',
              'dni_mean','dni_std','dni_min','dni_max',
              'dhi_mean','dhi_std','dhi_min','dhi_max',
              'reflected_mean','sun_altitude_at_h']

lag3_feats = base_feats + [c for c in df.columns if c.endswith(('lag1','lag2','lag3'))]
lag6_feats = base_feats + [c for c in df.columns if c.endswith(('lag1','lag2','lag3','lag4','lag5','lag6'))]

y_col = 'y_kt_next'

n = len(df)
i_tr, i_va = int(n*0.70), int(n*0.85)
print(f"n={n} train={i_tr} val={i_va-i_tr} test={n-i_va}")

def run(feats, label):
    feats = [f for f in feats if f in df.columns]
    X = df[feats].fillna(-999)
    y = df[y_col]
    Xtr, ytr = X.iloc[:i_tr], y.iloc[:i_tr]
    Xva, yva = X.iloc[i_tr:i_va], y.iloc[i_tr:i_va]
    Xte, yte = X.iloc[i_va:], y.iloc[i_va:]

    m = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, num_leaves=31, verbosity=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva,yva)], callbacks=[])
    pred = m.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    mae = mean_absolute_error(yte, pred)

    # persistence baseline (this hour's kt_mean = next hour's kt)
    pers_r2 = r2_score(yte, Xte['kt_mean'])
    pers_rmse = np.sqrt(mean_squared_error(yte, Xte['kt_mean']))

    # convert to GHI W/m2 using actual clearsky of target hour
    clearsky_next = df['y_clearsky_next'].iloc[i_va:].values
    ghi_pred = pred * clearsky_next
    ghi_true = df['y_ghi_next'].iloc[i_va:].values
    ghi_r2 = r2_score(ghi_true, ghi_pred)
    ghi_rmse = np.sqrt(mean_squared_error(ghi_true, ghi_pred))

    print(f"\n=== {label} ({len(feats)} features) ===")
    print(f"  Persistence  : RMSE(kt)={pers_rmse:.4f} R2(kt)={pers_r2:.4f}")
    print(f"  LightGBM     : RMSE(kt)={rmse:.4f} R2(kt)={r2:.4f} MAE(kt)={mae:.4f}")
    print(f"  GHI (W/m2)   : RMSE={ghi_rmse:.2f} R2={ghi_r2:.4f}")
    print(f"  Skill vs persistence: {1-(rmse**2/pers_rmse**2):.4f}")

    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    print("  Top 10 importance:")
    print(imp.head(10).to_string())
    return m, r2, ghi_r2

m3, r2_3, ghi_r2_3 = run(lag3_feats, "Stage 1 - lag 1-3h")
m6, r2_6, ghi_r2_6 = run(lag6_feats, "Stage 1 - lag 1-6h")
