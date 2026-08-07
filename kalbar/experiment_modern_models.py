"""
Bandingkan LightGBM (baseline kita) vs XGBoost vs CatBoost vs LSTM pada fitur/split yang
SAMA (train 2022-23, val 2024, test 2025) -- supaya perbandingan adil, tidak seperti
klaim-klaim di literatur yang beda metodologi evaluasi.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PARQUET = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_enhanced.parquet"
df = pd.read_parquet(PARQUET).sort_values("timestamp_wib").reset_index(drop=True)
df["year"] = df["timestamp_wib"].dt.year

FEATURES = [
    "CLOT_mean", "CLTT_mean", "CLTH_mean", "CLER_23_mean", "clp_cloud_present_int",
    "temp_air_c", "humidity_pct", "wind_speed_ms", "rainfall_mm", "sun_altitude",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "AOD_500nm", "angstrom_440_870", "precipitable_water_cm",
    "ghi_lag10m", "ghi_lag20m", "ghi_lag30m", "ghi_lag60m",
    "kt_lag10m", "kt_lag20m", "kt_lag30m", "kt_lag60m",
    "kt_roll30m_mean", "kt_roll60m_mean", "kt_roll30m_std",
    "delta_kt_10m", "delta_kt_30m", "clot_lag10m", "clot_lag30m", "delta_clot_30m", "delta_ghi_30m",
    "sun_altitude_future", "ghi_clearsky_future", "hour_sin_future", "hour_cos_future",
    "CLOT_std", "CLER_23_std", "pressure_hpa", "cloud_cover_oktas",
]
TARGET = "ghi_target_60m"

for c in FEATURES:
    df[c] = df[c].astype("float32")
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

train = df[df["year"].isin([2022, 2023])].reset_index(drop=True)
val = df[df["year"] == 2024].reset_index(drop=True)
test = df[df["year"] == 2025].reset_index(drop=True)

results = {}

def report(name, pred):
    r2 = r2_score(test[TARGET], pred)
    mae = mean_absolute_error(test[TARGET], pred)
    results[name] = r2
    print(f"{name:25s} R2={r2:.4f}  MAE={mae:7.2f}")

m_lgb = lgb.LGBMRegressor(n_estimators=2000, num_leaves=127, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
m_lgb.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
report("LightGBM (direct)", m_lgb.predict(test[FEATURES]))

m_xgb = xgb.XGBRegressor(n_estimators=2000, max_depth=8, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          early_stopping_rounds=50, eval_metric="rmse")
m_xgb.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])], verbose=False)
report("XGBoost (direct)", m_xgb.predict(test[FEATURES]))

m_cb = cb.CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.03,
                             random_state=42, verbose=False, early_stopping_rounds=50)
m_cb.fit(train[FEATURES], train[TARGET], eval_set=(val[FEATURES], val[TARGET]))
report("CatBoost (direct)", m_cb.predict(test[FEATURES]))

print("\nMenyiapkan sequence untuk LSTM (lookback 6 langkah = 60 menit)...")
LOOKBACK = 6
SEQ_COLS = ["kt", "CLOT_mean", "CLTT_mean", "sun_altitude", "temp_air_c", "humidity_pct"]
STATIC_COLS = ["sun_altitude_future", "hour_sin_future", "hour_cos_future",
               "ghi_clearsky_future", "doy_sin", "doy_cos"]

def build_sequences(frame):
    full = frame.sort_values("timestamp_wib").reset_index(drop=True)
    seq_data = full[SEQ_COLS].fillna(full[SEQ_COLS].median()).values.astype("float32")
    ts = full["timestamp_wib"].values
    valid_idx = full.index[full["anchor_valid"]].tolist()
    X_seq, X_static, y, idx_keep = [], [], [], []
    ts_pd = pd.Series(ts)
    for i in valid_idx:
        if i < LOOKBACK - 1:
            continue
        window_ts = ts_pd.iloc[i - LOOKBACK + 1:i + 1]
        gaps = window_ts.diff().dropna()
        if len(gaps) and not all(g == np.timedelta64(10, "m") for g in gaps):
            continue
        X_seq.append(seq_data[i - LOOKBACK + 1:i + 1])
        X_static.append(full.loc[i, STATIC_COLS].values.astype("float32"))
        y.append(full.loc[i, TARGET])
        idx_keep.append(i)
    return np.array(X_seq), np.array(X_static), np.array(y), idx_keep

con_full = pd.read_parquet(r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_direct.parquet")
con_full["year"] = con_full["timestamp_wib"].dt.year
train_full = con_full[con_full["year"].isin([2022, 2023])]
val_full = con_full[con_full["year"] == 2024]
test_full = con_full[con_full["year"] == 2025]

Xs_tr, Xst_tr, y_tr, _ = build_sequences(train_full)
Xs_va, Xst_va, y_va, _ = build_sequences(val_full)
Xs_te, Xst_te, y_te, _ = build_sequences(test_full)
print(f"LSTM sequences: train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

seq_mean, seq_std = Xs_tr.reshape(-1, Xs_tr.shape[-1]).mean(0), Xs_tr.reshape(-1, Xs_tr.shape[-1]).std(0) + 1e-6
static_mean, static_std = Xst_tr.mean(0), Xst_tr.std(0) + 1e-6
Xs_tr_n, Xs_va_n, Xs_te_n = [(x - seq_mean) / seq_std for x in (Xs_tr, Xs_va, Xs_te)]
Xst_tr_n, Xst_va_n, Xst_te_n = [(x - static_mean) / static_std for x in (Xst_tr, Xst_va, Xst_te)]

seq_in = layers.Input(shape=(LOOKBACK, len(SEQ_COLS)))
static_in = layers.Input(shape=(len(STATIC_COLS),))
x = layers.LSTM(48, return_sequences=False)(seq_in)
x = layers.Concatenate()([x, static_in])
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1)(x)
model = models.Model([seq_in, static_in], out)
model.compile(optimizer="adam", loss="mse")
es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.fit([Xs_tr_n, Xst_tr_n], y_tr, validation_data=([Xs_va_n, Xst_va_n], y_va),
          epochs=100, batch_size=256, callbacks=[es], verbose=0)
pred_lstm = model.predict([Xs_te_n, Xst_te_n], verbose=0).flatten()
r2_lstm = r2_score(y_te, pred_lstm)
mae_lstm = mean_absolute_error(y_te, pred_lstm)
results["LSTM"] = r2_lstm
print(f"{'LSTM (lookback 6 step)':25s} R2={r2_lstm:.4f}  MAE={mae_lstm:7.2f}")

print("\n=== Ringkasan semua model ===")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {k:25s} R2={v:.4f}")
