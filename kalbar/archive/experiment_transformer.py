"""
Uji model Transformer (multi-head self-attention) vs LightGBM/ensemble, split sama
(train 2022-23, val 2024, test 2025). Coba 2 panjang lookback: 6 langkah (60 menit,
setara LSTM sebelumnya) dan 18 langkah (3 jam, gaya arsitektur Jambi/pipeline lama)
untuk memberi Transformer kesempatan terbaik memanfaatkan konteks lebih panjang.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PARQUET_FULL = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_direct.parquet"
PARQUET_ENH = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_enhanced.parquet"
TARGET = "ghi_target_60m"
SEQ_COLS = ["kt", "CLOT_mean", "CLTT_mean", "CLTH_mean", "sun_altitude", "temp_air_c", "humidity_pct"]
STATIC_COLS = ["sun_altitude_future", "hour_sin_future", "hour_cos_future",
               "ghi_clearsky_future", "doy_sin", "doy_cos"]

df_full = pd.read_parquet(PARQUET_FULL).sort_values("timestamp_wib").reset_index(drop=True)
df_full["year"] = df_full["timestamp_wib"].dt.year

def build_sequences(frame, lookback):
    full = frame.sort_values("timestamp_wib").reset_index(drop=True)
    seq_data = full[SEQ_COLS].fillna(full[SEQ_COLS].median()).values.astype("float32")
    ts_pd = full["timestamp_wib"]
    valid_idx = full.index[full["anchor_valid"]].tolist()
    X_seq, X_static, y = [], [], []
    for i in valid_idx:
        if i < lookback - 1:
            continue
        window_ts = ts_pd.iloc[i - lookback + 1:i + 1]
        gaps = window_ts.diff().dropna()
        if len(gaps) and not all(g == np.timedelta64(10, "m") for g in gaps):
            continue
        X_seq.append(seq_data[i - lookback + 1:i + 1])
        X_static.append(full.loc[i, STATIC_COLS].values.astype("float32"))
        y.append(full.loc[i, TARGET])
    return np.array(X_seq), np.array(X_static), np.array(y)

def transformer_block(x, num_heads, key_dim, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.LayerNormalization()(x + ff)
    return x

def run_transformer(lookback, label):
    train_full = df_full[df_full["year"].isin([2022, 2023])]
    val_full = df_full[df_full["year"] == 2024]
    test_full = df_full[df_full["year"] == 2025]
    Xs_tr, Xst_tr, y_tr = build_sequences(train_full, lookback)
    Xs_va, Xst_va, y_va = build_sequences(val_full, lookback)
    Xs_te, Xst_te, y_te = build_sequences(test_full, lookback)
    print(f"  [{label}] n_train={len(y_tr)} n_val={len(y_va)} n_test={len(y_te)}")

    seq_mean = Xs_tr.reshape(-1, Xs_tr.shape[-1]).mean(0)
    seq_std = Xs_tr.reshape(-1, Xs_tr.shape[-1]).std(0) + 1e-6
    static_mean, static_std = Xst_tr.mean(0), Xst_tr.std(0) + 1e-6
    Xs_tr_n = (Xs_tr - seq_mean) / seq_std
    Xs_va_n = (Xs_va - seq_mean) / seq_std
    Xs_te_n = (Xs_te - seq_mean) / seq_std
    Xst_tr_n = (Xst_tr - static_mean) / static_std
    Xst_va_n = (Xst_va - static_mean) / static_std
    Xst_te_n = (Xst_te - static_mean) / static_std

    seq_in = layers.Input(shape=(lookback, len(SEQ_COLS)))
    x = layers.Dense(32)(seq_in)
    pos = layers.Embedding(input_dim=lookback, output_dim=32)(tf.range(lookback))
    x = x + pos
    x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=64)
    x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=64)
    x = layers.GlobalAveragePooling1D()(x)
    static_in = layers.Input(shape=(len(STATIC_COLS),))
    x = layers.Concatenate()([x, static_in])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    model = models.Model([seq_in, static_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit([Xs_tr_n, Xst_tr_n], y_tr, validation_data=([Xs_va_n, Xst_va_n], y_va),
              epochs=100, batch_size=256, callbacks=[es], verbose=0)
    pred = model.predict([Xs_te_n, Xst_te_n], verbose=0).flatten()
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    print(f"  [{label}] R2={r2:.4f}  MAE={mae:.2f}")
    return r2

print("Transformer lookback=6 (60 menit):")
r2_t6 = run_transformer(6, "Transformer-60m")
print("\nTransformer lookback=18 (3 jam):")
r2_t18 = run_transformer(18, "Transformer-3h")

print("\nLightGBM pembanding (fitur enhanced, 43 fitur):")
df_enh = pd.read_parquet(PARQUET_ENH).sort_values("timestamp_wib").reset_index(drop=True)
df_enh["year"] = df_enh["timestamp_wib"].dt.year
FEATURES = ["CLOT_mean","CLTT_mean","CLTH_mean","CLER_23_mean","clp_cloud_present_int",
    "temp_air_c","humidity_pct","wind_speed_ms","rainfall_mm","sun_altitude",
    "hour_sin","hour_cos","doy_sin","doy_cos","month",
    "AOD_500nm","angstrom_440_870","precipitable_water_cm",
    "ghi_lag10m","ghi_lag20m","ghi_lag30m","ghi_lag60m",
    "kt_lag10m","kt_lag20m","kt_lag30m","kt_lag60m",
    "kt_roll30m_mean","kt_roll60m_mean","kt_roll30m_std",
    "delta_kt_10m","delta_kt_30m","clot_lag10m","clot_lag30m","delta_clot_30m","delta_ghi_30m",
    "sun_altitude_future","ghi_clearsky_future","hour_sin_future","hour_cos_future",
    "CLOT_std","CLER_23_std","pressure_hpa","cloud_cover_oktas"]
for c in FEATURES:
    df_enh[c] = df_enh[c].astype("float32")
df_enh[FEATURES] = df_enh[FEATURES].fillna(df_enh[FEATURES].median())
train = df_enh[df_enh["year"].isin([2022, 2023])]
val = df_enh[df_enh["year"] == 2024]
test = df_enh[df_enh["year"] == 2025]
m = lgb.LGBMRegressor(n_estimators=2000, num_leaves=127, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
m.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
      callbacks=[lgb.early_stopping(50, verbose=False)])
pred_lgb = m.predict(test[FEATURES])
r2_lgb = r2_score(test[TARGET], pred_lgb)
print(f"  LightGBM R2={r2_lgb:.4f}")

print("\n=== Ringkasan ===")
for name, r2 in [("Transformer (lookback 60m)", r2_t6), ("Transformer (lookback 3h)", r2_t18),
                  ("LightGBM (43 fitur)", r2_lgb)]:
    print(f"  {name:30s} R2={r2:.4f}")
