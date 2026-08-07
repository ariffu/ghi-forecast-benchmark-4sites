"""
Varian Transformer lain: PatchTST-style (patching) dan Transformer ramping+regularized.
Plus verifikasi eksplisit bahwa target adalah T+60 menit genuine (bukan nowcast/trik).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

PARQUET_FULL = r"C:\Users\ariff\DuckDB_kalbar\training_ghi_1h_direct.parquet"
TARGET = "ghi_target_60m"
SEQ_COLS = ["kt", "CLOT_mean", "CLTT_mean", "CLTH_mean", "sun_altitude", "temp_air_c", "humidity_pct"]
STATIC_COLS = ["sun_altitude_future", "hour_sin_future", "hour_cos_future",
               "ghi_clearsky_future", "doy_sin", "doy_cos"]

df_full = pd.read_parquet(PARQUET_FULL).sort_values("timestamp_wib").reset_index(drop=True)
df_full["year"] = df_full["timestamp_wib"].dt.year

def build_sequences(frame, lookback, verify=False):
    full = frame.sort_values("timestamp_wib").reset_index(drop=True)
    seq_data = full[SEQ_COLS].fillna(full[SEQ_COLS].median()).values.astype("float32")
    ts_pd = full["timestamp_wib"]
    valid_idx = full.index[full["anchor_valid"]].tolist()
    X_seq, X_static, y, anchor_ts = [], [], [], []
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
        anchor_ts.append(full.loc[i, "timestamp_wib"])
    if verify:
        print("  Verifikasi horizon T+60 (5 sampel acak dari test set):")
        rng = np.random.default_rng(0)
        for idx in rng.choice(len(anchor_ts), 5, replace=False):
            t_anchor = pd.Timestamp(anchor_ts[idx])
            t_target_seharusnya = t_anchor + pd.Timedelta(minutes=60)
            row_target = full[full["timestamp_wib"] == t_target_seharusnya]
            ghi_aktual_di_t60 = row_target["ghi_final"].values[0] if len(row_target) else None
            cocok = "OK" if (len(row_target) and abs(ghi_aktual_di_t60 - y[idx]) < 0.01) else "MISMATCH"
            print(f"    anchor={t_anchor}  target_dipakai={y[idx]:.2f}  "
                  f"ghi_aktual@T+60({t_target_seharusnya})={ghi_aktual_di_t60}  [{cocok}]")
    return np.array(X_seq), np.array(X_static), np.array(y)

def normalize(Xs_tr, Xs_va, Xs_te, Xst_tr, Xst_va, Xst_te):
    seq_mean = Xs_tr.reshape(-1, Xs_tr.shape[-1]).mean(0)
    seq_std = Xs_tr.reshape(-1, Xs_tr.shape[-1]).std(0) + 1e-6
    static_mean, static_std = Xst_tr.mean(0), Xst_tr.std(0) + 1e-6
    return ((Xs_tr - seq_mean) / seq_std, (Xs_va - seq_mean) / seq_std, (Xs_te - seq_mean) / seq_std,
            (Xst_tr - static_mean) / static_std, (Xst_va - static_mean) / static_std, (Xst_te - static_mean) / static_std)

def get_split_sequences(lookback, verify=False):
    train_full = df_full[df_full["year"].isin([2022, 2023])]
    val_full = df_full[df_full["year"] == 2024]
    test_full = df_full[df_full["year"] == 2025]
    Xs_tr, Xst_tr, y_tr = build_sequences(train_full, lookback)
    Xs_va, Xst_va, y_va = build_sequences(val_full, lookback)
    Xs_te, Xst_te, y_te = build_sequences(test_full, lookback, verify=verify)
    return Xs_tr, Xst_tr, y_tr, Xs_va, Xst_va, y_va, Xs_te, Xst_te, y_te

def patchtst_model(lookback, patch_len, n_features, d_model=32, n_heads=4):
    n_patches = lookback // patch_len
    seq_in = layers.Input(shape=(lookback, n_features))
    x = layers.Reshape((n_patches, patch_len * n_features))(seq_in)
    x = layers.Dense(d_model)(x)
    pos = layers.Embedding(input_dim=n_patches, output_dim=d_model)(tf.range(n_patches))
    x = x + pos
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=0.15)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(d_model * 2, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    static_in = layers.Input(shape=(len(STATIC_COLS),))
    x = layers.Concatenate()([x, static_in])
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)
    return models.Model([seq_in, static_in], out)

def lean_transformer_model(lookback, n_features, d_model=16, n_heads=2):
    seq_in = layers.Input(shape=(lookback, n_features))
    x = layers.Dense(d_model, kernel_regularizer=regularizers.l2(1e-3))(seq_in)
    pos = layers.Embedding(input_dim=lookback, output_dim=d_model)(tf.range(lookback))
    x = x + pos
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=0.3)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(d_model, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    static_in = layers.Input(shape=(len(STATIC_COLS),))
    x = layers.Concatenate()([x, static_in])
    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1)(x)
    return models.Model([seq_in, static_in], out)

def train_eval(model, data, label, epochs=100, patience=10, lr=1e-3):
    Xs_tr, Xst_tr, y_tr, Xs_va, Xst_va, y_va, Xs_te, Xst_te, y_te = data
    Xs_tr_n, Xs_va_n, Xs_te_n, Xst_tr_n, Xst_va_n, Xst_te_n = normalize(Xs_tr, Xs_va, Xs_te, Xst_tr, Xst_va, Xst_te)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    es = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model.fit([Xs_tr_n, Xst_tr_n], y_tr, validation_data=([Xs_va_n, Xst_va_n], y_va),
              epochs=epochs, batch_size=256, callbacks=[es], verbose=0)
    pred = model.predict([Xs_te_n, Xst_te_n], verbose=0).flatten()
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    print(f"{label:35s} R2={r2:.4f}  MAE={mae:.2f}")
    return r2

print("=== Verifikasi horizon + bangun sequences (lookback 12 = 2 jam) ===")
data12 = get_split_sequences(12, verify=True)
n_feat = len(SEQ_COLS)

print("\n=== PatchTST-style (lookback 12, patch_len 3 -> 4 patch) ===")
m_patch = patchtst_model(12, patch_len=3, n_features=n_feat)
r2_patch = train_eval(m_patch, data12, "PatchTST-style (12 step, patch=3)")

print("\n=== Transformer ramping + regularized (lookback 12) ===")
m_lean = lean_transformer_model(12, n_features=n_feat)
r2_lean = train_eval(m_lean, data12, "Lean-Transformer-regularized (12 step)")

print("\n=== Pembanding: LightGBM ensemble & Transformer/LSTM sebelumnya ===")
print(f"{'LightGBM (43 fitur enhanced)':35s} R2=0.7234")
print(f"{'Ensemble 3-model boosting':35s} R2=0.7264")
print(f"{'LSTM (lookback 60m, sesi lalu)':35s} R2=0.6742")
print(f"{'Transformer standar (60m, sesi lalu)':35s} R2=0.6770")
print(f"{'Transformer standar (3h, sesi lalu)':35s} R2=0.4930")
print(f"{'PatchTST-style (lookback 2h)':35s} R2={r2_patch:.4f}")
print(f"{'Lean-Transformer-regularized (2h)':35s} R2={r2_lean:.4f}")
