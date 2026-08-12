"""
train_ghi_lstm.py  —  GHI Forecast dengan LSTM (Keras/TensorFlow)
══════════════════════════════════════════════════════════════════
Mengapa LSTM lebih baik dari LightGBM untuk prediksi deret waktu:
  - LightGBM pada wide-format lag: fitur tabular statis, urutan diabaikan
  - LSTM: memproses sekuens (18 × n_fitur) secara berurutan, memori internal
    menangkap pola temporal yang tidak bisa diekstrak dari lag biasa
  - Untuk GHI tropis dengan variabilitas awan tinggi, perbedaan R² ≈ 5-10%

Input : Parquet dari build_ghi_forecast_v3.py (Stage 2 atau 3)
Output: Model .h5, prediksi CSV, ringkasan R²

Arsitektur:
  Input (batch, 18, n_features)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → Dropout(0.2)
    → Dense(32, activation='relu')
    → Dense(6)   ← prediksi GHI t+1 … t+6

Cara pakai:
  pip install tensorflow scikit-learn pyarrow pandas numpy
  python train_ghi_lstm.py --stage 3
  python train_ghi_lstm.py --stage 3 --epochs 100 --batch 512
"""

import os, sys, time, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Konfigurasi ───────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path(__file__).parent
WINDOW      = 18
HORIZON     = 6
H_LABELS    = [f"t+{h*10}min" for h in range(1, HORIZON + 1)]

# Kolom per stage (harus sama dengan build_ghi_forecast_v3.py)
STAGE_BASE_FEATS = {
    1: ["ghi_wm2", "dni_wm2", "dhi_wm2", "sun_altitude", "kt_approx"],
    2: ["ghi_wm2", "dni_wm2", "dhi_wm2", "sun_altitude", "kt_approx",
        "cloud_present_bin", "cloud_optical_thick",
        "cloud_top_height_m", "cloud_top_temp_k", "cloud_eff_radius_um"],
    3: ["ghi_wm2", "dni_wm2", "dhi_wm2", "sun_altitude", "kt_approx",
        "cloud_present_bin", "cloud_optical_thick",
        "cloud_top_height_m", "cloud_top_temp_k", "cloud_eff_radius_um",
        "delta_ghi_1", "delta_ghi_3", "delta_ghi_6",
        "ghi_std6", "delta_cot_1", "cot_std6"],
}
# ─────────────────────────────────────────────────────────────────────────────


def load_parquet(stage):
    path = OUTPUT_DIR / f"ghi_forecast_v3_stage{stage}.parquet"
    if not path.exists():
        print(f"✗ {path.name} tidak ditemukan.")
        print(f"  Jalankan dulu: python build_ghi_forecast_v3.py --stages {stage}")
        sys.exit(1)
    print(f"📥 Memuat {path.name} ...")
    df = pd.read_parquet(path)
    df["anchor_ts"] = pd.to_datetime(df["anchor_ts"])
    print(f"  ✓ {len(df):,} sampel, {len(df.columns)} kolom")
    return df


def reshape_to_sequences(df, stage):
    """
    Dari wide-format (lag kolom) → 3D numpy array (sampel, WINDOW, n_fitur).
    Juga kembalikan target (sampel, HORIZON) dan clearsky future (sampel, HORIZON).
    """
    features = STAGE_BASE_FEATS[stage]
    n_f      = len(features)

    # Susun lag kolom dalam urutan: lag0 (sekarang), lag1, ..., lag17
    X_list = []
    for lag in range(WINDOW):
        cols = [f"{f}_lag{lag}" for f in features]
        X_list.append(df[cols].values)  # (n, n_f)

    # Stack → (n, WINDOW, n_f), urutan temporal: X[:,0,:] = lag0 (terbaru)
    # Untuk LSTM, biasanya input dalam urutan kronologis (lama → baru)
    # → balik urutan: X[:,0,:] = lag17 (terlama), X[:,17,:] = lag0 (terbaru)
    X = np.stack(X_list[::-1], axis=1).astype(np.float32)  # (n, 18, n_f)

    # Target: GHI t+1 … t+6
    tgt_cols = [f"ghi_t{h}" for h in range(1, HORIZON + 1)]
    y        = df[tgt_cols].values.astype(np.float32)

    # Clearsky future (untuk konversi opsional)
    cs_cols = [f"ghi_cs_t{h}" for h in range(1, HORIZON + 1)]
    cs      = df[cs_cols].values.astype(np.float32)

    # Fitur tambahan (waktu + future sun altitude) → digabung ke timestep terakhir
    # atau bisa diberikan sebagai input terpisah (metode: auxiliary input)
    # Di sini: tambahkan sebagai extra fitur pada timestep anchor (lag0 = index 17)
    aux_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    aux_cols += [f"sun_alt_t{h}" for h in range(1, HORIZON + 1)]
    aux_cols += [f"ghi_cs_t{h}"  for h in range(1, HORIZON + 1)]
    X_aux    = df[aux_cols].values.astype(np.float32)

    return X, X_aux, y, cs, df["anchor_ts"].values


def split_by_year(ts_arr, X, X_aux, y, cs):
    """Split temporal: 2022-2023 train | 2024 val | 2025 test."""
    year = pd.to_datetime(ts_arr).year
    tr_mask = (year == 2022) | (year == 2023)
    va_mask = year == 2024
    te_mask = year == 2025

    def sel(m):
        return X[m], X_aux[m], y[m], cs[m], ts_arr[m]

    return sel(tr_mask), sel(va_mask), sel(te_mask)


def normalize(X_tr, X_va, X_te, X_aux_tr, X_aux_va, X_aux_te, y_tr):
    """
    Normalisasi min-max AMAN: tidak menggunakan sklearn MinMaxScaler karena
    fitur konstan (min==max) menghasilkan NaN yang merusak LSTM sejak batch pertama.
    Semua NaN/Inf dalam input juga dibersihkan dengan nan_to_num.
    """
    # 1. Bersihkan NaN / Inf terlebih dahulu (safety net)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_aux_tr = np.nan_to_num(X_aux_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_aux_va = np.nan_to_num(X_aux_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_aux_te = np.nan_to_num(X_aux_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 2. Normalisasi X: (n, 18, n_f) → flatten → fit pada train → transform semua
    n_tr, w, f = X_tr.shape
    flat_tr = X_tr.reshape(-1, f)   # (n_tr*18, n_f)

    feat_min   = flat_tr.min(axis=0)                     # (n_f,)
    feat_max   = flat_tr.max(axis=0)                     # (n_f,)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0   # fitur konstan → range=1 → scaled=0 (tidak NaN!)

    def scale_X(arr):
        s = arr.shape
        flat = arr.reshape(-1, f)
        scaled = (flat - feat_min) / feat_range * 2.0 - 1.0   # → [-1, 1]
        return np.nan_to_num(scaled, nan=0.0).reshape(s).astype(np.float32)

    X_tr_n = scale_X(X_tr)
    X_va_n = scale_X(X_va)
    X_te_n = scale_X(X_te)

    # 3. Normalisasi auxiliary
    aux_min   = X_aux_tr.min(axis=0)
    aux_max   = X_aux_tr.max(axis=0)
    aux_range = aux_max - aux_min
    aux_range[aux_range == 0] = 1.0

    def scale_aux(arr):
        scaled = (arr - aux_min) / aux_range * 2.0 - 1.0
        return np.nan_to_num(scaled, nan=0.0).astype(np.float32)

    X_aux_tr_n = scale_aux(X_aux_tr)
    X_aux_va_n = scale_aux(X_aux_va)
    X_aux_te_n = scale_aux(X_aux_te)

    # 4. Target GHI: normalisasi 0–1200 W/m²
    y_max  = 1200.0
    y_tr_n = np.clip(y_tr / y_max, 0.0, 1.5).astype(np.float32)

    # 5. Validasi: pastikan tidak ada NaN yang lolos
    for name, arr in [("X_tr_n", X_tr_n), ("X_aux_tr_n", X_aux_tr_n),
                      ("y_tr_n", y_tr_n)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            print(f"  ⚠  {name} masih ada {n_nan:,} NaN — paksa 0")
            arr[:] = np.nan_to_num(arr, nan=0.0)

    print(f"  ✓ Normalisasi: X shape={X_tr_n.shape}, "
          f"NaN X_tr={np.isnan(X_tr_n).sum()}, "
          f"NaN y_tr={np.isnan(y_tr_n).sum()}")

    return (X_tr_n, X_va_n, X_te_n,
            X_aux_tr_n, X_aux_va_n, X_aux_te_n,
            y_tr_n, y_max)


def build_model(n_seq_feats, n_aux_feats, horizon, dropout=0.2):
    """
    Arsitektur: LSTM + Auxiliary input (waktu + clearsky)

    seq_input (batch, 18, n_seq_feats) → LSTM → encoded state
    aux_input (batch, n_aux_feats)     → Dense(32) → auxiliary features
    concat → Dense(64) → Dense(6)     → GHI t+1..t+6
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("✗ TensorFlow tidak ditemukan. Jalankan: pip install tensorflow")
        sys.exit(1)

    # Sequence branch (LSTM)
    seq_in = keras.Input(shape=(WINDOW, n_seq_feats), name="seq_input")
    x = layers.LSTM(128, return_sequences=True, name="lstm1")(seq_in)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(64, name="lstm2")(x)
    x = layers.Dropout(dropout)(x)
    seq_out = layers.Dense(64, activation="relu")(x)

    # Auxiliary branch (waktu + clearsky future = deterministik)
    aux_in  = keras.Input(shape=(n_aux_feats,), name="aux_input")
    a = layers.Dense(32, activation="relu")(aux_in)
    a = layers.Dense(32, activation="relu")(a)

    # Gabung + output
    merged = layers.Concatenate()([seq_out, a])
    z      = layers.Dense(64, activation="relu")(merged)
    z      = layers.Dropout(dropout)(z)
    z      = layers.Dense(32, activation="relu")(z)
    output = layers.Dense(horizon, activation="relu", name="ghi_pred")(z)

    model = keras.Model(inputs=[seq_in, aux_in], outputs=output, name="GHI_LSTM")
    return model


def train_model(model, data_tr, data_va, epochs, batch_size, model_path):
    import tensorflow as tf
    from tensorflow.keras import callbacks as cb

    X_tr, X_aux_tr, y_tr_n = data_tr
    X_va, X_aux_va, y_va_n = data_va

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4,
            clipnorm=1.0,   # gradient clipping — cegah exploding gradients
        ),
        loss="huber",             # robust terhadap outlier (awan lewat tiba-tiba)
        metrics=["mae"],
    )

    callbacks_list = [
        cb.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=0),
        cb.ModelCheckpoint(str(model_path), save_best_only=True, verbose=0),
        cb.TerminateOnNaN(),        # stop segera jika NaN muncul kembali
    ]

    print(f"\n  Mulai training: {len(X_tr):,} train | {len(X_va):,} val")
    t0 = time.time()
    history = model.fit(
        [X_tr, X_aux_tr], y_tr_n,
        validation_data=([X_va, X_aux_va], y_va_n),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
    )
    elapsed = time.time() - t0
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    print(f"\n  ✓ Selesai dalam {elapsed:.0f}s | best epoch={best_epoch}")
    return history


def evaluate(model, data_te, y_max, stage):
    from sklearn.metrics import r2_score, mean_absolute_error

    X_te, X_aux_te, y_te, cs_te = data_te

    pred_n = model.predict([X_te, X_aux_te], verbose=0)
    pred   = (pred_n * y_max).clip(0)        # denormalisasi

    print(f"\n── Stage {stage} — LSTM (Test 2025) ──")
    r2_list, mae_list = [], []

    for h_i, lbl in enumerate(H_LABELS):
        y_true = y_te[:, h_i]
        y_pred = pred[:, h_i]
        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2_list.append(r2)
        mae_list.append(mae)
        status = "✅" if r2 >= 0.90 else ("~" if r2 >= 0.85 else "○")
        print(f"  {lbl:>10s}  R²={r2:.4f}  MAE={mae:6.1f} W/m²  {status}")

    avg = np.mean(r2_list)
    print(f"  {'Rata-rata':>10s}  R²={avg:.4f}  "
          f"{'✅ TARGET TERCAPAI' if avg >= 0.90 else '─'}")

    return r2_list, mae_list, pred


def compare_with_lgbm(r2_lstm, stage):
    """Bandingkan LSTM vs LightGBM v1/v3."""
    # R² terbaik LightGBM per stage (dari v1 Stage 3)
    lgbm_best = [0.863, 0.826, 0.804, 0.786, 0.773, 0.764]

    print("\n" + "=" * 60)
    print("📊 LSTM vs LightGBM (Test R²)")
    print("=" * 60)
    print(f"{'Horizon':<12}  {'LGBM-v1S3':>10}  {'LSTM':>10}  {'Delta':>8}")
    print("─" * 46)
    for i, lbl in enumerate(H_LABELS):
        delta = r2_lstm[i] - lgbm_best[i]
        sign  = "+" if delta >= 0 else ""
        print(f"{lbl:<12}  {lgbm_best[i]:>10.4f}  {r2_lstm[i]:>10.4f}  "
              f"{sign}{delta:>+7.4f}")
    print("─" * 46)
    avg_lgbm = np.mean(lgbm_best)
    avg_lstm = np.mean(r2_lstm)
    delta_avg = avg_lstm - avg_lgbm
    print(f"{'Rata-rata':<12}  {avg_lgbm:>10.4f}  {avg_lstm:>10.4f}  "
          f"{'+' if delta_avg >= 0 else ''}{delta_avg:>+7.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--epochs",     type=int, default=80)
    parser.add_argument("--batch",      type=int, default=256)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--skip-train", action="store_true",
                        help="Load model yang sudah ada (.h5) tanpa training ulang")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        # Gunakan GPU jika tersedia
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"  GPU: {[g.name for g in gpus]}")
        else:
            print("  GPU: tidak tersedia, pakai CPU")
    except ImportError:
        print("✗ pip install tensorflow")
        sys.exit(1)

    print(f"\n╔══════════════════════════════════════════════════╗")
    print(f"║   GHI LSTM Forecast — Stage {args.stage}                    ║")
    print(f"╚══════════════════════════════════════════════════╝")

    t_total = time.time()

    # 1. Load data
    df = load_parquet(args.stage)

    # 2. Reshape ke 3D sequence
    X, X_aux, y, cs, ts = reshape_to_sequences(df, args.stage)
    print(f"  Sequence shape: {X.shape}  (sampel, window, fitur)")
    print(f"  Aux shape     : {X_aux.shape}")
    print(f"  Target shape  : {y.shape}")

    # 3. Split temporal
    (X_tr, X_aux_tr, y_tr, cs_tr, ts_tr), \
    (X_va, X_aux_va, y_va, cs_va, ts_va), \
    (X_te, X_aux_te, y_te, cs_te, ts_te) = split_by_year(ts, X, X_aux, y, cs)
    print(f"\n  Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")

    # 4. Normalisasi
    (X_tr_n, X_va_n, X_te_n,
     X_aux_tr_n, X_aux_va_n, X_aux_te_n,
     y_tr_n, y_max) = normalize(
        X_tr, X_va, X_te, X_aux_tr, X_aux_va, X_aux_te, y_tr
    )
    y_va_n = np.clip(y_va / y_max, 0.0, 1.5).astype(np.float32)

    # 5. Build / load model
    model_path = OUTPUT_DIR / f"ghi_lstm_stage{args.stage}.keras"
    n_seq_f    = X_tr_n.shape[2]
    n_aux_f    = X_aux_tr_n.shape[1]

    model = build_model(n_seq_f, n_aux_f, HORIZON, dropout=args.dropout)
    model.summary(print_fn=lambda s: print(f"  {s}"))

    if args.skip_train and model_path.exists():
        print(f"\n📂 Memuat model dari {model_path.name} ...")
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
    else:
        train_model(
            model,
            (X_tr_n, X_aux_tr_n, y_tr_n),
            (X_va_n, X_aux_va_n, y_va_n),
            epochs=args.epochs,
            batch_size=args.batch,
            model_path=model_path,
        )

    # 6. Evaluasi
    r2_list, mae_list, pred = evaluate(
        model,
        (X_te_n, X_aux_te_n, y_te, cs_te),
        y_max, args.stage,
    )

    # 7. Bandingkan dengan LightGBM
    compare_with_lgbm(r2_list, args.stage)

    # 8. Simpan prediksi
    pred_df = pd.DataFrame({"anchor_ts": ts_te})
    for h_i in range(HORIZON):
        pred_df[f"ghi_pred_t{h_i+1}"] = pred[:, h_i]
        pred_df[f"ghi_true_t{h_i+1}"] = y_te[:, h_i]
    pred_path = OUTPUT_DIR / f"ghi_lstm_predictions_stage{args.stage}.csv"
    pred_df.to_csv(pred_path, index=False)

    # 9. Simpan ringkasan evaluasi
    rows = []
    for i, lbl in enumerate(H_LABELS):
        rows.append({
            "model": f"LSTM_stage{args.stage}",
            "horizon": lbl,
            "test_r2": round(r2_list[i], 5),
            "test_mae": round(mae_list[i], 2),
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / f"ghi_lstm_eval_stage{args.stage}.csv", index=False
    )

    print(f"\n💾 Prediksi  → {pred_path.name}")
    print(f"💾 Model     → {model_path.name}")
    print(f"✅ Selesai dalam {time.time()-t_total:.1f} detik")


if __name__ == "__main__":
    main()
