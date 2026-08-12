"""
build_ghi_forecast_dataset.py
══════════════════════════════════════════════════════════════════════
Membangun dataset ML untuk prediksi GHI 1 jam ke depan (t+6 × 10 menit)
dari sliding window 3 jam sebelumnya (18 × 10 menit).

TAHAP FITUR:
  Stage 1 — Radiasi matahari (GHI, DNI, DHI, sun_altitude, kt)
  Stage 2 — Stage 1 + Meteorologi (suhu, RH, tekanan, angin, hujan)
  Stage 3 — Stage 2 + Cloud properties (Himawari-8/9: COT, CTH, CTT, CLER)

OUTPUT (per stage):
  ghi_forecast_stage{1,2,3}.parquet   ← dataset fitur lag + target
  ghi_forecast_eval.csv               ← ringkasan R² per stage per horizon

EVALUASI:
  LightGBM, split temporal: 2022-2023 train | 2024 val | 2025 test
  R² untuk t+10mnt … t+60mnt (t+1 … t+6)

Cara pakai:
  pip install lightgbm scikit-learn pyarrow
  python build_ghi_forecast_dataset.py
  python build_ghi_forecast_dataset.py --token YOUR_TOKEN
  python build_ghi_forecast_dataset.py --skip-train   # hanya buat dataset
  python build_ghi_forecast_dataset.py --stages 3     # hanya stage 3
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(__file__).parent
WINDOW       = 18     # langkah input ke belakang (18 × 10 mnt = 180 mnt = 3 jam)
HORIZON      = 6      # langkah prediksi ke depan  (6  × 10 mnt = 60 mnt  = 1 jam)
MIN_ALT      = 5.0    # sun_altitude minimum (derajat) — filter siang hari
MAX_GAP_MIN  = 10     # gap maksimum antar timestep (menit) — kontinuitas data

# ── Kolom fitur per stage ─────────────────────────────────────────────────────
# Lag features: setiap kolom di-lag dari lag0 (sekarang) s.d. lag{WINDOW-1}
RAD_COLS = [
    "ghi_wm2", "dni_wm2", "dhi_wm2",
    "sun_altitude", "kt",
]
MET_COLS = [
    "temp_air_c", "humidity_pct", "pressure_qff_mb",
    "wind_speed_ms", "wind_sin", "wind_cos", "rainfall_mm",
]
CLP_COLS = [
    "cloud_present_bin", "cloud_optical_thick",
    "cloud_top_height_m", "cloud_top_temp_k", "cloud_eff_radius_um",
]

STAGE_COLS = {
    1: RAD_COLS,
    2: RAD_COLS + MET_COLS,
    3: RAD_COLS + MET_COLS + CLP_COLS,
}
# ─────────────────────────────────────────────────────────────────────────────


def connect_motherduck(token=None):
    tok = token or os.environ.get("MOTHERDUCK_TOKEN", "")
    url = f"md:kalbar?motherduck_token={tok}" if tok else "md:kalbar"
    print(f"🔌 Menghubungkan ke MotherDuck ...")
    con = duckdb.connect(url)
    print("  ✓ Terhubung")
    return con


def load_data(con):
    print("\n📥 Memuat data dari solar_radiation_valid ...")
    t0 = time.time()

    df = con.execute("""
        SELECT
            timestamp_wib,
            sun_altitude,
            -- Radiasi matahari
            COALESCE(ghi_wm2, 0)       AS ghi_wm2,
            COALESCE(dni_wm2, 0)       AS dni_wm2,
            COALESCE(dhi_wm2, 0)       AS dhi_wm2,
            -- Meteorologi
            temp_air_c,
            humidity_pct,
            pressure_qff_mb,
            wind_speed_ms,
            wind_dir_deg,
            COALESCE(rainfall_mm, 0)   AS rainfall_mm,
            -- Cloud properties (Himawari-8/9)
            CASE WHEN cloud_present = TRUE  THEN 1.0
                 WHEN cloud_present = FALSE THEN 0.0
                 ELSE NULL END                       AS cloud_present_bin,
            cloud_optical_thick,
            cloud_top_height_m,
            cloud_top_temp_k,
            cloud_eff_radius_um
        FROM kalbar.main.solar_radiation_valid
        ORDER BY timestamp_wib
    """).df()

    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    print(f"  ✓ {len(df):,} baris ({time.time()-t0:.1f}s) | "
          f"{df['timestamp_wib'].min().date()} – {df['timestamp_wib'].max().date()}")
    return df


def feature_engineer(df):
    """Tambah fitur turunan: wind cyclical, clearsky GHI, clearness index kt."""
    print("\n🔧 Feature engineering ...")

    # Angin → komponen siklik (hindari discontinuity di 0°/360°)
    wind_rad      = np.radians(df["wind_dir_deg"].fillna(0.0))
    df["wind_sin"] = np.sin(wind_rad).astype(np.float32)
    df["wind_cos"] = np.cos(wind_rad).astype(np.float32)

    # Clear-sky GHI (model empiris Pontianak – cocok tropis dekat ekuator)
    sin_alt = np.sin(np.radians(df["sun_altitude"].clip(0))).clip(0)
    df["ghi_clearsky"] = np.where(
        df["sun_altitude"] > 0,
        (950.0 * np.power(sin_alt, 1.1)).astype(np.float32),
        0.0,
    )

    # Clearness index kt = GHI / GHI_cs (0–1.2)
    cs = df["ghi_clearsky"].values
    df["kt"] = np.where(
        cs > 10,
        (df["ghi_wm2"].values / cs).clip(0, 1.2),
        0.0,
    ).astype(np.float32)

    # Isi NULL cloud properties (malam hari = 0.0)
    for col in ["cloud_optical_thick", "cloud_top_height_m",
                "cloud_top_temp_k", "cloud_eff_radius_um", "cloud_present_bin"]:
        df[col] = df[col].fillna(0.0)

    # Isi NULL meteorologi (hapus baris jika masih NULL setelah ini
    # tidak dilakukan di sini karena sudah 100% coverage dari sesi sebelumnya)
    for col in ["temp_air_c", "humidity_pct", "pressure_qff_mb",
                "wind_speed_ms", "rainfall_mm"]:
        df[col] = df[col].ffill().bfill()

    print(f"  ✓ Kolom tambahan: wind_sin, wind_cos, ghi_clearsky, kt")
    return df


def compute_valid_anchors(df):
    """
    Temukan indeks anchor yang memenuhi syarat:
    1. Window [i-WINDOW+1 … i] dan horizon [i+1 … i+HORIZON] keduanya kontinu
       (tidak ada gap > MAX_GAP_MIN menit di seluruh rentang)
    2. Anchor daytime: sun_altitude[i] > MIN_ALT
    3. Target akhir daytime: sun_altitude[i+HORIZON] > MIN_ALT
       (prediksi 60 menit ke depan bermakna, bukan trivial nol)
    """
    print("\n🔍 Menghitung valid anchor positions ...")
    t0 = time.time()

    n   = len(df)
    alt = df["sun_altitude"].values
    ts  = df["timestamp_wib"].values

    # Deteksi break point: gap antar baris > MAX_GAP_MIN menit
    diffs   = pd.to_datetime(ts).to_series().diff().dt.total_seconds().div(60).fillna(0)
    breaks  = (diffs > MAX_GAP_MIN).astype(int).values
    cum_b   = np.cumsum(breaks)   # kumulatif breaks untuk range check O(1)

    valid = []
    for i in range(WINDOW - 1, n - HORIZON):
        start = i - WINDOW + 1
        end   = i + HORIZON
        # 1) Kontinu
        if cum_b[end] - cum_b[start] > 0:
            continue
        # 2) Anchor siang hari
        if alt[i] <= MIN_ALT:
            continue
        # 3) Target akhir (t+6 = 60 mnt ke depan) siang hari
        if alt[i + HORIZON] <= MIN_ALT:
            continue
        valid.append(i)

    n_valid = len(valid)
    pct     = n_valid / n * 100
    print(f"  ✓ {n_valid:,} valid anchors dari {n:,} baris ({pct:.1f}%) "
          f"[{time.time()-t0:.1f}s]")
    return np.array(valid, dtype=np.int64)


def build_dataset(df, valid_anchors, stage):
    """
    Buat wide-format DataFrame untuk satu stage.

    Struktur kolom:
      anchor_ts                   ← timestamp anchor
      {fitur}_lag{0..W-1}         ← fitur pada t, t-1, ..., t-(W-1)
                                      lag0 = sekarang (anchor), lag17 = 170 mnt lalu
      hour_sin, hour_cos,
      doy_sin, doy_cos            ← waktu siklik anchor
      sun_alt_t{1..H}             ← sun altitude masa depan (deterministik)
      ghi_cs_t{1..H}              ← clearsky GHI masa depan (deterministik)
      ghi_t{1..H}                 ← TARGET: GHI di t+1 … t+H
    """
    features = STAGE_COLS[stage]
    n_feats  = len(features)
    n_samp   = len(valid_anchors)

    print(f"\n📦 Membangun Stage {stage} "
          f"({n_feats} fitur × {WINDOW} lag = {n_feats*WINDOW} kolom lag) ...")
    t0 = time.time()

    # Ekstrak numpy arrays untuk kecepatan
    feat_np  = df[features].values.astype(np.float32)
    ghi_np   = df["ghi_wm2"].values.astype(np.float32)
    alt_np   = df["sun_altitude"].values.astype(np.float32)
    cs_np    = df["ghi_clearsky"].values.astype(np.float32)
    ts_np    = df["timestamp_wib"].values

    # Fitur waktu anchor
    ts_pd     = pd.to_datetime(ts_np)
    hour_frac = ts_pd.hour + ts_pd.minute / 60.0
    doy_arr   = ts_pd.dayofyear.values.astype(np.float32)
    hs = np.sin(2 * np.pi * hour_frac / 24).astype(np.float32)
    hc = np.cos(2 * np.pi * hour_frac / 24).astype(np.float32)
    ds = np.sin(2 * np.pi * doy_arr / 365.0).astype(np.float32)
    dc = np.cos(2 * np.pi * doy_arr / 365.0).astype(np.float32)

    # Alokasi output
    n_lag_cols  = n_feats * WINDOW
    n_time_cols = 4
    n_fut_cols  = HORIZON * 2   # alt + clearsky
    n_tgt_cols  = HORIZON
    n_total     = n_lag_cols + n_time_cols + n_fut_cols + n_tgt_cols

    out = np.empty((n_samp, n_total), dtype=np.float32)

    for j, i in enumerate(valid_anchors):
        ptr = 0
        # Lag features: lag0 = anchor (t), lag1 = t-1, ..., lag{W-1} = t-(W-1)
        for lag in range(WINDOW):
            out[j, ptr:ptr + n_feats] = feat_np[i - lag]
            ptr += n_feats
        # Waktu siklik (anchor)
        out[j, ptr:ptr + 4] = [hs[i], hc[i], ds[i], dc[i]]
        ptr += 4
        # Fitur masa depan deterministik
        for h in range(1, HORIZON + 1):
            out[j, ptr]     = alt_np[i + h]
            out[j, ptr + 1] = cs_np[i + h]
            ptr += 2
        # Target
        for h in range(1, HORIZON + 1):
            out[j, ptr] = ghi_np[i + h]
            ptr += 1

    # Susun nama kolom
    col_names = []
    for lag in range(WINDOW):
        for f in features:
            col_names.append(f"{f}_lag{lag}")
    col_names += ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    for h in range(1, HORIZON + 1):
        col_names += [f"sun_alt_t{h}", f"ghi_cs_t{h}"]
    for h in range(1, HORIZON + 1):
        col_names.append(f"ghi_t{h}")

    result = pd.DataFrame(out, columns=col_names)
    result.insert(0, "anchor_ts", ts_np[valid_anchors])

    mem_mb = result.memory_usage(deep=True).sum() / (1024**2)
    print(f"  ✓ {len(result):,} sampel × {len(result.columns)} kolom "
          f"({mem_mb:.0f} MB RAM) [{time.time()-t0:.1f}s]")
    return result


def save_parquet(df, stage):
    path = OUTPUT_DIR / f"ghi_forecast_stage{stage}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / (1024**2)
    print(f"  💾 Disimpan → {path.name} ({size_mb:.1f} MB)")
    return path


def train_and_evaluate(datasets, skip_train=False):
    if skip_train:
        print("\n⏭  --skip-train: melewati training model")
        return

    try:
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("\n⚠  lightgbm/scikit-learn tidak ditemukan. Jalankan:")
        print("   pip install lightgbm scikit-learn")
        return

    print("\n" + "=" * 64)
    print("🤖 TRAINING & EVALUASI LIGHTGBM")
    print("=" * 64)
    print("Split temporal: 2022–2023 train | 2024 val | 2025 test")

    TARGET_COLS = [f"ghi_t{h}" for h in range(1, HORIZON + 1)]
    H_LABELS    = [f"t+{h*10}min" for h in range(1, HORIZON + 1)]

    LGBM_PARAMS = dict(
        n_estimators   = 600,
        learning_rate  = 0.05,
        num_leaves     = 127,
        min_child_samples = 20,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        reg_alpha      = 0.1,
        reg_lambda     = 1.0,
        n_jobs         = -1,
        verbose        = -1,
    )

    all_results = {}

    for stage, df in datasets.items():
        print(f"\n── Stage {stage} ({len(STAGE_COLS[stage])} fitur base) ──")
        year = pd.to_datetime(df["anchor_ts"]).dt.year
        train_df = df[year.isin([2022, 2023])]
        val_df   = df[year == 2024]
        test_df  = df[year == 2025]
        print(f"  Sampel → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        feat_cols = [c for c in df.columns if c not in ["anchor_ts"] + TARGET_COLS]
        X_tr = train_df[feat_cols].values
        X_va = val_df[feat_cols].values
        X_te = test_df[feat_cols].values

        r2_v_list, r2_t_list, mae_t_list = [], [], []

        for h_i, tgt in enumerate(TARGET_COLS):
            y_tr = train_df[tgt].values
            y_va = val_df[tgt].values
            y_te = test_df[tgt].values

            m = lgb.LGBMRegressor(**LGBM_PARAMS)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            pred_va = m.predict(X_va).clip(0)
            pred_te = m.predict(X_te).clip(0)

            r2_v  = r2_score(y_va, pred_va)
            r2_t  = r2_score(y_te, pred_te)
            mae_t = mean_absolute_error(y_te, pred_te)

            r2_v_list.append(r2_v)
            r2_t_list.append(r2_t)
            mae_t_list.append(mae_t)

            status = "✅" if r2_t >= 0.90 else ("⚠" if r2_t >= 0.80 else "✗")
            print(f"  {H_LABELS[h_i]:>10s}  Val R²={r2_v:.4f}  "
                  f"Test R²={r2_t:.4f}  MAE={mae_t:6.1f} W/m²  {status}")

        avg = np.mean(r2_t_list)
        tgt_status = "✅ TARGET TERCAPAI" if avg >= 0.90 else \
                     f"{'~' if avg >= 0.85 else '✗'} Rata-rata {avg:.4f}"
        print(f"  {'':>10s}  {'─'*44}")
        print(f"  {'Rata-rata':>10s}  Test R²={avg:.4f}  {tgt_status}")

        all_results[stage] = {"val": r2_v_list, "test": r2_t_list, "mae": mae_t_list}

    # ── Ringkasan perbandingan ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("📊 PERBANDINGAN STAGE (Test R²)")
    print("=" * 64)

    stage_list = sorted(all_results.keys())
    header = f"{'Horizon':<12}" + "".join(f"{'Stage '+str(s):>10}" for s in stage_list)
    print(header)
    print("─" * len(header))
    for h_i, lbl in enumerate(H_LABELS):
        row = f"{lbl:<12}" + "".join(
            f"{all_results[s]['test'][h_i]:>10.4f}" for s in stage_list
        )
        print(row)
    print("─" * len(header))
    avg_row = f"{'Rata-rata':<12}" + "".join(
        f"{np.mean(all_results[s]['test']):>10.4f}" for s in stage_list
    )
    print(avg_row)
    print("=" * 64)

    # Peningkatan stage 1→2→3
    if len(stage_list) >= 2:
        print("\n📈 Peningkatan R² (rata-rata):")
        for a, b in zip(stage_list, stage_list[1:]):
            delta = np.mean(all_results[b]["test"]) - np.mean(all_results[a]["test"])
            sign  = "+" if delta >= 0 else ""
            print(f"   Stage {a} → Stage {b}: {sign}{delta:+.4f}")

    # ── Simpan CSV evaluasi ───────────────────────────────────────────────────
    rows_out = []
    for s in stage_list:
        for h_i, lbl in enumerate(H_LABELS):
            rows_out.append({
                "stage"    : s,
                "horizon"  : lbl,
                "val_r2"   : round(all_results[s]["val"][h_i], 5),
                "test_r2"  : round(all_results[s]["test"][h_i], 5),
                "test_mae" : round(all_results[s]["mae"][h_i], 2),
            })
    eval_path = OUTPUT_DIR / "ghi_forecast_eval.csv"
    pd.DataFrame(rows_out).to_csv(eval_path, index=False)
    print(f"\n💾 Evaluasi disimpan → {eval_path.name}")


def print_dataset_info(df, stage):
    """Tampilkan statistik ringkas dataset."""
    tgt_cols = [c for c in df.columns if c.startswith("ghi_t")]
    feat_cols = [c for c in df.columns if c not in ["anchor_ts"] + tgt_cols]

    ts  = pd.to_datetime(df["anchor_ts"])
    ghi6 = df["ghi_t6"].values

    print(f"\n  📋 Stage {stage} info:")
    print(f"     Fitur    : {len(feat_cols)} kolom")
    print(f"     Sampel   : {len(df):,}")
    print(f"     Rentang  : {ts.min().date()} – {ts.max().date()}")
    print(f"     Target ghi_t6 → mean={ghi6.mean():.1f}  "
          f"std={ghi6.std():.1f}  max={ghi6.max():.1f} W/m²")


def main():
    parser = argparse.ArgumentParser(
        description="Build GHI forecast dataset (3-stage) dari MotherDuck kalbar"
    )
    parser.add_argument("--token",      help="MotherDuck token (atau env MOTHERDUCK_TOKEN)")
    parser.add_argument("--stages",     nargs="+", type=int, default=[1, 2, 3],
                        help="Stage yang dibuat, mis: --stages 2 3")
    parser.add_argument("--skip-train", action="store_true",
                        help="Buat dataset saja, tanpa training LightGBM")
    parser.add_argument("--load-local", action="store_true",
                        help="Muat dari Parquet lokal yang sudah ada (tidak ke MD)")
    args = parser.parse_args()

    t_total = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   GHI Forecast Dataset Builder — Kalimantan Barat        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Window  : {WINDOW} langkah × 10 mnt = {WINDOW*10} mnt ke belakang")
    print(f"Horizon : {HORIZON} langkah × 10 mnt = {HORIZON*10} mnt ke depan")
    print(f"Stage   : {args.stages}")
    print(f"Output  : {OUTPUT_DIR}")

    if args.load_local:
        # ── Mode: load Parquet lokal ─────────────────────────────────────────
        print("\n📂 Mode: load dari Parquet lokal ...")
        datasets = {}
        for s in args.stages:
            p = OUTPUT_DIR / f"ghi_forecast_stage{s}.parquet"
            if not p.exists():
                print(f"  ✗ {p.name} tidak ditemukan. Jalankan tanpa --load-local dulu.")
                sys.exit(1)
            datasets[s] = pd.read_parquet(p)
            print_dataset_info(datasets[s], s)
    else:
        # ── Mode: build dari MotherDuck ──────────────────────────────────────
        con = connect_motherduck(args.token)
        df  = load_data(con)
        con.close()

        df   = feature_engineer(df)
        anch = compute_valid_anchors(df)

        datasets = {}
        for s in args.stages:
            ds = build_dataset(df, anch, s)
            save_parquet(ds, s)
            print_dataset_info(ds, s)
            datasets[s] = ds

    # ── Train & evaluate ──────────────────────────────────────────────────────
    train_and_evaluate(datasets, skip_train=args.skip_train)

    print(f"\n✅ Selesai dalam {time.time()-t_total:.1f} detik")
    print(f"   Output : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()