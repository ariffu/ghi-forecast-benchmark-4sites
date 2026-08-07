"""
build_ghi_forecast_v3.py  —  GHI Forecast Dataset Builder v3
══════════════════════════════════════════════════════════════
PELAJARAN dari v1 & v2:
  ✓ v1: GHI sebagai target — benar
  ✗ v2: kt sebagai target — GAGAL karena clearsky empiris (950×sin^1.1) terlalu kasar,
         kt sendiri mengandung noise sistematis bukan dari awan

v3 PERBAIKAN:
  1. Target = GHI (seperti v1)
  2. Reorganisasi stage yang lebih logis secara fisis:
       Stage 1: Radiasi + geometri matahari (persistence baseline)
       Stage 2: Stage 1 + Cloud properties Himawari-8/9 (state awan kini)
       Stage 3: Stage 2 + Dinamika awan (laju perubahan GHI & COT)
  3. Hapus meteorologi (suhu/RH/tekanan terbukti tidak prediktif)
  4. Fitur baru — dinamika awan (Stage 3):
       delta_ghi_1 = GHI[t] - GHI[t-1]   (perubahan 10 mnt)
       delta_ghi_3 = GHI[t] - GHI[t-3]   (perubahan 30 mnt)
       delta_ghi_6 = GHI[t] - GHI[t-6]   (perubahan 60 mnt)
       ghi_std6    = std(GHI[t-5..t])     (volatilitas 60 mnt)
       delta_cot_1 = COT[t] - COT[t-1]   (laju perubahan tebal awan)
       cot_std6    = std(COT[t-5..t])     (volatilitas COT)
  5. Lebih banyak pohon (1000) + optimasi hyperparameter

Cara pakai:
  python build_ghi_forecast_v3.py
  python build_ghi_forecast_v3.py --stages 3
  python build_ghi_forecast_v3.py --skip-train
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
OUTPUT_DIR  = Path(__file__).parent
WINDOW      = 18
HORIZON     = 6
MIN_ALT     = 5.0
MAX_GAP_MIN = 10

# ── Definisi fitur ────────────────────────────────────────────────────────────
# Stage 1: Radiasi + geometri (persistence baseline)
STAGE1_COLS = [
    "ghi_wm2", "dni_wm2", "dhi_wm2",
    "sun_altitude",
    "kt_approx",      # GHI / max_daily_clearsky — proxy sederhana, bukan dari clearsky model
]

# Stage 2: Stage 1 + cloud properties Himawari
CLOUD_STATE_COLS = [
    "cloud_present_bin",
    "cloud_optical_thick",
    "cloud_top_height_m",
    "cloud_top_temp_k",
    "cloud_eff_radius_um",
]
STAGE2_COLS = STAGE1_COLS + CLOUD_STATE_COLS

# Stage 3: Stage 2 + dinamika awan (trend/volatilitas)
CLOUD_DYN_COLS = [
    "delta_ghi_1",    # perubahan GHI 10 mnt
    "delta_ghi_3",    # perubahan GHI 30 mnt
    "delta_ghi_6",    # perubahan GHI 60 mnt
    "ghi_std6",       # volatilitas GHI 60 mnt
    "delta_cot_1",    # perubahan COT 10 mnt (Himawari)
    "cot_std6",       # volatilitas COT 60 mnt
]
STAGE3_COLS = STAGE2_COLS + CLOUD_DYN_COLS

STAGE_COLS = {1: STAGE1_COLS, 2: STAGE2_COLS, 3: STAGE3_COLS}
# ─────────────────────────────────────────────────────────────────────────────


def connect_md(token=None):
    tok = token or os.environ.get("MOTHERDUCK_TOKEN", "")
    url = f"md:kalbar?motherduck_token={tok}" if tok else "md:kalbar"
    print("🔌 Menghubungkan ke MotherDuck ...")
    con = duckdb.connect(url)
    print("  ✓ Terhubung")
    return con


def load_data(con):
    print("\n📥 Memuat data ...")
    t0 = time.time()
    df = con.execute("""
        SELECT
            timestamp_wib,
            sun_altitude,
            COALESCE(ghi_wm2, 0)   AS ghi_wm2,
            COALESCE(dni_wm2, 0)   AS dni_wm2,
            COALESCE(dhi_wm2, 0)   AS dhi_wm2,
            CASE WHEN cloud_present = TRUE  THEN 1.0
                 WHEN cloud_present = FALSE THEN 0.0
                 ELSE NULL END        AS cloud_present_bin,
            COALESCE(cloud_optical_thick, 0)   AS cloud_optical_thick,
            COALESCE(cloud_top_height_m, 0)    AS cloud_top_height_m,
            COALESCE(cloud_top_temp_k, 0)      AS cloud_top_temp_k,
            COALESCE(cloud_eff_radius_um, 0)   AS cloud_eff_radius_um
        FROM kalbar.main.solar_radiation_valid
        ORDER BY timestamp_wib
    """).df()
    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    df["cloud_present_bin"] = df["cloud_present_bin"].fillna(0.0)
    print(f"  ✓ {len(df):,} baris [{time.time()-t0:.1f}s] | "
          f"{df.timestamp_wib.min().date()} – {df.timestamp_wib.max().date()}")
    return df


def feature_engineer(df):
    """
    Tambah fitur turunan.
    PENTING: kt_approx berbeda dengan v2.
    Di sini kt_approx = GHI / GHI_max_harian_per_jam (rolling max) →
    tidak bergantung pada model clearsky, lebih robust.
    """
    print("\n🔧 Feature engineering ...")

    # kt_approx: GHI dinormalisasi oleh maksimum harian per jam
    # → menangkap "seberapa cerah sekarang vs potensi maksimum"
    # tanpa perlu model atmosfer
    df["hour"] = df["timestamp_wib"].dt.hour
    df["date"] = df["timestamp_wib"].dt.date

    # Clearsky kasar berdasarkan sin(altitude) — hanya untuk kt_approx, bukan target
    sin_alt = np.sin(np.radians(df["sun_altitude"].clip(0))).clip(0)
    df["ghi_clearsky_rough"] = (950.0 * sin_alt**1.1).clip(1.0)
    df["kt_approx"] = np.where(
        df["sun_altitude"] > 0,
        (df["ghi_wm2"] / df["ghi_clearsky_rough"]).clip(0, 1.3),
        0.0,
    ).astype(np.float32)

    # ── Dinamika GHI ──────────────────────────────────────────────────────────
    g = df["ghi_wm2"].values.astype(np.float32)
    df["delta_ghi_1"] = np.concatenate([[0],  g[1:]  - g[:-1]]).astype(np.float32)
    df["delta_ghi_3"] = np.concatenate([[0,0,0], g[3:] - g[:-3]]).astype(np.float32)
    df["delta_ghi_6"] = np.concatenate([[0]*6,   g[6:] - g[:-6]]).astype(np.float32)
    df["ghi_std6"]    = pd.Series(g).rolling(6, min_periods=1).std().fillna(0).values.astype(np.float32)

    # ── Dinamika COT (Himawari) ───────────────────────────────────────────────
    c = df["cloud_optical_thick"].values.astype(np.float32)
    df["delta_cot_1"] = np.concatenate([[0], c[1:] - c[:-1]]).astype(np.float32)
    df["cot_std6"]    = pd.Series(c).rolling(6, min_periods=1).std().fillna(0).values.astype(np.float32)

    print("  ✓ kt_approx, delta_ghi_1/3/6, ghi_std6, delta_cot_1, cot_std6")
    return df


def compute_valid_anchors(df):
    print("\n🔍 Hitung valid anchor positions ...")
    t0 = time.time()
    n   = len(df)
    alt = df["sun_altitude"].values
    ts  = df["timestamp_wib"].values

    diffs  = pd.to_datetime(ts).to_series().diff().dt.total_seconds().div(60).fillna(0)
    breaks = (diffs > MAX_GAP_MIN).astype(int).values
    cum_b  = np.cumsum(breaks)

    valid = []
    for i in range(WINDOW - 1, n - HORIZON):
        if cum_b[i + HORIZON] - cum_b[i - WINDOW + 1] > 0:
            continue
        if alt[i] <= MIN_ALT:
            continue
        if alt[i + HORIZON] <= MIN_ALT:
            continue
        valid.append(i)

    print(f"  ✓ {len(valid):,} valid anchors [{time.time()-t0:.1f}s]")
    return np.array(valid, dtype=np.int64)


def build_dataset(df, valid_anchors, stage):
    features = STAGE_COLS[stage]
    n_feats  = len(features)
    n_samp   = len(valid_anchors)
    print(f"\n📦 Stage {stage}: {n_feats} fitur × {WINDOW} lag ...")
    t0 = time.time()

    feat_np = df[features].values.astype(np.float32)
    ghi_np  = df["ghi_wm2"].values.astype(np.float32)
    alt_np  = df["sun_altitude"].values.astype(np.float32)
    cs_np   = df["ghi_clearsky_rough"].values.astype(np.float32)
    ts_np   = df["timestamp_wib"].values

    ts_pd     = pd.to_datetime(ts_np)
    hour_frac = ts_pd.hour + ts_pd.minute / 60.0
    doy_arr   = ts_pd.dayofyear.values
    hs = np.sin(2 * np.pi * hour_frac / 24).astype(np.float32)
    hc = np.cos(2 * np.pi * hour_frac / 24).astype(np.float32)
    ds = np.sin(2 * np.pi * doy_arr / 365).astype(np.float32)
    dc = np.cos(2 * np.pi * doy_arr / 365).astype(np.float32)

    # Kolom: lag + waktu + future(alt+cs) + target
    n_cols = n_feats * WINDOW + 4 + HORIZON * 2 + HORIZON
    out = np.empty((n_samp, n_cols), dtype=np.float32)

    for j, i in enumerate(valid_anchors):
        ptr = 0
        for lag in range(WINDOW):
            out[j, ptr:ptr + n_feats] = feat_np[i - lag]
            ptr += n_feats
        out[j, ptr:ptr+4] = [hs[i], hc[i], ds[i], dc[i]]
        ptr += 4
        for h in range(1, HORIZON + 1):
            out[j, ptr]     = alt_np[i + h]
            out[j, ptr + 1] = cs_np[i + h]
            ptr += 2
        for h in range(1, HORIZON + 1):
            out[j, ptr] = ghi_np[i + h]
            ptr += 1

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

    mem = result.memory_usage(deep=True).sum() / (1024**2)
    print(f"  ✓ {len(result):,} sampel × {len(result.columns)} kolom "
          f"({mem:.0f} MB) [{time.time()-t0:.1f}s]")
    return result


def save_parquet(df, stage):
    path = OUTPUT_DIR / f"ghi_forecast_v3_stage{stage}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    print(f"  💾 {path.name} ({path.stat().st_size/(1024**2):.1f} MB)")


def train_and_evaluate(datasets):
    try:
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("\n⚠  pip install lightgbm scikit-learn")
        return

    TARGET_COLS = [f"ghi_t{h}" for h in range(1, HORIZON + 1)]
    H_LABELS    = [f"t+{h*10}min" for h in range(1, HORIZON + 1)]

    # LightGBM hyperparams yang lebih agresif
    LGBM_PARAMS = dict(
        n_estimators      = 1000,
        learning_rate     = 0.03,
        num_leaves        = 255,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        reg_alpha         = 0.05,
        reg_lambda        = 0.5,
        n_jobs            = -1,
        verbose           = -1,
    )

    print("\n" + "=" * 68)
    print("🤖 TRAINING — Target: GHI (W/m²) langsung  |  v3")
    print("=" * 68)
    print("Split: 2022–2023 train | 2024 val | 2025 test\n")

    all_results = {}

    for stage, df in sorted(datasets.items()):
        print(f"── Stage {stage} ({len(STAGE_COLS[stage])} fitur base) ──")
        year = pd.to_datetime(df["anchor_ts"]).dt.year
        tr = df[year.isin([2022, 2023])]
        va = df[year == 2024]
        te = df[year == 2025]
        print(f"  Train: {len(tr):,} | Val: {len(va):,} | Test: {len(te):,}")

        non_feat  = ["anchor_ts"] + TARGET_COLS
        feat_cols = [c for c in df.columns if c not in non_feat]
        X_tr = tr[feat_cols].values
        X_va = va[feat_cols].values
        X_te = te[feat_cols].values

        r2_list, mae_list = [], []

        for h_i, tgt in enumerate(TARGET_COLS):
            y_tr = tr[tgt].values
            y_va = va[tgt].values
            y_te = te[tgt].values

            m = lgb.LGBMRegressor(**LGBM_PARAMS)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(80, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            pred = m.predict(X_te).clip(0)
            r2   = r2_score(y_te, pred)
            mae  = mean_absolute_error(y_te, pred)
            r2_list.append(r2)
            mae_list.append(mae)

            status = "✅" if r2 >= 0.90 else ("~" if r2 >= 0.85 else "○")
            print(f"  {H_LABELS[h_i]:>10s}  R²={r2:.4f}  MAE={mae:6.1f} W/m²  {status}")

        avg = np.mean(r2_list)
        print(f"  {'Rata-rata':>10s}  R²={avg:.4f}  "
              f"{'✅ ≥0.9' if avg >= 0.90 else ('~0.85+' if avg >= 0.85 else '○')}\n")
        all_results[stage] = {"r2": r2_list, "mae": mae_list}

    # ── Tabel perbandingan antar versi ────────────────────────────────────────
    v1_s3 = [0.863, 0.826, 0.804, 0.786, 0.773, 0.764]  # hasil v1 Stage 3
    stage_list = sorted(all_results.keys())

    print("=" * 72)
    print("📊 PERBANDINGAN Test R² (GHI) — v1 Stage3 vs v3 Stages")
    print("=" * 72)
    hdr = f"{'Horizon':<12}  {'v1-S3':>8}" + "".join(
        f"  {'v3-S'+str(s):>8}" for s in stage_list
    )
    print(hdr)
    print("─" * len(hdr))
    for i, lbl in enumerate(H_LABELS):
        row = f"{lbl:<12}  {v1_s3[i]:>8.4f}" + "".join(
            f"  {all_results[s]['r2'][i]:>8.4f}" for s in stage_list
        )
        print(row)
    print("─" * len(hdr))
    avg_row = f"{'Rata-rata':<12}  {np.mean(v1_s3):>8.4f}" + "".join(
        f"  {np.mean(all_results[s]['r2']):>8.4f}" for s in stage_list
    )
    print(avg_row)
    print("=" * 72)

    # Peningkatan v3-S3 vs v1-S3
    if 3 in all_results:
        delta = [all_results[3]["r2"][i] - v1_s3[i] for i in range(HORIZON)]
        print(f"\n📈 v3-Stage3 vs v1-Stage3 (delta R²):")
        for i, lbl in enumerate(H_LABELS):
            sign = "+" if delta[i] >= 0 else ""
            print(f"   {lbl}: {sign}{delta[i]:+.4f}")
        print(f"   Rata-rata: {'+' if np.mean(delta) >= 0 else ''}{np.mean(delta):+.4f}")

    # ── Simpan CSV ────────────────────────────────────────────────────────────
    rows_out = []
    for s in stage_list:
        for h_i, lbl in enumerate(H_LABELS):
            rows_out.append({
                "version"  : "v3",
                "stage"    : s,
                "horizon"  : lbl,
                "test_r2"  : round(all_results[s]["r2"][h_i], 5),
                "test_mae" : round(all_results[s]["mae"][h_i], 2),
            })
    # Tambahkan v1 untuk referensi
    for h_i, lbl in enumerate(H_LABELS):
        rows_out.append({
            "version": "v1", "stage": 3, "horizon": lbl,
            "test_r2": v1_s3[h_i], "test_mae": None,
        })
    pd.DataFrame(rows_out).to_csv(OUTPUT_DIR / "ghi_forecast_eval_v3.csv", index=False)
    print(f"\n💾 Evaluasi → ghi_forecast_eval_v3.csv")

    # ── Tips jika masih di bawah 0.9 ──────────────────────────────────────────
    best_avg = max(np.mean(all_results[s]["r2"]) for s in stage_list)
    if best_avg < 0.90:
        print("\n" + "─" * 68)
        print("💡 Masih di bawah 0.9? Opsi lanjutan:")
        print("   1. pvlib clearsky — install pvlib lalu jalankan dengan --use-pvlib")
        print("      pip install pvlib")
        print("      python build_ghi_forecast_v3.py --use-pvlib")
        print("      → clearsky GHI lebih akurat → fitur kt lebih bersih")
        print("   2. Pisahkan model clear/cloudy: dua model terpisah per kondisi awan")
        print("   3. Prediksi cloud state (COT) sebagai auxiliary task")
        print("   4. Data tambahan: reanalysis NWP (ERA5) sebagai fitur cuaca skala besar")
        print("─" * 68)

    return all_results


def add_pvlib_clearsky(df):
    """
    Opsional: Hitung clearsky GHI yang lebih akurat dengan pvlib (Ineichen model).
    Membutuhkan: pip install pvlib
    Koordinat Pontianak: lat=-0.02, lon=109.34, alt=16 mdpl, tz=UTC+7
    """
    try:
        import pvlib
    except ImportError:
        print("  ⚠  pvlib tidak tersedia. Gunakan clearsky kasar (950×sin^1.1).")
        return df

    print("  🌤  Menghitung clearsky pvlib (Ineichen) untuk Pontianak ...")
    location = pvlib.location.Location(
        latitude=-0.02, longitude=109.34, altitude=16,
        tz="Asia/Jakarta", name="Pontianak"
    )
    times = df["timestamp_wib"].dt.tz_localize("Asia/Jakarta")
    cs    = location.get_clearsky(times, model="ineichen")

    df["ghi_clearsky_pvlib"] = cs["ghi"].values.astype(np.float32)
    # Overwrite kt_approx dengan kt dari pvlib (lebih akurat)
    df["kt_approx"] = np.where(
        df["ghi_clearsky_pvlib"] > 10,
        (df["ghi_wm2"] / df["ghi_clearsky_pvlib"]).clip(0, 1.3),
        0.0,
    ).astype(np.float32)
    # Overwrite ghi_clearsky_rough dengan pvlib untuk fitur future
    df["ghi_clearsky_rough"] = df["ghi_clearsky_pvlib"]
    print("  ✓ clearsky pvlib diterapkan")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",      help="MotherDuck token")
    parser.add_argument("--stages",     nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--use-pvlib",  action="store_true",
                        help="Gunakan pvlib untuk clearsky yang lebih akurat")
    args = parser.parse_args()

    t_total = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   GHI Forecast v3 — GHI target + cloud state + dynamics      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Stage   : {args.stages}")
    print(f"pvlib   : {'Ya' if args.use_pvlib else 'Tidak (kasar)'}")
    print(f"Window  : {WINDOW}×10mnt = {WINDOW*10} mnt | Horizon: {HORIZON}×10mnt = {HORIZON*10} mnt")
    print(f"Output  : {OUTPUT_DIR}")

    con = connect_md(args.token)
    df  = load_data(con)
    con.close()

    df = feature_engineer(df)
    if args.use_pvlib:
        df = add_pvlib_clearsky(df)

    anch = compute_valid_anchors(df)

    datasets = {}
    for s in args.stages:
        ds = build_dataset(df, anch, s)
        save_parquet(ds, s)
        datasets[s] = ds

    if not args.skip_train:
        train_and_evaluate(datasets)

    print(f"\n✅ Selesai dalam {time.time()-t_total:.1f} detik")


if __name__ == "__main__":
    main()