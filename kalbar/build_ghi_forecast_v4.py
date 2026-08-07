"""
build_ghi_forecast_v4.py  --  GHI Forecast Dataset Builder v4
==============================================================
TAMBAHAN vs v3:
  Stage 4 = Stage 3 + Fitur Aerosol (ARP Himawari + AERONET PWV)

Sumber data aerosol:
  - kalbar.main.arp_pontianak
      AOT_mean, AE_mean, fine_mode_aot_proxy, aerosol_retrieval_valid
      Resolusi: 10 menit, coverage 99.7%, valid AOT ~55% (sisa berawan)
  - kalbar.main.aeronetpontianak
      Precipitable_Water_cm (PWV) -- hanya tersedia saat langit cerah,
      diagregasi ke harian lalu di-forward-fill

Fitur baru (Stage 4 vs Stage 3):
  aot_mean        -- AOT dari Himawari (0 jika tidak valid / berawan)
  aot_valid_bin   -- 1 jika AOT berhasil diukur, 0 jika tidak
  ae_mean         -- Angstrom exponent (penanda tipe aerosol: asap vs debu)
  fine_mode_aot   -- fine-mode AOT proxy (tinggi saat kebakaran gambut)
  pwv_daily       -- precipitable water (cm) dari AERONET, forward-filled harian
  kt_aerosol      -- clearness index lebih akurat: GHI / clearsky_v2
                     clearsky_v2 = 950 x sin(alt)^1.1 x pwv_factor x aot_factor
  delta_aot_1     -- perubahan AOT 10 menit (dinamika haze)
  aot_std6        -- volatilitas AOT 60 menit

Cara pakai:
  python build_ghi_forecast_v4.py
  python build_ghi_forecast_v4.py --stages 4
  python build_ghi_forecast_v4.py --skip-train
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

# ---- Konfigurasi ------------------------------------------------------------
OUTPUT_DIR  = Path(__file__).parent
WINDOW      = 18
HORIZON     = 6
MIN_ALT     = 5.0
MAX_GAP_MIN = 10
PWV_DEFAULT = 5.0   # cm -- nilai tipikal Pontianak jika AERONET tidak tersedia

# ---- Definisi fitur ---------------------------------------------------------
STAGE1_COLS = [
    "ghi_wm2", "dni_wm2", "dhi_wm2",
    "sun_altitude", "kt_approx",
]
CLOUD_STATE_COLS = [
    "cloud_present_bin", "cloud_optical_thick",
    "cloud_top_height_m", "cloud_top_temp_k", "cloud_eff_radius_um",
]
CLOUD_DYN_COLS = [
    "delta_ghi_1", "delta_ghi_3", "delta_ghi_6",
    "ghi_std6", "delta_cot_1", "cot_std6",
]
AEROSOL_COLS = [
    "aot_mean",        # AOT dari Himawari (0 = tidak valid/berawan)
    "aot_valid_bin",   # 1 = AOT terukur, 0 = berawan/tidak tersedia
    "ae_mean",         # Angstrom exponent
    "fine_mode_aot",   # fine-mode AOT proxy (asap kebakaran)
    "pwv_daily",       # precipitable water harian dari AERONET (cm)
    "kt_aerosol",      # clearness index dengan koreksi aerosol
    "delta_aot_1",     # perubahan AOT 10 menit
    "aot_std6",        # volatilitas AOT 60 menit
]

STAGE2_COLS = STAGE1_COLS + CLOUD_STATE_COLS
STAGE3_COLS = STAGE2_COLS + CLOUD_DYN_COLS
STAGE4_COLS = STAGE3_COLS + AEROSOL_COLS

STAGE_COLS = {1: STAGE1_COLS, 2: STAGE2_COLS, 3: STAGE3_COLS, 4: STAGE4_COLS}


# ---- Koneksi MotherDuck -----------------------------------------------------
def connect_md(token=None):
    tok = token or os.environ.get("MOTHERDUCK_TOKEN", "")
    url = f"md:kalbar?motherduck_token={tok}" if tok else "md:kalbar"
    print("Menghubungkan ke MotherDuck ...")
    con = duckdb.connect(url)
    print("  Terhubung")
    return con


# ---- Load data utama + aerosol ----------------------------------------------
def load_data(con):
    print("\nMemuat data radiasi ...")
    t0 = time.time()

    df = con.execute("""
        SELECT
            s.timestamp_wib,
            s.sun_altitude,
            COALESCE(s.ghi_wm2, 0)            AS ghi_wm2,
            COALESCE(s.dni_wm2, 0)            AS dni_wm2,
            COALESCE(s.dhi_wm2, 0)            AS dhi_wm2,
            CASE WHEN s.cloud_present = TRUE  THEN 1.0
                 WHEN s.cloud_present = FALSE THEN 0.0
                 ELSE 0.0 END                  AS cloud_present_bin,
            COALESCE(s.cloud_optical_thick, 0) AS cloud_optical_thick,
            COALESCE(s.cloud_top_height_m, 0)  AS cloud_top_height_m,
            COALESCE(s.cloud_top_temp_k, 0)    AS cloud_top_temp_k,
            COALESCE(s.cloud_eff_radius_um, 0) AS cloud_eff_radius_um,
            -- ARP aerosol (10-min grid, 99.7% match rate)
            COALESCE(a.aerosol_retrieval_valid::INT, 0) AS aot_valid_bin,
            COALESCE(CASE WHEN a.aerosol_retrieval_valid THEN a.AOT_mean  END, 0.0) AS aot_mean,
            COALESCE(CASE WHEN a.aerosol_retrieval_valid THEN a.AE_mean   END, 0.0) AS ae_mean,
            COALESCE(CASE WHEN a.aerosol_retrieval_valid THEN a.fine_mode_aot_proxy END, 0.0) AS fine_mode_aot
        FROM kalbar.main.solar_radiation_valid s
        LEFT JOIN kalbar.main.arp_pontianak a
          ON s.timestamp_wib = a.timestamp_wib
        ORDER BY s.timestamp_wib
    """).df()

    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"])
    print(f"  {len(df):,} baris [{time.time()-t0:.1f}s] | "
          f"{df.timestamp_wib.min().date()} -- {df.timestamp_wib.max().date()}")
    print(f"  AOT valid: {df['aot_valid_bin'].mean()*100:.1f}% baris")
    return df


def load_aeronet_pwv(con):
    """Harian PWV dari AERONET, di-forward-fill untuk mengisi hari tanpa observasi."""
    print("Memuat AERONET Precipitable Water ...")
    pwv = con.execute("""
        SELECT date_wib AS date,
               AVG(Precipitable_Water_cm) AS pwv_daily
        FROM kalbar.main.aeronetpontianak
        WHERE Precipitable_Water_cm IS NOT NULL
          AND Precipitable_Water_cm > 0
        GROUP BY date_wib
        ORDER BY date_wib
    """).df()
    pwv["date"] = pd.to_datetime(pwv["date"])
    print(f"  {len(pwv)} hari dengan data PWV "
          f"({pwv['date'].min().date()} -- {pwv['date'].max().date()})")
    print(f"  PWV mean={pwv['pwv_daily'].mean():.2f} cm  "
          f"std={pwv['pwv_daily'].std():.2f} cm")
    return pwv


# ---- Feature engineering ----------------------------------------------------
def feature_engineer(df, pwv_df):
    print("\nFeature engineering ...")

    # ---- Clearsky kasar (sama seperti v3) -----------------------------------
    sin_alt = np.sin(np.radians(df["sun_altitude"].clip(0))).clip(0)
    df["ghi_clearsky_rough"] = (950.0 * sin_alt ** 1.1).clip(1.0)
    df["kt_approx"] = np.where(
        df["sun_altitude"] > 0,
        (df["ghi_wm2"] / df["ghi_clearsky_rough"]).clip(0, 1.3),
        0.0,
    ).astype(np.float32)

    # ---- Dinamika GHI (sama seperti v3) -------------------------------------
    g = df["ghi_wm2"].values.astype(np.float32)
    df["delta_ghi_1"] = np.concatenate([[0],     g[1:]  - g[:-1]]).astype(np.float32)
    df["delta_ghi_3"] = np.concatenate([[0,0,0], g[3:]  - g[:-3]]).astype(np.float32)
    df["delta_ghi_6"] = np.concatenate([[0]*6,   g[6:]  - g[:-6]]).astype(np.float32)
    df["ghi_std6"]    = (pd.Series(g).rolling(6, min_periods=1).std()
                         .fillna(0).values.astype(np.float32))

    # ---- Dinamika COT (sama seperti v3) -------------------------------------
    c = df["cloud_optical_thick"].values.astype(np.float32)
    df["delta_cot_1"] = np.concatenate([[0], c[1:] - c[:-1]]).astype(np.float32)
    df["cot_std6"]    = (pd.Series(c).rolling(6, min_periods=1).std()
                         .fillna(0).values.astype(np.float32))

    # ---- PWV harian dari AERONET (forward-fill + backward-fill) -------------
    df["date"] = df["timestamp_wib"].dt.date.astype(str)
    pwv_df["date_str"] = pwv_df["date"].dt.strftime("%Y-%m-%d")
    pwv_map = pwv_df.set_index("date_str")["pwv_daily"].to_dict()

    # Buat full daily index dan forward-fill
    all_dates = pd.date_range(
        df["timestamp_wib"].min().date(),
        df["timestamp_wib"].max().date(),
        freq="D"
    )
    pwv_series = pd.Series(
        {d.strftime("%Y-%m-%d"): pwv_map.get(d.strftime("%Y-%m-%d"), np.nan)
         for d in all_dates}
    ).ffill().bfill().fillna(PWV_DEFAULT)

    df["pwv_daily"] = df["date"].map(pwv_series).astype(np.float32)
    df.drop(columns=["date"], inplace=True)

    # ---- Clearsky v2 dengan koreksi aerosol + PWV ---------------------------
    # pwv_factor = (PWV_DEFAULT / pwv)^0.15 -- water vapor absorption correction
    # aot_factor = exp(-aot * 1.5) -- aerosol extinction (AM=1.5 approximate mean)
    pwv_arr = df["pwv_daily"].values
    aot_arr = df["aot_mean"].values
    pwv_factor = (PWV_DEFAULT / pwv_arr.clip(0.5)) ** 0.15
    aot_factor = np.exp(-aot_arr * 1.5)
    ghi_clearsky_v2 = df["ghi_clearsky_rough"].values * pwv_factor * aot_factor
    ghi_clearsky_v2 = np.where(
        df["sun_altitude"].values > 0,
        ghi_clearsky_v2.clip(1.0),
        1.0
    )
    df["kt_aerosol"] = np.where(
        (df["sun_altitude"].values > 0) & (df["aot_valid_bin"].values > 0),
        (df["ghi_wm2"].values / ghi_clearsky_v2).clip(0, 1.3),
        df["kt_approx"].values  # fallback ke kt_approx jika AOT tidak valid
    ).astype(np.float32)

    # ---- Dinamika AOT -------------------------------------------------------
    a = df["aot_mean"].values.astype(np.float32)
    df["delta_aot_1"] = np.concatenate([[0], a[1:] - a[:-1]]).astype(np.float32)
    df["aot_std6"]    = (pd.Series(a).rolling(6, min_periods=1).std()
                         .fillna(0).values.astype(np.float32))

    print("  kt_approx, clearsky_rough, delta_ghi, ghi_std6, delta_cot, cot_std6")
    print(f"  PWV: mean={df['pwv_daily'].mean():.2f} cm  "
          f"(AERONET coverage={100*pwv_df['pwv_daily'].notna().sum()/len(all_dates):.0f}%)")
    print(f"  AOT: mean={df['aot_mean'].mean():.3f}  "
          f"kt_aerosol mean={df['kt_aerosol'].mean():.3f}")
    return df


# ---- Valid anchors ----------------------------------------------------------
def compute_valid_anchors(df):
    print("\nHitung valid anchor positions ...")
    t0   = time.time()
    n    = len(df)
    alt  = df["sun_altitude"].values
    ts   = df["timestamp_wib"].values

    diffs  = pd.to_datetime(ts).to_series().diff().dt.total_seconds().div(60).fillna(0)
    breaks = (diffs > MAX_GAP_MIN).astype(int).values
    cum_b  = np.cumsum(breaks)

    valid = []
    for i in range(WINDOW - 1, n - HORIZON):
        if cum_b[i + HORIZON] - cum_b[i - WINDOW + 1] > 0:
            continue
        if alt[i] <= MIN_ALT or alt[i + HORIZON] <= MIN_ALT:
            continue
        valid.append(i)

    print(f"  {len(valid):,} valid anchors [{time.time()-t0:.1f}s]")
    return np.array(valid, dtype=np.int64)


# ---- Build lag dataset ------------------------------------------------------
def build_dataset(df, valid_anchors, stage):
    features = STAGE_COLS[stage]
    n_feats  = len(features)
    n_samp   = len(valid_anchors)
    print(f"\nStage {stage}: {n_feats} fitur base x {WINDOW} lag ...")
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
    ms = np.sin(2 * np.pi * ts_pd.month.values / 12).astype(np.float32)
    mc = np.cos(2 * np.pi * ts_pd.month.values / 12).astype(np.float32)
    ds = np.sin(2 * np.pi * doy_arr / 365).astype(np.float32)
    dc = np.cos(2 * np.pi * doy_arr / 365).astype(np.float32)

    n_cols = n_feats * WINDOW + 6 + HORIZON * 2 + HORIZON
    out    = np.empty((n_samp, n_cols), dtype=np.float32)

    for j, i in enumerate(valid_anchors):
        ptr = 0
        for lag in range(WINDOW):
            out[j, ptr:ptr + n_feats] = feat_np[i - lag]
            ptr += n_feats
        out[j, ptr:ptr+6] = [hs[i], hc[i], ms[i], mc[i], ds[i], dc[i]]
        ptr += 6
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
    col_names += ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    for h in range(1, HORIZON + 1):
        col_names += [f"sun_alt_t{h}", f"ghi_cs_t{h}"]
    for h in range(1, HORIZON + 1):
        col_names.append(f"ghi_t{h}")

    result = pd.DataFrame(out, columns=col_names)
    result.insert(0, "anchor_ts", ts_np[valid_anchors])

    mem = result.memory_usage(deep=True).sum() / (1024**2)
    print(f"  {len(result):,} sampel x {len(result.columns)} kolom "
          f"({mem:.0f} MB) [{time.time()-t0:.1f}s]")
    return result


# ---- Save parquet -----------------------------------------------------------
def save_parquet(df, stage):
    path = OUTPUT_DIR / f"ghi_forecast_v4_stage{stage}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Disimpan: {path.name} ({path.stat().st_size/(1024**2):.1f} MB)")


# ---- Train & evaluate -------------------------------------------------------
def train_and_evaluate(datasets):
    try:
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("\n  pip install lightgbm scikit-learn")
        return

    TARGET_COLS = [f"ghi_t{h}" for h in range(1, HORIZON + 1)]
    H_LABELS    = [f"t+{h*10}min" for h in range(1, HORIZON + 1)]

    LGBM_PARAMS = dict(
        n_estimators      = 2500,
        learning_rate     = 0.04,
        num_leaves        = 127,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        n_jobs            = -1,
        verbose           = -1,
    )

    print("\n" + "=" * 68)
    print("TRAINING -- Target: GHI (W/m2)  |  v4 dengan aerosol")
    print("=" * 68)
    print("Split: 2022-2023 train | 2024 val | 2025 test\n")

    v3_s3 = [0.8625, 0.8261, 0.8027, 0.7861, 0.7733, 0.7648]  # baseline v3

    all_results = {}
    for stage, df in sorted(datasets.items()):
        print(f"-- Stage {stage} ({len(STAGE_COLS[stage])} fitur base) --")
        year = pd.to_datetime(df["anchor_ts"]).dt.year
        tr   = df[year.isin([2022, 2023])]
        va   = df[year == 2024]
        te   = df[year == 2025]
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
                eval_set  = [(X_va, y_va)],
                callbacks = [
                    lgb.early_stopping(120, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            pred = m.predict(X_te).clip(0)
            r2   = r2_score(y_te, pred)
            mae  = mean_absolute_error(y_te, pred)
            r2_list.append(r2)
            mae_list.append(mae)

            delta = r2 - v3_s3[h_i]
            marker = "OK" if r2 >= 0.90 else ("~" if r2 >= 0.85 else "o")
            print(f"  {H_LABELS[h_i]:>10s}  R2={r2:.4f}  MAE={mae:6.1f} W/m2  "
                  f"vs v3: {'+' if delta>=0 else ''}{delta:+.4f}  {marker}")

        avg  = np.mean(r2_list)
        davg = avg - np.mean(v3_s3)
        print(f"  {'Rata-rata':>10s}  R2={avg:.4f}  "
              f"vs v3-S3 baseline: {'+' if davg>=0 else ''}{davg:+.4f}\n")
        all_results[stage] = {"r2": r2_list, "mae": mae_list}

    # ---- Ringkasan -----------------------------------------------------------
    print("=" * 72)
    print("PERBANDINGAN Test R2: v3-Stage3 baseline vs v4 Stages")
    print("=" * 72)
    stage_list = sorted(all_results.keys())
    hdr = f"{'Horizon':<12}  {'v3-S3':>8}" + "".join(
        f"  {'v4-S'+str(s):>8}" for s in stage_list)
    print(hdr)
    print("-" * len(hdr))
    for i, lbl in enumerate(H_LABELS):
        row = f"{lbl:<12}  {v3_s3[i]:>8.4f}" + "".join(
            f"  {all_results[s]['r2'][i]:>8.4f}" for s in stage_list)
        print(row)
    print("-" * len(hdr))
    avg_row = f"{'Rata-rata':<12}  {np.mean(v3_s3):>8.4f}" + "".join(
        f"  {np.mean(all_results[s]['r2']):>8.4f}" for s in stage_list)
    print(avg_row)
    print("=" * 72)

    # ---- Simpan CSV evaluasi -------------------------------------------------
    rows_out = []
    for s in stage_list:
        for h_i, lbl in enumerate(H_LABELS):
            rows_out.append({
                "version": "v4", "stage": s, "horizon": lbl,
                "test_r2": round(all_results[s]["r2"][h_i], 5),
                "test_mae": round(all_results[s]["mae"][h_i], 2),
            })
    for h_i, lbl in enumerate(H_LABELS):
        rows_out.append({
            "version": "v3", "stage": 3, "horizon": lbl,
            "test_r2": v3_s3[h_i], "test_mae": None,
        })
    out_csv = OUTPUT_DIR / "ghi_forecast_eval_v4.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\nEvaluasi disimpan: {out_csv.name}")

    # ---- Save parquet untuk stage 4 -----------------------------------------
    best_avg = max(np.mean(all_results[s]["r2"]) for s in stage_list)
    if best_avg >= 0.90:
        print("\nTarget R2=0.90 TERCAPAI!")
    else:
        print(f"\nJarak ke target R2=0.90: {0.9 - best_avg:.4f}")

    return all_results


# ---- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",      help="MotherDuck token")
    parser.add_argument("--stages",     nargs="+", type=int, default=[3, 4])
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 62)
    print("  GHI Forecast v4 -- Aerosol + Cloud + Radiation features")
    print("=" * 62)
    print(f"  Stages  : {args.stages}")
    print(f"  Window  : {WINDOW}x10mnt = {WINDOW*10} mnt")
    print(f"  Horizon : {HORIZON}x10mnt = {HORIZON*10} mnt")
    print(f"  Output  : {OUTPUT_DIR}")

    con    = connect_md(args.token)
    df     = load_data(con)
    pwv_df = load_aeronet_pwv(con)
    con.close()

    df = feature_engineer(df, pwv_df)

    valid_anchors = compute_valid_anchors(df)

    datasets = {}
    for stage in args.stages:
        print(f"\n{'='*40}")
        ds = build_dataset(df, valid_anchors, stage)
        save_parquet(ds, stage)
        datasets[stage] = ds

    if not args.skip_train:
        train_and_evaluate(datasets)

    print(f"\nSelesai. Total waktu: {(time.time()-t_total)/60:.1f} menit")


if __name__ == "__main__":
    main()
