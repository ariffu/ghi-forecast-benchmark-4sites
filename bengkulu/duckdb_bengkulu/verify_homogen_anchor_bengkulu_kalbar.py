#!/usr/bin/env python3
"""
Verifikasi anchor homogen §2.3 untuk Bengkulu dan Kalbar.
Terkoreksi dari verify_homogen_anchor_bengkulu_kalbar.py (Obsidian vault):
  - Bengkulu GHI col: asrs_ghi_w_m2 (bukan ghi_now)
  - Kalbar ts col: timestamp_wib (bukan ts_wib), GHI col: ghi_final
  - Elevasi: vektorisasi NumPy (bukan pandas apply, jauh lebih cepat)
  - Bengkulu sudah diaudit sebelumnya (audit_tabel1_bengkulu.py) — dilaporkan ulang
"""

import duckdb, numpy as np, pandas as pd
from pathlib import Path

# ── koordinat ─────────────────────────────────────────────────────────────────
SITES = {
    "bengkulu": {
        "db_path": "C:/Users/ariff/DuckDB_bengkulu/bengkulu.duckdb",
        "attach":  True,
        "schema":  "bengkulu_sch",
        "table":   "bengkulu_master_10min_quality_final",
        "ts_col":  "ts_wib",
        "ghi_col": "asrs_ghi_w_m2",
        "lat": -3.8607, "lon": 102.3381,
        "ref_anchor_paper": 109_196,   # referensi Tabel 1 paper / §2.3
        "ref_pipeline":     105_051,   # R1/R8 yang sudah dipakai di Results
        "ref_train": 59_114, "ref_val": 23_226, "ref_test": 22_711,
        "ref_r2": 0.792,
    },
    "kalbar": {
        "db_path": "C:/Users/ariff/DuckDB_kalbar/kalbar_local.db",
        "attach":  False,
        "schema":  None,
        "table":   "training_ghi_1h_direct",
        "ts_col":  "timestamp_wib",
        "ghi_col": "ghi_final",
        "lat": -0.0356, "lon": 109.3384,
        "ref_anchor_paper": 90_579,    # angka §2.3 yang diklaim di draft paper
        "ref_pipeline":     81_851,    # R1/R8 pipeline yang sudah dipakai
        "ref_train": 39_759, "ref_val": 20_706, "ref_test": 21_386,
        "ref_r2": 0.7217,
    },
}

MERIDIAN = 105.0  # WIB

def astro_elev(ts_series: pd.Series, lat: float, lon: float) -> np.ndarray:
    ts  = pd.DatetimeIndex(ts_series)
    doy = ts.dayofyear.values.astype(float)
    h   = ts.hour.values + ts.minute.values / 60.0
    decl = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
    ha   = (h + 4 * (lon - MERIDIAN) / 60 - 12) * 15
    sin_e = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
             + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl))
             * np.cos(np.deg2rad(ha)))
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))


def run_site(name: str, cfg: dict):
    print(f"\n{'='*65}")
    print(f"  {name.upper()}")
    print(f"{'='*65}")

    # ── koneksi ──────────────────────────────────────────────────────────────
    if cfg["attach"]:
        con = duckdb.connect(":memory:")
        con.execute(f"ATTACH '{cfg['db_path']}' AS bdb (READ_ONLY)")
        tbl = f"bdb.{cfg['schema']}.{cfg['table']}"
    else:
        con = duckdb.connect(cfg["db_path"], read_only=True)
        tbl = cfg["table"]

    # ── muat mentah ──────────────────────────────────────────────────────────
    ts_c  = cfg["ts_col"]
    ghi_c = cfg["ghi_col"]
    q = (f"SELECT {ts_c} AS ts, {ghi_c}::DOUBLE AS ghi FROM {tbl}"
         f" WHERE YEAR({ts_c}) BETWEEN 2021 AND 2025 ORDER BY ts")
    df = con.execute(q).fetchdf()
    con.close()

    df["ts"]  = pd.to_datetime(df["ts"])
    df["ghi"] = pd.to_numeric(df["ghi"], errors="coerce")
    print(f"  Raw baris (2021–2025): {len(df):,}")

    # ── elevasi (vektorisasi) ─────────────────────────────────────────────────
    lat, lon = cfg["lat"], cfg["lon"]
    df["elev_a"]   = astro_elev(df["ts"], lat, lon)
    df["elev_t60"] = astro_elev(df["ts"] + pd.Timedelta(minutes=60), lat, lon)
    df["ghi_t60"]  = df["ghi"].shift(-6)   # GHI di t+60 (6 langkah ke depan)

    # ── filter §2.3 ──────────────────────────────────────────────────────────
    f_elev = (df["elev_a"] > 5) & (df["elev_t60"] > 5)
    f_ghi  = df["ghi"].between(0, 1400)
    f_ghi60= df["ghi_t60"].between(0, 1400)   # §2.3: GHI t+60 juga valid

    # kontinuitas 18 langkah berturutan (toleransi ±30 detik)
    diff_sec = df["ts"].diff().dt.total_seconds().fillna(0)
    step_ok  = ((diff_sec >= 570) & (diff_sec <= 630)).astype(int)
    f_cont   = (step_ok.rolling(window=18, min_periods=18).sum().fillna(0) == 18)

    mask = f_elev & f_ghi & f_ghi60 & f_cont
    anchor = df[mask].copy()
    n_total = len(anchor)

    # ── split ────────────────────────────────────────────────────────────────
    yr = anchor["ts"].dt.year
    tr = (anchor["ts"] < pd.Timestamp("2024-01-01")).sum()
    va = ((anchor["ts"] >= pd.Timestamp("2024-01-01")) & (anchor["ts"] < pd.Timestamp("2025-01-01"))).sum()
    te = (anchor["ts"] >= pd.Timestamp("2025-01-01")).sum()
    per_yr = yr.value_counts().sort_index().to_dict()

    # ── cetak ─────────────────────────────────────────────────────────────────
    print(f"\n  Anchor §2.3 (hitung ulang skrip ini):  {n_total:,}")
    print(f"  Referensi Tabel 1 paper (§2.3):        {cfg['ref_anchor_paper']:,}")
    dev_anchor = 100*(n_total - cfg["ref_anchor_paper"]) / cfg["ref_anchor_paper"]
    print(f"  Deviasi vs referensi:                  {dev_anchor:+.2f}%")
    status_a = "✓ VALID (<1%)" if abs(dev_anchor) <= 1 else "⚠ DEVIASI >1%"
    print(f"  Status:                                {status_a}")

    print(f"\n  Split (ts-only §2.3):  train={tr:,}  val={va:,}  test={te:,}")
    print(f"  Per tahun: {per_yr}")

    print(f"\n  Pipeline R1/R8 (dipakai di Results): {cfg['ref_pipeline']:,}")
    print(f"    train={cfg['ref_train']:,}  val={cfg['ref_val']:,}  test={cfg['ref_test']:,}")
    dev_pipe = 100*(n_total - cfg["ref_pipeline"]) / cfg["ref_pipeline"]
    dev_test = 100*(te - cfg["ref_test"]) / cfg["ref_test"]
    print(f"  Selisih anchor §2.3 vs pipeline: {dev_pipe:+.2f}%")
    print(f"  Selisih test §2.3 vs pipeline:   {dev_test:+.2f}%")

    # ── apakah perlu retrain? ──────────────────────────────────────────────────
    if abs(dev_test) <= 2.0:
        verdict = ("→ Selisih test <2% — kemungkinan R² tidak berubah secara signifikan "
                   "(seperti Banten dR²=-0.0004). Update Tabel 1 saja, tidak perlu retrain.")
    else:
        verdict = (f"→ Selisih test {dev_test:+.1f}% — PERLU retrain di set §2.3 "
                   "dan bandingkan R² sebelum update paper.")
    print(f"\n  VERDICT: {verdict}")

    return {"n_total": n_total, "train": tr, "val": va, "test": te,
            "dev_anchor_pct": dev_anchor, "dev_test_pct": dev_test}


def main():
    results = {}
    for site, cfg in SITES.items():
        try:
            results[site] = run_site(site, cfg)
        except Exception as e:
            print(f"\n[GAGAL] {site}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*65}")
    print("  RINGKASAN")
    print(f"{'='*65}")
    for site, r in results.items():
        print(f"\n  {site.upper()}")
        print(f"    §2.3 total: {r['n_total']:,}  (ref: {SITES[site]['ref_anchor_paper']:,},"
              f" dev {r['dev_anchor_pct']:+.2f}%)")
        print(f"    train/val/test: {r['train']:,}/{r['val']:,}/{r['test']:,}")
        print(f"    Selisih test vs pipeline: {r['dev_test_pct']:+.2f}%")
        print(f"    Perlu retrain: {'YA' if abs(r['dev_test_pct']) > 2 else 'TIDAK'}")

    print("\n  Bengkulu sudah diaudit lebih lengkap di audit_tabel1_bengkulu.py:")
    print("    §2.3 ts-only = 109,717  |  strict = 109,669  |  ref = 109,196 (+0.48%)")

if __name__ == "__main__":
    main()
