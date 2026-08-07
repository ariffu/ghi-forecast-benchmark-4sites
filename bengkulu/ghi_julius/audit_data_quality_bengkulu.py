#!/usr/bin/env python3
"""
Audit Konsistensi Fisis + Closure + Sentinel -- BENGKULU (Prioritas A2, 2026-07-25)
Metodologi identik dengan audit_data_quality_kalbar.py -- lihat docstring di
sana untuk definisi lengkap tiap metrik. Sumber data: rollback table
(ghi_forecast_1h_train_3h_rollback_2021_2025), yaitu tabel yang benar-benar
dipakai untuk training model (bukan tabel gabungan mentah asrs_bengkulu_combined
yang menghasilkan angka closure 12.9% di audit sebelumnya -- di sini closure
dihitung ULANG di titik yang lebih hilir, setelah join+filter, supaya
sebanding dengan lokasi lain yang juga diaudit di titik yang sama).

Run:
    python audit_data_quality_bengkulu.py
"""
import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOCAL_DB_PATH = Path("/sessions/jolly-gifted-ritchie/mnt/DuckDB_bengkulu/bengkulu.duckdb")
OUTPUT_DIR = Path("outputs_audit_konsistensi_fisis")
OUTPUT_DIR.mkdir(exist_ok=True)
LOCATION = "Bengkulu"
CLOSURE_THRESHOLD = 0.15
STUCK_MIN_RUN = 6


def load_base():
    con = duckdb.connect(database=":memory:")
    con.execute(f"ATTACH '{LOCAL_DB_PATH.as_posix()}' AS bdb (READ_ONLY)")
    df = con.execute("""
        SELECT ts_wib AS ts, ghi_now AS ghi, dni_now AS dni, dhi_now AS dhi,
               solar_elev_deg AS elev, clp_cot, clp_cloud_present AS clp_cloud_present_raw
        FROM bdb.bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025
        ORDER BY ts_wib
    """).fetchdf()
    con.close()
    return df


def sentinel_and_stuck(df_day, cols):
    """NOTE (2026-07-25 fix): dihitung HANYA di baris siang (elev>5) dan run
    bernilai TEPAT NOL di-exclude, supaya tidak mencampur 'sensor macet
    genuine' dengan 'GHI/DNI/DHI=0 yang valid secara fisis waktu malam/senja'."""
    rows = []
    for c in cols:
        if c not in df_day.columns:
            continue
        s = df_day[c]
        n = s.notna().sum()
        sentinel_n = int(s.isin([9999, -9999, 999, -999, 99999, -99999]).sum())
        vals = s.values
        stuck_rows = 0
        run_len = 1
        for i in range(1, len(vals)):
            a, b = vals[i - 1], vals[i]
            if pd.notna(a) and pd.notna(b) and a == b:
                run_len += 1
            else:
                if run_len >= STUCK_MIN_RUN and pd.notna(a) and a != 0:
                    stuck_rows += run_len
                run_len = 1
        if run_len >= STUCK_MIN_RUN and pd.notna(vals[-1]) and vals[-1] != 0:
            stuck_rows += run_len
        rows.append({
            "column": c, "n_nonnull": int(n),
            "sentinel_pct": round(100.0 * sentinel_n / max(n, 1), 3),
            "sentinel_n": sentinel_n,
            "stuck_run_pct_daytime_nonzero": round(100.0 * stuck_rows / max(n, 1), 3),
            "stuck_run_n_daytime_nonzero": stuck_rows,
        })
    return pd.DataFrame(rows)


def main():
    df = load_base()
    print(f"Rows loaded: {len(df):,}")
    df["kt"] = df["ghi"] / np.maximum(1100.0 * np.maximum(np.sin(np.radians(df["elev"])), 0.02), 20.0)
    day = df[df["elev"] > 5].copy()
    print(f"Day rows (elev>5): {len(day):,}")

    result = {"location": LOCATION, "n_total": len(df), "n_day": len(day),
              "source_table": "ghi_forecast_1h_train_3h_rollback_2021_2025"}

    closure_pred = day["dhi"] + day["dni"] * np.sin(np.radians(day["elev"]))
    closure_err = (closure_pred - day["ghi"]).abs() / np.maximum(day["ghi"], 20.0)
    closure_valid_mask = closure_err.notna()
    violation_rate = float((closure_err[closure_valid_mask] > CLOSURE_THRESHOLD).mean())
    result["closure_n_checked"] = int(closure_valid_mask.sum())
    result["closure_violation_pct_gt15"] = round(100.0 * violation_rate, 2)
    result["valid_row_pct_standardized"] = round(100.0 * (1.0 - violation_rate), 2)

    sentinel_df = sentinel_and_stuck(day.sort_values("ts"), ["ghi", "dni", "dhi", "clp_cot"])
    sentinel_df.to_csv(OUTPUT_DIR / "sentinel_audit_bengkulu.csv", index=False)
    result["sentinel_summary"] = sentinel_df.to_dict("records")

    day_sorted = day.sort_values("ts")
    kt_diff = day_sorted["kt"].diff().abs().mean()
    kt_std = day_sorted["kt"].std()
    cot_diff = day_sorted["clp_cot"].diff().abs().mean()
    cot_std = day_sorted["clp_cot"].std()
    kt_vol = kt_diff / kt_std if kt_std else np.nan
    cot_vol = cot_diff / cot_std if cot_std else np.nan
    result["volatility_ratio_kt"] = round(float(kt_vol), 4)
    result["volatility_ratio_clp_cot"] = round(float(cot_vol), 4)
    result["volatility_ratio_kt_over_cot"] = round(float(kt_vol / cot_vol), 3) if cot_vol else None

    full = df.sort_values("ts").reset_index(drop=True)
    lags_min = list(range(-60, 61, 10))
    corrs = {}
    for lag in lags_min:
        shift_steps = lag // 10
        shifted = full["kt"].shift(-shift_steps)
        mask = (full["elev"] > 5) & full["clp_cot"].notna() & shifted.notna()
        if mask.sum() > 50:
            corrs[lag] = float(np.corrcoef(full.loc[mask, "clp_cot"], shifted[mask])[0, 1])
        else:
            corrs[lag] = np.nan
    peak_lag = max(corrs, key=lambda k: abs(corrs[k]) if not np.isnan(corrs[k]) else -1)
    result["cross_corr_by_lag_min"] = {str(k): (round(v, 4) if not np.isnan(v) else None) for k, v in corrs.items()}
    result["cross_corr_peak_lag_min"] = peak_lag
    result["cross_corr_peak_value"] = round(corrs[peak_lag], 4)
    result["cross_corr_lag0_value"] = round(corrs[0], 4)

    valid_cot = day["clp_cot"].dropna()
    p75, p25 = valid_cot.quantile(0.75), valid_cot.quantile(0.25)
    cond = day["clp_cot"].notna() & day["kt"].notna()
    sub = day[cond]
    contra = ((sub["clp_cot"] > p75) & (sub["kt"] > 0.7)) | ((sub["clp_cot"] < p25) & (sub["kt"] < 0.3))
    result["clot_kt_contradiction_pct"] = round(100.0 * contra.mean(), 2)
    result["clot_kt_contradiction_n"] = int(contra.sum())
    result["clot_kt_contradiction_n_checked"] = int(cond.sum())
    result["clot_p75"] = round(float(p75), 2)
    result["clot_p25"] = round(float(p25), 2)

    result["two_product_n_checked"] = 0
    result["two_product_disagreement_pct"] = None
    result["two_product_note"] = "Tidak ada tabel ARP/aerosol-retrieval terpisah di Bengkulu -- tidak bisa direplikasi (temuan tersendiri: gap dokumentasi/sumber data, lihat 08_Standardisasi_Data_Mentah.md)."

    with open(OUTPUT_DIR / "audit_konsistensi_fisis_bengkulu.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in result.items() if k not in ("cross_corr_by_lag_min", "sentinel_summary")}, indent=2, default=str))
    print(f"\nSaved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
