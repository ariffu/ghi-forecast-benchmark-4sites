#!/usr/bin/env python3
"""
Audit Konsistensi Fisis + Closure + Sentinel -- KALBAR (Prioritas A2, 2026-07-25)

Replikasi metodologi `note_09_audit_konsistensi_fisis.md` (dikutip di
Restrukturisasi/08_Standardisasi_Data_Mentah.md SS1) dengan skrip TUNGGAL yang
dijalankan IDENTIK di keempat lokasi, supaya angka benar-benar sebanding
(bukan diambil dari audit lama yang metodologinya berbeda-beda per lokasi).
Kalbar sendiri direcompute di sini juga (bukan cuma dikutip dari note_09)
untuk memastikan definisi threshold 100% sama dengan 3 lokasi lain.

Metrik yang dihitung (semua di atas grid 10-menit solar_kalbar_10m + join):
  1. Closure violation: |({dhi}+{dni}*sin(elev)) - ghi| / max(ghi,20) > 15%,
     dihitung di baris siang (elev>5). Threshold 15% dipilih supaya konsisten
     dengan definisi Jambi (satu-satunya lokasi yang sudah eksplisit).
  2. "Baris valid": 1 - closure_violation_rate (definisi standar baru).
  3. Sentinel value: hitung 9999/-9999/999/-999 di kolom radiasi & meteo utama,
     + "stuck value" (>=6 pembacaan berurutan identik non-null, ~1 jam).
  4. Rasio volatilitas: mean(|diff 10-menit|)/std, dibandingkan antara kt
     (proxy target) vs clp_cot (fitur satelit) -- rasio >1 berarti target
     berubah lebih cepat dari resolusi temporal fitur cloud bisa tangkap.
  5. Cross-correlation lag: corr(clp_cot(t), kt(t+lag)) untuk lag -60..+60
     menit (step 10), dicari lag dengan |korelasi| puncak.
  6. Kontradiksi CLOT vs kt: baris siang dengan CLOT > P75 TAPI kt > 0.7
     (awan tebal dilaporkan tapi radiasi hampir clear-sky), ATAU CLOT < P25
     TAPI kt < 0.3 (awan tipis dilaporkan tapi radiasi rendah/berawan).
  7. [Kalbar only, produk kedua tersedia] Kontradiksi CLP cloud_present vs
     ARP clear_sky (arp_pontianak) -- replikasi persis Temuan #2 note_09.

Run:
    python audit_data_quality_kalbar.py
"""
import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = "kalbar_local.db"
OUTPUT_DIR = Path("outputs_audit_konsistensi_fisis")
OUTPUT_DIR.mkdir(exist_ok=True)
LOCATION = "Kalbar"
CLOSURE_THRESHOLD = 0.15
STUCK_MIN_RUN = 6  # 6 x 10min = 1 jam


def load_base():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            s.timestamp_wib AS ts,
            s.ghi_final AS ghi, s.dni_final AS dni, s.dhi_final AS dhi,
            s.sun_altitude AS elev,
            c.CLOT_mean AS clp_cot, c.cloud_present AS clp_cloud_present_raw
        FROM solar_kalbar_10m s
        LEFT JOIN clp_pontianak_20km c ON s.timestamp_wib = c.timestamp
        ORDER BY s.timestamp_wib
    """).fetchdf()
    arp = con.execute("""
        SELECT timestamp_wib AS ts, clear_sky, aerosol_retrieval_valid
        FROM arp_pontianak ORDER BY timestamp_wib
    """).fetchdf()
    con.close()
    return df, arp


def sentinel_and_stuck(df_day, cols):
    """NOTE (2026-07-25 fix): dihitung HANYA di baris siang (elev>5) dan run
    bernilai TEPAT NOL di-exclude, supaya tidak mencampur 'sensor macet
    genuine' dengan 'GHI/DNI/DHI=0 yang valid secara fisis waktu malam/senja'
    -- versi pertama audit ini (sebelum fix) melaporkan stuck_run_pct 50-60%
    yang ternyata murni artefak nol-malam, bukan temuan kualitas data riil."""
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
        run_val = vals[0] if len(vals) else None
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
    df, arp = load_base()
    print(f"Rows loaded: {len(df):,}")
    df["kt"] = df["ghi"] / np.maximum(1100.0 * np.maximum(np.sin(np.radians(df["elev"])), 0.02), 20.0)
    day = df[df["elev"] > 5].copy()
    print(f"Day rows (elev>5): {len(day):,}")

    result = {"location": LOCATION, "n_total": len(df), "n_day": len(day)}

    # 1+2. Closure + valid-row
    closure_pred = day["dhi"] + day["dni"] * np.sin(np.radians(day["elev"]))
    closure_err = (closure_pred - day["ghi"]).abs() / np.maximum(day["ghi"], 20.0)
    closure_valid_mask = closure_err.notna()
    violation_rate = float((closure_err[closure_valid_mask] > CLOSURE_THRESHOLD).mean())
    result["closure_n_checked"] = int(closure_valid_mask.sum())
    result["closure_violation_pct_gt15"] = round(100.0 * violation_rate, 2)
    result["valid_row_pct_standardized"] = round(100.0 * (1.0 - violation_rate), 2)

    # 3. Sentinel + stuck
    sentinel_df = sentinel_and_stuck(day.sort_values("ts"), ["ghi", "dni", "dhi", "clp_cot"])
    sentinel_df.to_csv(OUTPUT_DIR / "sentinel_audit_kalbar.csv", index=False)
    result["sentinel_summary"] = sentinel_df.to_dict("records")

    # 4. Volatility ratio
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

    # 5. Cross-correlation lag (on FULL series incl. night, to preserve calendar time; corr computed on day subset)
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

    # 6. CLOT vs kt contradiction
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

    # 7. Two-product contradiction: CLP cloud_present vs ARP clear_sky (Kalbar only)
    merged = pd.merge(day, arp, on="ts", how="inner")
    merged = merged[merged["clp_cloud_present_raw"].notna() & merged["clear_sky"].notna()]
    if len(merged) > 50:
        clp_says_cloud = merged["clp_cloud_present_raw"].astype(bool)
        arp_says_clear = merged["clear_sky"].astype(bool)
        disagree = (clp_says_cloud & arp_says_clear) | (~clp_says_cloud & ~arp_says_clear)
        # "false positive": CLP says cloud but radiation looks clear (kt high)
        false_positive = clp_says_cloud & arp_says_clear & (merged["kt"] > 0.7)
        result["two_product_n_checked"] = int(len(merged))
        result["two_product_disagreement_pct"] = round(100.0 * disagree.mean(), 1)
        result["two_product_false_positive_pct_of_all_day"] = round(100.0 * false_positive.sum() / len(day), 1)
    else:
        result["two_product_n_checked"] = 0
        result["two_product_disagreement_pct"] = None

    with open(OUTPUT_DIR / "audit_konsistensi_fisis_kalbar.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in result.items() if k not in ("cross_corr_by_lag_min", "sentinel_summary")}, indent=2, default=str))
    print(f"\nSaved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
