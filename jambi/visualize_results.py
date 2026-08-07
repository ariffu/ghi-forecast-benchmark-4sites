#!/usr/bin/env python3
"""
visualize_results.py
=====================
Visualisasi GHI multi-horizon 6 jam — Stasiun Jambi.

Jalankan dari direktori proyek:
  python visualize_results.py

Output: models_multistep/figures/*.png
"""

import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
DATASET  = ROOT / "dataset_multistep"
MODELS   = ROOT / "models_multistep"
FIG_DIR  = MODELS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HORIZON  = 6
HORIZONS = list(range(1, HORIZON + 1))

# ─── WARNA ────────────────────────────────────────────────────────────────────
C_HUJAN   = "#2E86AB"
C_KEMARAU = "#E07A5F"
C_GLOBAL  = "#3D405B"
C_STAGES  = ["#9DC3C1", "#5B8DB8", "#3D405B", "#E07A5F"]

STAGE_LABELS = {1: "S1: Solar", 2: "S2: +Cloud", 3: "S3: +Meteo", 4: "S4: +Musim"}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "grid.alpha":       0.3,
})

# ─── UTILITAS ─────────────────────────────────────────────────────────────────
def r2(yt, yp, mask=None):
    if mask is not None:
        yt, yp = yt[mask], yp[mask]
    m = ~(np.isnan(yt) | np.isnan(yp))
    return r2_score(yt[m], yp[m]) if m.sum() >= 5 else np.nan

def rmse_fn(yt, yp, mask=None):
    if mask is not None:
        yt, yp = yt[mask], yp[mask]
    m = ~(np.isnan(yt) | np.isnan(yp))
    return np.sqrt(mean_squared_error(yt[m], yp[m])) if m.sum() >= 5 else np.nan

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
def load_data():
    tr = pd.read_parquet(DATASET / "jambi_ms_train.parquet")
    va = pd.read_parquet(DATASET / "jambi_ms_val.parquet")
    te = pd.read_parquet(DATASET / "jambi_ms_test.parquet")
    return tr, va, te

# ─── LOAD PREDICTIONS ────────────────────────────────────────────────────────
def predict_all(tr, va, te):
    sys.path.insert(0, str(ROOT))
    try:
        from train_multistep_lgbm import STAGE_FEATS
    except ImportError:
        print("ERROR: Tidak bisa import train_multistep_lgbm.py")
        return {}

    preds = {}
    for s in [1, 2, 3, 4]:
        req = STAGE_FEATS[s]
        fc  = [c for c in req if c in tr.columns]
        for h in HORIZONS:
            mp = MODELS / f"ms_s{s}_h{h}.txt"
            if not mp.exists():
                continue
            model = lgb.Booster(model_file=str(mp))
            best  = model.best_iteration if model.best_iteration > 0 else None
            for split, df in [("val", va), ("test", te)]:
                X = df[fc].values.astype(float)
                preds[(s, h, split)] = model.predict(X, num_iteration=best)

    print(f"  Loaded: {len(preds)} model-split prediksi")
    return preds


# ─── FIG 1: R² CURVES ─────────────────────────────────────────────────────────
def fig1_r2_curves(preds, va, te):
    print("  Fig 1: R² curves ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GHI Multi-Horizon Forecasting — Stasiun Jambi (Test 2025)",
                 fontsize=13, fontweight="bold", y=1.01)
    xlbls = [f"h+{h}\n({h}h)" for h in HORIZONS]

    # Panel A: Stage comparison
    ax = axes[0]
    for s, clr in zip([1,2,3,4], C_STAGES):
        vals = [r2(te[f"ghi_h{h}"].values, preds.get((s,h,"test"), np.full(len(te),np.nan)))
                for h in HORIZONS]
        ax.plot(HORIZONS, vals, "o-", color=clr, lw=2, ms=6, label=STAGE_LABELS[s])
    pers = [r2(te[f"ghi_h{h}"].values, te["ghi_h"].values) for h in HORIZONS]
    ax.plot(HORIZONS, pers, "s--", color="#999", lw=1.5, ms=5, label="Persistence")
    ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.5)
    ax.set(xlabel="Horizon", ylabel="R²", title="A. Perbandingan Stage")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(xlbls)
    ax.legend(fontsize=8); ax.set_ylim(-0.25, 0.85)
    ax.grid(True, axis="y")

    # Panel B: U-shape
    ax = axes[1]
    for split, df_s, clr, ls, label in [
        ("val",  va, "#5B8DB8", "-",  "Val 2024"),
        ("test", te, C_GLOBAL,  "--", "Test 2025"),
    ]:
        vals = [r2(df_s[f"ghi_h{h}"].values, preds.get((3,h,split), np.full(len(df_s),np.nan)))
                for h in HORIZONS]
        ax.plot(HORIZONS, vals, "o"+ls, color=clr, lw=2.5, ms=8, label=f"S3 {label}", zorder=3)
    ax.axvspan(2.5, 4.5, alpha=0.08, color="red")
    ax.text(3.5, 0.05, "Dead zone\n(autocorr habis,\ndiurnal belum dominan)",
            ha="center", fontsize=8, color="darkred", style="italic")
    ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.5)
    ax.set(xlabel="Horizon", ylabel="R²", title="B. Pola U-Shape (Stage 3)")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(xlbls)
    ax.legend(fontsize=8); ax.set_ylim(-0.15, 0.85)
    ax.grid(True, axis="y")

    # Panel C: Season comparison
    ax = axes[2]
    wet = te["is_wet_season"].values.astype(bool)
    dry = ~wet
    for musim, mask, clr in [("Hujan",wet,C_HUJAN),("Kemarau",dry,C_KEMARAU)]:
        vals = [r2(te[f"ghi_h{h}"].values,
                   preds.get((3,h,"test"), np.full(len(te),np.nan)), mask)
                for h in HORIZONS]
        ax.plot(HORIZONS, vals, "o-", color=clr, lw=2.5, ms=8, label=musim)
    for h in HORIZONS:
        yp = preds.get((3,h,"test"), np.full(len(te),np.nan))
        rh = r2(te[f"ghi_h{h}"].values, yp, wet)
        rk = r2(te[f"ghi_h{h}"].values, yp, dry)
        if not (np.isnan(rh) or np.isnan(rk)):
            ax.text(h, (rh+rk)/2 + 0.03, f"+{rk-rh:.2f}", ha="center", fontsize=7.5, color="#555")
    ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.5)
    ax.set(xlabel="Horizon", ylabel="R²", title="C. Hujan vs Kemarau (angka=ΔR²)")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(xlbls)
    handles = [Patch(color=C_HUJAN, label="Musim Hujan (Okt-Apr)"),
               Patch(color=C_KEMARAU, label="Musim Kemarau (Mei-Sep)")]
    ax.legend(handles=handles, fontsize=8); ax.set_ylim(-0.15, 0.85)
    ax.grid(True, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "fig1_r2_curves.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── FIG 2: ERROR PER JAM ANCHOR ─────────────────────────────────────────────
def fig2_error_by_hour(preds, te):
    print("  Fig 2: Error per jam anchor ...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("RMSE per Jam Anchor — Stage 3, Test 2025", fontsize=12, fontweight="bold")

    wet   = te["is_wet_season"].values.astype(bool)
    dry   = ~wet
    hours = te["hour_wib"].dt.hour.values

    for idx, h in enumerate(HORIZONS):
        ax   = axes[idx // 3][idx % 3]
        col  = f"ghi_h{h}"
        yt   = te[col].values
        yp   = preds.get((3, h, "test"), np.full(len(te), np.nan))
        err2 = (yp - yt) ** 2

        for i, (musim, mask, clr) in enumerate([("Hujan",wet,C_HUJAN),("Kemarau",dry,C_KEMARAU)]):
            hr_list, rmse_list, n_list = [], [], []
            for hr in range(6, 12):
                m = mask & (hours == hr) & ~np.isnan(yt) & ~np.isnan(yp)
                if m.sum() >= 5:
                    hr_list.append(hr)
                    rmse_list.append(np.sqrt(err2[m].mean()))
                    n_list.append(int(m.sum()))
            if not hr_list:
                continue
            offset = -0.2 if i == 0 else 0.2
            bars   = ax.bar([hr + offset for hr in hr_list], rmse_list,
                            width=0.35, color=clr, alpha=0.80, label=musim, edgecolor="white")
            for bar, n in zip(bars, n_list):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 2, str(n), ha="center", fontsize=6.5, color="#555")

        ax.set(title=f"h+{h} ({h}h ke depan)",
               xlabel="Jam Anchor (WIB)", ylabel="RMSE (W/m²)")
        ax.set_xticks(range(6, 12))
        ax.set_xticklabels([f"{j}:00" for j in range(6, 12)])
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig2_error_by_hour.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── FIG 3: SCATTER ───────────────────────────────────────────────────────────
def fig3_scatter(preds, te):
    print("  Fig 3: Scatter ...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Prediksi vs Aktual — Stage 3, Test 2025", fontsize=12, fontweight="bold")

    wet = te["is_wet_season"].values.astype(bool)
    dry = ~wet

    for idx, h in enumerate(HORIZONS):
        ax  = axes[idx // 3][idx % 3]
        col = f"ghi_h{h}"
        yt  = te[col].values
        yp  = preds.get((3, h, "test"), np.full(len(te), np.nan))

        for mask, clr, lbl in [(wet,C_HUJAN,"Hujan"),(dry,C_KEMARAU,"Kemarau")]:
            m  = mask & ~np.isnan(yt) & ~np.isnan(yp)
            rv = r2(yt[m], yp[m]) if m.sum() >= 5 else np.nan
            ax.scatter(yt[m], yp[m], s=4, alpha=0.25, color=clr, rasterized=True,
                       label=f"{lbl} R²={rv:.3f}" if not np.isnan(rv) else lbl)

        m_all = ~np.isnan(yt) & ~np.isnan(yp)
        vmax  = max(yt[m_all].max(), yp[m_all].max()) * 1.05 if m_all.any() else 1000
        ax.plot([0, vmax], [0, vmax], "k--", lw=0.9, alpha=0.5)
        r_all = r2(yt[m_all], yp[m_all]) if m_all.sum() >= 5 else np.nan
        ax.set(title=f"h+{h} ({h}h) — Global R²={r_all:.3f}",
               xlabel="GHI Aktual (W/m²)", ylabel="GHI Prediksi (W/m²)")
        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
        ax.legend(fontsize=7.5, markerscale=4)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "fig3_scatter.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── FIG 4: CLOUD REGIME ──────────────────────────────────────────────────────
def fig4_cloud_regime(preds, te):
    print("  Fig 4: Cloud regime ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("R² per Cloud Regime — Stage 3, Test 2025", fontsize=12, fontweight="bold")

    regimes = [(0,2,"Clear\n(0-2)"),(3,5,"Scattered\n(3-5)"),
               (6,7,"Broken\n(6-7)"),(8,9,"Overcast\n(8+)")]
    x     = np.arange(len(regimes))
    oktas = te["cloud_oktas"].values
    wet   = te["is_wet_season"].values.astype(bool)
    dry   = ~wet

    for ax, h in zip(axes, [1, 6]):
        col = f"ghi_h{h}"
        yt  = te[col].values
        yp  = preds.get((3, h, "test"), np.full(len(te), np.nan))

        for i, (musim, mask, clr) in enumerate([("Hujan",wet,C_HUJAN),("Kemarau",dry,C_KEMARAU)]):
            r2s, ns = [], []
            for lo, hi, _ in regimes:
                m = mask & (oktas >= lo) & (oktas <= hi) & ~np.isnan(yt) & ~np.isnan(yp)
                r2s.append(r2(yt[m], yp[m]) if m.sum() >= 10 else np.nan)
                ns.append(int(m.sum()))

            bars = ax.bar(x + i*0.35 - 0.175, r2s, 0.35,
                          label=musim, color=clr, alpha=0.82, edgecolor="white")
            for bar, rv, n in zip(bars, r2s, ns):
                if not np.isnan(rv):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            max(rv,0) + 0.01, f"n={n}", ha="center", fontsize=7, color="#333")

        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set(xlabel="Cloud Regime", ylabel="R²", title=f"h+{h} ({h}h ke depan)")
        ax.set_xticks(x); ax.set_xticklabels([r[2] for r in regimes])
        ax.set_ylim(-0.25, 0.95); ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fig4_cloud_regime.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── FIG 5: HEATMAP ───────────────────────────────────────────────────────────
def fig5_heatmap(preds, va, te):
    print("  Fig 5: Heatmap ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Heatmap R² — Stage × Horizon", fontsize=12, fontweight="bold")

    for ax, (split, df_s) in zip(axes, [("Val 2024",va),("Test 2025",te)]):
        sp_key = "val" if "Val" in split else "test"
        mat    = np.full((4, HORIZON), np.nan)
        for si, s in enumerate([1,2,3,4]):
            for hi, h in enumerate(HORIZONS):
                yt = df_s[f"ghi_h{h}"].values
                yp = preds.get((s,h,sp_key), np.full(len(df_s),np.nan))
                mat[si,hi] = r2(yt, yp)

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.15, vmax=0.85, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label="R²")
        for si in range(4):
            for hi in range(HORIZON):
                v = mat[si,hi]
                if not np.isnan(v):
                    c = "white" if v < 0.2 or v > 0.72 else "black"
                    ax.text(hi, si, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color=c, fontweight="bold")
        ax.set_xticks(range(HORIZON))
        ax.set_xticklabels([f"h+{h}" for h in HORIZONS])
        ax.set_yticks(range(4))
        ax.set_yticklabels([STAGE_LABELS[s] for s in [1,2,3,4]], fontsize=9)
        ax.set(xlabel="Horizon", title=split)

    plt.tight_layout()
    out = FIG_DIR / "fig5_heatmap.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── FIG 6: FEATURE IMPORTANCE ────────────────────────────────────────────────
def fig6_feature_importance():
    print("  Fig 6: Feature importance ...")

    combos = [
        ("ms_s3_h1_imp.csv",        "Global h+1 — horizon pendek\n(current state dominan)",   C_GLOBAL),
        ("ms_s3_h6_imp.csv",        "Global h+6 — horizon jauh\n(diurnal cycle dominan)",      C_GLOBAL),
        ("ms_s3_h1_hujan_imp.csv",  "Musim Hujan, h+1\n(konveksi kuat)",                       C_HUJAN),
        ("ms_s3_h1_kemarau_imp.csv","Musim Kemarau, h+1\n(atmosfer stabil)",                    C_KEMARAU),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Top 15 Feature Importance — Stage 3 (Gain)", fontsize=12, fontweight="bold")

    def shorten(name):
        for old, new in [
            ("clearsky_pvlib_h", "cs_pvlib_h"),
            ("sun_alt_pvlib_h",  "sa_pvlib_h"),
            ("sun_az_pvlib_h",   "sz_pvlib_h"),
            ("ghi_last_lag",     "ghi_last_"),
        ]:
            name = name.replace(old, new)
        return name

    for ax, (fname, title, clr) in zip(axes.flat, combos):
        fp = MODELS / fname
        if not fp.exists():
            ax.text(0.5, 0.5, f"File tidak ada:\n{fname}\n(jalankan train_multistep_lgbm.py dulu)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666")
            ax.set_title(title, fontsize=10); continue

        imp = pd.read_csv(fp).head(15)
        imp["short"] = imp["feature"].apply(shorten)
        bars = ax.barh(range(len(imp)), imp["pct"], color=clr, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["short"].tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel="Importance (%)", title=title)
        ax.grid(True, axis="x", alpha=0.3)
        for bar, pct in zip(bars, imp["pct"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{pct:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "fig6_feature_importance.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── FIG 7: SEASONAL SUMMARY ──────────────────────────────────────────────────
def fig7_season_summary(preds, va, te):
    print("  Fig 7: Season summary ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Ringkasan Analisis Musim — Stage 3", fontsize=12, fontweight="bold")

    wet_te = te["is_wet_season"].values.astype(bool)
    dry_te = ~wet_te
    wet_va = va["is_wet_season"].values.astype(bool)
    dry_va = ~wet_va
    xlbls  = [f"h+{h}\n({h}h)" for h in HORIZONS]

    # Panel A: R² per musim (val + test)
    ax = axes[0]
    for split, df_s, wm, dm, ls, alpha in [
        ("val",  va, wet_va, dry_va, "--", 0.5),
        ("test", te, wet_te, dry_te, "-",  1.0),
    ]:
        for musim, mask, clr in [("Hujan",wm,C_HUJAN),("Kemarau",dm,C_KEMARAU)]:
            vals = [r2(df_s[f"ghi_h{h}"].values,
                       preds.get((3,h,split), np.full(len(df_s),np.nan)), mask)
                    for h in HORIZONS]
            lbl = musim if split == "test" else None
            ax.plot(HORIZONS, vals, "o"+ls, color=clr, lw=2.5,
                    ms=7 if split=="test" else 5, alpha=alpha, label=lbl, zorder=3)

    r2_h = [r2(te[f"ghi_h{h}"].values, preds.get((3,h,"test"),np.full(len(te),np.nan)), wet_te) for h in HORIZONS]
    r2_k = [r2(te[f"ghi_h{h}"].values, preds.get((3,h,"test"),np.full(len(te),np.nan)), dry_te) for h in HORIZONS]
    ax.fill_between(HORIZONS, r2_h, r2_k, alpha=0.08, color="#999")

    ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.5)
    ax.set(xlabel="Horizon", ylabel="R²",
           title="R² per Musim (putus=Val 2024, penuh=Test 2025)")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(xlbls)
    handles = [Patch(color=C_HUJAN,   label="Musim Hujan (Okt-Apr)"),
               Patch(color=C_KEMARAU, label="Musim Kemarau (Mei-Sep)")]
    ax.legend(handles=handles, fontsize=9)
    ax.set_ylim(-0.1, 0.85); ax.grid(True, axis="y")

    # Panel B: ΔR² per horizon
    ax = axes[1]
    delta = []
    for h in HORIZONS:
        yp = preds.get((3, h, "test"), np.full(len(te), np.nan))
        yt = te[f"ghi_h{h}"].values
        rh = r2(yt, yp, wet_te)
        rk = r2(yt, yp, dry_te)
        delta.append(rk - rh if not (np.isnan(rh) or np.isnan(rk)) else np.nan)

    colors = [C_KEMARAU if not np.isnan(d) and d > 0 else C_HUJAN for d in delta]
    bars   = ax.bar(HORIZONS, delta, color=colors, alpha=0.82, edgecolor="white")
    for bar, d in zip(bars, delta):
        if not np.isnan(d):
            ypos = d + 0.004 if d >= 0 else d - 0.009
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{d:+.3f}", ha="center", fontsize=9, fontweight="bold", color="#333")

    mean_d = np.nanmean(delta)
    ax.axhline(0,      color="black", lw=1.0, alpha=0.7)
    ax.axhline(mean_d, color="#666",  lw=1.5, ls="--",
               label=f"Rata-rata ΔR² = {mean_d:+.3f}")
    ax.set(xlabel="Horizon", ylabel="ΔR² (Kemarau − Hujan)",
           title="Keunggulan Kemarau atas Hujan\n(+= Kemarau lebih mudah diprediksi)")
    ax.set_xticks(HORIZONS); ax.set_xticklabels(xlbls)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fig7_season_summary.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print(f"    → {out.name}")


# ─── PRINT TABEL NUMERIK ──────────────────────────────────────────────────────
def print_summary(preds, va, te):
    print("\n" + "="*65)
    print("TABEL NUMERIK — Stage 3, Test 2025")
    print("="*65)
    wet = te["is_wet_season"].values.astype(bool)
    dry = ~wet
    print(f"\n  {'Horizon':>8} | {'Global':>8} | {'Hujan':>8} | {'Kemarau':>8} | {'ΔR²':>7}")
    print(f"  {'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*7}")
    for h in HORIZONS:
        yt = te[f"ghi_h{h}"].values
        yp = preds.get((3,h,"test"), np.full(len(te),np.nan))
        rg = r2(yt, yp)
        rh = r2(yt, yp, wet)
        rk = r2(yt, yp, dry)
        dr = rk - rh if not (np.isnan(rh) or np.isnan(rk)) else np.nan
        print(f"  h+{h} ({h:2d}h) | {rg:8.4f} | {rh:8.4f} | {rk:8.4f} | {dr:+7.4f}")
    print(f"\n  n(hujan)={wet.sum():,}  n(kemarau)={dry.sum():,}")
    print("="*65)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("VISUALISASI — GHI Multi-Horizon Jambi")
    print("=" * 55)

    tr, va, te = load_data()
    print(f"  Data: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    preds = predict_all(tr, va, te)
    if not preds:
        print("Tidak ada prediksi. Pastikan models_multistep/*.txt tersedia.")
        sys.exit(1)

    print(f"\nGenerating figures → {FIG_DIR.relative_to(ROOT)}")
    fig1_r2_curves(preds, va, te)
    fig2_error_by_hour(preds, te)
    fig3_scatter(preds, te)
    fig4_cloud_regime(preds, te)
    fig5_heatmap(preds, va, te)
    fig6_feature_importance()
    fig7_season_summary(preds, va, te)

    print_summary(preds, va=va, te=te)

    print("\n✓ Selesai! File PNG tersimpan di: models_multistep/figures/")
    print("  fig1_r2_curves.png       — U-shape + stage + musim")
    print("  fig2_error_by_hour.png   — RMSE per jam anchor (6-11 WIB)")
    print("  fig3_scatter.png         — Scatter pred vs aktual per musim")
    print("  fig4_cloud_regime.png    — R² per oktas regime × musim")
    print("  fig5_heatmap.png         — Heatmap R² stage × horizon")
    print("  fig6_feature_importance.png — Top-15 fitur per kondisi")
    print("  fig7_season_summary.png  — ΔR² Kemarau−Hujan per horizon")