"""
RUN R2 (paper review #7) — Jambi: target GHI mentah TITIK t+60 (bukan kt, bukan rata-rata).
Mengisi baris Jambi Tabel 2a + angka efek target §4.3 (delta titik -> rata-rata,
fitur/model/split SAMA).

Spek harmonis (mengikuti R1):
- Split kronologis: train s/d 2023-12-31, val 2024, test 2025
- Filter sun_altitude > 5 derajat (anchor DAN waktu target t+60)
- Anchor tanpa gap: toleransi +/-30 detik (lebih ketat dari pipeline internal +/-90s)
- Metrik: R2, MAE, RMSE, skill vs smart-persistence (kt_now x ghi_cs_t6) — test 2025
- Fitur: 96 fitur produksi v3 (identik utk kedua target — syarat perbandingan §4.3)
- Model: CatBoost tuned (4000it, lr0.015, d8, l2=5) = konfigurasi produksi;
  varian train-window: (a) train+val s/d 2024 [headline, konsisten angka historis
  lintas situs], (b) train s/d 2023 saja [sensitivitas, spek R1 literal]
Output: r2_run_jambi_targets.csv
"""
import pandas as pd
import numpy as np
import warnings, time
import pvlib
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

t0 = time.time()
LAT, LON, ELEV = -1.5833, 103.6667, 35.0

df = pd.read_parquet('dfm_with_clp_stats.parquet')
df = df.sort_values('ts').reset_index(drop=True)
ts_arr = (df['ts'].values.astype('int64') / 1_000_000).astype(np.int64)

def add_delta_fast(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90):
    val = df[col].values.astype(float)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n - n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lag_arr = np.full(n, np.nan)
    lag_arr[n_steps:][valid] = val[:n - n_steps][valid]
    df[col + '_delta' + str(n_steps * 10) + 'm'] = val - lag_arr
    return df

def add_lag_fast(df, col, n_steps, ts_arr, step_sec=600, tol_sec=90, name=None):
    val = df[col].values.astype(float)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n - n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lag_arr = np.full(n, np.nan)
    lag_arr[n_steps:][valid] = val[:n - n_steps][valid]
    df[name or (col + '_lag' + str(n_steps * 10) + 'm')] = lag_arr
    return df

def add_lead_strict(df, col, n_steps, ts_arr, step_sec=600, tol_sec=30, name=None):
    """Nilai col di T + n_steps*10 menit, valid hanya jika gap persis (toleransi +/-30s)."""
    val = df[col].values.astype(float)
    n = len(df)
    gaps = ts_arr[n_steps:] - ts_arr[:n - n_steps]
    valid = np.abs(gaps - n_steps * step_sec) <= tol_sec
    lead_arr = np.full(n, np.nan)
    lead_arr[:n - n_steps][valid] = val[n_steps:][valid]
    df[name or (col + '_lead' + str(n_steps * 10) + 'm')] = lead_arr
    return df

# ---------- target titik t+60 (gap-validated +/-30s) ----------
df = add_lead_strict(df, 'ghi_consolidated', 6, ts_arr, tol_sec=30, name='ghi_point_t60')
df = add_lead_strict(df, 'sun_altitude', 6, ts_arr, tol_sec=30, name='sun_alt_at_t60')

# ---------- fitur v3/v4 + future_cs + clp_lags (identik produksi v3) ----------
for var in ['cloud_top_height_m', 'cloud_top_temp_c', 'cloud_eff_radius_um']:
    df = add_delta_fast(df, var, 3, ts_arr)
    df = add_delta_fast(df, var, 6, ts_arr)
df['dni_ratio'] = df['dni_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df['dhi_fraction'] = df['dhi_consolidated'] / df['ghi_consolidated'].clip(lower=1)
df.loc[df['ghi_consolidated'] < 10, ['dni_ratio', 'dhi_fraction']] = np.nan
df['clot_std_roll30m'] = df['clot_std'].rolling(4, min_periods=2).mean()
df = add_delta_fast(df, 'clot_std', 3, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 3, ts_arr)

log("pvlib future clearsky...")
loc = pvlib.location.Location(latitude=LAT, longitude=LON, altitude=ELEV, tz='Asia/Jakarta')
ts_base = pd.DatetimeIndex(df['ts']).tz_localize('Asia/Jakarta')
for k in range(1, 7):
    ts_f = ts_base + pd.Timedelta(minutes=10 * k)
    sp = loc.get_solarposition(ts_f)
    cs = loc.get_clearsky(ts_f, model='ineichen')
    df[f'ghi_cs_t{k}'] = cs['ghi'].values
    df[f'sun_alt_t{k}'] = sp['apparent_elevation'].values
df['ghi_cs_next_mean'] = df[[f'ghi_cs_t{k}' for k in range(1, 7)]].mean(axis=1)
df['ghi_persist_next'] = df['kt'] * df['ghi_cs_next_mean']
df['ghi_persist_t60'] = df['kt'] * df['ghi_cs_t6']   # smart-persistence utk target titik

for n in [1, 2, 3]:
    df = add_lag_fast(df, 'clot_median', n, ts_arr)
df = add_delta_fast(df, 'clot_median', 1, ts_arr)
df = add_delta_fast(df, 'clot_median', 3, ts_arr)
df['clot_median_roll30m'] = df['clot_median'].rolling(4, min_periods=2).mean()
for n in [1, 2]:
    df = add_lag_fast(df, 'cler23_coverage', n, ts_arr)
df = add_delta_fast(df, 'cler23_coverage', 1, ts_arr)

FEATS_V2 = [
    'sun_altitude', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'month',
    'kt', 'kt_lag10m', 'kt_lag20m', 'kt_lag30m', 'kt_lag60m',
    'kt_roll30m_mean', 'kt_roll60m_mean', 'kt_roll30m_std',
    'kt_short_lag1m', 'kt_short_lag2m', 'kt_short_lag3m', 'kt_short_lag5m',
    'kt_short_roll5m_mean', 'kt_short_roll5m_std',
    'kt_delta_1m', 'kt_delta_5m', 'kt_delta_10m', 'kt_slope_5m',
    'cloud_optical_thickness', 'cloud_top_temp_c', 'cloud_top_height_m',
    'cloud_eff_radius_um', 'sat_cloud_present',
    'cloud_cover_oktas', 'cloud_low_type', 'cloud_med_type', 'cloud_high_type',
    'cloud_optical_thickness_lag10m', 'cloud_optical_thickness_delta10m',
    'cloud_cover_oktas_lag10m', 'cloud_cover_oktas_delta10m',
    'cloud_optical_thickness_roll60m_mean', 'cloud_optical_thickness_roll60m_std',
    'cloud_cover_oktas_roll60m_mean', 'cloud_cover_oktas_roll60m_std',
    'angstrom_exp_440_870', 'precipitable_water_cm', 'AOD_planck_avg',
    'cloud_optical_thickness_delta30m', 'cloud_optical_thickness_delta60m',
    'cloud_cover_oktas_delta30m', 'cloud_cover_oktas_delta60m',
]
V3_NEW = [
    'dni_ratio', 'dhi_fraction',
    'cloud_top_height_m_delta30m', 'cloud_top_temp_c_delta30m', 'cloud_eff_radius_um_delta30m',
    'cloud_top_height_m_delta60m', 'cloud_top_temp_c_delta60m', 'cloud_eff_radius_um_delta60m',
]
V4_CLP = [
    'clot_std', 'clot_median', 'clot_coverage',
    'cler23_coverage', 'cler23_std', 'cler23_mean',
    'clth_mean', 'clth_std', 'clth_median',
    'cltt_mean', 'cltt_std', 'cltt_median',
    'cloud_class_code_new',
    'cler23_coverage_delta30m', 'clot_std_delta30m', 'clot_std_roll30m',
]
FUTURE_CS = ([f'ghi_cs_t{k}' for k in range(1, 7)] +
             [f'sun_alt_t{k}' for k in range(1, 7)] +
             ['ghi_cs_next_mean', 'ghi_persist_next'])
CLP_LAGS = [
    'clot_median_lag10m', 'clot_median_lag20m', 'clot_median_lag30m',
    'clot_median_delta10m', 'clot_median_delta30m', 'clot_median_roll30m',
    'cler23_coverage_lag10m', 'cler23_coverage_lag20m', 'cler23_coverage_delta10m',
]
FEATS = FEATS_V2 + ['ghi_clearsky'] + V3_NEW + V4_CLP + FUTURE_CS + CLP_LAGS

# ---------- split harmonis R1 ----------
ts = pd.to_datetime(df['ts'])
idx_train = ts < '2024-01-01'
idx_val = (ts >= '2024-01-01') & (ts < '2025-01-01')
idx_test = ts >= '2025-01-01'
log(f"Split: train<=2023 n={idx_train.sum()}, val 2024 n={idx_val.sum()}, test 2025 n={idx_test.sum()}")

TARGETS = {
    'ghi_point_t60': dict(sp_col='ghi_persist_t60',
                          extra_mask=lambda d: d['sun_alt_at_t60'] > 5),
    'ghi_next_1h_mean': dict(sp_col='ghi_persist_next',
                             extra_mask=lambda d: pd.Series(True, index=d.index)),
}
TUNED = dict(iterations=4000, learning_rate=0.015, depth=8, l2_leaf_reg=5,
             min_data_in_leaf=20, loss_function='Quantile:alpha=0.5',
             verbose=0, random_seed=42, allow_writing_files=False)

rows = []
for tgt, cfg in TARGETS.items():
    base_mask = df[tgt].notna() & (df['sun_altitude'] > 5) & cfg['extra_mask'](df)
    m_te = base_mask & idx_test
    y_te = df.loc[m_te, tgt].astype(float)
    X_te = df.loc[m_te, FEATS].astype(float)

    # smart-persistence
    sp = df.loc[m_te, cfg['sp_col']].astype(float)
    ok = sp.notna()
    r2_sp = r2_score(y_te[ok], sp[ok])
    mse_sp = ((y_te[ok] - sp[ok]) ** 2).mean()
    rows.append(dict(target=tgt, config='smart_persistence', train_window='-',
                     r2=r2_sp, mae=mean_absolute_error(y_te[ok], sp[ok]),
                     rmse=float(np.sqrt(mse_sp)), skill_vs_sp=0.0, n_test=int(ok.sum())))
    log(f"\n[{tgt}] n_test={m_te.sum()}  smart-persistence R2={r2_sp:.4f}")

    for twname, m_tr in [('train<=2024 (train+val)', base_mask & (idx_train | idx_val)),
                         ('train<=2023 (spek literal)', base_mask & idx_train)]:
        m = CatBoostRegressor(**TUNED)
        m.fit(df.loc[m_tr, FEATS].astype(float), df.loc[m_tr, tgt].astype(float))
        pred = m.predict(X_te)
        r2 = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        rmse = float(np.sqrt(((y_te - pred) ** 2).mean()))
        skill = 1 - ((y_te - pred) ** 2).mean() / mse_sp
        rows.append(dict(target=tgt, config='catboost_tuned_4000_d8', train_window=twname,
                         r2=r2, mae=mae, rmse=rmse, skill_vs_sp=skill, n_test=int(m_te.sum())))
        log(f"  {twname:28s}: R2={r2:.4f}  MAE={mae:.1f}  RMSE={rmse:.1f}  skill={skill:+.3f}  ({time.time()-t0:.0f}s)")

out = pd.DataFrame(rows)
out.to_csv('r2_run_jambi_targets.csv', index=False)
log("\n=== RINGKASAN (headline = train<=2024) ===")
log(out.to_string(index=False))
log(f"\nSaved r2_run_jambi_targets.csv  (total {time.time()-t0:.0f}s)")
