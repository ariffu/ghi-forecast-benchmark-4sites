#!/usr/bin/env python3
"""
Focused 60-minute-ahead GHI residual benchmark for Bengkulu.

Goal:
    Re-test the model families that performed best for 60m residual forecasting:
        - LightGBM residual
        - LightGBM weighted residual for high-GHI cases
        - LSTM residual
        - Attention-LSTM residual
        - Transformer residual
        - Ensembles of residual models

Dataset period:
    2021-2025 only, 2026 excluded.

Split:
    Train      2021-2023
    Validation 2024
    Test       2025

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow lightgbm torch tqdm

Run:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal after setx
    python train_ghi_60m_residual_lgbm_lstm_attention.py

Outputs:
    outputs_60m_residual_models/metrics.csv
    outputs_60m_residual_models/metrics_by_segment.csv
    outputs_60m_residual_models/predictions_test.csv
    outputs_60m_residual_models/feature_importance_lgbm.csv
    outputs_60m_residual_models/diagnostics.png
    outputs_60m_residual_models/models/*.joblib / *.pt
"""

import os
from pathlib import Path
import warnings

import duckdb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
OUTPUT_DIR = Path("outputs_60m_residual_models")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_COL = "target_ghi_60m"
TARGET_TS_COL = "target_ts_60m"
DELTA_COL = "target_delta_60m"
PRED_MIN = 0.0
PRED_MAX = 1400.0
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42
SEQ_LEN = 19
BATCH_SIZE = 512
MAX_EPOCHS = 45
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

TABULAR_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "daylight_flag", "sun_above_5deg_flag",
    "ghi_now", "dhi_now", "dni_now", "reflected_now", "nett_rad_now", "solar_elev_deg",
    "asrs_n_obs_1min", "asrs_ok_obs", "aws_temp_c", "aws_temp_min_c", "aws_temp_max_c",
    "aws_rh_pct", "aws_pressure_hpa", "aws_ws_avg", "aws_ws_max", "aws_wd_deg", "aws_rain_mm",
    "aws_sr_avg_w_m2", "clp_cot", "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "clp_clear_flag", "clp_thin_cloud_flag", "clp_moderate_cloud_flag", "clp_thick_cloud_flag",
    "synop_temp_c", "synop_dewpoint_c", "synop_rh_pct", "synop_wind_speed", "synop_wind_dir_deg",
    "synop_visibility", "synop_rainfall_24h_mm", "synop_solar_rad_24h",
    "ghi_lag_10m", "ghi_lag_30m", "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "dhi_lag_60m", "dni_lag_60m", "aws_temp_lag_60m", "aws_rh_lag_60m", "aws_pressure_lag_60m",
    "clp_cot_lag_60m", "clp_cth_lag_60m", "ghi_roll_30m_mean", "ghi_roll_30m_min", "ghi_roll_30m_max",
    "ghi_roll_30m_std", "ghi_roll_60m_mean", "ghi_roll_60m_min", "ghi_roll_60m_max", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_min", "ghi_roll_180m_max", "ghi_roll_180m_std",
    "dhi_roll_180m_mean", "dni_roll_180m_mean", "aws_temp_roll_180m_mean", "aws_rh_roll_180m_mean",
    "aws_ws_roll_180m_mean", "aws_rain_sum_180m", "clp_cot_roll_180m_mean", "clp_cth_roll_180m_mean",
    "ghi_delta_10m", "ghi_delta_60m", "aws_temp_delta_60m", "aws_rh_delta_60m",
    "solar_elev_sin", "solar_elev_sin_clip", "clear_sky_ghi_now", "clear_sky_ghi_target_60m", "kt_now",
    "ghi_to_aws_sr_ratio", "dhi_fraction", "dni_fraction", "diffuse_to_global_ratio", "temp_rh_interaction",
    "vpd_proxy", "wind_u", "wind_v", "clp_cot_delta_60m", "clp_cth_delta_60m", "clp_cot_delta_180m",
    "clp_cth_delta_180m", "ghi_roll_180m_range", "ghi_roll_60m_range", "ghi_ramp_ratio_60m",
    "aws_temp_range", "cloud_opacity_proxy", "cloud_height_temp_interaction"
]

SEQ_FEATURES = [
    "ghi_now", "dhi_now", "dni_now", "solar_elev_deg", "clear_sky_ghi_now", "kt_now",
    "aws_sr_avg_w_m2", "aws_temp_c", "aws_rh_pct", "aws_pressure_hpa", "aws_ws_avg", "aws_rain_mm",
    "clp_cot", "clp_cth_m", "clp_ctt_k", "clp_cloud_present",
    "clp_clear_flag", "clp_thin_cloud_flag", "clp_moderate_cloud_flag", "clp_thick_cloud_flag",
    "synop_rh_pct", "synop_visibility", "hour_sin", "hour_cos", "month_sin", "month_cos"
]


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN environment variable.")
    os.environ["motherduck_token"] = token


def connect_motherduck():
    require_token()
    con = duckdb.connect(database=":memory:")
    con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
    return con


def build_sql():
    return """
    WITH master AS (
        SELECT m.*
        FROM bengkulu_db.bengkulu_sch.bengkulu_master_10min_quality_final m
        WHERE m.ts_wib >= TIMESTAMP '2021-01-01'
          AND m.ts_wib < TIMESTAMP '2026-01-01'
    ), feat AS (
        SELECT
            m.ts_wib,
            m.ts_wib + INTERVAL '60 minutes' AS target_ts_60m,
            LEAD(m.ts_wib, 6) OVER (ORDER BY m.ts_wib) AS observed_target_ts_60m,
            LEAD(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS target_ghi_60m,
            LEAD(m.asrs_solar_elev_deg, 6) OVER (ORDER BY m.ts_wib) AS target_solar_elev_deg_60m,
            m.year, m.month, m.day, m.hour, m.minute,
            SIN(2 * PI() * m.hour / 24.0) AS hour_sin,
            COS(2 * PI() * m.hour / 24.0) AS hour_cos,
            SIN(2 * PI() * m.month / 12.0) AS month_sin,
            COS(2 * PI() * m.month / 12.0) AS month_cos,
            CASE WHEN m.asrs_solar_elev_deg > 0 THEN 1 ELSE 0 END AS daylight_flag,
            CASE WHEN m.asrs_solar_elev_deg > 5 THEN 1 ELSE 0 END AS sun_above_5deg_flag,
            m.asrs_ghi_w_m2 AS ghi_now,
            m.asrs_dhi_w_m2 AS dhi_now,
            m.asrs_dni_w_m2 AS dni_now,
            m.asrs_reflected_w_m2 AS reflected_now,
            m.asrs_nett_rad_w_m2 AS nett_rad_now,
            m.asrs_solar_elev_deg AS solar_elev_deg,
            m.asrs_n_obs_1min, m.asrs_ok_obs,
            m.aws_temp_c, m.aws_temp_min_c, m.aws_temp_max_c, m.aws_rh_pct, m.aws_pressure_hpa,
            m.aws_ws_avg, m.aws_ws_max, m.aws_wd_deg, m.aws_rain_mm, m.aws_sr_avg_w_m2,
            m.clp_cot, m.clp_cth_m, m.clp_ctt_k, m.clp_cer,
            CAST(m.clp_cloud_present AS INT) AS clp_cloud_present,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%clear%' THEN 1 ELSE 0 END AS clp_clear_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%thin%' THEN 1 ELSE 0 END AS clp_thin_cloud_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%moderate%' THEN 1 ELSE 0 END AS clp_moderate_cloud_flag,
            CASE WHEN lower(m.clp_cloud_class) LIKE '%thick%' THEN 1 ELSE 0 END AS clp_thick_cloud_flag,
            m.synop_temp_c, m.synop_dewpoint_c, m.synop_rh_pct, m.synop_wind_speed, m.synop_wind_dir_deg,
            m.synop_visibility, m.synop_rainfall_24h_mm, m.synop_solar_rad_24h,
            m.has_asrs, m.has_aws, m.has_clp, m.has_synop, m.master_qc_status,
            m.asrs_qc_status, m.aws_qc_status, m.clp_qc_status, m.synop_qc_status,
            LAG(m.asrs_ghi_w_m2, 1) OVER (ORDER BY m.ts_wib) AS ghi_lag_10m,
            LAG(m.asrs_ghi_w_m2, 3) OVER (ORDER BY m.ts_wib) AS ghi_lag_30m,
            LAG(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS ghi_lag_60m,
            LAG(m.asrs_ghi_w_m2, 12) OVER (ORDER BY m.ts_wib) AS ghi_lag_120m,
            LAG(m.asrs_ghi_w_m2, 18) OVER (ORDER BY m.ts_wib) AS ghi_lag_180m,
            LAG(m.asrs_dhi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS dhi_lag_60m,
            LAG(m.asrs_dni_w_m2, 6) OVER (ORDER BY m.ts_wib) AS dni_lag_60m,
            LAG(m.aws_temp_c, 6) OVER (ORDER BY m.ts_wib) AS aws_temp_lag_60m,
            LAG(m.aws_rh_pct, 6) OVER (ORDER BY m.ts_wib) AS aws_rh_lag_60m,
            LAG(m.aws_pressure_hpa, 6) OVER (ORDER BY m.ts_wib) AS aws_pressure_lag_60m,
            LAG(m.clp_cot, 6) OVER (ORDER BY m.ts_wib) AS clp_cot_lag_60m,
            LAG(m.clp_cth_m, 6) OVER (ORDER BY m.ts_wib) AS clp_cth_lag_60m,
            AVG(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_mean,
            MIN(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_min,
            MAX(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_max,
            STDDEV_SAMP(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ghi_roll_30m_std,
            AVG(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_mean,
            MIN(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_min,
            MAX(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_max,
            STDDEV_SAMP(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ghi_roll_60m_std,
            AVG(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_mean,
            MIN(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_min,
            MAX(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_max,
            STDDEV_SAMP(m.asrs_ghi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS ghi_roll_180m_std,
            AVG(m.asrs_dhi_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS dhi_roll_180m_mean,
            AVG(m.asrs_dni_w_m2) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS dni_roll_180m_mean,
            AVG(m.aws_temp_c) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_temp_roll_180m_mean,
            AVG(m.aws_rh_pct) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_rh_roll_180m_mean,
            AVG(m.aws_ws_avg) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_ws_roll_180m_mean,
            SUM(COALESCE(m.aws_rain_mm, 0)) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS aws_rain_sum_180m,
            AVG(m.clp_cot) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cot_roll_180m_mean,
            AVG(m.clp_cth_m) OVER (ORDER BY m.ts_wib ROWS BETWEEN 17 PRECEDING AND CURRENT ROW) AS clp_cth_roll_180m_mean,
            m.asrs_ghi_w_m2 - LAG(m.asrs_ghi_w_m2, 1) OVER (ORDER BY m.ts_wib) AS ghi_delta_10m,
            m.asrs_ghi_w_m2 - LAG(m.asrs_ghi_w_m2, 6) OVER (ORDER BY m.ts_wib) AS ghi_delta_60m,
            m.aws_temp_c - LAG(m.aws_temp_c, 6) OVER (ORDER BY m.ts_wib) AS aws_temp_delta_60m,
            m.aws_rh_pct - LAG(m.aws_rh_pct, 6) OVER (ORDER BY m.ts_wib) AS aws_rh_delta_60m
        FROM master m
    )
    SELECT * FROM feat
    WHERE observed_target_ts_60m = target_ts_60m
      AND target_ts_60m < TIMESTAMP '2026-01-01'
      AND target_ghi_60m IS NOT NULL
      AND ghi_now IS NOT NULL
    ORDER BY ts_wib
    """


def add_engineered(df):
    out = df.copy()
    elev_rad = np.deg2rad(out["solar_elev_deg"].astype(float))
    elev_sin = np.sin(elev_rad)
    out["solar_elev_sin"] = elev_sin
    out["solar_elev_sin_clip"] = np.maximum(elev_sin, 0.02)
    out["clear_sky_ghi_now"] = 1100.0 * out["solar_elev_sin_clip"]
    target_sin = np.maximum(np.sin(np.deg2rad(out["target_solar_elev_deg_60m"].astype(float))), 0.02)
    out["clear_sky_ghi_target_60m"] = 1100.0 * target_sin
    out["kt_now"] = out["ghi_now"] / np.maximum(out["clear_sky_ghi_now"], 20.0)
    out["ghi_to_aws_sr_ratio"] = out["ghi_now"] / np.maximum(out["aws_sr_avg_w_m2"], 20.0)
    out["dhi_fraction"] = out["dhi_now"] / np.maximum(out["ghi_now"], 20.0)
    out["dni_fraction"] = out["dni_now"] / np.maximum(out["ghi_now"], 20.0)
    out["diffuse_to_global_ratio"] = out["dhi_now"] / np.maximum(out["ghi_now"], 20.0)
    out["temp_rh_interaction"] = out["aws_temp_c"] * out["aws_rh_pct"]
    out["vpd_proxy"] = out["aws_temp_c"] * (100.0 - out["aws_rh_pct"]) / 100.0
    wd_rad = np.deg2rad(out["aws_wd_deg"].astype(float))
    out["wind_u"] = out["aws_ws_avg"] * np.sin(wd_rad)
    out["wind_v"] = out["aws_ws_avg"] * np.cos(wd_rad)
    out["clp_cot_delta_60m"] = out["clp_cot"] - out["clp_cot_lag_60m"]
    out["clp_cth_delta_60m"] = out["clp_cth_m"] - out["clp_cth_lag_60m"]
    out["clp_cot_delta_180m"] = out["clp_cot"] - out["clp_cot_roll_180m_mean"]
    out["clp_cth_delta_180m"] = out["clp_cth_m"] - out["clp_cth_roll_180m_mean"]
    out["ghi_roll_180m_range"] = out["ghi_roll_180m_max"] - out["ghi_roll_180m_min"]
    out["ghi_roll_60m_range"] = out["ghi_roll_60m_max"] - out["ghi_roll_60m_min"]
    out["ghi_ramp_ratio_60m"] = out["ghi_delta_60m"] / np.maximum(out["ghi_lag_60m"].abs(), 20.0)
    out["aws_temp_range"] = out["aws_temp_max_c"] - out["aws_temp_min_c"]
    out["cloud_opacity_proxy"] = out["clp_cot"] * out["clp_cloud_present"].fillna(0)
    out["cloud_height_temp_interaction"] = out["clp_cth_m"] * out["clp_ctt_k"]
    out[DELTA_COL] = out[TARGET_COL] - out["ghi_now"]
    out["target_ghi_bin"] = pd.cut(out[TARGET_COL], bins=[0, 100, 300, 600, 900, 1400], labels=["0-100", "100-300", "300-600", "600-900", "900+"])
    out["solar_segment"] = pd.cut(out["solar_elev_deg"], bins=[-90, 15, 35, 90], labels=["low", "medium", "high"])
    return out


def ready_mask(df):
    return (
        (df[TARGET_COL].between(0, 1400)) &
        (df["ghi_now"].between(0, 1400)) &
        (df["daylight_flag"] == 1) &
        (df["target_solar_elev_deg_60m"] > 0) &
        (df["has_asrs"] == True) &
        (df["has_aws"] == True) &
        (df["asrs_qc_status"].isin(["ok", "warn_partial_asrs_qc"])) &
        (df["aws_qc_status"] == "ok") &
        (df["ghi_lag_180m"].notna()) &
        (df["ghi_roll_180m_mean"].notna())
    )


def split_masks(df):
    mask = ready_mask(df)
    train = mask & (df[TIME_COL] < pd.Timestamp(TRAIN_END))
    valid = mask & (df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))
    test = mask & (df[TIME_COL] >= pd.Timestamp(VALID_END))
    return train, valid, test


def high_weight(y):
    return 1.0 + 1.0 * (y >= 600) + 2.0 * (y >= 900)


def make_lgbm(weighted=False):
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2400,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.2,
        reg_lambda=2.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", reg)])


class SequenceDataset(Dataset):
    def __init__(self, x_seq, y_delta, ghi_now, target_ghi):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y_delta = torch.tensor(y_delta, dtype=torch.float32).view(-1, 1)
        self.ghi_now = torch.tensor(ghi_now, dtype=torch.float32).view(-1, 1)
        self.target_ghi = torch.tensor(target_ghi, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.y_delta)
    def __getitem__(self, idx):
        return self.x_seq[idx], self.y_delta[idx], self.ghi_now[idx], self.target_ghi[idx]


class LSTMResidual(nn.Module):
    def __init__(self, n_features, hidden_size=96, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class AttentionLSTMResidual(nn.Module):
    def __init__(self, n_features, hidden_size=96, num_layers=1, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)
        return self.head(context)


class TransformerResidual(nn.Module):
    def __init__(self, n_features, d_model=96, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=192, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        z = self.proj(x) + self.pos
        z = self.encoder(z)
        return self.head(z[:, -1, :])


def make_sequences(df, train_mask):
    raw = df[SEQ_FEATURES].copy()
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler_mean = None
    scaler_std = None
    train_vals = imp.fit_transform(raw.loc[train_mask])
    scaler_mean = train_vals.mean(axis=0)
    scaler_std = train_vals.std(axis=0)
    scaler_std[scaler_std == 0] = 1.0
    all_vals = (imp.transform(raw) - scaler_mean) / scaler_std
    ts_vals = df[TIME_COL].values.astype("datetime64[ns]")
    ready = ready_mask(df).values
    y_delta = df[DELTA_COL].values.astype("float32")
    ghi_now = df["ghi_now"].values.astype("float32")
    target = df[TARGET_COL].values.astype("float32")
    ten_min = np.timedelta64(10, "m")
    x_list, y_list, ghi_list, target_list, idx_list = [], [], [], [], []
    for idx in tqdm(range(SEQ_LEN - 1, len(df)), desc="Building sequences"):
        if not ready[idx]:
            continue
        start = idx - SEQ_LEN + 1
        if ts_vals[start] != ts_vals[idx] - ten_min * (SEQ_LEN - 1):
            continue
        if not np.all(np.diff(ts_vals[start:idx + 1]) == ten_min):
            continue
        x_list.append(all_vals[start:idx + 1])
        y_list.append(y_delta[idx])
        ghi_list.append(ghi_now[idx])
        target_list.append(target[idx])
        idx_list.append(idx)
    return np.asarray(x_list, dtype="float32"), np.asarray(y_list, dtype="float32"), np.asarray(ghi_list, dtype="float32"), np.asarray(target_list, dtype="float32"), np.asarray(idx_list, dtype="int64"), imp, scaler_mean, scaler_std


def train_torch_model(model, train_loader, valid_loader, weighted=False):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = np.inf
    best_state = None
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb, _, target in train_loader:
            xb, yb, target = xb.to(DEVICE), yb.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss_raw = torch.nn.functional.smooth_l1_loss(pred, yb, beta=50.0, reduction="none")
            if weighted:
                w = 1.0 + 1.0 * (target >= 600).float() + 2.0 * (target >= 900).float()
                loss = (loss_raw * w).mean()
            else:
                loss = loss_raw.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, _, target in valid_loader:
                xb, yb, target = xb.to(DEVICE), yb.to(DEVICE), target.to(DEVICE)
                pred = model(xb)
                loss_raw = torch.nn.functional.smooth_l1_loss(pred, yb, beta=50.0, reduction="none")
                if weighted:
                    w = 1.0 + 1.0 * (target >= 600).float() + 2.0 * (target >= 900).float()
                    val_loss = (loss_raw * w).mean()
                else:
                    val_loss = loss_raw.mean()
                val_losses.append(val_loss.item())
        val = float(np.mean(val_losses))
        print("epoch " + str(epoch) + " train_loss " + str(round(float(np.mean(losses)), 4)) + " valid_loss " + str(round(val, 4)))
        if val < best_loss:
            best_loss = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


def predict_torch(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _, _, _ in loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy().reshape(-1))
    return np.concatenate(preds) if preds else np.array([])


def metric_row(y, pred, name, persistence_rmse):
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    mbe = float(np.mean(pred - y))
    skill = 1.0 - rmse / persistence_rmse if persistence_rmse > 0 else np.nan
    if name == "persistence":
        skill = 0.0
    return {"model": name, "n_rows": len(y), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def evaluate(pred_df, pred_cols):
    y = pred_df[TARGET_COL].values
    persistence_rmse = float(np.sqrt(mean_squared_error(y, pred_df["persistence"])))
    return pd.DataFrame([metric_row(y, pred_df[c].values, c, persistence_rmse) for c in pred_cols])


def segment_metrics(pred_df, best_col):
    rows = []
    for seg_col in ["target_ghi_bin", "hour", "solar_segment", "month", "has_clp"]:
        for val, group in pred_df.groupby(seg_col, dropna=False):
            if len(group) < 30:
                continue
            persistence_rmse = float(np.sqrt(mean_squared_error(group[TARGET_COL], group["persistence"])))
            for col in ["persistence", "lgbm_residual", best_col]:
                rows.append(metric_row(group[TARGET_COL].values, group[col].values, col, persistence_rmse) | {"segment_col": seg_col, "segment_val": str(val)})
    return pd.DataFrame(rows)


def main():
    print("Device: " + DEVICE)
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading direct 60m residual dataset...")
    df = con.execute(build_sql()).fetchdf()
    con.close()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_TS_COL] = pd.to_datetime(df[TARGET_TS_COL])
    df = add_engineered(df)
    train_mask, valid_mask, test_mask = split_masks(df)
    print("Rows train " + str(int(train_mask.sum())) + " valid " + str(int(valid_mask.sum())) + " test " + str(int(test_mask.sum())))

    print("Training LightGBM residual...")
    lgbm_res = make_lgbm(False)
    lgbm_res.fit(df.loc[train_mask, TABULAR_FEATURES], df.loc[train_mask, DELTA_COL], model__eval_set=[(df.loc[valid_mask, TABULAR_FEATURES], df.loc[valid_mask, DELTA_COL])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(120, verbose=False)])

    print("Training weighted LightGBM residual for high-GHI cases...")
    lgbm_weighted = make_lgbm(True)
    weights = high_weight(df.loc[train_mask, TARGET_COL].values)
    lgbm_weighted.fit(df.loc[train_mask, TABULAR_FEATURES], df.loc[train_mask, DELTA_COL], model__sample_weight=weights, model__eval_set=[(df.loc[valid_mask, TABULAR_FEATURES], df.loc[valid_mask, DELTA_COL])], model__eval_metric="rmse", model__callbacks=[lgb.early_stopping(120, verbose=False)])

    print("Building sequence tensors...")
    x_seq, y_seq, ghi_seq, target_seq, idx_seq, imp, mean, std = make_sequences(df, train_mask)
    seq_df = df.iloc[idx_seq].copy().reset_index(drop=True)
    seq_df["seq_pos"] = np.arange(len(seq_df))
    seq_train = seq_df[TIME_COL] < pd.Timestamp(TRAIN_END)
    seq_valid = (seq_df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (seq_df[TIME_COL] < pd.Timestamp(VALID_END))
    seq_test = seq_df[TIME_COL] >= pd.Timestamp(VALID_END)

    def loader(mask, shuffle=False):
        pos = seq_df.loc[mask, "seq_pos"].values
        return DataLoader(SequenceDataset(x_seq[pos], y_seq[pos], ghi_seq[pos], target_seq[pos]), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = loader(seq_train, True)
    valid_loader = loader(seq_valid, False)
    test_loader = loader(seq_test, False)
    print("Sequence rows train " + str(int(seq_train.sum())) + " valid " + str(int(seq_valid.sum())) + " test " + str(int(seq_test.sum())))

    print("Training LSTM residual...")
    lstm = train_torch_model(LSTMResidual(len(SEQ_FEATURES)), train_loader, valid_loader, weighted=False)

    print("Training Attention-LSTM residual weighted for high-GHI...")
    attn_lstm = train_torch_model(AttentionLSTMResidual(len(SEQ_FEATURES)), train_loader, valid_loader, weighted=True)

    print("Training Transformer residual weighted for high-GHI...")
    transformer = train_torch_model(TransformerResidual(len(SEQ_FEATURES)), train_loader, valid_loader, weighted=True)

    test_df = df.loc[test_mask, [TIME_COL, TARGET_TS_COL, TARGET_COL, "ghi_now", "hour", "month", "solar_segment", "target_ghi_bin", "has_clp", "solar_elev_deg"]].copy()
    test_df["persistence"] = clip_ghi(test_df["ghi_now"].values)
    test_df["lgbm_residual"] = clip_ghi(df.loc[test_mask, "ghi_now"].values + lgbm_res.predict(df.loc[test_mask, TABULAR_FEATURES]))
    test_df["lgbm_weighted_residual"] = clip_ghi(df.loc[test_mask, "ghi_now"].values + lgbm_weighted.predict(df.loc[test_mask, TABULAR_FEATURES]))

    seq_test_df = seq_df.loc[seq_test].copy()
    seq_test_df["lstm_delta"] = predict_torch(lstm, test_loader)
    seq_test_df["attn_lstm_delta"] = predict_torch(attn_lstm, test_loader)
    seq_test_df["transformer_delta"] = predict_torch(transformer, test_loader)
    seq_preds = seq_test_df[[TIME_COL, "lstm_delta", "attn_lstm_delta", "transformer_delta"]].copy()
    test_df = test_df.merge(seq_preds, on=TIME_COL, how="left")
    test_df["lstm_residual"] = clip_ghi(test_df["ghi_now"].values + test_df["lstm_delta"].values)
    test_df["attention_lstm_residual"] = clip_ghi(test_df["ghi_now"].values + test_df["attn_lstm_delta"].values)
    test_df["transformer_residual"] = clip_ghi(test_df["ghi_now"].values + test_df["transformer_delta"].values)
    test_df["ensemble_lgbm_attention"] = clip_ghi(0.5 * test_df["lgbm_residual"].values + 0.5 * test_df["attention_lstm_residual"].values)
    test_df["ensemble_lgbm_transformer"] = clip_ghi(0.5 * test_df["lgbm_residual"].values + 0.5 * test_df["transformer_residual"].values)
    test_df["ensemble_all_residual"] = clip_ghi((test_df["lgbm_residual"].values + test_df["attention_lstm_residual"].values + test_df["transformer_residual"].values) / 3.0)
    test_df["ensemble_weighted_high"] = clip_ghi(0.4 * test_df["lgbm_weighted_residual"].values + 0.3 * test_df["attention_lstm_residual"].values + 0.3 * test_df["transformer_residual"].values)

    pred_cols = [c for c in test_df.columns if c in ["persistence", "lgbm_residual", "lgbm_weighted_residual", "lstm_residual", "attention_lstm_residual", "transformer_residual", "ensemble_lgbm_attention", "ensemble_lgbm_transformer", "ensemble_all_residual", "ensemble_weighted_high"]]
    metrics = evaluate(test_df, pred_cols).sort_values("rmse")
    best = metrics.iloc[0]["model"]
    seg = segment_metrics(test_df, best).sort_values("rmse", ascending=False)
    imp_df = pd.DataFrame({"feature": TABULAR_FEATURES, "importance_lgbm_residual": lgbm_res.named_steps["model"].feature_importances_, "importance_lgbm_weighted": lgbm_weighted.named_steps["model"].feature_importances_}).sort_values("importance_lgbm_residual", ascending=False)

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    seg.to_csv(OUTPUT_DIR / "metrics_by_segment.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)
    imp_df.to_csv(OUTPUT_DIR / "feature_importance_lgbm.csv", index=False)
    joblib.dump({"pipeline": lgbm_res, "features": TABULAR_FEATURES}, MODEL_DIR / "lgbm_residual.joblib")
    joblib.dump({"pipeline": lgbm_weighted, "features": TABULAR_FEATURES}, MODEL_DIR / "lgbm_weighted_residual.joblib")
    torch.save({"state_dict": lstm.state_dict(), "seq_features": SEQ_FEATURES}, MODEL_DIR / "lstm_residual.pt")
    torch.save({"state_dict": attn_lstm.state_dict(), "seq_features": SEQ_FEATURES}, MODEL_DIR / "attention_lstm_residual.pt")
    torch.save({"state_dict": transformer.state_dict(), "seq_features": SEQ_FEATURES}, MODEL_DIR / "transformer_residual.pt")

    print("Best 60m residual/attention models")
    print(metrics.to_string(index=False))
    print("Worst segments for best model: " + best)
    print(seg.head(40).to_string(index=False))

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=metrics, y="model", x="r2")
    plt.axvline(0.9, color="red", linestyle="--", linewidth=1)
    plt.title("60m residual model comparison - test R2")
    plt.xlabel("R2")
    plt.ylabel("")

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=test_df, x=TARGET_COL, y=best, hue="target_ghi_bin", s=8, alpha=0.35)
    plt.plot([0, 1200], [0, 1200], color="black", linewidth=1)
    plt.title("Best model prediction vs actual: " + best)
    plt.xlabel("Actual GHI +60m")
    plt.ylabel("Predicted")

    plt.subplot(2, 2, 3)
    for col in ["persistence", "lgbm_residual", best]:
        sns.kdeplot(test_df[col] - test_df[TARGET_COL], label=col)
    plt.title("Error distribution")
    plt.xlabel("Error")
    plt.legend()

    plt.subplot(2, 2, 4)
    top_imp = imp_df.head(20).sort_values("importance_lgbm_residual")
    plt.barh(top_imp["feature"], top_imp["importance_lgbm_residual"])
    plt.title("LightGBM residual feature importance")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnostics.png", dpi=160)
    plt.close()

    print("Saved outputs:")
    for path in [OUTPUT_DIR / "metrics.csv", OUTPUT_DIR / "metrics_by_segment.csv", OUTPUT_DIR / "predictions_test.csv", OUTPUT_DIR / "feature_importance_lgbm.csv", OUTPUT_DIR / "diagnostics.png"]:
        print(str(path))
    print("Models saved under: " + str(MODEL_DIR))


if __name__ == "__main__":
    main()
