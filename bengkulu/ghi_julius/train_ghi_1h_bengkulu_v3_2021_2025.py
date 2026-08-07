#!/usr/bin/env python3
"""
V3 Bengkulu GHI 1-hour-ahead forecasting benchmark for 2021-2025 only.

Models included:
    1. Persistence baseline
    2. LightGBM direct model
    3. LightGBM residual model
    4. LightGBM blended residual/persistence models
    5. LSTM sequence residual model
    6. Transformer sequence residual model
    7. Ensemble of best LightGBM residual + sequence residual

Important:
    R2=0.90 on 2025 holdout may be unrealistic if the 2025 holdout has distribution shift or incomplete CLP/SYNOP.
    This script is designed to test stronger models fairly against persistence.

Install dependencies:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow lightgbm torch tqdm

Run:
    export MOTHERDUCK_TOKEN="your_token_here"
    python train_ghi_1h_bengkulu_v3_models.py

Windows PowerShell:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal
    python train_ghi_1h_bengkulu_v3_models.py

Outputs:
    outputs_v3_2021_2025/ghi_1h_v3_metrics.csv
    outputs_v3_2021_2025/ghi_1h_v3_metrics_by_hour.csv
    outputs_v3_2021_2025/ghi_1h_v3_predictions_test.csv
    outputs_v3_2021_2025/ghi_1h_v3_diagnostics.png
    outputs_v3_2021_2025/lgbm_direct_model.joblib
    outputs_v3_2021_2025/lgbm_residual_model.joblib
    outputs_v3_2021_2025/lstm_residual_model.pt
    outputs_v3_2021_2025/transformer_residual_model.pt
"""

import os
from pathlib import Path
import warnings

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError("LightGBM is required. Install with: pip install lightgbm") from exc

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    raise RuntimeError("PyTorch is required. Install with: pip install torch") from exc

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
TABLE_NAME = "ghi_forecast_1h_train_3h_rollback_2021_2025"
OUTPUT_DIR = Path("outputs_v3_2021_2025")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_TIME_COL = "target_ts_wib"
TARGET_COL = "target_ghi_1h_ahead"
DELTA_TARGET_COL = "target_delta_ghi_1h"
PRED_MIN = 0.0
PRED_MAX = 1400.0

TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42
SEQ_LEN = 19
BATCH_SIZE = 512
MAX_EPOCHS = 60
PATIENCE = 8

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TABULAR_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "daylight_flag", "sun_above_5deg_flag",
    "ghi_now", "dhi_now", "dni_now", "reflected_now", "nett_rad_now", "solar_elev_deg",
    "asrs_n_obs_1min", "asrs_ok_obs",
    "aws_temp_c", "aws_temp_min_c", "aws_temp_max_c", "aws_rh_pct", "aws_pressure_hpa",
    "aws_ws_avg", "aws_ws_max", "aws_wd_deg", "aws_rain_mm", "aws_sr_avg_w_m2",
    "clp_cot", "clp_cth_m", "clp_ctt_k", "clp_cer", "clp_cloud_present",
    "clp_clear_flag", "clp_thin_cloud_flag", "clp_moderate_cloud_flag", "clp_thick_cloud_flag",
    "synop_temp_c", "synop_dewpoint_c", "synop_rh_pct", "synop_wind_speed",
    "synop_wind_dir_deg", "synop_visibility", "synop_rainfall_24h_mm", "synop_solar_rad_24h",
    "ghi_lag_10m", "ghi_lag_30m", "ghi_lag_60m", "ghi_lag_120m", "ghi_lag_180m",
    "dhi_lag_60m", "dni_lag_60m",
    "aws_temp_lag_60m", "aws_rh_lag_60m", "aws_pressure_lag_60m",
    "clp_cot_lag_60m", "clp_cth_lag_60m",
    "ghi_roll_30m_mean", "ghi_roll_30m_min", "ghi_roll_30m_max", "ghi_roll_30m_std",
    "ghi_roll_60m_mean", "ghi_roll_60m_min", "ghi_roll_60m_max", "ghi_roll_60m_std",
    "ghi_roll_180m_mean", "ghi_roll_180m_min", "ghi_roll_180m_max", "ghi_roll_180m_std",
    "dhi_roll_180m_mean", "dni_roll_180m_mean",
    "aws_temp_roll_180m_mean", "aws_rh_roll_180m_mean", "aws_ws_roll_180m_mean", "aws_rain_sum_180m",
    "clp_cot_roll_180m_mean", "clp_cth_roll_180m_mean",
    "ghi_delta_10m", "ghi_delta_60m", "aws_temp_delta_60m", "aws_rh_delta_60m"
]

SEQ_FEATURES = [
    "ghi_now", "dhi_now", "dni_now", "solar_elev_deg",
    "aws_sr_avg_w_m2", "aws_temp_c", "aws_rh_pct", "aws_pressure_hpa",
    "aws_ws_avg", "aws_rain_mm", "clp_cot", "clp_cth_m", "clp_ctt_k",
    "clp_cloud_present", "synop_rh_pct", "synop_visibility",
    "hour_sin", "hour_cos", "month_sin", "month_cos"
]


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError("Missing MOTHERDUCK_TOKEN environment variable.")
    os.environ["motherduck_token"] = token
    return token


def connect_motherduck():
    require_token()
    con = duckdb.connect(database=":memory:")
    con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
    return con


def load_data(con):
    cols = [TIME_COL, TARGET_TIME_COL, TARGET_COL, "is_model_ready", "has_continuous_3h_history"]
    cols = cols + sorted(set(TABULAR_FEATURES + SEQ_FEATURES))
    sql_text = """
    SELECT
        """ + ",\n        ".join(cols) + """
    FROM """ + ATTACH_ALIAS + "." + SCHEMA_NAME + "." + TABLE_NAME + """
    WHERE observed_target_ts_wib = target_ts_wib
      AND target_ghi_1h_ahead IS NOT NULL
      AND ghi_now IS NOT NULL
    ORDER BY ts_wib
    """
    df = con.execute(sql_text).fetchdf()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_TIME_COL] = pd.to_datetime(df[TARGET_TIME_COL])
    df[DELTA_TARGET_COL] = df[TARGET_COL] - df["ghi_now"]
    df["hour"] = df[TIME_COL].dt.hour
    df["model_ready_mask"] = (
        (df["is_model_ready"] == 1) &
        (df["has_continuous_3h_history"] == 1) &
        (df[TARGET_COL].between(0, 1400)) &
        (df["ghi_now"].between(0, 1400))
    )
    return df


def split_masks(df):
    train_mask = df[TIME_COL] < pd.Timestamp(TRAIN_END)
    valid_mask = (df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))
    test_mask = df[TIME_COL] >= pd.Timestamp(VALID_END)
    ready = df["model_ready_mask"]
    return train_mask & ready, valid_mask & ready, test_mask & ready


def build_lgbm(kind="direct"):
    if kind == "residual":
        reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=2500,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=47,
            min_child_samples=50,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", reg)
    ])
    return pipe


class SequenceDataset(Dataset):
    def __init__(self, x_seq, y_delta, ghi_now):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y_delta = torch.tensor(y_delta, dtype=torch.float32).view(-1, 1)
        self.ghi_now = torch.tensor(ghi_now, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.y_delta)
    def __getitem__(self, idx):
        return self.x_seq[idx], self.y_delta[idx], self.ghi_now[idx]


class LSTMResidual(nn.Module):
    def __init__(self, n_features, hidden_size=96, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerResidual(nn.Module):
    def __init__(self, n_features, d_model=96, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=192,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        z = self.proj(x) + self.pos
        z = self.encoder(z)
        return self.head(z[:, -1, :])


def make_sequences(df):
    seq_raw = df[SEQ_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_mask, valid_mask, test_mask = split_masks(df)
    imputer.fit(seq_raw.loc[train_mask])
    scaled_all = scaler.fit_transform(imputer.transform(seq_raw))

    ts_vals = df[TIME_COL].values.astype("datetime64[ns]")
    y_delta = df[DELTA_TARGET_COL].values.astype("float32")
    ghi_now = df["ghi_now"].values.astype("float32")
    ready = df["model_ready_mask"].values

    x_list = []
    y_list = []
    ghi_list = []
    idx_list = []
    ten_min = np.timedelta64(10, "m")
    for idx in tqdm(range(SEQ_LEN - 1, len(df)), desc="Building sequences"):
        if not ready[idx]:
            continue
        start_idx = idx - SEQ_LEN + 1
        expected_start = ts_vals[idx] - ten_min * (SEQ_LEN - 1)
        if ts_vals[start_idx] != expected_start:
            continue
        if not np.all(np.diff(ts_vals[start_idx:idx + 1]) == ten_min):
            continue
        x_list.append(scaled_all[start_idx:idx + 1])
        y_list.append(y_delta[idx])
        ghi_list.append(ghi_now[idx])
        idx_list.append(idx)
    x_seq = np.asarray(x_list, dtype="float32")
    y_arr = np.asarray(y_list, dtype="float32")
    ghi_arr = np.asarray(ghi_list, dtype="float32")
    idx_arr = np.asarray(idx_list, dtype="int64")
    return x_seq, y_arr, ghi_arr, idx_arr, imputer, scaler


def train_torch_model(model, train_loader, valid_loader):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=50.0)
    best_loss = np.inf
    best_state = None
    patience_count = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb, yb, _ in valid_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                valid_losses.append(loss_fn(pred, yb).item())
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float(np.mean(train_losses))
        print("epoch " + str(epoch) + " train_loss " + str(round(float(np.mean(train_losses)), 4)) + " valid_loss " + str(round(valid_loss, 4)))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch_delta(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds) if preds else np.array([])


def metric_row(y_true, y_pred, model_name, persistence_rmse=None):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mbe = float(np.mean(y_pred - y_true))
    skill = 0.0 if model_name.endswith("persistence") else np.nan
    if persistence_rmse and persistence_rmse > 0:
        skill = 1.0 - rmse / persistence_rmse
    return {"model": model_name, "n_rows": len(y_true), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def evaluate_predictions(y_true, pred_dict, prefix):
    rows = []
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, pred_dict["persistence"])))
    for name, pred in pred_dict.items():
        rows.append(metric_row(y_true, pred, prefix + "_" + name, persistence_rmse))
    return pd.DataFrame(rows)


def main():
    print("Device: " + DEVICE)
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading data once from MotherDuck...")
    df = load_data(con)
    con.close()
    print("Rows loaded: " + str(len(df)))
    print("Ready rows: " + str(int(df["model_ready_mask"].sum())))

    train_mask, valid_mask, test_mask = split_masks(df)
    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    test_df = df.loc[test_mask].copy()
    print("Train rows: " + str(len(train_df)))
    print("Valid rows: " + str(len(valid_df)))
    print("Test rows: " + str(len(test_df)))
    print("Split policy: train=2021-2023, validation=2024, test=2025; 2026 is excluded.")

    print("Training LightGBM direct...")
    lgbm_direct = build_lgbm("direct")
    lgbm_direct.fit(
        train_df[TABULAR_FEATURES], train_df[TARGET_COL],
        model__eval_set=[(valid_df[TABULAR_FEATURES], valid_df[TARGET_COL])],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    print("Training LightGBM residual...")
    lgbm_residual = build_lgbm("residual")
    lgbm_residual.fit(
        train_df[TABULAR_FEATURES], train_df[DELTA_TARGET_COL],
        model__eval_set=[(valid_df[TABULAR_FEATURES], valid_df[DELTA_TARGET_COL])],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    def lgbm_preds(part_df):
        persistence = clip_ghi(part_df["ghi_now"].values)
        direct = clip_ghi(lgbm_direct.predict(part_df[TABULAR_FEATURES]))
        residual = clip_ghi(part_df["ghi_now"].values + lgbm_residual.predict(part_df[TABULAR_FEATURES]))
        return {
            "persistence": persistence,
            "lgbm_direct": direct,
            "lgbm_residual": residual,
            "blend_70persistence_30_lgbm_residual": clip_ghi(0.7 * persistence + 0.3 * residual),
            "blend_50persistence_50_lgbm_residual": clip_ghi(0.5 * persistence + 0.5 * residual),
            "blend_30persistence_70_lgbm_residual": clip_ghi(0.3 * persistence + 0.7 * residual),
        }

    metrics = []
    lgbm_pred_parts = {}
    for split_name, part_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        preds = lgbm_preds(part_df)
        lgbm_pred_parts[split_name] = preds
        metrics.append(evaluate_predictions(part_df[TARGET_COL].values, preds, split_name))
    metrics_df = pd.concat(metrics, ignore_index=True)
    print(metrics_df.to_string(index=False))

    print("Building sequence tensors...")
    x_seq, y_seq, ghi_seq, idx_seq, seq_imputer, seq_scaler = make_sequences(df)
    seq_df = df.iloc[idx_seq].copy().reset_index(drop=True)
    seq_df["seq_pos"] = np.arange(len(seq_df))
    seq_train_mask = seq_df[TIME_COL] < pd.Timestamp(TRAIN_END)
    seq_valid_mask = (seq_df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (seq_df[TIME_COL] < pd.Timestamp(VALID_END))
    seq_test_mask = seq_df[TIME_COL] >= pd.Timestamp(VALID_END)

    def make_loader(mask, shuffle=False):
        pos = seq_df.loc[mask, "seq_pos"].values
        ds = SequenceDataset(x_seq[pos], y_seq[pos], ghi_seq[pos])
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(seq_train_mask, shuffle=True)
    valid_loader = make_loader(seq_valid_mask, shuffle=False)
    test_loader = make_loader(seq_test_mask, shuffle=False)
    print("Sequence train rows: " + str(int(seq_train_mask.sum())))
    print("Sequence valid rows: " + str(int(seq_valid_mask.sum())))
    print("Sequence test rows: " + str(int(seq_test_mask.sum())))

    print("Training LSTM residual...")
    lstm_model = LSTMResidual(n_features=len(SEQ_FEATURES))
    lstm_model = train_torch_model(lstm_model, train_loader, valid_loader)

    print("Training Transformer residual...")
    transformer_model = TransformerResidual(n_features=len(SEQ_FEATURES))
    transformer_model = train_torch_model(transformer_model, train_loader, valid_loader)

    seq_test_df = seq_df.loc[seq_test_mask].copy()
    seq_test_pos = seq_test_df["seq_pos"].values
    y_test_seq = seq_test_df[TARGET_COL].values
    ghi_test_seq = seq_test_df["ghi_now"].values
    persistence_seq = clip_ghi(ghi_test_seq)

    lstm_delta = predict_torch_delta(lstm_model, test_loader)
    transformer_delta = predict_torch_delta(transformer_model, test_loader)
    lstm_pred = clip_ghi(ghi_test_seq + lstm_delta)
    transformer_pred = clip_ghi(ghi_test_seq + transformer_delta)

    seq_lgbm_residual_pred = clip_ghi(seq_test_df["ghi_now"].values + lgbm_residual.predict(seq_test_df[TABULAR_FEATURES]))
    ensemble_lgbm_lstm = clip_ghi(0.5 * seq_lgbm_residual_pred + 0.5 * lstm_pred)
    ensemble_lgbm_transformer = clip_ghi(0.5 * seq_lgbm_residual_pred + 0.5 * transformer_pred)
    ensemble_all = clip_ghi((seq_lgbm_residual_pred + lstm_pred + transformer_pred) / 3.0)
    blend_best_safe = clip_ghi(0.5 * persistence_seq + 0.5 * ensemble_all)

    seq_preds = {
        "persistence": persistence_seq,
        "lgbm_residual_on_seq_rows": seq_lgbm_residual_pred,
        "lstm_residual": lstm_pred,
        "transformer_residual": transformer_pred,
        "ensemble_lgbm_lstm": ensemble_lgbm_lstm,
        "ensemble_lgbm_transformer": ensemble_lgbm_transformer,
        "ensemble_lgbm_lstm_transformer": ensemble_all,
        "blend_50persistence_50_ensemble_all": blend_best_safe,
    }
    seq_metrics_df = evaluate_predictions(y_test_seq, seq_preds, "testseq")
    metrics_df = pd.concat([metrics_df, seq_metrics_df], ignore_index=True)
    print(seq_metrics_df.to_string(index=False))

    print("Best test rows by RMSE:")
    print(metrics_df[metrics_df["model"].str.contains("test")].sort_values("rmse").head(20).to_string(index=False))

    print("Computing hourly test metrics for best candidates...")
    pred_test_df = test_df[[TIME_COL, TARGET_TIME_COL, TARGET_COL, "ghi_now", "hour"]].copy()
    test_preds = lgbm_pred_parts["test"]
    for key, val in test_preds.items():
        pred_test_df[key] = val
    hourly_rows = []
    for hour_val, group in pred_test_df.groupby("hour"):
        y_true = group[TARGET_COL].values
        persistence_rmse = float(np.sqrt(mean_squared_error(y_true, group["persistence"].values)))
        for model_name in test_preds.keys():
            hourly_rows.append(metric_row(y_true, group[model_name].values, model_name, persistence_rmse) | {"hour": hour_val})
    hourly_df = pd.DataFrame(hourly_rows)

    metrics_path = OUTPUT_DIR / "ghi_1h_v3_metrics.csv"
    hourly_path = OUTPUT_DIR / "ghi_1h_v3_metrics_by_hour.csv"
    pred_path = OUTPUT_DIR / "ghi_1h_v3_predictions_test.csv"
    plot_path = OUTPUT_DIR / "ghi_1h_v3_diagnostics.png"
    metrics_df.to_csv(metrics_path, index=False)
    hourly_df.to_csv(hourly_path, index=False)

    pred_test_df.to_csv(pred_path, index=False)
    joblib.dump({"pipeline": lgbm_direct, "features": TABULAR_FEATURES}, OUTPUT_DIR / "lgbm_direct_model.joblib")
    joblib.dump({"pipeline": lgbm_residual, "features": TABULAR_FEATURES, "target": DELTA_TARGET_COL}, OUTPUT_DIR / "lgbm_residual_model.joblib")
    torch.save({"state_dict": lstm_model.state_dict(), "seq_features": SEQ_FEATURES}, OUTPUT_DIR / "lstm_residual_model.pt")
    torch.save({"state_dict": transformer_model.state_dict(), "seq_features": SEQ_FEATURES}, OUTPUT_DIR / "transformer_residual_model.pt")

    plt.figure(figsize=(15, 10))
    best_test = metrics_df[metrics_df["model"].str.contains("test")].sort_values("rmse").iloc[0]["model"]
    plt.subplot(2, 2, 1)
    compare_cols = ["persistence", "lgbm_residual", "blend_50persistence_50_lgbm_residual"]
    for col in compare_cols:
        plt.scatter(pred_test_df[TARGET_COL], pred_test_df[col], s=5, alpha=0.18, label=col)
    plt.plot([0, 1200], [0, 1200], color="black", linewidth=1)
    plt.xlabel("Actual GHI 1h ahead")
    plt.ylabel("Predicted GHI")
    plt.title("Tabular Test Predictions")
    plt.legend()

    plt.subplot(2, 2, 2)
    for col in compare_cols:
        sns.kdeplot(pred_test_df[col] - pred_test_df[TARGET_COL], label=col)
    plt.title("Error Distribution")
    plt.xlabel("Error")
    plt.legend()

    plt.subplot(2, 2, 3)
    plot_df = pred_test_df.tail(min(800, len(pred_test_df)))
    plt.plot(plot_df[TARGET_TIME_COL], plot_df[TARGET_COL], label="actual", linewidth=1)
    plt.plot(plot_df[TARGET_TIME_COL], plot_df["persistence"], label="persistence", linewidth=1, alpha=0.7)
    plt.plot(plot_df[TARGET_TIME_COL], plot_df["blend_50persistence_50_lgbm_residual"], label="blend50_lgbm_residual", linewidth=1)
    plt.xticks(rotation=30)
    plt.title("Recent Test Period")
    plt.legend()

    plt.subplot(2, 2, 4)
    lgbm_importance = pd.DataFrame({
        "feature": TABULAR_FEATURES,
        "importance": lgbm_residual.named_steps["model"].feature_importances_
    }).sort_values("importance", ascending=False).head(18).sort_values("importance")
    plt.barh(lgbm_importance["feature"], lgbm_importance["importance"])
    plt.title("LightGBM Residual Feature Importance")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("Saved outputs:")
    for path in [metrics_path, hourly_path, pred_path, plot_path]:
        print(str(path))
    print(str(OUTPUT_DIR / "lgbm_direct_model.joblib"))
    print(str(OUTPUT_DIR / "lgbm_residual_model.joblib"))
    print(str(OUTPUT_DIR / "lstm_residual_model.pt"))
    print(str(OUTPUT_DIR / "transformer_residual_model.pt"))


if __name__ == "__main__":
    main()
