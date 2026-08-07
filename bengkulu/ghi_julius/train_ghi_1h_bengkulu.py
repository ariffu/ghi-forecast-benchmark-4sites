#!/usr/bin/env python3
"""
Train a 1-hour-ahead GHI forecast model from MotherDuck Bengkulu training dataset.

How to run in VS Code terminal:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow
    export MOTHERDUCK_TOKEN="your_token_here"
    python train_ghi_1h_bengkulu.py

On Windows PowerShell:
    setx MOTHERDUCK_TOKEN "your_token_here"
    python train_ghi_1h_bengkulu.py

Outputs:
    ./outputs/ghi_1h_model.joblib
    ./outputs/ghi_1h_feature_importance.csv
    ./outputs/ghi_1h_predictions_test.csv
    ./outputs/ghi_1h_diagnostics.png
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
TABLE_NAME = "ghi_forecast_1h_train_3h_rollback"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "target_ghi_1h_ahead"
TIME_COL = "ts_wib"
TARGET_TIME_COL = "target_ts_wib"

TRAIN_END = "2025-01-01"
VALID_END = "2026-01-01"

RANDOM_STATE = 42

FEATURE_COLS = [
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


def require_token():
    token = os.getenv("MOTHERDUCK_TOKEN") or os.getenv("motherduck_token")
    if not token:
        raise RuntimeError(
            "Missing MotherDuck token. Set MOTHERDUCK_TOKEN first. "
            "Example macOS/Linux: export MOTHERDUCK_TOKEN='...'"
        )
    os.environ["motherduck_token"] = token
    return token


def connect_motherduck():
    require_token()
    con = duckdb.connect(database=":memory:")
    con.execute("ATTACH 'md:" + DB_NAME + "' AS " + ATTACH_ALIAS)
    return con


def load_training_data(con):
    select_cols = [TIME_COL, TARGET_TIME_COL, TARGET_COL, "is_model_ready", "has_continuous_3h_history"] + FEATURE_COLS
    select_sql = ",\n        ".join(select_cols)
    full_table = ATTACH_ALIAS + "." + SCHEMA_NAME + "." + TABLE_NAME
    sql_text = """
    SELECT
        """ + select_sql + """
    FROM """ + full_table + """
    WHERE is_model_ready = 1
      AND has_continuous_3h_history = 1
      AND """ + TARGET_COL + """ BETWEEN 0 AND 1400
      AND ghi_now BETWEEN 0 AND 1400
    ORDER BY """ + TIME_COL + """
    """
    df = con.execute(sql_text).fetchdf()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_TIME_COL] = pd.to_datetime(df[TARGET_TIME_COL])
    return df


def temporal_split(df):
    train_df = df[df[TIME_COL] < pd.Timestamp(TRAIN_END)].copy()
    valid_df = df[(df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))].copy()
    test_df = df[df[TIME_COL] >= pd.Timestamp(VALID_END)].copy()
    return train_df, valid_df, test_df


def build_model():
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[("num", numeric_preprocess, FEATURE_COLS)],
        remainder="drop"
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


def metric_frame(y_true, y_pred, label):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    mbe_val = np.mean(y_pred - y_true)
    return {
        "split": label,
        "n_rows": len(y_true),
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_val,
        "mbe": mbe_val
    }


def evaluate_model(pipe, train_df, valid_df, test_df):
    rows = []
    predictions = {}
    for label, part_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if len(part_df) == 0:
            continue
        x_part = part_df[FEATURE_COLS]
        y_part = part_df[TARGET_COL]
        y_pred = pipe.predict(x_part)
        y_persist = part_df["ghi_now"].values
        rows.append(metric_frame(y_part, y_pred, label + "_model"))
        rows.append(metric_frame(y_part, y_persist, label + "_persistence"))
        predictions[label] = y_pred
    metrics_df = pd.DataFrame(rows)
    return metrics_df, predictions


def permutation_importance_fast(pipe, test_df, max_rows=20000):
    if len(test_df) == 0:
        return pd.DataFrame(columns=["feature", "importance_rmse_increase"])
    sample_df = test_df.sample(n=min(max_rows, len(test_df)), random_state=RANDOM_STATE).copy()
    x_base = sample_df[FEATURE_COLS].copy()
    y_true = sample_df[TARGET_COL].values
    base_pred = pipe.predict(x_base)
    base_rmse = np.sqrt(mean_squared_error(y_true, base_pred))
    importances = []
    rng = np.random.default_rng(RANDOM_STATE)
    for col in FEATURE_COLS:
        x_perm = x_base.copy()
        x_perm[col] = rng.permutation(x_perm[col].values)
        perm_pred = pipe.predict(x_perm)
        perm_rmse = np.sqrt(mean_squared_error(y_true, perm_pred))
        importances.append({"feature": col, "importance_rmse_increase": perm_rmse - base_rmse})
    importance_df = pd.DataFrame(importances).sort_values("importance_rmse_increase", ascending=False)
    return importance_df


def save_outputs(pipe, metrics_df, importance_df, test_df, test_pred):
    model_path = OUTPUT_DIR / "ghi_1h_model.joblib"
    metrics_path = OUTPUT_DIR / "ghi_1h_metrics.csv"
    importance_path = OUTPUT_DIR / "ghi_1h_feature_importance.csv"
    pred_path = OUTPUT_DIR / "ghi_1h_predictions_test.csv"
    plot_path = OUTPUT_DIR / "ghi_1h_diagnostics.png"

    joblib.dump({"pipeline": pipe, "feature_cols": FEATURE_COLS, "target_col": TARGET_COL}, model_path)
    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    if len(test_df) > 0:
        pred_df = test_df[[TIME_COL, TARGET_TIME_COL, TARGET_COL, "ghi_now"]].copy()
        pred_df["prediction_ghi_1h"] = test_pred
        pred_df["persistence_prediction"] = pred_df["ghi_now"]
        pred_df["error_model"] = pred_df["prediction_ghi_1h"] - pred_df[TARGET_COL]
        pred_df["error_persistence"] = pred_df["persistence_prediction"] - pred_df[TARGET_COL]
        pred_df.to_csv(pred_path, index=False)

        plt.figure(figsize=(13, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(pred_df[TARGET_COL], pred_df["prediction_ghi_1h"], s=5, alpha=0.25)
        plt.plot([0, 1200], [0, 1200], color="red", linewidth=1)
        plt.xlabel("Actual GHI 1h ahead")
        plt.ylabel("Predicted GHI")
        plt.title("Prediction vs Actual")

        plt.subplot(2, 2, 2)
        sns.histplot(pred_df["error_model"], bins=60, kde=True)
        plt.xlabel("Prediction error")
        plt.title("Model Error Distribution")

        plt.subplot(2, 2, 3)
        plot_df = pred_df.tail(min(800, len(pred_df)))
        plt.plot(plot_df[TARGET_TIME_COL], plot_df[TARGET_COL], label="actual", linewidth=1)
        plt.plot(plot_df[TARGET_TIME_COL], plot_df["prediction_ghi_1h"], label="model", linewidth=1)
        plt.plot(plot_df[TARGET_TIME_COL], plot_df["persistence_prediction"], label="persistence", linewidth=1, alpha=0.7)
        plt.xticks(rotation=30)
        plt.ylabel("GHI")
        plt.title("Recent Test Period")
        plt.legend()

        plt.subplot(2, 2, 4)
        top_imp = importance_df.head(15).sort_values("importance_rmse_increase")
        plt.barh(top_imp["feature"], top_imp["importance_rmse_increase"])
        plt.xlabel("RMSE increase after permutation")
        plt.title("Top Feature Importance")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
    return model_path, metrics_path, importance_path, pred_path, plot_path


def main():
    print("Connecting to MotherDuck...")
    con = connect_motherduck()

    print("Loading model-ready training rows...")
    df = load_training_data(con)
    print("Loaded rows: " + str(len(df)))
    print("Date range: " + str(df[TIME_COL].min()) + " to " + str(df[TIME_COL].max()))

    train_df, valid_df, test_df = temporal_split(df)
    print("Train rows: " + str(len(train_df)))
    print("Valid rows: " + str(len(valid_df)))
    print("Test rows: " + str(len(test_df)))

    if len(train_df) == 0:
        raise RuntimeError("No training rows found before TRAIN_END. Adjust split dates.")

    pipe = build_model()
    print("Training model...")
    pipe.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    print("Evaluating model and persistence baseline...")
    metrics_df, predictions = evaluate_model(pipe, train_df, valid_df, test_df)
    print(metrics_df.to_string(index=False))

    print("Computing permutation importance on test sample...")
    importance_df = permutation_importance_fast(pipe, test_df)
    print(importance_df.head(20).to_string(index=False))

    test_pred = predictions.get("test", np.array([]))
    paths = save_outputs(pipe, metrics_df, importance_df, test_df, test_pred)
    print("Saved outputs:")
    for path_val in paths:
        print(str(path_val))

    con.close()


if __name__ == "__main__":
    main()
