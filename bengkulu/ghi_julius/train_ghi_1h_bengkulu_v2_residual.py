#!/usr/bin/env python3
"""
Train robust 1-hour-ahead GHI forecast models from MotherDuck Bengkulu dataset.

V2 changes:
    1. Trains a direct model for target_ghi_1h_ahead.
    2. Trains a residual model for target_delta_ghi_1h = target_ghi_1h_ahead - ghi_now.
    3. Final residual prediction = ghi_now + predicted_delta.
    4. Clips final predictions to physical range 0-1400 W/m2.
    5. Compares against persistence baseline and reports skill score.
    6. Produces diagnostics by split and by hour.

How to run in VS Code terminal:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow
    export MOTHERDUCK_TOKEN="your_token_here"
    python train_ghi_1h_bengkulu_v2_residual.py

Windows PowerShell:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal after setx
    python train_ghi_1h_bengkulu_v2_residual.py

Outputs:
    outputs_v2/ghi_1h_direct_model.joblib
    outputs_v2/ghi_1h_residual_model.joblib
    outputs_v2/ghi_1h_v2_metrics.csv
    outputs_v2/ghi_1h_v2_metrics_by_hour.csv
    outputs_v2/ghi_1h_v2_feature_importance_residual.csv
    outputs_v2/ghi_1h_v2_predictions_test.csv
    outputs_v2/ghi_1h_v2_diagnostics.png
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
TABLE_NAME = "ghi_forecast_1h_train_3h_rollback"
OUTPUT_DIR = Path("outputs_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_TIME_COL = "target_ts_wib"
TARGET_COL = "target_ghi_1h_ahead"
DELTA_TARGET_COL = "target_delta_ghi_1h"
PRED_MIN = 0.0
PRED_MAX = 1400.0

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
    df[DELTA_TARGET_COL] = df[TARGET_COL] - df["ghi_now"]
    return df


def temporal_split(df):
    train_df = df[df[TIME_COL] < pd.Timestamp(TRAIN_END)].copy()
    valid_df = df[(df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))].copy()
    test_df = df[df[TIME_COL] >= pd.Timestamp(VALID_END)].copy()
    return train_df, valid_df, test_df


def build_hgb_model(target_type="direct"):
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
    if target_type == "residual":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=350,
            max_leaf_nodes=23,
            min_samples_leaf=40,
            l2_regularization=0.25,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def clip_ghi(values):
    return np.clip(values, PRED_MIN, PRED_MAX)


def metric_row(y_true, y_pred, label, persistence_rmse=None):
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_val = float(mean_absolute_error(y_true, y_pred))
    r2_val = float(r2_score(y_true, y_pred))
    mbe_val = float(np.mean(y_pred - y_true))
    skill_val = np.nan
    if persistence_rmse and persistence_rmse > 0:
        skill_val = 1.0 - rmse_val / persistence_rmse
    return {
        "model": label,
        "n_rows": len(y_true),
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_val,
        "mbe": mbe_val,
        "skill_vs_persistence": skill_val
    }


def predict_all(direct_pipe, residual_pipe, part_df):
    x_part = part_df[FEATURE_COLS]
    y_true = part_df[TARGET_COL].values
    pred_persistence = clip_ghi(part_df["ghi_now"].values)
    pred_direct = clip_ghi(direct_pipe.predict(x_part))
    pred_delta = residual_pipe.predict(x_part)
    pred_residual = clip_ghi(part_df["ghi_now"].values + pred_delta)
    pred_blend_7030 = clip_ghi(0.7 * pred_persistence + 0.3 * pred_residual)
    pred_blend_5050 = clip_ghi(0.5 * pred_persistence + 0.5 * pred_residual)
    return {
        "actual": y_true,
        "persistence": pred_persistence,
        "direct_model": pred_direct,
        "residual_model": pred_residual,
        "blend_70persistence_30residual": pred_blend_7030,
        "blend_50persistence_50residual": pred_blend_5050,
    }


def evaluate_split(direct_pipe, residual_pipe, part_df, split_name):
    preds = predict_all(direct_pipe, residual_pipe, part_df)
    y_true = preds["actual"]
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, preds["persistence"])))
    rows = []
    for model_name in ["persistence", "direct_model", "residual_model", "blend_70persistence_30residual", "blend_50persistence_50residual"]:
        row = metric_row(y_true, preds[model_name], split_name + "_" + model_name, persistence_rmse)
        rows.append(row)
    return pd.DataFrame(rows), preds


def evaluate_all(direct_pipe, residual_pipe, train_df, valid_df, test_df):
    metric_parts = []
    pred_parts = {}
    for split_name, part_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if len(part_df) == 0:
            continue
        metrics_df, preds = evaluate_split(direct_pipe, residual_pipe, part_df, split_name)
        metric_parts.append(metrics_df)
        pred_parts[split_name] = preds
    return pd.concat(metric_parts, ignore_index=True), pred_parts


def metrics_by_hour(test_df, test_preds):
    rows = []
    tmp_df = test_df[[TIME_COL, TARGET_COL, "hour"]].copy()
    tmp_df["actual"] = test_preds["actual"]
    tmp_df["persistence"] = test_preds["persistence"]
    tmp_df["direct_model"] = test_preds["direct_model"]
    tmp_df["residual_model"] = test_preds["residual_model"]
    tmp_df["blend_70persistence_30residual"] = test_preds["blend_70persistence_30residual"]
    tmp_df["blend_50persistence_50residual"] = test_preds["blend_50persistence_50residual"]
    for hour_val, group_df in tmp_df.groupby("hour"):
        y_true = group_df["actual"].values
        persistence_rmse = float(np.sqrt(mean_squared_error(y_true, group_df["persistence"].values)))
        for model_name in ["persistence", "direct_model", "residual_model", "blend_70persistence_30residual", "blend_50persistence_50residual"]:
            row = metric_row(y_true, group_df[model_name].values, model_name, persistence_rmse)
            row["hour"] = hour_val
            rows.append(row)
    return pd.DataFrame(rows)


def permutation_importance_fast(pipe, test_df, target_col, prediction_mode="residual", max_rows=20000):
    if len(test_df) == 0:
        return pd.DataFrame(columns=["feature", "importance_rmse_increase"])
    sample_df = test_df.sample(n=min(max_rows, len(test_df)), random_state=RANDOM_STATE).copy()
    x_base = sample_df[FEATURE_COLS].copy()
    y_true = sample_df[TARGET_COL].values
    if prediction_mode == "residual":
        base_pred = clip_ghi(sample_df["ghi_now"].values + pipe.predict(x_base))
    else:
        base_pred = clip_ghi(pipe.predict(x_base))
    base_rmse = np.sqrt(mean_squared_error(y_true, base_pred))
    importances = []
    rng = np.random.default_rng(RANDOM_STATE)
    for col in FEATURE_COLS:
        x_perm = x_base.copy()
        x_perm[col] = rng.permutation(x_perm[col].values)
        if prediction_mode == "residual":
            perm_pred = clip_ghi(sample_df["ghi_now"].values + pipe.predict(x_perm))
        else:
            perm_pred = clip_ghi(pipe.predict(x_perm))
        perm_rmse = np.sqrt(mean_squared_error(y_true, perm_pred))
        importances.append({"feature": col, "importance_rmse_increase": perm_rmse - base_rmse})
    return pd.DataFrame(importances).sort_values("importance_rmse_increase", ascending=False)


def save_outputs(direct_pipe, residual_pipe, metrics_df, hourly_df, importance_df, test_df, test_preds):
    direct_model_path = OUTPUT_DIR / "ghi_1h_direct_model.joblib"
    residual_model_path = OUTPUT_DIR / "ghi_1h_residual_model.joblib"
    metrics_path = OUTPUT_DIR / "ghi_1h_v2_metrics.csv"
    hourly_path = OUTPUT_DIR / "ghi_1h_v2_metrics_by_hour.csv"
    importance_path = OUTPUT_DIR / "ghi_1h_v2_feature_importance_residual.csv"
    pred_path = OUTPUT_DIR / "ghi_1h_v2_predictions_test.csv"
    plot_path = OUTPUT_DIR / "ghi_1h_v2_diagnostics.png"

    joblib.dump({"pipeline": direct_pipe, "feature_cols": FEATURE_COLS, "target_col": TARGET_COL}, direct_model_path)
    joblib.dump({"pipeline": residual_pipe, "feature_cols": FEATURE_COLS, "target_col": DELTA_TARGET_COL, "final_prediction": "ghi_now + predicted_delta clipped to 0-1400"}, residual_model_path)
    metrics_df.to_csv(metrics_path, index=False)
    hourly_df.to_csv(hourly_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    if len(test_df) > 0:
        pred_df = test_df[[TIME_COL, TARGET_TIME_COL, TARGET_COL, "ghi_now", "hour"]].copy()
        for key_name, values in test_preds.items():
            if key_name != "actual":
                pred_df[key_name] = values
                pred_df["error_" + key_name] = pred_df[key_name] - pred_df[TARGET_COL]
        pred_df.to_csv(pred_path, index=False)

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.scatter(pred_df[TARGET_COL], pred_df["persistence"], s=5, alpha=0.20, label="persistence")
        plt.scatter(pred_df[TARGET_COL], pred_df["residual_model"], s=5, alpha=0.20, label="residual")
        plt.plot([0, 1200], [0, 1200], color="black", linewidth=1)
        plt.xlabel("Actual GHI 1h ahead")
        plt.ylabel("Predicted GHI")
        plt.title("Test Prediction vs Actual")
        plt.legend()

        plt.subplot(2, 2, 2)
        sns.histplot(pred_df["error_persistence"], bins=60, kde=True, color="gray", label="persistence", alpha=0.5)
        sns.histplot(pred_df["error_residual_model"], bins=60, kde=True, color="tab:blue", label="residual", alpha=0.5)
        plt.xlabel("Error")
        plt.title("Test Error Distribution")
        plt.legend()

        plt.subplot(2, 2, 3)
        plot_df = pred_df.tail(min(800, len(pred_df)))
        plt.plot(plot_df[TARGET_TIME_COL], plot_df[TARGET_COL], label="actual", linewidth=1)
        plt.plot(plot_df[TARGET_TIME_COL], plot_df["persistence"], label="persistence", linewidth=1, alpha=0.7)
        plt.plot(plot_df[TARGET_TIME_COL], plot_df["residual_model"], label="residual", linewidth=1, alpha=0.9)
        plt.xticks(rotation=30)
        plt.ylabel("GHI")
        plt.title("Recent Test Period")
        plt.legend()

        plt.subplot(2, 2, 4)
        top_imp = importance_df.head(15).sort_values("importance_rmse_increase")
        plt.barh(top_imp["feature"], top_imp["importance_rmse_increase"])
        plt.xlabel("RMSE increase after permutation")
        plt.title("Residual Model Feature Importance")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

    return [direct_model_path, residual_model_path, metrics_path, hourly_path, importance_path, pred_path, plot_path]


def print_best_by_split(metrics_df):
    summary_rows = []
    metrics_df = metrics_df.copy()
    metrics_df["split"] = metrics_df["model"].str.split("_").str[0]
    for split_name, split_df in metrics_df.groupby("split"):
        best = split_df.sort_values("rmse").iloc[0]
        persistence = split_df[split_df["model"].str.contains("persistence")].iloc[0]
        summary_rows.append({
            "split": split_name,
            "best_model": best["model"],
            "best_rmse": best["rmse"],
            "persistence_rmse": persistence["rmse"],
            "skill_vs_persistence": 1.0 - best["rmse"] / persistence["rmse"] if persistence["rmse"] > 0 else np.nan
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))


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

    direct_pipe = build_hgb_model(target_type="direct")
    residual_pipe = build_hgb_model(target_type="residual")

    print("Training direct GHI model...")
    direct_pipe.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    print("Training residual delta-GHI model...")
    residual_pipe.fit(train_df[FEATURE_COLS], train_df[DELTA_TARGET_COL])

    print("Evaluating models...")
    metrics_df, pred_parts = evaluate_all(direct_pipe, residual_pipe, train_df, valid_df, test_df)
    print(metrics_df.to_string(index=False))

    print("Best model by split:")
    print_best_by_split(metrics_df)

    print("Computing test metrics by hour...")
    test_preds = pred_parts.get("test", {})
    hourly_df = metrics_by_hour(test_df, test_preds) if len(test_df) > 0 else pd.DataFrame()
    if len(hourly_df) > 0:
        print(hourly_df.sort_values(["hour", "rmse"]).head(50).to_string(index=False))

    print("Computing residual model permutation importance on test sample...")
    importance_df = permutation_importance_fast(residual_pipe, test_df, DELTA_TARGET_COL, prediction_mode="residual")
    print(importance_df.head(20).to_string(index=False))

    paths = save_outputs(direct_pipe, residual_pipe, metrics_df, hourly_df, importance_df, test_df, test_preds)
    print("Saved outputs:")
    for path_val in paths:
        print(str(path_val))

    con.close()


if __name__ == "__main__":
    main()
