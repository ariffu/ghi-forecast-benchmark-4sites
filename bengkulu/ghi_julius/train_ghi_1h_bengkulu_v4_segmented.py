#!/usr/bin/env python3
"""
V4 Bengkulu GHI 1-hour-ahead forecasting experiments.

Goal:
    Push beyond the V3 ceiling by testing higher-signal subsets and segmented LightGBM models.

Dataset:
    bengkulu_sch.ghi_forecast_1h_train_3h_rollback_2021_2025

Split:
    Train      2021-2023
    Validation 2024
    Test       2025
    2026 is not used.

Experiments:
    1. Global all-daylight LightGBM direct and residual models
    2. CLP-required subset
    3. High-sun subset
    4. Midday subset
    5. CLP-required high-sun subset
    6. Solar-elevation segmented models
    7. Hour-of-day segmented models
    8. Simple ensemble/selection diagnostics

Install:
    pip install duckdb pandas numpy scikit-learn matplotlib seaborn joblib pyarrow lightgbm tqdm

Run:
    export MOTHERDUCK_TOKEN="your_token_here"
    python train_ghi_1h_bengkulu_v4_segmented.py

Windows PowerShell:
    setx MOTHERDUCK_TOKEN "your_token_here"
    # reopen terminal
    python train_ghi_1h_bengkulu_v4_segmented.py

Outputs:
    outputs_v4_segmented/ghi_1h_v4_metrics.csv
    outputs_v4_segmented/ghi_1h_v4_segment_metrics.csv
    outputs_v4_segmented/ghi_1h_v4_predictions_test.csv
    outputs_v4_segmented/ghi_1h_v4_feature_importance.csv
    outputs_v4_segmented/ghi_1h_v4_diagnostics.png
    outputs_v4_segmented/models/*.joblib
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

warnings.filterwarnings("ignore")

DB_NAME = "bengkulu"
ATTACH_ALIAS = "bengkulu_db"
SCHEMA_NAME = "bengkulu_sch"
TABLE_NAME = "ghi_forecast_1h_train_3h_rollback_2021_2025"
OUTPUT_DIR = Path("outputs_v4_segmented")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TIME_COL = "ts_wib"
TARGET_TIME_COL = "target_ts_wib"
TARGET_COL = "target_ghi_1h_ahead"
DELTA_TARGET_COL = "target_delta_ghi_1h"
PRED_MIN = 0.0
PRED_MAX = 1400.0
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"
RANDOM_STATE = 42

BASE_FEATURES = [
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

ENGINEERED_FEATURES = [
    "solar_elev_sin", "solar_elev_sin_clip", "ghi_clear_proxy", "target_clear_proxy_now",
    "ghi_to_aws_sr_ratio", "dhi_fraction", "dni_fraction", "diffuse_to_global_ratio",
    "temp_rh_interaction", "vpd_proxy", "wind_u", "wind_v",
    "clp_cot_delta_60m", "clp_cth_delta_60m", "clp_cot_delta_180m", "clp_cth_delta_180m",
    "ghi_roll_180m_range", "ghi_roll_60m_range", "ghi_ramp_ratio_60m",
    "aws_temp_range", "cloud_opacity_proxy", "cloud_height_temp_interaction"
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

STATUS_COLS = [
    "is_model_ready", "has_continuous_3h_history", "has_asrs", "has_aws", "has_clp", "has_synop",
    "master_qc_status", "asrs_qc_status", "aws_qc_status", "clp_qc_status", "synop_qc_status"
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
    cols = [TIME_COL, TARGET_TIME_COL, TARGET_COL] + STATUS_COLS + BASE_FEATURES
    cols = list(dict.fromkeys(cols))
    sql_text = """
    SELECT
        """ + ",\n        ".join(cols) + """
    FROM """ + ATTACH_ALIAS + "." + SCHEMA_NAME + "." + TABLE_NAME + """
    WHERE observed_target_ts_wib = target_ts_wib
      AND target_ghi_1h_ahead IS NOT NULL
      AND ghi_now IS NOT NULL
      AND ts_wib >= TIMESTAMP '2021-01-01'
      AND target_ts_wib < TIMESTAMP '2026-01-01'
    ORDER BY ts_wib
    """
    df = con.execute(sql_text).fetchdf()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_TIME_COL] = pd.to_datetime(df[TARGET_TIME_COL])
    df["hour"] = df[TIME_COL].dt.hour
    df["month"] = df[TIME_COL].dt.month
    df[DELTA_TARGET_COL] = df[TARGET_COL] - df["ghi_now"]
    return df


def add_engineered_features(df):
    out = df.copy()
    elev_rad = np.deg2rad(out["solar_elev_deg"].astype(float))
    elev_sin = np.sin(elev_rad)
    out["solar_elev_sin"] = elev_sin
    out["solar_elev_sin_clip"] = np.maximum(elev_sin, 0.02)
    clear_proxy_den = 1100.0 * out["solar_elev_sin_clip"]
    out["ghi_clear_proxy"] = out["ghi_now"] / clear_proxy_den
    out["target_clear_proxy_now"] = out[TARGET_COL] / clear_proxy_den
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
    return out


def make_masks(df):
    base_ready = (
        (df["is_model_ready"] == 1) &
        (df["has_continuous_3h_history"] == 1) &
        (df[TARGET_COL].between(0, 1400)) &
        (df["ghi_now"].between(0, 1400)) &
        (df["daylight_flag"] == 1)
    )
    masks = {
        "all_daylight": base_ready,
        "clp_required": base_ready & (df["has_clp"] == True) & (df["clp_qc_status"] == "ok"),
        "high_sun": base_ready & (df["solar_elev_deg"] >= 15) & (df[TARGET_COL] >= 50),
        "midday_08_15": base_ready & (df["hour"].between(8, 15)) & (df[TARGET_COL] >= 50),
        "clp_high_sun": base_ready & (df["has_clp"] == True) & (df["clp_qc_status"] == "ok") & (df["solar_elev_deg"] >= 15) & (df[TARGET_COL] >= 50),
        "very_high_sun": base_ready & (df["solar_elev_deg"] >= 35) & (df[TARGET_COL] >= 100),
        "clp_very_high_sun": base_ready & (df["has_clp"] == True) & (df["clp_qc_status"] == "ok") & (df["solar_elev_deg"] >= 35) & (df[TARGET_COL] >= 100),
    }
    return masks


def split_masks(df, mask):
    train = mask & (df[TIME_COL] < pd.Timestamp(TRAIN_END))
    valid = mask & (df[TIME_COL] >= pd.Timestamp(TRAIN_END)) & (df[TIME_COL] < pd.Timestamp(VALID_END))
    test = mask & (df[TIME_COL] >= pd.Timestamp(VALID_END))
    return train, valid, test


def build_lgbm(kind="direct", segment_size="normal"):
    if segment_size == "small":
        num_leaves = 15
        min_child = 50
        n_estimators = 1600
    elif segment_size == "large":
        num_leaves = 47
        min_child = 70
        n_estimators = 2400
    else:
        num_leaves = 31
        min_child = 60
        n_estimators = 2000
    if kind == "residual":
        reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=0.025,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1
        )
    else:
        reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])


def metric_row(y_true, y_pred, label, persistence_rmse=None):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    mbe = float(np.mean(y_pred - y_true))
    skill = 0.0 if label.endswith("persistence") else np.nan
    if persistence_rmse and persistence_rmse > 0:
        skill = 1.0 - rmse / persistence_rmse
    return {"model": label, "n_rows": len(y_true), "mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe, "skill_vs_persistence": skill}


def eval_preds(y_true, preds, prefix):
    rows = []
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, preds["persistence"])))
    for name, pred in preds.items():
        rows.append(metric_row(y_true, pred, prefix + "_" + name, persistence_rmse))
    return pd.DataFrame(rows)


def train_experiment(df, exp_name, mask, feature_cols):
    train_mask, valid_mask, test_mask = split_masks(df, mask)
    n_train = int(train_mask.sum())
    n_valid = int(valid_mask.sum())
    n_test = int(test_mask.sum())
    if n_train < 1000 or n_valid < 200 or n_test < 200:
        return None, None, None
    segment_size = "large" if n_train > 50000 else "small" if n_train < 10000 else "normal"
    direct = build_lgbm("direct", segment_size)
    residual = build_lgbm("residual", segment_size)
    direct.fit(
        df.loc[train_mask, feature_cols], df.loc[train_mask, TARGET_COL],
        model__eval_set=[(df.loc[valid_mask, feature_cols], df.loc[valid_mask, TARGET_COL])],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    residual.fit(
        df.loc[train_mask, feature_cols], df.loc[train_mask, DELTA_TARGET_COL],
        model__eval_set=[(df.loc[valid_mask, feature_cols], df.loc[valid_mask, DELTA_TARGET_COL])],
        model__eval_metric="rmse",
        model__callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    metric_parts = []
    pred_test = None
    for split_name, part_mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        part = df.loc[part_mask].copy()
        y = part[TARGET_COL].values
        persistence = clip_ghi(part["ghi_now"].values)
        direct_pred = clip_ghi(direct.predict(part[feature_cols]))
        residual_pred = clip_ghi(part["ghi_now"].values + residual.predict(part[feature_cols]))
        preds = {
            "persistence": persistence,
            "lgbm_direct": direct_pred,
            "lgbm_residual": residual_pred,
            "blend_30persistence_70residual": clip_ghi(0.3 * persistence + 0.7 * residual_pred),
            "blend_50persistence_50residual": clip_ghi(0.5 * persistence + 0.5 * residual_pred),
            "blend_70persistence_30residual": clip_ghi(0.7 * persistence + 0.3 * residual_pred),
        }
        met = eval_preds(y, preds, exp_name + "_" + split_name)
        met["experiment"] = exp_name
        met["split"] = split_name
        met["n_train"] = n_train
        met["n_valid"] = n_valid
        met["n_test"] = n_test
        metric_parts.append(met)
        if split_name == "test":
            pred_test = part[[TIME_COL, TARGET_TIME_COL, TARGET_COL, "ghi_now", "hour", "solar_elev_deg", "has_clp", "clp_qc_status"]].copy()
            for key, val in preds.items():
                pred_test[key] = val
    metrics = pd.concat(metric_parts, ignore_index=True)
    importance = pd.DataFrame({
        "experiment": exp_name,
        "feature": feature_cols,
        "importance_direct": direct.named_steps["model"].feature_importances_,
        "importance_residual": residual.named_steps["model"].feature_importances_,
    }).sort_values(["importance_direct", "importance_residual"], ascending=False)
    joblib.dump({"pipeline": direct, "features": feature_cols, "experiment": exp_name}, MODEL_DIR / (exp_name + "_direct.joblib"))
    joblib.dump({"pipeline": residual, "features": feature_cols, "experiment": exp_name, "target": DELTA_TARGET_COL}, MODEL_DIR / (exp_name + "_residual.joblib"))
    return metrics, pred_test, importance


def solar_segment_name(df):
    conds = [df["solar_elev_deg"] < 15, df["solar_elev_deg"].between(15, 35), df["solar_elev_deg"] > 35]
    return np.select(conds, ["low_sun", "medium_sun", "high_sun"], default="unknown")


def hour_segment_name(df):
    conds = [df["hour"].between(5, 9), df["hour"].between(10, 13), df["hour"].between(14, 18)]
    return np.select(conds, ["morning", "midday", "afternoon"], default="other")


def main():
    print("Connecting to MotherDuck...")
    con = connect_motherduck()
    print("Loading 2021-2025 dataset...")
    df = load_data(con)
    con.close()
    df = add_engineered_features(df)
    df["solar_segment"] = solar_segment_name(df)
    df["hour_segment"] = hour_segment_name(df)
    print("Rows loaded: " + str(len(df)))
    print("Date range: " + str(df[TIME_COL].min()) + " to " + str(df[TIME_COL].max()))
    print("Model-ready rows: " + str(int(((df["is_model_ready"] == 1) & (df["has_continuous_3h_history"] == 1)).sum())))
    print("Split policy: train=2021-2023, validation=2024, test=2025; 2026 excluded.")

    masks = make_masks(df)
    all_metrics = []
    all_predictions = []
    all_importance = []

    print("Running subset experiments...")
    for exp_name, mask in tqdm(masks.items(), desc="Subset experiments"):
        result = train_experiment(df, exp_name, mask, FEATURES)
        metrics, pred_test, importance = result
        if metrics is not None:
            all_metrics.append(metrics)
            pred_test["experiment"] = exp_name
            all_predictions.append(pred_test)
            all_importance.append(importance)
            best_test = metrics[metrics["split"] == "test"].sort_values("rmse").iloc[0]
            print(exp_name + " best_test " + best_test["model"] + " rmse " + str(round(best_test["rmse"], 3)) + " r2 " + str(round(best_test["r2"], 4)))
        else:
            print(exp_name + " skipped due to insufficient rows")

    print("Running solar elevation segmented models...")
    base_masks = make_masks(df)
    base_ready = base_masks["all_daylight"]
    for seg_name in ["low_sun", "medium_sun", "high_sun"]:
        seg_mask = base_ready & (df["solar_segment"] == seg_name)
        exp_name = "seg_solar_" + seg_name
        metrics, pred_test, importance = train_experiment(df, exp_name, seg_mask, FEATURES)
        if metrics is not None:
            all_metrics.append(metrics)
            pred_test["experiment"] = exp_name
            all_predictions.append(pred_test)
            all_importance.append(importance)
            best_test = metrics[metrics["split"] == "test"].sort_values("rmse").iloc[0]
            print(exp_name + " best_test " + best_test["model"] + " rmse " + str(round(best_test["rmse"], 3)) + " r2 " + str(round(best_test["r2"], 4)))

    print("Running hour segmented models...")
    for seg_name in ["morning", "midday", "afternoon"]:
        seg_mask = base_ready & (df["hour_segment"] == seg_name)
        exp_name = "seg_hour_" + seg_name
        metrics, pred_test, importance = train_experiment(df, exp_name, seg_mask, FEATURES)
        if metrics is not None:
            all_metrics.append(metrics)
            pred_test["experiment"] = exp_name
            all_predictions.append(pred_test)
            all_importance.append(importance)
            best_test = metrics[metrics["split"] == "test"].sort_values("rmse").iloc[0]
            print(exp_name + " best_test " + best_test["model"] + " rmse " + str(round(best_test["rmse"], 3)) + " r2 " + str(round(best_test["r2"], 4)))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True)

    best_tests = metrics_df[metrics_df["split"] == "test"].sort_values("rmse")
    print("Best test models overall:")
    print(best_tests.head(30).to_string(index=False))

    segment_metrics = []
    for exp_name, pred_group in predictions_df.groupby("experiment"):
        for model_col in ["persistence", "lgbm_direct", "lgbm_residual", "blend_30persistence_70residual", "blend_50persistence_50residual", "blend_70persistence_30residual"]:
            if model_col not in pred_group.columns:
                continue
            y = pred_group[TARGET_COL].values
            persistence_rmse = float(np.sqrt(mean_squared_error(y, pred_group["persistence"].values)))
            row = metric_row(y, pred_group[model_col].values, model_col, persistence_rmse)
            row["experiment"] = exp_name
            row["test_rows"] = len(pred_group)
            segment_metrics.append(row)
    segment_metrics_df = pd.DataFrame(segment_metrics).sort_values("rmse")

    metrics_path = OUTPUT_DIR / "ghi_1h_v4_metrics.csv"
    segment_path = OUTPUT_DIR / "ghi_1h_v4_segment_metrics.csv"
    pred_path = OUTPUT_DIR / "ghi_1h_v4_predictions_test.csv"
    importance_path = OUTPUT_DIR / "ghi_1h_v4_feature_importance.csv"
    plot_path = OUTPUT_DIR / "ghi_1h_v4_diagnostics.png"
    metrics_df.to_csv(metrics_path, index=False)
    segment_metrics_df.to_csv(segment_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    top_plot = best_tests.head(20).copy()
    sns.barplot(data=top_plot, y="model", x="r2", hue="experiment", dodge=False)
    plt.axvline(0.9, color="red", linestyle="--", linewidth=1, label="R2 target 0.9")
    plt.title("Top Test R2 by Experiment")
    plt.xlabel("R2")
    plt.ylabel("")
    plt.legend(fontsize=7)

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=best_tests, x="n_test", y="r2", hue="experiment", size="skill_vs_persistence", sizes=(30, 160))
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.title("R2 vs Test Rows")
    plt.xlabel("Test rows")
    plt.ylabel("R2")
    plt.legend(fontsize=7)

    plt.subplot(2, 2, 3)
    best_exp = best_tests.iloc[0]["experiment"]
    best_pred = predictions_df[predictions_df["experiment"] == best_exp].copy()
    best_model_name = best_tests.iloc[0]["model"].split("_test_")[-1]
    if best_model_name in best_pred.columns:
        plt.scatter(best_pred[TARGET_COL], best_pred[best_model_name], s=5, alpha=0.25)
        plt.plot([0, 1200], [0, 1200], color="black", linewidth=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Best Experiment Prediction vs Actual: " + best_exp)

    plt.subplot(2, 2, 4)
    top_imp = importance_df[importance_df["experiment"] == best_exp].sort_values("importance_direct", ascending=False).head(18).sort_values("importance_direct")
    plt.barh(top_imp["feature"], top_imp["importance_direct"])
    plt.title("Top Direct Feature Importance: " + best_exp)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("Saved outputs:")
    for path in [metrics_path, segment_path, pred_path, importance_path, plot_path]:
        print(str(path))
    print("Models saved under: " + str(MODEL_DIR))


if __name__ == "__main__":
    main()
