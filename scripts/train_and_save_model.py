#!/usr/bin/env python3
"""Train and save the TSLA next-day direction model artifacts.

This script follows the notebook's Logistic Regression path:
- Phase 1 pruned feature set
- StandardScaler + LogisticRegression
- final 20% holdout split before any selection
- TimeSeriesSplit with gap=1 on the pre-holdout training/tuning window
- validation OOF probability threshold tuning without final-holdout rows

It does not modify notebooks or data/processed files. It writes the threshold
results CSV from the same OOF table used for the saved model artifacts so the
selected threshold stays consistent across outputs.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_latest_prediction_input import HISTORICAL_MODEL_FEATURE_COLUMNS
from scripts.tsla_experimental_features import (
    EXCLUDED_EXPERIMENTAL_FEATURES,
    EXPERIMENTAL_FEATURE_SET_NAME,
    FEATURE_FORMULAS,
    FORBIDDEN_FEATURE_COLUMNS,
    INCLUDED_EXPERIMENTAL_FEATURES,
    add_experimental_features,
    build_experimental_feature_columns,
    validate_no_forbidden_features,
)


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", category=UserWarning)


TRAINING_DATA_PATH = PROJECT_ROOT / "data/processed/tsla_fused_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
THRESHOLD_RESULTS_PATH = RESULTS_DIR / "threshold_tuning_validation_oof.csv"
PRODUCTION_CANDIDATE_METRICS_PATH = RESULTS_DIR / "tsla_production_candidate_metrics.csv"
PRODUCTION_CANDIDATE_SUMMARY_PATH = RESULTS_DIR / "tsla_production_candidate_summary.md"
MODEL_PATH = MODELS_DIR / "tsla_direction_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
SELECTED_THRESHOLD_PATH = MODELS_DIR / "selected_threshold.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

CV_GAP = 1
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
FINAL_HOLDOUT_FRACTION = 0.20
RANDOM_STATE = 42
FEATURE_SET_NAME = EXPERIMENTAL_FEATURE_SET_NAME
MODEL_TYPE = "StandardScaler + LogisticRegression (TSLA Experimental Research Model)"
MODEL_STATUS = "experimental_research_only"

LR_PARAM_CANDIDATES = [
    {"C": 0.01, "penalty": "l1", "solver": "liblinear", "class_weight": None},
    {"C": 0.10, "penalty": "l1", "solver": "liblinear", "class_weight": None},
    {"C": 1.00, "penalty": "l1", "solver": "liblinear", "class_weight": None},
    {"C": 10.0, "penalty": "l1", "solver": "liblinear", "class_weight": None},
    {"C": 0.01, "penalty": "l2", "solver": "liblinear", "class_weight": None},
    {"C": 0.10, "penalty": "l2", "solver": "liblinear", "class_weight": None},
    {"C": 1.00, "penalty": "l2", "solver": "liblinear", "class_weight": None},
    {"C": 10.0, "penalty": "l2", "solver": "liblinear", "class_weight": None},
    {"C": 0.01, "penalty": "l1", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 0.10, "penalty": "l1", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 1.00, "penalty": "l1", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 10.0, "penalty": "l1", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 0.01, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 0.10, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 1.00, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
    {"C": 10.0, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
]


def fail(message: str, exit_code: int = 1, **extra: Any) -> None:
    payload = {"success": False, "error": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(exit_code)


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        fail(
            "Missing training dependency. Install requirements.txt or run with the project .venv.",
            missing_dependency=str(exc),
        )

    return {
        "np": np,
        "pd": pd,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "roc_auc_score": roc_auc_score,
        "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
        "TimeSeriesSplit": TimeSeriesSplit,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def to_jsonable(value: Any) -> Any:
    libs = require_runtime()
    np = libs["np"]
    pd = libs["pd"]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_threshold_table(threshold_table, selected_threshold: float) -> None:
    selected_rows = threshold_table[threshold_table["selected_probability_threshold"].astype(int) == 1]
    if len(selected_rows) != 1:
        fail(
            "Threshold table must contain exactly one selected_probability_threshold row.",
            selected_row_count=len(selected_rows),
        )
    table_threshold = float(selected_rows.iloc[0]["probability_threshold"])
    if table_threshold != float(selected_threshold):
        fail(
            "Selected threshold mismatch between threshold table and model artifact payload.",
            table_threshold=table_threshold,
            selected_threshold=selected_threshold,
        )


def build_model(params: dict[str, Any]):
    libs = require_runtime()
    return libs["Pipeline"](
        steps=[
            ("scaler", libs["StandardScaler"]()),
            (
                "model",
                libs["LogisticRegression"](
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    **params,
                ),
            ),
        ]
    )


def metric_dict(y_true, y_pred) -> dict[str, float]:
    libs = require_runtime()
    return {
        "accuracy": float(libs["accuracy_score"](y_true, y_pred)),
        "precision": float(libs["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(libs["recall_score"](y_true, y_pred, zero_division=0)),
        "f1_score": float(libs["f1_score"](y_true, y_pred, zero_division=0)),
    }


def probability_metric_dict(y_true, y_pred, y_score) -> dict[str, float]:
    libs = require_runtime()
    pd = libs["pd"]
    y_true_series = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).astype(int).reset_index(drop=True)
    score_series = pd.Series(y_score).astype(float).reset_index(drop=True).clip(0, 1)
    metrics = metric_dict(y_true_series, y_pred_series)
    metrics.update(
        {
            "balanced_accuracy": float(libs["balanced_accuracy_score"](y_true_series, y_pred_series)),
            "roc_auc": float(libs["roc_auc_score"](y_true_series, score_series)),
            "pr_auc": float(libs["average_precision_score"](y_true_series, score_series)),
            "brier_score": float(libs["brier_score_loss"](y_true_series, score_series)),
        }
    )
    return metrics


def load_training_frame():
    libs = require_runtime()
    pd = libs["pd"]
    np = libs["np"]

    if not TRAINING_DATA_PATH.exists():
        fail("Training data CSV is missing.", path=str(TRAINING_DATA_PATH))

    df = pd.read_csv(TRAINING_DATA_PATH)
    df = add_experimental_features(df)
    feature_columns = build_experimental_feature_columns(HISTORICAL_MODEL_FEATURE_COLUMNS)
    validate_no_forbidden_features(feature_columns)

    required_columns = ["Date", "Target"] + feature_columns
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        fail("Training data is missing required columns.", missing_columns=missing_columns)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        fail("Training data contains unparseable Date values.")

    df = df.sort_values("Date").reset_index(drop=True)
    feature_df = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    target = df["Target"].astype(int)

    missing_feature_values = feature_df.isna().sum()
    missing_feature_values = missing_feature_values[missing_feature_values > 0].to_dict()
    if missing_feature_values:
        fail("Training feature matrix contains missing values.", missing_feature_values=missing_feature_values)
    if target.isna().any():
        fail("Training target contains missing values.")

    return df, feature_df, target, feature_columns


def date_range_payload(dates) -> dict[str, Any]:
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "rows": int(len(dates)),
    }


def build_final_holdout_split(df):
    libs = require_runtime()
    np = libs["np"]
    split_idx = int(np.floor(len(df) * (1 - FINAL_HOLDOUT_FRACTION)))
    if split_idx <= INNER_N_SPLITS + CV_GAP:
        fail(
            "Training/tuning window is too small for inner TimeSeriesSplit.",
            rows=len(df),
            split_idx=split_idx,
        )
    if len(df) - split_idx <= 0:
        fail("Final holdout split produced no holdout rows.", rows=len(df), split_idx=split_idx)

    train_tune_idx = np.arange(0, split_idx)
    holdout_idx = np.arange(split_idx, len(df))
    if df["Date"].iloc[train_tune_idx].max() >= df["Date"].iloc[holdout_idx].min():
        fail("Final holdout split is not strictly time ordered.")
    return train_tune_idx, holdout_idx


def build_split_summary(df, train_tune_idx, holdout_idx, oof_df=None) -> dict[str, Any]:
    summary = {
        "final_holdout_fraction": FINAL_HOLDOUT_FRACTION,
        "train_tune": date_range_payload(df["Date"].iloc[train_tune_idx]),
        "strict_final_20pct_holdout": date_range_payload(df["Date"].iloc[holdout_idx]),
    }
    if oof_df is not None and not oof_df.empty:
        oof_dates = require_runtime()["pd"].to_datetime(oof_df["date"], errors="raise")
        summary["validation_oof"] = date_range_payload(oof_dates)
        holdout_dates = set(df["Date"].iloc[holdout_idx].dt.date)
        oof_date_set = set(oof_dates.dt.date)
        overlap_dates = sorted(oof_date_set.intersection(holdout_dates))
        summary["oof_holdout_overlap_date_count"] = len(overlap_dates)
        summary["oof_holdout_overlap_dates"] = [day.isoformat() for day in overlap_dates]
    return summary


def validate_oof_holdout_separation(df, holdout_idx, oof_df) -> None:
    pd = require_runtime()["pd"]
    holdout_dates = set(df["Date"].iloc[holdout_idx].dt.date)
    oof_dates = set(pd.to_datetime(oof_df["date"], errors="raise").dt.date)
    overlap_dates = sorted(oof_dates.intersection(holdout_dates))
    if overlap_dates:
        fail(
            "Validation OOF dates overlap the final holdout dates.",
            overlap_dates=[day.isoformat() for day in overlap_dates[:10]],
            overlap_date_count=len(overlap_dates),
        )


def select_lr_params(X, y) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    libs = require_runtime()
    TimeSeriesSplit = libs["TimeSeriesSplit"]
    f1_score = libs["f1_score"]
    accuracy_score = libs["accuracy_score"]
    np = libs["np"]
    pd = libs["pd"]

    inner_tscv = TimeSeriesSplit(n_splits=INNER_N_SPLITS, gap=CV_GAP)
    rows: list[dict[str, Any]] = []

    for params in LR_PARAM_CANDIDATES:
        acc_scores = []
        f1_scores = []

        for train_idx, val_idx in inner_tscv.split(X):
            model = build_model(params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_val_pred = model.predict(X.iloc[val_idx])
            acc_scores.append(accuracy_score(y.iloc[val_idx], y_val_pred))
            f1_scores.append(f1_score(y.iloc[val_idx], y_val_pred, zero_division=0))

        rows.append(
            {
                "C": params["C"],
                "penalty": params["penalty"],
                "solver": params["solver"],
                "class_weight": "None" if params["class_weight"] is None else params["class_weight"],
                "Accuracy_Mean": float(np.mean(acc_scores)),
                "F1_Mean": float(np.mean(f1_scores)),
            }
        )

    search_df = pd.DataFrame(rows).sort_values(["Accuracy_Mean", "F1_Mean"], ascending=False).reset_index(drop=True)
    best_row = search_df.iloc[0]
    best_params = {
        "C": float(best_row["C"]),
        "penalty": str(best_row["penalty"]),
        "solver": "liblinear",
        "class_weight": None if best_row["class_weight"] == "None" else str(best_row["class_weight"]),
    }
    return best_params, search_df.to_dict(orient="records")


def build_validation_oof(X, y, dates, params: dict[str, Any]):
    libs = require_runtime()
    TimeSeriesSplit = libs["TimeSeriesSplit"]
    pd = libs["pd"]

    splitter = TimeSeriesSplit(n_splits=OUTER_N_SPLITS, gap=CV_GAP)
    fold_rows = []
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = build_model(params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        y_proba = model.predict_proba(X.iloc[test_idx])[:, 1]

        fold_metric = metric_dict(y.iloc[test_idx], y_pred)
        fold_metric.update(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_start_date": dates.iloc[train_idx].min().date().isoformat(),
                "train_end_date": dates.iloc[train_idx].max().date().isoformat(),
                "test_start_date": dates.iloc[test_idx].min().date().isoformat(),
                "test_end_date": dates.iloc[test_idx].max().date().isoformat(),
            }
        )
        fold_rows.append(fold_metric)

        oof_rows.append(
            pd.DataFrame(
                {
                    "row_index": test_idx,
                    "date": dates.iloc[test_idx].dt.date.astype(str).to_numpy(),
                    "y_true": y.iloc[test_idx].to_numpy(),
                    "y_pred_default_threshold": y_pred,
                    "y_proba": y_proba,
                    "fold": fold,
                }
            )
        )

    oof_df = pd.concat(oof_rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    return fold_rows, oof_df


def tune_threshold(oof_df):
    libs = require_runtime()
    np = libs["np"]
    pd = libs["pd"]

    y_true = oof_df["y_true"].astype(int).to_numpy()
    y_proba = oof_df["y_proba"].astype(float).to_numpy()
    rows = []

    for threshold in np.round(np.arange(0.30, 0.71, 0.02), 2):
        y_pred = (y_proba >= threshold).astype(int)
        row = {"probability_threshold": float(threshold)}
        row.update(metric_dict(y_true, y_pred))
        rows.append(row)

    threshold_df = pd.DataFrame(rows)
    best_idx = threshold_df["f1_score"].idxmax()
    selected_threshold = float(threshold_df.loc[best_idx, "probability_threshold"])
    threshold_df["selected_probability_threshold"] = threshold_df["probability_threshold"].eq(selected_threshold).astype(int)
    selected_metrics = threshold_df.loc[best_idx].drop(labels=["selected_probability_threshold"]).to_dict()
    return selected_threshold, selected_metrics, threshold_df


def summarize_fold_metrics(fold_rows: list[dict[str, Any]]) -> dict[str, float]:
    libs = require_runtime()
    pd = libs["pd"]
    fold_df = pd.DataFrame(fold_rows)
    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        summary[f"{metric}_mean"] = float(fold_df[metric].mean())
        summary[f"{metric}_std"] = float(fold_df[metric].std(ddof=0))
    return summary


def evaluate_holdout(
    df,
    X,
    y,
    params: dict[str, Any],
    selected_threshold: float,
    train_idx,
    holdout_idx,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    libs = require_runtime()
    np = libs["np"]

    model = build_model(params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    probabilities = model.predict_proba(X.iloc[holdout_idx])[:, 1]

    rows: list[dict[str, Any]] = []
    for label, threshold in [
        ("model_default_threshold_0p5", 0.5),
        ("model_research_selected_threshold", selected_threshold),
    ]:
        predictions = (probabilities >= threshold).astype(int)
        row = {
            "scope": "strict_final_20pct_holdout",
            "name": label,
            "kind": "model",
            "threshold": float(threshold),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(holdout_idx)),
            "train_start_date": df["Date"].iloc[train_idx].min().date().isoformat(),
            "train_end_date": df["Date"].iloc[train_idx].max().date().isoformat(),
            "test_start_date": df["Date"].iloc[holdout_idx].min().date().isoformat(),
            "test_end_date": df["Date"].iloc[holdout_idx].max().date().isoformat(),
            "predicted_up_ratio": float(np.mean(predictions)),
            "actual_up_ratio": float(y.iloc[holdout_idx].mean()),
        }
        row.update(probability_metric_dict(y.iloc[holdout_idx], predictions, probabilities))
        rows.append(row)

    baseline_map = {
        "always_up": np.ones(len(holdout_idx), dtype=int),
        "always_down": np.zeros(len(holdout_idx), dtype=int),
        "tsla_momentum_return_1": (df["Return_1"].iloc[holdout_idx].to_numpy() > 0).astype(int),
        "tsla_momentum_return_5": (df["Return_5"].iloc[holdout_idx].to_numpy() > 0).astype(int),
        "tsla_relative_to_qqq_momentum": (df["TSLA_vs_QQQ_Return_1"].iloc[holdout_idx].to_numpy() > 0).astype(int),
    }
    for name, predictions in baseline_map.items():
        row = {
            "scope": "strict_final_20pct_holdout",
            "name": name,
            "kind": "baseline",
            "threshold": "",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(holdout_idx)),
            "train_start_date": df["Date"].iloc[train_idx].min().date().isoformat(),
            "train_end_date": df["Date"].iloc[train_idx].max().date().isoformat(),
            "test_start_date": df["Date"].iloc[holdout_idx].min().date().isoformat(),
            "test_end_date": df["Date"].iloc[holdout_idx].max().date().isoformat(),
            "predicted_up_ratio": float(np.mean(predictions)),
            "actual_up_ratio": float(y.iloc[holdout_idx].mean()),
        }
        row.update(probability_metric_dict(y.iloc[holdout_idx], predictions, predictions.astype(float)))
        rows.append(row)

    model_default = next(row for row in rows if row["name"] == "model_default_threshold_0p5")
    best_baseline = sorted(
        [row for row in rows if row["kind"] == "baseline"],
        key=lambda row: (row["accuracy"], row["balanced_accuracy"], row["roc_auc"]),
        reverse=True,
    )[0]
    summary = {
        "split": {
            "train_start_date": model_default["train_start_date"],
            "train_end_date": model_default["train_end_date"],
            "holdout_start_date": model_default["test_start_date"],
            "holdout_end_date": model_default["test_end_date"],
            "train_rows": model_default["train_rows"],
            "holdout_rows": model_default["test_rows"],
        },
        "model_default_threshold": model_default,
        "best_baseline": best_baseline,
        "delta_accuracy_vs_best_baseline": float(model_default["accuracy"] - best_baseline["accuracy"]),
        "delta_roc_auc_vs_best_baseline": float(model_default["roc_auc"] - best_baseline["roc_auc"]),
    }
    return rows, summary


def build_candidate_metrics_table(
    oof_df,
    default_oof_metrics: dict[str, float],
    selected_oof_metrics: dict[str, float],
    selected_threshold: float,
    holdout_rows: list[dict[str, Any]],
):
    libs = require_runtime()
    pd = libs["pd"]
    oof_dates = pd.to_datetime(oof_df["date"], errors="raise") if "date" in oof_df.columns else pd.Series(dtype="datetime64[ns]")
    oof_start = oof_dates.min().date().isoformat() if not oof_dates.empty else ""
    oof_end = oof_dates.max().date().isoformat() if not oof_dates.empty else ""
    oof_rows = [
        {
            "scope": "validation_oof",
            "name": "model_default_threshold_0p5",
            "kind": "model",
            "threshold": 0.5,
            "train_rows": "",
            "test_rows": int(len(oof_df)),
            "train_start_date": "",
            "train_end_date": "",
            "test_start_date": oof_start,
            "test_end_date": oof_end,
            "predicted_up_ratio": float(oof_df["y_pred_default_threshold"].astype(int).mean()),
            "actual_up_ratio": float(oof_df["y_true"].astype(int).mean()),
            **default_oof_metrics,
        },
        {
            "scope": "validation_oof",
            "name": "model_research_selected_threshold",
            "kind": "model",
            "threshold": selected_threshold,
            "train_rows": "",
            "test_rows": int(len(oof_df)),
            "train_start_date": "",
            "train_end_date": "",
            "test_start_date": oof_start,
            "test_end_date": oof_end,
            "predicted_up_ratio": float((oof_df["y_proba"].astype(float) >= selected_threshold).mean()),
            "actual_up_ratio": float(oof_df["y_true"].astype(int).mean()),
            **selected_oof_metrics,
        },
    ]
    return pd.DataFrame([*oof_rows, *holdout_rows])


def write_candidate_summary(
    metadata: dict[str, Any],
    candidate_metrics,
    holdout_summary: dict[str, Any],
) -> None:
    libs = require_runtime()
    pd = libs["pd"]
    display_columns = [
        "scope",
        "name",
        "kind",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "predicted_up_ratio",
        "actual_up_ratio",
    ]
    printable = candidate_metrics[[column for column in display_columns if column in candidate_metrics.columns]].copy()
    for column in printable.columns:
        if pd.api.types.is_numeric_dtype(printable[column]):
            printable[column] = printable[column].round(4)
    table_header = "| " + " | ".join(printable.columns) + " |"
    table_separator = "| " + " | ".join(["---"] * len(printable.columns)) + " |"
    table_rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in printable.to_numpy()
    ]
    table = "\n".join([table_header, table_separator, *table_rows])

    lines = [
        "# TSLA Production Candidate Summary",
        "",
        "Status: experimental / research only. This is not a reliable prediction system and is not financial advice.",
        "",
        f"Feature set: `{metadata['feature_set_name']}`",
        f"Model type: `{metadata['model_type']}`",
        f"Training date range: `{metadata['train_date_range']['start']}` to `{metadata['train_date_range']['end']}`",
        f"Feature count: `{metadata['feature_count']}`",
        "",
        "## Included Experimental Features",
        "",
        "\n".join(f"- `{feature}`: {FEATURE_FORMULAS[feature]}" for feature in INCLUDED_EXPERIMENTAL_FEATURES),
        "",
        "## Explicitly Excluded Features",
        "",
        "\n".join(f"- `{feature}`: {reason}" for feature, reason in EXCLUDED_EXPERIMENTAL_FEATURES.items()),
        "",
        "## Metrics",
        "",
        table,
        "",
        "## Holdout Baseline Check",
        "",
        f"- Holdout model accuracy at threshold 0.5: `{holdout_summary['model_default_threshold']['accuracy']:.4f}`.",
        f"- Best holdout baseline: `{holdout_summary['best_baseline']['name']}` accuracy `{holdout_summary['best_baseline']['accuracy']:.4f}`.",
        f"- Delta accuracy vs best baseline: `{holdout_summary['delta_accuracy_vs_best_baseline']:+.4f}`.",
        "",
        "Interpretation: any improvement is limited and should not be presented as a reliable trading edge.",
    ]
    PRODUCTION_CANDIDATE_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training_pipeline() -> dict[str, Any]:
    df, X, y, feature_columns = load_training_frame()
    created_at = datetime.now(timezone.utc).isoformat()

    train_tune_idx, holdout_idx = build_final_holdout_split(df)
    X_train_tune = X.iloc[train_tune_idx].reset_index(drop=True)
    y_train_tune = y.iloc[train_tune_idx].reset_index(drop=True)
    train_tune_dates = df["Date"].iloc[train_tune_idx].reset_index(drop=True)

    best_params, lr_search_results = select_lr_params(X_train_tune, y_train_tune)
    fold_rows, oof_df = build_validation_oof(X_train_tune, y_train_tune, train_tune_dates, best_params)
    validate_oof_holdout_separation(df, holdout_idx, oof_df)
    selected_threshold, selected_threshold_metrics, threshold_table = tune_threshold(oof_df)
    validate_threshold_table(threshold_table, selected_threshold)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(THRESHOLD_RESULTS_PATH, index=False)

    default_oof_metrics = probability_metric_dict(
        oof_df["y_true"].astype(int),
        oof_df["y_pred_default_threshold"].astype(int),
        oof_df["y_proba"].astype(float),
    )
    selected_oof_pred = (oof_df["y_proba"].astype(float) >= selected_threshold).astype(int)
    selected_oof_metrics = probability_metric_dict(
        oof_df["y_true"].astype(int),
        selected_oof_pred,
        oof_df["y_proba"].astype(float),
    )
    holdout_rows, holdout_summary = evaluate_holdout(
        df,
        X,
        y,
        best_params,
        selected_threshold,
        train_tune_idx,
        holdout_idx,
    )
    split_summary = build_split_summary(df, train_tune_idx, holdout_idx, oof_df)

    final_model = build_model(best_params)
    final_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_model, handle)

    write_json(FEATURE_COLUMNS_PATH, feature_columns)
    write_json(
        SELECTED_THRESHOLD_PATH,
        {
            "selected_threshold": selected_threshold,
            "selection_metric": "f1_score",
            "selection_source": (
                f"Pre-holdout TimeSeriesSplit(n_splits={OUTER_N_SPLITS}, gap={CV_GAP}) "
                "OOF probabilities only"
            ),
            "threshold_results_path": str(THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "metrics_at_selected_threshold": selected_threshold_metrics,
            "model_status": MODEL_STATUS,
            "experimental": True,
            "recommended_conservative_threshold": 0.5,
            "split_summary": split_summary,
            "warning": "Research threshold is selected for validation F1 only and does not use the final holdout.",
        },
    )
    candidate_metrics = build_candidate_metrics_table(
        oof_df,
        default_oof_metrics,
        selected_oof_metrics,
        selected_threshold,
        holdout_rows,
    )
    candidate_metrics.to_csv(PRODUCTION_CANDIDATE_METRICS_PATH, index=False)

    metadata = {
        "model_status": MODEL_STATUS,
        "experimental": True,
        "scope": "TSLA-only US equity research model",
        "not_financial_advice": True,
        "training_data_path": str(TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)),
        "train_date_range": {
            "start": df["Date"].min().date().isoformat(),
            "end": df["Date"].max().date().isoformat(),
        },
        "evaluation_split": split_summary,
        "row_count": int(len(df)),
        "feature_count": len(feature_columns),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH.relative_to(PROJECT_ROOT)),
        "feature_set_name": FEATURE_SET_NAME,
        "base_feature_count": len(HISTORICAL_MODEL_FEATURE_COLUMNS),
        "included_experimental_features": list(INCLUDED_EXPERIMENTAL_FEATURES),
        "included_experimental_feature_formulas": FEATURE_FORMULAS,
        "excluded_candidate_features": EXCLUDED_EXPERIMENTAL_FEATURES,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "model_type": MODEL_TYPE,
        "model_params": best_params,
        "selected_threshold": selected_threshold,
        "recommended_conservative_threshold": 0.5,
        "validation_metrics": {
            "cv_strategy": f"Pre-holdout TimeSeriesSplit(n_splits={OUTER_N_SPLITS}, gap={CV_GAP})",
            "inner_param_search_strategy": f"Pre-holdout TimeSeriesSplit(n_splits={INNER_N_SPLITS}, gap={CV_GAP})",
            "threshold_results_path": str(THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "production_candidate_metrics_path": str(PRODUCTION_CANDIDATE_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "production_candidate_summary_path": str(PRODUCTION_CANDIDATE_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "default_threshold_oof_metrics": default_oof_metrics,
            "selected_threshold_oof_metrics": selected_oof_metrics,
            "strict_final_20pct_holdout": holdout_summary,
            "fold_metrics_default_threshold": fold_rows,
            "fold_metric_summary_default_threshold": summarize_fold_metrics(fold_rows),
            "threshold_grid": threshold_table.to_dict(orient="records"),
            "lr_param_search_results": lr_search_results,
        },
        "created_at": created_at,
        "notes": [
            "Parameter selection, OOF probabilities, and threshold tuning use only the pre-holdout train/tune window.",
            "Strict final 20% holdout is evaluated once after all selection and threshold tuning are complete.",
            "Saved final artifact is fitted on the full historical dataset after strict holdout evaluation for future inference only.",
            "No notebooks or data/processed files are modified by this script.",
            "The threshold tuning CSV and model threshold artifacts are written from the same pre-holdout OOF table.",
            "This artifact is experimental/research-only and must not be described as a reliable trading signal.",
            "The UI defaults to the conservative 0.5 threshold; the selected research threshold is retained for validation diagnostics.",
        ],
    }
    write_json(MODEL_METADATA_PATH, metadata)
    write_candidate_summary(metadata, candidate_metrics, holdout_summary)

    return {
        "success": True,
        "model_path": str(MODEL_PATH),
        "threshold_results_path": str(THRESHOLD_RESULTS_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "selected_threshold_path": str(SELECTED_THRESHOLD_PATH),
        "model_metadata_path": str(MODEL_METADATA_PATH),
        "feature_count": len(feature_columns),
        "model_type": MODEL_TYPE,
        "model_status": MODEL_STATUS,
        "selected_threshold": selected_threshold,
        "validation_metrics": selected_oof_metrics,
        "holdout_metrics": holdout_summary,
        "evaluation_split": split_summary,
        "production_candidate_metrics_path": str(PRODUCTION_CANDIDATE_METRICS_PATH),
        "production_candidate_summary_path": str(PRODUCTION_CANDIDATE_SUMMARY_PATH),
        "created_at": created_at,
    }


def main() -> int:
    payload = run_training_pipeline()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
