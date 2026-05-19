#!/usr/bin/env python3
"""Predict the latest TSLA next-day direction from a prepared input row.

This script only uses local files. It does not fetch news, refresh market data,
train a model, place trades, or connect to any trading account.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_io import load_feature_columns, load_model, load_model_metadata, load_selected_threshold
from scripts.tsla_experimental_features import add_experimental_features

LATEST_INPUT_PATH = PROJECT_ROOT / "data/latest/tsla_latest_prediction_input.csv"
MARKET_CACHE_PATH = PROJECT_ROOT / "data/raw/market_ohlcv_2020_2024.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/latest/tsla_latest_prediction_output.csv"
MODEL_PATH = PROJECT_ROOT / "models/tsla_direction_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models/feature_columns.json"
SELECTED_THRESHOLD_PATH = PROJECT_ROOT / "models/selected_threshold.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"


def fail(message: str, exit_code: int = 1, **extra: Any) -> None:
    payload = {"success": False, "error": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(exit_code)


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        fail(
            "Missing prediction dependency. Install requirements.txt or run with the project .venv.",
            missing_dependency=str(exc),
        )
    return pd, np


def text_value(value: Any) -> str:
    pd, _ = require_runtime()
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def collect_warnings(row) -> str:
    warning_parts = []
    for column in ["warnings", "blocking_context_warnings", "feature_context_warnings"]:
        if column in row.index:
            value = text_value(row[column])
            if value:
                warning_parts.extend(part.strip() for part in value.split("|") if part.strip())

    if "can_predict_with_historical_model" in row.index and text_value(row["can_predict_with_historical_model"]) in {"0", "False", "false"}:
        warning_parts.append("latest_input_can_predict_with_historical_model_is_0")

    if "feature_alignment_status" in row.index:
        status = text_value(row["feature_alignment_status"])
        if status and status != "aligned":
            warning_parts.append(f"feature_alignment_status={status}")

    warning_parts.append("This is not financial advice.")
    warning_parts.append("Research model, not financial advice.")
    warning_parts.append("Experimental TSLA model; not a reliable trading signal.")
    return " | ".join(dict.fromkeys(warning_parts))


def confidence_bucket(probability_up: float, selected_threshold: float) -> str:
    distance = abs(probability_up - selected_threshold)
    if distance >= 0.20:
        return "high"
    if distance >= 0.10:
        return "medium"
    return "low"


def run_latest_prediction(
    input_path: str | Path = LATEST_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    pd, np = require_runtime()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    for path in [input_path, MODEL_PATH, FEATURE_COLUMNS_PATH, SELECTED_THRESHOLD_PATH]:
        if not path.exists():
            fail("Required input or model artifact is missing.", path=str(path))

    latest_df = pd.read_csv(input_path)
    if latest_df.empty:
        fail("Latest prediction input CSV contains no rows.", path=str(input_path))

    try:
        feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
        selected_threshold = load_selected_threshold(SELECTED_THRESHOLD_PATH)
        model_metadata = load_model_metadata(MODEL_METADATA_PATH)
        model = load_model(MODEL_PATH)
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    if "Date" not in latest_df.columns and "market_feature_date" in latest_df.columns:
        latest_df = latest_df.copy()
        latest_df["Date"] = pd.to_datetime(latest_df["market_feature_date"], errors="coerce")
    latest_df = add_experimental_features(latest_df, MARKET_CACHE_PATH)

    missing_columns = [column for column in feature_columns if column not in latest_df.columns]
    if missing_columns:
        fail("Latest input is missing required training features.", missing_feature_columns=missing_columns)

    X = latest_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_value_columns = X.columns[X.isna().any(axis=0)].tolist()
    if missing_value_columns:
        missing_by_row = {
            str(index): X.columns[X.iloc[index].isna()].tolist()
            for index in range(len(X))
            if X.iloc[index].isna().any()
        }
        fail(
            "Latest input has missing or non-numeric values in required training features.",
            missing_feature_columns=missing_value_columns,
            missing_by_row=missing_by_row,
        )

    probabilities_up = model.predict_proba(X)[:, 1]
    prediction_cutoff_values = (
        latest_df["prediction_cutoff"].astype(str).tolist()
        if "prediction_cutoff" in latest_df.columns
        else [""] * len(latest_df)
    )

    output_rows = []
    for row_idx, probability_up in enumerate(probabilities_up):
        probability = float(probability_up)
        predicted_direction = "up" if probability >= selected_threshold else "down"
        output_rows.append(
            {
                "prediction_cutoff": prediction_cutoff_values[row_idx],
                "model_created_at": model_metadata.get("created_at", ""),
                "probability_up": probability,
                "selected_threshold": selected_threshold,
                "predicted_direction": predicted_direction,
                "confidence_bucket": confidence_bucket(probability, selected_threshold),
                "feature_count": len(feature_columns),
                "warnings": collect_warnings(latest_df.iloc[row_idx]),
                "model_type": model_metadata.get("model_type", ""),
                "feature_alignment_status": latest_df.iloc[row_idx].get("feature_alignment_status", ""),
                "can_predict_with_historical_model": latest_df.iloc[row_idx].get("can_predict_with_historical_model", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, index=False)

    return {
        "success": True,
        "output_path": str(output_path),
        "rows": len(output_rows),
        "feature_count": len(feature_columns),
        "feature_columns_match": list(X.columns) == feature_columns,
        "selected_threshold": selected_threshold,
        "predictions": output_rows,
    }


def main() -> int:
    payload = run_latest_prediction()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
