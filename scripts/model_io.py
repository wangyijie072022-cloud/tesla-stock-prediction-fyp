"""Shared model artifact loading helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/tsla_direction_model.pkl"
DEFAULT_FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models/feature_columns.json"
DEFAULT_SELECTED_THRESHOLD_PATH = PROJECT_ROOT / "models/selected_threshold.json"
DEFAULT_MODEL_METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_columns(path: Path = DEFAULT_FEATURE_COLUMNS_PATH) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, dict) and isinstance(payload.get("feature_columns"), list):
        columns = payload["feature_columns"]
    else:
        raise ValueError("feature_columns.json must contain a list of feature names.")
    if not columns or not all(isinstance(column, str) for column in columns):
        raise ValueError("feature_columns.json contains invalid feature names.")
    return columns


def load_selected_threshold(path: Path = DEFAULT_SELECTED_THRESHOLD_PATH) -> float:
    payload = read_json(path)
    raw_value = payload.get("selected_threshold") if isinstance(payload, dict) else payload
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid selected_threshold value: {raw_value}") from exc


def load_model(path: Path = DEFAULT_MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact is missing: {path}")
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")
    return model


def load_model_metadata(path: Path = DEFAULT_MODEL_METADATA_PATH) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def load_model_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
    feature_columns_path: Path = DEFAULT_FEATURE_COLUMNS_PATH,
    selected_threshold_path: Path = DEFAULT_SELECTED_THRESHOLD_PATH,
    model_metadata_path: Path = DEFAULT_MODEL_METADATA_PATH,
) -> tuple[Any, list[str], float, dict[str, Any]]:
    model = load_model(model_path)
    feature_columns = load_feature_columns(feature_columns_path)
    selected_threshold = load_selected_threshold(selected_threshold_path)
    metadata = load_model_metadata(model_metadata_path)
    return model, feature_columns, selected_threshold, metadata
