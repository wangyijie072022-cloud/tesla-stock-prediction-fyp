#!/usr/bin/env python3
"""Date-driven TSLA next-trading-day direction prediction.

Modes:
- historical: replay a date inside the processed historical feature table.
- live: use the existing latest prediction input artifact.

The default path is fully local and does not call Doubao, yfinance, Alpha
Vantage, or any trading account.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    latest_closed_trading_day,
    next_available_date,
    next_calendar_trading_day,
    previous_available_date,
    previous_calendar_trading_day,
)
from scripts.model_io import load_feature_columns, load_model_bundle
from scripts.train_and_save_model import (
    build_model,
    build_validation_oof,
    select_lr_params,
    tune_threshold,
)
from scripts.tsla_experimental_features import add_experimental_features

HISTORICAL_DATA_PATH = PROJECT_ROOT / "data/processed/tsla_fused_dataset.csv"
MARKET_CACHE_PATH = PROJECT_ROOT / "data/raw/market_ohlcv_2020_2024.csv"
LATEST_INPUT_PATH = PROJECT_ROOT / "data/latest/tsla_latest_prediction_input.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/latest/tsla_prediction_for_date.csv"
MODEL_PATH = PROJECT_ROOT / "models/tsla_direction_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models/feature_columns.json"
SELECTED_THRESHOLD_PATH = PROJECT_ROOT / "models/selected_threshold.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"
CONSERVATIVE_THRESHOLD = 0.5
LOW_CONFIDENCE_MIN = 0.45
LOW_CONFIDENCE_MAX = 0.55
MIN_WALK_FORWARD_TRAIN_ROWS = 120


class PredictionError(RuntimeError):
    """Raised for user-facing prediction errors."""


class InsufficientHistoryError(PredictionError):
    """Raised when a date is too early for walk-forward replay."""

    def __init__(self, requested_date: Date, prior_rows: int, earliest_predictable_date: Date | None):
        self.requested_date = requested_date
        self.prior_rows = prior_rows
        self.earliest_predictable_date = earliest_predictable_date
        earliest_text = earliest_predictable_date.isoformat() if earliest_predictable_date else "not available"
        super().__init__(
            "Insufficient historical samples for walk-forward replay. "
            f"Requested date: {requested_date.isoformat()}. "
            f"Need at least {MIN_WALK_FORWARD_TRAIN_ROWS} prior rows, found {prior_rows}. "
            f"Earliest predictable date: {earliest_text}."
        )


@dataclass
class HistoricalReplayContext:
    historical_df: Any
    feature_columns: list[str]
    feature_matrix: Any
    target: Any
    processed_dates: list[Date]
    market_dates: list[Date]
    market_latest_date: Date | None
    earliest_predictable_date: Date | None


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise PredictionError(f"Missing prediction dependency: {exc}") from exc
    return pd, np


def resolve_decision_threshold(research_threshold: float, decision_threshold: float | None) -> float:
    active_threshold = research_threshold if decision_threshold is None else float(decision_threshold)
    if active_threshold < 0 or active_threshold > 1:
        raise PredictionError("decision_threshold must be between 0 and 1.")
    return active_threshold


def load_saved_model_bundle() -> tuple[Any, list[str], float, dict[str, Any]]:
    try:
        return load_model_bundle(MODEL_PATH, FEATURE_COLUMNS_PATH, SELECTED_THRESHOLD_PATH, MODEL_METADATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        raise PredictionError(str(exc)) from exc


def parse_input_date(value: str):
    pd, _ = require_runtime()
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.normalize()


def parse_date_only(value: str | Any) -> Date:
    parsed = parse_input_date(str(value))
    return parsed.date()


def load_market_dates() -> list[Date]:
    if not MARKET_CACHE_PATH.exists():
        return []
    pd, _ = require_runtime()
    market_df = pd.read_csv(MARKET_CACHE_PATH)
    if "Date" not in market_df.columns:
        return []
    dates = pd.to_datetime(market_df["Date"], errors="coerce").dropna()
    return sorted({item.date() for item in dates})


def resolve_input_trading_date(day: Date, available_dates: list[Date]) -> tuple[Date, str, str]:
    if day in available_dates:
        return day, "exact", ""
    mapped = previous_available_date(day, available_dates)
    if mapped is None:
        raise PredictionError("No available trading day exists at or before the requested date.")
    message = f"Input date is not a trading day; mapped to previous available trading day: {mapped.isoformat()}."
    return mapped, "mapped_to_previous_available_trading_day", message


def expected_input_trading_date(
    day: Date,
    market_dates: list[Date],
    latest_closed_day: Date | None = None,
) -> tuple[Date, str, str]:
    if latest_closed_day is not None and day > latest_closed_day:
        message = (
            "Input date is after the latest closed U.S. market session; "
            f"mapped to latest closed trading day: {latest_closed_day.isoformat()}."
        )
        return latest_closed_day, "mapped_to_latest_closed_trading_day", message

    market_latest = max(market_dates) if market_dates else None
    if market_latest and day <= market_latest:
        return resolve_input_trading_date(day, market_dates)

    expected = previous_calendar_trading_day(day)
    if latest_closed_day is not None and expected > latest_closed_day:
        message = (
            "Input date is before U.S. market close; "
            f"mapped to latest closed trading day: {latest_closed_day.isoformat()}."
        )
        return latest_closed_day, "mapped_to_latest_closed_trading_day", message
    if expected == day:
        return expected, "exact", ""
    message = f"Input date is not a trading day; mapped to previous available trading day: {expected.isoformat()}."
    return expected, "mapped_to_previous_available_trading_day", message


def infer_next_trading_day(input_trading_date: Date, available_dates: list[Date]) -> Date:
    next_from_cache = next_available_date(input_trading_date, available_dates)
    if next_from_cache:
        return next_from_cache
    return next_calendar_trading_day(input_trading_date)


def text_value(value: Any) -> str:
    pd, _ = require_runtime()
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def split_warning_text(value: Any) -> list[str]:
    text = text_value(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def confidence_bucket(probability_up: float, selected_threshold: float | None = None) -> str:
    del selected_threshold
    if LOW_CONFIDENCE_MIN <= probability_up <= LOW_CONFIDENCE_MAX:
        return "low"
    distance = abs(probability_up - CONSERVATIVE_THRESHOLD)
    if distance >= 0.20:
        return "high"
    if distance >= 0.10:
        return "medium"
    return "medium"


def decision_warnings(probability_up: float | None, decision_threshold: float, threshold_mode: str) -> list[str]:
    warnings: list[str] = []
    if decision_threshold < LOW_CONFIDENCE_MIN:
        warnings.append("Decision threshold is below 0.5; UP predictions may be frequent.")
    if threshold_mode == "research":
        warnings.append(
            "Research threshold is tuned for validation F1 and is not a high-confidence trading signal."
        )
    if probability_up is not None and LOW_CONFIDENCE_MIN <= probability_up <= LOW_CONFIDENCE_MAX:
        warnings.append("Low confidence.")
    return warnings


def direction_from_target(value: Any) -> str:
    pd, _ = require_runtime()
    if value is None or pd.isna(value):
        return ""
    return "up" if int(value) == 1 else "down"


def make_feature_matrix(source_df, feature_columns: list[str]):
    pd, np = require_runtime()
    missing_columns = [column for column in feature_columns if column not in source_df.columns]
    if missing_columns:
        raise PredictionError(f"Missing required training features: {missing_columns}")
    matrix = source_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_value_columns = matrix.columns[matrix.isna().any(axis=0)].tolist()
    if missing_value_columns:
        missing_by_row = {
            str(index): matrix.columns[matrix.iloc[index].isna()].tolist()
            for index in range(len(matrix))
            if matrix.iloc[index].isna().any()
        }
        raise PredictionError(
            "Required training features contain missing or non-numeric values: "
            f"{missing_by_row}"
        )
    return matrix


def earliest_walk_forward_date(historical_df, target) -> Date | None:
    valid_count = 0
    for idx, row in historical_df.iterrows():
        if valid_count >= MIN_WALK_FORWARD_TRAIN_ROWS:
            return row["Date"].date()
        if int(target.iloc[idx]) in {0, 1}:
            valid_count += 1
    return None


def load_historical_replay_context() -> HistoricalReplayContext:
    pd, _ = require_runtime()
    if not HISTORICAL_DATA_PATH.exists():
        raise PredictionError(f"Historical dataset is missing: {HISTORICAL_DATA_PATH}")

    try:
        feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        raise PredictionError(str(exc)) from exc

    historical_df = pd.read_csv(HISTORICAL_DATA_PATH)
    if "Date" not in historical_df.columns:
        raise PredictionError("Historical dataset is missing Date column.")

    historical_df = historical_df.copy()
    historical_df["Date"] = pd.to_datetime(historical_df["Date"], errors="coerce")
    historical_df = historical_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    historical_df = add_available_experimental_features(historical_df)
    if historical_df.empty:
        raise PredictionError("Historical dataset contains no valid Date rows.")

    target = pd.to_numeric(historical_df["Target"], errors="coerce")
    if not target.isin([0, 1]).all():
        raise PredictionError("Historical dataset contains invalid Target values.")
    target = target.astype(int).reset_index(drop=True)
    feature_matrix = make_feature_matrix(historical_df, feature_columns).reset_index(drop=True)
    processed_dates = sorted({item.date() for item in historical_df["Date"]})
    market_dates = load_market_dates()
    market_latest_date = max(market_dates) if market_dates else None
    earliest_date = earliest_walk_forward_date(historical_df, target)
    return HistoricalReplayContext(
        historical_df=historical_df,
        feature_columns=feature_columns,
        feature_matrix=feature_matrix,
        target=target,
        processed_dates=processed_dates,
        market_dates=market_dates,
        market_latest_date=market_latest_date,
        earliest_predictable_date=earliest_date,
    )


def add_available_experimental_features(source_df):
    pd, _ = require_runtime()
    if "Date" not in source_df.columns:
        for candidate in ["market_feature_date", "effective_feature_date", "input_trading_date"]:
            if candidate in source_df.columns:
                output = source_df.copy()
                output["Date"] = pd.to_datetime(output[candidate], errors="coerce")
                return add_experimental_features(output, MARKET_CACHE_PATH)
        return source_df
    return add_experimental_features(source_df, MARKET_CACHE_PATH)


def get_next_trading_day(feature_date, historical_df) -> str:
    feature_day = feature_date.date() if hasattr(feature_date, "date") else parse_date_only(feature_date)
    later_dates = historical_df.loc[historical_df["Date"].dt.date > feature_day, "Date"]
    if not later_dates.empty:
        return later_dates.min().date().isoformat()
    market_next = next_available_date(feature_day, load_market_dates())
    if market_next:
        return market_next.isoformat()
    return next_calendar_trading_day(feature_day).isoformat()


def build_walk_forward_replay(
    context: HistoricalReplayContext,
    selected_pos: int,
    input_trading_date: Date,
    decision_threshold: float | None,
    threshold_mode: str,
) -> dict[str, Any]:
    pd, _ = require_runtime()
    prior_target_raw = context.target.iloc[:selected_pos].reset_index(drop=True)
    valid_prior = prior_target_raw.isin([0, 1])
    prior_rows = int(valid_prior.sum())
    if prior_rows < MIN_WALK_FORWARD_TRAIN_ROWS:
        raise InsufficientHistoryError(input_trading_date, prior_rows, context.earliest_predictable_date)

    prior_target = prior_target_raw.loc[valid_prior].astype(int).reset_index(drop=True)
    if prior_target.nunique() < 2:
        raise PredictionError("Walk-forward training window contains only one target class.")

    X_prior = context.feature_matrix.iloc[:selected_pos].reset_index(drop=True).loc[valid_prior].reset_index(drop=True)
    prior_dates = context.historical_df["Date"].iloc[:selected_pos].reset_index(drop=True).loc[valid_prior].reset_index(drop=True)
    try:
        best_params, param_search_rows = select_lr_params(X_prior, prior_target)
        _, oof_df = build_validation_oof(X_prior, prior_target, prior_dates, best_params)
        walk_forward_threshold, threshold_metrics, _ = tune_threshold(oof_df)
    except SystemExit as exc:
        raise PredictionError("Walk-forward model selection failed.") from exc
    except Exception as exc:
        raise PredictionError(f"Walk-forward model selection failed: {exc}") from exc

    threshold_warning = ""
    if threshold_mode == "research":
        active_threshold = walk_forward_threshold
        if decision_threshold is not None and abs(float(decision_threshold) - walk_forward_threshold) > 1e-9:
            threshold_warning = "Saved research threshold ignored for historical replay; prior-only walk-forward threshold was used."
    else:
        active_threshold = resolve_decision_threshold(0.5, decision_threshold)

    model = build_model(best_params)
    model.fit(X_prior, prior_target)
    X_selected = context.feature_matrix.iloc[[selected_pos]]
    probability_up = float(model.predict_proba(X_selected)[:, 1][0])
    predicted_direction = "up" if probability_up >= active_threshold else "down"

    oof_dates = pd.to_datetime(oof_df["date"], errors="raise")
    diagnostics = {
        "probability_up": probability_up,
        "predicted_direction": predicted_direction,
        "active_threshold": active_threshold,
        "walk_forward_threshold": walk_forward_threshold,
        "threshold_metrics": threshold_metrics,
        "best_params": best_params,
        "param_search_rows": len(param_search_rows),
        "train_rows": int(len(X_prior)),
        "train_start_date": prior_dates.min().date().isoformat(),
        "train_end_date": prior_dates.max().date().isoformat(),
        "oof_rows": int(len(oof_df)),
        "oof_start_date": oof_dates.min().date().isoformat(),
        "oof_end_date": oof_dates.max().date().isoformat(),
        "threshold_warning": threshold_warning,
    }
    return diagnostics


def predict_historical(
    input_date: str,
    decision_threshold: float | None = None,
    threshold_mode: str = "research",
) -> dict[str, Any]:
    context = load_historical_replay_context()
    return predict_historical_from_context(context, input_date, decision_threshold, threshold_mode)


def predict_historical_from_context(
    context: HistoricalReplayContext,
    input_date: str,
    decision_threshold: float | None = None,
    threshold_mode: str = "research",
) -> dict[str, Any]:
    pd, _ = require_runtime()
    requested_date = parse_input_date(input_date)
    requested_day = requested_date.date()

    min_date = min(context.processed_dates)
    max_date = max(context.processed_dates)
    if requested_day < min_date:
        raise PredictionError(
            f"Input date is before the earliest available historical feature date: {min_date}"
        )
    if requested_day > max_date:
        raise PredictionError(
            f"Input date is after the latest available historical feature date: {max_date}. "
            "Use Live Prediction for current or out-of-range dates."
        )

    input_trading_date, date_mapping_status, mapping_note = resolve_input_trading_date(requested_day, context.processed_dates)
    available_rows = context.historical_df[context.historical_df["Date"].dt.date == input_trading_date]
    if available_rows.empty:
        raise PredictionError("No historical feature row is available for the resolved input trading date.")

    selected_pos = int(available_rows.index[0])
    selected_row = available_rows.iloc[0]
    feature_date = selected_row["Date"]
    was_mapped = date_mapping_status != "exact"
    next_trading_day = get_next_trading_day(feature_date, context.historical_df)
    market_data_stale = bool(context.market_latest_date and input_trading_date > context.market_latest_date)
    if market_data_stale:
        raise PredictionError("Market data stale. Please click Update Market Data before running prediction.")

    replay = build_walk_forward_replay(
        context,
        selected_pos,
        input_trading_date,
        decision_threshold,
        threshold_mode,
    )
    active_threshold = replay["active_threshold"]
    probability_up = replay["probability_up"]
    predicted_direction = replay["predicted_direction"]
    actual_direction = direction_from_target(selected_row.get("Target"))
    prediction_correct = (
        predicted_direction == actual_direction
        if actual_direction in {"up", "down"}
        else ""
    )

    warnings = [
        "This is not financial advice.",
        "Prediction is probabilistic and may be wrong.",
        "Historical replay uses walk-forward training on rows before the selected feature date only.",
    ]
    if mapping_note:
        warnings.append(mapping_note)
    if replay["threshold_warning"]:
        warnings.append(replay["threshold_warning"])
    warnings.extend(decision_warnings(probability_up, active_threshold, threshold_mode))

    return {
        "mode": "historical",
        "requested_date": requested_day.isoformat(),
        "input_date": requested_day.isoformat(),
        "input_trading_date": input_trading_date.isoformat(),
        "effective_feature_date": input_trading_date.isoformat(),
        "was_mapped_to_previous_trading_day": int(was_mapped),
        "date_mapping_status": date_mapping_status,
        "mapping_note": mapping_note,
        "next_trading_day": next_trading_day,
        "market_data_latest_date": context.market_latest_date.isoformat() if context.market_latest_date else "",
        "market_data_stale": int(market_data_stale),
        "prediction_allowed": 1,
        "data_cutoff": replay["train_end_date"],
        "model_created_at": "walk_forward_runtime",
        "probability_up": probability_up,
        "selected_threshold": active_threshold,
        "research_threshold": replay["walk_forward_threshold"],
        "threshold_mode": threshold_mode,
        "decision_margin": probability_up - active_threshold,
        "predicted_direction": predicted_direction,
        "confidence_bucket": confidence_bucket(probability_up, active_threshold),
        "actual_direction": actual_direction,
        "next_day_return": selected_row.get("Next_Day_Return", ""),
        "target": selected_row.get("Target", ""),
        "prediction_correct": prediction_correct,
        "feature_count": len(context.feature_columns),
        "feature_alignment_status": "full_match",
        "can_predict_with_historical_model": 1,
        "prediction_context": "historical_walk_forward_replay",
        "walk_forward_train_rows": replay["train_rows"],
        "walk_forward_train_start_date": replay["train_start_date"],
        "walk_forward_train_end_date": replay["train_end_date"],
        "walk_forward_oof_rows": replay["oof_rows"],
        "walk_forward_oof_start_date": replay["oof_start_date"],
        "walk_forward_oof_end_date": replay["oof_end_date"],
        "warnings": " | ".join(dict.fromkeys(warnings)),
        "model_type": "Walk-forward StandardScaler + LogisticRegression",
    }


def predict_live(
    input_date: str,
    decision_threshold: float | None = None,
    threshold_mode: str = "research",
) -> dict[str, Any]:
    pd, _ = require_runtime()
    requested_date = parse_input_date(input_date)
    requested_day = requested_date.date()
    market_dates = load_market_dates()
    if not market_dates:
        raise PredictionError("Market cache is missing or contains no usable trading dates.")

    market_latest_date = max(market_dates)
    latest_closed_day = latest_closed_trading_day()
    input_trading_date, date_mapping_status, mapping_note = expected_input_trading_date(
        requested_day,
        market_dates,
        latest_closed_day,
    )
    next_trading_day = infer_next_trading_day(input_trading_date, market_dates)
    market_data_stale = input_trading_date > market_latest_date

    _, feature_columns, research_threshold, metadata = load_saved_model_bundle()
    active_threshold = resolve_decision_threshold(research_threshold, decision_threshold)
    base_denied_result = {
        "mode": "live",
        "requested_date": requested_day.isoformat(),
        "input_date": requested_day.isoformat(),
        "input_trading_date": input_trading_date.isoformat(),
        "effective_feature_date": "",
        "was_mapped_to_previous_trading_day": int(date_mapping_status != "exact"),
        "date_mapping_status": date_mapping_status,
        "mapping_note": mapping_note,
        "next_trading_day": next_trading_day.isoformat(),
        "market_data_latest_date": market_latest_date.isoformat(),
        "latest_closed_market_date": latest_closed_day.isoformat(),
        "market_data_stale": int(market_data_stale),
        "prediction_allowed": 0,
        "data_cutoff": "",
        "market_feature_date": "",
        "model_created_at": metadata.get("created_at", ""),
        "probability_up": "",
        "selected_threshold": active_threshold,
        "research_threshold": research_threshold,
        "threshold_mode": threshold_mode,
        "decision_margin": "",
        "predicted_direction": "",
        "confidence_bucket": "",
        "actual_direction": "",
        "next_day_return": "",
        "target": "",
        "prediction_correct": "",
        "feature_count": len(feature_columns),
        "feature_alignment_status": "",
        "raw_feature_alignment_status": "",
        "can_predict_with_historical_model": "",
        "prediction_context": "live_blocked",
        "warnings": "",
        "model_type": metadata.get("model_type", ""),
    }

    if market_data_stale:
        warning_parts = [
            "Market data stale. Please click Update Market Data before running a formal prediction.",
            "This is not financial advice.",
            "Prediction is probabilistic and may be wrong.",
        ]
        if mapping_note:
            warning_parts.insert(0, mapping_note)
        result = dict(base_denied_result)
        result["warnings"] = " | ".join(dict.fromkeys(warning_parts))
        return result

    if not LATEST_INPUT_PATH.exists():
        raise PredictionError(f"Latest prediction input is missing: {LATEST_INPUT_PATH}")

    model, feature_columns, research_threshold, metadata = load_saved_model_bundle()
    active_threshold = resolve_decision_threshold(research_threshold, decision_threshold)
    latest_df = pd.read_csv(LATEST_INPUT_PATH)
    if latest_df.empty:
        raise PredictionError("Latest prediction input CSV contains no rows.")

    latest_row_df = add_available_experimental_features(latest_df.tail(1).copy())
    latest_row = latest_row_df.iloc[0]
    market_feature_date = text_value(latest_row.get("market_feature_date"))
    if market_feature_date != input_trading_date.isoformat():
        warning_parts = [
            f"Latest prediction input is built for market_feature_date={market_feature_date or 'missing'}, "
            f"but input_trading_date={input_trading_date.isoformat()}. Please click Update Market Data first.",
            "This is not financial advice.",
            "Prediction is probabilistic and may be wrong.",
        ]
        if mapping_note:
            warning_parts.insert(0, mapping_note)
        result = dict(base_denied_result)
        result.update(
            {
                "market_feature_date": market_feature_date,
                "effective_feature_date": market_feature_date,
                "data_cutoff": text_value(latest_row.get("prediction_cutoff")) or market_feature_date,
                "feature_alignment_status": text_value(latest_row.get("feature_alignment_status")),
                "raw_feature_alignment_status": text_value(latest_row.get("feature_alignment_status")),
                "can_predict_with_historical_model": text_value(latest_row.get("can_predict_with_historical_model")),
                "warnings": " | ".join(dict.fromkeys(warning_parts)),
            }
        )
        return result

    X = make_feature_matrix(latest_row_df, feature_columns)
    probability_up = float(model.predict_proba(X)[:, 1][0])
    predicted_direction = "up" if probability_up >= active_threshold else "down"

    raw_alignment = text_value(latest_row.get("feature_alignment_status"))
    can_predict = text_value(latest_row.get("can_predict_with_historical_model"))
    full_match = raw_alignment in {"full_match", "aligned"} and can_predict in {"1", "True", "true", "yes"}
    output_alignment = "full_match" if full_match else (raw_alignment or "unknown")

    warning_parts = []
    for column in ["blocking_context_warnings", "feature_context_warnings", "warnings"]:
        warning_parts.extend(split_warning_text(latest_row.get(column)))
    if mapping_note:
        warning_parts.append(mapping_note)
    if not full_match:
        warning_parts.append("experimental / partial context: feature alignment is not full_match or can_predict_with_historical_model is not 1")
    warning_parts.extend(
        [
            "This is not financial advice.",
            "Prediction is probabilistic and may be wrong.",
        ]
    )
    warning_parts.extend(decision_warnings(probability_up, active_threshold, threshold_mode))

    prediction_cutoff = text_value(latest_row.get("prediction_cutoff"))
    data_cutoff = prediction_cutoff or market_feature_date or requested_date.date().isoformat()

    return {
        "mode": "live",
        "requested_date": requested_day.isoformat(),
        "input_date": requested_day.isoformat(),
        "input_trading_date": input_trading_date.isoformat(),
        "effective_feature_date": market_feature_date,
        "was_mapped_to_previous_trading_day": int(date_mapping_status != "exact"),
        "date_mapping_status": date_mapping_status,
        "mapping_note": mapping_note,
        "next_trading_day": next_trading_day.isoformat(),
        "market_data_latest_date": market_latest_date.isoformat(),
        "latest_closed_market_date": latest_closed_day.isoformat(),
        "market_data_stale": 0,
        "prediction_allowed": 1,
        "data_cutoff": data_cutoff,
        "market_feature_date": market_feature_date,
        "model_created_at": metadata.get("created_at", ""),
        "probability_up": probability_up,
        "selected_threshold": active_threshold,
        "research_threshold": research_threshold,
        "threshold_mode": threshold_mode,
        "decision_margin": probability_up - active_threshold,
        "predicted_direction": predicted_direction,
        "confidence_bucket": confidence_bucket(probability_up, active_threshold),
        "actual_direction": "",
        "next_day_return": "",
        "target": "",
        "prediction_correct": "",
        "feature_count": len(feature_columns),
        "feature_alignment_status": output_alignment,
        "raw_feature_alignment_status": raw_alignment,
        "can_predict_with_historical_model": can_predict,
        "prediction_context": "live_experimental" if not full_match else "live_full_match",
        "warnings": " | ".join(dict.fromkeys(warning_parts)),
        "model_type": metadata.get("model_type", ""),
    }


def run_prediction_for_date(
    input_date: str,
    mode: str,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    decision_threshold: float | None = None,
    threshold_mode: str = "research",
) -> dict[str, Any]:
    if mode not in {"historical", "live"}:
        raise PredictionError("mode must be historical or live.")

    if mode == "historical":
        result = predict_historical(input_date, decision_threshold, threshold_mode)
    else:
        result = predict_live(input_date, decision_threshold, threshold_mode)
    pd, _ = require_runtime()
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False)
    result["output_path"] = str(path)
    return result


def skipped_insufficient_history_row(
    input_day: Date,
    exc: InsufficientHistoryError,
) -> dict[str, Any]:
    warnings = [
        str(exc),
        "Historical replay skipped because the selected date does not have enough prior samples.",
    ]
    earliest = exc.earliest_predictable_date.isoformat() if exc.earliest_predictable_date else ""
    return {
        "mode": "historical",
        "requested_date": input_day.isoformat(),
        "input_date": input_day.isoformat(),
        "input_trading_date": input_day.isoformat(),
        "effective_feature_date": input_day.isoformat(),
        "was_mapped_to_previous_trading_day": 0,
        "date_mapping_status": "exact",
        "mapping_note": "",
        "next_trading_day": "",
        "market_data_latest_date": "",
        "market_data_stale": "",
        "prediction_allowed": 0,
        "data_cutoff": "",
        "model_created_at": "walk_forward_runtime",
        "probability_up": "",
        "selected_threshold": "",
        "research_threshold": "",
        "threshold_mode": "",
        "decision_margin": "",
        "predicted_direction": "",
        "confidence_bucket": "",
        "actual_direction": "",
        "next_day_return": "",
        "target": "",
        "prediction_correct": "",
        "feature_count": "",
        "feature_alignment_status": "",
        "can_predict_with_historical_model": 0,
        "prediction_context": "skipped_insufficient_history",
        "walk_forward_train_rows": exc.prior_rows,
        "walk_forward_min_required_rows": MIN_WALK_FORWARD_TRAIN_ROWS,
        "earliest_predictable_date": earliest,
        "warnings": " | ".join(dict.fromkeys(warnings)),
        "model_type": "Walk-forward StandardScaler + LogisticRegression",
    }


def run_historical_batch_replay(
    batch_start: str,
    batch_end: str,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    decision_threshold: float | None = None,
    threshold_mode: str = "research",
) -> dict[str, Any]:
    pd, _ = require_runtime()
    start_day = parse_date_only(batch_start)
    end_day = parse_date_only(batch_end)
    if start_day > end_day:
        raise PredictionError("--batch-start must be on or before --batch-end.")

    context = load_historical_replay_context()
    requested_days = [
        day for day in context.processed_dates
        if start_day <= day <= end_day
    ]
    if not requested_days:
        raise PredictionError("No historical feature rows exist inside the requested batch date range.")

    rows: list[dict[str, Any]] = []
    skipped_count = 0
    for day in requested_days:
        try:
            rows.append(
                predict_historical_from_context(
                    context,
                    day.isoformat(),
                    decision_threshold=decision_threshold,
                    threshold_mode=threshold_mode,
                )
            )
        except InsufficientHistoryError as exc:
            rows.append(skipped_insufficient_history_row(day, exc))
            skipped_count += 1

    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return {
        "mode": "historical_batch",
        "batch_start": start_day.isoformat(),
        "batch_end": end_day.isoformat(),
        "rows": len(rows),
        "predicted_rows": len(rows) - skipped_count,
        "skipped_insufficient_history_rows": skipped_count,
        "earliest_predictable_date": (
            context.earliest_predictable_date.isoformat()
            if context.earliest_predictable_date
            else ""
        ),
        "output_path": str(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict TSLA direction for a selected date.")
    parser.add_argument("--date", default=None, help="Input date in YYYY-MM-DD format. Defaults to the latest historical feature date.")
    parser.add_argument("--mode", default="historical", choices=["historical", "live"], help="Prediction mode. Default: historical.")
    parser.add_argument("--batch-start", default=None, help="Historical batch replay start date in YYYY-MM-DD format.")
    parser.add_argument("--batch-end", default=None, help="Historical batch replay end date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Optional active decision threshold. Defaults to the saved research threshold.",
    )
    parser.add_argument(
        "--threshold-mode",
        default="research",
        choices=["research", "conservative"],
        help="Threshold label saved in the prediction output.",
    )
    return parser.parse_args()


def default_prediction_date() -> str:
    pd, _ = require_runtime()
    if HISTORICAL_DATA_PATH.exists():
        historical = pd.read_csv(HISTORICAL_DATA_PATH, usecols=["Date"])
        dates = pd.to_datetime(historical["Date"], errors="coerce").dropna()
        if not dates.empty:
            return dates.max().date().isoformat()
    return Date.today().isoformat()


def main() -> int:
    args = parse_args()
    input_date = args.date or default_prediction_date()
    try:
        if args.batch_start or args.batch_end:
            if args.mode != "historical":
                raise PredictionError("Batch replay is only available for --mode historical.")
            if not args.batch_start or not args.batch_end:
                raise PredictionError("Both --batch-start and --batch-end are required for batch replay.")
            result = run_historical_batch_replay(
                args.batch_start,
                args.batch_end,
                args.output,
                decision_threshold=args.decision_threshold,
                threshold_mode=args.threshold_mode,
            )
        else:
            result = run_prediction_for_date(
                input_date,
                args.mode,
                args.output,
                decision_threshold=args.decision_threshold,
                threshold_mode=args.threshold_mode,
            )
    except Exception as exc:
        payload = {"success": False, "error": str(exc)}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 1

    payload = {"success": True, **result}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
