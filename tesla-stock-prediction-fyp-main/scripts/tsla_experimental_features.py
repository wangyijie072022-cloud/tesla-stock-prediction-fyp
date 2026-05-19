"""Shared TSLA-only experimental feature construction.

These helpers intentionally construct a small feature set only. They exclude
candidate features that earlier diagnostics marked as noisy or insufficiently
supported by reliable local calendars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_PATH = PROJECT_ROOT / "data/raw/market_ohlcv_2020_2024.csv"

FORBIDDEN_FEATURE_COLUMNS = {
    "Target",
    "Next_Day_Close",
    "Next_Day_Return",
    "TSLA_forward_return_1d",
    "TSLA_forward_return_3d",
    "TSLA_forward_return_5d",
    "QQQ_forward_return_1d",
    "QQQ_forward_return_5d",
    "SPY_forward_return_1d",
    "SPY_forward_return_5d",
}

EXPERIMENTAL_FEATURE_SET_NAME = "TSLA_Experimental_Current61_Plus_Conservative_TSLA_State"

INCLUDED_EXPERIMENTAL_FEATURES = [
    "tsla_residual_return_vs_qqq",
    "gap_open_prev_close",
    "volume_shock_20",
    "rolling_volatility_20",
    "distance_to_20d_high",
    "distance_to_20d_low",
]

EXCLUDED_EXPERIMENTAL_FEATURES = {
    "tsla_rolling_beta_to_qqq": "Diagnostic results showed mixed behavior; beta is used internally to compute residual return but excluded as a model feature.",
    "market_state": "Market-state group was unstable and often dragged 5-day target diagnostics.",
    "all_new_features": "All-feature bundle increased noise and was not stable across walk-forward windows.",
    "event_window_pre_features": "Reliable earnings/delivery/FOMC calendars are not present locally; pre-window features would risk future event-date leakage.",
    "intraday_reversal": "Single-feature diagnostics showed calibration/accuracy drag; excluded by default.",
    "previous_large_up": "Not stable enough to include by default.",
    "previous_large_down": "Single-feature diagnostics showed worse PR AUC/Brier behavior; excluded by default.",
}

FEATURE_FORMULAS = {
    "tsla_residual_return_vs_qqq": "TSLA return_1 - rolling 60-day beta(TSLA, QQQ) * QQQ return_1; beta is not saved as a model feature.",
    "gap_open_prev_close": "TSLA_Open / previous TSLA_Close - 1.",
    "volume_shock_20": "1 when abs(TSLA volume z-score versus trailing 20 trading days) >= 2, else 0.",
    "rolling_volatility_20": "Trailing 20-trading-day standard deviation of TSLA close-to-close return.",
    "distance_to_20d_high": "TSLA_Close / trailing 20-trading-day high - 1.",
    "distance_to_20d_low": "TSLA_Close / trailing 20-trading-day low - 1.",
}


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - caller reports this.
        raise RuntimeError(f"Missing feature-engineering dependency: {exc}") from exc
    return pd, np


def unique_features(features: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for feature in features:
        if feature not in seen:
            seen.add(feature)
            result.append(feature)
    return result


def build_experimental_feature_columns(base_features: Iterable[str]) -> list[str]:
    return unique_features([*base_features, *INCLUDED_EXPERIMENTAL_FEATURES])


def validate_no_forbidden_features(feature_columns: Iterable[str]) -> None:
    forbidden = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(feature_columns))
    if forbidden:
        raise ValueError(f"Forbidden future/target columns found in feature set: {forbidden}")


def _normalized_market_frame(market_df):
    pd, _ = require_runtime()
    df = market_df.copy()
    if "Date" not in df.columns:
        raise ValueError("Market frame is missing Date column.")

    rename_map = {}
    if "TSLA_Open" not in df.columns and "Open" in df.columns:
        rename_map["Open"] = "TSLA_Open"
    if "TSLA_High" not in df.columns and "High" in df.columns:
        rename_map["High"] = "TSLA_High"
    if "TSLA_Low" not in df.columns and "Low" in df.columns:
        rename_map["Low"] = "TSLA_Low"
    if "TSLA_Close" not in df.columns and "Close" in df.columns:
        rename_map["Close"] = "TSLA_Close"
    if "TSLA_Volume" not in df.columns and "Volume" in df.columns:
        rename_map["Volume"] = "TSLA_Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["Date", "TSLA_Open", "TSLA_High", "TSLA_Low", "TSLA_Close", "TSLA_Volume", "QQQ_Close"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Market frame is missing columns required for TSLA experimental features: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.normalize()
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def compute_experimental_features_from_market(market_df):
    pd, np = require_runtime()
    market = _normalized_market_frame(market_df)

    tsla_return_1 = market["TSLA_Close"].pct_change(1)
    qqq_return_1 = market["QQQ_Close"].pct_change(1)
    beta_cov = tsla_return_1.rolling(window=60, min_periods=20).cov(qqq_return_1)
    beta_var = qqq_return_1.rolling(window=60, min_periods=20).var().replace(0, np.nan)
    rolling_beta_to_qqq = beta_cov / beta_var
    volume_mean_20 = market["TSLA_Volume"].rolling(window=20, min_periods=20).mean()
    volume_std_20 = market["TSLA_Volume"].rolling(window=20, min_periods=20).std().replace(0, np.nan)
    volume_zscore_20 = (market["TSLA_Volume"] - volume_mean_20) / volume_std_20

    features = pd.DataFrame(
        {
            "Date": market["Date"],
            "tsla_residual_return_vs_qqq": tsla_return_1 - rolling_beta_to_qqq * qqq_return_1,
            "gap_open_prev_close": market["TSLA_Open"] / market["TSLA_Close"].shift(1) - 1,
            "volume_shock_20": (volume_zscore_20.abs() >= 2).astype(int),
            "rolling_volatility_20": tsla_return_1.rolling(window=20, min_periods=20).std(),
            "distance_to_20d_high": market["TSLA_Close"]
            / market["TSLA_Close"].rolling(window=20, min_periods=20).max()
            - 1,
            "distance_to_20d_low": market["TSLA_Close"]
            / market["TSLA_Close"].rolling(window=20, min_periods=20).min()
            - 1,
        }
    )
    return features.replace([np.inf, -np.inf], np.nan)


def load_experimental_feature_frame(market_path: Path = DEFAULT_MARKET_PATH):
    pd, _ = require_runtime()
    if not market_path.exists():
        raise FileNotFoundError(f"Market OHLCV cache is missing: {market_path}")
    market = pd.read_csv(market_path)
    return compute_experimental_features_from_market(market)


def add_experimental_features(df, market_path: Path = DEFAULT_MARKET_PATH):
    feature_frame = load_experimental_feature_frame(market_path)
    output = df.copy()
    pd, _ = require_runtime()
    output["Date"] = pd.to_datetime(output["Date"], errors="raise").dt.normalize()
    for column in INCLUDED_EXPERIMENTAL_FEATURES:
        if column in output.columns:
            output = output.drop(columns=[column])
    return output.merge(feature_frame, on="Date", how="left", validate="many_to_one")
