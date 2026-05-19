#!/usr/bin/env python3
"""Build one latest TSLA next-trading-day prediction input row.

This script is for live inference preparation only. It does not train a model
and does not write data/processed or results artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    closed_market_cutoff_note,
    latest_closed_trading_day,
    next_available_date,
    next_calendar_trading_day,
)
from scripts.tsla_experimental_features import (
    INCLUDED_EXPERIMENTAL_FEATURES,
    build_experimental_feature_columns,
    compute_experimental_features_from_market,
)

LATEST_NEWS_PATH = PROJECT_ROOT / "data/raw/doubao_tesla_latest_news_raw.csv"
MARKET_CACHE_PATH = PROJECT_ROOT / "data/raw/market_ohlcv_2020_2024.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/latest/tsla_latest_prediction_input.csv"
MARKET_TICKERS = ["TSLA", "QQQ", "SPY", "^VIX"]
MARKET_PREFIX_MAP = {"TSLA": "TSLA", "QQQ": "QQQ", "SPY": "SPY", "^VIX": "VIX"}
MARKET_FIELD_MAP = {
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Adj Close": "Adj_Close",
    "Volume": "Volume",
}
FINBERT_MODEL = "ProsusAI/finbert"

TECHNICAL_COLUMNS = [
    "MA7", "MA14", "EMA12", "EMA26",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "Return_1", "Return_3", "Return_5",
    "Volume_Change", "Volatility_5", "Volatility_10",
    "Price_Range", "Open_Close_Return",
    "Gap_Return",
    "MA7_Gap", "MA14_Gap", "EMA12_Gap", "EMA26_Gap",
    "MA_Cross", "EMA_Cross",
    "Momentum_10", "Momentum_20",
    "Volume_Ratio_5", "Volume_Ratio_10",
    "OBV_Change_5",
    "ATR_Ratio_14",
    "Bollinger_Width_20", "Bollinger_Position_20",
    "Stoch_K_14", "Stoch_D_3",
    "QQQ_Return_1", "QQQ_Return_5", "QQQ_Volatility_5",
    "SPY_Return_1", "SPY_Return_5",
    "VIX_Return_1", "VIX_Change_5", "VIX_Relative_10",
    "TSLA_vs_QQQ_Return_1", "TSLA_vs_SPY_Return_1",
]

SENTIMENT_FEATURE_COLUMNS = [
    "weighted_sentiment_score",
    "net_sentiment",
    "news_count",
    "negative_ratio",
    "news_available",
    "weighted_sentiment_score_lag1",
    "net_sentiment_lag1",
    "weighted_sentiment_score_3d_mean",
    "weighted_sentiment_score_5d_mean",
    "news_count_3d_sum",
    "negative_ratio_3d_mean",
    "strong_negative_count_3d_sum",
]

PHASE1_RELATIVE_PRICE_COLUMNS = ["close_position_in_day", "body_to_range"]
PHASE1_SENTIMENT_ENHANCEMENT_COLUMNS = ["sentiment_surprise_3", "abnormal_news_flag"]
PHASE1_INTERACTION_COLUMNS = ["sentiment_x_volatility5", "net_sentiment_x_news"]
HISTORICAL_MODEL_FEATURE_COLUMNS = (
    TECHNICAL_COLUMNS
    + PHASE1_RELATIVE_PRICE_COLUMNS
    + SENTIMENT_FEATURE_COLUMNS
    + PHASE1_SENTIMENT_ENHANCEMENT_COLUMNS
    + PHASE1_INTERACTION_COLUMNS
)
EXPERIMENTAL_MODEL_FEATURE_COLUMNS = build_experimental_feature_columns(HISTORICAL_MODEL_FEATURE_COLUMNS)


def fail(message: str, exit_code: int = 1, **extra: Any) -> None:
    payload = {"success": False, "error": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


def require_runtime():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        fail(
            "Missing required runtime dependency. Install requirements.txt or run with the project .venv.",
            missing_dependency=str(exc),
        )
    return pd, np


def parse_cutoff(value: str | None):
    pd, _ = require_runtime()
    if not value:
        return pd.Timestamp.now(tz="UTC")
    cutoff = pd.to_datetime(value, errors="raise", utc=True)
    return cutoff


def prepare_market_model_frame(cache_df):
    pd, _ = require_runtime()
    required_cache_cols = ["Date"] + [
        f"{prefix}_{field}"
        for prefix in ["TSLA", "QQQ", "SPY", "VIX"]
        for field in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
    ]
    missing_cols = [col for col in required_cache_cols if col not in cache_df.columns]
    if missing_cols:
        fail("Market cache is missing required columns.", missing_columns=missing_cols)

    model_df = cache_df[
        [
            "Date", "TSLA_Open", "TSLA_High", "TSLA_Low", "TSLA_Close", "TSLA_Volume",
            "QQQ_Close", "SPY_Close", "VIX_Close",
        ]
    ].copy()
    model_df = model_df.rename(
        columns={
            "TSLA_Open": "Open",
            "TSLA_High": "High",
            "TSLA_Low": "Low",
            "TSLA_Close": "Close",
            "TSLA_Volume": "Volume",
        }
    )
    model_df["Date"] = pd.to_datetime(model_df["Date"]).dt.normalize()
    return model_df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def compute_technical_features(market_df):
    pd, np = require_runtime()
    df = market_df.sort_values("Date").reset_index(drop=True).copy()
    df["MA7"] = df["Close"].rolling(window=7, min_periods=7).mean()
    df["MA14"] = df["Close"].rolling(window=14, min_periods=14).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "RSI"] = 0
    df.loc[(avg_gain == 0) & (avg_loss == 0), "RSI"] = 50

    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volatility_5"] = df["Close"].pct_change().rolling(window=5, min_periods=5).std()
    df["Volatility_10"] = df["Close"].pct_change().rolling(window=10, min_periods=10).std()
    df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Return"] = (df["Close"] - df["Open"]) / df["Open"]

    df["Gap_Return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["MA7_Gap"] = (df["Close"] - df["MA7"]) / df["MA7"]
    df["MA14_Gap"] = (df["Close"] - df["MA14"]) / df["MA14"]
    df["EMA12_Gap"] = (df["Close"] - df["EMA12"]) / df["EMA12"]
    df["EMA26_Gap"] = (df["Close"] - df["EMA26"]) / df["EMA26"]
    df["MA_Cross"] = (df["MA7"] - df["MA14"]) / df["MA14"]
    df["EMA_Cross"] = (df["EMA12"] - df["EMA26"]) / df["EMA26"]
    df["Momentum_10"] = df["Close"].pct_change(10)
    df["Momentum_20"] = df["Close"].pct_change(20)
    df["Volume_Ratio_5"] = df["Volume"] / df["Volume"].rolling(window=5, min_periods=5).mean()
    df["Volume_Ratio_10"] = df["Volume"] / df["Volume"].rolling(window=10, min_periods=10).mean()

    price_direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (price_direction * df["Volume"]).cumsum()
    df["OBV_Change_5"] = df["OBV"].pct_change(5)

    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14, min_periods=14).mean()
    df["ATR_Ratio_14"] = df["ATR_14"] / df["Close"]

    bb_mid = df["Close"].rolling(window=20, min_periods=20).mean()
    bb_std = df["Close"].rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["Bollinger_Width_20"] = (bb_upper - bb_lower) / bb_mid
    df["Bollinger_Position_20"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)

    lowest_low_14 = df["Low"].rolling(window=14, min_periods=14).min()
    highest_high_14 = df["High"].rolling(window=14, min_periods=14).max()
    df["Stoch_K_14"] = 100 * (df["Close"] - lowest_low_14) / (highest_high_14 - lowest_low_14)
    df["Stoch_D_3"] = df["Stoch_K_14"].rolling(window=3, min_periods=3).mean()

    df["QQQ_Return_1"] = df["QQQ_Close"].pct_change(1)
    df["QQQ_Return_5"] = df["QQQ_Close"].pct_change(5)
    df["QQQ_Volatility_5"] = df["QQQ_Close"].pct_change().rolling(window=5, min_periods=5).std()
    df["SPY_Return_1"] = df["SPY_Close"].pct_change(1)
    df["SPY_Return_5"] = df["SPY_Close"].pct_change(5)
    df["VIX_Return_1"] = df["VIX_Close"].pct_change(1)
    df["VIX_Change_5"] = df["VIX_Close"].pct_change(5)
    df["VIX_Relative_10"] = (df["VIX_Close"] / df["VIX_Close"].rolling(window=10, min_periods=10).mean()) - 1
    df["TSLA_vs_QQQ_Return_1"] = df["Return_1"] - df["QQQ_Return_1"]
    df["TSLA_vs_SPY_Return_1"] = df["Return_1"] - df["SPY_Return_1"]
    experimental = compute_experimental_features_from_market(df)
    df = df.merge(experimental[["Date", *INCLUDED_EXPERIMENTAL_FEATURES]], on="Date", how="left", validate="one_to_one")
    return df.replace([np.inf, -np.inf], np.nan)


def download_market_cache(existing_cache):
    pd, _ = require_runtime()
    try:
        import yfinance as yf
    except ImportError as exc:
        fail("yfinance is required for --update-market.", missing_dependency=str(exc))

    start_date = pd.to_datetime(existing_cache["Date"]).min().date().isoformat()
    end_date = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    raw = yf.download(
        tickers=MARKET_TICKERS,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    if raw.empty:
        fail("yfinance returned no market data.")

    cache_df = pd.DataFrame({"Date": pd.to_datetime(raw.index).tz_localize(None).date})
    for ticker in MARKET_TICKERS:
        if ticker not in raw.columns.get_level_values(0):
            fail("Missing ticker in yfinance response.", ticker=ticker)
        ticker_df = raw[ticker]
        prefix = MARKET_PREFIX_MAP[ticker]
        for yf_field, out_field in MARKET_FIELD_MAP.items():
            if yf_field not in ticker_df.columns:
                fail("Missing field in yfinance response.", ticker=ticker, field=yf_field)
            cache_df[f"{prefix}_{out_field}"] = ticker_df[yf_field].values
    return cache_df.sort_values("Date").reset_index(drop=True)


def parse_news_timestamp(row, cutoff):
    pd, _ = require_runtime()
    warnings = []
    raw_utc_value = row.get("publish_time_utc", "")
    raw_utc = "" if pd.isna(raw_utc_value) else str(raw_utc_value).strip()
    if raw_utc:
        ts = pd.to_datetime(raw_utc, errors="coerce", utc=True)
        if pd.notna(ts):
            return ts, warnings
        warnings.append("publish_time_utc_unparseable_used_fetched_at")

    fetched_at_value = row.get("fetched_at", "")
    fetched_at = "" if pd.isna(fetched_at_value) else str(fetched_at_value).strip()
    ts = pd.to_datetime(fetched_at, errors="coerce", utc=True)
    if pd.isna(ts):
        warnings.append("publish_time_utc_empty_and_fetched_at_unparseable")
        return cutoff + pd.Timedelta(days=36500), warnings
    warnings.append("publish_time_utc_empty_used_fetched_at")
    return ts, warnings


def run_finbert(texts):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        fail("transformers and torch are required to compute FinBERT sentiment for live input.", missing_dependency=str(exc))

    try:
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL,
            local_files_only=True,
            use_safetensors=False,
        )
    except OSError as exc:
        fail(
            "FinBERT model files are not available in the local Hugging Face cache. "
            "This script does not download model files automatically.",
            model=FINBERT_MODEL,
            detail=str(exc),
        )
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline(texts, truncation=True)


def aggregate_latest_news(news_df, cutoff):
    pd, np = require_runtime()
    warnings: list[str] = []
    if news_df.empty:
        return build_zero_sentiment_features(), ["no_latest_news_rows_available"]

    timestamps = []
    for _, row in news_df.iterrows():
        ts, row_warnings = parse_news_timestamp(row, cutoff)
        timestamps.append(ts)
        warnings.extend(row_warnings)

    df = news_df.copy()
    df["news_timestamp_utc"] = timestamps
    df = df[df["news_timestamp_utc"] <= cutoff].copy()
    if df.empty:
        return build_zero_sentiment_features(), warnings + ["no_news_at_or_before_prediction_cutoff"]

    texts = (df["title"].fillna("").astype(str) + ". " + df["summary"].fillna("").astype(str)).str.strip().tolist()
    sentiment_results = run_finbert(texts)
    labels = [item["label"].lower() for item in sentiment_results]
    confidence = np.array([float(item["score"]) for item in sentiment_results])
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    scores = np.array([score_map.get(label, 0) for label in labels])
    weighted_scores = scores * confidence
    n = len(labels)
    positive_count = sum(1 for label in labels if label == "positive")
    negative_count = sum(1 for label in labels if label == "negative")
    neutral_count = sum(1 for label in labels if label == "neutral")

    features = {
        "daily_sentiment_score": float(scores.mean()) if n else 0.0,
        "weighted_sentiment_score": float(weighted_scores.mean()) if n else 0.0,
        "positive_ratio": positive_count / n if n else 0.0,
        "negative_ratio": negative_count / n if n else 0.0,
        "neutral_ratio": neutral_count / n if n else 0.0,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "news_count": n,
        "avg_confidence": float(confidence.mean()) if n else 0.0,
        "sentiment_std": float(np.std(scores, ddof=1)) if n > 1 else 0.0,
        "strong_positive_count": int(sum(1 for label, conf in zip(labels, confidence) if label == "positive" and conf >= 0.90)),
        "strong_negative_count": int(sum(1 for label, conf in zip(labels, confidence) if label == "negative" and conf >= 0.90)),
    }
    features["net_sentiment"] = features["positive_ratio"] - features["negative_ratio"]
    return features, sorted(set(warnings))


def build_zero_sentiment_features():
    return {
        "daily_sentiment_score": 0.0,
        "weighted_sentiment_score": 0.0,
        "positive_ratio": 0.0,
        "negative_ratio": 0.0,
        "neutral_ratio": 0.0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "news_count": 0,
        "avg_confidence": 0.0,
        "sentiment_std": 0.0,
        "strong_positive_count": 0,
        "strong_negative_count": 0,
        "net_sentiment": 0.0,
    }


def build_live_feature_row(market_features, sentiment_features):
    pd, np = require_runtime()
    latest = market_features.iloc[-1].copy()
    eps = 1e-8
    row = latest.to_dict()
    row.update(sentiment_features)
    row["news_available"] = int(row.get("news_count", 0) > 0)
    row["no_news_flag"] = int(row["news_available"] == 0)

    # Only latest-news supplement is available here. Lag features cannot be
    # reconstructed from the full historical sentiment timeline, so keep them
    # explicit and conservative.
    row["weighted_sentiment_score_lag1"] = 0.0
    row["net_sentiment_lag1"] = 0.0
    row["weighted_sentiment_score_3d_mean"] = row["weighted_sentiment_score"]
    row["weighted_sentiment_score_5d_mean"] = row["weighted_sentiment_score"]
    row["news_count_3d_sum"] = row["news_count"]
    row["negative_ratio_3d_mean"] = row["negative_ratio"]
    row["strong_negative_count_3d_sum"] = row["strong_negative_count"]

    row["close_position_in_day"] = (row["Close"] - row["Low"]) / ((row["High"] - row["Low"]) + eps)
    row["body_to_range"] = (row["Close"] - row["Open"]) / ((row["High"] - row["Low"]) + eps)
    row["sentiment_surprise_3"] = row["weighted_sentiment_score"] - row["weighted_sentiment_score_3d_mean"]
    row["abnormal_news_flag"] = int(row["news_count"] > (row["news_count_3d_sum"] / 3))
    row["sentiment_x_volatility5"] = row["weighted_sentiment_score"] * row["Volatility_5"]
    row["net_sentiment_x_news"] = row["net_sentiment"] * row["news_available"]

    missing = [feature for feature in EXPERIMENTAL_MODEL_FEATURE_COLUMNS if feature not in row or pd.isna(row[feature])]
    for feature in missing:
        row[feature] = np.nan
    row["missing_feature_columns"] = "|".join(missing)
    return row, missing


def classify_feature_alignment(missing_features, warnings):
    blocking_fragments = [
        "market_cache_is_stale",
        "latest_news_cutoff_is_after_latest_market_date",
        "sentiment lag/rolling features",
    ]
    blocking_warnings = [
        warning
        for warning in sorted(set(warnings))
        if any(fragment in warning for fragment in blocking_fragments)
    ]
    if missing_features:
        return "partial_missing_features", 0, blocking_warnings
    if blocking_warnings:
        return "partial_context", 0, blocking_warnings
    return "full_match", 1, []


def parse_args():
    parser = argparse.ArgumentParser(description="Build one latest TSLA live prediction feature row.")
    parser.add_argument("--prediction-cutoff", help="UTC or timezone-aware cutoff timestamp. Default: now UTC.")
    parser.add_argument("--update-market", action="store_true", help="Update raw market cache with yfinance before building features.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}")
    return parser.parse_args()


def build_latest_prediction_input(
    prediction_cutoff: str | None = None,
    update_market: bool = False,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    pd, np = require_runtime()
    cutoff = parse_cutoff(prediction_cutoff)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not LATEST_NEWS_PATH.exists():
        fail("Latest Doubao news raw CSV is missing.", path=str(LATEST_NEWS_PATH))
    if not MARKET_CACHE_PATH.exists():
        fail("Market OHLCV cache is missing.", path=str(MARKET_CACHE_PATH))

    market_cache = pd.read_csv(MARKET_CACHE_PATH)
    if update_market:
        market_cache = download_market_cache(market_cache)
        market_cache.to_csv(MARKET_CACHE_PATH, index=False)

    model_market = prepare_market_model_frame(market_cache)
    cutoff_date = cutoff.tz_convert("UTC").date() if hasattr(cutoff, "tz_convert") else cutoff.date()
    market_dates = sorted({item.date() for item in model_market["Date"]})
    market_latest_date = max(market_dates)
    closed_market_date = latest_closed_trading_day(cutoff)
    market_data_stale = closed_market_date > market_latest_date
    market_feature_limit_date = market_latest_date if market_data_stale else closed_market_date
    market_close_note = closed_market_cutoff_note(cutoff, closed_market_date)
    available_market = model_market[model_market["Date"].dt.date <= market_feature_limit_date].copy()
    if available_market.empty:
        fail(
            "No market rows are available at or before the latest closed market cutoff.",
            prediction_cutoff=str(cutoff),
            latest_closed_market_date=closed_market_date.isoformat(),
        )

    technical_df = compute_technical_features(available_market)
    if technical_df[TECHNICAL_COLUMNS].tail(1).isna().any(axis=1).iloc[0]:
        missing = technical_df[TECHNICAL_COLUMNS].tail(1).columns[
            technical_df[TECHNICAL_COLUMNS].tail(1).isna().iloc[0]
        ].tolist()
        fail("Latest market row lacks required technical indicators.", missing_technical_columns=missing)

    latest_news = pd.read_csv(LATEST_NEWS_PATH)
    sentiment_features, news_warnings = aggregate_latest_news(latest_news, cutoff)
    feature_row, missing_features = build_live_feature_row(technical_df, sentiment_features)

    market_date = pd.to_datetime(feature_row["Date"]).date()
    next_market_date = next_available_date(market_date, market_dates) or next_calendar_trading_day(market_date)
    if market_data_stale:
        date_mapping_status = "market_data_stale"
        date_mapping_note = "Market data stale. Please click Update Market Data before running a formal prediction."
    elif market_date == closed_market_date and not market_close_note:
        date_mapping_status = "exact"
        date_mapping_note = ""
    elif market_date == closed_market_date:
        date_mapping_status = "mapped_to_latest_closed_trading_day"
        date_mapping_note = market_close_note
    else:
        date_mapping_status = "mapped_to_previous_available_trading_day"
        date_mapping_note = f"Input date is not a closed trading day; mapped to previous available trading day: {market_date.isoformat()}."
    warnings = list(news_warnings)
    if date_mapping_note:
        warnings.append(date_mapping_note)
    if (closed_market_date - market_date).days > 7:
        warnings.append("market_cache_is_stale_relative_to_prediction_cutoff; consider --update-market")
    if sentiment_features["news_count"] and market_date < cutoff_date:
        warnings.append("latest_news_cutoff_is_after_latest_market_date; update market data before real inference")
    warnings.append("sentiment lag/rolling features are latest-news-only approximations, not full historical sentiment context")
    alignment_status, can_predict, blocking_warnings = classify_feature_alignment(missing_features, warnings)

    output_row = {feature: feature_row.get(feature, np.nan) for feature in EXPERIMENTAL_MODEL_FEATURE_COLUMNS}
    metadata = {
        "requested_date": cutoff_date.isoformat(),
        "input_trading_date": market_date.isoformat(),
        "next_trading_day": next_market_date.isoformat(),
        "date_mapping_status": date_mapping_status,
        "market_data_latest_date": market_latest_date.isoformat(),
        "latest_closed_market_date": closed_market_date.isoformat(),
        "market_data_stale": int(market_data_stale),
        "prediction_allowed": int(bool(can_predict) and not market_data_stale),
        "prediction_cutoff": cutoff.isoformat(),
        "market_feature_date": pd.to_datetime(feature_row["Date"]).date().isoformat(),
        "market_feature_cutoff_date": market_feature_limit_date.isoformat(),
        "target_horizon": "next_trading_day_after_market_feature_date",
        "latest_news_rows_used": int(sentiment_features["news_count"]),
        "source_news_path": str(LATEST_NEWS_PATH),
        "source_market_path": str(MARKET_CACHE_PATH),
        "historical_model_feature_count": len(EXPERIMENTAL_MODEL_FEATURE_COLUMNS),
        "included_experimental_features": "|".join(INCLUDED_EXPERIMENTAL_FEATURES),
        "missing_feature_count": len(missing_features),
        "missing_feature_columns": "|".join(missing_features),
        "feature_alignment_status": alignment_status,
        "can_predict_with_historical_model": can_predict,
        "blocking_context_warnings": " | ".join(blocking_warnings),
        "feature_context_warnings": " | ".join(sorted(set(warnings))),
    }
    output_row.update(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([output_row]).to_csv(output_path, index=False)
    return {
        "success": True,
        "output_path": str(output_path),
        "prediction_cutoff": cutoff.isoformat(),
        "requested_date": metadata["requested_date"],
        "input_trading_date": metadata["input_trading_date"],
        "next_trading_day": metadata["next_trading_day"],
        "date_mapping_status": metadata["date_mapping_status"],
        "market_data_latest_date": metadata["market_data_latest_date"],
        "latest_closed_market_date": metadata["latest_closed_market_date"],
        "market_data_stale": bool(market_data_stale),
        "prediction_allowed": bool(metadata["prediction_allowed"]),
        "market_feature_date": metadata["market_feature_date"],
        "market_feature_cutoff_date": metadata["market_feature_cutoff_date"],
        "latest_news_rows_used": metadata["latest_news_rows_used"],
        "missing_feature_count": len(missing_features),
        "missing_feature_columns": missing_features,
        "feature_alignment_status": alignment_status,
        "can_predict_with_historical_model": bool(can_predict),
        "blocking_context_warnings": blocking_warnings,
        "warnings": sorted(set(warnings)),
    }


def main() -> int:
    args = parse_args()
    payload = build_latest_prediction_input(
        prediction_cutoff=args.prediction_cutoff,
        update_market=args.update_market,
        output_path=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
