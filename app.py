from __future__ import annotations

import html
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from inspect import signature
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from scripts.predict_for_date import PredictionError, run_prediction_for_date
from scripts.tsla_experimental_features import add_experimental_features


BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "fused": "data/processed/tsla_fused_dataset.csv",
    "sentiment": "data/processed/tesla_daily_sentiment_updated.csv",
    "indicators": "data/processed/tsla_processed_with_indicators.csv",
    "alignment": "results/alignment_audit.csv",
    "threshold": "results/threshold_tuning_validation_oof.csv",
    "latest": "data/latest/tsla_latest_prediction_input.csv",
    "prediction_output": "data/latest/tsla_latest_prediction_output.csv",
    "date_prediction": "data/latest/tsla_prediction_for_date.csv",
    "market": "data/raw/market_ohlcv_2020_2024.csv",
    "kaggle_raw": "data/raw/kaggle_tesla_news_2020_2022_raw.csv",
    "alphavantage_raw": "data/raw/alphavantage_news_2023_2024_raw.csv",
    "combined_raw": "data/raw/tesla_news_2020_2024_combined_raw.csv",
    "doubao_raw": "data/raw/doubao_tesla_latest_news_raw.csv",
    "model": "models/tsla_direction_model.pkl",
    "feature_columns": "models/feature_columns.json",
    "selected_threshold": "models/selected_threshold.json",
    "model_metadata": "models/model_metadata.json",
    "readme": "README.md",
}

MODEL_ARTIFACT_PATTERNS = [
    "models/**/*.pkl",
    "models/**/*.pickle",
    "models/**/*.joblib",
    "models/**/*.pt",
    "models/**/*.onnx",
    "artifacts/**/*.pkl",
    "artifacts/**/*.pickle",
    "artifacts/**/*.joblib",
    "artifacts/**/*.pt",
    "artifacts/**/*.onnx",
    "results/**/*.pkl",
    "results/**/*.pickle",
    "results/**/*.joblib",
    "results/**/*.pt",
    "results/**/*.onnx",
]

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score"]
DATE_CANDIDATES = ["Date", "date", "published_date", "effective_date", "market_feature_date"]
DATAFRAME_SUPPORTS_WIDTH = "width" in signature(st.dataframe).parameters
CONSERVATIVE_THRESHOLD = 0.5
LOW_CONFIDENCE_MIN = 0.45
LOW_CONFIDENCE_MAX = 0.55
UP_BIAS_WARNING_RATIO = 0.70
RESEARCH_THRESHOLD_OPTION = "Research threshold"
CONSERVATIVE_THRESHOLD_OPTION = "Conservative threshold 0.5"


@dataclass(frozen=True)
class CsvResult:
    path: Path
    rel_path: str
    exists: bool
    df: pd.DataFrame | None = None
    error: str | None = None


def abs_path(rel_path: str) -> Path:
    return BASE_DIR / rel_path


def format_mtime(path: Path) -> str:
    if not path.exists():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def file_status_row(label: str, rel_path: str, role: str, required: str) -> dict[str, object]:
    path = abs_path(rel_path)
    return {
        "file": label,
        "path": rel_path,
        "status": "available" if path.exists() else "missing",
        "required_status": required,
        "role": role,
        "last_modified": format_mtime(path),
        "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else None,
    }


@st.cache_data(show_spinner=False)
def read_csv_cached(path_string: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_string)


def load_csv(rel_path: str) -> CsvResult:
    path = abs_path(rel_path)
    if not path.exists():
        return CsvResult(path=path, rel_path=rel_path, exists=False, error=f"Missing file: {rel_path}")

    try:
        df = read_csv_cached(str(path), path.stat().st_mtime_ns)
    except pd.errors.EmptyDataError:
        return CsvResult(path=path, rel_path=rel_path, exists=True, error=f"Empty CSV: {rel_path}")
    except pd.errors.ParserError as exc:
        return CsvResult(path=path, rel_path=rel_path, exists=True, error=f"CSV parse error in {rel_path}: {exc}")
    except Exception as exc:  # Streamlit should report the file problem instead of crashing.
        return CsvResult(path=path, rel_path=rel_path, exists=True, error=f"Could not read {rel_path}: {exc}")

    return CsvResult(path=path, rel_path=rel_path, exists=True, df=df)


def show_csv_issue(result: CsvResult) -> bool:
    if result.error:
        st.warning(result.error)
        return True
    return False


def parse_date_series(df: pd.DataFrame, candidates: Iterable[str] = DATE_CANDIDATES) -> tuple[pd.Series | None, str | None]:
    for column in candidates:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
            if parsed.notna().any():
                return parsed, column
    return None, None


def date_range_text(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "not available"
    parsed, _ = parse_date_series(df)
    if parsed is None:
        return "date column not available"
    valid = parsed.dropna()
    if valid.empty:
        return "date values not available"
    return f"{valid.min().date()} to {valid.max().date()}"


def metric_value(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def numeric_text(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def display_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    text = str(value).strip()
    return text if text else "n/a"


def status_text(value: object) -> str:
    parsed = boolish(value)
    if parsed is True:
        return "Yes"
    if parsed is False:
        return "No"
    return display_text(value)


def float_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentage_text(value: object, digits: int = 1) -> str:
    parsed = float_value(value)
    if parsed is None:
        return "n/a"
    return f"{parsed * 100:.{digits}f}%"


def is_low_confidence_probability(probability_up: float | None) -> bool:
    return probability_up is not None and LOW_CONFIDENCE_MIN <= probability_up <= LOW_CONFIDENCE_MAX


def confidence_level_text(probability_up: float | None) -> str:
    if probability_up is None:
        return "n/a"
    if is_low_confidence_probability(probability_up):
        return "Low confidence"
    if abs(probability_up - CONSERVATIVE_THRESHOLD) >= 0.20:
        return "High confidence"
    return "Moderate confidence"


def predicted_direction_for_threshold(probability_up: float | None, threshold: float | None) -> str:
    if probability_up is None or threshold is None:
        return "n/a"
    return "up" if probability_up >= threshold else "down"


def load_research_threshold_value() -> float | None:
    path = abs_path(FILES["selected_threshold"])
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_value = payload.get("selected_threshold") if isinstance(payload, dict) else payload
    return float_value(raw_value)


def value_counts_text(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    return ", ".join(f"{label}: {count}" for label, count in counts.items())


def show_dataframe(df: pd.DataFrame, hide_index: bool = True) -> None:
    if DATAFRAME_SUPPORTS_WIDTH:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    else:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def boolish(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def model_artifacts() -> list[Path]:
    artifacts: list[Path] = []
    for pattern in MODEL_ARTIFACT_PATTERNS:
        artifacts.extend(path for path in BASE_DIR.glob(pattern) if path.is_file())
    return sorted(set(artifacts))


def model_file_rows() -> list[dict[str, object]]:
    return [
        file_status_row("Model pickle", FILES["model"], "Trained prediction model", "required for latest prediction"),
        file_status_row("Feature columns", FILES["feature_columns"], "Training feature order", "required for latest prediction"),
        file_status_row("Selected threshold", FILES["selected_threshold"], "Validation-selected decision threshold", "required for latest prediction"),
        file_status_row("Model metadata", FILES["model_metadata"], "Training metadata and validation metrics", "required for latest prediction"),
    ]


def classification_metric_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    positives = y_true == 1
    negatives = y_true == 0
    predicted_positive = y_pred == 1
    predicted_negative = y_pred == 0

    tp = int((positives & predicted_positive).sum())
    tn = int((negatives & predicted_negative).sum())
    fp = int((negatives & predicted_positive).sum())
    fn = int((positives & predicted_negative).sum())
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def saved_model_replay_diagnostics(decision_threshold: float) -> tuple[dict[str, object] | None, str | None]:
    fused = load_csv(FILES["fused"])
    if fused.error:
        return None, fused.error
    if fused.df is None or fused.df.empty:
        return None, "Historical dataset is not available."

    model_path = abs_path(FILES["model"])
    feature_path = abs_path(FILES["feature_columns"])
    if not model_path.exists():
        return None, f"Missing model artifact: {FILES['model']}"
    if not feature_path.exists():
        return None, f"Missing feature artifact: {FILES['feature_columns']}"

    try:
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read feature columns: {exc}"
    if isinstance(payload, list):
        feature_columns = payload
    elif isinstance(payload, dict) and isinstance(payload.get("feature_columns"), list):
        feature_columns = payload["feature_columns"]
    else:
        return None, "feature_columns.json does not contain a feature list."

    df = fused.df.copy()
    if "Target" not in df.columns:
        return None, "Historical dataset is missing Target column."
    try:
        df = add_experimental_features(df, abs_path(FILES["market"]))
    except Exception as exc:
        return None, f"Could not construct TSLA experimental replay features: {exc}"
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        return None, f"Historical dataset is missing model features: {missing_columns[:8]}"

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    y = pd.to_numeric(df["Target"], errors="coerce")
    valid_target = y.isin([0, 1])
    X = X.loc[valid_target]
    y = y.loc[valid_target].astype(int)
    if X.empty or y.empty:
        return None, "Historical dataset has no usable Target rows."
    if X.isna().to_numpy().any():
        missing_value_columns = X.columns[X.isna().any(axis=0)].tolist()
        return None, f"Historical feature matrix has missing values: {missing_value_columns[:8]}"

    try:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
    except OSError as exc:
        return None, f"Could not read model artifact: {exc}"
    if not hasattr(model, "predict_proba"):
        return None, "Model artifact does not support predict_proba."

    probabilities = pd.Series(model.predict_proba(X)[:, 1], index=X.index, dtype=float)
    predicted = (probabilities >= decision_threshold).astype(int)
    always_up = pd.Series(1, index=y.index, dtype=int)
    metric_values = classification_metric_summary(y, predicted)
    always_up_metrics = classification_metric_summary(y, always_up)

    dates = pd.to_datetime(df.loc[y.index, "Date"], errors="coerce") if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    date_start = dates.min().date().isoformat() if not dates.dropna().empty else ""
    date_end = dates.max().date().isoformat() if not dates.dropna().empty else ""

    return {
        "rows": int(len(y)),
        "date_start": date_start,
        "date_end": date_end,
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "probability_median": float(probabilities.median()),
        "probability_q25": float(probabilities.quantile(0.25)),
        "probability_q75": float(probabilities.quantile(0.75)),
        "predicted_up_count": int((predicted == 1).sum()),
        "predicted_down_count": int((predicted == 0).sum()),
        "predicted_up_ratio": float((predicted == 1).mean()),
        "predicted_down_ratio": float((predicted == 0).mean()),
        "actual_up_count": int((y == 1).sum()),
        "actual_down_count": int((y == 0).sum()),
        "actual_up_ratio": float((y == 1).mean()),
        "actual_down_ratio": float((y == 0).mean()),
        "low_confidence_ratio": float(((probabilities >= LOW_CONFIDENCE_MIN) & (probabilities <= LOW_CONFIDENCE_MAX)).mean()),
        "always_up_accuracy": always_up_metrics["accuracy"],
        "always_up_precision": always_up_metrics["precision"],
        "always_up_recall": always_up_metrics["recall"],
        "always_up_f1_score": always_up_metrics["f1_score"],
        **metric_values,
    }, None


def key_file_rows() -> list[dict[str, object]]:
    return [
        file_status_row("Fused modelling dataset", FILES["fused"], "Main technical + sentiment modelling table", "required output"),
        file_status_row("Daily sentiment", FILES["sentiment"], "Aggregated daily news sentiment features", "required output"),
        file_status_row("Technical indicators", FILES["indicators"], "Technical-only feature table", "required output"),
        file_status_row("Alignment audit", FILES["alignment"], "News timestamp and trading-day alignment audit", "required result"),
        file_status_row("Threshold tuning", FILES["threshold"], "Validation OOF threshold tuning metrics", "required result"),
        file_status_row("Kaggle raw news", FILES["kaggle_raw"], "2020-2022 local raw news cache required by README", "required raw source"),
        file_status_row("Alpha Vantage raw news", FILES["alphavantage_raw"], "2023-2024 local raw news cache required by README", "required raw source"),
        file_status_row("Combined raw news", FILES["combined_raw"], "Merged raw news cache generated by notebook rerun", "required raw source"),
        file_status_row("Market OHLCV cache", FILES["market"], "Offline market cache for TSLA, QQQ, SPY and VIX", "required for offline rerun"),
        file_status_row("Doubao latest news", FILES["doubao_raw"], "Optional latest-news supplement", "optional supplement"),
        file_status_row("Latest prediction input", FILES["latest"], "Optional live-input construction artifact", "optional supplement"),
        file_status_row("Latest prediction output", FILES["prediction_output"], "Latest local model prediction output", "optional generated output"),
        file_status_row("Date prediction output", FILES["date_prediction"], "Date-driven prediction output", "optional generated output"),
        file_status_row("Sample README", "data/sample/README.md", "Documents optional demo/sample directory only", "optional documentation"),
        *model_file_rows(),
    ]


def section_header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tesla-red: #E82127;
            --ink: #171A20;
            --muted: #5C6470;
            --line: #E6E8EC;
            --panel: #FFFFFF;
            --soft: #F4F5F7;
            --success: #128A48;
        }

        .stApp {
            background: linear-gradient(180deg, #FBFBFC 0%, #F3F4F6 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1rem 0.85rem;
            box-shadow: 0 8px 22px rgba(23, 26, 32, 0.05);
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: #111318;
            border-right: 1px solid #242832;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #E9ECEF;
        }

        .sidebar-brand {
            border-bottom: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 1.1rem;
            padding: 0.35rem 0 1rem;
        }

        .sidebar-logo {
            color: #FFFFFF;
            font-weight: 800;
            font-size: 1.55rem;
            line-height: 1.15;
        }

        .sidebar-accent {
            color: var(--tesla-red);
        }

        .sidebar-subtitle {
            color: #A8AFBA;
            font-size: 0.84rem;
            line-height: 1.35;
            margin-top: 0.45rem;
        }

        .tsla-hero {
            background:
                radial-gradient(circle at top right, rgba(232,33,39,0.14), transparent 34%),
                linear-gradient(135deg, #FFFFFF 0%, #F6F7F9 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 16px 38px rgba(23, 26, 32, 0.08);
            padding: 1.65rem 1.75rem;
            margin-bottom: 1.45rem;
        }

        .hero-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            flex-wrap: wrap;
        }

        .word-logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 132px;
            border: 2px solid var(--tesla-red);
            color: var(--tesla-red);
            border-radius: 6px;
            padding: 0.42rem 0.7rem;
            font-size: 1.75rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }

        .hero-copy h1 {
            margin: 0.8rem 0 0.4rem;
            font-size: 2.1rem;
            line-height: 1.15;
            font-weight: 800;
        }

        .hero-copy p {
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
        }

        .research-pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(232,33,39,0.28);
            background: rgba(232,33,39,0.08);
            color: #B4151B;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(23, 26, 32, 0.055);
            padding: 1rem;
            min-height: 142px;
            margin-bottom: 0.35rem;
        }

        .metric-card-title {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.45rem;
        }

        .metric-card-value {
            color: var(--ink);
            font-size: 1.18rem;
            line-height: 1.25;
            font-weight: 800;
            overflow-wrap: anywhere;
            margin-bottom: 0.6rem;
        }

        .metric-card-caption {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.35;
        }

        .section-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(23, 26, 32, 0.045);
            padding: 1.15rem 1.25rem;
            margin: 1rem 0 1.25rem;
        }

        .section-panel h3 {
            margin: 0 0 0.65rem;
            font-size: 1.18rem;
        }

        .section-panel p {
            color: #3F4650;
            line-height: 1.65;
            margin: 0;
        }

        .status-card {
            border-radius: 8px;
            padding: 1rem 1.15rem;
            border: 1px solid rgba(18,138,72,0.24);
            background: #F0FAF4;
            color: #0B5B31;
            margin: 0.65rem 0 1rem;
        }

        .status-card.warning {
            border-color: rgba(232,33,39,0.20);
            background: #FFF4F4;
            color: #9A181D;
        }

        .styled-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(23, 26, 32, 0.045);
            margin: 0.65rem 0 1.1rem;
        }

        .styled-table th {
            background: #F3F4F6;
            color: #3E4651;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            text-align: left;
            padding: 0.78rem 0.85rem;
            border-bottom: 1px solid var(--line);
        }

        .styled-table td {
            padding: 0.78rem 0.85rem;
            border-bottom: 1px solid #EEF0F3;
            color: #2E3440;
            vertical-align: top;
            font-size: 0.92rem;
            overflow-wrap: anywhere;
        }

        .styled-table tr:last-child td {
            border-bottom: none;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid #CCD2DA;
            background: #F8F9FB;
            color: #49515C;
            white-space: nowrap;
        }

        .status-badge.available {
            border-color: rgba(18,138,72,0.22);
            background: #EAF7EF;
            color: var(--success);
        }

        .status-badge.missing {
            border-color: rgba(232,33,39,0.22);
            background: #FFF0F1;
            color: #B4151B;
        }

        .table-note {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.45;
            margin: -0.2rem 0 0.35rem;
        }

        @media (max-width: 900px) {
            .hero-copy h1 {
                font-size: 1.65rem;
            }
            .word-logo {
                font-size: 1.35rem;
                min-width: 110px;
            }
            .metric-card {
                min-height: 128px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return html.escape(str(value))


def render_hero() -> None:
    st.markdown(
        """
        <div class="tsla-hero">
          <div class="hero-row">
            <div>
              <div class="word-logo">TESLA</div>
              <div class="hero-copy">
                <h1>TSLA Next-Day Stock Movement Prediction</h1>
                <p>Technical indicators + FinBERT-based news sentiment + time-aware validation</p>
              </div>
            </div>
            <div class="research-pill">Research use only</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-card-title">{html_text(title)}</div>
          <div class="metric-card-value">{html_text(value)}</div>
          <div class="metric-card-caption">{html_text(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="section-panel">
          <h3>{html_text(title)}</h3>
          <p>{body_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(value: object) -> str:
    text = html_text(value)
    css_class = text.strip().lower().replace(" ", "-")
    return f'<span class="status-badge {css_class}">{text}</span>'


def render_status_message(success: bool, success_text: str, warning_text: str) -> None:
    if success:
        card_class = "status-card"
        body = success_text
    else:
        card_class = "status-card warning"
        body = warning_text
    st.markdown(f'<div class="{card_class}">{html_text(body)}</div>', unsafe_allow_html=True)


def render_status_table(rows: list[dict[str, object]]) -> None:
    columns = ["file", "path", "status", "required_status", "role", "last_modified", "size_kb"]
    headers = "".join(f"<th>{html_text(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            if column == "status":
                cells.append(f"<td>{render_status_badge(row.get(column))}</td>")
            else:
                cells.append(f"<td>{html_text(row.get(column))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<table class="styled-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_file_update_table() -> None:
    rows = [
        file_status_row("Fused dataset", FILES["fused"], "processed", "required output"),
        file_status_row("Daily sentiment", FILES["sentiment"], "processed", "required output"),
        file_status_row("Technical indicators", FILES["indicators"], "processed", "required output"),
        file_status_row("Alignment audit", FILES["alignment"], "results", "required result"),
        file_status_row("Threshold tuning", FILES["threshold"], "results", "required result"),
    ]
    render_status_table(rows)


def render_overview() -> None:
    render_hero()

    fused = load_csv(FILES["fused"])
    sentiment = load_csv(FILES["sentiment"])
    threshold = load_csv(FILES["threshold"])
    alignment = load_csv(FILES["alignment"])
    latest = load_csv(FILES["latest"])

    required_rows = key_file_rows()
    required_available = all(row["status"] == "available" for row in required_rows if str(row["required_status"]).startswith("required"))
    artifacts = model_artifacts()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            "Fused date range",
            date_range_text(fused.df),
            "Main modelling table used for the direction-classification experiment.",
        )
    with col2:
        render_metric_card(
            "Daily sentiment range",
            date_range_text(sentiment.df),
            "Aggregated Tesla news sentiment coverage for model features.",
        )
    with col3:
        render_metric_card(
            "Offline reproducibility inputs",
            "available" if required_available else "incomplete",
            "Required raw caches, processed outputs, and result artifacts.",
        )
    with col4:
        render_metric_card(
            "Model artifact",
            f"{len(artifacts)} found" if artifacts else "not available",
            "Saved model and metadata available for local review.",
        )

    render_section_card(
        "Project Summary",
        (
            "This Final Year Project studies <strong>Tesla</strong> "
            "<strong>next-day movement direction</strong> prediction using "
            "<strong>technical market features</strong>, "
            "<strong>FinBERT-based sentiment</strong>, "
            "<strong>time-aware validation</strong>, and "
            "<strong>threshold tuning</strong>. The dashboard is designed as a "
            "read-only research interface over existing local CSV artifacts and saved model outputs."
        ),
    )

    st.subheader("Processed / Results Update Time")
    st.markdown(
        '<div class="table-note">Availability and modified time for the core processed datasets and evaluation artifacts used by the dashboard.</div>',
        unsafe_allow_html=True,
    )
    render_file_update_table()

    st.subheader("Reproducibility Status")
    render_status_message(
        required_available,
        "Required raw caches, processed datasets, and result CSVs are present for local review and offline rerun preparation.",
        "Some required files are missing. The dashboard can still render available sections, but full local reproducibility is incomplete.",
    )

    if artifacts:
        show_dataframe(pd.DataFrame({"model_artifact": [str(path.relative_to(BASE_DIR)) for path in artifacts]}))
    else:
        st.info("model artifact not available")

    st.subheader("Data Leakage Fix Status")
    st.write(
        "README states the notebook was rerun on 2026-05-15 using source-aware news merge logic, "
        "conservative trading-day alignment, and TimeSeriesSplit(gap=1). "
        "The audit below summarizes the available alignment evidence."
    )
    if show_csv_issue(alignment):
        return

    audit = alignment.df
    summary_rows = []
    if audit is not None and not audit.empty:
        summary_rows.append({"check": "alignment_audit_rows", "value": len(audit)})
        for column in ["source_dataset", "has_intraday_time", "was_remapped_to_next_date", "is_after_hours", "is_weekend_published"]:
            if column in audit.columns:
                summary_rows.append({"check": column, "value": value_counts_text(audit[column])})
        summary_df = pd.DataFrame(summary_rows)
        summary_df["value"] = summary_df["value"].astype(str)
        show_dataframe(summary_df)
    else:
        st.info("Alignment audit is empty.")

    st.subheader("Latest Input Status")
    if latest.error:
        st.info("Latest prediction input is optional and is not currently available.")
    elif latest.df is not None and not latest.df.empty:
        row = latest.df.iloc[0]
        can_predict = boolish(row.get("can_predict_with_historical_model"))
        status = row.get("feature_alignment_status", "n/a")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("prediction_cutoff", str(row.get("prediction_cutoff", "n/a")))
        col_b.metric("feature_alignment_status", str(status))
        col_c.metric("can_predict_with_historical_model", str(row.get("can_predict_with_historical_model", "n/a")))
        if can_predict is False:
            st.warning("This row is an input-construction artifact, not a formal prediction result.")

    if threshold.error:
        st.info("Threshold tuning result is not available.")


def render_data() -> None:
    section_header("Data", "Dataset shape, missing values, sentiment trends and TSLA market movement.")

    fused = load_csv(FILES["fused"])
    sentiment = load_csv(FILES["sentiment"])
    market = load_csv(FILES["market"])
    indicators = load_csv(FILES["indicators"])

    if not show_csv_issue(fused):
        df = fused.df
        assert df is not None
        col1, col2, col3 = st.columns(3)
        col1.metric("Fused rows", f"{len(df):,}")
        col2.metric("Fused columns", f"{len(df.columns):,}")
        col3.metric("Date range", date_range_text(df))

        st.subheader("Missing Values")
        missing = (
            df.isna()
            .sum()
            .reset_index()
            .rename(columns={"index": "column", 0: "missing_count"})
        )
        missing["missing_pct"] = (missing["missing_count"] / max(len(df), 1) * 100).round(2)
        missing = missing.sort_values(["missing_count", "column"], ascending=[False, True])
        show_dataframe(missing)

    st.subheader("Daily Sentiment Trend")
    if not show_csv_issue(sentiment):
        sentiment_df = sentiment.df
        assert sentiment_df is not None
        parsed, date_col = parse_date_series(sentiment_df)
        candidate_cols = [
            col
            for col in ["weighted_sentiment_score", "daily_sentiment_score", "net_sentiment", "intraday_weighted_sentiment_score"]
            if col in sentiment_df.columns
        ]
        if parsed is None or not candidate_cols:
            st.info("Sentiment trend columns are not available.")
        else:
            chart_df = sentiment_df.assign(_date=parsed.dt.date).set_index("_date")[candidate_cols]
            st.line_chart(chart_df)
            st.caption(f"Date column: {date_col}")

    st.subheader("TSLA Close / Return")
    price_df = fused.df if fused.df is not None else indicators.df if indicators.df is not None else market.df
    if price_df is None:
        st.warning("No TSLA price dataset is available.")
    else:
        parsed, date_col = parse_date_series(price_df)
        close_col = "Close" if "Close" in price_df.columns else "TSLA_Close" if "TSLA_Close" in price_df.columns else None
        return_col = "Return_1" if "Return_1" in price_df.columns else None
        if parsed is None or close_col is None:
            st.info("TSLA close chart columns are not available.")
        else:
            chart_base = pd.DataFrame(index=parsed.dt.date)
            chart_base["TSLA close"] = pd.to_numeric(price_df[close_col], errors="coerce").to_numpy()
            st.line_chart(chart_base[["TSLA close"]])
            if return_col:
                chart_base["TSLA return"] = pd.to_numeric(price_df[return_col], errors="coerce").to_numpy()
                st.line_chart(chart_base[["TSLA return"]])
            st.caption(f"Date column: {date_col}")

    st.subheader("Source Dataset Snapshots")
    snapshot_rows = []
    for label, result in [
        ("fused", fused),
        ("sentiment", sentiment),
        ("indicators", indicators),
        ("market", market),
    ]:
        if result.df is not None:
            snapshot_rows.append(
                {
                    "dataset": label,
                    "path": result.rel_path,
                    "rows": len(result.df),
                    "columns": len(result.df.columns),
                    "date_range": date_range_text(result.df),
                    "last_modified": format_mtime(result.path),
                }
            )
        else:
            snapshot_rows.append(
                {
                    "dataset": label,
                    "path": result.rel_path,
                    "rows": None,
                    "columns": None,
                    "date_range": result.error or "not available",
                    "last_modified": format_mtime(result.path),
                }
            )
    show_dataframe(pd.DataFrame(snapshot_rows))


def replay_summary_row(label: str, threshold: float, diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        "threshold_mode": label,
        "threshold": round(threshold, 3),
        "rows": diagnostics["rows"],
        "date_range": f"{diagnostics['date_start']} to {diagnostics['date_end']}",
        "prob_min": round(float(diagnostics["probability_min"]), 3),
        "prob_max": round(float(diagnostics["probability_max"]), 3),
        "prob_mean": round(float(diagnostics["probability_mean"]), 3),
        "prob_median": round(float(diagnostics["probability_median"]), 3),
        "prob_25pct": round(float(diagnostics["probability_q25"]), 3),
        "prob_75pct": round(float(diagnostics["probability_q75"]), 3),
        "predicted_up_ratio": percentage_text(diagnostics["predicted_up_ratio"]),
        "predicted_down_ratio": percentage_text(diagnostics["predicted_down_ratio"]),
        "low_confidence_probability_ratio": percentage_text(diagnostics["low_confidence_ratio"]),
    }


def render_saved_model_replay_diagnostics() -> None:
    st.subheader("Saved Model In-Sample Diagnostics")
    st.caption(
        "This uses the final saved model on rows it was fitted on, so it is not a backtest or OOF validation. "
        "Accuracy-style metrics are intentionally not shown here."
    )

    research_threshold = load_research_threshold_value()
    threshold_specs = [(CONSERVATIVE_THRESHOLD_OPTION, CONSERVATIVE_THRESHOLD)]
    if research_threshold is not None:
        threshold_specs.append((RESEARCH_THRESHOLD_OPTION, research_threshold))

    rows = []
    for label, threshold in threshold_specs:
        diagnostics, error = saved_model_replay_diagnostics(threshold)
        if error:
            st.warning(error)
            continue
        assert diagnostics is not None
        if float(diagnostics["predicted_up_ratio"]) > UP_BIAS_WARNING_RATIO:
            st.warning("Model is biased toward UP predictions.")
        rows.append(replay_summary_row(label, threshold, diagnostics))

    if rows:
        show_dataframe(pd.DataFrame(rows))
    else:
        st.info("Saved-model replay diagnostics are not available.")


def render_model_evaluation() -> None:
    section_header("Model Evaluation", "Validation threshold tuning and alignment audit artifacts.")

    threshold = load_csv(FILES["threshold"])
    alignment = load_csv(FILES["alignment"])

    st.subheader("Threshold Tuning")
    if not show_csv_issue(threshold):
        threshold_df = threshold.df
        assert threshold_df is not None
        show_dataframe(threshold_df)

        selected = pd.DataFrame()
        if "selected_probability_threshold" in threshold_df.columns:
            selected = threshold_df[threshold_df["selected_probability_threshold"].astype(str).str.lower().isin(["1", "true"])]
        if selected.empty and "f1_score" in threshold_df.columns:
            selected = threshold_df.sort_values("f1_score", ascending=False).head(1)

        if not selected.empty:
            row = selected.iloc[0]
            threshold_value = row.get("probability_threshold", "n/a")
            threshold_float = float_value(threshold_value)
            research_threshold = load_research_threshold_value()
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Selected threshold", metric_value(threshold_value, 2))
            col2.metric("Accuracy", metric_value(row.get("accuracy")))
            col3.metric("Precision", metric_value(row.get("precision")))
            col4.metric("Recall", metric_value(row.get("recall")))
            col5.metric("F1 score", metric_value(row.get("f1_score")))
            if (
                threshold_float is not None
                and research_threshold is not None
                and abs(threshold_float - research_threshold) > 1e-9
            ):
                st.warning(
                    f"Threshold CSV marks {threshold_float:.2f} as selected, "
                    f"but the saved model artifact uses {research_threshold:.2f}."
                )
        else:
            st.info("Selected threshold row is not available.")

        chart_cols = [col for col in METRIC_COLUMNS if col in threshold_df.columns]
        if "probability_threshold" in threshold_df.columns and chart_cols:
            chart_df = threshold_df.set_index("probability_threshold")[chart_cols]
            st.line_chart(chart_df)
        else:
            st.info("Metric chart columns are not available.")

    render_saved_model_replay_diagnostics()

    st.subheader("Alignment Audit")
    if not show_csv_issue(alignment):
        alignment_df = alignment.df
        assert alignment_df is not None
        show_dataframe(alignment_df)

        summary_rows = []
        for column in ["source_dataset", "has_intraday_time", "is_after_hours", "was_remapped_to_next_date"]:
            if column in alignment_df.columns:
                counts = alignment_df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
                counts["field"] = column
                counts[column] = counts[column].astype(str)
                summary_rows.append(counts[["field", column, "count"]].rename(columns={column: "value"}))
        if summary_rows:
            alignment_summary_df = pd.concat(summary_rows, ignore_index=True)
            alignment_summary_df["value"] = alignment_summary_df["value"].astype(str)
            show_dataframe(alignment_summary_df)


def run_command(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", *args],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def parse_json_objects(text: str) -> list[object]:
    decoder = json.JSONDecoder()
    objects: list[object] = []
    index = 0
    while index < len(text):
        next_indexes = [pos for pos in (text.find("{", index), text.find("[", index)) if pos != -1]
        if not next_indexes:
            break
        index = min(next_indexes)
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        objects.append(parsed)
        index += end
    return objects


def latest_json_payload(stdout: str) -> dict[str, object] | None:
    for item in reversed(parse_json_objects(stdout)):
        if isinstance(item, dict) and "success" in item:
            return item
    return None


def looks_like_json_text(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def command_error_summary(result: subprocess.CompletedProcess[str], payload: dict[str, object] | None) -> str:
    if payload:
        for field in ["error", "message", "warning"]:
            value = payload.get(field)
            if value:
                return str(value)

    for source in [result.stderr, result.stdout]:
        if looks_like_json_text(source):
            continue
        for line in source.splitlines():
            stripped = line.strip()
            if stripped and stripped not in {"{", "}", "[", "]"}:
                return stripped[:240]

    return f"Command failed with exit code {result.returncode}."


def show_technical_details(result: subprocess.CompletedProcess[str], payload: dict[str, object] | None = None) -> None:
    with st.expander("Technical details", expanded=False):
        st.write(f"Exit code: {result.returncode}")
        if payload:
            st.write(f"Parsed output fields: {', '.join(payload.keys())}")
            if payload.get("error"):
                st.write(f"Error: {payload['error']}")
            st.caption("Raw JSON output is hidden from the dashboard view.")
        elif result.stdout.strip():
            if looks_like_json_text(result.stdout):
                st.caption("Raw JSON output is hidden from the dashboard view.")
            else:
                st.text(result.stdout.strip()[:3000])
        if result.stderr.strip():
            if looks_like_json_text(result.stderr):
                st.caption("Raw error JSON is hidden from the dashboard view.")
            else:
                st.text(result.stderr.strip()[:3000])


def show_command_result(
    result: subprocess.CompletedProcess[str],
    success_message: str,
    failure_prefix: str = "Operation failed.",
) -> dict[str, object] | None:
    payload = latest_json_payload(result.stdout)
    if result.returncode == 0:
        st.success(success_message)
    else:
        st.error(f"{failure_prefix} {command_error_summary(result, payload)}")
        show_technical_details(result, payload)
    return payload


def show_latest_news_result(result: subprocess.CompletedProcess[str]) -> dict[str, object] | None:
    payload = latest_json_payload(result.stdout)
    if result.returncode != 0:
        st.error(f"Latest news refresh failed. {command_error_summary(result, payload)}")
        show_technical_details(result, payload)
        return payload

    st.success("Latest news refreshed successfully.")
    warnings = payload.get("warnings", []) if payload else []
    failed_queries = payload.get("failed_queries", []) if payload else []
    final_rows = payload.get("final_rows", "n/a") if payload else "n/a"

    col1, col2, col3 = st.columns(3)
    col1.metric("Final rows", display_text(final_rows))
    col2.metric("Failed query count", str(len(failed_queries) if isinstance(failed_queries, list) else 0))
    col3.metric("Warning count", str(len(warnings) if isinstance(warnings, list) else 0))
    return payload


def clear_csv_cache() -> None:
    read_csv_cached.clear()


def refresh_prediction_output(selected_date: date, mode: str, decision_threshold: float, threshold_mode: str) -> bool:
    try:
        run_prediction_for_date(
            selected_date.isoformat(),
            mode,
            abs_path(FILES["date_prediction"]),
            decision_threshold=decision_threshold,
            threshold_mode=threshold_mode,
        )
    except PredictionError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return False

    clear_csv_cache()
    st.success("Prediction output refreshed.")
    return True


WARNING_TEXT_OVERRIDES = {
    "latest_news_cutoff_is_after_latest_market_date; update market data before real inference": (
        "Latest news cutoff is after the latest market date; update market data before real inference."
    ),
    "market_cache_is_stale_relative_to_prediction_cutoff; consider --update-market": (
        "Market cache is stale relative to the prediction cutoff; consider updating market data."
    ),
    "publish_time_utc_empty_used_fetched_at": (
        "Some news rows did not include publish_time_utc, so fetched_at was used."
    ),
    "publish_time_utc_unparseable_used_fetched_at": (
        "Some news publish_time_utc values could not be parsed, so fetched_at was used."
    ),
    "publish_time_utc_empty_and_fetched_at_unparseable": (
        "Some news rows did not have a usable publish_time_utc or fetched_at timestamp."
    ),
    "no_news_at_or_before_prediction_cutoff": (
        "No latest-news rows were available at or before the prediction cutoff."
    ),
}


def humanize_warning_message(message: object) -> str:
    text = str(message).strip()
    if not text:
        return ""
    if text in WARNING_TEXT_OVERRIDES:
        return WARNING_TEXT_OVERRIDES[text]
    if "_" in text and " " not in text:
        text = text.replace("_", " ")
    return text[:1].upper() + text[1:] if text else text


def warning_messages(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        messages: list[str] = []
        for item in value:
            messages.extend(warning_messages(item))
        return list(dict.fromkeys(messages))
    if isinstance(value, dict):
        query = display_text(value.get("query"))
        message = value.get("warning") or value.get("error") or value.get("message")
        if message:
            text = humanize_warning_message(message)
            return [f"{query}: {text}" if query != "n/a" else text]
        return []
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text[0] in {"[", "{"}:
        try:
            return warning_messages(json.loads(text))
        except json.JSONDecodeError:
            pass

    parts = []
    for part in text.replace("\n", " | ").split("|"):
        cleaned = humanize_warning_message(part)
        if cleaned:
            parts.append(cleaned)
    return list(dict.fromkeys(parts))


def render_warnings(value: object) -> None:
    messages = warning_messages(value)
    if not messages:
        st.info("No warnings recorded.")
        return
    st.warning("\n".join(f"- {message}" for message in messages))


def render_prediction_result(
    output_df: pd.DataFrame,
    decision_threshold: float,
    threshold_mode: str,
    research_threshold: float | None,
    replay_diagnostics: dict[str, object] | None,
) -> None:
    if output_df.empty:
        st.info("Prediction output CSV exists but contains no rows.")
        return

    output_row = output_df.iloc[-1]
    probability_up = float_value(output_row.get("probability_up"))
    margin = probability_up - decision_threshold if probability_up is not None else None
    confidence_level = confidence_level_text(probability_up)
    predicted_direction = predicted_direction_for_threshold(probability_up, decision_threshold)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability up", numeric_text(probability_up, 3))
    col2.metric("Decision threshold", numeric_text(decision_threshold, 2))
    col3.metric("Margin", numeric_text(margin, 3))
    col4.metric("Confidence level", confidence_level)

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Predicted direction", predicted_direction)
    col6.metric("Threshold mode", threshold_mode)
    col7.metric("Research threshold", numeric_text(research_threshold, 2))
    col8.metric("Prediction allowed", status_text(output_row.get("prediction_allowed")))

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Requested date", display_text(output_row.get("requested_date")))
    col10.metric("Input trading date", display_text(output_row.get("input_trading_date")))
    col11.metric("Next trading day", display_text(output_row.get("next_trading_day")))
    col12.metric("Market data latest date", display_text(output_row.get("market_data_latest_date")))

    st.caption(f"Date mapping status: {display_text(output_row.get('date_mapping_status'))}")

    output_threshold = float_value(output_row.get("selected_threshold"))
    if output_threshold is not None and abs(output_threshold - decision_threshold) > 1e-9:
        st.info("Displayed decision threshold follows the current UI selection. Run prediction again to update the output CSV.")

    if boolish(output_row.get("market_data_stale")) is True:
        st.error("Market data stale. Please click Update Market Data before running a formal prediction.")

    context = str(output_row.get("prediction_context", ""))
    if "experimental" in context or "partial" in context:
        st.warning("Experimental / partial context prediction.")
    if decision_threshold < LOW_CONFIDENCE_MIN:
        st.warning("Decision threshold is below 0.5; UP predictions may be frequent.")
    if threshold_mode == "research":
        st.warning("Research threshold is tuned for validation F1 and is not a high-confidence trading signal.")
    if is_low_confidence_probability(probability_up):
        st.warning("Low confidence.")
    if replay_diagnostics and float(replay_diagnostics.get("predicted_up_ratio", 0.0)) > UP_BIAS_WARNING_RATIO:
        st.warning("Model is biased toward UP predictions.")

    st.subheader("Warnings")
    render_warnings(output_row.get("warnings", ""))


def render_latest_prediction_input() -> None:
    section_header("TSLA Experimental Prediction", "Research model, not financial advice.")
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.75rem;margin:0.25rem 0 1rem 0;">
          <div style="border:1px solid #d33;border-radius:6px;padding:0.35rem 0.65rem;font-weight:800;color:#d33;letter-spacing:0.08em;">TSLA</div>
          <div>
            <div style="font-weight:700;">Tesla / TSLA research-only model</div>
            <div style="font-size:0.9rem;color:#666;">Experimental output for academic diagnostics. Not a reliable trading signal.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("Research model, not financial advice.")
    st.warning("Prediction is probabilistic, experimental, and may be wrong.")

    st.subheader("Model Artifact Status")
    model_status_df = pd.DataFrame(model_file_rows())
    show_dataframe(model_status_df)

    st.subheader("Prediction Controls")
    research_threshold = load_research_threshold_value()
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        selected_date = st.date_input("Input date", value=date.today())
    with control_col2:
        mode_label = st.radio("Mode", ["Historical Replay", "Live Prediction"], horizontal=True, index=1)
    with control_col3:
        threshold_label = st.radio(
            "Decision threshold",
            [CONSERVATIVE_THRESHOLD_OPTION, RESEARCH_THRESHOLD_OPTION],
            horizontal=False,
            index=0,
        )

    mode = "historical" if mode_label == "Historical Replay" else "live"
    threshold_mode = "research" if threshold_label == RESEARCH_THRESHOLD_OPTION else "conservative"
    if threshold_mode == "research":
        if research_threshold is None:
            st.warning("Research threshold artifact is missing; using conservative threshold 0.5.")
            decision_threshold = CONSERVATIVE_THRESHOLD
            threshold_mode = "conservative"
        else:
            decision_threshold = research_threshold
            st.warning("Research threshold is tuned for validation F1 and is not a high-confidence trading signal.")
    else:
        decision_threshold = CONSERVATIVE_THRESHOLD

    st.caption(
        f"Recommended conservative threshold: {CONSERVATIVE_THRESHOLD:.2f}. "
        f"Saved research threshold: {numeric_text(research_threshold, 2)}."
    )
    if decision_threshold < LOW_CONFIDENCE_MIN:
        st.warning("Decision threshold is below 0.5; UP predictions may be frequent.")

    replay_diagnostics, replay_error = saved_model_replay_diagnostics(decision_threshold)
    if replay_error:
        st.info(f"Saved-model replay diagnostic unavailable: {replay_error}")
    elif replay_diagnostics and float(replay_diagnostics["predicted_up_ratio"]) > UP_BIAS_WARNING_RATIO:
        st.warning("Model is biased toward UP predictions.")

    button_col1, button_col2, button_col3 = st.columns(3)
    with button_col1:
        run_prediction_clicked = st.button("Run Prediction", type="primary")
    with button_col2:
        update_market_clicked = st.button("Update Market Data")
    with button_col3:
        refresh_news_clicked = st.button("Refresh Latest News")

    had_date_prediction = abs_path(FILES["date_prediction"]).exists()

    if run_prediction_clicked:
        refresh_prediction_output(selected_date, mode, decision_threshold, threshold_mode)

    if update_market_clicked:
        cutoff = f"{selected_date.isoformat()}T23:59:59+00:00"
        with st.spinner("Updating market cache and rebuilding latest input. This may use yfinance."):
            result = run_command(
                [
                    "scripts/build_latest_prediction_input.py",
                    "--prediction-cutoff",
                    cutoff,
                    "--update-market",
                    "--output",
                    FILES["latest"],
                ]
            )
        clear_csv_cache()
        show_command_result(result, "Market data updated successfully.", "Market data update failed.")
        if result.returncode == 0:
            refresh_prediction_output(selected_date, mode, decision_threshold, threshold_mode)

    if refresh_news_clicked:
        cutoff = f"{selected_date.isoformat()}T23:59:59+00:00"
        with st.spinner("Refreshing latest news with Doubao, then rebuilding latest input without market update."):
            fetch_result = run_command(["scripts/fetch_latest_doubao_news.py"])
        show_latest_news_result(fetch_result)
        clear_csv_cache()
        if fetch_result.returncode == 0:
            with st.spinner("Rebuilding latest input."):
                build_result = run_command(
                    [
                        "scripts/build_latest_prediction_input.py",
                        "--prediction-cutoff",
                        cutoff,
                        "--output",
                        FILES["latest"],
                    ]
                )
            clear_csv_cache()
            if build_result.returncode != 0:
                show_command_result(build_result, "Latest input rebuilt successfully.", "Latest input rebuild failed.")
            elif had_date_prediction:
                refresh_prediction_output(selected_date, mode, decision_threshold, threshold_mode)

    st.subheader("Date Prediction Output")
    date_prediction = load_csv(FILES["date_prediction"])
    if not show_csv_issue(date_prediction):
        output_df = date_prediction.df
        assert output_df is not None
        render_prediction_result(output_df, decision_threshold, threshold_mode, research_threshold, replay_diagnostics)
    else:
        st.info("Choose a date and mode, then click Run Prediction.")

    st.subheader("Latest Prediction Input")
    latest = load_csv(FILES["latest"])
    if show_csv_issue(latest):
        st.info("This optional file can be generated by the existing input builder, but the dashboard does not run that builder.")
        return

    latest_df = latest.df
    assert latest_df is not None
    if latest_df.empty:
        st.warning("Latest prediction input CSV exists but contains no rows.")
        return

    row = latest_df.iloc[0]
    can_predict_value = row.get("can_predict_with_historical_model", "n/a")
    can_predict = boolish(can_predict_value)

    col1, col2, col3 = st.columns(3)
    col1.metric("prediction_cutoff", str(row.get("prediction_cutoff", "n/a")))
    col2.metric("feature_alignment_status", str(row.get("feature_alignment_status", "n/a")))
    col3.metric("can_predict_with_historical_model", str(can_predict_value))

    if can_predict is False:
        st.warning("This row is an input-construction artifact, not a formal prediction result.")
    elif can_predict is True:
        artifacts = model_artifacts()
        if artifacts:
            st.success("Input row is marked as compatible and a local model artifact was found.")
        else:
            st.info("Input row is marked as compatible, but model artifact not available.")
    else:
        st.info("Prediction compatibility status is not available.")

    artifacts = model_artifacts()
    if artifacts:
        show_dataframe(pd.DataFrame({"model_artifact": [str(path.relative_to(BASE_DIR)) for path in artifacts]}))
    else:
        st.info("model artifact not available")

    metadata_columns = [
        "requested_date",
        "input_trading_date",
        "next_trading_day",
        "date_mapping_status",
        "market_data_latest_date",
        "latest_closed_market_date",
        "market_data_stale",
        "prediction_allowed",
        "prediction_cutoff",
        "market_feature_date",
        "market_feature_cutoff_date",
        "target_horizon",
        "latest_news_rows_used",
        "source_news_path",
        "source_market_path",
        "historical_model_feature_count",
        "missing_feature_count",
        "missing_feature_columns",
        "feature_alignment_status",
        "can_predict_with_historical_model",
        "blocking_context_warnings",
        "feature_context_warnings",
    ]
    available_metadata = [col for col in metadata_columns if col in latest_df.columns]
    if available_metadata:
        st.subheader("Status Fields")
        status_fields = latest_df[available_metadata].T.rename(columns={0: "value"})
        status_fields["value"] = status_fields["value"].astype(str)
        show_dataframe(status_fields, hide_index=False)

    st.subheader("Latest Input Row")
    show_dataframe(latest_df)


def render_files_reproducibility() -> None:
    section_header("Files / Reproducibility", "Local file availability and README data-source status.")

    rows = key_file_rows()
    status_df = pd.DataFrame(rows)
    st.subheader("Key File Status")
    show_dataframe(status_df)

    required_df = status_df[status_df["required_status"].astype(str).str.startswith("required")]
    optional_df = status_df[status_df["required_status"].astype(str).str.startswith("optional")]

    col1, col2, col3 = st.columns(3)
    col1.metric("Required files available", f"{(required_df['status'] == 'available').sum()} / {len(required_df)}")
    col2.metric("Optional supplement files available", f"{(optional_df['status'] == 'available').sum()} / {len(optional_df)}")
    col3.metric("Model artifact", "available" if model_artifacts() else "not available")

    st.subheader("README Data Source Status")
    st.write(
        "README requires local Kaggle news, Alpha Vantage raw news, combined raw news, and the market OHLCV cache "
        "for offline raw-to-output reproduction. The dashboard checks file presence only; it does not refresh or download sources."
    )
    readme_sources = status_df[
        status_df["file"].isin(["Kaggle raw news", "Alpha Vantage raw news", "Combined raw news", "Market OHLCV cache"])
    ]
    show_dataframe(readme_sources)

    st.subheader("Optional Supplements")
    st.write(
        "Doubao latest news and latest prediction input are optional supplement artifacts. "
        "They are not part of the historical model evaluation pipeline."
    )
    show_dataframe(optional_df)


def main() -> None:
    st.set_page_config(
        page_title="TSLA Stock Movement Prediction Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo"><span class="sidebar-accent">TSLA</span> Prediction Dashboard</div>
          <div class="sidebar-subtitle">Research model, not financial advice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Data", "Model Evaluation", "Latest Prediction", "Files / Reproducibility"],
    )

    if page == "Overview":
        render_overview()
    elif page == "Data":
        render_data()
    elif page == "Model Evaluation":
        render_model_evaluation()
    elif page == "Latest Prediction":
        render_latest_prediction_input()
    else:
        render_files_reproducibility()


if __name__ == "__main__":
    main()
