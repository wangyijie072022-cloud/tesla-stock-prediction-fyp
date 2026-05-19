# Tesla Next-Day Stock Movement Direction Prediction

## Project Objective

This final year project studies **next-trading-day Tesla stock movement direction prediction** using market features, news sentiment features, and time-aware machine learning evaluation. It predicts directional labels, not exact prices, and should not be interpreted as financial advice.

## Current Project Structure

```text
.
├── notebooks/
│   └── TSLA_Data_Collection_Preprocessing.ipynb
├── scripts/
│   ├── common.py
│   ├── model_io.py
│   ├── tsla_experimental_features.py
│   ├── train_and_save_model.py
│   ├── predict_for_date.py
│   ├── build_latest_prediction_input.py
│   ├── predict_latest.py
│   └── fetch_latest_doubao_news.py
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── latest/
│   └── sample/
├── models/
├── results/
├── logs/
└── requirements.txt
```

## Roles

- `notebooks/TSLA_Data_Collection_Preprocessing.ipynb` is the final runnable report. It displays data loading, feature preparation, strict final holdout evaluation, OOF threshold tuning, walk-forward Historical Replay, live cutoff checks, latest prediction, and artifact summaries.
- `scripts/*.py` contains the authoritative implementation. The notebook imports these functions instead of copying training, replay, live cutoff, model loading, or feature logic.
- `app.py` is the Streamlit dashboard entry point. It uses the same script-backed prediction functions and artifact paths.
- `data/raw/` contains source and cache inputs. These should not be deleted during cleanup.
- `data/processed/` contains model-ready historical datasets used by training and replay.
- `data/latest/` contains dashboard/CLI/notebook latest prediction artifacts.
- `models/` contains the current trained model, feature list, selected threshold, and metadata.
- `results/` contains current evaluation artifacts used by the report/dashboard.

## Core Scripts

- `scripts/common.py`: shared U.S. market calendar, trading-day, early-close, and cutoff helpers.
- `scripts/model_io.py`: shared model, threshold, metadata, and feature-column loading.
- `scripts/tsla_experimental_features.py`: shared feature construction used by training and prediction.
- `scripts/train_and_save_model.py`: strict time-series training and evaluation pipeline.
  - Splits the final 20% holdout before parameter selection.
  - Runs parameter search, OOF validation, and threshold tuning only on the pre-holdout window.
  - Evaluates the strict holdout once after selection and tuning.
  - Saves model artifacts and current metrics.
- `scripts/predict_for_date.py`: date-driven prediction.
  - Historical mode uses prior-only walk-forward replay.
  - Batch replay marks early dates as `skipped_insufficient_history`.
  - Live mode enforces latest closed market date constraints.
- `scripts/build_latest_prediction_input.py`: builds the latest live input row using closed-market cutoff logic.
- `scripts/predict_latest.py`: loads the current model through `model_io.py` and predicts from a prepared latest input row.
- `scripts/fetch_latest_doubao_news.py`: optional latest-news fetcher for the dashboard/live supplement.

## Current Artifacts

Keep these files unless intentionally regenerating them:

```text
models/tsla_direction_model.pkl
models/feature_columns.json
models/selected_threshold.json
models/model_metadata.json
results/threshold_tuning_validation_oof.csv
results/tsla_production_candidate_metrics.csv
results/tsla_production_candidate_summary.md
data/latest/tsla_latest_prediction_input.csv
data/latest/tsla_latest_prediction_output.csv
data/latest/tsla_prediction_for_date.csv
```

`results/alignment_audit.csv` is retained for dashboard audit display and historical data-alignment inspection. It is not the final model metric source.

## Data Sources

The checked-in local data supports offline report runs:

- `data/raw/kaggle_tesla_news_2020_2022_raw.csv`
- `data/raw/alphavantage_news_2023_2024_raw.csv`
- `data/raw/tesla_news_2020_2024_combined_raw.csv`
- `data/raw/market_ohlcv_2020_2024.csv`
- `data/processed/tsla_fused_dataset.csv`
- `data/processed/tesla_daily_sentiment_updated.csv`
- `data/processed/tsla_processed_with_indicators.csv`

Credentials are not stored in this repository. If optional latest-news fetching is needed, provide credentials through environment variables.

## Running the Report Notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run:

```text
notebooks/TSLA_Data_Collection_Preprocessing.ipynb
```

The notebook calls script functions directly:

- `run_training_pipeline()`
- `run_prediction_for_date()`
- `run_historical_batch_replay()`
- `build_latest_prediction_input()`
- `run_latest_prediction()`
- `load_model_bundle()`

It does not contain a duplicated exploratory model pipeline.

## CLI Commands

Train and regenerate model artifacts:

```bash
python scripts/train_and_save_model.py
```

Historical walk-forward prediction for one date:

```bash
python scripts/predict_for_date.py --mode historical --date 2024-12-30
```

Historical batch replay with insufficient-history rows marked:

```bash
python scripts/predict_for_date.py --mode historical --batch-start 2020-01-31 --batch-end 2020-02-05
```

Build latest live input without changing market cache:

```bash
python scripts/build_latest_prediction_input.py
```

Build latest live input and refresh market cache:

```bash
python scripts/build_latest_prediction_input.py --update-market
```

Predict from latest input:

```bash
python scripts/predict_latest.py
```

Run dashboard:

```bash
streamlit run app.py
```

## Leakage Controls

- The final 20% holdout is split before model selection, OOF validation, and threshold tuning.
- Threshold tuning uses pre-holdout OOF probabilities only.
- OOF dates and final holdout dates are asserted not to overlap.
- Historical Replay is walk-forward and uses only rows before the target prediction date.
- Live prediction uses `scripts/common.py` trading calendar helpers and does not treat an unclosed or early-close trading day as a complete OHLCV row.
- Early historical replay rows with fewer than 120 prior samples are not predicted; batch outputs mark them as `skipped_insufficient_history`.

## Interpretation

The current model is an academic research artifact. Metrics should be discussed as directional-classification experiment results over the selected data period, not as evidence of a reliable trading edge.
