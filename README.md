# Tesla Next-Day Stock Movement Direction Prediction Using News Sentiment and Machine Learning

## Project Objective

This repository contains a final year project that investigates **next-day Tesla stock movement direction prediction** (up/down), not exact future price forecasting. The study combines technical market features with financial news sentiment and evaluates whether sentiment information improves directional classification performance.

## Repository Structure

```text
.
├── notebooks/
│   └── TSLA_Data_Collection_Preprocessing.ipynb
├── data/
│   ├── raw/
│   │   └── alphavantage_news_2023_2024_raw.csv
│   └── processed/
│       ├── tesla_daily_sentiment_updated.csv
│       ├── tsla_fused_dataset.csv
│       └── tsla_processed_with_indicators.csv
├── logs/
│   └── alphavantage_fetch_log.csv
├── results/
│   ├── alignment_audit.csv
│   └── threshold_tuning_validation_oof.csv
└── README.md
```

### Folder Purpose

- **`notebooks/`**: Main end-to-end workflow for data preparation, feature engineering, modeling, and evaluation.
- **`data/raw/`**: Unprocessed collected news data.
- **`data/processed/`**: Cleaned/engineered datasets used for modeling.
- **`logs/`**: Data collection audit and fetch trace files.
- **`results/`**: Saved evaluation artifacts such as alignment checks and validation-based threshold tuning outputs.

## Data Sources

The workflow references and/or uses the following data sources:

1. **Tesla market data** for creating return-based targets and technical indicators.
2. **Financial news data** from:
   - A Tesla news sample dataset (2020-2022) path referenced in the notebook.
   - **Alpha Vantage News API** collection for 2023-2024 (`data/raw/alphavantage_news_2023_2024_raw.csv`).
3. **Derived sentiment and fused datasets** stored under `data/processed/`.

> Note: API-based collection requires valid credentials in the execution environment. Credentials are not stored in this repository.

## Methodology Overview

The project pipeline is designed to remain explainable and suitable for academic analysis:

1. **Market Data Preparation**
   - Build daily Tesla market features.
   - Generate technical indicators.
2. **Target Construction**
   - Construct a binary next-day movement label from next-day return direction using a fixed thresholding logic for clearer class definition.
3. **News Processing and Sentiment Analysis**
   - Clean and timestamp-align news with trading days.
   - Use **FinBERT** (`ProsusAI/finbert`) sentiment inference.
   - Aggregate article-level sentiment to daily features.
4. **Feature Fusion**
   - Merge technical and sentiment features into a unified modeling table (`tsla_fused_dataset.csv`).
5. **Ablation Design**
   - Compare technical-only and technical+sentiment feature group settings.

## Machine Learning Workflow

The notebook applies time-aware model development and comparison:

- **Validation strategy**: `TimeSeriesSplit` (expanding-window style) to reduce leakage risk.
- **Baseline and comparative models** include:
  - Logistic Regression
  - Random Forest
  - XGBoost (if available in environment)
  - LightGBM (if available in environment)
- **Model-selection utilities** include validation-based threshold tuning on out-of-fold probabilities.

## Evaluation Approach

The repository emphasizes classification evaluation for direction prediction:

- Fold-wise and aggregated metrics such as:
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - F1-score
- Confusion-matrix based interpretation.
- Validation out-of-fold threshold search (saved to `results/threshold_tuning_validation_oof.csv`).
- Alignment audit artifacts (saved to `results/alignment_audit.csv`) for checking data/label consistency.

This project does **not** claim exact price prediction; results should be interpreted as directional classification outcomes under the chosen dataset period and validation setup.

## How to Read or Run the Notebook

### Recommended Reading Order

1. Open `notebooks/TSLA_Data_Collection_Preprocessing.ipynb`.
2. Follow sections in order: data loading → preprocessing → sentiment inference → feature fusion → modeling → evaluation.
3. Inspect CSV outputs in `data/processed/`, `logs/`, and `results/` to verify intermediate and final artifacts.

### Execution Notes

- The notebook contains Kaggle-oriented setup logic but also includes local path detection.
- Ensure required Python packages are installed (e.g., pandas, numpy, scikit-learn, transformers, matplotlib; plus optional xgboost/lightgbm).
- If re-running API collection cells, provide an Alpha Vantage API key via environment variable as expected by the notebook code.
- If optional model libraries are unavailable, the notebook is designed to continue with available models.

## Scope and Interpretation

This repository is intended for academic experimentation and comparative analysis of feature sets and models for **next-day Tesla movement direction classification**. It should be used with appropriate caution regarding market non-stationarity, data-period dependence, and practical trading constraints.
