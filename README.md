# Tesla Stock Price Movement Prediction Using News Sentiment and Machine Learning

## Project Objective
This final year project investigates whether financial news sentiment can improve next-day Tesla stock movement prediction.

The prediction target is binary direction classification:
- **1** = upward movement
- **0** = downward movement

The project compares technical-only features against technical + sentiment feature sets under time-series evaluation.

---

## Dataset Description
This repository contains sample, raw, and processed data artifacts related to Tesla stock/news modeling.

### Data sources used in the project
- Tesla historical stock market data
- Tesla-related financial news from Kaggle dataset
- Additional Tesla news collected via Alpha Vantage

### Current repository data folders
- `data/sample/`: small sample dataset for quick reference
- `data/raw/`: raw collected news files
- `data/processed/`: processed feature datasets for modeling
- `logs/`: API collection logs

---

## Methodology Summary
The notebook workflow includes:
1. **Stock data collection and preprocessing**
2. **Technical indicator engineering**
3. **Target construction for next-day prediction**
4. **News sentiment processing using FinBERT**
5. **Daily sentiment aggregation and date alignment**
6. **Feature fusion (technical + sentiment)**
7. **Time-series model evaluation**
8. **Ablation analysis** (technical-only vs technical + sentiment)

Models include baseline and non-baseline classifiers (e.g., Logistic Regression, tree-based variants) evaluated with time-aware validation.

---

## Repository Structure
```text
notebooks/
  tsla-data-collection-preprocessing.ipynb   # Main project notebook

data/
  raw/
    alphavantage_news_2023_2024_raw.csv      # Raw API-collected news
  processed/
    tsla_processed_with_indicators.csv        # Processed stock + indicators
  sample/
    tesla_news_2020_2022_sample.csv           # Sample reference dataset

logs/
  alphavantage_fetch_log.csv                  # Monthly fetch log summary
```

---

## How to Run the Notebook
Please refer to [run.md](run.md) for full setup instructions.

Quick start:
1. Install dependencies from `requirements.txt`
2. Open `notebooks/tsla-data-collection-preprocessing.ipynb`
3. Run cells top-to-bottom in order

> Note: The notebook was originally developed in Kaggle and includes Kaggle-style paths.

---

## API Keys and Private Credentials
- API keys, passwords, and tokens are **not** included in this repository.
- Do not hardcode private credentials in notebook cells or scripts.
- Use secure environment variables / Kaggle Secrets for API-based data collection.

---

## 3-Minute Defense Navigation (Supervisor Demo)
Use this quick flow for presentation:

### Minute 0:00 – 0:45 | Research Problem
- Explain why next-day TSLA movement is hard to predict.
- State hypothesis: sentiment features can complement technical indicators.

### Minute 0:45 – 1:45 | Pipeline Overview
- Show notebook sections:
  - data preprocessing
  - sentiment extraction (FinBERT)
  - feature fusion
  - time-series model evaluation

### Minute 1:45 – 2:30 | Core Results
- Highlight technical-only vs technical+sentiment comparison.
- Explain primary metrics (Accuracy / F1) and time-series split rationale.

### Minute 2:30 – 3:00 | Contribution + Limitation
- Contribution: integrated end-to-end Tesla sentiment-ML pipeline.
- Limitation: single stock focus, moderate predictive power, future extension plans.

---

## Notes
This repository is maintained for academic explanation, reproducibility improvement, and supervisor discussion support.
