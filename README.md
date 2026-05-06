# Tesla Stock Price Prediction Using News Sentiment Analysis and Machine Learning

This repository contains my final year project on predicting Tesla stock price movement using financial news sentiment analysis and machine learning.

## Project Objective

The main objective of this project is to study whether news sentiment can help improve the prediction of Tesla stock price movement. The target variable is the next-day stock movement, classified as either upward or downward movement.

## Main Methods

This project uses:

- Tesla historical stock price data
- Tesla financial news data
- FinBERT sentiment analysis
- Technical indicators
- Machine learning classification models
- Time-series validation
- Technical-only and technical + sentiment feature comparison

## Data Sources

The project uses Tesla historical stock price data and Tesla-related financial news data. The news data includes a Kaggle Tesla news dataset and additional news data collected from Alpha Vantage.

Only sample or processed data files are included in this repository. API keys and private credentials are not included.

## Repository Structure

```text
notebooks/
  Main Kaggle notebook for data collection, preprocessing, feature engineering, and model preparation.

data/
  raw/
    Raw collected news data.
  processed/
    Processed dataset with stock indicators and sentiment features.
  sample/
    Small sample dataset for reference.

logs/
  API fetching log files.
