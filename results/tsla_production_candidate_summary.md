# TSLA Production Candidate Summary

Status: experimental / research only. This is not a reliable prediction system and is not financial advice.

Feature set: `TSLA_Experimental_Current61_Plus_Conservative_TSLA_State`
Model type: `StandardScaler + LogisticRegression (TSLA Experimental Research Model)`
Training date range: `2020-01-31` to `2024-12-30`
Feature count: `67`

## Included Experimental Features

- `tsla_residual_return_vs_qqq`: TSLA return_1 - rolling 60-day beta(TSLA, QQQ) * QQQ return_1; beta is not saved as a model feature.
- `gap_open_prev_close`: TSLA_Open / previous TSLA_Close - 1.
- `volume_shock_20`: 1 when abs(TSLA volume z-score versus trailing 20 trading days) >= 2, else 0.
- `rolling_volatility_20`: Trailing 20-trading-day standard deviation of TSLA close-to-close return.
- `distance_to_20d_high`: TSLA_Close / trailing 20-trading-day high - 1.
- `distance_to_20d_low`: TSLA_Close / trailing 20-trading-day low - 1.

## Explicitly Excluded Features

- `tsla_rolling_beta_to_qqq`: Diagnostic results showed mixed behavior; beta is used internally to compute residual return but excluded as a model feature.
- `market_state`: Market-state group was unstable and often dragged 5-day target diagnostics.
- `all_new_features`: All-feature bundle increased noise and was not stable across walk-forward windows.
- `event_window_pre_features`: Reliable earnings/delivery/FOMC calendars are not present locally; pre-window features would risk future event-date leakage.
- `intraday_reversal`: Single-feature diagnostics showed calibration/accuracy drag; excluded by default.
- `previous_large_up`: Not stable enough to include by default.
- `previous_large_down`: Single-feature diagnostics showed worse PR AUC/Brier behavior; excluded by default.

## Metrics

| scope | name | kind | threshold | accuracy | balanced_accuracy | precision | recall | f1_score | roc_auc | pr_auc | brier_score | predicted_up_ratio | actual_up_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_oof | model_default_threshold_0p5 | model | 0.5 | 0.4977 | 0.4972 | 0.5175 | 0.5116 | 0.5145 | 0.5074 | 0.5489 | 0.2505 | 0.5143 | 0.5203 |
| validation_oof | model_research_selected_threshold | model | 0.3 | 0.5203 | 0.5 | 0.5203 | 1.0 | 0.6845 | 0.5074 | 0.5489 | 0.2505 | 1.0 | 0.5203 |
| strict_final_20pct_holdout | model_default_threshold_0p5 | model | 0.5 | 0.54 | 0.5403 | 0.5333 | 0.5657 | 0.549 | 0.5318 | 0.5498 | 0.2506 | 0.525 | 0.495 |
| strict_final_20pct_holdout | model_research_selected_threshold | model | 0.3 | 0.49 | 0.4949 | 0.4925 | 0.9899 | 0.6577 | 0.5318 | 0.5498 | 0.2506 | 0.995 | 0.495 |
| strict_final_20pct_holdout | always_up | baseline |  | 0.495 | 0.5 | 0.495 | 1.0 | 0.6622 | 0.5 | 0.495 | 0.505 | 1.0 | 0.495 |
| strict_final_20pct_holdout | always_down | baseline |  | 0.505 | 0.5 | 0.0 | 0.0 | 0.0 | 0.5 | 0.495 | 0.495 | 0.0 | 0.495 |
| strict_final_20pct_holdout | tsla_momentum_return_1 | baseline |  | 0.51 | 0.5102 | 0.5049 | 0.5253 | 0.5149 | 0.5102 | 0.5002 | 0.49 | 0.515 | 0.495 |
| strict_final_20pct_holdout | tsla_momentum_return_5 | baseline |  | 0.495 | 0.4951 | 0.4902 | 0.5051 | 0.4975 | 0.4951 | 0.4926 | 0.505 | 0.51 | 0.495 |
| strict_final_20pct_holdout | tsla_relative_to_qqq_momentum | baseline |  | 0.525 | 0.525 | 0.52 | 0.5253 | 0.5226 | 0.525 | 0.5081 | 0.475 | 0.5 | 0.495 |

## Holdout Baseline Check

- Holdout model accuracy at threshold 0.5: `0.5400`.
- Best holdout baseline: `tsla_relative_to_qqq_momentum` accuracy `0.5250`.
- Delta accuracy vs best baseline: `+0.0150`.

Interpretation: any improvement is limited and should not be presented as a reliable trading edge.
