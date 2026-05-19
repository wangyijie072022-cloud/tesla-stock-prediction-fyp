#!/usr/bin/env python3
"""Regenerate model comparison evidence for the D6 report.

This report-only script uses the current project feature matrix and the same
strict chronological final holdout split as scripts/train_and_save_model.py.
It writes comparison metrics under report_assets/ only and does not modify the
project's saved model, results, data, notebook, or production scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.train_and_save_model import build_final_holdout_split, load_training_frame


OUTPUT_PATH = PROJECT_ROOT / "report_assets/model_comparison_metrics_regenerated.csv"
SUMMARY_PATH = PROJECT_ROOT / "report_assets/model_comparison_metadata.json"
RANDOM_STATE = 42


def candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        penalty="l1",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE),
    }


def probability_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min())
    return model.predict(X)


def main() -> int:
    df, X, y, feature_columns = load_training_frame()
    train_idx, holdout_idx = build_final_holdout_split(df)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[holdout_idx]
    y_test = y.iloc[holdout_idx]

    rows: list[dict[str, object]] = []
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = probability_scores(model, X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": name,
                "train_rows": int(len(train_idx)),
                "holdout_rows": int(len(holdout_idx)),
                "feature_count": int(len(feature_columns)),
                "holdout_start": df["Date"].iloc[holdout_idx].min().date().isoformat(),
                "holdout_end": df["Date"].iloc[holdout_idx].max().date().isoformat(),
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(["accuracy", "f1_score", "balanced_accuracy"], ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "source": "regenerated_from_current_project_scripts",
                "notebook_original_comparison_output_found": False,
                "reason": "Current notebook and results folder preserve only final Logistic Regression plus baselines.",
                "data_path": "data/processed/tsla_fused_dataset.csv",
                "same_feature_matrix_as_training_script": True,
                "feature_count": len(feature_columns),
                "train_rows": int(len(train_idx)),
                "holdout_rows": int(len(holdout_idx)),
                "holdout_start": comparison["holdout_start"].iloc[0],
                "holdout_end": comparison["holdout_end"].iloc[0],
                "evaluated_models": comparison["model"].tolist(),
                "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
