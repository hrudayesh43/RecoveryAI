"""Anomaly detection utilities for RecoverAI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy with anomaly labels and scores for selected numeric columns.

    label -1 means flagged as anomalous; label 1 means not flagged.
    Scores are relative model scores, not probabilities.
    """
    if not 0.0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5")

    if df.empty:
        raise ValueError("Cannot detect anomalies in an empty dataframe.")

    if columns is None:
        columns = list(df.select_dtypes(include="number").columns)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {missing}")

    usable = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not usable:
        raise ValueError("Choose at least one numeric column.")

    features = df[usable].replace([np.inf, -np.inf], np.nan)
    if features.dropna(how="all").empty:
        raise ValueError("Selected columns contain no usable numeric values.")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,
        ),
    )
    labels = model.fit_predict(features)
    scores = model.decision_function(features)

    result = df.copy(deep=True)
    result["anomaly_label"] = labels
    result["anomaly_score"] = scores
    result["is_anomaly"] = labels == -1
    return result


def evaluate_detection(
    injected_ground_truth: pd.Series | np.ndarray,
    predicted_anomalies: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calculate binary detection metrics against known injected anomaly labels."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    y_true = np.asarray(injected_ground_truth, dtype=bool)
    y_pred = np.asarray(predicted_anomalies, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("Ground-truth labels and predictions must have the same shape.")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate empty labels.")

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_anomalies": int(y_true.sum()),
        "detected_anomalies": int(y_pred.sum()),
    }
