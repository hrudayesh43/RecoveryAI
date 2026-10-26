
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import detect_anomalies, evaluate_detection


def test_detector_returns_expected_columns():
    df = pd.DataFrame({
        "x": [1, 1, 1, 1, 100],
        "y": [2, 2, 2, 2, 100],
    })

    result = detect_anomalies(
        df, ["x", "y"], contamination=0.2
    )

    assert {
        "anomaly_label",
        "anomaly_score",
        "is_anomaly",
    }.issubset(result.columns)

    assert result["is_anomaly"].dtype == bool


def test_detector_does_not_modify_input():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})
    original = df.copy(deep=True)

    detect_anomalies(df, ["x"], contamination=0.2)

    pd.testing.assert_frame_equal(df, original)


def test_detector_rejects_no_numeric_columns():
    df = pd.DataFrame({"category": ["a", "b", "c"]})

    with pytest.raises(ValueError):
        detect_anomalies(df)


def test_evaluation_metrics_with_known_labels():
    result = evaluate_detection(
        np.array([False, False, True, True]),
        np.array([False, True, True, True]),
    )

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)
