import numpy as np
import pandas as pd
import pytest

from src.ml_impact import evaluate_ml_impact


def make_sample_data(n_rows=80):
    idx = np.arange(n_rows, dtype=float)

    original = pd.DataFrame({
        "feature": idx,
        "secondary_feature": idx**2 / 7 + np.sin(idx),
        "category": np.where(idx % 2 == 0, "even", "odd"),
        "target": 3 * idx + np.sin(idx / 4),
    })

    corrupted = original.copy(deep=True)
    corrupted.loc[corrupted.index[::8], "feature"] = np.nan
    corrupted.loc[corrupted.index[5::11], "category"] = np.nan

    return original, corrupted


EXPECTED_STRATEGIES = {
    "Mean-prediction baseline",
    "Original-data reference",
    "Corrupted complete-case",
    "Imputation only",
    "Controlled repair",
}


def test_evaluator_returns_all_five_strategies():
    original, corrupted = make_sample_data()
    original_before = original.copy(deep=True)
    corrupted_before = corrupted.copy(deep=True)

    summary, details = evaluate_ml_impact(
        original,
        corrupted,
        target_column="target",
        n_runs=2,
        random_state=10,
    )

    assert set(summary["experiment"]) == EXPECTED_STRATEGIES
    assert set(details["experiment"]) == EXPECTED_STRATEGIES
    assert len(details) == 10
    assert details.groupby("run").size().eq(5).all()
    assert details.groupby("run")["test_rows"].nunique().eq(1).all()
    assert (details["test_rows"] == 16).all()
    assert np.isfinite(details["MAE"]).all()
    assert np.isfinite(details["R2"]).all()

    pd.testing.assert_frame_equal(original, original_before)
    pd.testing.assert_frame_equal(corrupted, corrupted_before)


def test_evaluator_rejects_missing_target_column():
    original, corrupted = make_sample_data()

    with pytest.raises(ValueError, match="target column is missing"):
        evaluate_ml_impact(
            original, corrupted, target_column="missing_target", n_runs=1
        )


def test_evaluator_rejects_nonnumeric_target():
    original, corrupted = make_sample_data()
    original["target"] = original["target"].astype(str)
    corrupted["target"] = corrupted["target"].astype(str)

    with pytest.raises(ValueError, match="numeric target"):
        evaluate_ml_impact(
            original, corrupted, target_column="target", n_runs=1
        )


def test_evaluator_rejects_mismatched_row_counts():
    original, corrupted = make_sample_data()
    corrupted = corrupted.iloc[:-1].copy()

    with pytest.raises(ValueError, match="different row counts"):
        evaluate_ml_impact(
            original, corrupted, target_column="target", n_runs=1
        )
