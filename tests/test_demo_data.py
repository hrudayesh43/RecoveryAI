import pandas as pd
import pytest

from src.demo_data import make_demo_dataset


def test_demo_dataset_is_ready_for_ml_evaluation():
    df = make_demo_dataset()

    assert len(df) == 120
    assert {"age", "income", "visits", "segment", "spend"}.issubset(
        df.columns
    )
    assert df.notna().all().all()
    assert pd.api.types.is_numeric_dtype(df["spend"])
    assert df["spend"].nunique() > 10


def test_demo_dataset_is_reproducible():
    pd.testing.assert_frame_equal(
        make_demo_dataset(),
        make_demo_dataset(),
    )


def test_demo_dataset_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 30"):
        make_demo_dataset(20)

