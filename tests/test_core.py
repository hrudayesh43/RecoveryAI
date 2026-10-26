import pandas as pd
from src.profiling import profile_dataframe
from src.corruption import corrupt_dataframe
from src.repair import propose_repairs
from src.validation import validate_dataframe

def test_profile_counts_missing_values():
    df = pd.DataFrame({"x": [1, None, 3], "category": ["a", "a", "b"]})
    profile = profile_dataframe(df)
    row = profile.loc[profile["column"] == "x"].iloc[0]
    assert row["missing_count"] == 1

def test_corruption_does_not_mutate_input():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    before = df.copy(deep=True)
    _ = corrupt_dataframe(df, "Missing values", 0.5, seed=7)
    pd.testing.assert_frame_equal(df, before)

def test_repair_fills_missing_and_logs_action():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    repaired, audit = propose_repairs(df)
    assert repaired["x"].isna().sum() == 0
    assert "impute_missing" in audit["action"].tolist()

def test_validation_reports_duplicates():
    df = pd.DataFrame({"x": [1, 1]})
    assert any("duplicate" in issue.lower() for issue in validate_dataframe(df))
