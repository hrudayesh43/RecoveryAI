
import pandas as pd


def calculate_quality_metrics(df: pd.DataFrame) -> dict:
    """Calculate basic, explainable dataset quality metrics."""

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        completeness = 100.0
    else:
        completeness = (
            1 - df.isna().sum().sum() / total_cells
        ) * 100

    if len(df) == 0:
        uniqueness = 100.0
        duplicate_rows = 0
    else:
        duplicate_rows = int(df.duplicated().sum())
        uniqueness = (
            1 - duplicate_rows / len(df)
        ) * 100

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "completeness_pct": round(completeness, 2),
        "duplicate_rows": duplicate_rows,
        "uniqueness_pct": round(uniqueness, 2),
    }
