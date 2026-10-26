import numpy as np
import pandas as pd


def make_demo_dataset(n_rows: int = 120) -> pd.DataFrame:
    """Create a reproducible synthetic dataset for RecoverAI."""

    if (
        isinstance(n_rows, bool)
        or not isinstance(n_rows, int)
        or n_rows < 30
    ):
        raise ValueError("n_rows must be an integer of at least 30.")

    i = np.arange(n_rows, dtype=int)

    age = 21 + (i * 7) % 43
    income = 28000 + (i * 2371) % 72000
    visits = 1 + (i * 3) % 12

    segment = np.where(
        i % 3 == 0,
        "student",
        np.where(i % 3 == 1, "professional", "other"),
    )

    spend = np.round(
        120
        + income * 0.012
        + visits * 22
        - age * 0.7
        + 35 * np.sin(i / 5),
        2,
    )

    return pd.DataFrame({
        "age": age,
        "income": income,
        "visits": visits,
        "segment": segment,
        "spend": spend,
    })
