import pandas as pd

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, explainable column-level quality profile."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": str(col),
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2) if len(s) else 0.0,
            "unique_count": int(s.nunique(dropna=True)),
            "duplicate_rows": int(df.duplicated().sum()),
        })
    return pd.DataFrame(rows)
