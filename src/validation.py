import pandas as pd

def validate_dataframe(df: pd.DataFrame) -> list[str]:
    issues = []
    if df.empty:
        issues.append("Dataset is empty.")
    if df.columns.duplicated().any():
        issues.append("Duplicate column names detected.")
    missing = int(df.isna().sum().sum())
    if missing:
        issues.append(f"{missing} missing cell(s) detected.")
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(f"{duplicates} exact duplicate row(s) detected.")
    for col in df.select_dtypes(include="number").columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_count = int((~df[col].dropna().map(lambda x: __import__('math').isfinite(float(x)))).sum())
            if inf_count:
                issues.append(f"Column '{col}' contains {inf_count} infinite value(s).")
    return issues
