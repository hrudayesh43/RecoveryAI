import pandas as pd

def propose_repairs(df: pd.DataFrame):
    """Conservative repairs for missing values and exact duplicates, with an audit trail."""
    repaired = df.copy(deep=True)
    actions = []
    before_missing = int(repaired.isna().sum().sum())
    for col in repaired.columns:
        missing = int(repaired[col].isna().sum())
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(repaired[col]):
            fill_value = repaired[col].median()
            if pd.isna(fill_value):
                actions.append({"action": "manual_review", "column": str(col), "rows_affected": missing,
                                "details": "Entire column is missing or has no numeric median; not auto-filled."})
                continue
        else:
            mode = repaired[col].mode(dropna=True)
            if mode.empty:
                actions.append({"action": "manual_review", "column": str(col), "rows_affected": missing,
                                "details": "No non-missing category available; not auto-filled."})
                continue
            fill_value = mode.iloc[0]
        repaired[col] = repaired[col].fillna(fill_value)
        actions.append({"action": "impute_missing", "column": str(col), "rows_affected": missing,
                        "details": f"Filled missing values using {'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'}."})
    duplicates = int(repaired.duplicated().sum())
    if duplicates:
        repaired = repaired.drop_duplicates().reset_index(drop=True)
        actions.append({"action": "remove_exact_duplicates", "column": "*", "rows_affected": duplicates,
                        "details": "Removed exact duplicate rows. Review domain requirements before accepting."})
    audit = pd.DataFrame(actions, columns=["action", "column", "rows_affected", "details"])
    return repaired, audit
