
import pandas as pd


def controlled_repair(
    df: pd.DataFrame,
    missing_threshold: float = 0.40,
):
    """
    Repair missing values conservatively and remove only exact
    duplicates present before imputation.

    Rows that become identical after imputation are preserved.
    Returns the repaired dataset, audit log, and review items.
    """
    if not 0 <= missing_threshold <= 1:
        raise ValueError(
            "missing_threshold must be between 0 and 1."
        )

    repaired = df.copy(deep=True)
    audit_log = []
    review_items = []

    original_rows = len(repaired)
    original_columns = list(repaired.columns)

    # Step 1: Remove exact duplicates present in the input.
    initial_duplicates = int(repaired.duplicated().sum())

    if initial_duplicates:
        repaired = repaired.drop_duplicates(
            keep="first"
        ).copy()

    audit_log.append({
        "action": "remove_initial_duplicates",
        "column": "*",
        "affected_rows": initial_duplicates,
        "details": (
            f"Input rows: {original_rows}; "
            f"removed {initial_duplicates} exact duplicates "
            "that existed before imputation."
        ),
    })

    # Step 2: Impute missing values only when missingness is
    # at or below the configured threshold.
    for column in repaired.columns:
        missing_count = int(repaired[column].isna().sum())

        if missing_count == 0:
            continue

        missing_ratio = (
            missing_count / len(repaired)
            if len(repaired) else 0
        )

        if missing_ratio > missing_threshold:
            review_items.append({
                "column": str(column),
                "missing_values": missing_count,
                "missing_percentage": round(
                    missing_ratio * 100, 2
                ),
                "reason": (
                    "Missingness exceeds the configured threshold. "
                    "Manual review is recommended."
                ),
            })
            continue

        if repaired[column].isna().all():
            review_items.append({
                "column": str(column),
                "missing_values": missing_count,
                "missing_percentage": 100.0,
                "reason": "Column contains no observed values.",
            })
            continue

        if pd.api.types.is_numeric_dtype(repaired[column]):
            fill_value = repaired[column].median()
            method = "median_imputation"
        else:
            modes = repaired[column].mode(dropna=True)

            if modes.empty:
                review_items.append({
                    "column": str(column),
                    "missing_values": missing_count,
                    "missing_percentage": round(
                        missing_ratio * 100, 2
                    ),
                    "reason": "No valid replacement value was found.",
                })
                continue

            fill_value = modes.iloc[0]
            method = "mode_imputation"

        repaired[column] = repaired[column].fillna(fill_value)

        audit_log.append({
            "action": method,
            "column": str(column),
            "affected_rows": missing_count,
            "details": (
                f"Filled {missing_count} missing values "
                f"using {fill_value!r}."
            ),
        })

    # Step 3: Validate structure. Do NOT deduplicate again:
    # distinct records can become identical after imputation.
    if list(repaired.columns) != original_columns:
        raise RuntimeError(
            "Repair unexpectedly changed the columns."
        )

    if len(repaired) > original_rows:
        raise RuntimeError(
            "Repair unexpectedly increased the row count."
        )

    remaining_duplicates = int(repaired.duplicated().sum())

    audit_log.append({
        "action": "validation_summary",
        "column": "*",
        "affected_rows": 0,
        "details": (
            f"Input rows: {original_rows}; "
            f"output rows: {len(repaired)}; "
            f"initial duplicates removed: {initial_duplicates}; "
            f"exact duplicate rows in output: {remaining_duplicates}. "
            "Output duplicates may include records that became "
            "identical after imputation and were intentionally preserved."
        ),
    })

    return (
        repaired,
        pd.DataFrame(audit_log),
        pd.DataFrame(review_items),
    )
