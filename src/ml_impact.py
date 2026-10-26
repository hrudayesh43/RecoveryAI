
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _feature_types(X):
    """Identify numeric and categorical feature columns."""
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = list(X.select_dtypes(exclude="number").columns)
    return numeric, categorical


def _make_model(numeric_columns, categorical_columns, random_state):
    """Build a model with preprocessing learned from training data only."""
    transformers = []

    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_columns,
        ))

    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))

    if not transformers:
        raise ValueError("No usable feature columns are available.")

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def _score_experiment(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    run_number,
    seed,
    records,
):
    """Fit one model and record its test performance."""
    if len(X_train) < 20:
        raise ValueError(
            f"Only {len(X_train)} training rows remain for '{name}'. "
            "At least 20 are required."
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    records.append({
        "run": run_number,
        "seed": seed,
        "experiment": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    })


def evaluate_ml_impact(
    original_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.20,
    random_state: int = 42,
    n_runs: int = 5,
):
    """
    Compare five regression strategies on matched test records.

    1. Mean-prediction baseline.
    2. Original-data reference.
    3. Corrupted complete-case.
    4. Imputation only.
    5. Controlled repair.

    Corruption must preserve row count, row order, and column order.
    Target labels come from the original dataset and are never repaired.
    Repair and preprocessing are performed using training data only.

    Returns:
        summary: Aggregated metrics across runs.
        details: Metrics for every experiment and run.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------
    if not isinstance(original_df, pd.DataFrame):
        raise TypeError("original_df must be a pandas DataFrame.")

    if not isinstance(corrupted_df, pd.DataFrame):
        raise TypeError("corrupted_df must be a pandas DataFrame.")

    if target_column not in original_df.columns:
        raise ValueError(
            "The target column is missing from the original dataset."
        )

    if target_column not in corrupted_df.columns:
        raise ValueError(
            "The target column is missing from the corrupted dataset."
        )

    if not pd.api.types.is_numeric_dtype(original_df[target_column]):
        raise ValueError(
            "Choose a numeric target column for regression."
        )

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    if (
        isinstance(n_runs, bool)
        or not isinstance(n_runs, int)
        or not 1 <= n_runs <= 10
    ):
        raise ValueError("n_runs must be an integer from 1 to 10.")

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
    ):
        raise ValueError("random_state must be an integer.")

    clean = original_df.copy(deep=True).reset_index(drop=True)
    corrupted = corrupted_df.copy(deep=True).reset_index(drop=True)

    if len(clean) != len(corrupted):
        raise ValueError(
            "Original and corrupted datasets have different row counts. "
            "This experiment requires row-preserving corruption."
        )

    if list(clean.columns) != list(corrupted.columns):
        raise ValueError(
            "Original and corrupted datasets must have identical "
            "columns in the same order."
        )

    if len(clean) < 30:
        raise ValueError(
            "At least 30 rows are required for ML impact evaluation."
        )

    # Preserve the original target labels.
    known_target = clean[target_column].notna()
    clean = clean.loc[known_target].copy()
    corrupted = corrupted.loc[known_target].copy()

    y = clean[target_column].copy()
    X_clean = clean.drop(columns=[target_column])
    X_corrupted = corrupted.drop(columns=[target_column])

    # Drop columns that contain no feature information in the original.
    empty_features = [
        column for column in X_clean.columns
        if X_clean[column].isna().all()
    ]

    X_clean = X_clean.drop(columns=empty_features)
    X_corrupted = X_corrupted.drop(columns=empty_features)

    if X_clean.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    from src.controlled_repair import controlled_repair

    records = []

    # ---------------------------------------------------------
    # 2. Repeated matched train/test experiments
    # ---------------------------------------------------------
    for run_idx in range(n_runs):
        run_number = run_idx + 1
        seed = random_state + run_idx

        train_idx, test_idx = train_test_split(
            X_clean.index,
            test_size=test_size,
            random_state=seed,
        )

        X_clean_train = X_clean.loc[train_idx].copy()
        X_corrupt_train = X_corrupted.loc[train_idx].copy()

        # All strategies are evaluated on the same original test features.
        X_test_clean = X_clean.loc[test_idx].copy()

        y_train = y.loc[train_idx].copy()
        y_test = y.loc[test_idx].copy()

        numeric_columns, categorical_columns = _feature_types(
            X_clean_train
        )

        # -----------------------------------------------------
        # 1. Mean-prediction baseline
        # -----------------------------------------------------
        dummy_model = Pipeline([
            ("preprocessing", ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                        ]),
                        numeric_columns,
                    ),
                    (
                        "categorical",
                        Pipeline([
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]),
                        categorical_columns,
                    ),
                ],
                remainder="drop",
            )),
            ("model", DummyRegressor(strategy="mean")),
        ])

        _score_experiment(
            "Mean-prediction baseline",
            dummy_model,
            X_clean_train,
            y_train,
            X_test_clean,
            y_test,
            run_number,
            seed,
            records,
        )

        # -----------------------------------------------------
        # 2. Original-data reference
        # -----------------------------------------------------
        _score_experiment(
            "Original-data reference",
            _make_model(
                numeric_columns,
                categorical_columns,
                seed,
            ),
            X_clean_train,
            y_train,
            X_test_clean,
            y_test,
            run_number,
            seed,
            records,
        )

        # -----------------------------------------------------
        # 3. Corrupted complete-case
        # -----------------------------------------------------
        complete_mask = X_corrupt_train.notna().all(axis=1)

        X_complete = X_corrupt_train.loc[complete_mask].copy()
        y_complete = y_train.loc[X_complete.index].copy()

        _score_experiment(
            "Corrupted complete-case",
            _make_model(
                numeric_columns,
                categorical_columns,
                seed,
            ),
            X_complete,
            y_complete,
            X_test_clean,
            y_test,
            run_number,
            seed,
            records,
        )

        # -----------------------------------------------------
        # 4. Imputation only
        # -----------------------------------------------------
        _score_experiment(
            "Imputation only",
            _make_model(
                numeric_columns,
                categorical_columns,
                seed,
            ),
            X_corrupt_train,
            y_train,
            X_test_clean,
            y_test,
            run_number,
            seed,
            records,
        )

        # -----------------------------------------------------
        # 5. Controlled repair
        # -----------------------------------------------------
        repaired_train, _, _ = controlled_repair(
            X_corrupt_train,
            missing_threshold=0.40,
        )

        if not isinstance(repaired_train, pd.DataFrame):
            raise TypeError(
                "controlled_repair must return a DataFrame first."
            )

        if not repaired_train.index.is_unique:
            raise ValueError(
                "Controlled repair returned duplicate indices."
            )

        unknown_indices = repaired_train.index.difference(
            X_corrupt_train.index
        )

        if len(unknown_indices) > 0:
            raise ValueError(
                "Controlled repair returned unknown training indices."
            )

        repaired_train = repaired_train.reindex(
            columns=X_corrupt_train.columns
        )

        repaired_targets = y_train.loc[repaired_train.index]

        _score_experiment(
            "Controlled repair",
            _make_model(
                numeric_columns,
                categorical_columns,
                seed,
            ),
            repaired_train,
            repaired_targets,
            X_test_clean,
            y_test,
            run_number,
            seed,
            records,
        )

    # ---------------------------------------------------------
    # 3. Summary across runs
    # ---------------------------------------------------------
    details = pd.DataFrame(records)

    summary = (
        details.groupby("experiment", sort=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            training_rows_mean=("training_rows", "mean"),
            training_rows_std=("training_rows", "std"),
        )
        .reset_index()
    )

    summary[
        ["MAE_std", "R2_std", "training_rows_std"]
    ] = summary[
        ["MAE_std", "R2_std", "training_rows_std"]
    ].fillna(0)

    return summary, details
