import pandas as pd
import streamlit as st

from src.ml_impact import evaluate_ml_impact
from src.anomaly_detection import detect_anomalies
from src.controlled_repair import controlled_repair
from src.corruption import corrupt_dataframe
from src.profiling import profile_dataframe
from src.quality_metrics import calculate_quality_metrics
from src.repair import propose_repairs
from src.validation import validate_dataframe
from src.demo_data import make_demo_dataset


st.set_page_config(
    page_title="RecoverAI",
    page_icon="🛠️",
    layout="wide",
)

st.title("RecoverAI")
st.caption(
    "A reproducible prototype for data-quality monitoring, "
    "AI anomaly detection, and controlled repair."
)


# --------------------------------------------------
# 1. Dataset input and controlled corruption
# --------------------------------------------------

with st.sidebar:
    st.header("Dataset configuration")

    uploaded = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
    )

    scenario = st.selectbox(
        "Controlled corruption scenario",
        [
            "None",
            "Missing values",
            "Duplicate rows",
            "Numeric outliers",
            "Feature drift",
        ],
    )

    intensity = st.slider(
        "Corruption intensity (%)",
        min_value=1,
        max_value=30,
        value=10,
    )


if uploaded is not None:
    try:
        original = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    original = make_demo_dataset()

    st.info(
        "Using the built-in 120-row synthetic demo dataset. "
        "Choose a corruption scenario to explore the full workflow."
    )


working = original.copy(deep=True)

if scenario != "None":
    working = corrupt_dataframe(
        working,
        scenario,
        intensity / 100,
        seed=42,
    )


# --------------------------------------------------
# 2. Dataset overview
# --------------------------------------------------

st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(working):,}")
c2.metric("Columns", f"{working.shape[1]:,}")
c3.metric(
    "Missing cells",
    f"{int(working.isna().sum().sum()):,}",
)
c4.metric(
    "Exact duplicate rows",
    f"{int(working.duplicated().sum()):,}",
)


# --------------------------------------------------
# 3. AI anomaly detection
# --------------------------------------------------

st.header("AI Anomaly Detection")

st.write(
    "Use Isolation Forest to identify unusual records. "
    "Flagged records are candidates for review, not confirmed errors."
)

numeric_columns = list(
    working.select_dtypes(include="number").columns
)

if len(working) < 2:
    st.info("At least two rows are required for anomaly detection.")

elif not numeric_columns:
    st.info("No numeric columns are available for anomaly detection.")

else:
    selected_columns = st.multiselect(
        "Select numeric features",
        options=numeric_columns,
        default=numeric_columns[:min(3, len(numeric_columns))],
        key="anomaly_columns",
    )

    contamination = st.slider(
        "Expected proportion of anomalies",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        key="anomaly_contamination",
    )

    if st.button(
        "Run AI Anomaly Detection",
        key="run_anomaly_detection",
    ):
        if not selected_columns:
            st.warning("Select at least one numeric feature.")

        else:
            try:
                anomaly_results = detect_anomalies(
                    working,
                    columns=selected_columns,
                    contamination=contamination,
                )

                flagged = anomaly_results[
                    anomaly_results["is_anomaly"]
                ]

                a, b, c = st.columns(3)

                a.metric(
                    "Rows analysed",
                    len(anomaly_results),
                )
                b.metric(
                    "Rows flagged",
                    len(flagged),
                )
                c.metric(
                    "Flagged proportion",
                    f"{len(flagged) / len(anomaly_results):.1%}",
                )

                st.caption(
                    "Anomaly scores are relative model scores, "
                    "not probabilities. Review flagged records "
                    "before changing the data."
                )

                st.subheader("Flagged Records")
                st.dataframe(
                    flagged,
                    use_container_width=True,
                )

                st.download_button(
                    "Download anomaly results",
                    data=anomaly_results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="recoverai_anomaly_results.csv",
                    mime="text/csv",
                    key="download_anomaly_results",
                )

            except (ValueError, TypeError) as exc:
                st.error(
                    f"Anomaly detection could not run: {exc}"
                )


# --------------------------------------------------
# 4. Data profiling and validation
# --------------------------------------------------

st.header("Data Quality Profile")

tab1, tab2, tab3 = st.tabs([
    "Data Preview",
    "Quality Profile",
    "Repair Proposals",
])

with tab1:
    st.write(
        "Current dataset after the selected corruption scenario, "
        "if any."
    )

    st.dataframe(
        working.head(100),
        use_container_width=True,
    )

with tab2:
    st.subheader("Column Profile")

    st.dataframe(
        profile_dataframe(working),
        use_container_width=True,
    )

    st.subheader("Validation Checks")

    issues = validate_dataframe(working)

    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success(
            "No issues were found by the currently enabled "
            "basic validation checks."
        )

with tab3:
    st.write(
        "These are proposed repairs. Inspect the output "
        "before using it."
    )

    proposed_df, proposed_audit = propose_repairs(working)

    st.dataframe(
        proposed_df.head(100),
        use_container_width=True,
    )

    if proposed_audit.empty:
        st.info("No repair actions were proposed.")
    else:
        st.dataframe(
            proposed_audit,
            use_container_width=True,
        )

    st.download_button(
        "Download proposed repaired CSV",
        data=proposed_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="recoverai_proposed_repairs.csv",
        mime="text/csv",
        key="download_proposed_repairs",
    )


# --------------------------------------------------
# 5. Controlled repair engine
# --------------------------------------------------

st.header("Controlled Data Repair")

st.write(
    "Apply conservative repairs to a copy of the current "
    "dataset. Review the results before using them."
)

missing_threshold = st.slider(
    "Maximum missing-value percentage for automatic imputation",
    min_value=0.10,
    max_value=0.80,
    value=0.40,
    step=0.05,
    key="repair_missing_threshold",
    help=(
        "Columns exceeding this missingness threshold are "
        "sent for human review rather than automatically imputed."
    ),
)

if st.button(
    "Run Controlled Repair",
    key="run_controlled_repair",
):
    try:
        repaired_df, audit_log, review_items = controlled_repair(
            working.copy(deep=True),
            missing_threshold=missing_threshold,
        )

        # Store a snapshot of the input so the comparison
        # remains consistent with the repair just performed.
        st.session_state["recoverai_repair_input"] = (
            working.copy(deep=True)
        )
        st.session_state["recoverai_repaired_df"] = repaired_df
        st.session_state["recoverai_audit_log"] = audit_log
        st.session_state["recoverai_review_items"] = review_items

        st.success("Controlled repair completed successfully!")

    except (ValueError, TypeError, RuntimeError) as exc:
        st.error(f"Repair could not be completed: {exc}")


# --------------------------------------------------
# 6. Repair results and quality scorecard
# --------------------------------------------------

if "recoverai_repaired_df" in st.session_state:

    repaired_df = st.session_state["recoverai_repaired_df"]
    audit_log = st.session_state["recoverai_audit_log"]
    review_items = st.session_state["recoverai_review_items"]
    repair_input = st.session_state["recoverai_repair_input"]

    st.header("Repair Results")

    before_metrics = calculate_quality_metrics(repair_input)
    after_metrics = calculate_quality_metrics(repaired_df)

    st.caption(
        "Metrics compare the exact dataset snapshot used for "
        "the repair with the resulting dataset."
    )

    before_col, after_col = st.columns(2)

    with before_col:
        st.subheader("Before Repair")

        st.metric(
            "Completeness",
            f"{before_metrics['completeness_pct']:.2f}%",
        )
        st.metric(
            "Uniqueness",
            f"{before_metrics['uniqueness_pct']:.2f}%",
        )
        st.metric(
            "Missing Cells",
            before_metrics["missing_cells"],
        )
        st.metric(
            "Duplicate Rows",
            before_metrics["duplicate_rows"],
        )

    with after_col:
        st.subheader("After Repair")

        st.metric(
            "Completeness",
            f"{after_metrics['completeness_pct']:.2f}%",
            delta=round(
                after_metrics["completeness_pct"]
                - before_metrics["completeness_pct"],
                2,
            ),
        )
        st.metric(
            "Uniqueness",
            f"{after_metrics['uniqueness_pct']:.2f}%",
            delta=round(
                after_metrics["uniqueness_pct"]
                - before_metrics["uniqueness_pct"],
                2,
            ),
        )
        st.metric(
            "Missing Cells",
            after_metrics["missing_cells"],
            delta=(
                after_metrics["missing_cells"]
                - before_metrics["missing_cells"]
            ),
        )
        st.metric(
            "Duplicate Rows",
            after_metrics["duplicate_rows"],
            delta=(
                after_metrics["duplicate_rows"]
                - before_metrics["duplicate_rows"]
            ),
        )

    st.subheader("Repaired Dataset Preview")

    st.dataframe(
        repaired_df.head(100),
        use_container_width=True,
    )

    st.download_button(
        "Download repaired dataset",
        data=repaired_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="recoverai_repaired.csv",
        mime="text/csv",
        key="download_repaired_dataset",
    )

    st.subheader("Repair Audit Log")

    if audit_log.empty:
        st.info("No repairs were necessary.")
    else:
        st.dataframe(
            audit_log,
            use_container_width=True,
        )

        st.download_button(
            "Download repair audit log",
            data=audit_log.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="recoverai_audit_log.csv",
            mime="text/csv",
            key="download_repair_audit",
        )

    st.subheader("Issues Requiring Review")

    if review_items.empty:
        st.success(
            "No unresolved missing-value issues were identified "
            "by the current review rules."
        )
    else:
        st.warning(
            "These issues were not automatically repaired. "
            "Review them before using the dataset."
        )

        st.dataframe(
            review_items,
            use_container_width=True,
        )


# --------------------------------------------------
# 7. Machine Learning Impact Evaluation
# --------------------------------------------------

st.header("ML Impact Evaluation")

st.write(
    "Compare an original-data reference with corrupted-data training, "
    "imputation, and RecoverAI controlled repair."
)

st.caption(
    "For each run, all experiments are evaluated on the same held-out "
    "records using target labels from the original dataset. The target is "
    "never repaired. Use row-preserving corruption scenarios for this "
    "matched comparison; duplicate-row injection changes the row count."
)

target_candidates = list(
    original.select_dtypes(include="number").columns
)

if len(original) < 30:
    st.info(
        "Upload a dataset with at least 30 rows to run this experiment."
    )
elif not target_candidates:
    st.warning(
        "No numeric target columns are available. "
        "A regression experiment requires a numeric target."
    )
elif scenario == "Duplicate rows":
    st.warning(
        "For a matched ML comparison, choose a row-preserving "
        "corruption such as Missing values, Numeric outliers, or Feature drift. "
        "Duplicate-row injection changes the number of records and needs a "
        "separate evaluation design."
    )
else:
    selected_target = st.selectbox(
        "Select numeric target column",
        options=target_candidates,
        index=(
            target_candidates.index("spend")
            if "spend" in target_candidates
            else 0
        ),
        key="ml_target_column",
    )

    test_fraction = st.slider(
        "Test-set proportion",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        key="ml_test_fraction",
    )

    n_runs = st.slider(
        "Repeated train/test splits",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key="ml_n_runs",
        help="More runs take longer but show how sensitive results are to the split.",
    )

    if scenario == "None":
        st.info(
            "Select a corruption scenario in the sidebar to compare "
            "corrupted training data with the original-data reference."
        )

    if st.button(
        "Evaluate ML Impact",
        key="run_ml_impact",
    ):
        if scenario == "None":
            st.warning(
                "Choose a corruption scenario in the sidebar before running "
                "the ML impact experiment."
            )
        else:
            try:
                with st.spinner(
                    f"Training models across {n_runs} runs..."
                ):
                    ml_summary, ml_details = evaluate_ml_impact(
                        original_df=original,
                        corrupted_df=working,
                        target_column=selected_target,
                        test_size=test_fraction,
                        random_state=42,
                        n_runs=n_runs,
                    )

                st.session_state["recoverai_ml_summary"] = ml_summary
                st.session_state["recoverai_ml_details"] = ml_details
                st.session_state["recoverai_ml_target"] = selected_target
                st.session_state["recoverai_ml_scenario"] = scenario
                st.session_state["recoverai_ml_intensity"] = intensity
                st.success("ML impact evaluation completed!")

            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                st.error(f"ML evaluation could not run: {exc}")
            except Exception as exc:
                st.error(f"Unexpected ML evaluation error: {exc}")

if "recoverai_ml_summary" in st.session_state:
    ml_summary = st.session_state["recoverai_ml_summary"]
    ml_details = st.session_state["recoverai_ml_details"]

    st.subheader("Model Performance Across Repeated Runs")
    st.caption(
        f"Target: {st.session_state.get('recoverai_ml_target', 'not recorded')}. "
        f"Corruption: {st.session_state.get('recoverai_ml_scenario', 'not recorded')} "
        f"at {st.session_state.get('recoverai_ml_intensity', '—')}%. "
        "Lower MAE is better; higher R² is generally better. "
        "Standard deviations show variability across splits, not confidence intervals."
    )

    st.dataframe(
        ml_summary.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detailed Results by Run")
    st.dataframe(
        ml_details.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download ML summary",
        data=ml_summary.to_csv(index=False).encode("utf-8"),
        file_name="recoverai_ml_impact_summary.csv",
        mime="text/csv",
        key="download_ml_summary",
    )

    st.download_button(
        "Download detailed ML results",
        data=ml_details.to_csv(index=False).encode("utf-8"),
        file_name="recoverai_ml_impact_details.csv",
        mime="text/csv",
        key="download_ml_details",
    )

    # --------------------------------------------------
    # Research visualizations
    # --------------------------------------------------

    st.subheader("Research Visualizations")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("**Mean absolute error (MAE)**")
        mae_chart = ml_summary.set_index("experiment")[["MAE_mean"]]
        st.bar_chart(mae_chart)
        st.caption("Lower MAE indicates smaller average prediction errors.")

    with chart_right:
        st.markdown("**Mean R² score**")
        r2_chart = ml_summary.set_index("experiment")[["R2_mean"]]
        st.bar_chart(r2_chart)
        st.caption(
            "Higher R² is generally better. Negative R² means the model "
            "performs worse than a test-set mean-prediction benchmark."
        )

    if "training_rows_mean" in ml_summary.columns:
        st.markdown("**Average training rows retained**")
        rows_chart = ml_summary.set_index("experiment")[["training_rows_mean"]]
        st.bar_chart(rows_chart)
        st.caption(
            "Fewer rows may indicate complete-case filtering or row removal "
            "during controlled repair."
        )

    # --------------------------------------------------
    # Data-quality before/after visualization
    # --------------------------------------------------

    if (
        "recoverai_repair_input" in st.session_state
        and "recoverai_repaired_df" in st.session_state
    ):
        st.subheader("Data Quality Before and After Repair")
        repair_input = st.session_state["recoverai_repair_input"]
        repaired_df = st.session_state["recoverai_repaired_df"]
        before_metrics = calculate_quality_metrics(repair_input)
        after_metrics = calculate_quality_metrics(repaired_df)

        quality_comparison = pd.DataFrame({
            "Before repair": {
                "Completeness (%)": before_metrics["completeness_pct"],
                "Uniqueness (%)": before_metrics["uniqueness_pct"],
                "Missing cells": before_metrics["missing_cells"],
                "Duplicate rows": before_metrics["duplicate_rows"],
                "Rows": len(repair_input),
            },
            "After repair": {
                "Completeness (%)": after_metrics["completeness_pct"],
                "Uniqueness (%)": after_metrics["uniqueness_pct"],
                "Missing cells": after_metrics["missing_cells"],
                "Duplicate rows": after_metrics["duplicate_rows"],
                "Rows": len(repaired_df),
            },
        })

        st.dataframe(
            quality_comparison.round(2),
            use_container_width=True,
        )

        quality_pct = quality_comparison.loc[
            ["Completeness (%)", "Uniqueness (%)"]
        ]
        st.markdown("**Percentage-based quality metrics**")
        st.bar_chart(quality_pct)

        count_metrics = quality_comparison.loc[
            ["Missing cells", "Duplicate rows", "Rows"]
        ]
        st.markdown("**Record and issue counts**")
        st.bar_chart(count_metrics)

    # --------------------------------------------------
    # Downloadable research report
    # --------------------------------------------------

    st.subheader("Experiment Report")

    target_name = st.session_state.get("recoverai_ml_target", "not recorded")
    scenario_name = st.session_state.get("recoverai_ml_scenario", "not recorded")
    intensity_value = st.session_state.get("recoverai_ml_intensity", "not recorded")

    n_evaluated_runs = (
        int(ml_details["run"].nunique())
        if "run" in ml_details
        else 0
    )

    test_rows_summary = (
        int(ml_details["test_rows"].min())
        if "test_rows" in ml_details and not ml_details.empty
        else "not recorded"
    )

    best_mae_row = ml_summary.loc[ml_summary["MAE_mean"].idxmin()]
    best_r2_row = ml_summary.loc[ml_summary["R2_mean"].idxmax()]

    report_lines = [
        "# RecoverAI ML Impact Experiment Report",
        "",
        "## Configuration",
        f"- Target column: {target_name}",
        f"- Corruption scenario: {scenario_name}",
        f"- Corruption intensity setting: {intensity_value}%",
        f"- Repeated train/test runs: {n_evaluated_runs}",
        f"- Test rows per run (minimum recorded): {test_rows_summary}",
        "",
        "## Summary results",
        ml_summary.round(4).to_csv(index=False),
        "",
        "## Interpretation",
        f"- Lowest mean MAE: {best_mae_row['experiment']} "
        f"(MAE = {best_mae_row['MAE_mean']:.4f}).",
        f"- Highest mean R²: {best_r2_row['experiment']} "
        f"(R² = {best_r2_row['R2_mean']:.4f}).",
        "- Lower MAE is better; higher R² is generally better.",
        "- Standard deviations describe variability across the repeated splits; "
        "they are not confidence intervals.",
        "- The original-data reference may contain pre-existing missing values. "
        "Do not describe it as completely clean unless that has been verified.",
        "- Results show association under this experimental setup; they do not "
        "prove that repair will improve every dataset or predictive task.",
        "",
        "## Data-quality results",
    ]

    if (
        "recoverai_repair_input" in st.session_state
        and "recoverai_repaired_df" in st.session_state
    ):
        report_lines.extend([
            quality_comparison.to_csv(),
            "",
        ])
    else:
        report_lines.extend([
            "No stored repair comparison was available when this report was generated.",
            "",
        ])

    report_lines.extend([
        "## Reproducibility notes",
        "- Keep the original CSV unchanged.",
        "- Record the corruption scenario and intensity for each run.",
        "- Use matched held-out records across compared strategies.",
        "- Fit imputation and repair decisions using training data only.",
        "- Review the detailed per-run CSV alongside these summary metrics.",
        "",
    ])

    report_text = "\n".join(report_lines)

    st.download_button(
        "Download experiment report (Markdown)",
        data=report_text.encode("utf-8"),
        file_name="recoverai_experiment_report.md",
        mime="text/markdown",
        key="download_experiment_report",
    )