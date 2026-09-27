"""
Descriptive Evidence-State Analysis for Baseline CXR Report Generation.

Calculates report-level and nested finding-level descriptive metrics
across the five evidence states:
    - sufficient
    - incomplete
    - irrelevant
    - conflicting
    - insufficient

Outputs:
    results/tables/evidence_state_breakdown_analysis.csv

This module is purely descriptive. It does not perform inferential
hypothesis tests (such as Fisher's exact test or Mann-Whitney U) because
finding-level comparison rows are nested within cases and cannot be assumed
to be independent.

Prohibited terms: hallucination rate, reliability score, factuality F1,
overall accuracy, RadGraph-F1, CheXpert-F1, ECE, Brier.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDITION_BASELINE_PATH = ROOT / "results/tables/condition_baseline_analysis.csv"
FACTUALITY_SUMMARY_PATH = (
    ROOT / "results/tables/reference_grounded_factuality_by_condition.csv"
)
FACTUALITY_DETAIL_PATH = (
    ROOT / "results/tables/reference_grounded_factuality.csv"
)
OUTPUT_PATH = ROOT / "results/tables/evidence_state_breakdown_analysis.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_input_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load baseline analysis and factuality summary/detail tables."""

    for path in [
        CONDITION_BASELINE_PATH,
        FACTUALITY_SUMMARY_PATH,
        FACTUALITY_DETAIL_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required input table missing: {path}")

    condition_df = pd.read_csv(CONDITION_BASELINE_PATH)
    factuality_summary_df = pd.read_csv(FACTUALITY_SUMMARY_PATH)
    factuality_detail_df = pd.read_csv(FACTUALITY_DETAIL_PATH)

    return condition_df, factuality_summary_df, factuality_detail_df


def compute_evidence_state_breakdown(
    condition_df: pd.DataFrame,
    factuality_summary_df: pd.DataFrame,
    factuality_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the descriptive breakdown table across evidence states.

    Combines report-level metrics (81 generation records) and
    finding-level nested factuality proportions (365 comparison rows).
    """

    # Index input tables by condition
    cond_map = condition_df.set_index("condition").to_dict(orient="index")
    fact_map = factuality_summary_df.set_index("condition").to_dict(
        orient="index"
    )

    rows = []

    for condition in REQUIRED_CONDITIONS:
        if condition not in cond_map:
            raise ValueError(
                f"Condition '{condition}' missing from baseline analysis table."
            )
        if condition not in fact_map:
            raise ValueError(
                f"Condition '{condition}' missing from factuality summary table."
            )

        c_info = cond_map[condition]
        f_info = fact_map[condition]

        case_count = int(c_info["case_count"])
        duplicate_report_rate = float(c_info["duplicate_report_rate"])
        unique_report_count = int(c_info["unique_report_count"])
        mean_report_length_chars = float(c_info["mean_report_length_chars"])
        median_report_length_chars = float(c_info["median_report_length_chars"])
        claim_count = int(c_info["claim_count"])

        claim_density = (
            float(claim_count / case_count) if case_count > 0 else 0.0
        )

        comparison_row_count = int(f_info["comparison_row_count"])

        # Label counts from factuality summary
        supported_count = int(f_info.get("supported", 0))
        omission_count = int(f_info.get("omission", 0))
        extra_negation_count = int(f_info.get("extra_negation", 0))
        ungrounded_aff_count = int(f_info.get("ungrounded_affirmation", 0))
        unsupported_aff_count = int(f_info.get("unsupported_affirmation", 0))
        contradiction_count = int(f_info.get("contradiction", 0))
        unclear_count = int(f_info.get("unclear", 0))

        # Factuality label proportions (denominator = nested comparison rows)
        if comparison_row_count > 0:
            supported_prop = supported_count / comparison_row_count
            omission_prop = omission_count / comparison_row_count
            extra_negation_prop = extra_negation_count / comparison_row_count
            ungrounded_aff_prop = ungrounded_aff_count / comparison_row_count
            unsupported_aff_prop = unsupported_aff_count / comparison_row_count
            contradiction_prop = contradiction_count / comparison_row_count
            unclear_prop = unclear_count / comparison_row_count
        else:
            supported_prop = 0.0
            omission_prop = 0.0
            extra_negation_prop = 0.0
            ungrounded_aff_prop = 0.0
            unsupported_aff_prop = 0.0
            contradiction_prop = 0.0
            unclear_prop = 0.0

        rows.append(
            {
                "condition": condition,
                "case_count": case_count,
                "duplicate_report_rate": duplicate_report_rate,
                "unique_report_count": unique_report_count,
                "mean_report_length_chars": mean_report_length_chars,
                "median_report_length_chars": median_report_length_chars,
                "claim_count": claim_count,
                "claim_density": claim_density,
                "supported_proportion": supported_prop,
                "omission_proportion": omission_prop,
                "extra_negation_proportion": extra_negation_prop,
                "ungrounded_affirmation_proportion": ungrounded_aff_prop,
                "unsupported_affirmation_proportion": unsupported_aff_prop,
                "contradiction_proportion": contradiction_prop,
                "unclear_proportion": unclear_prop,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    condition_df, factuality_summary_df, factuality_detail_df = (
        load_input_tables()
    )

    breakdown_df = compute_evidence_state_breakdown(
        condition_df,
        factuality_summary_df,
        factuality_detail_df,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    breakdown_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("PHASE 8: EVIDENCE-STATE DESCRIPTIVE BREAKDOWN ANALYSIS")
    print("=" * 80)
    print(
        "Disclaimer: These metrics describe baseline generation outputs "
        "and polarity agreement across 81 development cases.\n"
        "Comparison rows (N=365) are nested within cases. No inferential "
        "hypothesis testing or claims of statistical significance are made."
    )
    print("=" * 80)
    print(breakdown_df.to_string(index=False))
    print("\nOutput written to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
