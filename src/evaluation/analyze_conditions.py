from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "results/tables/baseline_generation_results.csv"
)

CLAIMS_PATH = Path(
    "results/tables/extracted_claims.csv"
)

OUTPUT_PATH = Path(
    "results/tables/condition_baseline_analysis.csv"
)


EXPECTED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline generation and extracted claim tables."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {INPUT_PATH}"
        )

    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"Extracted claims not found: {CLAIMS_PATH}"
        )

    baseline_df = pd.read_csv(INPUT_PATH)
    claims_df = pd.read_csv(CLAIMS_PATH)

    required_baseline = {
        "sample_id",
        "condition",
        "generated_report",
        "inference_time_seconds",
    }

    required_claims = {
        "sample_id",
        "condition",
        "finding",
        "polarity",
    }

    missing_baseline = (
        required_baseline - set(baseline_df.columns)
    )

    missing_claims = (
        required_claims - set(claims_df.columns)
    )

    if missing_baseline:
        raise ValueError(
            "Missing baseline columns: "
            f"{sorted(missing_baseline)}"
        )

    if missing_claims:
        raise ValueError(
            "Missing claim columns: "
            f"{sorted(missing_claims)}"
        )

    return baseline_df, claims_df


def calculate_duplicate_rate(
    reports: pd.Series,
) -> float:
    """Calculate proportion of reports duplicated within a condition."""

    if len(reports) == 0:
        return 0.0

    normalized = (
        reports.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    duplicated = normalized.duplicated(
        keep=False
    )

    return float(duplicated.mean())


def analyze_condition(
    condition: str,
    baseline_df: pd.DataFrame,
    claims_df: pd.DataFrame,
) -> dict:

    condition_baseline = baseline_df[
        baseline_df["condition"] == condition
    ].copy()

    condition_claims = claims_df[
        claims_df["condition"] == condition
    ].copy()

    case_count = len(condition_baseline)

    reports = (
        condition_baseline["generated_report"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_report_count = int(
        (reports == "").sum()
    )

    unique_report_count = int(
        reports.nunique()
    )

    duplicate_report_rate = calculate_duplicate_rate(
        reports
    )

    report_lengths = reports.str.len()

    mean_report_length = float(
        report_lengths.mean()
    ) if case_count else 0.0

    median_report_length = float(
        report_lengths.median()
    ) if case_count else 0.0

    mean_inference_time = float(
        condition_baseline[
            "inference_time_seconds"
        ].mean()
    ) if case_count else 0.0

    claim_count = len(condition_claims)

    affirmed_claim_count = int(
        (
            condition_claims["polarity"]
            == "AFFIRMED"
        ).sum()
    )

    negated_claim_count = int(
        (
            condition_claims["polarity"]
            == "NEGATED"
        ).sum()
    )

    unclear_claim_count = int(
        (
            condition_claims["polarity"]
            == "MENTIONED_UNCLEAR"
        ).sum()
    )

    return {
        "condition": condition,
        "case_count": case_count,
        "empty_report_count": empty_report_count,
        "unique_report_count": unique_report_count,
        "duplicate_report_rate": duplicate_report_rate,
        "mean_report_length_chars": mean_report_length,
        "median_report_length_chars": median_report_length,
        "mean_inference_time_seconds": mean_inference_time,
        "claim_count": claim_count,
        "affirmed_claim_count": affirmed_claim_count,
        "negated_claim_count": negated_claim_count,
        "unclear_claim_count": unclear_claim_count,
    }


def main() -> None:

    baseline_df, claims_df = load_inputs()

    actual_conditions = set(
        baseline_df["condition"].astype(str)
    )

    unexpected = (
        actual_conditions
        - set(EXPECTED_CONDITIONS)
    )

    if unexpected:
        raise ValueError(
            "Unexpected conditions found: "
            f"{sorted(unexpected)}"
        )

    missing_conditions = (
        set(EXPECTED_CONDITIONS)
        - actual_conditions
    )

    if missing_conditions:
        raise ValueError(
            "Expected conditions missing: "
            f"{sorted(missing_conditions)}"
        )

    results = [
        analyze_condition(
            condition,
            baseline_df,
            claims_df,
        )
        for condition in EXPECTED_CONDITIONS
    ]

    output_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("CONDITION-LEVEL BASELINE ANALYSIS")
    print("=" * 70)

    print(
        f"Baseline records: {len(baseline_df)}"
    )

    print(
        f"Extracted claims: {len(claims_df)}"
    )

    print("\nCondition-level results:")
    print(
        output_df.to_string(
            index=False
        )
    )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()