from pathlib import Path
import re

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_CSV = Path(
    "results/tables/baseline_generation_results.csv"
)

PERTURBATION_CSV = Path(
    "data/processed/iu_xray/perturbations/perturbations.csv"
)

CONFLICT_CONFIG = Path(
    "configs/conflict_cases.yaml"
)

OUTPUT_AUDIT_CSV = Path(
    "results/tables/baseline_audit.csv"
)

OUTPUT_CONFLICT_CSV = Path(
    "results/tables/baseline_conflict_analysis.csv"
)


# ---------------------------------------------------------------------
# Expected experimental design
# ---------------------------------------------------------------------

EXPECTED_COUNTS = {
    "sufficient": 17,
    "incomplete": 9,
    "irrelevant": 20,
    "conflicting": 15,
    "insufficient": 20,
}


# ---------------------------------------------------------------------
# Controlled conflict targets
#
# These correspond to the approved conflict specifications used to
# construct the experimental condition.
# ---------------------------------------------------------------------

CONFLICT_TARGETS = {
    "IU_1": "pleural effusion",
    "IU_2": "cardiomegaly",
    "IU_5": "pleural effusion",
    "IU_6": "focal airspace consolidation",
    "IU_7": "basilar atelectasis",
    "IU_8": "focal airspace opacity",
    "IU_10": "pleural effusion",
    "IU_11": "pleural effusion",
    "IU_14": "pulmonary edema",
    "IU_17": "pleural effusion",
    "IU_18": "pulmonary edema",
    "IU_19": "pleural effusion",
    "IU_20": "pneumothorax",
    "IU_22": "pneumothorax",
    "IU_23": "pleural effusion",
}


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for duplicate analysis."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    """Return whitespace-based word count."""
    text = str(text).strip()

    if not text:
        return 0

    return len(text.split())


def target_pattern(target: str) -> str:
    """
    Create a simple case-insensitive word-boundary pattern.

    This is deliberately conservative and is used only for the
    controlled conflict-target analysis.
    """
    words = target.lower().split()

    escaped_words = [
        re.escape(word)
        for word in words
    ]

    return r"\b" + r"\s+".join(escaped_words) + r"\b"


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:

    required_columns = {
        "sample_id",
        "condition",
        "clinical_context",
        "reference_findings",
        "reference_impression",
        "generated_report",
        "model_name",
        "inference_time_seconds",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Baseline results CSV is empty."
        )

    duplicate_pairs = df.duplicated(
        subset=["sample_id", "condition"]
    )

    if duplicate_pairs.any():
        duplicates = df.loc[
            duplicate_pairs,
            ["sample_id", "condition"],
        ]

        raise ValueError(
            "Duplicate sample/condition pairs found:\n"
            f"{duplicates.to_string(index=False)}"
        )


# ---------------------------------------------------------------------
# Condition-count audit
# ---------------------------------------------------------------------

def audit_condition_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        df["condition"]
        .value_counts()
        .rename_axis("condition")
        .reset_index(name="record_count")
    )

    expected = pd.DataFrame(
        {
            "condition": list(
                EXPECTED_COUNTS.keys()
            ),
            "expected_count": list(
                EXPECTED_COUNTS.values()
            ),
        }
    )

    result = expected.merge(
        counts,
        on="condition",
        how="left",
    )

    result["record_count"] = (
        result["record_count"]
        .fillna(0)
        .astype(int)
    )

    result["count_matches_expected"] = (
        result["record_count"]
        == result["expected_count"]
    )

    return result


# ---------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------

def audit_generation_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working["report_word_count"] = (
        working["generated_report"]
        .fillna("")
        .apply(word_count)
    )

    result = (
        working.groupby("condition")
        .agg(
            mean_inference_time_seconds=(
                "inference_time_seconds",
                "mean",
            ),
            median_inference_time_seconds=(
                "inference_time_seconds",
                "median",
            ),
            min_inference_time_seconds=(
                "inference_time_seconds",
                "min",
            ),
            max_inference_time_seconds=(
                "inference_time_seconds",
                "max",
            ),
            mean_report_word_count=(
                "report_word_count",
                "mean",
            ),
            median_report_word_count=(
                "report_word_count",
                "median",
            ),
            min_report_word_count=(
                "report_word_count",
                "min",
            ),
            max_report_word_count=(
                "report_word_count",
                "max",
            ),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Empty-report audit
# ---------------------------------------------------------------------

def audit_empty_reports(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working["report_word_count"] = (
        working["generated_report"]
        .fillna("")
        .apply(word_count)
    )

    working["empty_report"] = (
        working["report_word_count"] == 0
    )

    result = (
        working.groupby("condition")
        .agg(
            empty_report_count=(
                "empty_report",
                "sum",
            ),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Duplicate-report audit
# ---------------------------------------------------------------------

def audit_duplicate_reports(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working["normalized_report"] = (
        working["generated_report"]
        .fillna("")
        .apply(normalize_text)
    )

    report_frequency = (
        working["normalized_report"]
        .value_counts()
    )

    working["report_frequency"] = (
        working["normalized_report"]
        .map(report_frequency)
    )

    working["duplicate_report"] = (
        working["report_frequency"] > 1
    )

    result = (
        working.groupby("condition")
        .agg(
            duplicate_report_count=(
                "duplicate_report",
                "sum",
            ),
            unique_report_count=(
                "normalized_report",
                "nunique",
            ),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Detailed duplicate-report analysis
# ---------------------------------------------------------------------

def build_duplicate_details(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working["normalized_report"] = (
        working["generated_report"]
        .fillna("")
        .apply(normalize_text)
    )

    frequencies = (
        working["normalized_report"]
        .value_counts()
        .rename("frequency")
        .reset_index()
        .rename(
            columns={
                "index": "normalized_report"
            }
        )
    )

    duplicates = frequencies[
        frequencies["frequency"] > 1
    ].copy()

    if duplicates.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "frequency",
                "sample_ids",
                "report",
            ]
        )

    rows = []

    for _, duplicate in duplicates.iterrows():

        normalized = duplicate[
            "normalized_report"
        ]

        matches = working[
            working["normalized_report"]
            == normalized
        ]

        for condition, group in matches.groupby(
            "condition"
        ):

            rows.append(
                {
                    "condition": condition,
                    "frequency": len(group),
                    "sample_ids": ",".join(
                        sorted(
                            group["sample_id"]
                            .astype(str)
                            .tolist()
                        )
                    ),
                    "report": group.iloc[0][
                        "generated_report"
                    ],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Conflict polarity detection
# ---------------------------------------------------------------------

def classify_conflict_response(
    report: str,
    target: str,
) -> tuple[str, bool]:
    """
    Classify the response to a controlled conflict target.

    This evaluator is intentionally restricted to the controlled
    conflict experiment. It is not the final clinical claim evaluator.

    Categories:
        AFFIRMED
        NEGATED
        MENTIONED_UNCLEAR
        NOT_MENTIONED
    """

    report = str(report).lower().strip()
    target = str(target).lower().strip()

        # ---------------------------------------------------------
    # Remove the clinical indication from the generated report.
    #
    # The model output may reproduce the supplied clinical
    # context. That reproduced context must not be counted as
    # a generated clinical finding.
    # ---------------------------------------------------------

    if "findings :" in report:
        report = report.split("findings :", 1)[1]

    elif "findings:" in report:
        report = report.split("findings:", 1)[1]
    # ---------------------------------------------------------
    # Normalize generated-report formatting.
    # ---------------------------------------------------------

    report = re.sub(r"\s+", " ", report)

    # ---------------------------------------------------------
    # Target-specific lexical variants.
    #
    # These mappings are ONLY for the controlled perturbation
    # targets used in this experiment.
    # ---------------------------------------------------------

    target_variants = {
        "pleural effusion": [
            "pleural effusion",
            "pleural effusions",
        ],
        "pneumothorax": [
            "pneumothorax",
        ],
        "pulmonary edema": [
            "pulmonary edema",
            "pulmonary oedema",
            "edema",
            "oedema",
        ],
        "cardiomegaly": [
            "cardiomegaly",
            "enlarged heart",
            "enlarged cardiac silhouette",
        ],
        "basilar atelectasis": [
            "basilar atelectasis",
            "basal atelectasis",
            "bibasilar atelectasis",
            "bibasal atelectasis",
            "atelectasis",
        ],
        "focal airspace consolidation": [
            "focal airspace consolidation",
            "airspace consolidation",
            "consolidation",
        ],
        "focal airspace opacity": [
            "focal airspace opacity",
            "airspace opacity",
            "focal opacity",
            "opacity",
        ],
    }

    variants = target_variants.get(
        target,
        [target],
    )

    # ---------------------------------------------------------
    # Find every relevant lexical variant in the report.
    # ---------------------------------------------------------

    matches = []

    for variant in variants:
        for match in re.finditer(
            re.escape(variant),
            report,
            flags=re.IGNORECASE,
        ):
            matches.append(
                (
                    match.start(),
                    match.end(),
                    variant,
                )
            )

    # No target or target variant appears.
    if not matches:
        return "NOT_MENTIONED", False

    # ---------------------------------------------------------
    # Sort occurrences so that the report is evaluated
    # deterministically.
    # ---------------------------------------------------------

    matches.sort(key=lambda item: item[0])

    # ---------------------------------------------------------
    # Evaluate every occurrence.
    #
    # We prioritize explicit NEGATION because the controlled
    # conflict experiment specifically asks whether the generated
    # report resists the injected contradictory context.
    # ---------------------------------------------------------

    for start, end, variant in matches:

        # Keep enough surrounding text to capture constructions such as:
        #
        #   no pleural effusion
        #   there is no pneumothorax
        #   without consolidation
        #   negative for pneumothorax
        #
        window_start = max(0, start - 120)
        window_end = min(len(report), end + 120)

        window = report[window_start:window_end]

        # -----------------------------------------------------
        # Explicit negation.
        # -----------------------------------------------------

        negative_patterns = [
            rf"\bno\s+(?:\w+\s+){{0,4}}{re.escape(variant)}\b",
            rf"\bwithout\s+(?:\w+\s+){{0,4}}{re.escape(variant)}\b",
            rf"\babsence\s+of\s+(?:\w+\s+){{0,4}}{re.escape(variant)}\b",
            rf"\bnegative\s+for\s+(?:\w+\s+){{0,4}}{re.escape(variant)}\b",
            rf"\bno\s+evidence\s+of\s+(?:\w+\s+){{0,4}}{re.escape(variant)}\b",
            rf"\b{re.escape(variant)}\s+is\s+absent\b",
            rf"\b{re.escape(variant)}\s+is\s+not\s+seen\b",
            rf"\b{re.escape(variant)}\s+is\s+not\s+identified\b",
            rf"\b{re.escape(variant)}\s+is\s+not\s+present\b",
        ]

        for pattern in negative_patterns:
            if re.search(
                pattern,
                window,
                flags=re.IGNORECASE,
            ):
                return "NEGATED", True

        # -----------------------------------------------------
        # Coordinated negative constructions.
        #
        # Examples:
        #
        #   no pleural effusion or pneumothorax
        #   without consolidation or edema
        #
        # These are common in radiology-style reports.
        # -----------------------------------------------------

        coordinated_patterns = [
            rf"\bno\b[^.]*\b{re.escape(variant)}\b",
            rf"\bwithout\b[^.]*\b{re.escape(variant)}\b",
        ]

        for pattern in coordinated_patterns:
            if re.search(
                pattern,
                window,
                flags=re.IGNORECASE,
            ):
                return "NEGATED", True

    # ---------------------------------------------------------
    # If no explicit negation was found, evaluate positive
    # constructions.
    # ---------------------------------------------------------

    for start, end, variant in matches:

        window_start = max(0, start - 120)
        window_end = min(len(report), end + 120)

        window = report[window_start:window_end]

        positive_patterns = [
            rf"\b{re.escape(variant)}\s+is\s+present\b",
            rf"\bthere\s+is\s+(?:a\s+|an\s+)?{re.escape(variant)}\b",
            rf"\bthere\s+are\s+(?:\w+\s+)?{re.escape(variant)}\b",
            rf"\bdemonstrates?\s+(?:a\s+|an\s+)?{re.escape(variant)}\b",
            rf"\bshows?\s+(?:a\s+|an\s+)?{re.escape(variant)}\b",
            rf"\bwith\s+(?:a\s+|an\s+)?{re.escape(variant)}\b",
        ]

        for pattern in positive_patterns:
            if re.search(
                pattern,
                window,
                flags=re.IGNORECASE,
            ):
                return "AFFIRMED", True

    # ---------------------------------------------------------
    # The target is mentioned but its polarity is unclear.
    # ---------------------------------------------------------

    return "MENTIONED_UNCLEAR", True


# ---------------------------------------------------------------------
# Controlled conflict analysis
# ---------------------------------------------------------------------

def build_conflict_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    conflicting = df[
        df["condition"] == "conflicting"
    ].copy()

    if conflicting.empty:
        return pd.DataFrame()

    rows = []

    for _, row in conflicting.iterrows():

        sample_id = str(
            row["sample_id"]
        )

        target = CONFLICT_TARGETS.get(
            sample_id
        )

        if target is None:
            raise ValueError(
                f"No conflict target defined for "
                f"{sample_id}"
            )

        response, mentioned = (
            classify_conflict_response(
                row["generated_report"],
                target,
            )
        )

        rows.append(
            {
                "sample_id": sample_id,
                "target": target,
                "clinical_context": row[
                    "clinical_context"
                ],
                "generated_report": row[
                    "generated_report"
                ],
                "target_in_report": mentioned,
                "response_category": response,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "sample_id"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("BASELINE GENERATION AUDIT")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    if not PERTURBATION_CSV.exists():
        raise FileNotFoundError(
            f"Perturbation file not found: "
            f"{PERTURBATION_CSV}"
        )

    if not CONFLICT_CONFIG.exists():
        raise FileNotFoundError(
            f"Conflict configuration not found: "
            f"{CONFLICT_CONFIG}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"Input records: {len(df)}")

    validate_input(df)

    print("[PASS] Required columns exist.")
    print("[PASS] No duplicate sample/condition pairs.")

    # -------------------------------------------------------------
    # Condition counts
    # -------------------------------------------------------------

    condition_audit = audit_condition_counts(
        df
    )

    print("\nCondition counts:")
    print(
        condition_audit.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Generation statistics
    # -------------------------------------------------------------

    statistics = audit_generation_statistics(
        df
    )

    print("\nGeneration statistics:")
    print(
        statistics.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Empty reports
    # -------------------------------------------------------------

    empty_reports = audit_empty_reports(
        df
    )

    print("\nEmpty-report audit:")
    print(
        empty_reports.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Duplicate reports
    # -------------------------------------------------------------

    duplicates = audit_duplicate_reports(
        df
    )

    print("\nDuplicate-report audit:")
    print(
        duplicates.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Detailed duplicate reports
    # -------------------------------------------------------------

    duplicate_details = (
        build_duplicate_details(df)
    )

    print("\nDetailed duplicate reports:")

    if duplicate_details.empty:
        print("No duplicate reports found.")

    else:
        print(
            duplicate_details.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # Controlled conflict analysis
    # -------------------------------------------------------------

    conflict_analysis = (
        build_conflict_analysis(df)
    )

    print("\nControlled conflict analysis:")

    print(
        conflict_analysis[
            [
                "sample_id",
                "target",
                "target_in_report",
                "response_category",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # Conflict response summary
    # -------------------------------------------------------------

    conflict_summary = (
        conflict_analysis[
            "response_category"
        ]
        .value_counts()
        .rename_axis(
            "response_category"
        )
        .reset_index(
            name="count"
        )
    )

    print("\nConflict response summary:")
    print(
        conflict_summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Save condition-level audit
    # -------------------------------------------------------------

    summary = (
        condition_audit
        .merge(
            statistics,
            on="condition",
        )
        .merge(
            empty_reports,
            on="condition",
        )
        .merge(
            duplicates,
            on="condition",
        )
    )

    OUTPUT_AUDIT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_AUDIT_CSV,
        index=False,
    )

    # -------------------------------------------------------------
    # Save conflict analysis
    # -------------------------------------------------------------

    conflict_analysis.to_csv(
        OUTPUT_CONFLICT_CSV,
        index=False,
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETED")
    print("=" * 70)

    print(
        f"Condition audit: {OUTPUT_AUDIT_CSV}"
    )

    print(
        f"Conflict analysis: "
        f"{OUTPUT_CONFLICT_CSV}"
    )


if __name__ == "__main__":
    main()