"""
Reference-grounded factuality comparison.

Compares generated-report claim candidates against reference
claim candidates at the case-level key:

    (sample_id, condition, finding)

This evaluator is condition-aware.

The Phase 6 conflict evaluator in claim_evaluator.py remains
condition-blind (sample_id + finding only). That limitation is
not modified here.

Labels (do not rename):

    supported
    contradiction
    omission
    unsupported_affirmation
    ungrounded_affirmation
    extra_negation
    unclear

These counts are not hallucination rate, factuality F1,
reliability score, overall accuracy, RadGraph-F1, CheXpert-F1,
ECE, Brier, coverage, or abstention.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    VALID_LABELS,
    assign_factuality_label,
)


GENERATION_PATH = Path(
    "results/tables/baseline_generation_results.csv"
)

REFERENCE_PATH = Path(
    "results/tables/reference_claim_candidates.csv"
)

GENERATED_CLAIMS_PATH = Path(
    "results/tables/generated_claim_candidates.csv"
)

DETAIL_OUTPUT_PATH = Path(
    "results/tables/reference_grounded_factuality.csv"
)

SUMMARY_OUTPUT_PATH = Path(
    "results/tables/reference_grounded_factuality_by_condition.csv"
)

EXPECTED_GENERATION_RECORDS = 81

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def collapse_reference_polarities(
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse reference rows to one polarity per
    (sample_id, finding).

    Conflicting polarities become MENTIONED_UNCLEAR.
    """

    required = {"sample_id", "finding", "polarity"}
    missing = required - set(reference_df.columns)

    if missing:
        raise ValueError(
            "Missing reference columns: "
            f"{sorted(missing)}"
        )

    rows = []

    grouped = reference_df.groupby(
        ["sample_id", "finding"],
        sort=False,
    )

    for (sample_id, finding), group in grouped:

        polarities = sorted(
            set(
                str(polarity)
                for polarity in group["polarity"].tolist()
            )
        )

        if (
            "AFFIRMED" in polarities
            and "NEGATED" in polarities
        ):
            polarity = "MENTIONED_UNCLEAR"
        elif len(polarities) == 1:
            polarity = polarities[0]
        else:
            polarity = "MENTIONED_UNCLEAR"

        rows.append(
            {
                "sample_id": str(sample_id),
                "finding": str(finding),
                "reference_polarity": polarity,
            }
        )

    return pd.DataFrame(rows)


def build_factuality_table(
    generation_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    generated_claims_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one comparison row per
    (sample_id, condition, finding) in the union of
    reference findings for the sample and generated findings
    for that condition.
    """

    required_generation = {
        "sample_id",
        "condition",
    }

    missing_generation = (
        required_generation - set(generation_df.columns)
    )

    if missing_generation:
        raise ValueError(
            "Missing generation columns: "
            f"{sorted(missing_generation)}"
        )

    reference = collapse_reference_polarities(
        reference_df
    )

    generated = generated_claims_df.copy()

    if generated.empty:
        generated = pd.DataFrame(
            columns=[
                "sample_id",
                "condition",
                "finding",
                "polarity",
            ]
        )
    else:
        required_generated = {
            "sample_id",
            "condition",
            "finding",
            "polarity",
        }

        missing_generated = (
            required_generated - set(generated.columns)
        )

        if missing_generated:
            raise ValueError(
                "Missing generated-claim columns: "
                f"{sorted(missing_generated)}"
            )

        generated["sample_id"] = (
            generated["sample_id"].astype(str)
        )
        generated["condition"] = (
            generated["condition"].astype(str)
        )
        generated["finding"] = (
            generated["finding"].astype(str)
        )

    generated_lookup = {
        (
            str(row["sample_id"]),
            str(row["condition"]),
            str(row["finding"]),
        ): str(row["polarity"])
        for _, row in generated.iterrows()
    }

    reference_by_sample: dict[str, dict[str, str]] = {}

    for _, row in reference.iterrows():
        sample_id = str(row["sample_id"])
        finding = str(row["finding"])
        reference_by_sample.setdefault(sample_id, {})
        reference_by_sample[sample_id][finding] = str(
            row["reference_polarity"]
        )

    rows = []

    generation_pairs = (
        generation_df[
            ["sample_id", "condition"]
        ]
        .astype(str)
        .drop_duplicates()
        .sort_values(["sample_id", "condition"])
    )

    for _, pair in generation_pairs.iterrows():

        sample_id = str(pair["sample_id"])
        condition = str(pair["condition"])

        reference_findings = reference_by_sample.get(
            sample_id,
            {},
        )

        generated_findings = {
            finding: polarity
            for (
                gen_sample,
                gen_condition,
                finding,
            ), polarity in generated_lookup.items()
            if gen_sample == sample_id
            and gen_condition == condition
        }

        findings = sorted(
            set(reference_findings) | set(generated_findings)
        )

        if not findings:
            rows.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "finding": "",
                    "generated_polarity": "NOT_MENTIONED",
                    "reference_polarity": "",
                    "label": "unclear",
                }
            )
            continue

        for finding in findings:

            generated_polarity = generated_findings.get(
                finding,
                "NOT_MENTIONED",
            )

            reference_polarity = reference_findings.get(
                finding
            )

            label = assign_factuality_label(
                generated_polarity,
                reference_polarity,
            )

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Unexpected label: {label}"
                )

            rows.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "finding": finding,
                    "generated_polarity": generated_polarity,
                    "reference_polarity": (
                        ""
                        if reference_polarity is None
                        else reference_polarity
                    ),
                    "label": label,
                }
            )

    output = pd.DataFrame(rows)

    represented_pairs = set(
        zip(
            output["sample_id"].astype(str),
            output["condition"].astype(str),
        )
    )

    expected_pairs = set(
        zip(
            generation_df["sample_id"].astype(str),
            generation_df["condition"].astype(str),
        )
    )

    missing_pairs = expected_pairs - represented_pairs

    if missing_pairs:
        raise RuntimeError(
            "Generation records missing from factuality "
            f"table: {sorted(missing_pairs)}"
        )

    return (
        output
        .sort_values(
            ["sample_id", "condition", "finding"]
        )
        .reset_index(drop=True)
    )


def summarize_by_condition(
    detail_df: pd.DataFrame,
    generation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count labels by evidence-state condition."""

    generation_counts = (
        generation_df["condition"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    rows = []

    for condition in REQUIRED_CONDITIONS:

        subset = detail_df[
            detail_df["condition"].astype(str)
            == condition
        ]

        represented_pairs = subset[
            ["sample_id", "condition"]
        ].drop_duplicates()

        row = {
            "condition": condition,
            "generation_record_count": int(
                generation_counts.get(condition, 0)
            ),
            "represented_generation_records": len(
                represented_pairs
            ),
            "comparison_row_count": len(subset),
        }

        for label in sorted(VALID_LABELS):
            row[label] = int(
                (subset["label"] == label).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:

    for path in [
        GENERATION_PATH,
        REFERENCE_PATH,
        GENERATED_CLAIMS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    generation_df = pd.read_csv(GENERATION_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)
    generated_claims_df = pd.read_csv(
        GENERATED_CLAIMS_PATH
    )

    # Updated in Phase 7B following conservative reference-vocabulary expansion (73 candidate rows).
    if len(reference_df) != 73:
        raise RuntimeError(
            "Reference candidate row count changed: "
            f"expected 73, found {len(reference_df)}."
        )


    generation_pairs = (
        generation_df[["sample_id", "condition"]]
        .astype(str)
        .drop_duplicates()
    )

    if len(generation_pairs) != EXPECTED_GENERATION_RECORDS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_GENERATION_RECORDS} generation "
            "records, found "
            f"{len(generation_pairs)}."
        )

    detail_df = build_factuality_table(
        generation_df,
        reference_df,
        generated_claims_df,
    )

    summary_df = summarize_by_condition(
        detail_df,
        generation_df,
    )

    DETAIL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_df.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("REFERENCE-GROUNDED FACTUALITY COMPARISON")
    print("=" * 70)

    print(
        "These counts are polarity comparisons against "
        "reference claim candidates."
    )
    print(
        "They are not hallucination rate, factuality F1, "
        "reliability score, or overall accuracy."
    )

    print(
        f"\nGeneration records: {len(generation_df)}"
    )
    print(
        f"Reference candidate rows: {len(reference_df)}"
    )
    print(
        f"Generated claim rows: {len(generated_claims_df)}"
    )
    print(
        f"Comparison rows: {len(detail_df)}"
    )

    represented = detail_df[
        ["sample_id", "condition"]
    ].drop_duplicates()

    print(
        "Represented generation records: "
        f"{len(represented)}"
    )

    print("\nLabel counts:")
    print(
        detail_df["label"]
        .value_counts()
        .reindex(sorted(VALID_LABELS), fill_value=0)
        .to_string()
    )

    print("\nBy condition:")
    print(summary_df.to_string(index=False))

    print(
        f"\nDetail output: {DETAIL_OUTPUT_PATH}"
    )
    print(
        f"Summary output: {SUMMARY_OUTPUT_PATH}"
    )

    print(
        "\nPhase 6 limitation (unchanged): "
        "claim_evaluator.py matches by (sample_id, finding) "
        "and ignores condition."
    )


if __name__ == "__main__":
    main()
