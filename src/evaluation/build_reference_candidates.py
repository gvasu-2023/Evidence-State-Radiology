from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.claim_patterns import (
    extract_raw_claim_records,
)


DATASET_PATH = Path(
    "data/processed/iu_xray/metadata/dataset.csv"
)

OUTPUT_PATH = Path(
    "results/tables/reference_claim_candidates.csv"
)


def extract_reference_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract candidate reference findings from findings +
    impression text.
    """

    records = []

    for _, row in df.iterrows():

        sample_id = str(
            row["sample_id"]
        )

        findings = str(
            row["findings"]
        ).lower()

        impression = str(
            row["impression"]
        ).lower()

        text = (
            findings
            + " "
            + impression
        )

        for match in extract_raw_claim_records(text):

            records.append(
                {
                    "sample_id": sample_id,
                    "finding": match["finding"],
                    "polarity": match["polarity"],
                    "matched_pattern": match[
                        "matched_pattern"
                    ],
                }
            )

    output = pd.DataFrame(records)

    if output.empty:

        return pd.DataFrame(
            columns=[
                "sample_id",
                "finding",
                "polarity",
                "matched_pattern",
            ]
        )

    output = (
        output
        .drop_duplicates(
            subset=[
                "sample_id",
                "finding",
                "polarity",
            ]
        )
        .sort_values(
            [
                "sample_id",
                "finding",
            ]
        )
        .reset_index(drop=True)
    )

    return output


def validate_known_cases(
    output: pd.DataFrame,
) -> None:
    """
    Validate known reference cases from the actual IU X-Ray
    development dataset.

    These checks prevent an incorrect polarity extractor from
    silently becoming the reference ground truth.
    """

    expected = {
        # IU_1
        ("IU_1", "consolidation"): "NEGATED",
        ("IU_1", "pleural effusion"): "NEGATED",
        ("IU_1", "pneumothorax"): "NEGATED",
        ("IU_1", "pulmonary edema"): "NEGATED",

        # IU_10
        ("IU_10", "airspace opacity"): "NEGATED",
        ("IU_10", "pleural effusion"): "NEGATED",
        ("IU_10", "pneumothorax"): "NEGATED",

        # IU_11
        ("IU_11", "pleural effusion"): "NEGATED",
        ("IU_11", "pneumothorax"): "NEGATED",

        # IU_12
        ("IU_12", "pleural effusion"): "NEGATED",
        ("IU_12", "pneumothorax"): "NEGATED",

        # IU_13
        ("IU_13", "cardiomegaly"): "AFFIRMED",
        ("IU_13", "pleural effusion"): "NEGATED",
        ("IU_13", "pneumothorax"): "NEGATED",

        # IU_14
        ("IU_14", "consolidation"): "NEGATED",
        ("IU_14", "hyperinflation"): "AFFIRMED",
        ("IU_14", "interstitial abnormality"): "AFFIRMED",
        ("IU_14", "pleural effusion"): "NEGATED",
        ("IU_14", "pulmonary edema"): "NEGATED",

        # IU_15
        ("IU_15", "granulomatous disease"): "AFFIRMED",
        ("IU_15", "pleural effusion"): "NEGATED",
        ("IU_15", "pneumothorax"): "NEGATED",

        # IU_17
        ("IU_17", "consolidation"): "NEGATED",
        ("IU_17", "pleural effusion"): "NEGATED",
        ("IU_17", "pneumothorax"): "NEGATED",

        # IU_18
        ("IU_18", "consolidation"): "NEGATED",
        ("IU_18", "pleural effusion"): "NEGATED",
        ("IU_18", "pneumothorax"): "NEGATED",
        ("IU_18", "pulmonary edema"): "NEGATED",

        # IU_19
        ("IU_19", "airspace opacity"): "NEGATED",
        ("IU_19", "degenerative change"): "AFFIRMED",
        ("IU_19", "pleural effusion"): "NEGATED",
        ("IU_19", "pneumothorax"): "NEGATED",

        # IU_20
        ("IU_20", "degenerative change"): "AFFIRMED",
        ("IU_20", "pneumothorax"): "NEGATED",

        # IU_22
        ("IU_22", "airspace opacity"): "NEGATED",
        ("IU_22", "pleural effusion"): "NEGATED",
        ("IU_22", "pneumothorax"): "NEGATED",

        # IU_23
        ("IU_23", "airspace opacity"): "NEGATED",
        ("IU_23", "pleural effusion"): "NEGATED",
        ("IU_23", "pneumothorax"): "NEGATED",

        # IU_4
        ("IU_4", "emphysema"): "AFFIRMED",
        ("IU_4", "interstitial abnormality"): "AFFIRMED",
        ("IU_4", "pleural effusion"): "NEGATED",
        ("IU_4", "pneumothorax"): "NEGATED",

        # IU_5
        ("IU_5", "consolidation"): "NEGATED",
        ("IU_5", "hyperinflation"): "AFFIRMED",
        ("IU_5", "pleural effusion"): "NEGATED",
        ("IU_5", "pleural thickening"): "AFFIRMED",
        ("IU_5", "pneumothorax"): "NEGATED",

        # IU_6
        ("IU_6", "consolidation"): "NEGATED",
        ("IU_6", "degenerative change"): "AFFIRMED",
        ("IU_6", "pleural effusion"): "NEGATED",
        ("IU_6", "pneumothorax"): "NEGATED",

        # IU_7
        ("IU_7", "atelectasis"): "AFFIRMED",
        ("IU_7", "consolidation"): "NEGATED",
        ("IU_7", "pleural effusion"): "NEGATED",

        # IU_8
        ("IU_8", "airspace opacity"): "NEGATED",
        ("IU_8", "pleural effusion"): "NEGATED",
        ("IU_8", "pneumothorax"): "NEGATED",

        # IU_9
        ("IU_9", "consolidation"): "NEGATED",
        ("IU_9", "granulomatous disease"): "AFFIRMED",
        ("IU_9", "pleural effusion"): "NEGATED",
        ("IU_9", "pneumothorax"): "NEGATED",
    }

    failures = []

    for key, expected_polarity in expected.items():

        sample_id, finding = key

        matches = output[
            (output["sample_id"] == sample_id)
            & (output["finding"] == finding)
        ]

        if matches.empty:

            failures.append(
                (
                    sample_id,
                    finding,
                    expected_polarity,
                    "MISSING",
                )
            )

            continue

        actual_polarities = set(
            matches["polarity"]
        )

        if expected_polarity not in actual_polarities:

            failures.append(
                (
                    sample_id,
                    finding,
                    expected_polarity,
                    ", ".join(
                        sorted(actual_polarities)
                    ),
                )
            )

    if failures:

        print("\nREFERENCE VALIDATION FAILED")
        print("=" * 70)

        for (
            sample_id,
            finding,
            expected,
            actual,
        ) in failures:

            print(
                f"{sample_id:8s} | "
                f"{finding:25s} | "
                f"expected={expected:8s} | "
                f"actual={actual}"
            )

        raise RuntimeError(
            f"Reference validation failed for "
            f"{len(failures)} cases."
        )

    print(
        "\nREFERENCE VALIDATION PASSED"
    )

    print(
        f"Validated expectations: {len(expected)}"
    )


def main() -> None:

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    output = extract_reference_candidates(
        df
    )

    # ---------------------------------------------------------
    # Validate before writing the reference CSV.
    # ---------------------------------------------------------

    validate_known_cases(
        output
    )

    # ---------------------------------------------------------
    # Save output.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    print("=" * 70)
    print(
        "REFERENCE CLAIM CANDIDATE EXTRACTION"
    )
    print("=" * 70)

    print(
        f"Cases: "
        f"{df['sample_id'].nunique()}"
    )

    print(
        f"Candidate claims: "
        f"{len(output)}"
    )

    print(
        "\nPolarity counts:"
    )

    if output.empty:

        print(
            "No candidate claims found."
        )

    else:

        print(
            output["polarity"]
            .value_counts()
            .to_string()
        )

    print(
        "\nCandidates:"
    )

    if output.empty:

        print(
            "No candidate claims found."
        )

    else:

        print(
            output.to_string(
                index=False
            )
        )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()