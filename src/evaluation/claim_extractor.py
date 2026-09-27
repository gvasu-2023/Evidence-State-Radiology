"""
Extract generated-report claim candidates.

This module extracts clinical-finding candidates from baseline
generated reports using the same REFERENCE_PATTERNS vocabulary
and polarity rules as the reference candidate extractor.

It writes:

    results/tables/generated_claim_candidates.csv

It does NOT overwrite:

    results/tables/extracted_claims.csv

Conflict-target evaluation remains in claim_evaluator.py.
That Phase 6 evaluator is condition-blind: it matches claims by
(sample_id, finding) and ignores condition. That behavior is
intentionally left unchanged. This extractor is condition-aware
and keys generated claims by (sample_id, condition, finding).
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.claim_patterns import (
    extract_raw_claim_records,
    strip_generated_report_body,
)


GENERATION_PATH = Path(
    "results/tables/baseline_generation_results.csv"
)

OUTPUT_PATH = Path(
    "results/tables/generated_claim_candidates.csv"
)

# Historical Phase 6 artifact. Never overwrite this file.
PHASE6_CLAIMS_PATH = Path(
    "results/tables/extracted_claims.csv"
)


def extract_generated_claim_candidates(
    generation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract case-level generated claims for each
    (sample_id, condition) generation record.
    """

    required_columns = {
        "sample_id",
        "condition",
        "generated_report",
    }

    missing = required_columns - set(
        generation_df.columns
    )

    if missing:
        raise ValueError(
            "Missing generation columns: "
            f"{sorted(missing)}"
        )

    records = []

    for _, row in generation_df.iterrows():

        sample_id = str(row["sample_id"])
        condition = str(row["condition"])

        body = strip_generated_report_body(
            row["generated_report"]
        )

        matches = extract_raw_claim_records(body)

        if not matches:
            continue

        match_df = pd.DataFrame(matches)

        match_df["sample_id"] = sample_id
        match_df["condition"] = condition

        # Collapse Findings / Impression repeats of the same
        # finding and polarity within one generated report.
        match_df = match_df.drop_duplicates(
            subset=[
                "sample_id",
                "condition",
                "finding",
                "polarity",
            ]
        )

        records.append(match_df)

    if not records:

        return pd.DataFrame(
            columns=[
                "sample_id",
                "condition",
                "finding",
                "polarity",
                "matched_pattern",
                "source_sentence",
            ]
        )

    combined = pd.concat(
        records,
        ignore_index=True,
    )

    collapsed_rows = []

    grouped = combined.groupby(
        ["sample_id", "condition", "finding"],
        sort=False,
    )

    for (
        sample_id,
        condition,
        finding,
    ), group in grouped:

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

        sentences = sorted(
            set(
                group["source_sentence"]
                .dropna()
                .astype(str)
                .tolist()
            )
        )

        patterns = sorted(
            set(
                group["matched_pattern"]
                .dropna()
                .astype(str)
                .tolist()
            )
        )

        collapsed_rows.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "finding": finding,
                "polarity": polarity,
                "matched_pattern": " | ".join(patterns),
                "source_sentence": " | ".join(sentences),
            }
        )

    output = pd.DataFrame(collapsed_rows)

    if output.empty:

        return pd.DataFrame(
            columns=[
                "sample_id",
                "condition",
                "finding",
                "polarity",
                "matched_pattern",
                "source_sentence",
            ]
        )

    output = (
        output
        .sort_values(
            [
                "sample_id",
                "condition",
                "finding",
            ]
        )
        .reset_index(drop=True)
    )

    return output


def main() -> None:

    if not GENERATION_PATH.exists():
        raise FileNotFoundError(
            f"Generation results not found: {GENERATION_PATH}"
        )

    if PHASE6_CLAIMS_PATH.exists():
        phase6_mtime_before = (
            PHASE6_CLAIMS_PATH.stat().st_mtime
        )
    else:
        phase6_mtime_before = None

    generation_df = pd.read_csv(GENERATION_PATH)

    output = extract_generated_claim_candidates(
        generation_df
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    if PHASE6_CLAIMS_PATH.exists():
        phase6_mtime_after = (
            PHASE6_CLAIMS_PATH.stat().st_mtime
        )

        if (
            phase6_mtime_before is not None
            and phase6_mtime_after != phase6_mtime_before
        ):
            raise RuntimeError(
                "extracted_claims.csv was modified. "
                "Phase 7 must not overwrite that artifact."
            )

    print("=" * 70)
    print("GENERATED CLAIM CANDIDATE EXTRACTION")
    print("=" * 70)

    print(
        f"Generation records: {len(generation_df)}"
    )

    print(
        f"Generated claim candidates: {len(output)}"
    )

    print(
        "\nPolarity counts:"
    )

    if output.empty:
        print("No generated claims found.")
    else:
        print(
            output["polarity"]
            .value_counts()
            .to_string()
        )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )

    print(
        "\nPhase 6 extracted_claims.csv was not overwritten."
    )

    print(
        "Conflict evaluation remains in claim_evaluator.py "
        "and is condition-blind."
    )


if __name__ == "__main__":
    main()
