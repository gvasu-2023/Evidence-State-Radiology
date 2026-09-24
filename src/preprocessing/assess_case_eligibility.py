from pathlib import Path

import pandas as pd


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Input and output paths
# ============================================================

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "metadata"
    / "dataset.csv"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "metadata"
    / "eligibility.csv"
)


# ============================================================
# Configuration
# ============================================================

MIN_CONTEXT_LENGTH = 15


# ============================================================
# Utility functions
# ============================================================

def contains_deidentification_marker(text: str) -> bool:
    """Check whether the clinical context contains XXXX markers."""

    return "XXXX" in text.upper()


# ============================================================
# Case eligibility assessment
# ============================================================

def assess_case(row: pd.Series) -> dict:
    """Determine which experimental conditions the case can support."""

    context = str(row["clinical_context"]).strip()

    context_length = len(context)

    has_context = bool(context)

    has_deid_marker = contains_deidentification_marker(context)

    # Remove de-identification markers before checking whether
    # meaningful clinical components are present.
    usable_context = context.replace("XXXX", "").strip()

    # --------------------------------------------------------
    # E1 — Sufficient
    # --------------------------------------------------------
    #
    # Requires a usable clinical indication of sufficient length.
    #

    eligible_sufficient = (
        has_context
        and context_length >= MIN_CONTEXT_LENGTH
    )

    # --------------------------------------------------------
    # E2 — Incomplete
    # --------------------------------------------------------
    #
    # Requires a context containing potentially separable
    # clinical components.
    #
    # We do NOT create incomplete context from a single
    # short clinical concept.
    #

    eligible_incomplete = (
        has_context
        and len(usable_context) >= MIN_CONTEXT_LENGTH
        and (
            "." in usable_context
            or "," in usable_context
            or " and " in usable_context.lower()
            or " with " in usable_context.lower()
        )
    )

    # --------------------------------------------------------
    # E3 — Irrelevant
    # --------------------------------------------------------
    #
    # Any case with an image can later be paired with
    # a clinical indication from another case.
    #

    eligible_irrelevant = True

    # --------------------------------------------------------
    # E4 — Conflicting
    # --------------------------------------------------------
    #
    # Requires enough original context to construct a
    # controlled contradiction later.
    #

    eligible_conflicting = (
        has_context
        and context_length >= MIN_CONTEXT_LENGTH
    )

    # --------------------------------------------------------
    # E5 — Insufficient
    # --------------------------------------------------------
    #
    # Every case can be evaluated with no clinical context.
    #

    eligible_insufficient = True

    return {
        "sample_id": row["sample_id"],
        "context_length": context_length,
        "contains_deid": has_deid_marker,
        "eligible_sufficient": eligible_sufficient,
        "eligible_incomplete": eligible_incomplete,
        "eligible_irrelevant": eligible_irrelevant,
        "eligible_conflicting": eligible_conflicting,
        "eligible_insufficient": eligible_insufficient,
    }


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found:\n{INPUT_CSV}"
        )

    # Load dataset
    df = pd.read_csv(INPUT_CSV)

    # Required columns
    required_columns = [
        "sample_id",
        "clinical_context",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Assess every case
    eligibility_records = [
        assess_case(row)
        for _, row in df.iterrows()
    ]

    eligibility_df = pd.DataFrame(
        eligibility_records
    )

    # Make sure output directory exists
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save eligibility table
    eligibility_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # Console report
    # ========================================================

    print("=" * 60)
    print("IU-Xray Case Eligibility Assessment")
    print("=" * 60)

    print(
        f"Total cases: {len(eligibility_df)}"
    )

    print()

    print("Eligibility counts:")

    for column in [
        "eligible_sufficient",
        "eligible_incomplete",
        "eligible_irrelevant",
        "eligible_conflicting",
        "eligible_insufficient",
    ]:

        count = int(
            eligibility_df[column].sum()
        )

        print(
            f"{column}: "
            f"{count}/{len(eligibility_df)}"
        )

    print()

    print("Cases containing XXXX in clinical context:")

    print(
        int(
            eligibility_df["contains_deid"].sum()
        ),
        f"/{len(eligibility_df)}",
    )

    print()

    print("Eligibility table:")

    print(
        eligibility_df.to_string(
            index=False
        )
    )

    print()

    print("Saved to:")

    print(OUTPUT_CSV)


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    main()