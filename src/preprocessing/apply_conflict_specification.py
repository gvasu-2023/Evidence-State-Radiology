from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Input files
# ============================================================

PERTURBATION_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "perturbations"
    / "perturbations.csv"
)

CONFLICT_CONFIG = (
    ROOT
    / "configs"
    / "conflict_cases.yaml"
)

OUTPUT_CSV = PERTURBATION_CSV


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not PERTURBATION_CSV.exists():
        raise FileNotFoundError(
            f"Perturbation file not found:\n{PERTURBATION_CSV}"
        )

    if not CONFLICT_CONFIG.exists():
        raise FileNotFoundError(
            f"Conflict configuration not found:\n{CONFLICT_CONFIG}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    perturbations = pd.read_csv(
        PERTURBATION_CSV
    )

    with open(
        CONFLICT_CONFIG,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    conflicts = config["conflicts"]

    # --------------------------------------------------------
    # Process E4 records
    # --------------------------------------------------------

    updated_records = []

    approved_count = 0
    excluded_count = 0

    for _, row in perturbations.iterrows():

        sample_id = row["sample_id"]
        condition = row["condition"]

        # Keep all non-conflicting records unchanged.
        if condition != "conflicting":

            updated_records.append(
                row.to_dict()
            )

            continue

        # Make sure every E4 case has a specification.
        if sample_id not in conflicts:

            raise ValueError(
                f"No conflict specification found for "
                f"{sample_id}"
            )

        specification = conflicts[sample_id]

        approved = bool(
            specification["approved"]
        )

        # ----------------------------------------------------
        # Excluded E4 case
        # ----------------------------------------------------

        if not approved:

            excluded_count += 1

            print(
                f"[EXCLUDED] {sample_id}"
            )

            continue

        # ----------------------------------------------------
        # Approved E4 case
        # ----------------------------------------------------

        conflicting_context = str(
            specification["conflicting_context"]
        ).strip()

        if not conflicting_context:

            raise ValueError(
                f"Empty conflicting context for "
                f"{sample_id}"
            )

        row["perturbed_context"] = (
            conflicting_context
        )

        row["transformation_type"] = (
            specification["type"]
        )

        row["source_context_id"] = (
            sample_id
        )

        row["manual_review_required"] = False

        approved_count += 1

        updated_records.append(
            row.to_dict()
        )

        print(
            f"[APPROVED] {sample_id}: "
            f"{conflicting_context}"
        )

    # --------------------------------------------------------
    # Create updated dataframe
    # --------------------------------------------------------

    updated_df = pd.DataFrame(
        updated_records
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    updated_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("E4 Conflict Specification Applied")
    print("=" * 60)

    print(
        f"Approved E4 records: {approved_count}"
    )

    print(
        f"Excluded E4 records: {excluded_count}"
    )

    print(
        f"Final perturbation records: "
        f"{len(updated_df)}"
    )

    print()
    print("Final condition counts:")

    print(
        updated_df["condition"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Saved to:")

    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()