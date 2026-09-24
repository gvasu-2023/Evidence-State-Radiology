from pathlib import Path

import pandas as pd


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PERTURBATION_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "perturbations"
    / "perturbations.csv"
)

DATASET_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "metadata"
    / "dataset.csv"
)


# ============================================================
# Expected conditions
# ============================================================

EXPECTED_CONDITIONS = {
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
}


# ============================================================
# Utility
# ============================================================

def normalize_text(value):
    """Convert missing values to an empty string."""

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# Validation
# ============================================================

def validate():

    if not PERTURBATION_CSV.exists():
        raise FileNotFoundError(
            f"Perturbation file not found:\n{PERTURBATION_CSV}"
        )

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{DATASET_CSV}"
        )

    perturbations = pd.read_csv(
        PERTURBATION_CSV
    )

    dataset = pd.read_csv(
        DATASET_CSV
    )

    print("=" * 60)
    print("IU-Xray Perturbation Validation")
    print("=" * 60)

    print(
        f"Perturbation records: {len(perturbations)}"
    )

    print(
        f"Original dataset cases: {len(dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "sample_id",
        "condition",
        "image_path",
        "original_context",
        "perturbed_context",
        "transformation_type",
        "source_context_id",
        "reference_findings",
        "reference_impression",
        "manual_review_required",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in perturbations.columns
    ]

    if missing_columns:
        raise AssertionError(
            f"Missing columns: {missing_columns}"
        )

    print("[PASS] Required columns exist.")

    # --------------------------------------------------------
    # Condition validation
    # --------------------------------------------------------

    actual_conditions = set(
        perturbations["condition"]
    )

    invalid_conditions = (
        actual_conditions
        - EXPECTED_CONDITIONS
    )

    if invalid_conditions:
        raise AssertionError(
            f"Invalid conditions: {invalid_conditions}"
        )

    print("[PASS] Condition values are valid.")

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicates = perturbations.duplicated(
        subset=["sample_id", "condition"],
        keep=False,
    )

    duplicate_count = int(
        duplicates.sum()
    )

    if duplicate_count > 0:
        print(
            f"[WARNING] Duplicate sample/condition rows: "
            f"{duplicate_count}"
        )
    else:
        print(
            "[PASS] No duplicate sample/condition pairs."
        )

    # --------------------------------------------------------
    # Sample ID validation
    # --------------------------------------------------------

    valid_sample_ids = set(
        dataset["sample_id"]
    )

    invalid_sample_ids = (
        set(perturbations["sample_id"])
        - valid_sample_ids
    )

    if invalid_sample_ids:
        raise AssertionError(
            f"Unknown sample IDs: {invalid_sample_ids}"
        )

    print("[PASS] All sample IDs exist in dataset.")

    # --------------------------------------------------------
    # Image validation
    # --------------------------------------------------------

    missing_images = []

    for _, row in perturbations.iterrows():

        image_path = normalize_text(
            row["image_path"]
        )

        full_path = (
            ROOT
            / "data"
            / "processed"
            / "iu_xray"
            / "images"
            / Path(image_path).name
        )

        if not full_path.exists():
            missing_images.append(
                (
                    row["sample_id"],
                    image_path,
                )
            )

    if missing_images:
        raise AssertionError(
            f"Missing images: {missing_images}"
        )

    print("[PASS] All referenced frontal images exist.")

    # --------------------------------------------------------
    # E1 — Sufficient
    # --------------------------------------------------------

    sufficient = perturbations[
        perturbations["condition"] == "sufficient"
    ]

    for _, row in sufficient.iterrows():

        original = normalize_text(
            row["original_context"]
        )

        perturbed = normalize_text(
            row["perturbed_context"]
        )

        if original != perturbed:
            raise AssertionError(
                f"E1 context mismatch: {row['sample_id']}"
            )

    print(
        "[PASS] E1 sufficient contexts equal original contexts."
    )

    # --------------------------------------------------------
    # E2 — Incomplete
    # --------------------------------------------------------

    incomplete = perturbations[
        perturbations["condition"] == "incomplete"
    ]

    for _, row in incomplete.iterrows():

        original = normalize_text(
            row["original_context"]
        )

        perturbed = normalize_text(
            row["perturbed_context"]
        )

        if not perturbed:
            raise AssertionError(
                f"E2 empty context: {row['sample_id']}"
            )

        if perturbed == original:
            raise AssertionError(
                f"E2 unchanged context: {row['sample_id']}"
            )

        if row["transformation_type"] != (
            "partial_context_reduction"
        ):
            raise AssertionError(
                f"Unexpected E2 transformation: "
                f"{row['sample_id']}"
            )

    print(
        "[PASS] E2 incomplete contexts are non-empty and reduced."
    )

    # --------------------------------------------------------
    # E3 — Irrelevant
    # --------------------------------------------------------

    irrelevant = perturbations[
        perturbations["condition"] == "irrelevant"
    ]

    for _, row in irrelevant.iterrows():

        if (
            row["sample_id"]
            == row["source_context_id"]
        ):
            raise AssertionError(
                f"E3 source equals target: "
                f"{row['sample_id']}"
            )

        if not normalize_text(
            row["perturbed_context"]
        ):
            raise AssertionError(
                f"E3 empty context: "
                f"{row['sample_id']}"
            )

    print(
        "[PASS] E3 contexts come from different cases."
    )

    # --------------------------------------------------------
    # E4 — Conflicting
    # --------------------------------------------------------

    conflicting = perturbations[
         perturbations["condition"] == "conflicting"
     ]

    for _, row in conflicting.iterrows():

        context = normalize_text(
           row["perturbed_context"]
        )

        if not context:
            raise AssertionError(
              f"E4 has empty conflicting context: "
              f"{row['sample_id']}"
            )

        if bool(
            row["manual_review_required"]
        ):
            raise AssertionError(
                f"E4 still requires manual review: "
                f"{row['sample_id']}"
            )

    print(
        "[PASS] E4 conflicting contexts are populated "
        "and finalized."
    )

    # --------------------------------------------------------
    # E5 — Insufficient
    # --------------------------------------------------------

    insufficient = perturbations[
        perturbations["condition"] == "insufficient"
    ]

    for _, row in insufficient.iterrows():

        context = normalize_text(
            row["perturbed_context"]
        )

        if context:
            raise AssertionError(
                f"E5 contains context: "
                f"{row['sample_id']}"
            )

    print(
        "[PASS] E5 contexts are empty."
    )

    # --------------------------------------------------------
    # Reference evidence validation
    # --------------------------------------------------------

    missing_reference = perturbations[
        (
            perturbations["reference_findings"]
            .fillna("")
            .str.strip()
            == ""
        )
        |
        (
            perturbations["reference_impression"]
            .fillna("")
            .str.strip()
            == ""
        )
    ]

    if not missing_reference.empty:

        print(
            "[WARNING] Records with missing reference evidence:"
        )

        print(
            missing_reference[
                [
                    "sample_id",
                    "condition",
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "[PASS] Reference findings and impressions "
            "are present."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("Condition counts:")
    print(
        perturbations["condition"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 60)
    print("VALIDATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    validate()