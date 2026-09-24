from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "metadata"
    / "dataset.csv"
)

ELIGIBILITY_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "metadata"
    / "eligibility.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "perturbations"
)

OUTPUT_CSV = OUTPUT_DIR / "perturbations.csv"


# ============================================================
# Constants
# ============================================================

CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


# ============================================================
# Utility functions
# ============================================================

def load_inputs():
    """Load dataset and eligibility metadata."""

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_CSV}"
        )

    if not ELIGIBILITY_CSV.exists():
        raise FileNotFoundError(
            f"Eligibility file not found:\n{ELIGIBILITY_CSV}"
        )

    dataset = pd.read_csv(DATASET_CSV)
    eligibility = pd.read_csv(ELIGIBILITY_CSV)

    return dataset, eligibility


def normalize_text(value):
    """Convert missing values to empty strings."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def create_incomplete_context(context):
    """
    Create a controlled incomplete-context version.

    The transformation removes one or more complete clinical
    components while retaining at least one meaningful component.

    No characters are truncated and no new medical information
    is introduced.

    Returns:
        str | None:
            Reduced context if a valid transformation is possible.
    """

    context = normalize_text(context)

    if not context:
        return None

    # --------------------------------------------------------
    # Step 1: Split explicit sentence-level components.
    # --------------------------------------------------------

    sentences = [
        sentence.strip()
        for sentence in context.split(".")
        if sentence.strip()
    ]

    # --------------------------------------------------------
    # Step 2: If multiple sentences exist, retain the most
    # clinically explicit sentence(s).
    # --------------------------------------------------------

    if len(sentences) >= 2:

        # Remove obvious demographic/de-identification-only
        # components where possible.
        meaningful_sentences = [
            sentence
            for sentence in sentences
            if sentence.replace("XXXX", "").strip()
        ]

        if len(meaningful_sentences) >= 2:
            # Retain the final meaningful component.
            # This avoids arbitrary character truncation.
            return meaningful_sentences[-1].strip() + "."

        if len(meaningful_sentences) == 1:
            candidate = meaningful_sentences[0].strip()

            if candidate:
                return candidate + "."

    # --------------------------------------------------------
    # Step 3: Handle comma-separated clinical components.
    # --------------------------------------------------------

    if "," in context:

        components = [
            component.strip()
            for component in context.split(",")
            if component.strip()
        ]

        meaningful_components = [
            component
            for component in components
            if component.replace("XXXX", "").strip()
        ]

        if len(meaningful_components) >= 2:

            # Retain the final meaningful component.
            candidate = meaningful_components[-1].strip()

            if candidate:
                return candidate

    # --------------------------------------------------------
    # Step 4: Handle "and"-separated components.
    # --------------------------------------------------------

    lower_context = context.lower()

    if " and " in lower_context:

        components = [
            component.strip()
            for component in context.split(" and ")
            if component.strip()
        ]

        meaningful_components = [
            component
            for component in components
            if component.replace("XXXX", "").strip()
        ]

        if len(meaningful_components) >= 2:

            candidate = meaningful_components[-1].strip()

            if candidate:
                return candidate

    # --------------------------------------------------------
    # Step 5: No defensible reduction available.
    # --------------------------------------------------------

    return None


def choose_irrelevant_source(
    target_index,
    dataset,
):
    """
    Select a deterministic clinical context from another case.

    The source case must be different from the target case.

    A simple deterministic next-case rule is used here so that
    the experiment is reproducible.
    """

    if len(dataset) < 2:
        raise ValueError(
            "At least two cases are required for irrelevant-context "
            "perturbation."
        )

    source_index = (target_index + 1) % len(dataset)

    source_row = dataset.iloc[source_index]

    return normalize_text(source_row["clinical_context"]), source_row["sample_id"]


# ============================================================
# Perturbation generation
# ============================================================

def generate_perturbations(dataset, eligibility):
    """
    Generate the controlled perturbation table.

    E1 Sufficient:
        Original image + original clinical context.

    E2 Incomplete:
        Original image + partial clinical context.

    E3 Irrelevant:
        Original image + context from another case.

    E4 Conflicting:
        Candidate entry only. No automatic medical contradiction
        is generated in this version.

    E5 Insufficient:
        Original image + no clinical context.
    """

    eligibility_lookup = eligibility.set_index("sample_id")

    records = []

    for target_index, row in dataset.iterrows():

        sample_id = row["sample_id"]

        if sample_id not in eligibility_lookup.index:
            raise ValueError(
                f"Missing eligibility information for {sample_id}"
            )

        eligible = eligibility_lookup.loc[sample_id]

        image_path = normalize_text(row["frontal_image"])
        original_context = normalize_text(row["clinical_context"])
        findings = normalize_text(row["findings"])
        impression = normalize_text(row["impression"])

        # ----------------------------------------------------
        # E1 — Sufficient
        # ----------------------------------------------------

        if bool(eligible["eligible_sufficient"]):

            records.append(
                {
                    "sample_id": sample_id,
                    "condition": "sufficient",
                    "image_path": image_path,
                    "original_context": original_context,
                    "perturbed_context": original_context,
                    "transformation_type": "none",
                    "source_context_id": sample_id,
                    "reference_findings": findings,
                    "reference_impression": impression,
                    "manual_review_required": False,
                }
            )

        # ----------------------------------------------------
        # E2 — Incomplete
        # ----------------------------------------------------

        if bool(eligible["eligible_incomplete"]):

            incomplete_context = create_incomplete_context(
                original_context
            )

            if incomplete_context is not None:

                records.append(
                    {
                        "sample_id": sample_id,
                        "condition": "incomplete",
                        "image_path": image_path,
                        "original_context": original_context,
                        "perturbed_context": incomplete_context,
                        "transformation_type": "partial_context_reduction",
                        "source_context_id": sample_id,
                        "reference_findings": findings,
                        "reference_impression": impression,
                        "manual_review_required": False,
                    }
                )

        # ----------------------------------------------------
        # E3 — Irrelevant
        # ----------------------------------------------------

        if bool(eligible["eligible_irrelevant"]):

            irrelevant_context, source_id = choose_irrelevant_source(
                target_index,
                dataset,
            )

            records.append(
                {
                    "sample_id": sample_id,
                    "condition": "irrelevant",
                    "image_path": image_path,
                    "original_context": original_context,
                    "perturbed_context": irrelevant_context,
                    "transformation_type": "cross_case_context_swap",
                    "source_context_id": source_id,
                    "reference_findings": findings,
                    "reference_impression": impression,
                    "manual_review_required": False,
                }
            )

        # ----------------------------------------------------
        # E4 — Conflicting
        # ----------------------------------------------------

        if bool(eligible["eligible_conflicting"]):

            records.append(
                {
                    "sample_id": sample_id,
                    "condition": "conflicting",
                    "image_path": image_path,
                    "original_context": original_context,
                    "perturbed_context": "",
                    "transformation_type": "manual_evidence_based_contradiction_required",
                    "source_context_id": sample_id,
                    "reference_findings": findings,
                    "reference_impression": impression,
                    "manual_review_required": True,
                }
            )

        # ----------------------------------------------------
        # E5 — Insufficient
        # ----------------------------------------------------

        if bool(eligible["eligible_insufficient"]):

            records.append(
                {
                    "sample_id": sample_id,
                    "condition": "insufficient",
                    "image_path": image_path,
                    "original_context": original_context,
                    "perturbed_context": "",
                    "transformation_type": "context_removed",
                    "source_context_id": sample_id,
                    "reference_findings": findings,
                    "reference_impression": impression,
                    "manual_review_required": False,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# Validation
# ============================================================

def validate_perturbations(df):
    """Run basic structural validation."""

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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing perturbation columns: {missing}"
        )

    invalid_conditions = set(df["condition"]) - set(CONDITIONS)

    if invalid_conditions:
        raise ValueError(
            f"Invalid conditions found: {invalid_conditions}"
        )

    # Sufficient must retain original context.
    sufficient = df[df["condition"] == "sufficient"]

    if not sufficient.empty:
        assert (
            sufficient["perturbed_context"]
            == sufficient["original_context"]
        ).all()

    # Insufficient must have empty context.
    insufficient = df[df["condition"] == "insufficient"]

    if not insufficient.empty:
        assert (
            insufficient["perturbed_context"].fillna("") == ""
        ).all()

    # Conflicting entries must currently be marked for manual review.
    conflicting = df[df["condition"] == "conflicting"]

    if not conflicting.empty:
        assert (
            conflicting["manual_review_required"]
        ).all()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("IU-Xray Controlled Perturbation Generator")
    print("=" * 60)

    dataset, eligibility = load_inputs()

    print(f"Dataset cases: {len(dataset)}")
    print(f"Eligibility records: {len(eligibility)}")
    print()

    perturbations = generate_perturbations(
        dataset,
        eligibility,
    )

    validate_perturbations(perturbations)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    perturbations.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("Generated perturbation records:")
    print()

    print(
        perturbations["condition"]
        .value_counts()
        .reindex(CONDITIONS, fill_value=0)
    )

    print()
    print(
        f"Total perturbation records: "
        f"{len(perturbations)}"
    )

    print()
    print("Manual-review records:")
    print(
        int(
            perturbations["manual_review_required"]
            .sum()
        )
    )

    print()
    print("Saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()