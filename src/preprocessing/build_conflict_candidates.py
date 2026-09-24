from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "perturbations"
    / "perturbations.csv"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "iu_xray"
    / "metadata"
    / "conflict_candidates.csv"
)


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Perturbation file not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    candidates = df[
        df["condition"] == "conflicting"
    ].copy()

    candidates = candidates[
        [
            "sample_id",
            "original_context",
            "reference_findings",
            "reference_impression",
            "transformation_type",
            "manual_review_required",
        ]
    ]

    # Fields to be completed during manual conflict design.
    candidates["conflict_target"] = ""
    candidates["conflict_type"] = ""
    candidates["conflicting_context"] = ""
    candidates["evidence_basis"] = ""
    candidates["conflict_approved"] = False

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("=" * 60)
    print("E4 Conflict Candidate Builder")
    print("=" * 60)

    print(
        f"Conflict candidates: {len(candidates)}"
    )

    print()

    print(
        candidates[
            [
                "sample_id",
                "original_context",
                "reference_impression",
            ]
        ].to_string(index=False)
    )

    print()

    print("Saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()