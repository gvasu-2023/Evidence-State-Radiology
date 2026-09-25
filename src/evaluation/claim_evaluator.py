from pathlib import Path

import pandas as pd
import yaml


CLAIMS_PATH = Path("results/tables/extracted_claims.csv")
CONFLICT_PATH = Path("configs/conflict_cases.yaml")
ONTOLOGY_PATH = Path("configs/claim_ontology.yaml")
OUTPUT_PATH = Path("results/tables/conflict_claim_evaluation.csv")


VALID_POLARITIES = {
    "AFFIRMED",
    "NEGATED",
    "MENTIONED_UNCLEAR",
}


def load_approved_conflicts() -> list[dict]:
    with CONFLICT_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    conflicts = data.get("conflicts")

    if not isinstance(conflicts, dict):
        raise ValueError("Expected 'conflicts' to be a mapping.")

    approved_cases = []

    for sample_id, case in conflicts.items():
        if not case.get("approved", False):
            continue

        target = str(case.get("target", "")).strip()

        if not target:
            raise ValueError(
                f"Approved case {sample_id} has an empty target."
            )

        approved_cases.append(
            {
                "sample_id": str(sample_id),
                "target": target,
                "type": str(case.get("type", "")),
                "conflicting_context": str(
                    case.get("conflicting_context", "")
                ),
            }
        )

    return approved_cases


def load_ontology() -> dict:
    with ONTOLOGY_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    mappings = data.get("finding_mappings")

    if not isinstance(mappings, dict):
        raise ValueError(
            "Expected 'finding_mappings' to be a mapping."
        )

    return mappings


def normalize_finding(finding: str) -> str:
    return str(finding).strip().lower()


def resolve_extractor_finding(
    target: str,
    ontology: dict,
) -> str:
    target_normalized = normalize_finding(target)

    mapping = ontology.get(target_normalized)

    if mapping is None:
        raise ValueError(
            f"No ontology mapping found for target: {target}"
        )

    return normalize_finding(
        mapping["extractor_finding"]
    )


def find_matching_claims(
    sample_id: str,
    extractor_finding: str,
    claims_df: pd.DataFrame,
) -> pd.DataFrame:

    return claims_df[
        (claims_df["sample_id"].astype(str) == sample_id)
        &
        (
            claims_df["finding"].map(normalize_finding)
            == extractor_finding
        )
    ].copy()


def evaluate_case(
    case: dict,
    ontology: dict,
    claims_df: pd.DataFrame,
) -> dict:

    sample_id = case["sample_id"]
    target = case["target"]

    extractor_finding = resolve_extractor_finding(
        target,
        ontology,
    )

    matching_claims = find_matching_claims(
        sample_id,
        extractor_finding,
        claims_df,
    )

    if matching_claims.empty:
        return {
            "sample_id": sample_id,
            "target": target,
            "extractor_finding": extractor_finding,
            "type": case["type"],
            "conflicting_context": case[
                "conflicting_context"
            ],
            "generated_polarity": "NOT_MENTIONED",
            "evaluation": "NOT_MENTIONED",
            "source_sentence": "",
        }

    polarities = [
        str(p)
        for p in matching_claims["polarity"].tolist()
        if str(p) in VALID_POLARITIES
    ]

    if not polarities:
        return {
            "sample_id": sample_id,
            "target": target,
            "extractor_finding": extractor_finding,
            "type": case["type"],
            "conflicting_context": case[
                "conflicting_context"
            ],
            "generated_polarity": "MENTIONED_UNCLEAR",
            "evaluation": "MENTIONED_UNCLEAR",
            "source_sentence": "",
        }

    unique_polarities = sorted(set(polarities))

    if len(unique_polarities) > 1:
        generated_polarity = "MENTIONED_UNCLEAR"
        evaluation = "MENTIONED_UNCLEAR"
    else:
        generated_polarity = unique_polarities[0]
        evaluation = generated_polarity

    sentences = sorted(
        set(
            matching_claims["source_sentence"]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    return {
        "sample_id": sample_id,
        "target": target,
        "extractor_finding": extractor_finding,
        "type": case["type"],
        "conflicting_context": case[
            "conflicting_context"
        ],
        "generated_polarity": generated_polarity,
        "evaluation": evaluation,
        "source_sentence": " | ".join(sentences),
    }


def main() -> None:

    for path in [
        CLAIMS_PATH,
        CONFLICT_PATH,
        ONTOLOGY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    claims_df = pd.read_csv(CLAIMS_PATH)

    required_columns = {
        "sample_id",
        "finding",
        "polarity",
        "source_sentence",
    }

    missing = required_columns - set(claims_df.columns)

    if missing:
        raise ValueError(
            f"Missing claim columns: {sorted(missing)}"
        )

    approved_cases = load_approved_conflicts()
    ontology = load_ontology()

    results = [
        evaluate_case(
            case,
            ontology,
            claims_df,
        )
        for case in approved_cases
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
    print("CONFLICT CLAIM EVALUATION")
    print("=" * 70)

    print(
        f"Approved conflict cases: {len(output_df)}"
    )

    print("\nEvaluation:")

    print(
        output_df["evaluation"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCase-level results:")

    print(
        output_df[
            [
                "sample_id",
                "target",
                "extractor_finding",
                "generated_polarity",
                "evaluation",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
