"""
Phase 12A: Evidence-State Analyzer Evaluation.

Evaluates the CURRENT EvidenceStateAnalyzer prototype directly against controlled
experimental perturbation conditions from data/processed/iu_xray/perturbations/perturbations.csv.

Outputs:
    results/tables/phase12_analyzer_evaluation.csv
    results/tables/phase12_analyzer_summary.csv

Terminology: Deterministic rule-execution evaluation of the current analyzer prototype.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.states import EvidenceState

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
EVALUATION_OUTPUT_PATH = ROOT / "results/tables/phase12_analyzer_evaluation.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/phase12_analyzer_summary.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_perturbations() -> pd.DataFrame:
    if not PERTURBATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Perturbations file missing: {PERTURBATIONS_PATH}"
        )

    df = pd.read_csv(PERTURBATIONS_PATH)
    if len(df) != 81:
        raise ValueError(f"Expected 81 perturbation records, got {len(df)}")
    return df


def evaluate_record(row: pd.Series, analyzer: EvidenceStateAnalyzer) -> dict:
    sample_id = str(row["sample_id"])
    condition = str(row["condition"])
    transformation_type = str(row["transformation_type"])

    image_path_str = str(row["image_path"]) if pd.notna(row["image_path"]) else ""
    image_available = bool(image_path_str and Path(image_path_str).exists())

    context_str = (
        str(row["perturbed_context"]).strip()
        if pd.notna(row["perturbed_context"])
        else ""
    )
    context_available = bool(context_str != "")
    perturbed_context_length = len(context_str)

    context_relevant = transformation_type != "cross_case_context_swap"
    context_consistent = transformation_type not in (
        "negative_finding_contradiction",
        "positive_finding_negation",
    )
    evidence_strength = 1.0

    assessment = EvidenceAssessment(
        image_available=image_available,
        context_available=context_available,
        context_relevant=context_relevant,
        context_consistent=context_consistent,
        evidence_strength=evidence_strength,
    )

    predicted_state_enum = analyzer.classify(assessment)
    predicted_state = predicted_state_enum.value
    expected_state = condition
    state_match = bool(predicted_state == expected_state)

    return {
        "sample_id": sample_id,
        "condition": condition,
        "predicted_state": predicted_state,
        "expected_state": expected_state,
        "state_match": state_match,
        "image_available": image_available,
        "context_available": context_available,
        "context_relevant": context_relevant,
        "context_consistent": context_consistent,
        "evidence_strength": evidence_strength,
        "transformation_type": transformation_type,
        "perturbed_context_length": perturbed_context_length,
    }


def run_analyzer_evaluation() -> pd.DataFrame:
    df = load_perturbations()
    analyzer = EvidenceStateAnalyzer()

    records = [evaluate_record(row, analyzer) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(records)

    if len(eval_df) != 81:
        raise RuntimeError(f"Expected 81 evaluation rows, got {len(eval_df)}")

    return eval_df


def build_analyzer_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary table containing overall performance, condition breakdown, and confusion matrix."""

    summary_rows = []

    subsets = [("overall", eval_df)] + [
        (c, eval_df[eval_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        total_cases = len(subset)
        correct_cases = int(subset["state_match"].sum())
        alignment_rate = (
            float(correct_cases / total_cases) if total_cases > 0 else 0.0
        )

        pred_dist = str(subset["predicted_state"].value_counts().to_dict())
        exp_dist = str(subset["expected_state"].value_counts().to_dict())

        # Confusion matrix counts for this scope
        pred_counts = subset["predicted_state"].value_counts().to_dict()
        pred_suff = int(pred_counts.get("sufficient", 0))
        pred_inc = int(pred_counts.get("incomplete", 0))
        pred_irr = int(pred_counts.get("irrelevant", 0))
        pred_conf = int(pred_counts.get("conflicting", 0))
        pred_insuff = int(pred_counts.get("insufficient", 0))

        summary_rows.append(
            {
                "condition": scope,
                "total_cases": total_cases,
                "correct_cases": correct_cases,
                "alignment_rate": alignment_rate,
                "predicted_state_distribution": pred_dist,
                "expected_state_distribution": exp_dist,
                "pred_sufficient": pred_suff,
                "pred_incomplete": pred_inc,
                "pred_irrelevant": pred_irr,
                "pred_conflicting": pred_conf,
                "pred_insufficient": pred_insuff,
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    eval_df = run_analyzer_evaluation()
    summary_df = build_analyzer_summary(eval_df)

    EVALUATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    total_cases = len(eval_df)
    correct_cases = int(eval_df["state_match"].sum())
    overall_alignment = correct_cases / total_cases

    print("=" * 80)
    print("PHASE 12A: EVIDENCE-STATE ANALYZER EVALUATION")
    print("Terminology: Deterministic rule-execution evaluation of current prototype")
    print("=" * 80)
    print(f"Total Evaluations: {total_cases}")
    print(f"Correct Predictions: {correct_cases}")
    print(f"Overall Alignment Rate: {overall_alignment:.4f} ({overall_alignment*100:.2f}%)")

    print("\n--- PER-CONDITION ALIGNMENT ---")
    cond_df = summary_df[summary_df["condition"] != "overall"]
    print(
        cond_df[
            [
                "condition",
                "total_cases",
                "correct_cases",
                "alignment_rate",
                "predicted_state_distribution",
            ]
        ].to_string(index=False)
    )

    print("\n--- CONFUSION MATRIX (Expected Condition vs Predicted State) ---")
    conf_matrix = pd.crosstab(
        eval_df["condition"],
        eval_df["predicted_state"],
        margins=True,
        margins_name="Total",
    ).reindex(
        index=REQUIRED_CONDITIONS + ["Total"],
        columns=REQUIRED_CONDITIONS + ["Total"],
        fill_value=0,
    )
    print(conf_matrix.to_string())

    print(f"\nEvaluation detail saved to: {EVALUATION_OUTPUT_PATH}")
    print(f"Summary report saved to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
