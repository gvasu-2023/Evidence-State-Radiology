"""
Deterministic rule-execution evaluation of the current prototype.

Evaluates EvidenceStateAnalyzer and ReliabilityGate on the 81 controlled
perturbation records from data/processed/iu_xray/perturbations/perturbations.csv.

Outputs:
    results/tables/reliability_gate_evaluation.csv

Terminology: This is a deterministic rule-execution evaluation of the current
prototype. It is not clinical validation, reliability improvement,
hallucination reduction, factuality improvement, model accuracy, or clinical accuracy.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.states import EvidenceState
from src.gating.reliability_gate import ReliabilityGate

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
OUTPUT_PATH = ROOT / "results/tables/reliability_gate_evaluation.csv"

EXPECTED_GATE_ACTIONS = {
    "sufficient": "generate",
    "incomplete": "qualify",
    "irrelevant": "discount_context",
    "conflicting": "qualify_or_abstain",
    "insufficient": "abstain",
}

EXPECTED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_perturbations() -> pd.DataFrame:
    """Load baseline controlled perturbation records."""
    if not PERTURBATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Perturbations CSV not found: {PERTURBATIONS_PATH}"
        )

    df = pd.read_csv(PERTURBATIONS_PATH)
    if len(df) != 81:
        raise ValueError(f"Expected 81 perturbation records, found {len(df)}")
    return df


def evaluate_record(row: pd.Series, analyzer: EvidenceStateAnalyzer, gate: ReliabilityGate) -> dict:
    """Evaluate a single perturbation record deterministically."""
    sample_id = str(row["sample_id"])
    experimental_condition = str(row["condition"])
    transformation_type = str(row["transformation_type"])

    image_path_str = str(row["image_path"]) if pd.notna(row["image_path"]) else ""
    image_available = bool(image_path_str and Path(image_path_str).exists())

    context_str = (
        str(row["perturbed_context"]).strip()
        if pd.notna(row["perturbed_context"])
        else ""
    )
    context_available = bool(context_str != "")

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
    predicted_evidence_state = predicted_state_enum.value

    gate_decision = gate.decide(predicted_state_enum)
    gate_action = gate_decision.action
    gate_reason = gate_decision.reason

    expected_gate_action = EXPECTED_GATE_ACTIONS.get(experimental_condition, "")

    state_matches_condition = bool(
        predicted_evidence_state == experimental_condition
    )
    gate_action_matches_expected = bool(
        gate_action == expected_gate_action
    )

    return {
        "sample_id": sample_id,
        "experimental_condition": experimental_condition,
        "transformation_type": transformation_type,
        "image_available": image_available,
        "context_available": context_available,
        "context_relevant": context_relevant,
        "context_consistent": context_consistent,
        "evidence_strength": evidence_strength,
        "predicted_evidence_state": predicted_evidence_state,
        "state_matches_condition": state_matches_condition,
        "gate_action": gate_action,
        "gate_reason": gate_reason,
        "expected_gate_action": expected_gate_action,
        "gate_action_matches_expected": gate_action_matches_expected,
    }


def run_evaluation() -> pd.DataFrame:
    """Run evaluation over all 81 perturbation records."""
    df = load_perturbations()
    analyzer = EvidenceStateAnalyzer()
    gate = ReliabilityGate()

    results = [evaluate_record(row, analyzer, gate) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(results)

    if len(eval_df) != 81:
        raise RuntimeError(f"Expected 81 evaluation output rows, got {len(eval_df)}")

    return eval_df


def main() -> None:
    eval_df = run_evaluation()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False)

    total_records = len(eval_df)
    overall_state_alignment_rate = eval_df["state_matches_condition"].mean()
    overall_gate_match_rate = eval_df["gate_action_matches_expected"].mean()

    print("=" * 80)
    print("PHASE 9: RELIABILITY GATE EVALUATION")
    print("Terminology: deterministic rule-execution evaluation of the current prototype")
    print("=" * 80)
    print(f"Total perturbation records evaluated: {total_records}")
    print(f"Overall State Alignment Rate: {overall_state_alignment_rate:.4f} ({eval_df['state_matches_condition'].sum()}/{total_records})")
    print(f"Overall Expected Gate-Action Match Rate: {overall_gate_match_rate:.4f} ({eval_df['gate_action_matches_expected'].sum()}/{total_records})")

    print("\n--- STATE ALIGNMENT BY CONDITION ---")
    cond_summary = []
    for cond in EXPECTED_CONDITIONS:
        subset = eval_df[eval_df["experimental_condition"] == cond]
        n_cond = len(subset)
        align_rate = subset["state_matches_condition"].mean() if n_cond else 0.0
        match_count = subset["state_matches_condition"].sum()
        states_predicted = subset["predicted_evidence_state"].value_counts().to_dict()
        cond_summary.append({
            "condition": cond,
            "count": n_cond,
            "aligned_count": match_count,
            "alignment_rate": f"{align_rate:.4f}",
            "predicted_states": str(states_predicted),
        })

    summary_df = pd.DataFrame(cond_summary)
    print(summary_df.to_string(index=False))

    print("\n--- GATE ACTION DISTRIBUTION BY CONDITION ---")
    gate_dist_rows = []
    for cond in EXPECTED_CONDITIONS:
        subset = eval_df[eval_df["experimental_condition"] == cond]
        n_cond = len(subset)
        actions = subset["gate_action"].value_counts().to_dict()
        gate_dist_rows.append({
            "condition": cond,
            "count": n_cond,
            "expected_action": EXPECTED_GATE_ACTIONS.get(cond, ""),
            "actual_actions": str(actions),
        })

    gate_dist_df = pd.DataFrame(gate_dist_rows)
    print(gate_dist_df.to_string(index=False))

    print("\n--- OBSERVED STRUCTURAL DISCREPANCIES (PROTOTYPE ANALYZER RULES) ---")
    print("1. incomplete condition (partial context reduction):")
    inc_subset = eval_df[eval_df["experimental_condition"] == "incomplete"]
    print(f"   - Predicted State: {set(inc_subset['predicted_evidence_state'])} (Expected: 'incomplete')")
    print(f"   - Triggered Gate Action: {set(inc_subset['gate_action'])} (Expected: 'qualify')")
    print("   - Root cause: Current analyzer lacks a context_complete field; present context yields 'sufficient'.")

    print("2. insufficient condition (context removed with ES=1.0):")
    ins_subset = eval_df[eval_df["experimental_condition"] == "insufficient"]
    print(f"   - Predicted State: {set(ins_subset['predicted_evidence_state'])} (Expected: 'insufficient')")
    print(f"   - Triggered Gate Action: {set(ins_subset['gate_action'])} (Expected: 'abstain')")
    print("   - Root cause: When context is missing but image is present (ES=1.0), analyzer predicts 'incomplete' ('qualify').")

    print(f"\nEvaluation table saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
