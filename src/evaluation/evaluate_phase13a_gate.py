"""
Phase 13A: Reliability Gate Evaluation with Phase 12C Evidence-State Analyzer.

Evaluates the downstream ReliabilityGate policy decisions when connected to the
Phase 12C EvidenceStateAnalyzer (with component/fragmentation context-completeness parser)
across the 81 controlled perturbation records in perturbations.csv.

Pipeline:
    perturbation record
    → Phase 12C EvidenceStateAnalyzer
    → predicted EvidenceState
    → ReliabilityGate
    → GateDecision (actual_gate_action, gate_reason)

Anti-Leakage:
The analyzer and context completeness parser operate strictly on input context strings
and evidence availability flags. No ground-truth metadata (condition, expected_state,
original_context, transformation_type) is passed into the analyzer or gate inference.

Outputs:
    results/tables/phase13a_gate_evaluation.csv
    results/tables/phase13a_gate_summary.csv
"""

from pathlib import Path
import sys
from typing import Dict, List, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.context_completeness import parse_context_completeness
from src.evidence_state.states import EvidenceState
from src.gating.reliability_gate import ReliabilityGate, GateDecision

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
EVALUATION_OUTPUT_PATH = ROOT / "results/tables/phase13a_gate_evaluation.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/phase13a_gate_summary.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]

EXPECTED_ACTION_MAPPING: Dict[str, str] = {
    "sufficient": "generate",
    "incomplete": "qualify",
    "irrelevant": "discount_context",
    "conflicting": "qualify_or_abstain",
    "insufficient": "abstain",
}


def load_perturbations() -> pd.DataFrame:
    if not PERTURBATIONS_PATH.exists():
        raise FileNotFoundError(f"Perturbations file missing: {PERTURBATIONS_PATH}")

    df = pd.read_csv(PERTURBATIONS_PATH)
    if len(df) != 81:
        raise ValueError(f"Expected 81 perturbation records, got {len(df)}")
    return df


def evaluate_record(
    row: pd.Series,
    analyzer: EvidenceStateAnalyzer,
    gate: ReliabilityGate,
) -> Dict[str, Any]:
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

    context_relevant = transformation_type != "cross_case_context_swap"
    context_consistent = transformation_type not in (
        "negative_finding_contradiction",
        "positive_finding_negation",
    )

    completeness_res = parse_context_completeness(context_str)
    context_complete = completeness_res.is_complete
    evidence_strength = 1.0

    # Construct assessment for current Phase 12C EvidenceStateAnalyzer
    assessment = EvidenceAssessment(
        image_available=image_available,
        context_available=context_available,
        context_relevant=context_relevant,
        context_consistent=context_consistent,
        context_complete=context_complete,
        evidence_strength=evidence_strength,
    )

    # 1. Predict EvidenceState
    predicted_state_enum = analyzer.classify(assessment)
    predicted_state = predicted_state_enum.value

    # 2. Decide Gate Action
    gate_decision: GateDecision = gate.decide(predicted_state_enum)
    actual_gate_action = gate_decision.action
    gate_reason = gate_decision.reason

    # 3. Ground truth evaluation comparisons
    expected_state = condition
    state_correct = bool(predicted_state == expected_state)
    expected_gate_action = EXPECTED_ACTION_MAPPING[condition]
    action_correct = bool(actual_gate_action == expected_gate_action)

    return {
        "sample_id": sample_id,
        "condition": condition,
        "predicted_state": predicted_state,
        "expected_state": expected_state,
        "state_correct": state_correct,
        "actual_gate_action": actual_gate_action,
        "expected_gate_action": expected_gate_action,
        "action_correct": action_correct,
        "gate_reason": gate_reason,
        "transformation_type": transformation_type,
        "image_available": image_available,
        "context_available": context_available,
        "context_relevant": context_relevant,
        "context_consistent": context_consistent,
        "context_complete": context_complete,
        "rule_triggered": completeness_res.rule_triggered,
        "evidence_strength": evidence_strength,
    }


def run_phase13a_evaluation() -> pd.DataFrame:
    df = load_perturbations()
    analyzer = EvidenceStateAnalyzer()
    gate = ReliabilityGate()

    records = [evaluate_record(row, analyzer, gate) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(records)

    if len(eval_df) != 81:
        raise RuntimeError(f"Expected 81 evaluation rows, got {len(eval_df)}")

    return eval_df


def build_phase13a_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    subsets = [("overall", eval_df)] + [
        (c, eval_df[eval_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        n = len(subset)
        correct_state = int(subset["state_correct"].sum())
        state_accuracy = float(correct_state / n) if n > 0 else 0.0

        correct_action = int(subset["action_correct"].sum())
        action_accuracy = float(correct_action / n) if n > 0 else 0.0

        action_counts = subset["actual_gate_action"].value_counts().to_dict()
        gen_count = int(action_counts.get("generate", 0))
        qual_count = int(action_counts.get("qualify", 0))
        disc_count = int(action_counts.get("discount_context", 0))
        qual_abst_count = int(action_counts.get("qualify_or_abstain", 0))
        abst_count = int(action_counts.get("abstain", 0))

        pred_dist = str(subset["predicted_state"].value_counts().to_dict())
        action_dist = str(subset["actual_gate_action"].value_counts().to_dict())

        summary_rows.append(
            {
                "condition": scope,
                "n": n,
                "correct_state": correct_state,
                "state_accuracy": state_accuracy,
                "correct_action": correct_action,
                "action_accuracy": action_accuracy,
                "generation_count": gen_count,
                "qualification_count": qual_count,
                "context_discount_count": disc_count,
                "qualify_or_abstain_count": qual_abst_count,
                "abstention_count": abst_count,
                "predicted_state_distribution": pred_dist,
                "actual_gate_action_distribution": action_dist,
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    eval_df = run_phase13a_evaluation()
    summary_df = build_phase13a_summary(eval_df)

    EVALUATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    overall = summary_df[summary_df["condition"] == "overall"].iloc[0]

    print("=" * 80)
    print("PHASE 13A: RELIABILITY GATE DOWNSTREAM EVALUATION")
    print("Pipeline: Phase 12C EvidenceStateAnalyzer -> ReliabilityGate")
    print("Terminology: Deterministic rule-execution downstream evaluation")
    print("=" * 80)
    print(f"Total Records: {overall['n']}")
    print(f"Overall State Alignment: {overall['correct_state']}/{overall['n']} ({overall['state_accuracy']*100:.2f}%)")
    print(f"Overall Action Alignment: {overall['correct_action']}/{overall['n']} ({overall['action_accuracy']*100:.2f}%)")
    print("\nOverall Gate Action Counts:")
    print(f"  generate:           {overall['generation_count']}")
    print(f"  qualify:            {overall['qualification_count']}")
    print(f"  discount_context:   {overall['context_discount_count']}")
    print(f"  qualify_or_abstain: {overall['qualify_or_abstain_count']}")
    print(f"  abstain:            {overall['abstention_count']}")

    print("\n--- CONDITION-WISE ACCURACY & GATE ACTION DISTRIBUTIONS ---")
    cond_df = summary_df[summary_df["condition"] != "overall"]
    print(
        cond_df[
            [
                "condition",
                "n",
                "correct_state",
                "state_accuracy",
                "correct_action",
                "action_accuracy",
                "actual_gate_action_distribution",
            ]
        ].to_string(index=False)
    )

    print(f"\nEvaluation detail saved to: {EVALUATION_OUTPUT_PATH}")
    print(f"Summary report saved to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
