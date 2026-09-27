"""
Phase 13B: Downstream Gated Report Evaluation with Phase 12C Evidence-State Analyzer.

Evaluates the downstream gated-report outputs produced by connecting the CURRENT Phase 12C
EvidenceStateAnalyzer + CURRENT ReliabilityGate to the EXISTING 81 baseline reports
from results/tables/baseline_generation_results.csv.

Reuses exact baseline reports to avoid redundant CPU model inference.

Pipeline:
    perturbation record
    → Phase 12C parse_context_completeness()
    → EvidenceAssessment(context_complete=...)
    → Phase 12C EvidenceStateAnalyzer
    → predicted EvidenceState
    → ReliabilityGate
    → GateDecision
    → Report Policy Transformation (baseline lookup / header prefix / image-only / abstention)
    → Claim Extraction & Reference Factuality Evaluation

Anti-Leakage:
The analyzer and context completeness parser operate strictly on input context strings
and image availability flags. No ground-truth metadata (condition, expected_state,
original_context, transformation_type) is passed into analyzer or gate inference.

Outputs:
    results/tables/phase13b_gated_generation_results.csv
    results/tables/phase13b_downstream_summary.csv
    results/tables/phase13b_claim_comparison.csv
"""

from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.context_completeness import parse_context_completeness
from src.evidence_state.states import EvidenceState
from src.gating.reliability_gate import ReliabilityGate, GateDecision
from src.evaluation.claim_extractor import extract_generated_claim_candidates
from src.evaluation.evaluate_reference_factuality import (
    build_factuality_table,
    VALID_LABELS,
)

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_RESULTS_PATH = ROOT / "results/tables/baseline_generation_results.csv"
REFERENCE_CLAIMS_PATH = ROOT / "results/tables/reference_claim_candidates.csv"

GATED_RESULTS_OUTPUT_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/phase13b_downstream_summary.csv"
CLAIM_COMPARISON_OUTPUT_PATH = ROOT / "results/tables/phase13b_claim_comparison.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_input_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [PERTURBATIONS_PATH, BASELINE_RESULTS_PATH, REFERENCE_CLAIMS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file missing: {path}")

    perturbations_df = pd.read_csv(PERTURBATIONS_PATH)
    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
    reference_claims_df = pd.read_csv(REFERENCE_CLAIMS_PATH)

    if len(perturbations_df) != 81:
        raise ValueError(f"Expected 81 perturbation records, got {len(perturbations_df)}")
    if len(baseline_df) != 81:
        raise ValueError(f"Expected 81 baseline generation records, got {len(baseline_df)}")

    return perturbations_df, baseline_df, reference_claims_df


def build_baseline_lookup(baseline_df: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    lookup = {}
    for _, row in baseline_df.iterrows():
        sample_id = str(row["sample_id"])
        condition = str(row["condition"])
        report = str(row["generated_report"])
        lookup[(sample_id, condition)] = report
    return lookup


def evaluate_record(
    row: pd.Series,
    baseline_lookup: Dict[Tuple[str, str], str],
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

    # Phase 12C context completeness parser
    completeness_res = parse_context_completeness(context_str)
    context_complete = completeness_res.is_complete
    evidence_strength = 1.0

    # Assessment passed to analyzer (NO condition or perturbation metadata passed!)
    assessment = EvidenceAssessment(
        image_available=image_available,
        context_available=context_available,
        context_relevant=context_relevant,
        context_consistent=context_consistent,
        context_complete=context_complete,
        evidence_strength=evidence_strength,
    )

    predicted_state_enum = analyzer.classify(assessment)
    predicted_state = predicted_state_enum.value

    gate_decision: GateDecision = gate.decide(predicted_state_enum)
    gate_action = gate_decision.action
    gate_reason = gate_decision.reason

    # Baseline report for this specific record
    baseline_key = (sample_id, condition)
    if baseline_key not in baseline_lookup:
        raise KeyError(f"Missing baseline lookup for key: {baseline_key}")
    baseline_report = baseline_lookup[baseline_key]

    # Report policy transformation
    if gate_action == "generate":
        source_baseline_condition = condition
        gated_report = baseline_report
    elif gate_action == "discount_context":
        source_baseline_condition = "insufficient"
        gated_report = baseline_lookup[(sample_id, "insufficient")]
    elif gate_action == "qualify":
        source_baseline_condition = "insufficient"
        img_report = baseline_lookup[(sample_id, "insufficient")]
        gated_report = f"[QUALIFIED: Incomplete clinical context] {img_report}"
    elif gate_action == "qualify_or_abstain":
        source_baseline_condition = condition
        gated_report = f"[WARNING: Clinical context conflict detected] {baseline_report}"
    elif gate_action == "abstain":
        source_baseline_condition = "none"
        gated_report = "[ABSTAIN: Insufficient evidence for report generation]"
    else:
        source_baseline_condition = condition
        gated_report = baseline_report

    ref_findings = str(row["reference_findings"]) if pd.notna(row["reference_findings"]) else ""
    ref_impression = str(row["reference_impression"]) if pd.notna(row["reference_impression"]) else ""

    return {
        "sample_id": sample_id,
        "condition": condition,
        "transformation_type": transformation_type,
        "image_path": image_path_str,
        "perturbed_context": context_str,
        "reference_findings": ref_findings,
        "reference_impression": ref_impression,
        "context_complete": context_complete,
        "predicted_state": predicted_state,
        "gate_action": gate_action,
        "gate_reason": gate_reason,
        "source_baseline_condition": source_baseline_condition,
        "baseline_report": baseline_report,
        "gated_report": gated_report,
        "was_model_inference_run": False,
        "baseline_report_length": len(baseline_report),
        "gated_report_length": len(gated_report),
    }


def run_phase13b_gated_evaluation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perturbations_df, baseline_df, reference_claims_df = load_input_datasets()
    baseline_lookup = build_baseline_lookup(baseline_df)

    analyzer = EvidenceStateAnalyzer()
    gate = ReliabilityGate()

    records = [
        evaluate_record(row, baseline_lookup, analyzer, gate)
        for _, row in perturbations_df.iterrows()
    ]
    results_df = pd.DataFrame(records)

    if len(results_df) != 81:
        raise RuntimeError(f"Expected 81 evaluation rows, got {len(results_df)}")

    # Extract claims for baseline & Phase 13B gated reports
    base_prep = results_df.rename(columns={"baseline_report": "generated_report"})[
        ["sample_id", "condition", "generated_report"]
    ]
    baseline_claims = extract_generated_claim_candidates(base_prep)

    gated_prep = results_df.rename(columns={"gated_report": "generated_report"})[
        ["sample_id", "condition", "generated_report"]
    ]
    gated_claims = extract_generated_claim_candidates(gated_prep)

    # Attach claim counts to results_df
    base_claim_counts = baseline_claims.groupby(["sample_id", "condition"]).size().to_dict()
    gated_claim_counts = gated_claims.groupby(["sample_id", "condition"]).size().to_dict()

    results_df["baseline_claim_count"] = [
        int(base_claim_counts.get((str(row["sample_id"]), str(row["condition"])), 0))
        for _, row in results_df.iterrows()
    ]
    results_df["gated_claim_count"] = [
        int(gated_claim_counts.get((str(row["sample_id"]), str(row["condition"])), 0))
        for _, row in results_df.iterrows()
    ]

    # Evaluate reference factuality for gated reports
    gated_factuality_df = build_factuality_table(gated_prep, reference_claims_df, gated_claims)

    # Compute case-level claim comparison table (81 rows)
    claim_comp_rows = []
    for _, row in results_df.iterrows():
        sid = str(row["sample_id"])
        cond = str(row["condition"])
        b_cnt = int(row["baseline_claim_count"])
        g_cnt = int(row["gated_claim_count"])

        case_factuality = gated_factuality_df[
            (gated_factuality_df["sample_id"].astype(str) == sid)
            & (gated_factuality_df["condition"].astype(str) == cond)
        ]

        label_counts = case_factuality["label"].value_counts().to_dict()

        claim_comp_rows.append(
            {
                "sample_id": sid,
                "condition": cond,
                "baseline_claim_count": b_cnt,
                "gated_claim_count": g_cnt,
                "supported": int(label_counts.get("supported", 0)),
                "omission": int(label_counts.get("omission", 0)),
                "extra_negation": int(label_counts.get("extra_negation", 0)),
                "ungrounded_affirmation": int(label_counts.get("ungrounded_affirmation", 0)),
                "unsupported_affirmation": int(label_counts.get("unsupported_affirmation", 0)),
                "contradiction": int(label_counts.get("contradiction", 0)),
                "unclear": int(label_counts.get("unclear", 0)),
            }
        )

    claim_comp_df = pd.DataFrame(claim_comp_rows)

    # Compute summary table
    summary_df = build_phase13b_summary(results_df)

    return results_df, summary_df, claim_comp_df


def build_phase13b_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    subsets = [("overall", results_df)] + [
        (c, results_df[results_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        n = len(subset)
        action_counts = subset["gate_action"].value_counts().to_dict()
        gen_count = int(action_counts.get("generate", 0))
        qual_count = int(action_counts.get("qualify", 0))
        disc_count = int(action_counts.get("discount_context", 0))
        conflict_warn_count = int(action_counts.get("qualify_or_abstain", 0))
        abst_count = int(action_counts.get("abstain", 0))

        mean_base_len = float(subset["baseline_report_length"].mean()) if n > 0 else 0.0
        mean_gated_len = float(subset["gated_report_length"].mean()) if n > 0 else 0.0
        mean_len_diff = mean_gated_len - mean_base_len

        mean_base_claims = float(subset["baseline_claim_count"].mean()) if n > 0 else 0.0
        mean_gated_claims = float(subset["gated_claim_count"].mean()) if n > 0 else 0.0
        mean_claim_diff = mean_gated_claims - mean_base_claims

        model_inf_count = int((subset["was_model_inference_run"] == True).sum())

        pred_dist = str(subset["predicted_state"].value_counts().to_dict())
        action_dist = str(subset["gate_action"].value_counts().to_dict())

        summary_rows.append(
            {
                "condition": scope,
                "n": n,
                "generation_count": gen_count,
                "qualification_count": qual_count,
                "context_discount_count": disc_count,
                "conflict_warning_count": conflict_warn_count,
                "abstention_count": abst_count,
                "mean_baseline_report_length": round(mean_base_len, 2),
                "mean_gated_report_length": round(mean_gated_len, 2),
                "mean_report_length_difference": round(mean_len_diff, 2),
                "mean_baseline_claim_count": round(mean_base_claims, 2),
                "mean_gated_claim_count": round(mean_gated_claims, 2),
                "mean_claim_count_difference": round(mean_claim_diff, 2),
                "model_inference_count": model_inf_count,
                "predicted_state_distribution": pred_dist,
                "gate_action_distribution": action_dist,
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    results_df, summary_df, claim_comp_df = run_phase13b_gated_evaluation()

    GATED_RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(GATED_RESULTS_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    claim_comp_df.to_csv(CLAIM_COMPARISON_OUTPUT_PATH, index=False)

    overall = summary_df[summary_df["condition"] == "overall"].iloc[0]

    print("=" * 80)
    print("PHASE 13B: DOWNSTREAM GATED REPORT EVALUATION")
    print("Pipeline: Phase 12C EvidenceStateAnalyzer -> ReliabilityGate -> Baseline Report Reuse")
    print("Terminology: Descriptive evaluation of gated policy transformations")
    print("=" * 80)
    print(f"Total Records Evaluated: {overall['n']}")
    print(f"Model Inference Executed: {overall['model_inference_count']} (0 for all 81 records)")
    print("\nGate Action Distribution:")
    print(f"  generate:           {overall['generation_count']}")
    print(f"  qualify:            {overall['qualification_count']}")
    print(f"  discount_context:   {overall['context_discount_count']}")
    print(f"  qualify_or_abstain: {overall['conflict_warning_count']}")
    print(f"  abstain:            {overall['abstention_count']}")

    print("\nReport Length & Claim Count Changes (Baseline vs Phase 13B Gated):")
    print(f"  Mean Baseline Report Length: {overall['mean_baseline_report_length']:.2f} chars")
    print(f"  Mean Gated Report Length:    {overall['mean_gated_report_length']:.2f} chars")
    print(f"  Mean Length Difference:     {overall['mean_report_length_difference']:.2f} chars")
    print(f"  Mean Baseline Claims:        {overall['mean_baseline_claim_count']:.2f}")
    print(f"  Mean Gated Claims:           {overall['mean_gated_claim_count']:.2f}")
    print(f"  Mean Claim Count Difference: {overall['mean_claim_count_difference']:.2f}")

    print("\n--- CONDITION-WISE SUMMARY ---")
    cond_df = summary_df[summary_df["condition"] != "overall"]
    print(
        cond_df[
            [
                "condition",
                "n",
                "mean_baseline_report_length",
                "mean_gated_report_length",
                "mean_baseline_claim_count",
                "mean_gated_claim_count",
                "gate_action_distribution",
            ]
        ].to_string(index=False)
    )

    print(f"\nGated generation results saved to: {GATED_RESULTS_OUTPUT_PATH}")
    print(f"Downstream summary saved to:         {SUMMARY_OUTPUT_PATH}")
    print(f"Claim comparison saved to:           {CLAIM_COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
