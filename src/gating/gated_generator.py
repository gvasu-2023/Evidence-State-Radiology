"""
Evidence-State Gated CXR Report Generator.

Applies EvidenceStateAnalyzer and ReliabilityGate to determine the generation
policy for each perturbation record, reusing exact input-matched baseline outputs
from baseline_generation_results.csv to avoid redundant CPU inference.

Terminology: paired reference-grounded evaluation of the current prototype's
gated generation policy.
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
BASELINE_RESULTS_PATH = ROOT / "results/tables/baseline_generation_results.csv"
OUTPUT_PATH = ROOT / "results/tables/gated_generation_results.csv"


class GatedGenerator:
    """
    Gated report generator wrapper around baseline outputs and reliability gate policy.
    """

    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df.copy()
        self.baseline_lookup = {
            (str(row["sample_id"]), str(row["condition"])): str(row["generated_report"])
            for _, row in self.baseline_df.iterrows()
        }
        self.analyzer = EvidenceStateAnalyzer()
        self.gate = ReliabilityGate()

    def process_record(self, row: pd.Series) -> dict:
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

        predicted_state_enum = self.analyzer.classify(assessment)
        predicted_evidence_state = predicted_state_enum.value

        gate_decision = self.gate.decide(predicted_state_enum)
        gate_action = gate_decision.action
        gate_reason = gate_decision.reason

        output_policy = gate_action
        was_model_inference_run = False
        baseline_pair_identifier = f"{sample_id}_{experimental_condition}"

        # Determine source baseline condition for exact input matching
        if gate_action == "generate":
            source_baseline_condition = experimental_condition
        elif gate_action == "discount_context":
            # Strip context -> use image-only baseline output (stored under insufficient condition)
            source_baseline_condition = "insufficient"
        elif gate_action == "qualify_or_abstain":
            source_baseline_condition = experimental_condition
        elif gate_action == "qualify":
            # Under prototype ES=1.0, insufficient condition maps to incomplete/qualify. Context is empty.
            source_baseline_condition = "insufficient"
        elif gate_action == "abstain":
            source_baseline_condition = "none"
        else:
            source_baseline_condition = experimental_condition

        # Retrieve report body from baseline lookup
        if source_baseline_condition == "none":
            baseline_report_body = ""
        else:
            key = (sample_id, source_baseline_condition)
            if key not in self.baseline_lookup:
                raise ValueError(
                    f"Baseline report key missing: {key} for sample {sample_id}, condition {experimental_condition}"
                )
            baseline_report_body = self.baseline_lookup[key]

        # Format gated output report according to policy
        if gate_action == "generate":
            generated_report = baseline_report_body
        elif gate_action == "discount_context":
            generated_report = baseline_report_body
        elif gate_action == "qualify":
            generated_report = f"[QUALIFIED: Incomplete clinical context] {baseline_report_body}"
        elif gate_action == "qualify_or_abstain":
            generated_report = f"[WARNING: Clinical context conflict detected] {baseline_report_body}"
        elif gate_action == "abstain":
            generated_report = "[ABSTAIN: Insufficient evidence for report generation]"
        else:
            generated_report = baseline_report_body

        return {
            "sample_id": sample_id,
            "experimental_condition": experimental_condition,
            "predicted_evidence_state": predicted_evidence_state,
            "gate_action": gate_action,
            "gate_reason": gate_reason,
            "output_policy": output_policy,
            "source_baseline_condition": source_baseline_condition,
            "generated_report": generated_report,
            "was_model_inference_run": was_model_inference_run,
            "baseline_pair_identifier": baseline_pair_identifier,
        }
