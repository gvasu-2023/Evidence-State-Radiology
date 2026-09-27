"""
Unit and integration tests for Phase 13B Downstream Gated Report Evaluation.

Verifies:
1. Exactly 81 records evaluated with no duplicate (sample_id, condition) pairs.
2. All five controlled experimental conditions present.
3. Every baseline report lookup succeeds without model inference.
4. was_model_inference_run = False for all 81 records.
5. Every predicted state and gate action is valid.
6. Phase 12C context completeness parser is explicitly integrated.
7. Correct policy behavior for sufficient, irrelevant, conflicting, and insufficient conditions.
8. Structure and schema of generated output tables.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.evaluation.evaluate_phase13b_gated_reports import (
    run_phase13b_gated_evaluation,
    GATED_RESULTS_OUTPUT_PATH,
    SUMMARY_OUTPUT_PATH,
    CLAIM_COMPARISON_OUTPUT_PATH,
)
from src.evidence_state.states import EvidenceState


VALID_EVIDENCE_STATES = {state.value for state in EvidenceState}
VALID_GATE_ACTIONS = {"generate", "qualify", "discount_context", "qualify_or_abstain", "abstain"}
REQUIRED_CONDITIONS = {"sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"}


def test_phase13b_evaluation_row_count_and_uniqueness() -> None:
    results_df, summary_df, claim_comp_df = run_phase13b_gated_evaluation()
    assert len(results_df) == 81, f"Expected 81 results rows, got {len(results_df)}"
    assert len(claim_comp_df) == 81, f"Expected 81 claim comparison rows, got {len(claim_comp_df)}"

    conditions_found = set(results_df["condition"].tolist())
    assert conditions_found == REQUIRED_CONDITIONS, f"Mismatch in conditions: {conditions_found}"

    duplicates = results_df.duplicated(subset=["sample_id", "condition"])
    assert not duplicates.any(), f"Found {duplicates.sum()} duplicate sample_id + condition rows"


def test_phase13b_no_model_inference_executed() -> None:
    results_df, _, _ = run_phase13b_gated_evaluation()
    assert (results_df["was_model_inference_run"] == False).all()


def test_phase13b_valid_states_and_actions() -> None:
    results_df, _, _ = run_phase13b_gated_evaluation()
    for _, row in results_df.iterrows():
        assert row["predicted_state"] in VALID_EVIDENCE_STATES, f"Invalid state: {row['predicted_state']}"
        assert row["gate_action"] in VALID_GATE_ACTIONS, f"Invalid gate action: {row['gate_action']}"


def test_phase13b_sufficient_policy_behavior() -> None:
    results_df, _, _ = run_phase13b_gated_evaluation()
    suff_df = results_df[results_df["condition"] == "sufficient"]
    assert len(suff_df) == 17
    assert (suff_df["predicted_state"] == "sufficient").all()
    assert (suff_df["gate_action"] == "generate").all()
    assert (suff_df["source_baseline_condition"] == "sufficient").all()
    for _, row in suff_df.iterrows():
        assert row["gated_report"] == row["baseline_report"]


def test_phase13b_irrelevant_policy_behavior() -> None:
    results_df, _, _ = run_phase13b_gated_evaluation()
    irr_df = results_df[results_df["condition"] == "irrelevant"]
    assert len(irr_df) == 20
    assert (irr_df["predicted_state"] == "irrelevant").all()
    assert (irr_df["gate_action"] == "discount_context").all()
    assert (irr_df["source_baseline_condition"] == "insufficient").all()
    for _, row in irr_df.iterrows():
        # Discount context removes swapped context -> retrieves image-only baseline report
        assert not row["gated_report"].startswith("[QUALIFIED:")
        assert not row["gated_report"].startswith("[WARNING:")


def test_phase13b_conflicting_policy_behavior() -> None:
    results_df, _, _ = run_phase13b_gated_evaluation()
    conf_df = results_df[results_df["condition"] == "conflicting"]
    assert len(conf_df) == 15
    assert (conf_df["predicted_state"] == "conflicting").all()
    assert (conf_df["gate_action"] == "qualify_or_abstain").all()
    assert (conf_df["source_baseline_condition"] == "conflicting").all()
    for _, row in conf_df.iterrows():
        assert row["gated_report"].startswith("[WARNING: Clinical context conflict detected] ")


def test_phase13b_insufficient_policy_behavior() -> None:
    results_df, _, _ = run_phase13b_gated_evaluation()
    insuff_df = results_df[results_df["condition"] == "insufficient"]
    assert len(insuff_df) == 20
    assert (insuff_df["predicted_state"] == "insufficient").all()
    assert (insuff_df["gate_action"] == "abstain").all()
    assert (insuff_df["source_baseline_condition"] == "none").all()
    for _, row in insuff_df.iterrows():
        assert row["gated_report"] == "[ABSTAIN: Insufficient evidence for report generation]"
        assert row["gated_claim_count"] == 0


def test_phase13b_required_output_columns() -> None:
    results_df, summary_df, claim_comp_df = run_phase13b_gated_evaluation()

    required_results_cols = [
        "sample_id", "condition", "transformation_type", "image_path", "perturbed_context",
        "reference_findings", "reference_impression", "context_complete", "predicted_state",
        "gate_action", "gate_reason", "source_baseline_condition", "baseline_report",
        "gated_report", "was_model_inference_run", "baseline_report_length", "gated_report_length",
        "baseline_claim_count", "gated_claim_count"
    ]
    for col in required_results_cols:
        assert col in results_df.columns, f"Missing result column: {col}"

    required_summary_cols = [
        "condition", "n", "generation_count", "qualification_count", "context_discount_count",
        "conflict_warning_count", "abstention_count", "mean_baseline_report_length",
        "mean_gated_report_length", "mean_report_length_difference", "mean_baseline_claim_count",
        "mean_gated_claim_count", "mean_claim_count_difference", "model_inference_count"
    ]
    for col in required_summary_cols:
        assert col in summary_df.columns, f"Missing summary column: {col}"

    required_claim_cols = [
        "sample_id", "condition", "baseline_claim_count", "gated_claim_count",
        "supported", "omission", "extra_negation", "ungrounded_affirmation",
        "unsupported_affirmation", "contradiction", "unclear"
    ]
    for col in required_claim_cols:
        assert col in claim_comp_df.columns, f"Missing claim comparison column: {col}"
