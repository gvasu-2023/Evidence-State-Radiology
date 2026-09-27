"""
Unit and integration tests for Phase 13A Reliability Gate evaluation.

Verifies:
1. Exactly 81 records evaluated with no duplicate (sample_id, condition) pairs.
2. All five controlled experimental conditions represented.
3. Predicted states and actual gate actions conform to valid enum/action sets.
4. Expected condition-to-action mapping correctness.
5. Per-condition analyzer -> gate accuracy (17/17 sufficient, 20/20 irrelevant, 15/15 conflicting).
6. Measurement of incomplete (0/9) and insufficient (20/20) behavior.
7. Output tables exist with required columns and row counts.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.evaluation.evaluate_phase13a_gate import (
    run_phase13a_evaluation,
    build_phase13a_summary,
    EXPECTED_ACTION_MAPPING,
    EVALUATION_OUTPUT_PATH,
    SUMMARY_OUTPUT_PATH,
)
from src.evidence_state.states import EvidenceState


VALID_EVIDENCE_STATES = {state.value for state in EvidenceState}
VALID_GATE_ACTIONS = {"generate", "qualify", "discount_context", "qualify_or_abstain", "abstain"}
REQUIRED_CONDITIONS = {"sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"}


def test_phase13a_evaluation_row_count_and_uniqueness() -> None:
    eval_df = run_phase13a_evaluation()
    assert len(eval_df) == 81, f"Expected 81 records, got {len(eval_df)}"

    # Verify all 5 conditions are represented
    conditions_found = set(eval_df["condition"].tolist())
    assert conditions_found == REQUIRED_CONDITIONS, f"Mismatch in conditions: {conditions_found}"

    # Verify no duplicate sample_id + condition pairs
    duplicates = eval_df.duplicated(subset=["sample_id", "condition"])
    assert not duplicates.any(), f"Found {duplicates.sum()} duplicate sample_id + condition rows"


def test_phase13a_required_columns() -> None:
    eval_df = run_phase13a_evaluation()
    required_columns = [
        "sample_id",
        "condition",
        "predicted_state",
        "expected_state",
        "state_correct",
        "actual_gate_action",
        "expected_gate_action",
        "action_correct",
        "gate_reason",
        "transformation_type",
        "image_available",
        "context_available",
        "context_relevant",
        "context_consistent",
        "context_complete",
    ]
    for col in required_columns:
        assert col in eval_df.columns, f"Missing required column: {col}"


def test_phase13a_valid_states_and_actions() -> None:
    eval_df = run_phase13a_evaluation()
    for _, row in eval_df.iterrows():
        assert row["predicted_state"] in VALID_EVIDENCE_STATES, f"Invalid state: {row['predicted_state']}"
        assert row["actual_gate_action"] in VALID_GATE_ACTIONS, f"Invalid gate action: {row['actual_gate_action']}"
        assert row["expected_gate_action"] in VALID_GATE_ACTIONS, f"Invalid expected action: {row['expected_gate_action']}"


def test_expected_action_mapping() -> None:
    assert EXPECTED_ACTION_MAPPING == {
        "sufficient": "generate",
        "incomplete": "qualify",
        "irrelevant": "discount_context",
        "conflicting": "qualify_or_abstain",
        "insufficient": "abstain",
    }


def test_phase13a_sufficient_condition_alignment() -> None:
    eval_df = run_phase13a_evaluation()
    suff_df = eval_df[eval_df["condition"] == "sufficient"]
    assert len(suff_df) == 17, f"Expected 17 sufficient cases, got {len(suff_df)}"
    assert (suff_df["state_correct"] == True).all()
    assert (suff_df["action_correct"] == True).all()
    assert (suff_df["actual_gate_action"] == "generate").all()


def test_phase13a_irrelevant_condition_alignment() -> None:
    eval_df = run_phase13a_evaluation()
    irr_df = eval_df[eval_df["condition"] == "irrelevant"]
    assert len(irr_df) == 20, f"Expected 20 irrelevant cases, got {len(irr_df)}"
    assert (irr_df["state_correct"] == True).all()
    assert (irr_df["action_correct"] == True).all()
    assert (irr_df["actual_gate_action"] == "discount_context").all()


def test_phase13a_conflicting_condition_alignment() -> None:
    eval_df = run_phase13a_evaluation()
    conf_df = eval_df[eval_df["condition"] == "conflicting"]
    assert len(conf_df) == 15, f"Expected 15 conflicting cases, got {len(conf_df)}"
    assert (conf_df["state_correct"] == True).all()
    assert (conf_df["action_correct"] == True).all()
    assert (conf_df["actual_gate_action"] == "qualify_or_abstain").all()


def test_phase13a_insufficient_condition_alignment() -> None:
    eval_df = run_phase13a_evaluation()
    insuff_df = eval_df[eval_df["condition"] == "insufficient"]
    assert len(insuff_df) == 20, f"Expected 20 insufficient cases, got {len(insuff_df)}"
    assert (insuff_df["state_correct"] == True).all()
    assert (insuff_df["action_correct"] == True).all()
    assert (insuff_df["actual_gate_action"] == "abstain").all()


def test_phase13a_incomplete_condition_measured_alignment() -> None:
    eval_df = run_phase13a_evaluation()
    inc_df = eval_df[eval_df["condition"] == "incomplete"]
    assert len(inc_df) == 9, f"Expected 9 incomplete cases, got {len(inc_df)}"
    # Measured result under Phase 12C parser: non-fragmented strings are parsed as complete
    assert (inc_df["state_correct"] == False).all()
    assert (inc_df["action_correct"] == False).all()
    assert (inc_df["actual_gate_action"] == "generate").all()


def test_phase13a_summary_structure() -> None:
    eval_df = run_phase13a_evaluation()
    summary_df = build_phase13a_summary(eval_df)
    assert len(summary_df) == 6, f"Expected 6 rows in summary (overall + 5 conditions), got {len(summary_df)}"

    overall = summary_df[summary_df["condition"] == "overall"].iloc[0]
    assert overall["n"] == 81
    assert overall["correct_state"] == 72
    assert overall["correct_action"] == 72
    assert overall["generation_count"] == 26
    assert overall["context_discount_count"] == 20
    assert overall["qualify_or_abstain_count"] == 15
    assert overall["abstention_count"] == 20
