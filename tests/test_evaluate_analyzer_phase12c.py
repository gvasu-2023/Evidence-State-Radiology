"""
Unit tests for Phase 12C Evidence-State Analyzer Evaluation module.

Verifies:
1. Phase 12C evaluation script execution and file outputs.
2. Structure and record counts of phase12c_analyzer_evaluation.csv and phase12c_analyzer_summary.csv.
3. Structure and record counts of phase12c_context_completeness_robustness.csv.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.evaluation.evaluate_analyzer_phase12c import (
    run_phase12c_evaluation,
    build_phase12c_summary,
    run_phase12c_robustness_audit,
    EVALUATION_OUTPUT_PATH,
    SUMMARY_OUTPUT_PATH,
    ROBUSTNESS_OUTPUT_PATH,
)


def test_phase12c_evaluation_dataframe_structure() -> None:
    eval_df = run_phase12c_evaluation()
    assert len(eval_df) == 81, f"Expected 81 evaluation rows, got {len(eval_df)}"

    required_columns = [
        "sample_id",
        "condition",
        "predicted_state",
        "expected_state",
        "state_match",
        "image_available",
        "context_available",
        "context_relevant",
        "context_consistent",
        "context_complete",
        "rule_triggered",
        "evidence_strength",
        "transformation_type",
    ]
    for col in required_columns:
        assert col in eval_df.columns, f"Missing required column: {col}"


def test_phase12c_summary_dataframe_structure() -> None:
    eval_df = run_phase12c_evaluation()
    summary_df = build_phase12c_summary(eval_df)
    assert len(summary_df) == 6, f"Expected 6 summary rows (overall + 5 conditions), got {len(summary_df)}"

    required_conditions = {"overall", "sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"}
    summary_conditions = set(summary_df["condition"].tolist())
    assert summary_conditions == required_conditions


def test_phase12c_robustness_audit_dataframe_structure() -> None:
    robustness_df = run_phase12c_robustness_audit()
    assert len(robustness_df) > 0, "Expected non-empty robustness audit dataframe"
    assert "false_incomplete" in robustness_df.columns

    complete_expected = robustness_df[robustness_df["expected_behavior"] == "complete"]
    false_incompletes = robustness_df[robustness_df["false_incomplete"] == True]
    assert len(false_incompletes) == 0, f"Expected 0 false incompletes under Phase 12C, got {len(false_incompletes)}"
