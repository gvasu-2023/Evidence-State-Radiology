from pathlib import Path

import pandas as pd
import pytest

from src.evidence_state.states import EvidenceState
from src.evaluation.evaluate_analyzer import (
    EVALUATION_OUTPUT_PATH,
    SUMMARY_OUTPUT_PATH,
    REQUIRED_CONDITIONS,
    build_analyzer_summary,
    run_analyzer_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase12_output_files_exist():
    assert EVALUATION_OUTPUT_PATH.exists(), "phase12_analyzer_evaluation.csv missing"
    assert SUMMARY_OUTPUT_PATH.exists(), "phase12_analyzer_summary.csv missing"


def test_exactly_81_analyzer_evaluations():
    eval_df = run_analyzer_evaluation()
    assert len(eval_df) == 81

    csv_df = pd.read_csv(EVALUATION_OUTPUT_PATH)
    assert len(csv_df) == 81


def test_expected_condition_counts_and_state_equals_condition():
    eval_df = run_analyzer_evaluation()

    counts = eval_df["condition"].value_counts().to_dict()
    expected_counts = {
        "sufficient": 17,
        "incomplete": 9,
        "irrelevant": 20,
        "conflicting": 15,
        "insufficient": 20,
    }
    assert counts == expected_counts

    # Expected state equals condition label
    assert (eval_df["expected_state"] == eval_df["condition"]).all()


def test_predicted_states_belong_to_enum():
    eval_df = run_analyzer_evaluation()
    valid_states = {state.value for state in EvidenceState}
    actual_states = set(eval_df["predicted_state"])
    assert actual_states.issubset(valid_states)


def test_state_match_logic():
    eval_df = run_analyzer_evaluation()
    expected_matches = eval_df["predicted_state"] == eval_df["expected_state"]
    assert (eval_df["state_match"] == expected_matches).all()


def test_summary_contains_all_conditions_and_confusion_matrix_sum():
    summary_df = pd.read_csv(SUMMARY_OUTPUT_PATH)
    assert len(summary_df) == 6  # overall + 5 conditions

    cond_subset = summary_df[summary_df["condition"] != "overall"]
    assert set(cond_subset["condition"]) == set(REQUIRED_CONDITIONS)

    # Verify confusion matrix columns sum to 81 across conditions
    pred_cols = [
        "pred_sufficient",
        "pred_incomplete",
        "pred_irrelevant",
        "pred_conflicting",
        "pred_insufficient",
    ]
    cm_sum = cond_subset[pred_cols].sum().sum()
    assert cm_sum == 81

    # Verify overall row total_cases equals 81 and correct_cases equals 52
    overall_row = summary_df[summary_df["condition"] == "overall"].iloc[0]
    assert overall_row["total_cases"] == 81
    assert overall_row["correct_cases"] == 52
    assert pytest.approx(overall_row["alignment_rate"], rel=1e-4) == 52 / 81
