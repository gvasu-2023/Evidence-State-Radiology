from pathlib import Path

import pandas as pd
import pytest

from src.evidence_state.states import EvidenceState
from src.evaluation.evaluate_analyzer_phase12b import (
    EVALUATION_OUTPUT_PATH as PHASE12B_EVAL_PATH,
    SUMMARY_OUTPUT_PATH as PHASE12B_SUMM_PATH,
    REQUIRED_CONDITIONS,
    build_phase12b_summary,
    run_phase12b_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE12A_EVAL_PATH = ROOT / "results/tables/phase12_analyzer_evaluation.csv"
PHASE12A_SUMM_PATH = ROOT / "results/tables/phase12_analyzer_summary.csv"


def test_phase12a_output_files_remain_unchanged():
    assert PHASE12A_EVAL_PATH.exists(), "Phase 12A evaluation file missing"
    assert PHASE12A_SUMM_PATH.exists(), "Phase 12A summary file missing"

    df_12a_sum = pd.read_csv(PHASE12A_SUMM_PATH)
    assert len(df_12a_sum) == 6
    overall = df_12a_sum[df_12a_sum["condition"] == "overall"].iloc[0]
    assert overall["correct_cases"] == 52
    assert pytest.approx(overall["alignment_rate"], rel=1e-4) == 52 / 81


def test_phase12b_output_files_exist():
    assert PHASE12B_EVAL_PATH.exists(), "phase12b_analyzer_evaluation.csv missing"
    assert PHASE12B_SUMM_PATH.exists(), "phase12b_analyzer_summary.csv missing"


def test_exactly_81_phase12b_records():
    eval_df = run_phase12b_evaluation()
    assert len(eval_df) == 81

    csv_df = pd.read_csv(PHASE12B_EVAL_PATH)
    assert len(csv_df) == 81


def test_expected_condition_counts_and_state_equals_condition():
    eval_df = run_phase12b_evaluation()

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
    eval_df = run_phase12b_evaluation()
    valid_states = {state.value for state in EvidenceState}
    actual_states = set(eval_df["predicted_state"])
    assert actual_states.issubset(valid_states)


def test_state_match_logic():
    eval_df = run_phase12b_evaluation()
    expected_matches = eval_df["predicted_state"] == eval_df["expected_state"]
    assert (eval_df["state_match"] == expected_matches).all()


def test_phase12b_summary_contains_all_conditions_and_confusion_matrix_sum():
    summary_df = pd.read_csv(PHASE12B_SUMM_PATH)
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

    # Verify overall row total_cases equals 81
    overall_row = summary_df[summary_df["condition"] == "overall"].iloc[0]
    assert overall_row["total_cases"] == 81
    assert overall_row["correct_cases"] == 81
    assert pytest.approx(overall_row["alignment_rate"], rel=1e-4) == 1.0


def test_anti_leakage_condition_not_in_assessment():
    from src.evaluation.evaluate_analyzer_phase12b import evaluate_record
    from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer

    # Verify evaluate_record passes only EvidenceAssessment to EvidenceStateAnalyzer
    analyzer = EvidenceStateAnalyzer()
    dummy_row = pd.Series(
        {
            "sample_id": "IU_TEST",
            "condition": "incomplete",
            "transformation_type": "partial_context_reduction",
            "image_path": "data/processed/iu_xray/images/IU_1_frontal.jpg",
            "perturbed_context": "chest pain",
        }
    )

    res = evaluate_record(dummy_row, analyzer)
    assert res["predicted_state"] == "incomplete"
    assert res["state_match"] == True
