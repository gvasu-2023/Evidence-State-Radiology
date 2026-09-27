from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.evaluate_reliability_gate import (
    EXPECTED_CONDITIONS,
    EXPECTED_GATE_ACTIONS,
    OUTPUT_PATH,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reliability_gate_evaluation_file_exists():
    assert OUTPUT_PATH.exists(), "reliability_gate_evaluation.csv missing"
    df = pd.read_csv(OUTPUT_PATH)
    assert len(df) == 81


def test_all_81_records_evaluated_and_unique_pairs():
    eval_df = run_evaluation()
    assert len(eval_df) == 81

    pairs = list(zip(eval_df["sample_id"], eval_df["experimental_condition"]))
    assert len(pairs) == len(set(pairs)), "Duplicate sample_id + experimental_condition pairs found"


def test_all_five_conditions_represented():
    eval_df = run_evaluation()
    actual_conditions = set(eval_df["experimental_condition"])
    assert actual_conditions == set(EXPECTED_CONDITIONS)


def test_output_schema():
    eval_df = run_evaluation()
    expected_cols = [
        "sample_id",
        "experimental_condition",
        "transformation_type",
        "image_available",
        "context_available",
        "context_relevant",
        "context_consistent",
        "evidence_strength",
        "predicted_evidence_state",
        "state_matches_condition",
        "gate_action",
        "gate_reason",
        "expected_gate_action",
        "gate_action_matches_expected",
    ]
    assert list(eval_df.columns) == expected_cols


def test_deterministic_analyzer_and_gate_mapping():
    eval_df = run_evaluation()

    # sufficient -> sufficient / generate
    suff = eval_df[eval_df["experimental_condition"] == "sufficient"]
    assert (suff["predicted_evidence_state"] == "sufficient").all()
    assert (suff["gate_action"] == "generate").all()
    assert (suff["state_matches_condition"] == True).all()

    # irrelevant -> irrelevant / discount_context
    irrel = eval_df[eval_df["experimental_condition"] == "irrelevant"]
    assert (irrel["predicted_evidence_state"] == "irrelevant").all()
    assert (irrel["gate_action"] == "discount_context").all()
    assert (irrel["state_matches_condition"] == True).all()

    # conflicting -> conflicting / qualify_or_abstain
    confl = eval_df[eval_df["experimental_condition"] == "conflicting"]
    assert (confl["predicted_evidence_state"] == "conflicting").all()
    assert (confl["gate_action"] == "qualify_or_abstain").all()
    assert (confl["state_matches_condition"] == True).all()


def test_observed_incomplete_structural_discrepancy():
    eval_df = run_evaluation()
    inc = eval_df[eval_df["experimental_condition"] == "incomplete"]
    assert len(inc) == 9
    # Prototype behavior: context_available is True, so analyzer predicts sufficient
    assert (inc["predicted_evidence_state"] == "sufficient").all()
    assert (inc["gate_action"] == "generate").all()
    assert (inc["state_matches_condition"] == False).all()


def test_observed_insufficient_structural_behavior():
    eval_df = run_evaluation()
    insuff = eval_df[eval_df["experimental_condition"] == "insufficient"]
    assert len(insuff) == 20
    # Calibrated hierarchy behavior: context removed with ES=1.0 -> analyzer predicts insufficient / abstain
    assert (insuff["predicted_evidence_state"] == "insufficient").all()
    assert (insuff["gate_action"] == "abstain").all()
    assert (insuff["state_matches_condition"] == True).all()

