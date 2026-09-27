from pathlib import Path

import pandas as pd
import pytest

from src.gating.gated_generator import (
    BASELINE_RESULTS_PATH,
    OUTPUT_PATH as GATED_RESULTS_PATH,
    PERTURBATIONS_PATH,
)
from src.gating.run_gated_experiment import (
    REQUIRED_COLUMNS,
    run_gated_experiment,
)
from src.evaluation.evaluate_gated_generation import (
    OUTPUT_PATH as COMPARISON_PATH,
    compute_gated_factuality_comparison,
    load_input_tables,
)

ROOT = Path(__file__).resolve().parents[1]


def test_gated_generation_results_file_exists():
    assert GATED_RESULTS_PATH.exists(), "gated_generation_results.csv missing"
    df = pd.read_csv(GATED_RESULTS_PATH)
    assert len(df) == 81
    assert list(df.columns) == REQUIRED_COLUMNS


def test_gated_factuality_comparison_file_exists():
    assert COMPARISON_PATH.exists(), "gated_factuality_comparison.csv missing"
    df = pd.read_csv(COMPARISON_PATH)
    assert len(df) == 5
    assert set(df["condition"]) == {
        "sufficient",
        "incomplete",
        "irrelevant",
        "conflicting",
        "insufficient",
    }


def test_exactly_81_records_and_unique_pairs():
    gated_df = run_gated_experiment()
    assert len(gated_df) == 81

    pairs = list(zip(gated_df["sample_id"], gated_df["experimental_condition"]))
    assert len(pairs) == len(set(pairs))
    assert len(set(gated_df["experimental_condition"])) == 5


def test_predicted_states_preserve_phase9_discrepancies():
    gated_df = run_gated_experiment()

    # sufficient -> sufficient / generate
    suff = gated_df[gated_df["experimental_condition"] == "sufficient"]
    assert (suff["predicted_evidence_state"] == "sufficient").all()
    assert (suff["gate_action"] == "generate").all()

    # incomplete -> sufficient / generate (Phase 9 structural discrepancy)
    inc = gated_df[gated_df["experimental_condition"] == "incomplete"]
    assert (inc["predicted_evidence_state"] == "sufficient").all()
    assert (inc["gate_action"] == "generate").all()

    # irrelevant -> irrelevant / discount_context
    irr = gated_df[gated_df["experimental_condition"] == "irrelevant"]
    assert (irr["predicted_evidence_state"] == "irrelevant").all()
    assert (irr["gate_action"] == "discount_context").all()

    # conflicting -> conflicting / qualify_or_abstain
    con = gated_df[gated_df["experimental_condition"] == "conflicting"]
    assert (con["predicted_evidence_state"] == "conflicting").all()
    assert (con["gate_action"] == "qualify_or_abstain").all()

    # insufficient -> incomplete / qualify (Phase 9 structural discrepancy)
    ins = gated_df[gated_df["experimental_condition"] == "insufficient"]
    assert (ins["predicted_evidence_state"] == "incomplete").all()
    assert (ins["gate_action"] == "qualify").all()


def test_zero_redundant_inference():
    gated_df = run_gated_experiment()
    # All 81 records must set was_model_inference_run == False
    assert (gated_df["was_model_inference_run"] == False).all()


def test_baseline_generation_results_remains_unchanged():
    assert BASELINE_RESULTS_PATH.exists()
    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
    assert len(baseline_df) == 81
    assert set(baseline_df["condition"]) == {
        "sufficient",
        "incomplete",
        "irrelevant",
        "conflicting",
        "insufficient",
    }


def test_policy_headers_do_not_introduce_false_clinical_claims():
    gated_df, baseline_breakdown_df, reference_df = load_input_tables()
    comparison_df, gated_claims_df, gated_detail_df = (
        compute_gated_factuality_comparison(
            gated_df, baseline_breakdown_df, reference_df
        )
    )

    # Policy headers: QUALIFIED, WARNING, ABSTAIN must not be extracted as clinical finding patterns
    extracted_findings = set(gated_claims_df["finding"])
    assert "qualified" not in extracted_findings
    assert "warning" not in extracted_findings
    assert "abstain" not in extracted_findings
