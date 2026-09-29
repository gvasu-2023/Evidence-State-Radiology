from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_phase21_risk_coverage import (
    OUTPUTS,
    build_phase21_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phase21():
    return build_phase21_outputs()


def test_phase21_coverage_and_condition_balance(phase21):
    outputs, docs, _, _ = phase21
    risk = outputs["risk"].set_index("view")
    overall = risk.loc["gated_all_records"]
    conditions = outputs["condition"].set_index("condition")
    actions = outputs["action"].set_index("gate_action")

    assert int(overall["eligible_n"]) == 576
    assert int(overall["covered_n"]) == 480
    assert int(overall["abstained_n"]) == 96
    assert np.isclose(overall["coverage"], 480 / 576)
    assert np.isclose(overall["abstention_rate"], 96 / 576)
    assert (conditions["n"] == 96).all()
    assert conditions.loc["insufficient", "covered_n"] == 0
    assert conditions.loc["insufficient", "abstained_n"] == 96
    assert (conditions.drop(index="insufficient")["covered_n"] == 96).all()
    assert actions["n"].to_dict() == {
        "generate": 192,
        "qualify": 96,
        "discount_context": 96,
        "qualify_or_abstain": 96,
        "abstain": 96,
    }
    assert "evidentiary_incomplete" in docs
    assert "not establish statistical significance" in docs


def test_phase21_claim_risk_uses_actual_claim_denominators(phase21):
    outputs, _, _, _ = phase21
    risk = outputs["risk"].set_index("view")

    assert risk.loc["baseline", "unsupported_claim_count"] == 120
    assert risk.loc["baseline", "evaluated_generated_claim_denominator"] == 1578
    assert np.isclose(risk.loc["baseline", "unsupported_claim_rate"], 120 / 1578)
    assert risk.loc["gated_all_records", "unsupported_claim_count"] == 106
    assert risk.loc["gated_all_records", "evaluated_generated_claim_denominator"] == 1460
    assert np.isclose(risk.loc["gated_all_records", "unsupported_claim_rate"], 106 / 1460)
    assert risk.loc["gated_covered_only", "unsupported_claim_count"] == 106
    assert risk.loc["gated_covered_only", "evaluated_generated_claim_denominator"] == 1460

    abstain = outputs["claim"].query(
        "summary_level == 'gate_action' and scope_value == 'abstain' and report_view == 'gated_all_records'"
    ).iloc[0]
    assert abstain["evaluated_generated_claim_denominator"] == 0
    assert abstain["unsupported_claim_count"] == 0
    assert abstain["omission_count"] > 0


def test_phase21_observed_analyzer_states_are_preserved(phase21):
    outputs, _, _, _ = phase21
    states = outputs["state"].set_index("analyzer_evidence_state")

    assert set(states.index) == {
        "sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"
    }
    assert states.loc["sufficient", "n"] == 192
    assert states.loc["incomplete", "n"] == 96
    assert states.loc["insufficient", "covered_n"] == 0


def test_phase21_quality_summaries_match_phase20_and_outputs_exist(phase21):
    outputs, _, _, _ = phase21
    risk = outputs["risk"].set_index("view")
    rad = pd.read_csv(ROOT / "results/tables/phase20_radgraph_f1_results.csv")
    chex = pd.read_csv(ROOT / "results/tables/phase20_chexpert_f1_results.csv")

    assert np.isclose(risk.loc["baseline", "baseline_radgraph_f1_mean"], rad["baseline_radgraph_f1"].mean())
    assert np.isclose(risk.loc["gated_all_records", "gated_all_radgraph_f1_mean"], rad["gated_radgraph_f1"].mean())
    assert np.isclose(risk.loc["gated_covered_only", "gated_covered_chexpert_f1_mean"], chex.loc[chex["gated_report_covered"], "gated_chexpert_f1"].mean())
    assert all(OUTPUTS[key].is_file() for key in ("risk", "condition", "state", "action", "claim", "docs"))
