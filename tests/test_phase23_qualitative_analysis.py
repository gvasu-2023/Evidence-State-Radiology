from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.run_phase23_qualitative_analysis import (
    OUTPUTS,
    _add_features,
    _deterministic_selection,
    _load_cohort,
    run_phase23,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phase23():
    return run_phase23()


def test_phase23_selection_is_deterministic_and_balanced(phase23):
    cases, patterns, _, _ = phase23
    cohort, _ = _load_cohort()
    rerun = _deterministic_selection(_add_features(cohort))

    assert len(cases) == 18
    assert cases.condition.value_counts().to_dict() == {
        "sufficient": 3,
        "syntactic_incomplete": 3,
        "evidentiary_incomplete": 3,
        "irrelevant": 3,
        "conflicting": 3,
        "insufficient": 3,
    }
    assert not cases.duplicated(["sample_id", "condition"]).any()
    assert cases[["sample_id", "uid", "condition"]].reset_index(drop=True).equals(
        rerun[["sample_id", "uid", "condition"]].reset_index(drop=True)
    )
    assert len(patterns) >= 8


def test_phase23_evidentiary_incomplete_limitation_is_explicit(phase23):
    cases, patterns, docs, _ = phase23
    subset = cases[cases.condition == "evidentiary_incomplete"]
    limitation = patterns[patterns.pattern == "evidentiary-incomplete analyzer limitation"].iloc[0]

    assert len(subset) == 3
    assert subset.analyzer_state.eq("sufficient").all()
    assert subset.gate_action.eq("generate").all()
    assert subset.report_unchanged.all()
    assert subset.qualitative_categories.str.contains("analyzer_misclassification").all()
    assert limitation.record_count == 96
    assert "not evidence that evidentiary incompleteness is harmless" in docs


def test_phase23_insufficient_examples_are_abstentions(phase23):
    cases, _, _, _ = phase23
    subset = cases[cases.condition == "insufficient"]

    assert len(subset) == 3
    assert subset.gate_action.eq("abstain").all()
    assert subset.qualitative_categories.str.contains("abstention_due_to_insufficient_evidence").all()
    assert subset.gated_report.str.contains("abstain", case=False).all()


def test_phase23_cases_include_context_reference_claims_and_existing_scores(phase23):
    cases, _, _, _ = phase23
    required = {
        "original_context", "perturbed_context", "findings", "impression",
        "baseline_report", "gated_report", "baseline_claim_labels", "gated_claim_labels",
        "baseline_behavior", "gated_behavior", "claim_level_change",
        "baseline_radgraph_f1", "gated_radgraph_f1", "baseline_chexpert_f1", "gated_chexpert_f1",
        "qualitative_categories", "qualitative_interpretation",
    }

    assert required.issubset(cases.columns)
    assert cases.findings.notna().all()
    assert cases.impression.notna().all()
    assert cases.baseline_report.notna().all() and cases.gated_report.notna().all()


def test_phase23_artifacts_exist_and_patterns_are_descriptive(phase23):
    _, patterns, docs, _ = phase23

    assert all(path.is_file() for path in OUTPUTS.values())
    assert "No subgroup hypothesis tests were performed" in docs
    assert "qualification can therefore leave evaluated contradiction labels in place" in docs
    assert "successful overall reduction" in docs
    assert "insufficient evidence abstention" in set(patterns.pattern)
    assert pd.read_csv(OUTPUTS["cases"]).shape[0] == 18
