import pytest

from src.evaluation.run_phase19_claim_evaluation import (
    LABELS,
    build_phase19_outputs,
)


@pytest.fixture(scope="module")
def phase19_outputs():
    return build_phase19_outputs()


def test_phase19_pairs_all_eligible_reports_and_keeps_both_state_labels(phase19_outputs):
    detail, _ = phase19_outputs
    record_keys = detail[["sample_id", "uid", "benchmark_condition"]].drop_duplicates()

    assert len(record_keys) == 576
    assert record_keys["benchmark_condition"].value_counts().to_dict() == {
        "sufficient": 96,
        "syntactic_incomplete": 96,
        "evidentiary_incomplete": 96,
        "irrelevant": 96,
        "conflicting": 96,
        "insufficient": 96,
    }
    assert not (set(record_keys["uid"].astype(int)) & {74, 597, 803, 885})
    assert not detail.duplicated(
        ["sample_id", "uid", "benchmark_condition", "finding"]
    ).any()
    assert set(detail["analyzer_evidence_state"]) == {
        "sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"
    }
    assert set(detail["gate_action"]) == {
        "generate", "qualify", "discount_context", "qualify_or_abstain", "abstain"
    }
    assert set(detail["baseline_label"].dropna()) <= set(LABELS) | {""}
    assert set(detail["gated_label"].dropna()) <= set(LABELS) | {""}


def test_phase19_summary_preserves_omission_and_unsupported_labels(phase19_outputs):
    _, summary = phase19_outputs
    overall = summary[summary["summary_level"] == "overall"].set_index("report_type")

    assert set(overall.index) == {"baseline", "gated"}
    assert int(overall.loc["baseline", "record_count"]) == 576
    assert int(overall.loc["gated", "record_count"]) == 576
    for label in LABELS:
        assert label in summary.columns
    assert int(overall.loc["baseline", "omission"]) > 0
    assert int(overall.loc["gated", "omission"]) > 0
    assert int(overall.loc["baseline", "unsupported_claim_count"]) == sum(
        int(overall.loc["baseline", label])
        for label in ("ungrounded_affirmation", "unsupported_affirmation", "contradiction")
    )
    assert int(overall.loc["gated", "unsupported_claim_count"]) == sum(
        int(overall.loc["gated", label])
        for label in ("ungrounded_affirmation", "unsupported_affirmation", "contradiction")
    )
    assert ((summary["unsupported_claim_rate"] >= 0) & (summary["unsupported_claim_rate"] <= 1)).all()
