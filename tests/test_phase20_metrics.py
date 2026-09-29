import pandas as pd

from src.evaluation.run_phase20_metrics import _summary


def test_phase20_summary_separates_all_gated_from_covered_only():
    results = pd.DataFrame(
        [
            {
                "sample_id": "IU_1",
                "uid": 1,
                "condition": "sufficient",
                "analyzer_evidence_state": "sufficient",
                "gate_action": "generate",
                "is_gated_abstention": False,
                "gated_report_covered": True,
                "baseline_radgraph_f1": 0.8,
                "gated_radgraph_f1": 0.7,
                "baseline_chexpert_f1": 0.6,
                "gated_chexpert_f1": 0.5,
            },
            {
                "sample_id": "IU_2",
                "uid": 2,
                "condition": "insufficient",
                "analyzer_evidence_state": "insufficient",
                "gate_action": "abstain",
                "is_gated_abstention": True,
                "gated_report_covered": False,
                "baseline_radgraph_f1": 0.4,
                "gated_radgraph_f1": 0.0,
                "baseline_chexpert_f1": 0.2,
                "gated_chexpert_f1": 0.0,
            },
        ]
    )

    summary = _summary(results)
    overall = summary[summary["summary_level"] == "overall"].iloc[0]
    abstain = summary[
        (summary["summary_level"] == "gate_action")
        & (summary["scope_value"] == "abstain")
    ].iloc[0]

    assert overall["n_records"] == 2
    assert overall["n_gated_covered"] == 1
    assert overall["n_gated_abstained"] == 1
    assert overall["overall_gated_radgraph_f1_mean"] == 0.35
    assert overall["covered_gated_radgraph_f1_mean"] == 0.7
    assert overall["overall_gated_chexpert_f1_mean"] == 0.25
    assert overall["covered_gated_chexpert_f1_mean"] == 0.5
    assert pd.isna(abstain["covered_gated_radgraph_f1_mean"])
    assert pd.isna(abstain["covered_gated_chexpert_f1_mean"])
