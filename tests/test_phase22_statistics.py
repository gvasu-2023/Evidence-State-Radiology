from pathlib import Path

import numpy as np

from src.evaluation.run_phase22_statistics import (
    _holm_adjust,
    _paired_effect_and_test,
    run_phase22,
)


def test_phase22_primary_and_covered_cohorts_are_paired_and_complete():
    outputs, docs, _ = run_phase22()
    primary = outputs["primary"].set_index("endpoint")
    covered = outputs["covered"].set_index("endpoint")

    assert primary["n"].to_dict() == {"radgraph_f1": 576, "chexpert_f1": 576}
    assert covered["n"].to_dict() == {"radgraph_f1": 480, "chexpert_f1": 480}
    assert np.isclose(primary.loc["radgraph_f1", "p_value_holm"], 1.242112e-9, rtol=1e-5)
    assert primary.loc["radgraph_f1", "significant_after_holm"]
    assert not primary.loc["chexpert_f1", "significant_after_holm"]
    assert not covered["significant_after_holm"].any()
    assert "Statistical significance does not establish clinical significance" in docs
    assert "No tests were run separately by condition" in docs


def test_phase22_difference_diagnostics_reconcile_to_cohort_sizes():
    outputs, _, _ = run_phase22()
    diagnostics = outputs["diagnostics"]

    assert len(diagnostics) == 4
    assert diagnostics["n"].to_dict() == {0: 576, 1: 576, 2: 480, 3: 480}
    assert (
        diagnostics["zero_differences"]
        + diagnostics["positive_differences"]
        + diagnostics["negative_differences"]
    ).equals(diagnostics["n"])


def test_phase22_rank_biserial_and_holm_follow_standard_formulas():
    statistic, p_value, effect = _paired_effect_and_test(np.array([1.0, 4.0, -2.0]))

    assert statistic == 2.0
    assert 0 <= p_value <= 1
    assert np.isclose(effect, 1 / 3)
    assert _holm_adjust([0.01, 0.04]) == [0.02, 0.04]


def test_phase22_claim_comparison_is_record_level_exact_mcnemar():
    outputs, _, _ = run_phase22()
    claim = outputs["claim"].iloc[0]

    assert claim["n"] == 576
    assert claim["baseline_positive_records"] == 74
    assert claim["gated_positive_records"] == 64
    assert claim["baseline_only_discordant_pairs"] == 14
    assert claim["gated_only_discordant_pairs"] == 4
    assert claim["discordant_pairs"] == 18
    assert np.isclose(claim["p_value"], 0.0308839, atol=1e-6)
    assert "Exact McNemar" in claim["test"]


def test_phase22_requested_artifacts_exist_after_runner_execution():
    root = Path(__file__).resolve().parents[1]
    outputs, _, _ = run_phase22()

    assert set(outputs) == {"primary", "covered", "claim", "diagnostics"}
    assert (root / "results/tables/phase22_primary_statistics.csv").is_file()
    assert (root / "results/tables/phase22_covered_statistics.csv").is_file()
    assert (root / "results/tables/phase22_claim_statistics.csv").is_file()
    assert (root / "results/tables/phase22_difference_diagnostics.csv").is_file()
    assert (root / "docs/phase22_statistical_analysis.md").is_file()
