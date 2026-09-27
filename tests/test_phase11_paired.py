from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.evaluate_phase11_paired import (
    CASE_METRICS_PATH,
    GATE_BEHAVIOR_PATH,
    PAIRED_COMP_PATH,
    STAT_TESTS_PATH,
    compute_case_level_metrics,
    compute_gate_behavior,
    compute_paired_comparison,
    load_input_datasets,
    run_statistical_tests,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase11_output_files_exist():
    assert CASE_METRICS_PATH.exists(), "phase11_case_level_metrics.csv missing"
    assert PAIRED_COMP_PATH.exists(), "phase11_paired_comparison.csv missing"
    assert STAT_TESTS_PATH.exists(), "phase11_statistical_tests.csv missing"
    assert GATE_BEHAVIOR_PATH.exists(), "phase11_gate_behavior.csv missing"


def test_phase11_csv_row_counts():
    case_df = pd.read_csv(CASE_METRICS_PATH)
    paired_df = pd.read_csv(PAIRED_COMP_PATH)
    stat_df = pd.read_csv(STAT_TESTS_PATH)
    gate_df = pd.read_csv(GATE_BEHAVIOR_PATH)

    assert len(case_df) == 162
    assert len(paired_df) == 81
    assert len(stat_df) == 80  # 8 scopes * 10 variables
    assert len(gate_df) == 6


def test_phase11_case_level_metrics_columns():
    case_df = pd.read_csv(CASE_METRICS_PATH)
    assert "claim_count_per_report" in case_df.columns
    assert "claim_density" not in case_df.columns  # Replaced with claim_count_per_report
    assert (case_df["claim_count_per_report"] == case_df["claim_count"]).all()


def test_phase11_statistical_scopes_stratification():
    stat_df = pd.read_csv(STAT_TESTS_PATH)
    scopes = set(stat_df["scope"])

    expected_scopes = {
        "overall_all_conditions",
        "non_irrelevant_conditions",
        "irrelevant_context_discount",
        "sufficient",
        "incomplete",
        "irrelevant",
        "conflicting",
        "insufficient",
    }
    assert scopes == expected_scopes

    # Check pair counts per scope
    overall_all = stat_df[stat_df["scope"] == "overall_all_conditions"]
    assert (overall_all["n_pairs"] == 81).all()

    non_irrel = stat_df[stat_df["scope"] == "non_irrelevant_conditions"]
    assert (non_irrel["n_pairs"] == 61).all()

    irrel_disc = stat_df[stat_df["scope"] == "irrelevant_context_discount"]
    assert (irrel_disc["n_pairs"] == 20).all()


def test_non_irrelevant_scope_test_statuses():
    stat_df = pd.read_csv(STAT_TESTS_PATH)
    non_irrel = stat_df[stat_df["scope"] == "non_irrelevant_conditions"]

    # In non_irrelevant_conditions, claim_count and factuality flags are identical (zero variation/discordance)
    claim_test = non_irrel[non_irrel["variable"] == "claim_count"].iloc[0]
    assert claim_test["test_status"] == "not_applicable"
    assert "zero variation" in str(claim_test["notes"])

    omission_test = non_irrel[non_irrel["variable"] == "has_omission"].iloc[0]
    assert omission_test["test_status"] == "not_applicable"
    assert "zero discordant" in str(omission_test["notes"])


def test_phase11_case_level_pairing_and_inference():
    gated_df = pd.read_csv(ROOT / "results/tables/gated_generation_results.csv")
    assert (gated_df["was_model_inference_run"] == False).all()

    paired_df = pd.read_csv(PAIRED_COMP_PATH)
    pairs = set(zip(paired_df["sample_id"].astype(str), paired_df["condition"].astype(str)))
    assert len(pairs) == 81


def test_phase11_gate_behavior_and_analyzer_alignment():
    gate_df = pd.read_csv(GATE_BEHAVIOR_PATH).set_index("condition")

    # Analyzer alignment discrepancies explicitly preserved
    assert gate_df.loc["sufficient", "analyzer_alignment_rate"] == 1.0
    assert gate_df.loc["incomplete", "analyzer_alignment_rate"] == 0.0
    assert gate_df.loc["irrelevant", "analyzer_alignment_rate"] == 1.0
    assert gate_df.loc["conflicting", "analyzer_alignment_rate"] == 1.0
    assert gate_df.loc["insufficient", "analyzer_alignment_rate"] == 0.0


def test_baseline_and_gated_files_unmodified():
    base_df = pd.read_csv(ROOT / "results/tables/baseline_generation_results.csv")
    gated_df = pd.read_csv(ROOT / "results/tables/gated_generation_results.csv")
    assert len(base_df) == 81
    assert len(gated_df) == 81
