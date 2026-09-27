from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.analyze_evidence_states import (
    compute_evidence_state_breakdown,
    REQUIRED_CONDITIONS,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results/tables/evidence_state_breakdown_analysis.csv"


def test_evidence_state_breakdown_file_exists():
    assert (
        OUTPUT_PATH.exists()
    ), "evidence_state_breakdown_analysis.csv missing"

    df = pd.read_csv(OUTPUT_PATH)
    assert len(df) == 5
    assert set(df["condition"]) == set(REQUIRED_CONDITIONS)

    expected_cols = [
        "condition",
        "case_count",
        "duplicate_report_rate",
        "unique_report_count",
        "mean_report_length_chars",
        "median_report_length_chars",
        "claim_count",
        "claim_density",
        "supported_proportion",
        "omission_proportion",
        "extra_negation_proportion",
        "ungrounded_affirmation_proportion",
        "unsupported_affirmation_proportion",
        "contradiction_proportion",
        "unclear_proportion",
    ]

    assert list(df.columns) == expected_cols


def test_case_counts_and_nested_proportions():
    df = pd.read_csv(OUTPUT_PATH)

    assert df["case_count"].sum() == 81

    # Verify factuality label proportions sum to 1.0 per condition
    prop_cols = [
        "supported_proportion",
        "omission_proportion",
        "extra_negation_proportion",
        "ungrounded_affirmation_proportion",
        "unsupported_affirmation_proportion",
        "contradiction_proportion",
        "unclear_proportion",
    ]

    sums = df[prop_cols].sum(axis=1)
    for idx, s in enumerate(sums):
        assert pytest.approx(s, rel=1e-5) == 1.0

    # Verify specific nested values
    suff = df[df["condition"] == "sufficient"].iloc[0]
    assert suff["case_count"] == 17
    assert pytest.approx(suff["supported_proportion"], rel=1e-4) == 33 / 74
    assert pytest.approx(suff["omission_proportion"], rel=1e-4) == 31 / 74

    insuff = df[df["condition"] == "insufficient"].iloc[0]
    assert insuff["case_count"] == 20
    assert pytest.approx(insuff["extra_negation_proportion"], rel=1e-4) == 31 / 104


def test_compute_evidence_state_breakdown_synthetic():
    condition_df = pd.DataFrame(
        [
            {
                "condition": "sufficient",
                "case_count": 10,
                "duplicate_report_rate": 0.0,
                "unique_report_count": 10,
                "mean_report_length_chars": 100.0,
                "median_report_length_chars": 100.0,
                "claim_count": 20,
            },
            {
                "condition": "incomplete",
                "case_count": 5,
                "duplicate_report_rate": 0.2,
                "unique_report_count": 4,
                "mean_report_length_chars": 90.0,
                "median_report_length_chars": 90.0,
                "claim_count": 10,
            },
            {
                "condition": "irrelevant",
                "case_count": 5,
                "duplicate_report_rate": 0.0,
                "unique_report_count": 5,
                "mean_report_length_chars": 95.0,
                "median_report_length_chars": 95.0,
                "claim_count": 10,
            },
            {
                "condition": "conflicting",
                "case_count": 5,
                "duplicate_report_rate": 0.0,
                "unique_report_count": 5,
                "mean_report_length_chars": 95.0,
                "median_report_length_chars": 95.0,
                "claim_count": 10,
            },
            {
                "condition": "insufficient",
                "case_count": 5,
                "duplicate_report_rate": 0.0,
                "unique_report_count": 5,
                "mean_report_length_chars": 95.0,
                "median_report_length_chars": 95.0,
                "claim_count": 10,
            },
        ]
    )

    fact_summary_df = pd.DataFrame(
        [
            {
                "condition": "sufficient",
                "comparison_row_count": 20,
                "supported": 10,
                "omission": 10,
                "extra_negation": 0,
            },
            {
                "condition": "incomplete",
                "comparison_row_count": 10,
                "supported": 5,
                "omission": 5,
                "extra_negation": 0,
            },
            {
                "condition": "irrelevant",
                "comparison_row_count": 10,
                "supported": 5,
                "omission": 5,
                "extra_negation": 0,
            },
            {
                "condition": "conflicting",
                "comparison_row_count": 10,
                "supported": 5,
                "omission": 5,
                "extra_negation": 0,
            },
            {
                "condition": "insufficient",
                "comparison_row_count": 10,
                "supported": 5,
                "omission": 5,
                "extra_negation": 0,
            },
        ]
    )

    fact_detail_df = pd.DataFrame()

    res = compute_evidence_state_breakdown(
        condition_df, fact_summary_df, fact_detail_df
    )

    assert len(res) == 5
    suff_res = res[res["condition"] == "sufficient"].iloc[0]
    assert suff_res["claim_density"] == 2.0
    assert suff_res["supported_proportion"] == 0.5
    assert suff_res["omission_proportion"] == 0.5
