"""
Tests for Phase 15 Evaluation Layer:
    - RadGraph output schema
    - CheXpert output schema & label vector preservation
    - Unsupported claim calculation & zero-denominator handling
    - Baseline / gated pairing by (sample_id, condition)
    - Condition grouping across 5 evidence states
    - Coverage & Risk calculations
    - Abstention metrics (precision, recall, rate, coverage)
    - Benchmark integrity audit (no accidental modification of frozen benchmark files)
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_chexpert_f1 import extract_chexpert_vector, compute_chexpert_f1, CHEXPERT_CATEGORIES
from src.evaluation.evaluate_unsupported_claims import compute_record_metrics
from src.evaluation.evaluate_abstention_metrics import compute_abstention_metrics_for_ref

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_PATH = ROOT / "results/tables/baseline_generation_results.csv"
GATED_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"

RADGRAPH_RES_PATH = ROOT / "results/tables/radgraph_f1_results.csv"
RADGRAPH_SUM_PATH = ROOT / "results/tables/radgraph_f1_summary.csv"
CHEXPERT_RES_PATH = ROOT / "results/tables/chexpert_f1_results.csv"
CHEXPERT_SUM_PATH = ROOT / "results/tables/chexpert_f1_summary.csv"
UNSUPP_RES_PATH = ROOT / "results/tables/unsupported_claim_rate.csv"
UNSUPP_SUM_PATH = ROOT / "results/tables/unsupported_claim_summary.csv"
RISK_PTS_PATH = ROOT / "results/tables/risk_coverage_points.csv"
RISK_SUM_PATH = ROOT / "results/tables/risk_coverage_summary.csv"
ABSTAIN_PATH = ROOT / "results/tables/abstention_metrics.csv"

REQUIRED_CONDITIONS = ["sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"]


def test_radgraph_output_schema():
    assert RADGRAPH_RES_PATH.exists(), f"Missing {RADGRAPH_RES_PATH}"
    assert RADGRAPH_SUM_PATH.exists(), f"Missing {RADGRAPH_SUM_PATH}"

    res_df = pd.read_csv(RADGRAPH_RES_PATH)
    sum_df = pd.read_csv(RADGRAPH_SUM_PATH)

    assert len(res_df) == 81
    expected_res_cols = {
        "sample_id", "condition",
        "baseline_radgraph_precision", "baseline_radgraph_recall", "baseline_radgraph_f1",
        "gated_radgraph_precision", "gated_radgraph_recall", "gated_radgraph_f1",
        "delta_radgraph_f1"
    }
    assert expected_res_cols.issubset(set(res_df.columns))

    expected_sum_conditions = {"overall", "sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"}
    assert set(sum_df["condition"].unique()) == expected_sum_conditions


def test_chexpert_output_schema_and_vectors():
    assert CHEXPERT_RES_PATH.exists(), f"Missing {CHEXPERT_RES_PATH}"
    assert CHEXPERT_SUM_PATH.exists(), f"Missing {CHEXPERT_SUM_PATH}"

    res_df = pd.read_csv(CHEXPERT_RES_PATH)
    sum_df = pd.read_csv(CHEXPERT_SUM_PATH)

    assert len(res_df) == 81
    expected_cols = {
        "sample_id", "condition",
        "baseline_chexpert_f1", "gated_chexpert_f1", "delta_chexpert_f1",
        "reference_chexpert_vector", "baseline_chexpert_vector", "gated_chexpert_vector"
    }
    assert expected_cols.issubset(set(res_df.columns))

    # Test label vector extraction function
    vec = extract_chexpert_vector("No cardiomegaly or pleural effusion. Mild atelectasis.")
    assert len(vec) == len(CHEXPERT_CATEGORIES)
    assert vec["Cardiomegaly"] == 0
    assert vec["Pleural Effusion"] == 0
    assert vec["Atelectasis"] == 1


def test_unsupported_claim_calculation_and_zero_denominator():
    # Test zero-denominator handling when generated claims == 0 (e.g. abstained report)
    mock_fact_df = pd.DataFrame([
        {
            "sample_id": "999",
            "condition": "insufficient",
            "finding": "cardiomegaly",
            "generated_polarity": "NOT_MENTIONED",
            "reference_polarity": "AFFIRMED",
            "label": "omission"
        }
    ])

    metrics = compute_record_metrics(mock_fact_df, "999", "insufficient")
    assert metrics["total_generated_claims"] == 0
    assert metrics["unsupported_claims"] == 0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["supported_claim_rate"] == 0.0


def test_baseline_gated_pairing():
    res_unsupp = pd.read_csv(UNSUPP_RES_PATH)
    assert len(res_unsupp) == 81

    pert_df = pd.read_csv(PERTURBATIONS_PATH)
    expected_pairs = set(zip(pert_df["sample_id"].astype(str), pert_df["condition"].astype(str)))
    actual_pairs = set(zip(res_unsupp["sample_id"].astype(str), res_unsupp["condition"].astype(str)))

    assert actual_pairs == expected_pairs, "Mismatch between perturbation pairs and evaluation pairs!"


def test_condition_grouping():
    sum_unsupp = pd.read_csv(UNSUPP_SUM_PATH)
    sum_conditions = set(sum_unsupp["condition"].unique())
    assert sum_conditions == {"overall", "sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"}


def test_coverage_and_risk_calculation():
    assert RISK_PTS_PATH.exists()
    assert RISK_SUM_PATH.exists()

    pts_df = pd.read_csv(RISK_PTS_PATH)
    sum_df = pd.read_csv(RISK_SUM_PATH)

    assert len(pts_df) == 81

    # Check baseline coverage = 1.0, gated coverage = 61/81 = 0.7531
    base_overall = sum_df[(sum_df["system"] == "baseline") & (sum_df["condition"] == "overall")].iloc[0]
    gated_overall = sum_df[(sum_df["system"] == "gated") & (sum_df["condition"] == "overall")].iloc[0]

    assert base_overall["coverage"] == 1.0
    assert gated_overall["coverage"] == 0.7531
    assert gated_overall["abstention_rate"] == 0.2469


def test_abstention_metrics():
    assert ABSTAIN_PATH.exists()
    df = pd.read_csv(ABSTAIN_PATH)
    assert len(df) == 2

    row_insuff = df[df["operational_reference"] == "ref_insufficient"].iloc[0]
    assert row_insuff["n_ref_abstain"] == 20
    assert row_insuff["n_gated_abstain"] == 20
    assert row_insuff["abstention_precision"] == 1.0
    assert row_insuff["abstention_recall"] == 1.0

    row_unsafe = df[df["operational_reference"] == "ref_unsafe_insufficient_or_conflicting"].iloc[0]
    assert row_unsafe["n_ref_abstain"] == 35
    assert row_unsafe["n_gated_abstain"] == 20
    assert row_unsafe["abstention_precision"] == 1.0
    assert row_unsafe["abstention_recall"] == 0.5714


def test_benchmark_integrity():
    pert_df = pd.read_csv(PERTURBATIONS_PATH)
    base_df = pd.read_csv(BASELINE_PATH)
    gated_df = pd.read_csv(GATED_PATH)

    assert len(pert_df) == 81
    assert len(base_df) == 81
    assert len(gated_df) == 81

    cond_counts = pert_df["condition"].value_counts().to_dict()
    assert cond_counts == {
        "irrelevant": 20,
        "insufficient": 20,
        "sufficient": 17,
        "conflicting": 15,
        "incomplete": 9,
    }
