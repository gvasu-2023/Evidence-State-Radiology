from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_phase24_consolidation import (
    FIGURE_DIR,
    OUTPUTS,
    build_phase24_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phase24():
    return build_phase24_outputs()


def test_phase24_master_totals_rates_and_metrics(phase24):
    tables, _, _ = phase24
    master = tables["master"].set_index("evaluation_view")

    assert master["N"].to_dict() == {
        "Baseline": 576,
        "Gated - all records": 576,
        "Gated - covered only": 480,
    }
    assert np.isclose(master.loc["Gated - all records", "coverage"], 480 / 576)
    assert np.isclose(master.loc["Gated - all records", "abstention_rate"], 96 / 576)
    assert master.loc["Baseline", "unsupported_claim_count"] == 120
    assert master.loc["Baseline", "generated_candidate_count"] == 1578
    assert np.isclose(master.loc["Baseline", "unsupported_claim_rate"], 120 / 1578)
    assert master.loc["Gated - all records", "unsupported_claim_count"] == 106
    assert master.loc["Gated - all records", "generated_candidate_count"] == 1460
    assert np.isclose(master.loc["Gated - covered only", "unsupported_claim_rate"], 106 / 1460)
    assert np.isclose(master.loc["Gated - all records", "radgraph_mean"], 0.16609375)


def test_phase24_condition_results_preserve_balance_and_coverage(phase24):
    tables, _, _ = phase24
    condition = tables["condition"]

    assert len(condition) == 6
    assert condition["N"].sum() == 576
    assert condition.set_index("condition")["N"].to_dict() == {
        "sufficient": 96,
        "syntactic_incomplete": 96,
        "evidentiary_incomplete": 96,
        "irrelevant": 96,
        "conflicting": 96,
        "insufficient": 96,
    }
    assert condition["covered"].sum() == 480
    assert condition["abstained"].sum() == 96
    insufficient = condition.set_index("condition").loc["insufficient"]
    assert insufficient["covered"] == 0
    assert insufficient["abstained"] == 96


def test_phase24_statistical_values_match_phase22(phase24):
    tables, _, _ = phase24
    stats = tables["statistics"]
    primary = stats[stats.analysis == "all-record primary family"].set_index("endpoint")
    covered = stats[stats.analysis == "covered-only secondary family"].set_index("endpoint")
    claim = stats[stats.analysis == "secondary claim-level analysis"].iloc[0]

    assert primary.loc["RadGraph-F1", "N"] == 576
    assert primary.loc["RadGraph-F1", "wilcoxon_W"] == 10532.5
    assert primary.loc["RadGraph-F1", "p_holm_adjusted"] == pytest.approx(1.2421122766592228e-9)
    assert primary.loc["CheXpert-F1", "p_holm_adjusted"] == pytest.approx(0.8426983960635385)
    assert covered.loc["RadGraph-F1", "N"] == 480
    assert covered.loc["RadGraph-F1", "p_holm_adjusted"] == pytest.approx(0.11945646173536378)
    assert covered.loc["CheXpert-F1", "p_holm_adjusted"] == pytest.approx(0.5766065444779812)
    assert claim["N"] == 576
    assert claim["baseline_positive_records"] == 74
    assert claim["gated_positive_records"] == 64
    assert claim["baseline_only_discordant"] == 14
    assert claim["gated_only_discordant"] == 4
    assert claim["p_raw"] == pytest.approx(0.0308837890625)
    assert pd.isna(claim["p_holm_adjusted"])


def test_phase24_gate_and_qualitative_summaries_keep_categories_separate(phase24):
    tables, docs, _ = phase24
    gate = tables["gate"]
    qualitative = tables["qualitative"]

    assert gate.category_type.value_counts().to_dict() == {"analyzer_state": 5, "gate_action": 5}
    assert set(gate[gate.category_type == "gate_action"].category) == {
        "generate", "qualify", "discount_context", "qualify_or_abstain", "abstain"
    }
    assert len(qualitative) == 6
    assert qualitative.selected_case_count.eq(3).all()
    assert "not 18 independent patients" in docs["synthesis"]
    assert "does not imply evidentiary incompleteness is harmless" in docs["synthesis"]


def test_phase24_output_tables_have_unique_rows_and_figures_exist(phase24):
    tables, _, _ = phase24
    assert not tables["master"].duplicated(["evaluation_view"]).any()
    assert not tables["statistics"].duplicated(["analysis", "endpoint"]).any()
    assert not tables["condition"].duplicated(["condition"]).any()
    assert not tables["gate"].duplicated(["category_type", "category"]).any()
    assert not tables["qualitative"].duplicated(["condition"]).any()
    assert len(list(FIGURE_DIR.glob("*.png"))) == 4
    assert all(path.stat().st_size > 0 for path in FIGURE_DIR.glob("*.png"))


def test_phase24_documents_and_csv_artifacts_exist(phase24):
    _, docs, _ = phase24
    assert all(path.is_file() for path in OUTPUTS.values())
    assert "generated-candidate denominator" in docs["synthesis"]
    assert "Holm-adjusted threshold" in docs["synthesis"]
    assert "McNemar" in docs["paper"]
    assert "213 passed" not in docs["addendum"]
    assert "219 passed" in docs["addendum"]
    assert pd.read_csv(OUTPUTS["master"]).shape[0] == 3
    assert pd.read_csv(OUTPUTS["qualitative"]).shape[0] == 6
