"""Lightweight provenance checks; these tests never load MAIRA-2."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "experiments/maira2/phase26d/phase26d_execution.ipynb"
PROVENANCE = ROOT / "docs/phase26d_execution_provenance.md"
README = ROOT / "experiments/maira2/phase26d/README.md"
GENERATION = ROOT / "results/maira2_contrastive/phase26d_evaluation/phase26d_evaluation_generation.csv"
CONDITIONS = {
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
}

REQUIRED_RESULTS = (
    "results/tables/phase26d_claim_evaluation/phase26d_evaluation_claims.csv",
    "results/tables/phase26d_claim_evaluation/phase26d_evaluation_claim_summary.csv",
    "results/tables/phase26d_claim_evaluation/phase26d_evaluation_paired_records.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_radgraph.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_radgraph_summary.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_radgraph_paired.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_radgraph_paired_stats.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_chexpert.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_chexpert_summary.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_chexpert_paired.csv",
    "results/tables/phase26d_metrics/phase26d_evaluation_chexpert_paired_stats.csv",
)


def test_provenance_records_and_required_artifacts_exist():
    assert NOTEBOOK.is_file()
    assert PROVENANCE.is_file()
    assert README.is_file()
    assert GENERATION.is_file()
    assert all((ROOT / relative_path).is_file() for relative_path in REQUIRED_RESULTS)


def test_extracted_module_imports_without_loading_a_model():
    module = importlib.import_module("src.generation.maira2_contrastive")

    assert callable(module.prepare_maira_inputs)
    assert callable(module.two_stream_contrastive_decode)
    assert callable(module.generate_maira_trajectory)


def test_lambda_selection_and_statistical_exception_are_documented():
    text = PROVENANCE.read_text(encoding="utf-8")

    assert "The selected value is `lambda = 0.25`, chosen on development only." in text
    assert "evaluation set was not used for lambda selection" in text
    assert "0.3363292101` remains a documented reproducibility exception" in text


def test_frozen_generation_structure_matches_notebook_audit():
    generation = pd.read_csv(GENERATION)

    assert len(generation) == 960
    assert generation["uid"].nunique() == 80
    assert generation["lambda"].value_counts().to_dict() == {0.0: 480, 0.25: 480}
    assert generation["condition"].value_counts().to_dict() == {
        condition: 160 for condition in CONDITIONS
    }
    assert generation["status"].eq("success").all()


def test_phase26d_result_csv_contents_match_head_before_provenance_work():
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    result_paths = [
        path
        for path in tracked
        if path.startswith("results/")
        and "phase26d" in path.lower()
        and path.lower().endswith(".csv")
    ]

    assert result_paths, "HEAD contains no tracked Phase 26D result CSVs."

    def normalize_line_endings(content: bytes) -> bytes:
        # Windows checkouts may use CRLF while Git blobs use LF. Compare all
        # other bytes exactly so this check detects content/value changes.
        return content.replace(bytes((13, 10)), bytes((10,))).replace(bytes((13,)), bytes((10,)))

    for relative_path in result_paths:
        current_path = ROOT / relative_path
        expected = subprocess.check_output(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=ROOT,
        )
        assert current_path.is_file(), f"Missing tracked Phase 26D result: {relative_path}"
        assert normalize_line_endings(current_path.read_bytes()) == normalize_line_endings(expected), (
            f"Phase 26D result CSV differs from HEAD: {relative_path}"
        )
