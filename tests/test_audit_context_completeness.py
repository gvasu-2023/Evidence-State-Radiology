from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.audit_context_completeness import (
    OUTPUT_PATH as AUDIT_OUTPUT_PATH,
    identify_triggered_rule,
    run_robustness_audit,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE12A_EVAL_PATH = ROOT / "results/tables/phase12_analyzer_evaluation.csv"
PHASE12A_SUMM_PATH = ROOT / "results/tables/phase12_analyzer_summary.csv"


def test_audit_context_completeness_file_exists():
    assert AUDIT_OUTPUT_PATH.exists(), "context_completeness_robustness_audit.csv missing"
    df = pd.read_csv(AUDIT_OUTPUT_PATH)
    assert len(df) > 0
    expected_cols = [
        "context_text",
        "expected_behavior",
        "predicted_context_complete",
        "false_incomplete",
        "rule_triggered",
        "source_type",
    ]
    assert list(df.columns) == expected_cols


def test_rule_identification_logic():
    assert identify_triggered_rule("chest pain") == "lowercase_initial"
    assert (
        identify_triggered_rule("History of mitral valve prolapse.")
        == "history_without_primary_indicator"
    )
    assert identify_triggered_rule("preop for surgery") == "lowercase_initial"  # lower initial takes precedence
    assert identify_triggered_rule("Preop for surgery") == "preop_prefix"
    assert identify_triggered_rule("Chest pain.") == "none"


def test_audit_identifies_false_incompleteness():
    audit_df = run_robustness_audit()
    assert len(audit_df) > 0

    # Verify standalone history contexts trigger false_incomplete with history_without_primary_indicator rule
    hist_case = audit_df[
        audit_df["context_text"] == "History of mitral valve prolapse."
    ]
    assert len(hist_case) > 0
    hist_row = hist_case.iloc[0]

    if hist_row["expected_behavior"] == "complete":
        assert hist_row["predicted_context_complete"] == False
        assert hist_row["false_incomplete"] == True
        assert hist_row["rule_triggered"] == "history_without_primary_indicator"


def test_phase12a_files_remain_frozen():
    assert PHASE12A_EVAL_PATH.exists()
    assert PHASE12A_SUMM_PATH.exists()

    df_12a_sum = pd.read_csv(PHASE12A_SUMM_PATH)
    overall = df_12a_sum[df_12a_sum["condition"] == "overall"].iloc[0]
    assert overall["correct_cases"] == 52
    assert pytest.approx(overall["alignment_rate"], rel=1e-4) == 52 / 81
