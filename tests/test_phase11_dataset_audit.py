from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.audit_phase11_dataset import (
    EXPECTED_CONDITION_COUNTS,
    OUTPUT_PATH as AUDIT_OUTPUT_PATH,
    run_dataset_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_audit_runs_and_passes():
    audit_df = run_dataset_audit()
    assert len(audit_df) == 13
    assert (audit_df["status"] == "PASSED").all()


def test_dataset_audit_csv_exists():
    assert AUDIT_OUTPUT_PATH.exists(), "phase11_dataset_audit.csv missing"
    df = pd.read_csv(AUDIT_OUTPUT_PATH)
    assert len(df) == 13
    assert (df["status"] == "PASSED").all()


def test_expected_condition_counts():
    pert_df = pd.read_csv(ROOT / "data/processed/iu_xray/perturbations/perturbations.csv")
    actual_counts = pert_df["condition"].value_counts().to_dict()
    assert actual_counts == EXPECTED_CONDITION_COUNTS
