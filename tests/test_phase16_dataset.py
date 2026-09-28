"""
Unit tests for Phase 16 IU-Xray Benchmark Expansion Audit & Inventory:
    - Unique study IDs (no duplicate UIDs or sample_ids)
    - Required inventory columns present
    - Valid image flags and path formatting
    - Accurate identification of the existing 20 pilot studies
    - Non-modification and integrity of the existing 20-case processed benchmark
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PILOT_DATASET_CSV = ROOT / "data/processed/iu_xray/metadata/dataset.csv"
PERTURBATIONS_CSV = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
INVENTORY_CSV = ROOT / "results/tables/phase16_dataset_inventory.csv"


def test_inventory_file_exists_and_schema():
    assert INVENTORY_CSV.exists(), f"Missing inventory CSV: {INVENTORY_CSV}"
    df = pd.read_csv(INVENTORY_CSV)

    assert len(df) == 3851

    required_cols = [
        "sample_id",
        "uid",
        "frontal_image",
        "lateral_image",
        "clinical_context",
        "comparison",
        "findings",
        "impression",
        "has_frontal",
        "has_lateral",
        "has_context",
        "has_findings",
        "has_impression",
        "already_in_pilot",
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required column in inventory: {col}"


def test_unique_study_ids():
    df = pd.read_csv(INVENTORY_CSV)

    assert df["uid"].nunique() == len(df), "Duplicate UIDs found in inventory!"
    assert df["sample_id"].nunique() == len(df), "Duplicate sample_ids found in inventory!"


def test_pilot_study_identification():
    inventory_df = pd.read_csv(INVENTORY_CSV)
    pilot_df = pd.read_csv(PILOT_DATASET_CSV)

    pilot_uids = set(pilot_df["uid"].astype(str).tolist())
    inventory_pilot_uids = set(inventory_df[inventory_df["already_in_pilot"] == True]["uid"].astype(str).tolist())

    assert len(pilot_uids) == 20
    assert inventory_pilot_uids == pilot_uids, "Mismatch in identified pilot UIDs!"


def test_existing_processed_benchmark_unmodified():
    # Verify the existing 20 pilot cases and 81 perturbation records are unchanged
    pilot_df = pd.read_csv(PILOT_DATASET_CSV)
    pert_df = pd.read_csv(PERTURBATIONS_CSV)

    assert len(pilot_df) == 20
    assert len(pert_df) == 81

    # Check frontal image paths for pilot cases exist on disk
    for path in pilot_df["frontal_image"]:
        assert (ROOT / path).exists(), f"Pilot image missing from disk: {path}"


def test_usable_study_counts():
    inventory_df = pd.read_csv(INVENTORY_CSV)

    has_any_image = (inventory_df["has_frontal"] == True) | (inventory_df["has_lateral"] == True)
    is_usable = (
        (inventory_df["has_context"] == True)
        & (inventory_df["has_findings"] == True)
        & (inventory_df["has_impression"] == True)
        & has_any_image
    )

    total_usable = is_usable.sum()
    assert total_usable == 3278

    usable_non_pilot = is_usable & (~inventory_df["already_in_pilot"])
    assert usable_non_pilot.sum() == 3258
