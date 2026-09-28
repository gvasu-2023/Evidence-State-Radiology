"""
Unit tests for Phase 16B Deterministic Selection of 100 Additional IU-Xray Studies:
    - Exactly 100 selected studies
    - All selected UIDs are unique
    - No selected UID belongs to the existing 20 pilot studies
    - All selected studies satisfy Phase 16 usability requirements
    - Selection is 100% reproducible with the same seed
    - Changing the seed produces a valid distinct selection
    - Required output columns exist
    - Source inventory file is not modified
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.select_phase16b_studies import (
    select_phase16b_studies,
    INVENTORY_PATH,
    SELECTED_OUTPUT_PATH,
    SEED,
)

PILOT_DATASET_CSV = ROOT / "data/processed/iu_xray/metadata/dataset.csv"


def test_selected_studies_output_file_and_schema():
    assert SELECTED_OUTPUT_PATH.exists(), f"Missing selected studies file: {SELECTED_OUTPUT_PATH}"
    df = pd.read_csv(SELECTED_OUTPUT_PATH)

    assert len(df) == 100, f"Expected 100 selected studies, found {len(df)}"

    required_cols = [
        "selection_order",
        "uid",
        "sample_id",
        "clinical_context",
        "comparison",
        "findings",
        "impression",
        "frontal_image",
        "lateral_image",
        "has_frontal",
        "has_lateral",
        "selection_seed",
        "selection_method",
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required column in selected studies CSV: {col}"


def test_unique_selected_uids():
    df = pd.read_csv(SELECTED_OUTPUT_PATH)

    assert df["uid"].nunique() == 100, "Duplicate UIDs found in selected studies!"
    assert df["sample_id"].nunique() == 100, "Duplicate sample_ids found in selected studies!"


def test_no_pilot_study_in_selected():
    selected_df = pd.read_csv(SELECTED_OUTPUT_PATH)
    pilot_df = pd.read_csv(PILOT_DATASET_CSV)

    pilot_uids = set(pilot_df["uid"].astype(str).tolist())
    selected_uids = set(selected_df["uid"].astype(str).tolist())

    overlap = selected_uids.intersection(pilot_uids)
    assert len(overlap) == 0, f"Selected studies contain pilot UIDs: {overlap}"


def test_usability_requirements_satisfied():
    selected_df = pd.read_csv(SELECTED_OUTPUT_PATH)

    # Check non-empty text fields and at least one image
    for _, row in selected_df.iterrows():
        assert str(row["clinical_context"]).strip() != "", f"Empty context in UID {row['uid']}"
        assert str(row["findings"]).strip() != "", f"Empty findings in UID {row['uid']}"
        assert str(row["impression"]).strip() != "", f"Empty impression in UID {row['uid']}"
        assert bool(row["has_frontal"]) or bool(row["has_lateral"]), f"No image available in UID {row['uid']}"


def test_reproducibility_with_same_seed():
    df1, _ = select_phase16b_studies(seed=SEED, n_select=100)
    df2, _ = select_phase16b_studies(seed=SEED, n_select=100)

    pd.testing.assert_frame_equal(df1, df2)


def test_changing_seed_produces_valid_distinct_selection():
    df_default, _ = select_phase16b_studies(seed=SEED, n_select=100)
    df_alt, _ = select_phase16b_studies(seed=12345, n_select=100)

    assert len(df_alt) == 100
    assert df_alt["uid"].nunique() == 100

    # Ensure different seeds produce different selections
    assert set(df_default["uid"].tolist()) != set(df_alt["uid"].tolist())


def test_inventory_unmodified():
    inventory_df = pd.read_csv(INVENTORY_PATH)
    assert len(inventory_df) == 3851
    assert inventory_df["already_in_pilot"].sum() == 20
