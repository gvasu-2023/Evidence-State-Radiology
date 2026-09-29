from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.preprocessing.prepare_phase17_vlm_eligibility import (
    BENCHMARK_PATH,
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_SELECTION_SHA256,
    EXCLUDED_UIDS,
    OUTPUT_PATH,
    ROOT,
    SELECTION_PATH,
    sha256_file,
)


@pytest.fixture(scope="module")
def eligibility():
    return pd.read_csv(OUTPUT_PATH)


def test_eligibility_artifact_has_600_rows_and_576_eligible(eligibility):
    assert len(eligibility) == 600
    assert eligibility["eligible_for_vlm"].sum() == 576
    assert (~eligibility["eligible_for_vlm"]).sum() == 24


def test_eligible_records_are_balanced_across_six_conditions(eligibility):
    counts = eligibility.loc[eligibility.eligible_for_vlm, "condition"].value_counts().to_dict()
    assert counts == {
        "sufficient": 96,
        "syntactic_incomplete": 96,
        "evidentiary_incomplete": 96,
        "irrelevant": 96,
        "conflicting": 96,
        "insufficient": 96,
    }


def test_only_the_four_unavailable_uids_are_excluded_six_times_each(eligibility):
    excluded = eligibility.loc[~eligibility.eligible_for_vlm]
    assert set(excluded.uid) == EXCLUDED_UIDS
    assert excluded.groupby("uid").size().to_dict() == {uid: 6 for uid in EXCLUDED_UIDS}
    assert excluded.exclusion_reason.eq("frontal_image_unavailable").all()
    assert not excluded.image_available.any()
    assert not excluded.image_readable.any()
    assert not excluded.image_verify_ok.any()


def test_every_eligible_image_exists_and_passes_integrity_checks(eligibility):
    eligible = eligibility.loc[eligibility.eligible_for_vlm]
    unique_images = eligible.drop_duplicates("image_path")
    assert len(unique_images) == 96
    for row in unique_images.itertuples(index=False):
        image_path = ROOT / Path(row.image_path)
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        assert row.image_available
        assert row.image_readable
        assert row.image_verify_ok
        assert row.width > 0 and row.height > 0
        with Image.open(image_path) as image:
            image.verify()


def test_eligible_records_use_frontal_not_lateral_images(eligibility):
    eligible = eligibility.loc[eligibility.eligible_for_vlm]
    assert eligible.image_path.str.lower().str.contains("_frontal.").all()
    assert not eligible.image_path.str.lower().str.contains("_lateral.").any()
    assert (eligible.image_path.str.lower() != eligible.lateral_image_path.fillna("").str.lower()).all()


def test_frozen_benchmark_inputs_remain_unchanged():
    assert sha256_file(BENCHMARK_PATH) == EXPECTED_BENCHMARK_SHA256
    assert sha256_file(SELECTION_PATH) == EXPECTED_SELECTION_SHA256
