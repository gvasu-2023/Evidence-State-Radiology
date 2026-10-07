from pathlib import Path

import pandas as pd
import pytest

from demo.data import (
    DemoDataError,
    find_quick_demo_sample,
    get_frozen_pair,
    get_study_context,
    load_generation_records,
    resolve_image_path,
)


def _records(tmp_path: Path) -> Path:
    path = tmp_path / "generation.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "study-a",
                "uid": 10,
                "condition": "sufficient",
                "evidence_state": "sufficient",
                "perturbed_context": "chest pain",
                "frontal_image": "missing.jpg",
                "lambda": lambda_value,
                "report": "saved report",
                "num_tokens": 2,
                "runtime_seconds": 1.2,
                "status": "success",
            }
            for lambda_value in (0.0, 0.25)
        ]
    ).to_csv(path, index=False)
    return path


def test_load_and_select_frozen_pair_without_substituting_missing_lambda(tmp_path):
    records = load_generation_records(_records(tmp_path))
    pair = get_frozen_pair(records, "study-a", "sufficient")

    assert set(pair) == {0.0, 0.25}
    assert get_study_context(records, "study-a", "sufficient") == "chest pain"
    assert get_frozen_pair(records, "unknown", "sufficient") == {}
    assert find_quick_demo_sample(records, "sufficient") == "study-a"


def test_context_uses_available_lambda_when_zero_is_absent(tmp_path):
    path = _records(tmp_path)
    records = pd.read_csv(path).loc[lambda frame: frame["lambda"].eq(0.25)]
    records.to_csv(path, index=False)
    records = load_generation_records(path)

    assert get_study_context(records, "study-a", "sufficient") == "chest pain"
    assert set(get_frozen_pair(records, "study-a", "sufficient")) == {0.25}


def test_malformed_generation_file_is_reported(tmp_path):
    path = tmp_path / "malformed.csv"
    pd.DataFrame({"sample_id": ["study-a"]}).to_csv(path, index=False)

    with pytest.raises(DemoDataError, match="missing required columns"):
        load_generation_records(path)


def test_image_resolution_uses_exact_existing_file_only(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image placeholder")

    assert resolve_image_path("image.jpg", root=tmp_path) == image.resolve()
    assert resolve_image_path("absent.jpg", root=tmp_path) is None
    assert resolve_image_path("", root=tmp_path) is None
