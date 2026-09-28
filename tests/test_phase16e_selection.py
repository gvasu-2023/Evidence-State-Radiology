"""Tests for the exact generator-backed Phase 16E candidate screen and selection."""

from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.generate_phase16c_perturbations import (
    SEED,
    _readable_evidence,
    generate_evidentiary_incomplete,
    generate_irrelevant_assignment,
)
from src.preprocessing.select_phase16e_studies import (
    ORIGINAL_SELECTION_PATH,
    SELECTED_OUTPUT_PATH,
    screen_candidate_pool,
    select_phase16e_studies,
)


@pytest.fixture(scope="module")
def screened():
    return screen_candidate_pool()


def _read_output():
    return pd.read_csv(SELECTED_OUTPUT_PATH, keep_default_na=False)


def test_final_selection_has_exactly_100_unique_studies():
    selected = _read_output()
    assert len(selected) == 100
    assert selected["uid"].astype(int).nunique() == 100


def test_screen_covers_full_phase16a_usable_non_pilot_pool(screened):
    assert len(screened) == 3258
    assert screened["uid"].astype(int).nunique() == 3258
    assert int(screened["evidentiary_eligible"].sum()) == 1190
    assert int(screened["all_six_states_eligible"].sum()) == 1150


def test_every_selected_uid_is_in_pool_and_passes_exact_evidentiary_dry_run(screened):
    selected = _read_output()
    screen = screened.set_index(screened["uid"].astype(int))
    for row in selected.itertuples(index=False):
        candidate = screen.loc[int(row.uid)]
        perturbed, removed, method, status = generate_evidentiary_incomplete(
            str(candidate["clinical_context"])
        )
        assert bool(candidate["evidentiary_eligible"])
        assert status == "success"
        assert method == row.evidentiary_method
        assert str(perturbed).strip() == row.evidentiary_perturbed_context
        assert str(removed).strip() == row.removed_evidence_text
        assert row.evidentiary_perturbed_context.strip().casefold() != "clinical evaluation."
        assert _readable_evidence(row.removed_evidence_text)


def test_every_selected_candidate_passes_all_six_state_screen_and_is_non_pilot(screened):
    selected = _read_output()
    screen = screened.set_index(screened["uid"].astype(int))
    for uid in selected["uid"].astype(int):
        candidate = screen.loc[uid]
        assert not bool(candidate["already_in_pilot"])
        assert bool(candidate["all_six_states_eligible"])
        assert bool(candidate["sufficient_eligible"])
        assert bool(candidate["syntactic_eligible"])
        assert bool(candidate["evidentiary_eligible"])
        assert bool(candidate["irrelevant_candidate_possible"])
        assert bool(candidate["conflicting_eligible"])
        assert bool(candidate["insufficient_eligible"])

    assignment = generate_irrelevant_assignment(selected, seed=SEED)
    assert len(assignment) == 100
    assert len(set(assignment.values())) == 100


def test_selection_is_deterministic_and_original_phase16b_file_is_unchanged(screened):
    original_before = ORIGINAL_SELECTION_PATH.read_bytes()
    selected_a, _ = select_phase16e_studies(screened, seed=SEED)
    selected_b, _ = select_phase16e_studies(screened, seed=SEED)
    original_after = ORIGINAL_SELECTION_PATH.read_bytes()

    assert_frame_equal(selected_a, selected_b)
    assert original_after == original_before
    artifact = _read_output()
    assert selected_a["uid"].astype(int).tolist() == artifact["uid"].astype(int).tolist()

