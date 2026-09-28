"""Integrity tests for the revised Phase 16D study selection."""

from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.select_phase16d_studies import (
    CANDIDATE_AUDIT_PATH,
    INVENTORY_PATH,
    ORIGINAL_COLUMNS,
    ORIGINAL_SELECTION_PATH,
    OUTPUT_PATH,
    build_phase16d_selection,
)


def _bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _read(path):
    return pd.read_csv(path, keep_default_na=False)


def test_revised_selection_has_exactly_100_unique_studies():
    selected = _read(OUTPUT_PATH)

    assert len(selected) == 100
    assert selected["uid"].astype(int).nunique() == 100
    assert selected["sample_id"].nunique() == 100


def test_revised_selection_excludes_pilot_and_uses_phase16a_pool():
    selected = _read(OUTPUT_PATH)
    inventory = _read(INVENTORY_PATH)
    pilot = _bool(inventory["already_in_pilot"])
    inventory["uid"] = inventory["uid"].astype(int)
    selected_uids = set(selected["uid"].astype(int))
    assert not selected_uids.intersection(
        set(inventory.loc[pilot, "uid"].astype(int))
    )

    usable = (
        ~pilot
        & _bool(inventory["has_context"])
        & _bool(inventory["has_findings"])
        & _bool(inventory["has_impression"])
        & (_bool(inventory["has_frontal"]) | _bool(inventory["has_lateral"]))
    )
    assert len(inventory.loc[usable]) == 3258
    assert selected_uids.issubset(set(inventory.loc[usable, "uid"].astype(int)))


def test_revised_selection_retains_86_removes_14_and_adds_audit_replacements():
    original = _read(ORIGINAL_SELECTION_PATH)
    selected = _read(OUTPUT_PATH)
    audit = _read(CANDIDATE_AUDIT_PATH)
    audit["uid"] = audit["uid"].astype(int)
    audit["currently_selected"] = _bool(audit["currently_selected"])
    audit["current_evidentiary_failure"] = _bool(audit["current_evidentiary_failure"])
    audit["recommended_replacement"] = _bool(audit["recommended_replacement"])

    original_uids = set(original["uid"].astype(int))
    removed = set(audit.loc[audit["current_evidentiary_failure"], "uid"])
    retained = original_uids - removed
    replacements = set(audit.loc[audit["recommended_replacement"], "uid"])
    selected_uids = set(selected["uid"].astype(int))

    assert len(original_uids) == 100
    assert len(removed) == 14
    assert len(retained) == 86
    assert len(replacements) == 14
    assert retained.issubset(selected_uids)
    assert not removed.intersection(selected_uids)
    assert replacements.issubset(selected_uids)
    assert not replacements.intersection(original_uids)
    assert selected_uids == retained | replacements

    # Every retained study's Phase 16B data fields remain byte-for-value equal.
    retained_original = original.loc[original["uid"].astype(int).isin(retained), ORIGINAL_COLUMNS]
    retained_revised = selected.loc[selected["uid"].astype(int).isin(retained), ORIGINAL_COLUMNS]
    assert_frame_equal(
        retained_original.reset_index(drop=True),
        retained_revised.reset_index(drop=True),
        check_dtype=False,
    )


def test_every_selected_uid_passes_phase16d_evidentiary_screen():
    selected = _read(OUTPUT_PATH)
    audit = _read(CANDIDATE_AUDIT_PATH)
    audit["uid"] = audit["uid"].astype(int)
    audit["evidentiary_eligible"] = _bool(audit["evidentiary_eligible"])
    selected_audit = audit.set_index("uid").reindex(selected["uid"].astype(int))

    assert not selected_audit["evidentiary_eligible"].isna().any()
    assert selected_audit["evidentiary_eligible"].all()


def test_replacements_pass_the_stricter_phase16d_screen():
    selected = _read(OUTPUT_PATH)
    audit = _read(CANDIDATE_AUDIT_PATH)
    audit["uid"] = audit["uid"].astype(int)
    for column in (
        "recommended_replacement",
        "evidentiary_eligible",
        "all_states_pre_donor_eligible",
        "frontal_available",
        "syntactic_eligible",
        "conflicting_eligible",
    ):
        audit[column] = _bool(audit[column])
    replacement_uids = set(
        selected.loc[selected["phase16d_selection_status"] == "replacement", "uid"].astype(int)
    )
    replacements = audit.set_index("uid").loc[sorted(replacement_uids)]

    assert replacements["recommended_replacement"].all()
    assert replacements["evidentiary_eligible"].all()
    assert replacements["all_states_pre_donor_eligible"].all()
    assert replacements["frontal_available"].all()
    assert replacements["syntactic_eligible"].all()
    assert replacements["conflicting_eligible"].all()
    assert replacements["evidentiary_method"].eq("clause_clinical_removal").all()


def test_selection_is_deterministic_and_original_file_is_unchanged():
    original_before = ORIGINAL_SELECTION_PATH.read_bytes()
    first, _ = build_phase16d_selection()
    second, _ = build_phase16d_selection()
    original_after = ORIGINAL_SELECTION_PATH.read_bytes()

    assert_frame_equal(first, second)
    assert original_after == original_before

    artifact = _read(OUTPUT_PATH)
    assert_frame_equal(first, artifact, check_dtype=False)

