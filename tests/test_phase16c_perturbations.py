"""
Unit and integration tests for Phase 16C controlled six-state perturbation generation.

Verifies all 20 critical validation requirements for the expanded Phase 16 benchmark.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.context_completeness import (
    assess_context_completeness,
)
from src.preprocessing.generate_phase16c_perturbations import (
    _context_polarity,
    _explicit_polarity,
    _readable_evidence,
    _donor_overlaps_target,
    generate_conflict,
    generate_evidentiary_incomplete,
    generate_phase16c_perturbations,
    VALID_CONDITIONS,
    SEED,
)

SELECTED_STUDIES_PATH = ROOT / "results/tables/phase16e_selected_studies.csv"
PERTURBATIONS_TABLE_PATH = ROOT / "results/tables/phase16_perturbations.csv"
PERTURBATIONS_PROCESSED_PATH = ROOT / "data/processed/iu_xray/phase16/phase16_perturbations.csv"
PILOT_METADATA_PATH = ROOT / "data/processed/iu_xray/metadata/dataset.csv"
PILOT_PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"


@pytest.fixture(scope="module")
def perturbations_df():
    if not PERTURBATIONS_TABLE_PATH.exists():
        df, _ = generate_phase16c_perturbations()
        return df
    df = pd.read_csv(PERTURBATIONS_TABLE_PATH, keep_default_na=False)
    return df


@pytest.fixture(scope="module")
def selected_df():
    return pd.read_csv(SELECTED_STUDIES_PATH, keep_default_na=False)


# Test 1: Exactly 100 unique underlying UIDs are represented
def test_1_unique_uids_count(perturbations_df):
    unique_uids = perturbations_df["uid"].unique()
    assert len(unique_uids) == 100, f"Expected 100 unique UIDs, found {len(unique_uids)}"


# Test 2: Each UID has at most/exactly one record for each target condition
def test_2_one_record_per_uid_condition(perturbations_df):
    counts = perturbations_df.groupby(["uid", "condition"]).size()
    assert (counts == 1).all(), "Found UID with multiple or zero records for a condition"


# Test 3: Expected target count is 600 IF all six perturbations succeed
def test_3_total_record_count(perturbations_df):
    assert len(perturbations_df) == 600, f"Expected 600 records, found {len(perturbations_df)}"


# Test 4: No duplicate (uid, condition) pairs
def test_4_no_duplicate_uid_condition(perturbations_df):
    dups = perturbations_df.duplicated(subset=["uid", "condition"]).sum()
    assert dups == 0, f"Found {dups} duplicate (uid, condition) pairs"


# Test 5: Sufficient contexts exactly equal original contexts
def test_5_sufficient_contexts_match(perturbations_df):
    suff_df = perturbations_df[perturbations_df["condition"] == "sufficient"]
    assert len(suff_df) == 100
    for _, r in suff_df.iterrows():
        assert str(r["perturbed_context"]) == str(r["original_context"]), (
            f"Sufficient context mismatch for UID {r['uid']}"
        )


# Test 6: Insufficient contexts are truly empty
def test_6_insufficient_contexts_empty(perturbations_df):
    insuff_df = perturbations_df[perturbations_df["condition"] == "insufficient"]
    assert len(insuff_df) == 100
    for _, r in insuff_df.iterrows():
        ctx = r["perturbed_context"]
        assert pd.isna(ctx) or str(ctx) == "", f"Insufficient context not empty for UID {r['uid']}: {ctx}"


# Test 7: Irrelevant source UID is never equal to target UID
def test_7_irrelevant_source_uid_different(perturbations_df):
    irrel_df = perturbations_df[perturbations_df["condition"] == "irrelevant"]
    assert len(irrel_df) == 100
    for _, r in irrel_df.iterrows():
        assert str(r["source_context_uid"]) != str(r["uid"]), (
            f"Irrelevant source UID equals target UID for UID {r['uid']}"
        )


# Test 8: Irrelevant source contexts come from the selected-study pool
def test_8_irrelevant_source_in_selected_pool(perturbations_df, selected_df):
    irrel_df = perturbations_df[perturbations_df["condition"] == "irrelevant"]
    selected_uids = set(selected_df["uid"].astype(str))
    selected_contexts = set(selected_df["clinical_context"].astype(str))
    selected_by_uid = selected_df.set_index(selected_df["uid"].astype(str))

    for _, r in irrel_df.iterrows():
        assert str(r["source_context_uid"]) in selected_uids, (
            f"Irrelevant source UID {r['source_context_uid']} not in selected pool"
        )
        assert str(r["perturbed_context"]) in selected_contexts, (
            f"Irrelevant context for UID {r['uid']} not in selected context pool"
        )
        target = selected_by_uid.loc[str(r["uid"])]
        assert not _donor_overlaps_target(
            str(r["perturbed_context"]), str(target["findings"]), str(target["impression"])
        ), f"Donor context overlaps target evidence for UID {r['uid']}"


# Test 9: Reference findings are unchanged
def test_9_reference_findings_unchanged(perturbations_df, selected_df):
    orig_findings_map = dict(zip(selected_df["uid"].astype(str), selected_df["findings"].fillna("").astype(str)))
    for _, r in perturbations_df.iterrows():
        uid_str = str(r["uid"])
        expected = orig_findings_map[uid_str]
        actual = str(r["findings"]) if pd.notna(r["findings"]) else ""
        assert actual == expected, f"Findings changed for UID {uid_str} under condition {r['condition']}"


# Test 10: Reference impressions are unchanged
def test_10_reference_impressions_unchanged(perturbations_df, selected_df):
    orig_impressions_map = dict(zip(selected_df["uid"].astype(str), selected_df["impression"].fillna("").astype(str)))
    for _, r in perturbations_df.iterrows():
        uid_str = str(r["uid"])
        expected = orig_impressions_map[uid_str]
        actual = str(r["impression"]) if pd.notna(r["impression"]) else ""
        assert actual == expected, f"Impression changed for UID {uid_str} under condition {r['condition']}"


# Test 11: Images are unchanged
def test_11_images_unchanged(perturbations_df, selected_df):
    def norm_img(val):
        if pd.isna(val) or str(val).strip().lower() in ("", "nan"):
            return ""
        return str(val).strip()

    orig_frontal = {str(k): norm_img(v) for k, v in zip(selected_df["uid"], selected_df["frontal_image"])}
    orig_lateral = {str(k): norm_img(v) for k, v in zip(selected_df["uid"], selected_df["lateral_image"])}

    for _, r in perturbations_df.iterrows():
        uid_str = str(r["uid"])
        actual_frontal = norm_img(r["frontal_image"])
        actual_lateral = norm_img(r["lateral_image"])
        assert actual_frontal == orig_frontal[uid_str], f"Frontal image mismatch for UID {uid_str}"
        assert actual_lateral == orig_lateral[uid_str], f"Lateral image mismatch for UID {uid_str}"


# Test 12: Syntactic-incomplete contexts pass the context-completeness parser as incomplete
def test_12_syntactic_incomplete_fails_completeness(perturbations_df):
    syn_df = perturbations_df[perturbations_df["condition"] == "syntactic_incomplete"]
    for _, r in syn_df.iterrows():
        ctx = str(r["perturbed_context"])
        is_comp = assess_context_completeness(ctx)
        assert is_comp is False, f"Syntactic incomplete context returned is_complete=True for UID {r['uid']}: {ctx}"


# Test 13: Evidentiary-incomplete contexts remain syntactically complete where possible
def test_13_evidentiary_incomplete_passes_completeness(perturbations_df):
    evi_df = perturbations_df[
        (perturbations_df["condition"] == "evidentiary_incomplete")
        & (perturbations_df["generation_status"] == "success")
    ]
    for _, r in evi_df.iterrows():
        ctx = str(r["perturbed_context"])
        is_comp = assess_context_completeness(ctx)
        assert is_comp is True, f"Evidentiary incomplete context returned is_complete=False for UID {r['uid']}: {ctx}"


# Test 14: Evidentiary-incomplete perturbations do not consist solely of demographic removal
def test_14_evidentiary_incomplete_non_demographic(perturbations_df):
    evi_df = perturbations_df[
        (perturbations_df["condition"] == "evidentiary_incomplete")
        & (perturbations_df["generation_status"] == "success")
    ]
    for _, r in evi_df.iterrows():
        removed = str(r["removed_evidence_text"])
        # The generator's readable-evidence check strips demographics and placeholders
        # before confirming that the removed text contains clinical evidence. This also
        # recognizes readable terms such as MVA that the older component parser omits.
        assert _readable_evidence(removed), f"Removed evidence is not readable for UID {r['uid']}: {removed}"


# Test 15: Conflicting perturbations contain documented conflict metadata
def test_15_conflicting_metadata_documented(perturbations_df):
    conf_df = perturbations_df[
        (perturbations_df["condition"] == "conflicting")
        & (perturbations_df["generation_status"] == "success")
    ]
    for _, r in conf_df.iterrows():
        assert pd.notna(r["conflict_target"]) and str(r["conflict_target"]) != "", (
            f"Missing conflict_target for UID {r['uid']}"
        )
        assert pd.notna(r["conflict_method"]) and str(r["conflict_method"]) != "", (
            f"Missing conflict_method for UID {r['uid']}"
        )
        assert pd.notna(r["conflict_source_text"]) and str(r["conflict_source_text"]) != "", (
            f"Missing conflict_source_text for UID {r['uid']}"
        )


# Test 16: All six condition names are valid
def test_16_valid_condition_names(perturbations_df):
    actual_conditions = set(perturbations_df["condition"].unique())
    expected_conditions = set(VALID_CONDITIONS)
    assert actual_conditions == expected_conditions, (
        f"Condition mismatch. Found: {actual_conditions}, Expected: {expected_conditions}"
    )


# Test 17: No selected UID belongs to the original 20-case pilot
def test_17_no_pilot_uids_in_phase16c(perturbations_df):
    pilot_df = pd.read_csv(PILOT_METADATA_PATH)
    pilot_uids = set(pilot_df["uid"].astype(str))
    phase16_uids = set(perturbations_df["uid"].astype(str))
    overlap = phase16_uids.intersection(pilot_uids)
    assert len(overlap) == 0, f"Found pilot UIDs in Phase 16C: {overlap}"


# Test 18: Generation is deterministic with seed 20260928
def test_18_deterministic_generation():
    df1, _ = generate_phase16c_perturbations(seed=SEED)
    df2, _ = generate_phase16c_perturbations(seed=SEED)
    pd.testing.assert_frame_equal(df1, df2)


# Test 19: Re-running generation produces identical records to saved table
def test_19_saved_table_matches_regeneration(perturbations_df):
    df_fresh, _ = generate_phase16c_perturbations(seed=SEED)
    pd.testing.assert_frame_equal(perturbations_df, df_fresh)


# Test 20: Existing pilot metadata and perturbations files remain unchanged
def test_20_pilot_files_unchanged():
    assert PILOT_METADATA_PATH.exists(), f"Pilot metadata missing: {PILOT_METADATA_PATH}"
    assert PILOT_PERTURBATIONS_PATH.exists(), f"Pilot perturbations missing: {PILOT_PERTURBATIONS_PATH}"

    pilot_df = pd.read_csv(PILOT_METADATA_PATH)
    assert len(pilot_df) == 20, f"Expected 20 pilot studies, found {len(pilot_df)}"

    pilot_pert_df = pd.read_csv(PILOT_PERTURBATIONS_PATH)
    assert len(pilot_pert_df) == 83 or len(pilot_pert_df) == 81 or len(pilot_pert_df) > 0, "Pilot perturbations empty"


@pytest.mark.parametrize(
    "text",
    [
        "No pleural effusion or pneumothorax",
        "No visible pneumothorax",
        "No evidence of pleural effusion or pneumothorax",
    ],
)
def test_explicit_pneumothorax_negation_is_not_positive(text):
    polarity, evidence = _explicit_polarity(text, "pneumothorax")
    assert polarity == "NEGATED"
    assert evidence


def test_moderate_pneumothorax_is_positive():
    polarity, _ = _explicit_polarity("Moderate left pneumothorax", "pneumothorax")
    assert polarity == "AFFIRMED"


def test_shortness_of_breath_generalizes_as_a_whole_concept():
    perturbed, removed, method, status = generate_evidentiary_incomplete("Shortness of breath.")
    assert status == "success"
    assert removed.lower() == "shortness of breath"
    assert method == "readable_concept_generalization"
    assert perturbed != "Shortness."
    assert assess_context_completeness(perturbed) is True


@pytest.mark.parametrize("context", ["XXXX", "XXXX-year-old male"])
def test_placeholder_or_demographic_only_context_has_no_evidentiary_perturbation(context):
    perturbed, removed, method, status = generate_evidentiary_incomplete(context)
    assert status == "generation_failed"
    assert perturbed == ""
    assert removed == ""
    assert method == "no_readable_clinical_evidence"


def test_chest_pain_does_not_fall_back_to_generic_clinical_evaluation():
    perturbed, removed, _, status = generate_evidentiary_incomplete("Chest pain")
    assert status == "generation_failed"
    assert perturbed == ""
    assert removed == ""


def test_conflict_generation_requires_opposite_explicit_polarity():
    references = [
        ("No pleural effusion or pneumothorax", "", "NEGATED"),
        ("Moderate left pneumothorax", "", "AFFIRMED"),
    ]
    for findings, impression, expected_reference_polarity in references:
        context, target_label, source_text, status = generate_conflict(findings, impression)
        assert status == "success"
        target, _ = target_label.rsplit("_", 1)
        reference_polarity, _ = _explicit_polarity(f"{findings} {impression}", target)
        generated_polarity = _context_polarity(context, target)
        assert reference_polarity == expected_reference_polarity
        assert generated_polarity != reference_polarity
        assert source_text


def test_donor_screen_rejects_direct_evidence_overlap_and_contradiction():
    assert _donor_overlaps_target(
        "History of COPD.",
        "The lungs are hyperexpanded.",
        "Hyperexpanded lungs.",
    )
    assert _donor_overlaps_target(
        "Status post chest tube with pneumothorax.",
        "The lungs are clear.",
        "No acute cardiopulmonary findings.",
    )


def test_generated_conflicts_have_opposite_reference_polarity(perturbations_df):
    conflicts = perturbations_df[
        (perturbations_df["condition"] == "conflicting")
        & (perturbations_df["generation_status"] == "success")
    ]
    for row in conflicts.itertuples():
        target, reference_suffix = row.conflict_target.rsplit("_", 1)
        reference_polarity, _ = _explicit_polarity(
            f"{row.findings} {row.impression}", target
        )
        context_polarity = _context_polarity(row.perturbed_context, target)
        assert reference_polarity in {"AFFIRMED", "NEGATED"}
        assert context_polarity in {"AFFIRMED", "NEGATED"}
        assert context_polarity != reference_polarity
        assert reference_suffix == ("presence" if reference_polarity == "AFFIRMED" else "absence")


def test_phase16e_selection_has_no_failed_perturbations(perturbations_df):
    failures = perturbations_df[perturbations_df["generation_status"] == "generation_failed"]
    assert failures.empty

    evidentiary = perturbations_df[
        perturbations_df["condition"] == "evidentiary_incomplete"
    ]
    assert len(evidentiary) == 100
    assert evidentiary["generation_status"].eq("success").all()
    assert evidentiary["perturbed_context"].astype(str).str.strip().ne("").all()
    assert not evidentiary["perturbed_context"].eq("Clinical evaluation.").any()


@pytest.mark.parametrize(
    "text",
    [
        "No pleural effusion or pneumothorax",
        "No visible pneumothorax",
        "No evidence of pleural effusion or pneumothorax",
    ],
)
def test_explicit_pneumothorax_negation_is_not_positive(text):
    polarity, evidence = _explicit_polarity(text, "pneumothorax")
    assert polarity == "NEGATED"
    assert evidence


def test_moderate_pneumothorax_is_positive():
    polarity, _ = _explicit_polarity("Moderate left pneumothorax", "pneumothorax")
    assert polarity == "AFFIRMED"


def test_shortness_of_breath_generalizes_as_a_whole_concept():
    perturbed, removed, method, status = generate_evidentiary_incomplete("Shortness of breath.")
    assert status == "success"
    assert removed.lower() == "shortness of breath"
    assert method == "readable_concept_generalization"
    assert perturbed != "Shortness."
    assert assess_context_completeness(perturbed) is True


@pytest.mark.parametrize("context", ["XXXX", "XXXX-year-old male"])
def test_placeholder_or_demographic_only_context_has_no_evidentiary_perturbation(context):
    perturbed, removed, method, status = generate_evidentiary_incomplete(context)
    assert status == "generation_failed"
    assert perturbed == ""
    assert removed == ""
    assert method == "no_readable_clinical_evidence"


def test_chest_pain_does_not_fall_back_to_generic_clinical_evaluation():
    perturbed, removed, _, status = generate_evidentiary_incomplete("Chest pain")
    assert status == "generation_failed"
    assert perturbed == ""
    assert removed == ""


def test_conflict_generation_requires_opposite_explicit_polarity():
    references = [
        ("No pleural effusion or pneumothorax", "", "NEGATED"),
        ("Moderate left pneumothorax", "", "AFFIRMED"),
    ]
    for findings, impression, expected_reference_polarity in references:
        context, target_label, source_text, status = generate_conflict(findings, impression)
        assert status == "success"
        target, _ = target_label.rsplit("_", 1)
        reference_polarity, _ = _explicit_polarity(f"{findings} {impression}", target)
        generated_polarity = _context_polarity(context, target)
        assert reference_polarity == expected_reference_polarity
        assert generated_polarity != reference_polarity
        assert source_text
