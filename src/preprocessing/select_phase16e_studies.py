"""Exact generator-backed evidentiary screening and Phase 16E selection."""

from pathlib import Path
import re
import sys
from typing import Any, Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.context_completeness import (
    assess_context_completeness,
    parse_context_completeness,
)
from src.preprocessing.generate_phase16c_perturbations import (
    SEED,
    _context_polarity,
    _donor_overlaps_target,
    _explicit_polarity,
    _normalized_clinical_concepts,
    _readable_evidence,
    generate_conflict,
    generate_evidentiary_incomplete,
    generate_irrelevant_assignment,
    generate_syntactic_incomplete,
)

INVENTORY_PATH = ROOT / "results/tables/phase16_dataset_inventory.csv"
PILOT_PATH = ROOT / "data/processed/iu_xray/metadata/dataset.csv"
ORIGINAL_SELECTION_PATH = ROOT / "results/tables/phase16_selected_studies.csv"
CANDIDATE_SCREEN_PATH = ROOT / "results/tables/phase16e_candidate_screen.csv"
SELECTED_OUTPUT_PATH = ROOT / "results/tables/phase16e_selected_studies.csv"

GENERIC_RESIDUAL = "clinical evaluation."
APPROVED_GENERALIZED_RESIDUALS = {
    "respiratory evaluation.",
    "cancer evaluation.",
    "preprocedure evaluation.",
    "infectious disease evaluation.",
}
GENERIC_REFERENCE_PATTERN = re.compile(
    r"\b(?:no acute (?:cardiopulmonary|pulmonary|intrathoracic)|without acute (?:cardiopulmonary|pulmonary))\b",
    re.IGNORECASE,
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _phase16a_usable_pool(inventory: pd.DataFrame) -> pd.DataFrame:
    required = {
        "uid", "sample_id", "clinical_context", "comparison", "findings", "impression",
        "frontal_image", "lateral_image", "already_in_pilot", "has_context",
        "has_findings", "has_impression", "has_frontal", "has_lateral",
    }
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Phase 16A inventory is missing columns: {sorted(missing)}")
    inventory = inventory.copy()
    for col in ("already_in_pilot", "has_context", "has_findings", "has_impression", "has_frontal", "has_lateral"):
        inventory[col] = _as_bool(inventory[col])
    inventory["uid"] = inventory["uid"].astype(int)
    usable = inventory.loc[
        ~inventory["already_in_pilot"]
        & inventory["has_context"]
        & inventory["has_findings"]
        & inventory["has_impression"]
        & (inventory["has_frontal"] | inventory["has_lateral"])
    ].copy()
    if len(usable) != 3258 or usable["uid"].nunique() != 3258:
        raise ValueError(f"Expected 3258 unique Phase 16A usable non-pilot candidates, found {len(usable)}")
    return usable.sort_values("uid").reset_index(drop=True)


def _evidentiary_result(context: str) -> Dict[str, Any]:
    perturbed, removed, method, status = generate_evidentiary_incomplete(context)
    perturbed = str(perturbed).strip()
    removed = str(removed).strip()
    failure = ""

    if status != "success":
        failure = method or "generation_failed"
    elif not removed or not _readable_evidence(removed):
        failure = "removed_text_not_readable_clinical_evidence"
    elif removed.casefold() not in str(context).casefold():
        failure = "removed_text_not_found_in_original_context"
    elif not perturbed:
        failure = "empty_residual_context"
    elif perturbed.casefold() == GENERIC_RESIDUAL:
        failure = "disallowed_generic_clinical_evaluation_residual"
    elif not assess_context_completeness(perturbed):
        failure = "malformed_residual_context"
    elif method == "clause_clinical_removal" and not _readable_evidence(perturbed):
        failure = "clause_removal_did_not_retain_readable_residual_evidence"
    elif method == "readable_concept_generalization" and perturbed.casefold() not in APPROVED_GENERALIZED_RESIDUALS:
        failure = "unrecognized_generalized_residual"
    elif method not in {"clause_clinical_removal", "readable_concept_generalization"}:
        failure = f"unrecognized_generation_method:{method}"

    eligible = not failure and status == "success"
    if eligible:
        return {
            "evidentiary_eligible": True,
            "evidentiary_method": method,
            "evidentiary_perturbed_context": perturbed,
            "removed_evidence_text": removed,
            "failure_reason": "",
            "evidentiary_generation_status": status,
        }

    return {
        "evidentiary_eligible": False,
        "evidentiary_method": method,
        "evidentiary_perturbed_context": perturbed,
        "removed_evidence_text": removed,
        "failure_reason": failure,
        "evidentiary_generation_status": status,
    }


def _state_results(row: pd.Series) -> Dict[str, Any]:
    context = str(row["clinical_context"])
    findings = str(row["findings"])
    impression = str(row["impression"])

    syn_context, syn_method, syn_status = generate_syntactic_incomplete(context)
    syn_parse = parse_context_completeness(syn_context)
    syntactic_ok = (
        syn_status == "success"
        and bool(str(syn_context).strip())
        and not syn_parse.is_complete
        and syn_parse.rule_triggered != "none"
    )

    conflict_context, conflict_target, conflict_source, conflict_status = generate_conflict(
        findings, impression
    )
    target = conflict_target.rsplit("_", 1)[0] if conflict_target else ""
    reference_polarity, checked_source = (
        _explicit_polarity(f"{findings} {impression}", target)
        if target
        else ("AMBIGUOUS", "")
    )
    generated_polarity = _context_polarity(conflict_context, target) if target else "AMBIGUOUS"
    conflict_ok = (
        conflict_status == "success"
        and bool(conflict_target)
        and bool(conflict_source)
        and reference_polarity in {"AFFIRMED", "NEGATED"}
        and generated_polarity in {"AFFIRMED", "NEGATED"}
        and reference_polarity != generated_polarity
        and conflict_source.casefold() == checked_source.casefold()
    )

    return {
        "syntactic_eligible": bool(syntactic_ok),
        "syntactic_method": syn_method,
        "syntactic_context_dry_run": str(syn_context),
        "syntactic_failure_reason": "" if syntactic_ok else "syntactic_generator_or_parser_check_failed",
        "conflicting_eligible": bool(conflict_ok),
        "conflict_method": "explicit_reference_polarity_contradiction" if conflict_ok else "",
        "conflict_target": conflict_target,
        "conflict_context_dry_run": str(conflict_context),
        "conflict_source_text": conflict_source,
        "reference_polarity": reference_polarity,
        "generated_polarity": generated_polarity,
        "conflict_failure_reason": "" if conflict_ok else (conflict_source or "explicit_opposite_polarity_check_failed"),
    }


def screen_candidate_pool(inventory_path: Path = INVENTORY_PATH) -> pd.DataFrame:
    """Dry-run current generators for every Phase 16A usable non-pilot candidate."""
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    candidates = _phase16a_usable_pool(inventory)

    evidentiary_rows = []
    state_rows = []
    for _, row in candidates.iterrows():
        evidentiary_rows.append(_evidentiary_result(str(row["clinical_context"])))
        state_rows.append(_state_results(row))
    screened = pd.concat(
        [candidates.reset_index(drop=True), pd.DataFrame(evidentiary_rows), pd.DataFrame(state_rows)],
        axis=1,
    )

    core_states = screened["evidentiary_eligible"] & screened["syntactic_eligible"] & screened["conflicting_eligible"]
    eligible_donor_rows = screened.loc[core_states]
    donor_concepts = {
        int(row.uid): _normalized_clinical_concepts(str(row.clinical_context))
        for row in eligible_donor_rows.itertuples(index=False)
    }
    reference_concepts = {
        int(row.uid): _normalized_clinical_concepts(f"{row.findings} {row.impression}")
        for row in screened.itertuples(index=False)
    }
    broad_normal_reference = {
        int(row.uid): bool(GENERIC_REFERENCE_PATTERN.search(f"{row.findings} {row.impression}"))
        for row in screened.itertuples(index=False)
    }
    donor_possible = []
    for row in screened.itertuples(index=False):
        uid = int(row.uid)
        if not (row.evidentiary_eligible and row.syntactic_eligible and row.conflicting_eligible):
            donor_possible.append(False)
            continue
        target_concepts = reference_concepts[uid]
        broad_normal = broad_normal_reference[uid]
        possible = any(
            donor_uid != uid
            and not (concepts & target_concepts)
            and not (concepts and broad_normal)
            for donor_uid, concepts in donor_concepts.items()
        )
        donor_possible.append(possible)
    screened["irrelevant_candidate_possible"] = donor_possible
    screened["sufficient_eligible"] = True
    screened["insufficient_eligible"] = True
    screened["all_six_states_eligible"] = (
        screened["sufficient_eligible"]
        & screened["syntactic_eligible"]
        & screened["evidentiary_eligible"]
        & screened["irrelevant_candidate_possible"]
        & screened["conflicting_eligible"]
        & screened["insufficient_eligible"]
    )
    return screened


def _fully_matched_selection(
    candidates: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """Prefer clause-removal cases, then find the first deterministic full donor match."""
    ordered = candidates.copy()
    ordered["uid"] = ordered["uid"].astype(int)
    ordered["evidence_tier"] = ordered["evidentiary_method"].map(
        {"clause_clinical_removal": 0, "readable_concept_generalization": 1}
    )
    ordered = ordered.sort_values(["evidence_tier", "uid"], kind="stable").reset_index(drop=True)
    if len(ordered) < 100:
        raise ValueError(f"Fewer than 100 all-six-state candidates: {len(ordered)}")

    for start in range(len(ordered) - 99):
        subset = ordered.iloc[start : start + 100].drop(columns="evidence_tier").copy()
        subset = subset.sort_values("uid", kind="stable").reset_index(drop=True)
        assignment = generate_irrelevant_assignment(subset, seed=seed)
        if len(assignment) == 100 and len(set(assignment.values())) == 100:
            return subset, assignment, start
    raise ValueError("No deterministic 100-study set in the eligible pool has a complete donor matching")


def select_phase16e_studies(
    screened: pd.DataFrame, seed: int = SEED
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select exactly 100 fully screened studies with deterministic donor feasibility."""
    eligible = screened.loc[screened["all_six_states_eligible"]].copy()
    selected, assignment, selected_window_start = _fully_matched_selection(eligible, seed)
    selected = selected.copy()
    selected["selection_order"] = range(1, 101)
    selected["selection_seed"] = seed
    selected["selection_method"] = "phase16e_tiered_uid_order_with_screened_donor_matching"
    selected["selection_status"] = "FINAL_BENCHMARK_ELIGIBLE"

    summary = {
        "candidate_pool_count": len(screened),
        "evidentiary_success_count": int(screened["evidentiary_eligible"].sum()),
        "evidentiary_failure_count": int((~screened["evidentiary_eligible"]).sum()),
        "all_six_state_success_count": int(screened["all_six_states_eligible"].sum()),
        "selected_count": len(selected),
        "seed": seed,
        "selection_method": "clause-removal tier first, then UID ascending; scan deterministic 100-row windows until screened donor matching is complete",
        "eligible_order_window_start": selected_window_start,
        "donor_assignment_count": len(assignment),
    }
    return selected, summary


def main() -> None:
    screened = screen_candidate_pool()
    if int(screened["all_six_states_eligible"].sum()) < 100:
        raise SystemExit(
            f"STOP: only {int(screened['all_six_states_eligible'].sum())} all-six-state candidates exist; "
            "no selection artifact was written."
        )
    selected, summary = select_phase16e_studies(screened)
    CANDIDATE_SCREEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_columns = [
        "selection_order", "uid", "sample_id", "clinical_context", "comparison", "findings", "impression",
        "frontal_image", "lateral_image", "has_frontal", "has_lateral", "selection_seed",
        "selection_method", "selection_status", "evidentiary_method", "evidentiary_perturbed_context",
        "removed_evidence_text", "all_six_states_eligible",
    ]
    selected[selected_columns].to_csv(SELECTED_OUTPUT_PATH, index=False)
    screened.to_csv(CANDIDATE_SCREEN_PATH, index=False)
    for name, value in summary.items():
        print(f"{name}: {value}")
    print(f"selected_uids: {selected.uid.astype(int).tolist()}")
    print(f"candidate_screen: {CANDIDATE_SCREEN_PATH}")
    print(f"selected_output: {SELECTED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
