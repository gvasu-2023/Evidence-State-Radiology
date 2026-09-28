"""Build the revised deterministic Phase 16D study selection.

The 86 selected studies with successful Phase 16C generation are retained. The
14 evidentiary-generation failures are replaced with the exact 14 candidates
flagged by the Phase 16D audit.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVENTORY_PATH = ROOT / "results/tables/phase16_dataset_inventory.csv"
ORIGINAL_SELECTION_PATH = ROOT / "results/tables/phase16_selected_studies.csv"
CANDIDATE_AUDIT_PATH = ROOT / "results/tables/phase16d_candidate_audit.csv"
OUTPUT_PATH = ROOT / "results/tables/phase16d_selected_studies.csv"

SEED = 20260928
ORIGINAL_COLUMNS = [
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
OUTPUT_COLUMNS = ORIGINAL_COLUMNS + [
    "phase16d_selection_status",
    "phase16d_selection_reason",
]


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse CSV booleans consistently, including string-valued inputs."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_phase16d_selection(
    inventory_path: Path = INVENTORY_PATH,
    original_selection_path: Path = ORIGINAL_SELECTION_PATH,
    candidate_audit_path: Path = CANDIDATE_AUDIT_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the revised 100-study selection and a compact audit summary."""
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    original = pd.read_csv(original_selection_path, keep_default_na=False)
    audit = pd.read_csv(candidate_audit_path, keep_default_na=False)

    for column in (
        "already_in_pilot",
        "has_context",
        "has_findings",
        "has_impression",
        "has_frontal",
        "has_lateral",
    ):
        inventory[column] = _as_bool(inventory[column])
    for column in (
        "currently_selected",
        "current_evidentiary_failure",
        "recommended_replacement",
        "evidentiary_eligible",
        "all_states_pre_donor_eligible",
        "frontal_available",
        "syntactic_eligible",
        "conflicting_eligible",
    ):
        audit[column] = _as_bool(audit[column])

    missing_original = set(ORIGINAL_COLUMNS) - set(original.columns)
    if missing_original:
        raise ValueError(f"Original selection is missing columns: {sorted(missing_original)}")
    if len(original) != 100 or original["uid"].astype(int).nunique() != 100:
        raise ValueError("Expected 100 unique studies in the Phase 16B selection")

    original = original.copy()
    original["uid"] = original["uid"].astype(int)
    audit = audit.copy()
    audit["uid"] = audit["uid"].astype(int)
    inventory = inventory.copy()
    inventory["uid"] = inventory["uid"].astype(int)

    failed_uids = set(audit.loc[audit["current_evidentiary_failure"], "uid"])
    if len(failed_uids) != 14 or not failed_uids.issubset(set(original["uid"])):
        raise ValueError("Phase 16D audit must identify 14 failures from the original selection")

    replacements = audit.loc[audit["recommended_replacement"]].copy()
    replacement_uids = set(replacements["uid"])
    if len(replacement_uids) != 14 or replacement_uids & set(original["uid"]):
        raise ValueError("Phase 16D audit must identify 14 unused replacement studies")

    replacement_ok = (
        replacements["evidentiary_eligible"]
        & replacements["all_states_pre_donor_eligible"]
        & replacements["frontal_available"]
        & replacements["syntactic_eligible"]
        & replacements["conflicting_eligible"]
        & replacements["evidentiary_method"].eq("clause_clinical_removal")
        & replacements["evidentiary_status"].eq("success")
        & replacements["evidence_terms"].astype(str).str.strip().ne("")
        & replacements["generated_residual_context"].astype(str).str.strip().ne("")
    )
    if not replacement_ok.all():
        raise ValueError("A recommended replacement fails the strict Phase 16D screen")

    # Re-apply the Phase 16A usability criteria to the source inventory.
    usable_mask = (
        ~inventory["already_in_pilot"]
        & inventory["has_context"]
        & inventory["has_findings"]
        & inventory["has_impression"]
        & (inventory["has_frontal"] | inventory["has_lateral"])
    )
    usable = inventory.loc[usable_mask].copy()
    if len(usable) != 3258:
        raise ValueError(f"Expected 3258 Phase 16A usable candidates, found {len(usable)}")
    if not replacement_uids.issubset(set(usable["uid"])):
        raise ValueError("A recommended replacement is outside the Phase 16A usable pool")

    retained = original.loc[~original["uid"].isin(failed_uids), ORIGINAL_COLUMNS].copy()
    if len(retained) != 86:
        raise ValueError(f"Expected to retain 86 original studies, found {len(retained)}")
    retained["phase16d_selection_status"] = "retained_valid_original"
    retained["phase16d_selection_reason"] = (
        "Retained from Phase 16B; all six Phase 16C generation records succeeded."
    )

    inventory_by_uid = usable.set_index("uid")
    replacements = replacements.sort_values("uid").reset_index(drop=True)
    replacement_rows = []
    for order, audit_row in enumerate(replacements.to_dict("records"), start=101):
        source = inventory_by_uid.loc[int(audit_row["uid"])]
        replacement_rows.append(
            {
                "selection_order": order,
                "uid": int(audit_row["uid"]),
                "sample_id": source["sample_id"],
                "clinical_context": source["clinical_context"],
                "comparison": source["comparison"],
                "findings": source["findings"],
                "impression": source["impression"],
                "frontal_image": source["frontal_image"],
                "lateral_image": source["lateral_image"],
                "has_frontal": bool(source["has_frontal"]),
                "has_lateral": bool(source["has_lateral"]),
                "selection_seed": SEED,
                "selection_method": "phase16d_sorted_uid_topup_no_sampling",
                "phase16d_selection_status": "replacement",
                "phase16d_selection_reason": (
                    "Unused Phase 16A candidate passing the strict Phase 16D screen; "
                    "selected by ascending UID."
                ),
            }
        )

    result = pd.concat(
        [retained, pd.DataFrame(replacement_rows, columns=OUTPUT_COLUMNS)],
        ignore_index=True,
    )
    result["uid"] = result["uid"].astype(int)
    result = result.sort_values("selection_order", kind="stable").reset_index(drop=True)
    if len(result) != 100 or result["uid"].nunique() != 100:
        raise ValueError("Revised selection must contain exactly 100 unique studies")

    selected_audit = audit.set_index("uid").reindex(result["uid"])
    if selected_audit["evidentiary_eligible"].isna().any() or not selected_audit[
        "evidentiary_eligible"
    ].all():
        raise ValueError("Every selected UID must pass the Phase 16D evidentiary screen")

    summary = {
        "original_count": len(original),
        "retained_count": len(retained),
        "removed_uids": sorted(failed_uids),
        "removed_count": len(failed_uids),
        "replacement_uids": sorted(replacement_uids),
        "replacement_count": len(replacement_uids),
        "final_count": len(result),
        "candidate_pool_count": len(usable),
        "seed": SEED,
    }
    return result[OUTPUT_COLUMNS], summary


def main() -> None:
    selected, summary = build_phase16d_selection()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved revised selection: {OUTPUT_PATH}")
    for name, value in summary.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
