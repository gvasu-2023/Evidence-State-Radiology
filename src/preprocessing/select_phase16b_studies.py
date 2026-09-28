"""
Phase 16B: Deterministic Selection of 100 Additional IU-Xray Underlying Studies.

Selects exactly 100 additional complete, non-pilot underlying studies from the inventory table
`results/tables/phase16_dataset_inventory.csv` using a fixed random seed (SEED = 20260928).

Eligibility Criteria:
    - `already_in_pilot` == False (Excludes the existing 20 pilot studies)
    - `has_context` == True
    - `has_findings` == True
    - `has_impression` == True
    - `has_frontal` == True OR `has_lateral` == True

Outputs:
    results/tables/phase16_selected_studies.csv
"""

from pathlib import Path
import sys
from typing import Dict, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVENTORY_PATH = ROOT / "results/tables/phase16_dataset_inventory.csv"
SELECTED_OUTPUT_PATH = ROOT / "results/tables/phase16_selected_studies.csv"

SEED = 20260928
SELECTION_METHOD = "pandas_sample_seed_20260928_sorted_uid"


def load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Missing dataset inventory: {INVENTORY_PATH}")
    inv_df = pd.read_csv(INVENTORY_PATH)
    if len(inv_df) != 3851:
        raise ValueError(f"Expected 3851 inventory rows, found {len(inv_df)}")
    return inv_df


def select_phase16b_studies(
    seed: int = SEED, n_select: int = 100
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    inv_df = load_inventory()

    # Filter out pilot studies
    non_pilot_df = inv_df[inv_df["already_in_pilot"] == False].copy()
    n_before_exclusion = len(inv_df)
    n_pilot_excluded = int(inv_df["already_in_pilot"].sum())
    n_after_exclusion = len(non_pilot_df)

    # Filter complete usable studies
    usable_mask = (
        (non_pilot_df["has_context"] == True)
        & (non_pilot_df["has_findings"] == True)
        & (non_pilot_df["has_impression"] == True)
        & ((non_pilot_df["has_frontal"] == True) | (non_pilot_df["has_lateral"] == True))
    )

    candidate_df = non_pilot_df[usable_mask].copy()
    n_usable_candidates = len(candidate_df)

    if n_usable_candidates < n_select:
        raise ValueError(f"Requested {n_select} studies, but only {n_usable_candidates} usable candidates available.")

    # Sort deterministically by integer UID before sampling
    candidate_df["uid_int"] = candidate_df["uid"].astype(int)
    sorted_candidate_df = candidate_df.sort_values("uid_int").reset_index(drop=True)

    # Deterministic sampling with seed
    sampled_df = sorted_candidate_df.sample(n=n_select, random_state=seed).copy()

    # Sort sampled studies back by uid_int for neat presentation
    sampled_df = sampled_df.sort_values("uid_int").reset_index(drop=True)

    # Assign selection order (1..100)
    sampled_df["selection_order"] = range(1, len(sampled_df) + 1)
    sampled_df["selection_seed"] = seed
    sampled_df["selection_method"] = SELECTION_METHOD

    # Fill NA comparison text with empty string
    sampled_df["comparison"] = sampled_df["comparison"].fillna("").astype(str)

    output_cols = [
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

    selected_export_df = sampled_df[output_cols].copy()

    stats = {
        "n_candidates_before_selection": n_before_exclusion,
        "n_pilot_excluded": n_pilot_excluded,
        "n_candidates_after_exclusion": n_after_exclusion,
        "n_usable_candidates": n_usable_candidates,
        "n_selected": len(selected_export_df),
        "seed": seed,
        "has_frontal_count": int((selected_export_df["has_frontal"] == True).sum()),
        "has_lateral_count": int((selected_export_df["has_lateral"] == True).sum()),
        "has_both_images_count": int(((selected_export_df["has_frontal"] == True) & (selected_export_df["has_lateral"] == True)).sum()),
    }

    return selected_export_df, stats


def main() -> None:
    print("=" * 70)
    print("PHASE 16B: DETERMINISTIC SELECTION OF 100 ADDITIONAL IU-XRAY STUDIES")
    print("=" * 70)

    selected_df, stats = select_phase16b_studies()

    SELECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(SELECTED_OUTPUT_PATH, index=False)

    print(f"\nSelected studies saved to: {SELECTED_OUTPUT_PATH}")
    print(f"Total Selected Rows: {len(selected_df)}")

    print("\n--- SELECTION AUDIT STATISTICS ---")
    for k, v in stats.items():
        print(f"  {k:32s}: {v}")

    print("\nPhase 16B selection complete.")


if __name__ == "__main__":
    main()
