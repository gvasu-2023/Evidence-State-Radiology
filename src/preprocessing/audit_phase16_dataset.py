"""
Phase 16: Full IU-Xray/Open-i Dataset Audit & Inventory Builder.

Audits the complete raw IU-Xray dataset present in `external_data/IU-Xray/data/` (3,851 total studies).
Identifies usable studies with complete clinical context, findings, impression, and image availability.
Maps existing 20-study pilot dataset overlap without modifying existing processed benchmark files.

Generates:
    results/tables/phase16_dataset_inventory.csv
"""

from pathlib import Path
import sys
import glob
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DATA_DIR = ROOT / "external_data/IU-Xray/data"
PILOT_DATASET_CSV = ROOT / "data/processed/iu_xray/metadata/dataset.csv"
INVENTORY_OUTPUT_PATH = ROOT / "results/tables/phase16_dataset_inventory.csv"


def check_bytes_present(val: Any) -> bool:
    """Determine whether an image field contains valid non-empty byte or path data."""
    if val is None or pd.isna(val):
        return False
    if isinstance(val, (bytes, bytearray)):
        return len(val) > 0
    if isinstance(val, dict) and "bytes" in val:
        return val["bytes"] is not None and len(val["bytes"]) > 0
    val_str = str(val).strip()
    return len(val_str) > 0 and val_str.lower() not in {"none", "nan", "null"}


def is_non_empty_text(val: Any) -> bool:
    """Determine whether a text field is non-empty and non-null."""
    if pd.isna(val) or val is None:
        return False
    val_str = str(val).strip()
    return len(val_str) > 0 and val_str.lower() not in {"none", "nan", "null"}


def build_phase16_inventory() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    parquet_files = sorted(glob.glob(str(RAW_DATA_DIR / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DATA_DIR}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    raw_df = pd.concat(dfs, ignore_index=True)

    total_studies = len(raw_df)
    unique_uids = raw_df["uid"].nunique()

    if total_studies != unique_uids:
        raise ValueError(f"Duplicate UIDs detected in raw dataset: {total_studies} rows vs {unique_uids} unique UIDs")

    # Pilot study UIDs
    if not PILOT_DATASET_CSV.exists():
        raise FileNotFoundError(f"Pilot dataset file missing: {PILOT_DATASET_CSV}")

    pilot_df = pd.read_csv(PILOT_DATASET_CSV)
    pilot_uids = set(pilot_df["uid"].astype(str).tolist())

    inventory_rows = []

    for _, row in raw_df.iterrows():
        uid_str = str(row["uid"]).strip()
        sample_id = f"IU_{uid_str}"

        already_in_pilot = uid_str in pilot_uids

        context_str = str(row["indication"]).strip() if is_non_empty_text(row["indication"]) else ""
        comparison_str = str(row["comparison"]).strip() if is_non_empty_text(row["comparison"]) else ""
        findings_str = str(row["findings"]).strip() if is_non_empty_text(row["findings"]) else ""
        impression_str = str(row["impression"]).strip() if is_non_empty_text(row["impression"]) else ""

        has_context = bool(context_str != "")
        has_findings = bool(findings_str != "")
        has_impression = bool(impression_str != "")

        has_frontal = check_bytes_present(row["img_frontal"])
        has_lateral = check_bytes_present(row["img_lateral"])

        # Format image path references
        if already_in_pilot:
            pilot_row = pilot_df[pilot_df["uid"].astype(str) == uid_str].iloc[0]
            frontal_path = str(pilot_row["frontal_image"]) if pd.notna(pilot_row["frontal_image"]) else ""
            lateral_path = str(pilot_row["lateral_image"]) if pd.notna(pilot_row["lateral_image"]) else ""
        else:
            frontal_path = f"external_data/IU-Xray/images/IU_{uid_str}_frontal.jpg" if has_frontal else ""
            lateral_path = f"external_data/IU-Xray/images/IU_{uid_str}_lateral.jpg" if has_lateral else ""

        inventory_rows.append(
            {
                "sample_id": sample_id,
                "uid": uid_str,
                "frontal_image": frontal_path,
                "lateral_image": lateral_path,
                "clinical_context": context_str,
                "comparison": comparison_str,
                "findings": findings_str,
                "impression": impression_str,
                "has_frontal": has_frontal,
                "has_lateral": has_lateral,
                "has_context": has_context,
                "has_findings": has_findings,
                "has_impression": has_impression,
                "already_in_pilot": already_in_pilot,
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)

    # Sort inventory numerically by UID
    inventory_df["uid_int"] = inventory_df["uid"].astype(int)
    inventory_df = inventory_df.sort_values("uid_int").drop(columns=["uid_int"]).reset_index(drop=True)

    # Calculate audit statistics
    has_any_image = inventory_df["has_frontal"] | inventory_df["has_lateral"]
    is_usable = (
        inventory_df["has_context"]
        & inventory_df["has_findings"]
        & inventory_df["has_impression"]
        & has_any_image
    )

    inventory_df["is_usable"] = is_usable

    stats = {
        "total_studies": total_studies,
        "unique_uids": unique_uids,
        "pilot_studies_count": len(pilot_uids),
        "pilot_studies_found_in_raw": int(inventory_df["already_in_pilot"].sum()),
        "has_context_count": int(inventory_df["has_context"].sum()),
        "has_findings_count": int(inventory_df["has_findings"].sum()),
        "has_impression_count": int(inventory_df["has_impression"].sum()),
        "has_frontal_count": int(inventory_df["has_frontal"].sum()),
        "has_lateral_count": int(inventory_df["has_lateral"].sum()),
        "has_both_images_count": int((inventory_df["has_frontal"] & inventory_df["has_lateral"]).sum()),
        "has_any_image_count": int(has_any_image.sum()),
        "missing_context_count": int((~inventory_df["has_context"]).sum()),
        "missing_findings_count": int((~inventory_df["has_findings"]).sum()),
        "missing_impression_count": int((~inventory_df["has_impression"]).sum()),
        "total_usable_studies": int(is_usable.sum()),
        "usable_pilot_studies": int((is_usable & inventory_df["already_in_pilot"]).sum()),
        "additional_usable_studies": int((is_usable & (~inventory_df["already_in_pilot"])).sum()),
    }

    # Drop temporary column before exporting
    export_df = inventory_df.drop(columns=["is_usable"])
    return export_df, stats


def main() -> None:
    print("=" * 70)
    print("PHASE 16: IU-XRAY BENCHMARK AUDIT & INVENTORY GENERATION")
    print("=" * 70)

    inventory_df, stats = build_phase16_inventory()

    INVENTORY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(INVENTORY_OUTPUT_PATH, index=False)

    print(f"\nInventory saved to: {INVENTORY_OUTPUT_PATH}")
    print(f"Total Rows Generated: {len(inventory_df)}")
    print(f"Columns: {list(inventory_df.columns)}\n")

    print("--- DATASET STATISTICAL AUDIT ---")
    for k, v in stats.items():
        print(f"  {k:32s}: {v}")

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
