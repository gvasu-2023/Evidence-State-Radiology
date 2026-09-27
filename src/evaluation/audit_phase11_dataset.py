"""
Dataset Integrity and Consistency Audit for Phase 11.

Verifies:
- 81 perturbation records, 81 baseline records, 81 gated records
- Unique (sample_id, condition) pairs in all datasets
- Five expected conditions with exact record counts:
    sufficient=17, incomplete=9, irrelevant=20, conflicting=15, insufficient=20
- Complete perturbation -> baseline pairing
- Complete perturbation -> gated pairing
- Image paths exist for all records
- No empty gated reports
- All gated records have was_model_inference_run == False

Outputs:
    results/tables/phase11_dataset_audit.csv
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_PATH = ROOT / "results/tables/baseline_generation_results.csv"
GATED_PATH = ROOT / "results/tables/gated_generation_results.csv"
OUTPUT_PATH = ROOT / "results/tables/phase11_dataset_audit.csv"

EXPECTED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]

EXPECTED_CONDITION_COUNTS = {
    "sufficient": 17,
    "incomplete": 9,
    "irrelevant": 20,
    "conflicting": 15,
    "insufficient": 20,
}


def run_dataset_audit() -> pd.DataFrame:
    """Perform comprehensive audit checks across perturbation, baseline, and gated datasets."""

    checks = []

    def add_check(
        check_id: str,
        check_name: str,
        passed: bool,
        expected: str,
        actual: str,
        details: str,
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "check_name": check_name,
                "status": "PASSED" if passed else "FAILED",
                "expected": expected,
                "actual": actual,
                "details": details,
            }
        )

    # 1. Input files exist
    files_exist = (
        PERTURBATIONS_PATH.exists()
        and BASELINE_PATH.exists()
        and GATED_PATH.exists()
    )
    add_check(
        "AUDIT_01",
        "Input Files Existence",
        files_exist,
        "All 3 files exist",
        f"pert={PERTURBATIONS_PATH.exists()}, base={BASELINE_PATH.exists()}, gated={GATED_PATH.exists()}",
        "Verify raw perturbation and baseline/gated table availability",
    )

    if not files_exist:
        return pd.DataFrame(checks)

    pert_df = pd.read_csv(PERTURBATIONS_PATH)
    base_df = pd.read_csv(BASELINE_PATH)
    gated_df = pd.read_csv(GATED_PATH)

    # 2. Record counts
    add_check(
        "AUDIT_02",
        "Perturbation Record Count",
        len(pert_df) == 81,
        "81",
        str(len(pert_df)),
        "Verify 81 controlled perturbation records",
    )

    add_check(
        "AUDIT_03",
        "Baseline Record Count",
        len(base_df) == 81,
        "81",
        str(len(base_df)),
        "Verify 81 baseline generation records",
    )

    add_check(
        "AUDIT_04",
        "Gated Record Count",
        len(gated_df) == 81,
        "81",
        str(len(gated_df)),
        "Verify 81 gated generation records",
    )

    # 3. Unique (sample_id, condition) pairs
    pert_pairs = set(zip(pert_df["sample_id"].astype(str), pert_df["condition"].astype(str)))
    base_pairs = set(zip(base_df["sample_id"].astype(str), base_df["condition"].astype(str)))
    gated_pairs = set(
        zip(gated_df["sample_id"].astype(str), gated_df["experimental_condition"].astype(str))
    )

    add_check(
        "AUDIT_05",
        "Perturbation Unique Pairs",
        len(pert_pairs) == 81 and len(pert_pairs) == len(pert_df),
        "81 unique pairs",
        f"{len(pert_pairs)} unique pairs",
        "No duplicate (sample_id, condition) in perturbations",
    )

    add_check(
        "AUDIT_06",
        "Baseline Unique Pairs",
        len(base_pairs) == 81 and len(base_pairs) == len(base_df),
        "81 unique pairs",
        f"{len(base_pairs)} unique pairs",
        "No duplicate (sample_id, condition) in baseline results",
    )

    add_check(
        "AUDIT_07",
        "Gated Unique Pairs",
        len(gated_pairs) == 81 and len(gated_pairs) == len(gated_df),
        "81 unique pairs",
        f"{len(gated_pairs)} unique pairs",
        "No duplicate (sample_id, experimental_condition) in gated results",
    )

    # 4. Condition counts
    pert_counts = pert_df["condition"].value_counts().to_dict()
    counts_match = pert_counts == EXPECTED_CONDITION_COUNTS

    add_check(
        "AUDIT_08",
        "Condition Record Counts",
        counts_match,
        str(EXPECTED_CONDITION_COUNTS),
        str(pert_counts),
        "Verify condition counts match expected distribution",
    )

    # 5. Pair alignment
    base_aligned = pert_pairs == base_pairs
    gated_aligned = pert_pairs == gated_pairs

    add_check(
        "AUDIT_09",
        "Perturbation -> Baseline Pairing Alignment",
        base_aligned,
        "100% pairing match",
        "Match" if base_aligned else f"Missing: {pert_pairs - base_pairs}",
        "All perturbation records match baseline output keys",
    )

    add_check(
        "AUDIT_10",
        "Perturbation -> Gated Pairing Alignment",
        gated_aligned,
        "100% pairing match",
        "Match" if gated_aligned else f"Missing: {pert_pairs - gated_pairs}",
        "All perturbation records match gated output keys",
    )

    # 6. Image paths existence
    missing_images = []
    for _, row in pert_df.iterrows():
        img_p = Path(str(row["image_path"]))
        if not img_p.exists():
            missing_images.append(str(img_p))

    add_check(
        "AUDIT_11",
        "Image Files Existence",
        len(missing_images) == 0,
        "0 missing images",
        f"{len(missing_images)} missing",
        "Verify CXR image files exist on disk",
    )

    # 7. Gated report non-emptiness
    empty_gated = (
        gated_df["generated_report"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    add_check(
        "AUDIT_12",
        "Gated Report Non-Emptiness",
        empty_gated == 0,
        "0 empty reports",
        f"{empty_gated} empty reports",
        "All gated reports contain valid policy strings or report text",
    )

    # 8. Zero redundant inference
    redundant_inference = (gated_df["was_model_inference_run"] == True).sum()

    add_check(
        "AUDIT_13",
        "Zero Redundant CPU Inference",
        redundant_inference == 0,
        "was_model_inference_run == False for all 81",
        f"{redundant_inference} true",
        "All 81 gated records reuse exact baseline output without model re-runs",
    )

    audit_df = pd.DataFrame(checks)
    return audit_df


def main() -> None:
    audit_df = run_dataset_audit()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("PHASE 11A: DATASET AUDIT")
    print("=" * 80)

    failed_count = (audit_df["status"] == "FAILED").sum()
    print(f"Total Audit Checks: {len(audit_df)}")
    print(f"Passed Checks: {(audit_df['status'] == 'PASSED').sum()}")
    print(f"Failed Checks: {failed_count}")
    print("=" * 80)

    print(audit_df.to_string(index=False))

    if failed_count > 0:
        print("\nERROR: Dataset audit failed!")
        sys.exit(1)

    print(f"\nAudit table saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
