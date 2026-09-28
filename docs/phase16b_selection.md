# Phase 16B Deterministic Selection Report: 100 Additional IU-Xray Underlying Studies

## 1. Executive Summary

This report documents the deterministic selection of **100 additional complete non-pilot underlying studies** from the OpenI / IU X-Ray dataset. Combined with the existing 20-case pilot benchmark (`data/processed/iu_xray/`), this selection establishes an expanded underlying dataset of **120 unique studies** for future evidence-state evaluation scaling.

---

## 2. Selection Objective & Constraints

- **Objective**: Select exactly 100 additional complete, non-pilot underlying IU X-Ray studies for benchmark expansion.
- **Source Inventory**: Machine-readable inventory [`results/tables/phase16_dataset_inventory.csv`](file:///C:/Evidence-State-Radiology/results/tables/phase16_dataset_inventory.csv) ($N=3,851$ total studies).
- **Frozen Benchmark Guarantee**: The 20 existing pilot studies remain untouched as historical development/audit data.
- **Scientific Integrity Constraints**:
  - Selection was performed strictly using pre-perturbation metadata from the inventory.
  - **Zero VLM inference** was performed.
  - **Zero perturbations** were generated.
  - The existing 20-case processed benchmark was not altered.

---

## 3. Candidate Filtering & Selection Pipeline

| Stage | Description | Study Count | Notes |
| :--- | :--- | :---: | :--- |
| **Total Inventory Candidates** | Total raw studies in `phase16_dataset_inventory.csv` | **3,851** | All unique UIDs in raw parquet dataset |
| **Pilot Studies Excluded** | Excluded because `already_in_pilot == True` | **20** | Original 20 pilot studies preserved intact |
| **Non-Pilot Candidates** | Studies remaining after pilot exclusion | **3,831** | Candidate pool for benchmark expansion |
| **Usable Complete Candidates** | Satisfy completeness criteria (`has_context`, `has_findings`, `has_impression`, `has_any_image`) | **3,258** | Complete non-pilot candidate pool |
| **Selected Studies** | Sampled deterministically with SEED = 20260928 | **100** | Exported to `phase16_selected_studies.csv` |

---

## 4. Deterministic Selection Methodology

- **Selection Script**: [`src/preprocessing/select_phase16b_studies.py`](file:///C:/Evidence-State-Radiology/src/preprocessing/select_phase16b_studies.py)
- **Random Seed**: `SEED = 20260928`
- **Selection Algorithm**:
  1. Filter non-pilot usable complete candidates (`already_in_pilot == False`, `has_context == True`, `has_findings == True`, `has_impression == True`, `has_frontal == True` OR `has_lateral == True`).
  2. Sort candidate rows deterministically by integer `uid` (`1` to `3851`).
  3. Execute pandas random sampling `candidate_df.sample(n=100, random_state=20260928)`.
  4. Assign `selection_order` (1 to 100) and export schema to [`results/tables/phase16_selected_studies.csv`](file:///C:/Evidence-State-Radiology/results/tables/phase16_selected_studies.csv).

---

## 5. Selected Dataset Statistics ($N=100$)

### 5.1 Image Availability Breakdown

| Image View | Count | Percentage |
| :--- | :---: | :---: |
| **Has Frontal Image (`has_frontal`)** | 99 | 99.0% |
| **Has Lateral Image (`has_lateral`)** | 94 | 94.0% |
| **Has Both Frontal & Lateral (`has_both`)** | 93 | 93.0% |
| **Has At Least One Image** | 100 | 100.0% |

### 5.2 Completeness Breakdown
- **Clinical Context (`clinical_context`)**: 100 / 100 (100.0%)
- **Findings (`findings`)**: 100 / 100 (100.0%)
- **Impression (`impression`)**: 100 / 100 (100.0%)
- **UID Duplicate Count**: 0 (All 100 UIDs are unique)

---

## 6. Verification & Confirmations

1. **No Perturbations Used**: Confirmed that selection relied exclusively on pre-perturbation metadata fields (`indication`, `findings`, `impression`, `img_frontal`, `img_lateral`).
2. **No VLM Inference Executed**: Confirmed that zero model forward passes were executed.
3. **Pilot Benchmark Preservation**: Confirmed that `data/processed/iu_xray/metadata/dataset.csv` (20 pilot cases) and `data/processed/iu_xray/perturbations/perturbations.csv` (81 pilot perturbation instances) were not modified.
4. **Reproducibility**: Selection is 100% reproducible by running `python src/preprocessing/select_phase16b_studies.py` or calling `select_phase16b_studies(seed=20260928)`.
