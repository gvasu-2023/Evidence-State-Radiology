# Phase 16 Complete IU-Xray Benchmark Dataset Audit Report

## 1. Executive Summary

This report presents a statistical audit of the complete locally available **OpenI / IU X-Ray dataset** stored under `external_data/IU-Xray/data/`. The goal of Phase 16 Task 1–3 is to audit the dataset and establish a machine-readable inventory (`results/tables/phase16_dataset_inventory.csv`) for future benchmark scaling without altering frozen methodology, retraining models, generating reports, or modifying the existing 20-case pilot benchmark (`data/processed/iu_xray/`).

---

## 2. Dataset Overview & Source Location

- **Source Dataset**: Indiana University Chest X-Ray (IU X-Ray / OpenI) Dataset.
- **Local Source Location**: `external_data/IU-Xray/data/`
- **Data Format**: 5 Parquet files (`train-00000-of-00005-c80629f6027d8eb1.parquet` through `train-00004-of-00005-208b7d689f1c7b70.parquet`).
- **Total Raw Studies**: **3,851** unique studies.
- **Unique Study Identifiers (UIDs)**: 3,851 (0 duplicate UIDs detected).

---

## 3. Data Schema & Field Mapping

| Raw Parquet Field | Target Inventory Field | Description & Data Characterization |
| :--- | :--- | :--- |
| `uid` | `uid` & `sample_id` | Unique study integer identifier (1 to 3851). `sample_id` formatted as `"IU_" + uid`. |
| `indication` | `clinical_context` | Patient clinical history, symptoms, or requisition reason. |
| `comparison` | `comparison` | Reference to previous imaging studies (if present). |
| `findings` | `findings` | Radiologist narrative report findings section. |
| `impression` | `impression` | Radiologist narrative report impression/summary section. |
| `img_frontal` | `frontal_image` | Frontal view chest X-ray image bytes. |
| `img_lateral` | `lateral_image` | Lateral view chest X-ray image bytes. |
| `MeSH` | N/A | Medical Subject Headings annotation tags. |
| `Problems` | N/A | Extracted clinical problem labels. |

---

## 4. Statistical Audit Results

### 4.1 Field Completeness Breakdown ($N=3,851$ total studies)

| Field | Description | Present Count | Present % | Missing Count | Missing % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `clinical_context` (`indication`) | Clinical requisition | 3,765 | 97.77% | 86 | 2.23% |
| `findings` | Report findings | 3,337 | 86.65% | 514 | 13.35% |
| `impression` | Report impression | 3,820 | 99.19% | 31 | 0.81% |
| `frontal_image` (`img_frontal`) | Frontal CXR image | 3,689 | 95.79% | 162 | 4.21% |
| `lateral_image` (`img_lateral`) | Lateral CXR image | 3,550 | 92.18% | 301 | 7.82% |
| `has_any_image` | Frontal OR Lateral | 3,851 | 100.00% | 0 | 0.00% |
| `has_both_images` | Frontal AND Lateral | 3,388 | 87.98% | 463 | 12.02% |

### 4.2 Complete & Usable Studies
A study is defined as **complete and usable** if it satisfies all four criteria:
1. Non-empty `clinical_context` (`indication`)
2. Non-empty `findings`
3. Non-empty `impression`
4. At least one available image (`has_frontal` or `has_lateral`)

- **Total Complete & Usable Studies**: **3,278** (85.12% of total dataset).
- **Incomplete / Excluded Studies**: **573** (14.88% of total dataset), primarily driven by missing `findings` (514 studies).

---

## 5. Overlap with Existing 20-Case Pilot Benchmark

- **Pilot Benchmark Location**: `data/processed/iu_xray/metadata/dataset.csv`
- **Pilot Study Count**: 20 cases (`IU_1`, `IU_2`, `IU_4`, `IU_5`, `IU_6`, `IU_7`, `IU_8`, `IU_9`, `IU_10`, `IU_11`, `IU_12`, `IU_13`, `IU_14`, `IU_15`, `IU_17`, `IU_18`, `IU_19`, `IU_20`, `IU_22`, `IU_23`).
- **Raw Dataset Overlap**: **20 / 20** pilot studies are present in the raw parquet files.
- **Pilot Completeness Verification**: All **20** pilot studies satisfy full complete-study eligibility criteria (100% complete).
- **Additional Usable Studies Available**: **3,258** complete studies are available in the raw dataset that are **NOT** part of the existing 20-case pilot.

---

## 6. Data Quality & Characterization Observations

1. **UID Uniqueness**: All 3,851 records have distinct, non-overlapping UIDs.
2. **Missing Findings**: 514 studies (13.35%) lack narrative `findings` text (containing only `impression` or `indication`). These studies cannot serve as ground-truth reference findings and must be excluded from evaluation benchmark scaling.
3. **Missing Context**: 86 studies (2.23%) lack clinical `indication` context.
4. **De-identification Placeholders**: Text fields contain standard OpenI de-identification markers (`XXXX`, `XXXX-year-old`, `Dr. XXXX`). These placeholders are consistent with the pilot dataset.

---

## 7. Recommendations for Benchmark Expansion

1. **Available Capacity**: With **3,258** additional complete studies available, the local dataset easily supports scaling the evaluation benchmark to **100**, **200**, or **500+** cases.
2. **Recommended Target Scale**: A target expansion of **100 to 200 additional studies** (totaling 120 to 220 studies) is recommended for Phase 16B selection.
3. **Selection Policy Recommendation**:
   - Retain the original 20 pilot studies intact to preserve baseline continuity.
   - Filter candidate studies strictly using the machine-readable inventory `results/tables/phase16_dataset_inventory.csv` (`has_context == True`, `has_findings == True`, `has_impression == True`, `has_frontal == True`).
