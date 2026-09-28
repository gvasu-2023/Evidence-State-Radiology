# Phase 16C: Controlled Six-State Perturbation Protocol

## 1. Overview

Phase 16C implements the controlled six-state evidence-state perturbation protocol for the expanded 100-study IU-Xray research benchmark. This protocol expands the evaluation benchmark from the 20-case pilot to 100 underlying clinical studies (600 controlled instances) while leaving the underlying imaging studies, reference reports, baseline VLM, and reliability gate architecture completely frozen.

Every underlying study generates exactly six controlled experimental perturbation records, modifying **ONLY** the clinical-context condition:
1. `sufficient`
2. `syntactic_incomplete`
3. `evidentiary_incomplete`
4. `irrelevant`
5. `conflicting`
6. `insufficient`

---

## 2. Six-State Evidence Taxonomy

| Condition | Evidence State | Description | Modifies Image? | Modifies Reference Findings? |
| :--- | :--- | :--- | :---: | :---: |
| `sufficient` | `sufficient` | Original clinical context provided without modification. | No | No |
| `syntactic_incomplete` | `incomplete` | Syntactically/grammatically unfinished phrase exhibiting dangling prepositions or conjunctions. | No | No |
| `evidentiary_incomplete` | `incomplete` | Syntactically complete phrase with diagnostic/evidentiary clinical information removed. | No | No |
| `irrelevant` | `irrelevant` | Clinical context swapped deterministically from a different study in the selected 100-study pool. | No | No |
| `conflicting` | `conflicting` | Controlled clinical-context contradiction constructed against reference findings/impression evidence. | No | No |
| `insufficient` | `insufficient` | Completely empty clinical context (`""`). | No | No |

---

## 3. Distinction Between Syntactic and Evidentiary Incompleteness

### 3.1 Syntactic Incompleteness (`syntactic_incomplete`)
- **Definition**: The clinical context is truncated to produce an ungrammatical or syntactically incomplete sentence structure (e.g., terminating on prepositions such as `"of"`, `"with"`, `"for"` or conjunctions like `"and"`).
- **Validation**: Evaluated using `assess_context_completeness(perturbed_context)` which MUST return `False` (`is_complete == False`).
- **Clinical Characterization**: Represents an incomplete referral note cut off mid-sentence during electronic health record (EHR) transmission or data ingestion.

### 3.2 Evidentiary Incompleteness (`evidentiary_incomplete`)
- **Definition**: The clinical context remains syntactically well-formed, but specific diagnostic evidence (e.g., disease history, presenting symptoms, procedures) is removed or generalized (e.g., `"History of COPD and shortness of breath."` $\rightarrow$ `"Shortness of breath."` or `"Chest pain."` $\rightarrow$ `"Clinical evaluation."`).
- **Validation**: Evaluated using `assess_context_completeness(perturbed_context)` which MUST return `True` (`is_complete == True`). Furthermore, removal MUST NOT consist solely of demographic information (e.g., removing only `"65-year-old male"` is prohibited).
- **Clinical Characterization**: Represents a referral note that is grammatically complete but lacks crucial historical or diagnostic details required for differential diagnosis.

---

## 4. Specific Perturbation Construction Rules

### 4.1 Irrelevant Context Construction (`irrelevant`)
- **Method**: Cross-case clinical context assignment.
- **Rule**: Each study $A_i$ receives the original clinical context of study $A_j$ where $i \neq j$.
- **Implementation**: A deterministic one-to-one donor assignment uses seed `20260928` and screens donor contexts against target reference evidence using normalized clinical concepts. Self-donors and obvious overlaps or contradictions are excluded. If no screened assignment is available, the slot is marked `generation_failed` with an empty context.
- **Metadata Recorded**: `source_context_uid`, `source_context_sample_id`.

### 4.2 Conflicting Context Construction (`conflicting`)
- **Method**: Conservative evidence-grounded contradiction.
- **Rule**: The clinical context is rewritten only when a normalized finding has unambiguous explicit polarity in reference `findings` or `impression`. The generated context is checked to have the opposite polarity. Ambiguous, uncertain, or unsupported targets fail closed; no generic fallback contradiction is used.
- **Metadata Recorded**: `conflict_target`, `conflict_method`, `conflict_source_text`.

### 4.3 Insufficient Context Construction (`insufficient`)
- **Method**: Complete context omission.
- **Rule**: `perturbed_context = ""`. No whitespace-only or filler substitutes permitted.

---

## 5. Anti-Leakage and Scientific Integrity Rules

1. **Patient Data Integrity**: The underlying study image (`frontal_image`, `lateral_image`), reference findings, reference impression, and comparison text MUST NOT be modified or corrupted.
2. **UID Partitioning**: No study selected in Phase 16B (UIDs 169–3994) overlaps with the original 20-case pilot dataset (`IU_1`–`IU_20`).
3. **No Silently Manufactured Data**: If automatic generation fails for any condition state, `generation_status` MUST be set to `generation_failed` and the exact failure reason recorded. Silent fabrication of data to reach 600 records is strictly forbidden.
4. **Deterministic Reproducibility**: Execution with `SEED = 20260928` is 100% deterministic and reproducible across runs.

---

## 6. Validation Suite & Quality Checks

The implementation is verified by 32 dedicated tests in `tests/test_phase16c_perturbations.py`:
1. Exactly 100 unique UIDs represented.
2. Exactly 1 record per `(uid, condition)` pair.
3. Total slot count equals 600; unavailable perturbations remain explicit failed slots rather than fabricated contexts.
4. Zero duplicate `(uid, condition)` keys.
5. `sufficient` contexts equal `original_context`.
6. `insufficient` contexts are empty strings (`""`).
7. `irrelevant` source UID is never equal to target UID.
8. `irrelevant` source contexts originate from the selected-study pool.
9. Reference findings remain identical.
10. Reference impressions remain identical.
11. Image path references remain identical.
12. `syntactic_incomplete` contexts evaluate as incomplete (`is_complete == False`).
13. `evidentiary_incomplete` contexts evaluate as complete (`is_complete == True`).
14. `evidentiary_incomplete` removals contain readable clinical evidence, not only demographics or placeholders.
15. Successful `conflicting` records have opposite, explicit reference/context polarity.
16. All 6 condition names are valid.
17. No selected UID overlaps with the 20 pilot studies.
18. Generation is deterministic with seed `20260928`.
19. Re-running generation matches disk artifacts.
20. Existing pilot files (`dataset.csv` and `perturbations.csv`) remain untouched.

---

## 7. Known Generation Results & Limitations

### Current Generator Results
- **Selected Studies**: 100 underlying IU-Xray studies
- **Generated Slots**: 600 (100 studies $\times$ 6 conditions)
- **Valid Perturbations**: 586
- **Explicit Generation Failures**: 14 evidentiary-incomplete slots; their perturbed contexts are empty and `generation_status == "generation_failed"`.
- **Complete Six-State Studies**: 86 / 100
- **Conflicting Perturbations**: 100 / 100 have explicit opposite reference/context polarity.
- **Quality Audit**: [`docs/phase16c_quality_audit.md`](phase16c_quality_audit.md); machine-readable counts are in [`results/tables/phase16c_quality_audit.csv`](../results/tables/phase16c_quality_audit.csv).

### Scope Characterization
The perturbations generated under this protocol are **controlled experimental perturbations** designed for standardized benchmark evaluation of medical VLM robustness. They do not claim to capture all nuances of real-world clinical documentation or human physician workflow variability.
