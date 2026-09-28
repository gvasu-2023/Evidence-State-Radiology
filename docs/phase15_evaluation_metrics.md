# Phase 15 Evaluation Metrics Specification & Benchmark Report

## 1. Objective

Phase 15 introduces a comprehensive, multi-dimensional evaluation layer to benchmark the factual reliability, clinical content alignment, selective prediction risk, and abstention performance of the Evidence-State-Aware Radiology Report Generation framework.

This evaluation layer operates strictly on the existing 81-instance controlled perturbation benchmark without altering underlying models, evidence-state taxonomies, perturbation records, or previously frozen baseline/gated generation outputs.

---

## 2. Benchmark Architecture & Instance Distinction

- **Underlying Cases**: 20 distinct patient CXR studies from the OpenI / IU X-Ray subset (`data/processed/iu_xray/`).
- **Controlled Evaluation Instances**: 81 perturbation records (`data/processed/iu_xray/perturbations/perturbations.csv`) spanning 5 evidence states:
  - `sufficient`: 17 records
  - `incomplete`: 9 records
  - `irrelevant`: 20 records
  - `conflicting`: 15 records
  - `insufficient`: 20 records
- *Distinction*: The 81 evaluation instances represent controlled clinical context perturbation conditions across 20 underlying cases. They are **never** described as 81 independent patients.

---

## 3. Metrics, Definitions & Mathematical Formulations

### 3.1 RadGraph-F1
- **Definition**: Evaluates clinical entity and relation graph overlap between generated reports (baseline or gated) and reference reports (reference findings + impression).
- **Implementation**: Official `radgraph` PyPI package (`radgraph 0.1.18`, using `F1RadGraph(reward_level="all", model_type="radgraph")`).
- **Characterization**: Clinical content / entity-relation alignment metric. It is **NOT** a direct measure of hallucination rate, factuality accuracy, or safety.

### 3.2 CheXpert-F1
- **Definition**: Evaluates clinical finding presence F1 across the 14 Stanford CheXpert observation categories:
  1. Enlarged Cardiomediastinum, 2. Cardiomegaly, 3. Lung Opacity, 4. Lung Lesion, 5. Edema, 6. Consolidation, 7. Pneumonia, 8. Atelectasis, 9. Pneumothorax, 10. Pleural Effusion, 11. Pleural Other, 12. Fracture, 13. Support Devices, 14. No Finding.
- **Label-Handling Policy**:
  - `Positive (1)`: Explicit positive finding mention.
  - `Negative (0)`: Explicit finding negation.
  - `Uncertain (-1)`: Equivocal mention (retained in label vectors for auditability).
  - `Unmentioned (nan)`: Finding absent from report text.
  - *Scoring*: Micro F1 across 14 categories where reference label is positive (1). Uncertain labels (-1) are **NOT** silently treated as positive or negative.

### 3.3 Unsupported Claim Rate (Evaluated Claim Ontology)
- **Operational Definition**:
  $$\text{unsupported clinical claims} = \text{ungrounded\_affirmation} + \text{unsupported\_affirmation} + \text{contradiction}$$
  $$\text{unsupported\_claim\_rate} = \frac{\text{unsupported clinical claims}}{\text{all evaluated generated clinical claims}}$$
- **Preserved Categories**: `omission`, `extra_negation`, and `unclear` are retained as separate categories.
- **Characterization Restriction**: The claim extractor evaluates claims within a target ontology (`configs/claim_ontology.yaml`). Therefore, this metric is strictly designated as **"unsupported claim rate within the evaluated claim ontology"**, **NOT** "overall hallucination rate."

### 3.4 Risk-Coverage & Selective Prediction
- **Coverage**:
  $$\text{Coverage} = \frac{\text{Accepted (Non-Abstained) Evaluation Cases}}{\text{Total Evaluation Cases}}$$
- **Risk**:
  $$\text{Risk} = \text{Unsupported claim rate among covered/generated outputs}$$
- **Operating Point**: The baseline system operates at Coverage = 1.0000, Abstention Rate = 0.0000. The gated system operates at Coverage = 0.7531 (61/81 cases), Abstention Rate = 0.2469 (20/81 cases).
- *AURC Note*: Because the current gate operates deterministically via discrete policy routing, AURC curves are omitted to prevent arbitrary confidence score fabrication.

### 3.5 Abstention Metrics
Evaluates the gate's decision to abstain on unsafe or insufficient evidence cases.
- **Operational References**:
  1. `ref_insufficient`: Ground-truth condition == `insufficient` (N=20 cases).
  2. `ref_unsafe_insufficient_or_conflicting`: Ground-truth condition in {`insufficient`, `conflicting`} (N=35 cases).
- **Formulas**:
  - $\text{Abstention Precision} = \frac{TP_{abstain}}{TP_{abstain} + FP_{abstain}}$
  - $\text{Abstention Recall} = \frac{TP_{abstain}}{TP_{abstain} + FN_{abstain}}$
  - $\text{Abstention Rate} = \frac{\text{Gated Abstentions}}{N_{total}}$

---

## 4. Summary Results Table

| Metric | System | Overall | Sufficient | Incomplete | Irrelevant | Conflicting | Insufficient |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RadGraph-F1** | Baseline | 0.2431 | 0.2236 | 0.2270 | 0.2739 | 0.2349 | 0.2423 |
| | Gated | 0.1755 | 0.2236 | 0.2270 | 0.2423 | 0.2353 | 0.0000 |
| | **Delta** | **-0.0676** | **0.0000** | **0.0000** | **-0.0316** | **+0.0003** | **-0.2423** |
| **CheXpert-F1** | Baseline | 0.2383 | 0.2784 | 0.4074 | 0.2700 | 0.0667 | 0.2250 |
| | Gated | 0.2457 | 0.2784 | 0.4074 | 0.2250 | 0.0667 | 0.3000 |
| | **Delta** | **+0.0074** | **0.0000** | **0.0000** | **-0.0450** | **0.0000** | **+0.0750** |
| **Unsupported Claim Rate** | Baseline | 0.0043 | 0.0000 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| | Gated | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **Delta** | **-0.0043** | **0.0000** | **0.0000** | **-0.0208** | **0.0000** | **0.0000** |
| **Coverage** | Baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| | Gated | **0.7531** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** |
| **Abstention Rate** | Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | Gated | **0.2469** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **1.0000** |

---

## 5. Abstention Performance Summary

| Operational Reference Target | $N_{ref}$ | $N_{gated\_abstain}$ | TP | FP | FN | Precision | Recall | Abstention Rate | Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ref_insufficient`** | 20 | 20 | 20 | 0 | 0 | **1.0000** | **1.0000** | 0.2469 | 0.7531 |
| **`ref_unsafe_insufficient_or_conflicting`** | 35 | 20 | 20 | 0 | 15 | **1.0000** | **0.5714** | 0.2469 | 0.7531 |

---

## 6. Input & Output File Index

### Inputs:
- `data/processed/iu_xray/perturbations/perturbations.csv` (81 perturbation records)
- `results/tables/baseline_generation_results.csv` (81 baseline report outputs)
- `results/tables/phase13b_gated_generation_results.csv` (81 gated report outputs)
- `results/tables/reference_claim_candidates.csv` (reference claim candidates)

### Outputs Generated in `results/tables/`:
- `radgraph_f1_results.csv` & `radgraph_f1_summary.csv`
- `chexpert_f1_results.csv` & `chexpert_f1_summary.csv`
- `unsupported_claim_rate.csv` & `unsupported_claim_summary.csv`
- `risk_coverage_points.csv` & `risk_coverage_summary.csv`
- `abstention_metrics.csv`

---

## 7. Limitations & Scientific Interpretation Constraints

1. **Deterministic Prototype**: The gate routing is deterministic based on discrete evidence state classification.
2. **Evaluation Ontology Boundary**: Unsupported claim rate is evaluated strictly within the defined claim ontology (`configs/claim_ontology.yaml`). It must NOT be described as overall hallucination rate.
3. **RadGraph Score Characterization**: RadGraph-F1 measures entity and relation overlap; zero score on abstained outputs reflects report text suppression rather than poor clinical alignment.

---

## 8. Reproducibility

To reproduce all Phase 15 evaluation tables and run the full test suite:

```powershell
# 1. Run Phase 15 Master Pipeline
.venv\Scripts\python.exe src\evaluation\run_phase15_evaluation.py

# 2. Run Full Test Suite
.venv\Scripts\python.exe -m pytest -q
```
