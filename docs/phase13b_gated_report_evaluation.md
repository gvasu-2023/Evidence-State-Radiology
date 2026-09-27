# Phase 13B: Downstream Gated Report Evaluation Report

## 1. Objective
The objective of Phase 13B is to evaluate the downstream report text transformations, report length changes, claim count shifts, and reference factuality label distributions produced by connecting the CURRENT Phase 12C `EvidenceStateAnalyzer` + CURRENT `ReliabilityGate` to the EXISTING 81 baseline reports.

## 2. Relationship to Phase 13A
- **Phase 13A**: Evaluated decision routing accuracy of the Phase 12C analyzer and ReliabilityGate against reference labels (`state_alignment = 88.89%`, `action_alignment = 88.89%`).
- **Phase 13B**: Evaluates the actual downstream report text outputs, report length changes, claim candidate extraction shifts, and reference-grounded factuality label distributions when gate actions are applied to baseline reports.

## 3. Data Sources & Baseline Reuse
- **Perturbations**: `data/processed/iu_xray/perturbations/perturbations.csv` (81 records).
- **Baseline Reports**: `results/tables/baseline_generation_results.csv` (81 baseline reports).
- **Reference Claim Candidates**: `results/tables/reference_claim_candidates.csv` (73 reference candidate rows).
- **Zero Model Inference**: No deep learning model weights were executed. Baseline reports were retrieved via `(sample_id, condition)` lookups (`was_model_inference_run = False` for all 81 records).

## 4. Phase 12C Analyzer & Gate Pipeline Integration
At inference time, each perturbation record is processed without ground-truth metadata leakage:

1. `perturbed_context` is evaluated by `parse_context_completeness(perturbed_context)` to populate `context_complete`.
2. `EvidenceAssessment` is constructed with `image_available`, `context_available`, `context_relevant`, `context_consistent`, and `context_complete`.
3. `EvidenceStateAnalyzer.classify(assessment)` predicts `predicted_state`.
4. `ReliabilityGate.decide(predicted_state)` determines `gate_action`.
5. Report policy transformation is applied to construct `gated_report`.

## 5. Report Policy Transformations
Reusing the exact policy behavior established in Phase 10:

| Gate Action | Source Baseline Condition | Output Gated Report Formatting |
| :--- | :--- | :--- |
| **`generate`** | `experimental_condition` | Baseline report body unchanged |
| **`discount_context`** | `insufficient` | Image-only baseline report body (context stripped) unchanged |
| **`qualify`** | `insufficient` | `"[QUALIFIED: Incomplete clinical context] " + image-only baseline report` |
| **`qualify_or_abstain`** | `experimental_condition` | `"[WARNING: Clinical context conflict detected] " + baseline report` |
| **`abstain`** | `none` | `"[ABSTAIN: Insufficient evidence for report generation]"` |

### Handling of Specific Actions:
- **`qualify_or_abstain`**: Preserves the warning header prefix without reinterpreting the action as a pure abstention.
- **`abstain`**: Replaces the clinical report text with a standardized abstention message. Claim candidate extraction on this message yields 0 claims.

## 6. Summary of Measured Results

### Overall Gate Action Distribution (81 Records):
- **`generate`**: 26 (17 sufficient + 9 incomplete predicted as complete)
- **`qualify`**: 0
- **`discount_context`**: 20 (20 irrelevant)
- **`qualify_or_abstain`**: 15 (15 conflicting)
- **`abstain`**: 20 (20 insufficient)

### Overall Report Length & Claim Count Changes:
- **Mean Baseline Report Length**: 303.22 characters
- **Mean Gated Report Length**: 246.02 characters
- **Mean Report Length Difference**: **-57.20 characters**
- **Mean Baseline Claim Count**: 2.90 claims / report
- **Mean Gated Claim Count**: 2.31 claims / report
- **Mean Claim Count Difference**: **-0.59 claims / report**

### Condition-Wise Performance Summary:

| Condition | $N$ | Mean Base Length | Mean Gated Length | Mean Length Diff | Mean Base Claims | Mean Gated Claims | Mean Claim Diff | Gate Action Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **sufficient** | 17 | 322.76 | 322.76 | **0.00** | 2.53 | 2.53 | **0.00** | `{'generate': 17}` |
| **incomplete** | 9 | 308.00 | 308.00 | **0.00** | 3.00 | 3.00 | **0.00** | `{'generate': 9}` |
| **irrelevant** | 20 | 320.15 | 252.95 | **-67.20** | 2.40 | 3.90 | **+1.50** | `{'discount_context': 20}` |
| **conflicting** | 15 | 322.67 | 368.67 | **+46.00** | 2.60 | 2.60 | **0.00** | `{'qualify_or_abstain': 15}` |
| **insufficient** | 20 | 252.95 | 54.00 | **-198.95** | 3.90 | 0.00 | **-3.90** | `{'abstain': 20}` |

## 7. Analysis of Downstream Policy Effects

1. **Sufficient Condition (17 cases)**: Reports are passed through without modification. Report length and claim counts are identical to baseline.
2. **Irrelevant Condition (20 cases)**: Swapped prompt context strings are discounted. The pipeline retrieves the image-only baseline output (generated without prompt context). Mean report length decreases by -67.20 characters, while mean image-based claims increase from 2.40 to 3.90 (reflecting unprompted visual findings).
3. **Conflicting Condition (15 cases)**: Conflict warning headers (`"[WARNING: Clinical context conflict detected]"`) are prepended. Mean report length increases by +46.00 characters; claim candidate content remains unchanged.
4. **Insufficient Condition (20 cases)**: Missing context cases trigger report abstention. The report body is replaced with the 54-character abstention string. Extracted claim count drops to 0.00.
5. **Incomplete Condition (9 cases)**: The 9 reduced context strings in the IU X-Ray perturbation subset are syntactically non-fragmented phrases (e.g. `"History of mitral valve prolapse."`). The Phase 12C context completeness parser classifies them as `complete`, predicting `sufficient` state and resulting in `generate` action instead of `qualify`.

## 8. Scientific Limitations & Boundaries
- **Research Prototype Evaluation**: Phase 13B evaluates a software architecture prototype via exact baseline report lookups.
- **Conservative Terminology**:
  - Shifts in claim counts and report lengths measure policy transformation effects, NOT clinical safety, reliability, or medical factuality.
  - No claims of hallucination reduction, clinical accuracy improvement, or deployment readiness are made.
- **Historical Output Protection**: Phase 10, Phase 11, Phase 12, and Phase 13A historical CSV artifacts remain frozen and unmodified.

## 9. Reproducibility Commands
To run the Phase 13B downstream report evaluation:
```powershell
.venv\Scripts\python.exe src\evaluation\evaluate_phase13b_gated_reports.py
```

To run the Phase 13B test suite:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_evaluate_phase13b_gated_reports.py -q
```
