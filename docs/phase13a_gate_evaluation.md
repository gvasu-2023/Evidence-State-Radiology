# Phase 13A: Reliability Gate Downstream Evaluation Report

## 1. Objective
The objective of Phase 13A is to evaluate the downstream generation policy decisions produced by the current `ReliabilityGate` when connected directly to the Phase 12C `EvidenceStateAnalyzer` (which incorporates the component/fragmentation context-completeness parser).

## 2. Data Source
- **Dataset**: Controlled 81 perturbation dataset located at `data/processed/iu_xray/perturbations/perturbations.csv`.
- **Case Count**: Exactly 81 records across 5 controlled experimental conditions.

## 3. 81-Record Experimental Design
The evaluation benchmark consists of:
- **Sufficient**: 17 records (unperturbed frontal CXR + matching relevant context)
- **Incomplete**: 9 records (partially reduced clinical context)
- **Irrelevant**: 20 records (cross-case swapped clinical context)
- **Conflicting**: 15 records (context contradicted by visual imaging evidence)
- **Insufficient**: 20 records (missing clinical context)
- **Total**: 81 records

## 4. Analyzer $\to$ Gate Pipeline Architecture
Each perturbation record is processed through the inference pipeline without ground-truth metadata leakage:

$$\text{Perturbation Record} \xrightarrow{\text{Assessment Extraction}} \text{EvidenceAssessment} \xrightarrow{\text{Phase 12C Analyzer}} \text{Predicted EvidenceState} \xrightarrow{\text{ReliabilityGate}} \text{GateDecision}$$

### Assessment Construction:
- `image_available`: `True` (frontal image file exists)
- `context_available`: `True` if `perturbed_context` non-empty
- `context_relevant`: `False` for `cross_case_context_swap`, `True` otherwise
- `context_consistent`: `False` for contradiction/negation perturbations, `True` otherwise
- `context_complete`: Output of `assess_context_completeness(perturbed_context)` (Phase 12C parser)
- `evidence_strength`: `1.0`

### Downstream Policy Mapping (`ReliabilityGate`):
- `SUFFICIENT` $\to$ `action = "generate"`
- `INCOMPLETE` $\to$ `action = "qualify"`
- `IRRELEVANT` $\to$ `action = "discount_context"`
- `CONFLICTING` $\to$ `action = "qualify_or_abstain"`
- `INSUFFICIENT` $\to$ `action = "abstain"`

## 5. Controlled Condition vs Expected Action Mapping
The reference controlled experimental condition defines an evaluation mapping to measure policy alignment:

| Experimental Condition | Reference Expected State | Evaluation Expected Action |
| :--- | :--- | :--- |
| **sufficient** | `sufficient` | `"generate"` |
| **incomplete** | `incomplete` | `"qualify"` |
| **irrelevant** | `irrelevant` | `"discount_context"` |
| **conflicting** | `conflicting` | `"qualify_or_abstain"` |
| **insufficient** | `insufficient` | `"abstain"` |

*Note: This expected mapping is used strictly for offline performance evaluation. The `ReliabilityGate` operates solely on the predicted `EvidenceState` output by the analyzer.*

## 6. Distinction Between Analyzer Prediction and Controlled Condition
- The **controlled experimental condition** is the ground-truth benchmark label assigned during perturbation dataset construction.
- The **analyzer prediction** is produced independently by `EvidenceStateAnalyzer.classify(assessment)` at inference time.
- The **ReliabilityGate** receives `predicted_state`, not `condition`. Discrepancies between the controlled condition and analyzer prediction directly propagate to downstream gate decisions.

## 7. Measured Results & Metrics

### Overall Benchmark Metrics (81 Records):
- **Total Records**: 81
- **Overall State Alignment**: **72 / 81 (88.89%)**
- **Overall Action Alignment**: **72 / 81 (88.89%)**

### Overall Gate Action Distribution:
- **`generate`**: 26 (17 sufficient + 9 incomplete predicted as complete)
- **`qualify`**: 0
- **`discount_context`**: 20 (20 irrelevant)
- **`qualify_or_abstain`**: 15 (15 conflicting)
- **`abstain`**: 20 (20 insufficient)

### Condition-Wise Performance Summary:

| Condition | $N$ | Correct State | State Accuracy | Correct Action | Action Accuracy | Actual Gate Action Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **sufficient** | 17 | 17 | **100.0%** | 17 | **100.0%** | `{'generate': 17}` |
| **incomplete** | 9 | 0 | **0.0%** | 0 | **0.0%** | `{'generate': 9}` |
| **irrelevant** | 20 | 20 | **100.0%** | 20 | **100.0%** | `{'discount_context': 20}` |
| **conflicting** | 15 | 15 | **100.0%** | 15 | **100.0%** | `{'qualify_or_abstain': 15}` |
| **insufficient** | 20 | 20 | **100.0%** | 20 | **100.0%** | `{'abstain': 20}` |

## 8. Analysis of Downstream Policy Behavior
- **Sufficient (17/17)**: The pipeline correctly routes all 17 sufficient context cases to full report generation (`"generate"`).
- **Irrelevant (20/20)**: The pipeline correctly routes all 20 swapped context cases to context discounting (`"discount_context"`), stripping irrelevant prompt text before generation.
- **Conflicting (15/15)**: The pipeline correctly routes all 15 contradiction cases to qualified generation or abstention (`"qualify_or_abstain"`).
- **Insufficient (20/20)**: The pipeline correctly routes all 20 missing context cases to abstention (`"abstain"`).
- **Incomplete (0/9)**: Under Phase 12C, the context completeness parser checks for observable syntactic structural fragmentation (e.g. dangling prepositions/conjunctions). Because the 9 reduced context strings in the IU X-Ray perturbation subset (e.g., `"History of mitral valve prolapse."`, `"nasal congestion."`, `"chest pain."`) are syntactically complete phrases, Phase 12C classifies them as `complete`. Consequently, the analyzer predicts `sufficient` and the gate routes them to `"generate"` rather than `"qualify"`.

## 9. Scientific Limitations & Boundaries
1. **Deterministic Rule Execution**: Phase 13A evaluates deterministic rule execution of a software prototype.
2. **Terminology Constraints**:
   - `state_accuracy` / `state_alignment` measures heuristic agreement with controlled labels, NOT clinical accuracy.
   - `action_accuracy` / `action_alignment` measures policy decision agreement with reference rules, NOT clinical reliability or safety.
   - No claims of medical factuality validation, hallucination reduction, or broad clinical generalization are made.
3. **No Inference Over Model Weights**: Phase 13A evaluates policy decision routing only. No report text generation model weights were executed.

## 10. Reproducibility Commands
To run the Phase 13A evaluation script and generate tabular artifacts:
```powershell
.venv\Scripts\python.exe src\evaluation\evaluate_phase13a_gate.py
```

To execute the Phase 13A test suite:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_evaluate_phase13a_gate.py -q
```
