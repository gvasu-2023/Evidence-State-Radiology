# Phase 28A: Local interactive research demonstrator

## Purpose and scope

Phase 28A provides a local interface to inspect Phase 26D frozen study records, compare saved generation outputs, and interact with the project's existing evidence-state analyzer and reliability gate. It is a research demonstrator, not a new evaluation or an inference pipeline. It makes no model calls and does not download model weights.

## Data and project components

- `results/maira2_contrastive/phase26d_evaluation/phase26d_evaluation_generation.csv` supplies the saved study-condition records, contexts, image paths, and reports at lambda 0.00 and 0.25.
- `results/tables/phase26d_master/phase26d_master_evaluation.csv` supplies the frozen unsupported-claim-rate, RadGraph F1, and custom CheXpert-style F1 snapshot.
- The app reuses `EvidenceStateAnalyzer`, `EvidenceAssessment`, `parse_context_completeness`, and `ReliabilityGate` from `src/`.
- `demo/data.py` validates the stored CSV structure and resolves only the image path named in the record. It never substitutes another report or image.

## Interaction and interpretation

The benchmark condition remains metadata and is not supplied to the analyzer. The analyzer receives image availability, context availability, context relevance, consistency, evidence strength, and context completeness. Relevance, consistency, and evidence strength are explicit reviewer controls because the current analyzer does not infer them from raw text or pixels. Quick walkthrough buttons seed these controls to illustrate a path; they are not measurements or model predictions.

The existing context parser reports structural fragmentation cues. It cannot determine whether the context is clinically complete. Phase 26D's `syntactic_incomplete` and `evidentiary_incomplete` labels are distinct saved conditions, while the analyzer has one `INCOMPLETE` state; the interface preserves that distinction in metadata and does not imply that the analyzer separates those subtypes.

The report panels show stored generations only. A missing lambda output, failed generation, or unavailable image is displayed as missing; values are never filled from a different record. The results snapshot displays the frozen master table's recorded values and interpretation.

## Scientific limits

The demo performs no new statistical test and generates no new result. The held-out evaluation did not demonstrate statistically significant overall improvement. Results vary across metrics and evidence conditions. Lambda 0.25 remains the frozen Phase 26D setting. The UI's reviewer controls are illustrative inputs and must not be described as automatic semantic understanding, clinical validation, or model performance.

## Launch

From the repository root on Windows, install the small UI dependency and launch:

```powershell
.venv\Scripts\python.exe -m pip install -r demo\requirements.txt
.venv\Scripts\python.exe -m streamlit run demo\app.py
```

See [demo/README.md](../demo/README.md) for the short walkthrough.
