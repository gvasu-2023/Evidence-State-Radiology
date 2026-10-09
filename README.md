# Evidence-State-Aware Radiology Report Generation under Incomplete and Conflicting Clinical Context

This project studies how incomplete, irrelevant, conflicting, or insufficient clinical context affects factual reliability in vision-language model (VLM) radiology reports, and whether evidence-state-aware generation control can reduce unsupported clinical claims.

## Research system

The research pipeline combines an Evidence-State Analyzer, Reliability Gate, and evidence-aware contrastive decoding. It is evaluated with MAIRA-2, claim-level factuality measures, RadGraph, and a custom CheXpert-style evaluation on a controlled evidence-state benchmark. The work also includes a Phase 28A local research demonstrator.

The six evidence states are:

1. `sufficient`
2. `syntactic_incomplete`
3. `evidentiary_incomplete`
4. `irrelevant`
5. `conflicting`
6. `insufficient`

## Dataset and experiment accounting

The controlled benchmark is based on IU-Xray and contains 100 selected studies and 600 frozen condition-level records: 100 records for each of the six conditions. Of the 100 studies, 96 are eligible for VLM evaluation, yielding 576 VLM condition-level records. The evaluation split uses 16 development studies and 80 held-out studies. UIDs 74, 597, 803, and 885 were excluded.

## MAIRA-2 held-out evaluation

The MAIRA-2 model is `microsoft/maira-2`. Its execution notebook is [`experiments/maira2/phase26d/phase26d_execution.ipynb`](experiments/maira2/phase26d/phase26d_execution.ipynb), and the extracted contrastive decoding implementation is [`src/generation/maira2_contrastive.py`](src/generation/maira2_contrastive.py). The contrastive weight, λ = 0.25, was selected on the development cohort. Held-out evaluation contains 960 trajectories: 480 at λ = 0 and 480 at λ = 0.25.

The held-out evaluation did **not** demonstrate statistically significant overall improvement. The reported aggregate comparisons (λ = 0 → λ = 0.25) are:

| Measure | λ = 0 | λ = 0.25 |
| --- | ---: | ---: |
| Unsupported claim rate | 18.2888% | 18.0451% |
| RadGraph F1 | 0.118235 | 0.120467 |
| Custom CheXpert-style F1 | 0.247887 | 0.243165 |

These results do not establish state-of-the-art performance or universal improvement. The RadGraph statistical-provenance exception is documented in the Phase 27 audit: the frozen historical p-value cannot be reproduced from the surviving Phase 26D artifacts.

## Provenance and demonstrator

The repository includes the Phase 26D execution provenance notebook, the Phase 27 scientific audit, and the Phase 28A research demonstrator. The demonstrator is a local interface for inspecting saved records and outputs and interacting with the analyzer and gate; it does not run inference.

Run it from the repository root:

```powershell
.venv\Scripts\python.exe -m streamlit run demo\app.py
```

## Repository structure

- `configs/` — experiment and claim ontology configuration
- `data/` — dataset locations and processed data
- `demo/` — Phase 28A local demonstrator
- `docs/` — methods, evaluation, provenance, and audit documentation
- `experiments/` — experiment notebooks and records
- `external_data/` — external evaluation resources
- `models/` — model-related files
- `notebooks/` — research notebooks
- `results/` — experiment outputs and tables
- `src/` — preprocessing, evidence-state analysis, gating, generation, and evaluation code
- `tests/` — automated test suite

## Tests

The current full test suite has **228 passing tests**. Run it with:

```powershell
.venv\Scripts\python.exe -m pytest -q
```
