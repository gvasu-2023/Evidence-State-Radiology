# Evidence-State-Radiology Agent Instructions

## Project

Project title:

Evidence-State-Aware Radiology Report Generation under Incomplete and Conflicting Clinical Context

Research question:

How does incomplete or conflicting clinical context affect the factual reliability of medical VLM-generated radiology reports, and can evidence-state-aware gating reduce unsupported clinical claims?

## Core research contribution

The central contribution is:

Evidence-State Analyzer + model-agnostic Reliability Gate.

The five evidence states are:

1. sufficient
2. incomplete
3. irrelevant
4. conflicting
5. insufficient

The conceptual pipeline is:

CXR image + clinical context
        ↓
visual/text evidence extraction
        ↓
Evidence-State Analyzer
        ↓
Reliability Gate
        ↓
generation / qualification / abstention
        ↓
claim extraction
        ↓
factuality and calibration evaluation

RAG and knowledge graphs are optional secondary components.
They must never be presented as substitutes for patient-specific visual evidence.

## Scientific integrity rules

1. Never fabricate experimental results.
2. Never invent dataset records, patient findings, labels, metrics, citations, or experiments.
3. Clearly distinguish:
   - measured results
   - expected results
   - hypotheses
   - proposed methodology
4. Never call an experiment successful unless it was actually executed and verified.
5. Never call a metric a factuality, hallucination, reliability, or calibration metric unless the implementation actually measures that property.
6. Never change research definitions merely to make results look better.
7. Preserve controlled perturbation conditions exactly unless a change is explicitly requested.
8. Do not silently alter reference findings.
9. Do not silently change ontology mappings.
10. Document every important ontology or evaluation assumption.

## Existing evidence states

Sufficient:
Relevant and consistent clinical context is available and visual evidence is adequate.

Incomplete:
Relevant clinical context exists but some information is missing.

Irrelevant:
Clinical context is not relevant to the imaging task.

Conflicting:
Clinical context conflicts with available imaging/reference evidence.

Insufficient:
Available evidence is inadequate for reliable generation.

## Existing project architecture

data/
src/
models/
experiments/
results/
notebooks/
configs/
tests/
docs/

Important modules include:

src/evidence_state/
src/gating/
src/generation/
src/evaluation/
src/preprocessing/

## Existing baseline

Baseline model:

nathansutton/generate-cxr

Architecture:

BLIP-based CXR report generation model.

Current baseline experiment contains:

81 controlled condition records.

Conditions:

sufficient
incomplete
irrelevant
conflicting
insufficient

Current baseline experiment must not be rerun unnecessarily because CPU inference is expensive.

Before rerunning a large experiment, check existing results and resume capability.

## Dataset

Primary development dataset:

IU X-Ray subset.

Current local processed dataset:

data/processed/iu_xray/

Do not commit raw external dataset files.

Do not commit large model weights.

Do not commit generated image datasets unless explicitly requested.

## Existing experiment counts

Current controlled perturbation dataset:

sufficient: 17
incomplete: 9
irrelevant: 20
conflicting: 15
insufficient: 20

Total: 81.

Do not change these counts without explicit justification and validation.

## Conflict evaluation

Approved conflict cases:

15.

Current explicit target-claim evaluation result:

NEGATED: 12
NOT_MENTIONED: 3

This result means:

"explicit target-claim polarity under controlled conflicting context"

It must NOT be described as:

- hallucination rate
- factuality score
- reliability score
- overall accuracy

unless a new validated evaluator establishes that metric.

## Reference claim extraction

Reference claims are extracted from findings and impressions.

Current reference candidate file:

results/tables/reference_claim_candidates.csv

Current validation:

64 expected reference claim cases validated.

Current candidate count:

66.

Current polarity counts:

NEGATED: 51
AFFIRMED: 15

These numbers are measured artifacts and must not be altered without rerunning the extraction/validation pipeline.

## Ontology

Current ontology file:

configs/claim_ontology.yaml

Ontology mappings are research assumptions.

For example:

calcified granuloma → granulomatous disease

airspace disease / airspace opacity / air space opacity may be normalized.

Do not present ontology-normalized equivalence as literal lexical equivalence.

## Evaluation principles

Prefer conservative evaluation.

If a generated report does not explicitly mention a finding, do not infer that it negates the finding merely because the report says "lungs are clear".

Do not count duplicate Findings/Impression mentions as independent case-level evidence.

Case-level deduplication should be applied where appropriate.

## Coding principles

1. Prefer simple, readable Python.
2. Use pathlib for paths.
3. Use pandas for tabular processing.
4. Add tests for non-trivial evaluation logic.
5. Do not duplicate functionality already present in the repository.
6. Inspect existing modules before creating new ones.
7. Reuse existing functions where appropriate.
8. Do not rewrite working modules unnecessarily.
9. Keep functions small and testable.
10. Avoid hard-coded absolute Windows paths.

## Validation requirements

After modifying code:

1. Run targeted tests.
2. Run relevant scripts.
3. Run the full test suite when practical:

python -m pytest -q

4. Report:
   - files changed
   - commands executed
   - test results
   - generated outputs
   - unresolved issues

Never claim validation without actually running it.

## Git rules

Never run:

git add .

unless explicitly instructed by the user.

Before staging:

git status --short

Review the exact files.

Do not commit:

.venv/
data/raw/
data/processed/
models/
large model files
generated image datasets
temporary outputs

Use focused staging.

Commit messages should follow:

<type>: <short professional description>

Examples:

feat: add reference-grounded factuality evaluator

fix: correct reference polarity detection

test: add conflict evaluation coverage

refactor: simplify claim normalization

## Agent autonomy

The agent may:

- inspect files
- inspect repository structure
- create implementation plans
- modify source code
- create tests
- run tests
- run validation scripts

The agent must request approval before:

- deleting important files
- changing research methodology
- changing dataset definitions
- changing evidence-state definitions
- changing evaluation metrics
- downloading large datasets/models
- modifying .gitignore
- force-resetting Git
- deleting experimental results
- pushing to GitHub
- making claims that experimental results are scientifically valid

## Research decision boundary

The agent implements engineering tasks.

The user/researcher decides:

- research question
- novelty claims
- methodology changes
- dataset inclusion/exclusion
- experimental conditions
- metrics
- interpretation of results
- publication claims

When uncertain about a scientific decision, stop and ask.

## Required completion report

At the end of every substantial task provide:

### Implemented
...

### Files changed
...

### Tests executed
...

### Results
...

### Research assumptions
...

### Remaining issues
...

Do not hide failed tests or unresolved issues.