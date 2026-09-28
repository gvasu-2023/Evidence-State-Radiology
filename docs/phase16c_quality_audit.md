# Final Phase 16C Quality Audit (Phase 16E Selection)

**Recommendation: `READY_FOR_VLM`**. The final benchmark was regenerated from `results/tables/phase16e_selected_studies.csv` and all 600 records passed structural and semantic audit. No generation failures, questionable records, or invalid records were found.

## Condition audit

| Condition | Total | Valid | Questionable | Invalid | Valid rate |
|---|---:|---:|---:|---:|---:|
| sufficient | 100 | 100 | 0 | 0 | 1.00 |
| syntactic_incomplete | 100 | 100 | 0 | 0 | 1.00 |
| evidentiary_incomplete | 100 | 100 | 0 | 0 | 1.00 |
| irrelevant | 100 | 100 | 0 | 0 | 1.00 |
| conflicting | 100 | 100 | 0 | 0 | 1.00 |
| insufficient | 100 | 100 | 0 | 0 | 1.00 |
| **Total** | **600** | **600** | **0** | **0** | **1.00** |

Overall validity: **600/600 (100%)**. The CSV contains a record-level VALID/QUESTIONABLE/INVALID classification and audit reason for each benchmark row, with condition totals repeated for filtering.

## Generation methods and failures

| Condition | Method distribution | Generation failures |
|---|---|---:|
| sufficient | identity_copy: 100 | 0 |
| syntactic_incomplete | preposition_truncation: 52; appended_conjunction_fallback: 28; pattern_truncation: 20 | 0 |
| evidentiary_incomplete | clause_clinical_removal: 100 | 0 |
| irrelevant | deterministic_screened_donor_assignment: 100 | 0 |
| conflicting | explicit_reference_polarity_contradiction: 100 | 0 |
| insufficient | empty_context: 100 | 0 |

Generation failures: **0**. Failure reasons: `{}`.

Questionable records: **0**. Invalid records: **0**.

## Structural validation
- PASS: records 600
- PASS: 100 unique uids
- PASS: one each condition
- PASS: 100 each condition
- PASS: no duplicate uid condition
- PASS: uids exactly phase16e
- PASS: no pilot overlap
- PASS: sufficient equals original
- PASS: insufficient empty
- PASS: reference fields and images unchanged
- PASS: all irrelevant sources nonself and match
- PASS: all conflicts opposite explicit

- Authoritative input: `results/tables/phase16e_selected_studies.csv` (100 studies).
- No UID from the 20-case pilot is present.
- Both generated CSV copies contain the same records and fields.

## Semantic review

- **Sufficient:** all 100 preserve the original selected-study context exactly.
- **Syntactic incomplete:** all 100 are parser-incomplete; the parser identifies a dangling conjunction or preposition in each case.
- **Evidentiary incomplete:** all 100 remove readable evidence present in the original context. All use clause-level removal, retain a parser-complete readable residual, and avoid demographic-only, placeholder-only, and generic "Clinical evaluation." reductions.
- **Irrelevant:** all 100 use a non-self donor from the selected studies; each perturbed context exactly matches that donor's source context and passes the target overlap screen.
- **Conflicting:** all 100 have explicit reference and generated-context polarities with opposite values, supported by the recorded reference text.
- **Insufficient:** all 100 perturbed contexts are empty.

## Representative examples

| Condition | UID | Original context | Perturbed context | Reference evidence | Method | Classification | Audit reason |
|---|---:|---|---|---|---|---|---|
| sufficient | 32 | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs | No acute disease. | identity copy | VALID | Original selected-study context is preserved exactly. |
| syntactic_incomplete | 32 | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs | WEAKNESS OF | No acute disease. | preposition_truncation | VALID | Existing parser detects unfinished syntax (dangling_preposition). |
| evidentiary_incomplete | 32 | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs | hx XXXX nodules, XXXX for changes in lungs. | No acute disease. | clause_clinical_removal | VALID | Readable evidence "WEAKNESS OF MUSCLES" is removed by clause_clinical_removal; the residual retains clinical indication and remains parser-complete. |
| irrelevant | 32 | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs | XXXX XXXX with COPD/emphysema and dyspnea on exertion. | No acute disease. | deterministic_screened_donor_assignment | VALID | Context provenance matches non-self donor UID 551; overlap screen passes. |
| conflicting | 32 | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs | Cardiomegaly is present. | the heart is normal in size | explicit_reference_polarity_contradiction | VALID | Explicit opposite polarity: reference NEGATED, context AFFIRMED for cardiomegaly. |
| insufficient | 32 | WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs |  | No acute disease. | empty context | VALID | Context is empty as required. |

## Outputs and tests

- `results/tables/phase16_perturbations.csv` and `data/processed/iu_xray/phase16/phase16_perturbations.csv` were regenerated from the same frame; byte comparison passed.
- Focused test command: `.venv\Scripts\python.exe -m pytest tests/test_phase16c_perturbations.py -q`.
- Full test command: `.venv\Scripts\python.exe -m pytest -q`.
- No VLM inference was run. No baseline, gate, selection, or taxonomy was changed.

## Acceptance status

**`READY_FOR_VLM`**: all six conditions are 100/100 valid; generation failures, questionable records, and invalid records are all zero.
