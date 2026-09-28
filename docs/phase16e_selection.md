# Phase 16E Exact Generator-Backed Selection

## Result

The complete Phase 16A usable non-pilot pool was screened by dry-running the current Phase 16C evidentiary generator and the current syntactic and conflict helpers. The final evidentiary screen accepts a candidate only when the generator returns success with readable evidence removed, the removed text appears in the original context, and the residual is non-empty and parser-complete. It rejects placeholder-only or demographic-only removals and the generic "Clinical evaluation." residual. Clause removals must retain readable clinical evidence; generalized residuals must match one of the generator's other explicit, more specific indications.

| Measure | Count |
|---|---:|
| Phase 16A usable non-pilot candidates | 3,258 |
| Evidentiary dry-run success | 1,190 |
| Evidentiary dry-run failure | 2,068 |
| Candidates passing all six construction checks | 1,150 |
| Final selected studies | 100 |

There are more than 100 all-six-state candidates. All 100 selected candidates use `clause_clinical_removal`; therefore, the final set does not depend on the more generalized evidentiary method. A complete 100-target irrelevant donor assignment was also confirmed for the selected set.

## Screening method

The candidate pool was rebuilt from `results/tables/phase16_dataset_inventory.csv` using the Phase 16A / Phase 16B usability rules: non-pilot, context/findings/impression present, and at least one frontal or lateral image. No candidate was accepted based only on non-empty context.

For every candidate, the screen called the actual current functions:

- `generate_evidentiary_incomplete` for the evidentiary dry-run. A success was accepted only when the removed evidence passed the generator's readable-evidence check, was present in the original context, and the residual was well formed. The generic "Clinical evaluation." residual was rejected. The other generalizations produced by the current generator were allowed only when the residual was parser-complete and matched an explicit generator-supported indication.
- `generate_syntactic_incomplete` plus the existing context parser. The output had to succeed and the parser had to identify unfinished syntax.
- `generate_conflict` plus explicit reference/context polarity checks. Both polarities had to be explicit, the polarity values had to differ, and the source text had to match the reference evidence.
- The existing donor-overlap helper for candidate-level irrelevant-donor availability. The final selected set was then passed to `generate_irrelevant_assignment` to confirm a complete one-to-one non-self donor assignment.
- Sufficient and insufficient construction checks. The source context is available for sufficient; the generator's insufficient context is empty.

Failure counts in the exact evidentiary dry-run were `no_readable_clinical_evidence: 902` and `no_safe_evidence_removal: 1,166`. The latter includes generator outputs that would rely on the disallowed generic residual or could not retain a defensible remainder. Candidate-level state details and exact dry-run outputs are in `results/tables/phase16e_candidate_screen.csv`.

## Deterministic selection

Seed/provenance is **20260928**. All-six-state eligible candidates were ranked with `clause_clinical_removal` first and UID ascending within each evidentiary-method tier. The selector scanned consecutive 100-candidate windows in that order and accepted the first window for which the existing irrelevant assignment helper produced 100 unique non-self donors using seed `20260928`. The first window (offset 0) passed; no candidate was sampled outside the exact eligible pool.

The selected UID order is:

`32, 36, 44, 49, 53, 66, 67, 74, 175, 185, 190, 195, 227, 247, 253, 254, 270, 275, 282, 287, 290, 297, 303, 304, 313, 317, 329, 340, 341, 372, 382, 388, 400, 417, 437, 451, 458, 460, 461, 463, 475, 486, 493, 494, 498, 502, 519, 527, 551, 567, 577, 581, 585, 586, 590, 594, 595, 597, 600, 604, 628, 633, 645, 653, 655, 688, 702, 729, 753, 772, 782, 783, 784, 793, 803, 806, 814, 827, 854, 871, 885, 927, 936, 960, 963, 965, 975, 989, 991, 1001, 1008, 1025, 1034, 1059, 1097, 1114, 1165, 1172, 1173, 1205`

All selected UIDs are from the full 3,258-study candidate pool, have unique IDs, and are non-pilot studies. The original Phase 16B selection file was only read and remains unchanged.

## Artifacts

- Full candidate dry-run: `results/tables/phase16e_candidate_screen.csv`
- Final deterministic selection: `results/tables/phase16e_selected_studies.csv`
- Reproducible selector: `src/preprocessing/select_phase16e_studies.py`
- Focused tests: `tests/test_phase16e_selection.py`

This phase performs candidate screening and study selection only. It does not regenerate or modify the final Phase 16C benchmark, change the six-state taxonomy, run VLM inference, or commit changes.
