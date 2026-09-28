# Phase 16D Revised Study Selection

## Purpose

Phase 16B selected 100 studies using general completeness and image availability. Phase 16C generation showed that 14 selected contexts could not support the required evidentiary-incomplete perturbation under the existing generator. Phase 16D refines the study selection for perturbation feasibility; it does **not** change the six-state taxonomy or weaken the evidentiary-incomplete definition.

## Selection counts

| Measure | Count |
|---|---:|
| Original Phase 16B selection | 100 |
| Retained valid studies | 86 |
| Removed failed studies | 14 |
| Replacement studies added | 14 |
| Final Phase 16D selection | 100 |
| Phase 16A usable non-pilot candidate pool | 3,258 |

### Removed UIDs

`884, 982, 1422, 1556, 1630, 1689, 1730, 1994, 2147, 2203, 2339, 3515, 3676, 3687`

These are the 14 selected studies flagged as evidentiary-generation failures in the Phase 16D candidate audit. Thirteen have no readable clinical evidence recognized by the existing generator; UID 2339 has no safe residual context under that generator.

### Replacement UIDs

`49, 53, 67, 175, 185, 195, 227, 247, 254, 270, 275, 287, 290, 297`

These are the exact recommended replacements flagged in the Phase 16D audit. Their contexts support clause-level evidence removal with a parser-complete residual. They also pass the audit's syntactic and explicit-reference conflict checks and have frontal images.

## Eligibility criteria

The Phase 16A usable pool follows the existing Phase 16B requirements: exclude pilot studies; require clinical context, findings, and impression; and require at least one frontal or lateral image.

All 100 final studies pass the Phase 16D evidentiary eligibility screen recorded in `phase16d_candidate_audit.csv`: the existing evidentiary generator succeeds with readable evidence removed and a non-empty residual accepted by the existing context-completeness parser. The 86 retained studies are preserved because all six Phase 16C generation records succeeded for them.

Replacement studies additionally meet the stricter audit preference: unused Phase 16A candidates, frontal image available, evidentiary method `clause_clinical_removal`, non-empty removed evidence and residual, and successful syntactic and conflict construction. Their residuals retain readable clinical evidence under the audit's existing-parser vocabulary screen. The recommendation uses the audit's 14 `recommended_replacement` rows without substituting other candidates.

The final set was also checked with the current deterministic irrelevant-donor assignment: all 100 targets receive a distinct non-self donor. Sufficient context continues to use the original context; insufficient context continues to be empty under the existing generator.

## Deterministic selection procedure and seed

1. Read the original Phase 16B selection and the Phase 16D candidate audit.
2. Retain the 86 original UIDs not flagged as evidentiary-generation failures. Their Phase 16B study fields and selection metadata are carried forward unchanged.
3. Take the 14 exact audit recommendations, whose UIDs are outside the original selection, and order them by integer UID ascending.
4. Append the replacements after the retained Phase 16B selection-order values. Their selection method is recorded as `phase16d_sorted_uid_topup_no_sampling`.

The Phase 16B seed remains **20260928** and is preserved for provenance. Phase 16D does not draw a random sample: the audit has already applied its eligibility screen, and the top-up uses the first 14 recommended candidates in ascending UID order. This makes the revised set deterministic without changing the original selection seed or altering the retained records.

## Pilot exclusion and retained-study integrity

The 20 pilot studies remain excluded. The Phase 16A inventory marks those studies, and the final 100 UIDs were checked against that flag and the Phase 16A usability criteria.

All 86 valid original studies are unchanged in the revised file across the original Phase 16B schema, including clinical context, images, findings, impression, comparison, and original selection metadata. Phase 16B's `results/tables/phase16_selected_studies.csv` remains the reproducibility source and is not overwritten.

## Artifact

The revised selection is saved at `results/tables/phase16d_selected_studies.csv`. It retains the original Phase 16B columns and adds `phase16d_selection_status` and `phase16d_selection_reason` to distinguish retained studies from replacements.
