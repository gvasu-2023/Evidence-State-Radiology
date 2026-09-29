# Phase 23: Qualitative Error Analysis

## Objective and source artifacts

This is a reproducible qualitative analysis of existing baseline, gated, claim-evaluation, and report-metric outputs. It introduces no new reports, clinical findings, or significance tests. Reference evidence is drawn only from the `findings` and `impression` fields in the frozen Phase 16 perturbation file; context fields are reported as recorded. All twelve authoritative input files were hashed before and after analysis and remained unchanged:

- `benchmark`: `3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53`
- `eligibility`: `cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6`
- `baseline`: `39f2591ada1deb8d7947c7e573a01b691dfd524c68269a44fc3b9d877371866f`
- `gated`: `287dd3bdf0dcdf9cf14ae839b34c40a924e74179a02f43373ba01ca9a4baa28e`
- `claims`: `0e60e5c836877b3c65fb67774a3211878599b8054b9f945ad06102da7d3f148a`
- `claim_summary`: `f90af7834beb0f888825285bee2df55602efe513dfed3c5be80a6a04a6050dc0`
- `radgraph`: `3817afa02a411991d2828c7984973e2f881bcd8f358df5543f79b1f289e366aa`
- `chexpert`: `7dda5e16f53acd132612c03f2cdcc174085834eccd51e94c4f57c23001ecc338`
- `phase21_condition`: `8d51ac87eb52e16312653fe2bfcddcb42ce335fd5fefc39c7ee4a60e1c902b5e`
- `phase22_primary`: `0e0497a0a5356e9c3f5dc602a569f1b97117e9684d2246e53f1e06c9a1331d63`
- `phase22_covered`: `36b42e696930d3cd7083564452f49f0aed6fd8df8911ce1fabc944bab2066bd4`
- `phase22_claim`: `f293703850452172baa348c53ba684c966eb901898db39d61d8b219ddbf72217`

## Deterministic selection procedure

The eligible Phase 17 cohort was joined one-to-one by `(sample_id, uid, condition)`. The sample contains three records per condition. Features used were exact report equality, character-level text change fraction (`1 - SequenceMatcher.ratio`), Phase 19 per-record label transitions and unsupported-claim counts (unsupported labels are `ungrounded_affirmation`, `unsupported_affirmation`, and `contradiction`), absolute paired RadGraph-F1 plus CheXpert-F1 delta, and stable identifiers.

Within each condition, selection slots were applied in order: (1) maximize observed removal of unsupported findings, then total claim-label transitions; if neither exists, retained unsupported claims resolve the slot; (2) maximize the sum of absolute RadGraph and CheXpert deltas, then text difference; (3) prefer an unchanged report as a control, otherwise a changed report with no claim-label transition, otherwise the largest remaining text change. Already selected records are excluded from later slots. Remaining ties resolve by ascending `sample_id`, then UID, using stable sorting. This rule selects observed examples and does not inspect qualitative conclusions.

Selected IDs and roles:

| case | sample_id | uid | condition | selection_role |
| --- | --- | --- | --- | --- |
| Q01 | IU_502 | 502 | conflicting | changed_report_without_claim_label_transition |
| Q02 | IU_275 | 275 | conflicting | largest_absolute_metric_delta |
| Q03 | IU_417 | 417 | conflicting | largest_unsupported_reduction_or_claim_label_change |
| Q04 | IU_1001 | 1001 | evidentiary_incomplete | largest_absolute_metric_delta |
| Q05 | IU_417 | 417 | evidentiary_incomplete | largest_unsupported_reduction_or_claim_label_change |
| Q06 | IU_1172 | 1172 | evidentiary_incomplete | unchanged_report_control |
| Q07 | IU_317 | 317 | insufficient | changed_report_without_claim_label_transition |
| Q08 | IU_247 | 247 | insufficient | largest_absolute_metric_delta |
| Q09 | IU_417 | 417 | insufficient | largest_unsupported_reduction_or_claim_label_change |
| Q10 | IU_303 | 303 | irrelevant | changed_report_without_claim_label_transition |
| Q11 | IU_254 | 254 | irrelevant | largest_absolute_metric_delta |
| Q12 | IU_628 | 628 | irrelevant | largest_unsupported_reduction_or_claim_label_change |
| Q13 | IU_1001 | 1001 | sufficient | largest_absolute_metric_delta |
| Q14 | IU_417 | 417 | sufficient | largest_unsupported_reduction_or_claim_label_change |
| Q15 | IU_1172 | 1172 | sufficient | unchanged_report_control |
| Q16 | IU_270 | 270 | syntactic_incomplete | changed_report_without_claim_label_transition |
| Q17 | IU_989 | 989 | syntactic_incomplete | largest_absolute_metric_delta |
| Q18 | IU_594 | 594 | syntactic_incomplete | largest_unsupported_reduction_or_claim_label_change |

## Representative case table

Full contexts, reference findings/impressions, reports, label counts, and claim transitions are in `results/tables/phase23_qualitative_cases.csv`. Text below is a compact index; interpretations refer to the recorded labels and score deltas and do not add clinical findings.

| case | condition | analyzer_state | gate_action | baseline_behavior | gated_behavior | claim_level_change | radgraph_delta | chexpert_delta | qualitative_categories | qualitative_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q01 | conflicting | conflicting | qualify_or_abstain | generated report; unsupported finding labels=3; labels: omission=2; contradiction=3 | qualification/warning added; report text changed; unsupported finding labels=3; labels: omission=2; contradiction=3 | unsupported findings retained: consolidation, pleural effusion, pneumothorax | 0.0 | 0.0 | unsupported_claim_retained;contradictory_content_retained;qualification_added;qualification_not_effective | text changed; analyzer=conflicting, action=qualify_or_abstain. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q02 | conflicting | conflicting | qualify_or_abstain | generated report; unsupported finding labels=0; labels: supported=3 | qualification/warning added; report text changed; unsupported finding labels=0; labels: supported=3 | labels unchanged (supported=3 → supported=3) | 0.0625 | 0.0 | qualification_added | text changed; analyzer=conflicting, action=qualify_or_abstain. Phase 19 unsupported labels 0→0; RadGraph Δ=+0.0625, CheXpert Δ=+0.0000. |
| Q03 | conflicting | conflicting | qualify_or_abstain | generated report; unsupported finding labels=3; labels: omission=1; contradiction=3 | qualification/warning added; report text changed; unsupported finding labels=3; labels: omission=1; contradiction=3 | unsupported findings retained: consolidation, pleural effusion, pneumothorax | 0.0 | 0.0 | unsupported_claim_retained;contradictory_content_retained;qualification_added;qualification_not_effective | text changed; analyzer=conflicting, action=qualify_or_abstain. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q04 | evidentiary_incomplete | sufficient | generate | generated report; unsupported finding labels=0; labels: omission=1; extra_negation=3 | no qualification notice; report text unchanged; unsupported finding labels=0; labels: omission=1; extra_negation=3 | labels unchanged (omission=1; extra_negation=3 → omission=1; extra_negation=3) | 0.0 | 0.0 | report_unchanged;analyzer_misclassification | text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 0→0; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q05 | evidentiary_incomplete | sufficient | generate | generated report; unsupported finding labels=3; labels: omission=1; contradiction=3 | no qualification notice; report text unchanged; unsupported finding labels=3; labels: omission=1; contradiction=3 | unsupported findings retained: consolidation, pleural effusion, pneumothorax | 0.0 | 0.0 | report_unchanged;unsupported_claim_retained;contradictory_content_retained;analyzer_misclassification | text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q06 | evidentiary_incomplete | sufficient | generate | generated report; unsupported finding labels=2; labels: omission=1; contradiction=2 | no qualification notice; report text unchanged; unsupported finding labels=2; labels: omission=1; contradiction=2 | unsupported findings retained: pleural effusion, pneumothorax | 0.0 | 0.0 | report_unchanged;unsupported_claim_retained;contradictory_content_retained;analyzer_misclassification | text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 2→2; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q07 | insufficient | insufficient | abstain | generated report; unsupported finding labels=0; labels: omission=3 | abstention output; report text changed; unsupported finding labels=0; labels: omission=3 | labels unchanged (omission=3 → omission=3) | 0.0 | 0.0 | abstention_due_to_insufficient_evidence | text changed; gated output is abstention; analyzer=insufficient, action=abstain. Phase 19 unsupported labels 0→0; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q08 | insufficient | insufficient | abstain | generated report; unsupported finding labels=0; labels: supported=3; extra_negation=1 | abstention output; report text changed; unsupported finding labels=0; labels: omission=3 | consolidation: supported → omission; pleural effusion: supported → omission; pneumothorax: supported → omission; pulmonary edema: extra_negation → blank | -0.5882 | 1.0 | supported_content_omitted;abstention_due_to_insufficient_evidence;mixed_change | text changed; gated output is abstention; analyzer=insufficient, action=abstain. Phase 19 unsupported labels 0→0; RadGraph Δ=-0.5882, CheXpert Δ=+1.0000. |
| Q09 | insufficient | insufficient | abstain | generated report; unsupported finding labels=3; labels: omission=1; extra_negation=1; contradiction=3 | abstention output; report text changed; unsupported finding labels=0; labels: omission=4 | unsupported findings removed: consolidation, pleural effusion, pneumothorax | -0.2667 | -0.4 | unsupported_claim_removed;contradictory_content_reduced;supported_content_omitted;abstention_due_to_insufficient_evidence | text changed; gated output is abstention; analyzer=insufficient, action=abstain. Phase 19 unsupported labels 3→0; RadGraph Δ=-0.2667, CheXpert Δ=-0.4000. |
| Q10 | irrelevant | irrelevant | discount_context | generated report; unsupported finding labels=2; labels: omission=4; extra_negation=3; ungrounded_affirmation=2 | no qualification notice; report text changed; unsupported finding labels=2; labels: omission=4; extra_negation=3; ungrounded_affirmation=2 | unsupported findings retained: interstitial abnormality, pulmonary edema | 0.0015000000000000013 | 0.0 | unsupported_claim_retained | text changed; analyzer=irrelevant, action=discount_context. Phase 19 unsupported labels 2→2; RadGraph Δ=+0.0015, CheXpert Δ=+0.0000. |
| Q11 | irrelevant | irrelevant | discount_context | generated report; unsupported finding labels=0; labels: unclear=1 | no qualification notice; report text changed; unsupported finding labels=0; labels: extra_negation=4 | consolidation: blank → extra_negation; pleural effusion: blank → extra_negation; pneumothorax: blank → extra_negation; pulmonary edema: blank → extra_negation; nan: unclear → blank | 0.08310000000000001 | 1.0 |  | text changed; analyzer=irrelevant, action=discount_context. Phase 19 unsupported labels 0→0; RadGraph Δ=+0.0831, CheXpert Δ=+1.0000. |
| Q12 | irrelevant | irrelevant | discount_context | generated report; unsupported finding labels=1; labels: supported=2; extra_negation=1; ungrounded_affirmation=1 | no qualification notice; report text changed; unsupported finding labels=0; labels: supported=2; extra_negation=2 | unsupported findings removed: hyperinflation | -0.02980000000000002 | 0.0 | unsupported_claim_removed;irrelevant_context_discounted | text changed; analyzer=irrelevant, action=discount_context. Phase 19 unsupported labels 1→0; RadGraph Δ=-0.0298, CheXpert Δ=+0.0000. |
| Q13 | sufficient | sufficient | generate | generated report; unsupported finding labels=0; labels: omission=1; extra_negation=3 | no qualification notice; report text unchanged; unsupported finding labels=0; labels: omission=1; extra_negation=3 | labels unchanged (omission=1; extra_negation=3 → omission=1; extra_negation=3) | 0.0 | 0.0 | report_unchanged | text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 0→0; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q14 | sufficient | sufficient | generate | generated report; unsupported finding labels=3; labels: omission=1; contradiction=3 | no qualification notice; report text unchanged; unsupported finding labels=3; labels: omission=1; contradiction=3 | unsupported findings retained: consolidation, pleural effusion, pneumothorax | 0.0 | 0.0 | report_unchanged;unsupported_claim_retained;contradictory_content_retained | text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q15 | sufficient | sufficient | generate | generated report; unsupported finding labels=2; labels: omission=1; extra_negation=1; contradiction=2 | no qualification notice; report text unchanged; unsupported finding labels=2; labels: omission=1; extra_negation=1; contradiction=2 | unsupported findings retained: pleural effusion, pneumothorax | 0.0 | 0.0 | report_unchanged;unsupported_claim_retained;contradictory_content_retained | text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 2→2; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| Q16 | syntactic_incomplete | incomplete | qualify | generated report; unsupported finding labels=2; labels: omission=1; extra_negation=2; contradiction=2 | qualification/warning added; report text changed; unsupported finding labels=2; labels: omission=1; extra_negation=2; contradiction=2 | unsupported findings retained: pleural effusion, pneumothorax | -0.057099999999999984 | 0.0 | unsupported_claim_retained;contradictory_content_retained;qualification_added;qualification_not_effective | text changed; analyzer=incomplete, action=qualify. Phase 19 unsupported labels 2→2; RadGraph Δ=-0.0571, CheXpert Δ=+0.0000. |
| Q17 | syntactic_incomplete | incomplete | qualify | generated report; unsupported finding labels=0; labels: extra_negation=2 | qualification/warning added; report text changed; unsupported finding labels=0; labels: extra_negation=4 | consolidation: blank → extra_negation; pulmonary edema: blank → extra_negation | 0.0822 | 1.0 | qualification_added | text changed; analyzer=incomplete, action=qualify. Phase 19 unsupported labels 0→0; RadGraph Δ=+0.0822, CheXpert Δ=+1.0000. |
| Q18 | syntactic_incomplete | incomplete | qualify | generated report; unsupported finding labels=1; labels: supported=1; omission=2; extra_negation=3; contradiction=1 | qualification/warning added; report text changed; unsupported finding labels=0; labels: supported=1; omission=3; extra_negation=3 | unsupported findings removed: airspace opacity | -0.0262 | 0.0 | unsupported_claim_removed;contradictory_content_reduced;supported_content_omitted;qualification_added | text changed; analyzer=incomplete, action=qualify. Phase 19 unsupported labels 1→0; RadGraph Δ=-0.0262, CheXpert Δ=+0.0000. |

## Recurring error patterns

Counts are descriptive summaries of the 576 eligible records and existing Phase 19–21 outputs. No subgroup hypothesis tests were performed.

| pattern | scope | condition | record_count | description | selected_cases |
| --- | --- | --- | --- | --- | --- |
| normal generation preservation | condition | sufficient | 96 | Reports unchanged in 96/96 cases; unsupported labels 20 baseline / 20 gated, retained where present. | Q13, Q14, Q15 |
| syntactically incomplete context | condition | syntactic_incomplete | 96 | Analyzer state distribution {'incomplete': 96}; reports changed in 96/96. Unsupported labels 16→21; omission labels 152→124. Text change does not imply claim-risk reduction. | Q16, Q17, Q18 |
| evidentiary-incomplete analyzer limitation | condition | evidentiary_incomplete | 96 | All 96 analyzer states were sufficient; gate action was generate; reports unchanged in 96/96. Removed-evidence records therefore received no gate intervention; unsupported labels remain 21→21. | Q04, Q05, Q06 |
| irrelevant context handling | condition | irrelevant | 96 | Discount-context action changed reports in 96/96. Unsupported labels were 19 baseline and 21 gated; this cohort-level count does not support calling the action successful suppression. | Q10, Q11, Q12 |
| conflicting context handling | condition | conflicting | 96 | Reports changed in 96/96 under qualification action. Unsupported labels 23→23; conflicting content was not removed in cases where Phase 19 continued to label contradictions. | Q01, Q02, Q03 |
| insufficient evidence abstention | condition | insufficient | 96 | Abstain action applied in 96/96; report text changed to the abstention output. Unsupported labels 21→0; gated candidates are not conventional generated reports. | Q07, Q08, Q09 |
| unchanged outputs | all eligible | sufficient + evidentiary_incomplete | 192 | The unchanged-output pool consists of sufficient and evidentiary_incomplete cases; the latter reflects analyzer classification as sufficient, not harmless evidence removal. | Q13, Q14, Q15; Q04, Q05, Q06 |
| report changed without claim-label transition | all eligible | conflicting | 96 | Conflicting reports can change by adding qualification while Phase 19 claim labels remain the same; a textual change alone is not evidence that the conflict was resolved. | Q01, Q02, Q03 |
| unsupported-claim labels by condition | all eligible | all six conditions | 576 | Phase 19 unsupported-label counts by condition (baseline→gated): sufficient: 20→20; syntactic_incomplete: 16→21; evidentiary_incomplete: 21→21; irrelevant: 19→21; conflicting: 23→23; insufficient: 21→0. These associations do not establish that context caused a particular claim. | Q13, Q14, Q15; Q16, Q17, Q18; Q04, Q05, Q06; Q10, Q11, Q12; Q01, Q02, Q03; Q07, Q08, Q09 |
| irrelevant-context claim transitions | condition | irrelevant | 96 | Discount-context reports changed for all 96 records. At least one unsupported finding was removed in 2 records and added in 4; aggregate unsupported labels rose from 19 to 21, so the pattern is mixed and is not summarized as a successful overall reduction. | Q10, Q11, Q12 |
| text changes without positive delta on either metric | selected conditions | syntactic_incomplete + irrelevant + conflicting | 165 | Count of records with changed report text and non-positive paired delta on both existing Phase 20 metrics. This is descriptive; unchanged overlap scores do not establish clinical harm or benefit. | Q16, Q17, Q18; Q10, Q11, Q12; Q01, Q02, Q03 |

### Required evidentiary-incomplete limitation

All 96 `evidentiary_incomplete` records had readable clinical evidence removed by the perturbation protocol, while the current analyzer classified all 96 as `sufficient`. The gate therefore did not intervene, and gated reports matched baseline. Unsupported claim labels remained present in the paired claim evaluation. This is an analyzer limitation; it is not evidence that evidentiary incompleteness is harmless. No analyzer or gate changes were made.

### Insufficient cases

The 96 insufficient records were handled with abstention outputs. These outputs are described as abstentions, not conventional generated reports or report-quality improvements. Phase 19 evaluates their claim consequences through omissions/absence of generated unsupported claims; they must be interpreted separately from generated-report content.

## Limitations

- The sample is a deterministic qualitative subset, not a statistical sample for inferential claims.
- For sufficient and evidentiary-incomplete conditions, all reports are unchanged, so no changed-versus-unchanged contrast exists within those conditions.
- Conflicting reports changed text, but Phase 19 recorded no claim-label transitions for the conflicting condition; qualification can therefore leave evaluated contradiction labels in place.
- Irrelevant-context discounting changed all reports, but cohort-level unsupported labels rose from 19 to 21. It is not characterized as successful unsupported-claim suppression.
- A report text or metric change does not by itself establish clinical benefit. RadGraph-F1 and CheXpert-F1 remain the existing overlap metrics.
- No new significance tests were run. Statistical or clinical significance is not inferred in this phase.

## Reproducibility

Run `.venv\Scripts\python.exe -m src.evaluation.run_phase23_qualitative_analysis` from the repository root. The runner validates cohort keys and condition balance, deterministically recomputes the 18-case set, and verifies file/tree hashes before and after writing outputs.
