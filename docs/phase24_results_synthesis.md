# Phase 24: Results Synthesis

## 1. Cohort and evaluation setup

The frozen benchmark contains 600 records across 100 studies and six controlled conditions. The image-backed inference cohort contains 576 eligible records, 96 per condition, across 96 studies. The four excluded image-unavailable UIDs contribute 24 excluded records and are not part of these result tables. Baseline and gated outputs are paired by record. The six benchmark conditions, five analyzer states, and five gate actions remain separate concepts.

## 2. Analyzer and gate behavior

Phase 18 action outputs, summarized descriptively in Phase 21, are shown separately by analyzer state and gate action below. The analyzer classified all 96 evidentiary-incomplete records as sufficient; they followed the `generate` action. This observed limitation is retained.

### By analyzer state

| category_type | category | N | covered | abstained | coverage | baseline_radgraph_f1 | gated_radgraph_f1 | baseline_chexpert_f1 | gated_chexpert_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analyzer_state | sufficient | 192 | 192 | 0 | 1 | 0.194721 | 0.194721 | 0.192463 | 0.192463 |
| analyzer_state | incomplete | 96 | 96 | 0 | 1 | 0.205313 | 0.203897 | 0.187154 | 0.210667 |
| analyzer_state | irrelevant | 96 | 96 | 0 | 1 | 0.184664 | 0.203939 | 0.202433 | 0.210667 |
| analyzer_state | conflicting | 96 | 96 | 0 | 1 | 0.198234 | 0.199285 | 0.178128 | 0.178128 |
| analyzer_state | insufficient | 96 | 0 | 96 | 0 | 0.203939 | 0 | 0.210667 | 0.166667 |

### By gate action

| category_type | category | N | covered | abstained | coverage | baseline_radgraph_f1 | gated_radgraph_f1 | baseline_chexpert_f1 | gated_chexpert_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate_action | generate | 192 | 192 | 0 | 1 | 0.194721 | 0.194721 | 0.192463 | 0.192463 |
| gate_action | qualify | 96 | 96 | 0 | 1 | 0.205313 | 0.203897 | 0.187154 | 0.210667 |
| gate_action | discount_context | 96 | 96 | 0 | 1 | 0.184664 | 0.203939 | 0.202433 | 0.210667 |
| gate_action | qualify_or_abstain | 96 | 96 | 0 | 1 | 0.198234 | 0.199285 | 0.178128 | 0.178128 |
| gate_action | abstain | 96 | 0 | 96 | 0 | 0.203939 | 0 | 0.210667 | 0.166667 |

## 3. Overall report-level results

RadGraph-F1 and CheXpert-F1 are the existing Phase 20 report-overlap metrics. The consolidated table separates baseline, gated all-record, and gated covered-only views.

| evaluation_view | N | coverage | abstention_rate | unsupported_claim_count | generated_candidate_count | unsupported_claim_rate | radgraph_mean | radgraph_median | chexpert_mean | chexpert_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 576 | 1 | 0 | 120 | 1578 | 0.0760456 | 0.196932 | 0.1714 | 0.193885 | 0 |
| Gated - all records | 576 | 0.833333 | 0.166667 | 106 | 1460 | 0.0726027 | 0.166094 | 0.1481 | 0.191842 | 0 |
| Gated - covered only | 480 | 0.833333 | 0.166667 | 106 | 1460 | 0.0726027 | 0.199312 | 0.1714 | 0.196877 | 0 |

## 4. Covered-only results

Covered-only quality uses the 480 records with a non-abstaining gated report. Mean RadGraph-F1 and CheXpert-F1 are slightly higher than baseline in this subset; neither covered-only paired comparison meets its Holm-adjusted threshold. These comparisons are conditional on gate coverage.

## 5. Unsupported-claim results

Phase 21's generated-candidate denominator is used throughout the consolidated risk table: baseline 120/1,578 (7.6046%); gated all-record 106/1,460 (7.2603%); gated covered-only 106/1,460 (7.2603%). This explicitly preserves the Phase 21 denominator definition and does not silently reconcile the historical Phase 19/Phase 21 denominator discrepancy. Omissions are not unsupported claims.

## 6. Statistical results

The primary all-record endpoints used two-sided paired Wilcoxon signed-rank tests with Holm correction across RadGraph-F1 and CheXpert-F1. The covered-only analyses form a separate Holm family. The record-level unsupported-claim presence comparison is separately reported as an exact two-sided McNemar test, with no new correction.

| analysis | endpoint | N | baseline_mean | gated_mean | mean_delta | baseline_median | gated_median | median_delta | wilcoxon_W | p_raw | p_holm_adjusted | rank_biserial_effect_size | baseline_positive_records | gated_positive_records | baseline_only_discordant | gated_only_discordant | test_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-record primary family | RadGraph-F1 | 576 | 0.196932 | 0.166094 | -0.030838 | 0.1714 | 0.1481 | 0 | 10532.5 | 6.21056e-10 | 1.24211e-09 | -0.432638 |  |  |  |  | Wilcoxon signed-rank, two-sided, zero_method=wilcox, method=approx, no continuity correction |
| all-record primary family | CheXpert-F1 | 576 | 0.193885 | 0.191842 | -0.00204236 | 0 | 0 | 0 | 2134 | 0.842698 | 0.842698 | -0.0235644 |  |  |  |  | Wilcoxon signed-rank, two-sided, zero_method=wilcox, method=approx, no continuity correction |
| covered-only secondary family | RadGraph-F1 | 480 | 0.19553 | 0.199313 | 0.00378208 | 0.1714 | 0.1714 | 0 | 6906.5 | 0.0597282 | 0.119456 | 0.161375 |  |  |  |  | Wilcoxon signed-rank, two-sided, zero_method=wilcox, method=approx, no continuity correction |
| covered-only secondary family | CheXpert-F1 | 480 | 0.190528 | 0.196877 | 0.00634917 | 0 | 0 | 0 | 556.5 | 0.576607 | 0.576607 | 0.0914286 |  |  |  |  | Wilcoxon signed-rank, two-sided, zero_method=wilcox, method=approx, no continuity correction |
| secondary claim-level analysis | record-level unsupported-claim presence | 576 |  |  |  |  |  |  |  | 0.0308838 |  |  | 74 | 64 | 14 | 4 | Exact two-sided McNemar test via binomial test; no new correction |

The all-record RadGraph-F1 mean is lower in the gated view; this includes all 96 abstention outputs as scored under Phase 20. The all-record Wilcoxon test has Holm-adjusted p=1.242112e-9. All-record CheXpert-F1 has adjusted p=0.842698. Covered-only RadGraph-F1 and CheXpert-F1 means increase slightly, but adjusted p-values are 0.119456 and 0.576607, respectively.

## 7. Condition-level behavior

No condition-level significance tests were added.

| condition | N | covered | abstained | coverage | baseline_radgraph_f1 | gated_radgraph_f1 | baseline_chexpert_f1 | gated_chexpert_f1 | baseline_unsupported_count | gated_all_unsupported_count | gated_covered_unsupported_count | baseline_unsupported_rate | gated_all_unsupported_rate | gated_covered_unsupported_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sufficient | 96 | 96 | 0 | 1 | 0.199831 | 0.199831 | 0.199505 | 0.199505 | 20 | 20 | 20 | 0.0843882 | 0.0843882 | 0.0843882 |
| syntactic_incomplete | 96 | 96 | 0 | 1 | 0.205313 | 0.203897 | 0.187154 | 0.210667 | 16 | 21 | 21 | 0.0655738 | 0.058011 | 0.058011 |
| evidentiary_incomplete | 96 | 96 | 0 | 1 | 0.18961 | 0.18961 | 0.18542 | 0.18542 | 21 | 21 | 21 | 0.0878661 | 0.0878661 | 0.0878661 |
| irrelevant | 96 | 96 | 0 | 1 | 0.184664 | 0.203939 | 0.202433 | 0.210667 | 19 | 21 | 21 | 0.0805085 | 0.058011 | 0.058011 |
| conflicting | 96 | 96 | 0 | 1 | 0.198234 | 0.199285 | 0.178128 | 0.178128 | 23 | 23 | 23 | 0.0884615 | 0.0884615 | 0.0884615 |
| insufficient | 96 | 0 | 96 | 0 | 0.203939 | 0 | 0.210667 | 0.166667 | 21 | 0 | 0 | 0.058011 | 0 | 0 |

## 8. Qualitative findings

Phase 23 contains 18 condition-level records (three per condition), not 18 independent patients. UIDs may repeat across conditions because each condition is a controlled perturbation of an underlying study.

| condition | selected_case_count | selected_case_ids | selected_sample_ids | major_qualitative_pattern | representative_interpretation |
| --- | --- | --- | --- | --- | --- |
| sufficient | 3 | Q13, Q14, Q15 | IU_1001, IU_417, IU_1172 | Reports unchanged in 96/96 cases; unsupported labels 20 baseline / 20 gated, retained where present. | Q14 (IU_417, UID 417): text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| syntactic_incomplete | 3 | Q16, Q17, Q18 | IU_270, IU_989, IU_594 | Analyzer state distribution {'incomplete': 96}; reports changed in 96/96. Unsupported labels 16→21; omission labels 152→124. Text change does not imply claim-risk reduction. | Q16 (IU_270, UID 270): text changed; analyzer=incomplete, action=qualify. Phase 19 unsupported labels 2→2; RadGraph Δ=-0.0571, CheXpert Δ=+0.0000. |
| evidentiary_incomplete | 3 | Q04, Q05, Q06 | IU_1001, IU_417, IU_1172 | All 96 analyzer states were sufficient; gate action was generate; reports unchanged in 96/96. Removed-evidence records therefore received no gate intervention; unsupported labels remain 21→21. | Q05 (IU_417, UID 417): text unchanged; analyzer=sufficient, action=generate. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| irrelevant | 3 | Q10, Q11, Q12 | IU_303, IU_254, IU_628 | Discount-context action changed reports in 96/96. Unsupported labels were 19 baseline and 21 gated; this cohort-level count does not support calling the action successful suppression. | Q12 (IU_628, UID 628): text changed; analyzer=irrelevant, action=discount_context. Phase 19 unsupported labels 1→0; RadGraph Δ=-0.0298, CheXpert Δ=+0.0000. |
| conflicting | 3 | Q01, Q02, Q03 | IU_502, IU_275, IU_417 | Reports changed in 96/96 under qualification action. Unsupported labels 23→23; conflicting content was not removed in cases where Phase 19 continued to label contradictions. | Q01 (IU_502, UID 502): text changed; analyzer=conflicting, action=qualify_or_abstain. Phase 19 unsupported labels 3→3; RadGraph Δ=+0.0000, CheXpert Δ=+0.0000. |
| insufficient | 3 | Q07, Q08, Q09 | IU_317, IU_247, IU_417 | Abstain action applied in 96/96; report text changed to the abstention output. Unsupported labels 21→0; gated candidates are not conventional generated reports. | Q09 (IU_417, UID 417): text changed; gated output is abstention; analyzer=insufficient, action=abstain. Phase 19 unsupported labels 3→0; RadGraph Δ=-0.2667, CheXpert Δ=-0.4000. |

The qualitative error-pattern summary includes unchanged sufficient/evidentiary-incomplete reports, incomplete-context qualifications, conflicting-context warnings, irrelevant-context changes, and insufficient-context abstentions. In the irrelevant and conflicting conditions, claim-level behavior was not uniformly improved.

## 9. Key limitation: evidentiary-incomplete misclassification

All 96 evidentiary-incomplete perturbations removed readable clinical evidence, but the analyzer classified them as sufficient. The gate did not intervene and outputs matched baseline. This does not imply evidentiary incompleteness is harmless.

## 10. Interpretation

The results show distinct behavior for report overlap, claim labels, and selective coverage. All-record RadGraph-F1 decreases when abstention outputs are included; covered-only report metrics move slightly upward but are not statistically supported after the specified correction. Unsupported-claim record presence falls from 74 to 64 with exact McNemar p=0.030884 as a secondary result. This does not establish universal improvement: irrelevant and conflicting context do not show universally improved claim-level behavior, and the evidentiary-incomplete condition exposes an analyzer limitation.

## 11. What the experiment does not establish

The experiment does not establish clinical safety, clinical significance, universal factuality improvement, generalization beyond this cohort/model/setup, or that a single operating point defines a continuous risk-coverage curve. No intermediate coverage points were created. Statistical significance does not establish clinical significance.
