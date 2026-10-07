# Phase 27 - Final Scientific Audit

**Audit status: PASS WITH DOCUMENTED REPRODUCIBILITY EXCEPTION**  
Checks passed: 49; documented exceptions: 1; failed: 0; total: 50.

## Integrity statements

No Phase 26D result files were modified. No evaluation-set tuning occurred. Lambda=0.25 remained frozen.

## Dataset accounting

The frozen selected cohort has 100 studies: 16 development and 84 nominal evaluation studies before frontal-image exclusion. The excluded UID set is [74, 597, 803, 885]; 80 evaluation studies remain eligible.

## Development/evaluation separation

The development IDs exactly match the specified 16-study cohort. The evaluation split is disjoint and contains 80 studies; generation contains no development or excluded UID.

## Inference accounting

The frozen generation table contains 960 successful trajectories, with lambda counts {0.0: 480, 0.25: 480} and condition counts {'conflicting': 160, 'evidentiary_incomplete': 160, 'insufficient': 160, 'irrelevant': 160, 'sufficient': 160, 'syntactic_incomplete': 160}. All reports are non-empty.

## Claim accounting

The raw table contains 3864 claim-comparison rows. Unsupported claim rates recomputed from raw labels are 171/935 = 0.1828877005 at lambda 0 and 168/931 = 0.1804511278 at lambda 0.25.

Across 480 paired study-condition records, unsupported status is positive for 126 baseline pairs and 129 selected pairs; discordants are 30 baseline-only and 33 selected-only. The paired-rate-vector Wilcoxon reconstruction gives p = 0.3186348599, matching the frozen value rounded to six decimals. Applying Wilcoxon directly to saved `delta_rate` gives p = 0.3388234605; this difference is retained and reported. Exact two-sided McNemar/binomial p-value recomputed from discordants is 0.8013064925; no separate Phase 26D McNemar p-value is stored.

## RadGraph accounting

Mean F1 is 0.118235092760 at lambda 0 and 0.120466825955 at lambda 0.25. The lambda-pair-vector reconstruction gives p = 0.3357641671 versus frozen p = 0.3363292101; Wilcoxon on saved `f1_delta` gives p = 0.3355760399. Neither available vector formulation reproduces the frozen p-value exactly.

## Custom CheXpert-style accounting

Mean F1 is 0.247886904762 at lambda 0 and 0.243164682540 at lambda 0.25. The lambda-pair-vector reconstruction gives p = 0.7119592028, reproducing the frozen p-value; Wilcoxon on saved `f1_delta` gives p = 0.7075048100.

## Lambda-selection integrity

The policy states that candidates were selected on the development cohort, lambda = 0.25 was selected, and the evaluation cohort was not used for selection. Lambda 0.25 was frozen before held-out evaluation.

## Statistical provenance exception

The Phase 27 runtime used SciPy 1.15.3. Repository requirements pin SciPy 1.15.3, but no Phase 26D execution log records the actual historical runtime. Commit `c660f21` added Phase 26D result tables without a corresponding analysis script or notebook. Matching paired vectors therefore reconstruct p-values but do not prove the historical call syntax. The RadGraph frozen p-value remains unreproducible from both available paired vectors.

This is a documented statistical-provenance limitation, not a failure of experiment integrity. Every current recomputed test is non-significant, so the discrepancy does not change the significance conclusion.

## Statistical interpretation

Held-out evaluation does not demonstrate statistically significant overall improvement. Unsupported claim rate changed only slightly (18.2888% -> 18.0451%); RadGraph F1 changed 0.118235 -> 0.120467; custom CheXpert-style F1 changed 0.247887 -> 0.243165. All frozen and independently recomputed p-values remain above 0.05. Condition-level behavior is heterogeneous.

## Final allowed scientific conclusion

In this held-out evaluation, lambda 0.25 was associated with a small reduction in the measured unsupported-claim rate and a small increase in RadGraph F1, alongside a decrease in custom CheXpert-style F1. The paired comparisons do not establish statistically significant overall improvement. These results do not support a claim of universal reliability improvement.

## Warnings against overclaiming

- Do not describe the small unsupported-claim-rate change as a statistically significant improvement.
- Do not treat any one metric as a complete measure of clinical factuality or reliability.
- Do not claim universal reliability improvement; observed condition-level behavior is heterogeneous.
- The exact McNemar/binomial result is recomputed from frozen paired-status records because no separate Phase 26D p-value is present in the supplied frozen CSVs.
- Phase 22 p = 0.0308837890625 belongs to a separate 576-record analysis with 14 and 4 discordances; it does not describe Phase 26D counts.
- The Phase 26D RadGraph p-value is not exactly reproducible from available paired CSV vectors; preserve it as a frozen value and disclose the unresolved provenance.

## Verified statistics table

Frozen historical values and current reconstructions are separated in `results/tables/phase27_verified_statistics.csv`.

## Check results

| Section | Check | Status | Expected | Observed |
|---|---|---:|---|---|
| A. Study split | 100 selected studies | PASS | 100 | 100 |
| A. Study split | 16 development studies | PASS | 16 | 16 |
| A. Study split | Development IDs exactly match frozen list | PASS | [195, 227, 253, 282, 290, 317, 458, 494, 577, 783, 793, 936, 960, 965, 1025, 1165] | [195, 227, 253, 282, 290, 317, 458, 494, 577, 783, 793, 936, 960, 965, 1025, 1165] |
| A. Study split | 84 nominal evaluation studies before frontal exclusion | PASS | 84 | 84 |
| A. Study split | Development and nominal evaluation disjoint | PASS | disjoint | 0 |
| A. Study split | Development and delivered evaluation split disjoint | PASS | disjoint | 0 |
| A. Study split | Four ineligible UIDs are in nominal evaluation cohort | PASS | [74, 597, 803, 885] | [74, 597, 803, 885] |
| A. Study split | 80 eligible evaluation studies | PASS | 80 | 80 |
| A. Study split | Split evaluation IDs match nominal cohort after exclusions | PASS | 80 | 80 |
| A. Study split | Frontal-ineligible UID set matches exclusions | PASS | [74, 597, 803, 885] | [74, 597, 803, 885] |
| A. Study split | All excluded UID condition records are ineligible | PASS | 24 ineligible records | 24 records; eligible=0 |
| B. Generation | 960 total trajectories | PASS | 960 | 960 |
| B. Generation | 80 unique evaluation studies | PASS | 80 | 80 |
| B. Generation | 480 trajectories per lambda | PASS | 0.00=480; 0.25=480 | {0.0: 480, 0.25: 480} |
| B. Generation | Exactly six expected conditions | PASS | ['conflicting', 'evidentiary_incomplete', 'insufficient', 'irrelevant', 'sufficient', 'syntactic_incomplete'] | ['conflicting', 'evidentiary_incomplete', 'insufficient', 'irrelevant', 'sufficient', 'syntactic_incomplete'] |
| B. Generation | 160 trajectories per condition | PASS | 160 each | {'conflicting': 160, 'evidentiary_incomplete': 160, 'insufficient': 160, 'irrelevant': 160, 'sufficient': 160, 'syntactic_incomplete': 160} |
| B. Generation | No development studies | PASS | none | [] |
| B. Generation | No excluded UID trajectories | PASS | none | [] |
| B. Generation | No duplicate sample-condition-lambda keys | PASS | 0 | 0 |
| B. Generation | All generation statuses succeeded | PASS | all success | {'success': 960} |
| B. Generation | All generated reports are non-empty | PASS | all non-empty | 0 empty |
| C. Claim evaluation | 3864 raw claim comparison rows | PASS | 3864 | 3864 |
| C. Claim evaluation | Unsupported labels exactly match approved set | PASS | ['contradiction', 'ungrounded_affirmation', 'unsupported_affirmation'] | ['contradiction', 'ungrounded_affirmation', 'unsupported_affirmation'] |
| C. Claim evaluation | No unrecognized claim labels | PASS | approved label vocabulary | ['contradiction', 'extra_negation', 'omission', 'supported', 'unclear', 'ungrounded_affirmation', 'unsupported_affirmation'] |
| C. Claim evaluation | Summary counts reproduce raw claim labels | PASS | exact grouped label counts | matched |
| C. Claim evaluation | Lambda 0 unsupported numerator/denominator/rate | PASS | 171 / 935 = 0.1828877005... | (171, 935, 0.18288770053475936) |
| C. Claim evaluation | Lambda 0.25 unsupported numerator/denominator/rate | PASS | 168 / 931 = 0.1804511278... | (168, 931, 0.18045112781954886) |
| C. Claim evaluation | Exactly 480 unique study-condition pairs | PASS | 480, no duplicates | rows=480, duplicates=0 |
| C. Claim evaluation | Paired unsupported-status counts | PASS | baseline=126, selected=129, baseline-only=30, selected-only=33, discordant=63 | baseline=126, selected=129, baseline-only=30, selected-only=33, discordant=63 |
| C. Claim evaluation | Frozen claim-rate p-value remains 0.318635 | PASS | 0.318635 | 0.318635 |
| C. Claim evaluation | Paired-rate vector reconstruction reproduces frozen claim p-value | PASS | 0.318635 | 0.318634859854 |
| C. Claim evaluation | Current-environment Wilcoxon from saved delta_rate computed | PASS | valid p-value | 0.338823460501 |
| C. Claim evaluation | Exact two-sided McNemar/binomial test recomputed | PASS | two-sided exact binomial, n=63, k=30 | 0.801306492504 |
| D. RadGraph | 480 paired records; no duplicate study-condition keys | PASS | 480, no duplicates | rows=480, duplicates=0 |
| D. RadGraph | Stored summary F1 means match frozen targets | PASS | lambda0=0.11823509276005441; lambda0.25=0.12046682595491279 | {0.0: 0.1182350927600544, 0.25: 0.1204668259549127} |
| D. RadGraph | Frozen paired-stat p-value remains at specified value | PASS | 0.3363292101 | 0.336329210118 |
| D. RadGraph | Paired-score vector reconstruction reproduces frozen p-value | EXCEPTION | 0.336329210118 | 0.335764167125 |
| D. RadGraph | Current-environment Wilcoxon from saved delta vector computed | PASS | valid p-value | 0.335576039938 |
| D. RadGraph | Stored paired-stat record count is 480 | PASS | 480 | 480 |
| E. Custom CheXpert-style | 480 paired records; no duplicate study-condition keys | PASS | 480, no duplicates | rows=480, duplicates=0 |
| E. Custom CheXpert-style | Stored summary F1 means match frozen targets | PASS | lambda0=0.24788690476190475; lambda0.25=0.24316468253968254 | {0.0: 0.2478869047619047, 0.25: 0.2431646825396825} |
| E. Custom CheXpert-style | Frozen paired-stat p-value remains at specified value | PASS | 0.7119592028 | 0.711959202795 |
| E. Custom CheXpert-style | Paired-score vector reconstruction reproduces frozen p-value | PASS | 0.711959202795 | 0.711959202795 |
| E. Custom CheXpert-style | Current-environment Wilcoxon from saved delta vector computed | PASS | valid p-value | 0.707504810033 |
| E. Custom CheXpert-style | Stored paired-stat record count is 480 | PASS | 480 | 480 |
| C-E. Statistical interpretation | All current recomputed tests are non-significant at alpha 0.05 | PASS | all p-values >= 0.05 | [0.31863485985425233, 0.33882346050116097, 0.8013064925040663, 0.3357641671254561, 0.33557603993823537, 0.7119592027948771, 0.7075048100334027] |
| F. Master tables | Master table contains exactly the five specified metrics | PASS | ['Custom CheXpert-style F1', 'RadGraph F1', 'RadGraph precision', 'RadGraph recall', 'Unsupported claim rate'] | ['Custom CheXpert-style F1', 'RadGraph F1', 'RadGraph precision', 'RadGraph recall', 'Unsupported claim rate'] |
| F. Master tables | Condition master has all 12 condition-lambda combinations | PASS | 6 conditions x 2 lambdas | 12 |
| F. Master tables | Condition-level results show heterogeneous behavior | PASS | at least one condition metric varies | checked=['unsupported_claim_rate', 'radgraph_f1_mean', 'chexpert_f1_mean'] |
| G. Lambda policy | Policy records development selection, lambda 0.25, and held-out exclusion | PASS | all three explicit statements | present |

## Failures and documented exceptions

- **Paired-score vector reconstruction reproduces frozen p-value**: lambda0 vs lambda0.25 score vectors; two-sided; zero_method=wilcox; correction=False; method=approx
