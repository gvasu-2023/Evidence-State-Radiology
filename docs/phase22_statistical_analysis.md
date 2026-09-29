# Phase 22: Statistical Analysis of Baseline vs Gated Results

## Objective and frozen inputs

This phase performs paired statistical analyses of the existing eligible Phase 17–20 records. It does not recompute reports or scores. Frozen benchmark, eligibility, baseline, gated, Phase 19, Phase 20, and Phase 21 artifacts were read as specified. SHA-256 values before analysis:

- `results\tables\phase17_vlm_inference_eligibility.csv`: `cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6`
- `results\baseline\phase17_full\baseline_results.csv`: `39f2591ada1deb8d7947c7e573a01b691dfd524c68269a44fc3b9d877371866f`
- `results\gated\phase18_full\gated_results.csv`: `287dd3bdf0dcdf9cf14ae839b34c40a924e74179a02f43373ba01ca9a4baa28e`
- `results\tables\phase19_claim_evaluation.csv`: `0e60e5c836877b3c65fb67774a3211878599b8054b9f945ad06102da7d3f148a`
- `results\tables\phase20_radgraph_f1_results.csv`: `3817afa02a411991d2828c7984973e2f881bcd8f358df5543f79b1f289e366aa`
- `results\tables\phase20_chexpert_f1_results.csv`: `7dda5e16f53acd132612c03f2cdcc174085834eccd51e94c4f57c23001ecc338`
- `results\tables\phase21_risk_coverage_summary.csv`: `dff3dca69a5854b065758c5611f9cafa0cfadf0b3216fac40e82c2885f3f3ab1`
- `results\tables\phase21_condition_coverage.csv`: `8d51ac87eb52e16312653fe2bfcddcb42ce335fd5fefc39c7ee4a60e1c902b5e`
- `results\tables\phase16_perturbations.csv`: `3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53`

## Pairing and integrity

The analysis includes 576 eligible records, 96 per benchmark condition. Baseline and gated scores were paired by `(sample_id, condition)` and UID equality was checked. Each paired score table had one row per eligible key; there were no excluded UIDs and no missing/non-finite RadGraph-F1 or CheXpert-F1 values. Phase 20 coverage flags agreed across both metrics; the covered-only subset contains 480 records.

## Primary all-record analysis

Endpoints are RadGraph-F1 and CheXpert-F1 for all 576 records. For each record, delta is gated minus baseline. Two-sided Wilcoxon signed-rank tests were used because the observations are paired and differences need not be normally distributed. Zeros were handled with SciPy `zero_method="wilcox"`; the normal approximation with tie correction (`method="approx"`) and no continuity correction was used. Holm correction was applied across the two primary endpoint p-values. Alpha is 0.05.

| endpoint | n | baseline_mean | gated_mean | mean_delta_gated_minus_baseline | baseline_median | gated_median | median_delta_gated_minus_baseline | wilcoxon_statistic | p_value_raw | p_value_holm | rank_biserial_correlation | significant_after_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| radgraph_f1 | 576 | 0.196932 | 0.166094 | -0.030838 | 0.1714 | 0.1481 | 0 | 10532.5 | 6.21056e-10 | 1.24211e-09 | -0.432638 | True |
| chexpert_f1 | 576 | 0.193885 | 0.191842 | -0.00204236 | 0 | 0 | 0 | 2134 | 0.842698 | 0.842698 | -0.0235644 | False |

## Secondary covered-only analysis

The same paired comparisons were repeated only for the 480 records with a covered gated report. The two covered-only p-values form a separate Holm family.

| endpoint | n | baseline_mean | gated_mean | mean_delta_gated_minus_baseline | baseline_median | gated_median | median_delta_gated_minus_baseline | wilcoxon_statistic | p_value_raw | p_value_holm | rank_biserial_correlation | significant_after_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| radgraph_f1 | 480 | 0.19553 | 0.199313 | 0.00378208 | 0.1714 | 0.1714 | 0 | 6906.5 | 0.0597282 | 0.119456 | 0.161375 | False |
| chexpert_f1 | 480 | 0.190528 | 0.196877 | 0.00634917 | 0 | 0 | 0 | 556.5 | 0.576607 | 0.576607 | 0.0914286 | False |

## Paired difference diagnostics

Counts and distribution summaries are computed on gated-minus-baseline paired differences. No normality test was used to select the test.

| cohort | endpoint | n | zero_differences | positive_differences | negative_differences | mean_difference | median_difference | difference_q1 | difference_q3 | difference_iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_records_primary | radgraph_f1 | 576 | 304 | 111 | 161 | -0.030838 | 0 | -0.029475 | 0 | 0.029475 |
| all_records_primary | chexpert_f1 | 576 | 483 | 42 | 51 | -0.00204236 | 0 | 0 | 0 | 0 |
| covered_only_secondary | radgraph_f1 | 480 | 299 | 111 | 70 | 0.00378208 | 0 | 0 | 0 | 0 |
| covered_only_secondary | chexpert_f1 | 480 | 431 | 27 | 22 | 0.00634917 | 0 | 0 | 0 | 0 |

## Paired effect size

The paired rank-biserial correlation is reported for each Wilcoxon comparison. It is calculated from nonzero paired differences as `(W_positive - W_negative) / (W_positive + W_negative)`, where positive and negative rank sums use average ranks of absolute differences. The sign therefore follows gated-minus-baseline direction. This is a descriptive numerical effect size and is not given a qualitative magnitude label.

## Secondary record-level unsupported-claim comparison

Phase 19 supports one binary outcome per report: an indicator is positive if that record contains at least one claim labeled `ungrounded_affirmation`, `unsupported_affirmation`, or `contradiction`. Other labels do not count. Claim rows were aggregated within record before testing; individual claims were not treated as independent observations. The result uses exact two-sided McNemar testing, implemented as a binomial test on the discordant pairs under probability 0.5. It is reported separately and is not Holm-adjusted with the F1 endpoint families.

| analysis | n | baseline_positive_records | gated_positive_records | baseline_only_discordant_pairs | gated_only_discordant_pairs | discordant_pairs | test | p_value | analysis_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| record_level_unsupported_claim_presence | 576 | 74 | 64 | 14 | 4 | 18 | Exact McNemar via two-sided binomial test on discordant pairs, null probability 0.5 | 0.0308838 | secondary, reported separately; no Holm correction |

## Limitations and interpretation

- The primary family is the two all-record F1 endpoints; the secondary family is the two covered-only endpoints. Holm correction is performed independently within each family.
- No tests were run separately by condition, analyzer state, or gate action. Phase 21 remains the descriptive subgroup analysis.
- A corrected p-value below 0.05 is the prespecified criterion for statistical significance. Statistical significance does not establish clinical significance.
- The covered-only comparison conditions on the gate's coverage decision and applies only to the 480 covered records.
- The all-record metric retains Phase 20's existing abstention scoring; scores were not recoded for this analysis.
- The analysis does not create intermediate thresholds, alter metrics, or exclude unfavorable records.

## Reproducibility

Run `.venv\Scripts\python.exe -m src.evaluation.run_phase22_statistics` from the repository root. Focused tests: `.venv\Scripts\python.exe -m pytest tests\test_phase22_statistics.py -q`. The runner checks cohort integrity and snapshots the listed input files and baseline/gated output trees before and after writing Phase 22 outputs.
