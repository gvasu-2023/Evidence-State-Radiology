# Phase 21: Risk-Coverage and Selective Prediction Analysis

## Objective and inputs

This descriptive analysis joins the frozen Phase 16 benchmark, Phase 17 eligibility and baseline, Phase 18 gated results, Phase 19 claim evaluation, and Phase 20 RadGraph/CheXpert per-record scores. It covers only records marked `eligible_for_vlm == True`.

References are the benchmark `findings` plus `impression`. Keys are joined on `(sample_id, uid, condition)` and verified against the required `sample_id + condition` pairing. No reports or upstream artifacts were changed.

## Methodology

Coverage is covered gated records divided by all eligible records. A record is covered when its recorded gate action is not `abstain`; abstention count comes from `gate_action == abstain` and is cross-checked against Phase 20 flags.

All-record gated quality averages the existing Phase 20 per-record scores, including each abstention output exactly as Phase 20 scored it. Covered-only quality averages scores for covered records only. Abstention scores were not replaced or recoded.

Unsupported claims retain the Phase 19 definition: `ungrounded_affirmation`, `unsupported_affirmation`, and `contradiction`. Omissions remain separate. The denominator counts actual generated candidate claims from Phase 19 generated polarities. Empty fallback rows and abstention reports with no generated claims add no claims to the denominator. A zero denominator yields rate 0, matching the existing zero-denominator convention.

RadGraph-F1 and CheXpert-F1 are report-quality metrics under their existing Phase 15 scoring procedures. This report keeps coverage, unsupported-claim risk, and report-quality scores as separate quantities.

## Overall results

Eligible records: 576. Gated covered: 480; abstained: 96. Coverage: 83.3333%; abstention rate: 16.6667%.

| View | N evaluated | Coverage | Unsupported count / generated candidates | Unsupported rate | RadGraph-F1 mean / median | CheXpert-F1 mean / median |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 576 | 100.00% | 120 / 1578 | 7.6046% | 0.1969 / 0.1714 | 0.1939 / 0.0000 |
| Gated, all eligible | 576 | 83.33% | 106 / 1460 | 7.2603% | 0.1661 / 0.1481 | 0.1918 / 0.0000 |
| Gated, covered only | 480 | 83.33% cohort coverage | 106 / 1460 | 7.2603% | 0.1993 / 0.1714 | 0.1969 / 0.0000 |

Mean score deltas (gated all minus baseline): RadGraph-F1 -0.0308; CheXpert-F1 -0.0020. Covered-only minus baseline: RadGraph-F1 +0.0024; CheXpert-F1 +0.0030.

## Condition-level results

| condition | n | covered_n | abstained_n | coverage | baseline_unsupported_claim_rate | gated_all_unsupported_claim_rate | gated_covered_unsupported_claim_rate | baseline_radgraph_f1_mean | gated_all_radgraph_f1_mean | gated_covered_radgraph_f1_mean | baseline_chexpert_f1_mean | gated_all_chexpert_f1_mean | gated_covered_chexpert_f1_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sufficient | 96 | 96 | 0 | 100.00% | 8.44% | 8.44% | 8.44% | 0.1998 | 0.1998 | 0.1998 | 0.1995 | 0.1995 | 0.1995 |
| syntactic_incomplete | 96 | 96 | 0 | 100.00% | 6.56% | 5.80% | 5.80% | 0.2053 | 0.2039 | 0.2039 | 0.1872 | 0.2107 | 0.2107 |
| evidentiary_incomplete | 96 | 96 | 0 | 100.00% | 8.79% | 8.79% | 8.79% | 0.1896 | 0.1896 | 0.1896 | 0.1854 | 0.1854 | 0.1854 |
| irrelevant | 96 | 96 | 0 | 100.00% | 8.05% | 5.80% | 5.80% | 0.1847 | 0.2039 | 0.2039 | 0.2024 | 0.2107 | 0.2107 |
| conflicting | 96 | 96 | 0 | 100.00% | 8.85% | 8.85% | 8.85% | 0.1982 | 0.1993 | 0.1993 | 0.1781 | 0.1781 | 0.1781 |
| insufficient | 96 | 0 | 96 | 0.00% | 5.80% | 0.00% | 0.00% | 0.2039 | 0.0000 | — | 0.2107 | 0.1667 | — |

## Analyzer-state results

States below are taken from Phase 18 analyzer output, not inferred from benchmark conditions.

| analyzer_evidence_state | n | covered_n | abstained_n | coverage | baseline_unsupported_claim_rate | gated_all_unsupported_claim_rate | gated_covered_unsupported_claim_rate | baseline_radgraph_f1_mean | gated_all_radgraph_f1_mean | gated_covered_radgraph_f1_mean | baseline_chexpert_f1_mean | gated_all_chexpert_f1_mean | gated_covered_chexpert_f1_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sufficient | 192 | 192 | 0 | 100.00% | 8.61% | 8.61% | 8.61% | 0.1947 | 0.1947 | 0.1947 | 0.1925 | 0.1925 | 0.1925 |
| incomplete | 96 | 96 | 0 | 100.00% | 6.56% | 5.80% | 5.80% | 0.2053 | 0.2039 | 0.2039 | 0.1872 | 0.2107 | 0.2107 |
| irrelevant | 96 | 96 | 0 | 100.00% | 8.05% | 5.80% | 5.80% | 0.1847 | 0.2039 | 0.2039 | 0.2024 | 0.2107 | 0.2107 |
| conflicting | 96 | 96 | 0 | 100.00% | 8.85% | 8.85% | 8.85% | 0.1982 | 0.1993 | 0.1993 | 0.1781 | 0.1781 | 0.1781 |
| insufficient | 96 | 0 | 96 | 0.00% | 5.80% | 0.00% | 0.00% | 0.2039 | 0.0000 | — | 0.2107 | 0.1667 | — |

## Gate-action results

| gate_action | n | covered_n | abstained_n | coverage | baseline_unsupported_claim_rate | gated_all_unsupported_claim_rate | gated_covered_unsupported_claim_rate | baseline_radgraph_f1_mean | gated_all_radgraph_f1_mean | gated_covered_radgraph_f1_mean | baseline_chexpert_f1_mean | gated_all_chexpert_f1_mean | gated_covered_chexpert_f1_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generate | 192 | 192 | 0 | 100.00% | 8.61% | 8.61% | 8.61% | 0.1947 | 0.1947 | 0.1947 | 0.1925 | 0.1925 | 0.1925 |
| qualify | 96 | 96 | 0 | 100.00% | 6.56% | 5.80% | 5.80% | 0.2053 | 0.2039 | 0.2039 | 0.1872 | 0.2107 | 0.2107 |
| discount_context | 96 | 96 | 0 | 100.00% | 8.05% | 5.80% | 5.80% | 0.1847 | 0.2039 | 0.2039 | 0.2024 | 0.2107 | 0.2107 |
| qualify_or_abstain | 96 | 96 | 0 | 100.00% | 8.85% | 8.85% | 8.85% | 0.1982 | 0.1993 | 0.1993 | 0.1781 | 0.1781 | 0.1781 |
| abstain | 96 | 0 | 96 | 0.00% | 5.80% | 0.00% | 0.00% | 0.2039 | 0.0000 | — | 0.2107 | 0.1667 | — |

## Interpretation and limitations

1. The gate abstains on all 96 insufficient records, yielding coverage of 83.33%.
2. Covered-only metrics describe reports the system chose to produce; all-record metrics include the abstention outputs as scored by Phase 20.
3. The `evidentiary_incomplete` condition was classified as `sufficient` by the current analyzer and remains fully covered. This observed analyzer limitation is preserved, not corrected here.
4. Only one deterministic operating point is evaluated. This is not a continuous risk-coverage curve.
5. Results are descriptive. They do not establish statistical significance or causal effects.
6. RadGraph-F1 and CheXpert-F1 measure overlap under their existing implementations; neither alone establishes clinical safety or overall factual reliability.
7. Abstention has no generated claim candidates. Its claim denominator is zero and it contributes no artificial claims; zero-denominator unsupported rate is reported as zero.

## Reproducibility

Run ` .venv\Scripts\python.exe -m src.evaluation.run_phase21_risk_coverage` from the repository root, then run focused tests with ` .venv\Scripts\python.exe -m pytest tests\test_phase21_risk_coverage.py -q`.
Input hashes and full baseline/gated output-tree hashes are checked before and after analysis. The script stops on cohort, key, state, action, or count mismatches.

Generated tables: `phase21_risk_coverage_summary.csv`, `phase21_condition_coverage.csv`, `phase21_analyzer_state_summary.csv`, `phase21_gate_action_summary.csv`, and `phase21_claim_risk_summary.csv`.
