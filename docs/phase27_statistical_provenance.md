# Phase 27 Statistical Provenance Investigation

## Scope and finding

This investigation read repository history, Phase 22 statistical code and outputs, and all checked-in Phase 26D statistical tables. No Phase 26D statistical-analysis script, notebook, or execution log exists in the repository history. Commit `c660f21aba26f24c56e5a791fc9ce873db67e537` (2026-10-06) adds the Phase 26D result tables, but no code that generated their Wilcoxon statistics.

**Interpretation: PASS WITH DOCUMENTED REPRODUCIBILITY EXCEPTION.** Substantive experiment-integrity checks pass. The exact historical RadGraph Wilcoxon p-value cannot be reproduced from surviving Phase 26D artifacts; this is recorded as a statistical-provenance limitation, not an experiment failure. All current recomputed paired and delta-vector tests are non-significant, so this exception does not change the significance conclusion. No Phase 26D result files were modified; no evaluation-set tuning occurred; lambda=0.25 remained frozen.

Accordingly, the vectors and settings below are **reconstructions that reproduce or test the frozen outputs**; they are not proof of the original Phase 26D call syntax. The repository's committed `requirements.txt` pins SciPy 1.15.3 (introduced in `5907f338ac290a64bbd01c0610a3f8d2b8f13b37`). The current `.venv` also uses SciPy 1.15.3. No record establishes that the historical Phase 26D process actually ran under that version.

## Wilcoxon results

For the reconstructions, the explicit current-environment call was two-sided Wilcoxon signed-rank with `zero_method="wilcox"`, `correction=False`, and `method="approx"`. Under these inputs this agrees with SciPy's default two-sided/uncorrected behavior. SciPy removes zero differences for ranking and uses average ranks for ties. None of the paired vectors contains NaNs; no additional record filtering was used.

| Endpoint | Available paired vectors | Frozen p (and W) | Reconstruction from paired x/y | Current audit from saved delta | Provenance conclusion |
|---|---|---:|---:|---:|---|
| Unsupported claim rate | `baseline_unsupported_rate`, `selected_unsupported_rate` in `phase26d_evaluation_paired_records.csv` (480 study-condition pairs; rates, not individual claims) | 0.318635 (master table rounds to 6 decimals) | 0.318634859854 (W=2328.5; 102 nonzero, 378 zeros) | 0.338823460501 from `delta_rate` (W=2341.0) | The paired-rate vectors reproduce the stored value at its recorded precision. The original Phase 26D code and whether it called `wilcoxon(x, y)` or an equivalent operation cannot be recovered. |
| RadGraph F1 | `f1_lambda0`, `f1_lambda025` in `phase26d_evaluation_radgraph_paired.csv` (480 pairs) | 0.3363292101181874 (W=17968.5) | 0.335764167125 (W=17967.0; 277 nonzero, 203 zeros) | 0.335576039938 from `f1_delta` (W=17966.5) | Unexplained. Neither available paired representation reproduces the frozen result. The original input precision, invocation, and any filtering are not recoverable. |
| Custom CheXpert-style F1 | `f1_lambda0`, `f1_lambda025` in `phase26d_evaluation_chexpert_paired.csv` (480 pairs) | 0.7119592027948771 (W=3973.0) | 0.711959202795 (W=3973.0; 128 nonzero, 352 zeros) | 0.707504810033 from `f1_delta` (W=3970.5) | The paired-score vectors reproduce the frozen value. The original Phase 26D code and exact invocation cannot be recovered. |

The saved deltas are numerically close to subtraction of the paired columns, but not bit-identical: 19 claim-rate deltas, 165 RadGraph deltas, and 35 CheXpert deltas differ from direct subtraction by at most `9.1e-17`, `1.4e-16`, and `1.3e-16`, respectively. With many zero differences and tied absolute differences, those last-bit differences can change rank assignments and the resulting Wilcoxon statistic. This explains why current tests on paired columns and saved delta columns can disagree for claim rate and CheXpert. It does **not** explain the RadGraph frozen p-value, which differs from both current reconstructions; attributing that discrepancy to floating-point ties alone would be unsupported.

The available Phase 26D tables do not establish whether the original call used `wilcoxon(x, y)`, `wilcoxon(x-y)`, or another equivalent formulation; nor do they establish the historical `zero_method`, `correction`, `alternative`, `method`, NaN handling, or pre-test filtering. The settings above describe reproducible current reconstructions only. The rate endpoint is computed over paired study-condition unsupported rates, not as a test over the 3,864 individual claim-comparison rows.

## McNemar result provenance

The value `p=0.0308837890625` exists in the Phase 22 generated output `results/tables/phase22_claim_statistics.csv` and in `docs/phase22_statistical_analysis.md`; it is repeated in Phase 24 summaries. The source procedure is `src/evaluation/run_phase22_statistics.py` at commit `5907f338ac290a64bbd01c0610a3f8d2b8f13b37`: claim labels were aggregated to one unsupported-presence indicator per report across 576 records, then tested by an exact two-sided binomial/McNemar test with null probability 0.5. Its counts were 74 baseline-positive, 64 gated-positive, 14 baseline-only, and 4 gated-only; `binomtest(min(14, 4), 18, p=0.5, alternative="two-sided")` reproduces the stored p-value exactly.

That is a separate Phase 22 experiment, not the Phase 26D evaluation. The current Phase 26D paired-status counts are baseline-only 30 and selected-only 33 (63 discordant pairs). Their exact two-sided result is `binomtest(30, 63, p=0.5, alternative="two-sided") = 0.801306492504`. No Phase 26D artifact or history entry was found containing `p=0.030884` for those counts.

## Files and history inspected

- `src/evaluation/run_phase22_statistics.py` and `tests/test_phase22_statistics.py`.
- `docs/phase22_statistical_analysis.md`, `results/tables/phase22_primary_statistics.csv`, `results/tables/phase22_covered_statistics.csv`, and `results/tables/phase22_claim_statistics.csv`.
- Phase 24 generated statistical tables and interpretation documents carrying forward the Phase 22 result.
- Phase 26D claim comparisons, claim summaries, paired claim records, RadGraph and CheXpert summaries/paired records/paired-stat tables, master tables, and generation table.
- `git log --all`, searches for the stored p-value strings, and the file lists for commits `5907f33` and `c660f21`.
- Notebook search: no checked-in `.ipynb` file was found for Phase 22 or Phase 26D. No Phase 26D statistical script was found.

## What the paper should report

Do not replace the frozen Phase 26D p-values. Use the independently reproducible paired-vector test in the paper-facing results where available, while preserving the frozen p-value in a separate column. The generated `results/tables/phase27_verified_statistics.csv` does this and separately records delta-vector recomputations. Disclose that the Phase 26D generating code/runtime is unavailable and that the frozen RadGraph p-value cannot be reproduced from either available paired vector. All frozen and reconstructed Phase 26D p-values remain above 0.05; the held-out evaluation does not demonstrate statistically significant overall improvement. Do not attribute the Phase 22 McNemar p-value to Phase 26D; report Phase 26D's paired-status comparison separately with its own counts and exact result if that analysis is included.
