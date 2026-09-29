# Phase 24 Manifest Addendum

This addendum records completion of Phases 20–24. Historical manifest sections were not rewritten.

- **Phase 20:** Existing RadGraph-F1 and CheXpert-F1 per-record and summary outputs were consolidated for the 576 image-eligible records.
- **Phase 21:** Descriptive risk/coverage analysis completed. Cohort coverage was 480/576 (83.33%) and abstention was 96/576 (16.67%). Unsupported rates in the consolidated package preserve Phase 21's generated-candidate denominator.
- **Phase 22:** Paired two-sided Wilcoxon signed-rank tests were run for all-record primary and covered-only secondary F1 endpoints, with separate Holm correction families. Rank-biserial effects and paired-difference diagnostics were reported. Record-level unsupported-claim presence used exact two-sided McNemar testing as a separate secondary analysis.
- **Phase 23:** Deterministic qualitative sample of 18 condition-level records (three per condition) and an 11-row error-pattern summary were produced. Repeated study IDs across conditions are expected; this is not an 18-patient sample.
- **Phase 24:** Final master, statistical, condition, gate, and qualitative tables; synthesis and paper-ready statements; and four reproducible figures were generated from existing artifacts.

## Frozen integrity hashes

- Phase 16 benchmark SHA-256: `3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53`
- Phase 17 eligibility SHA-256: `cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6`

## Validation

- Current full pytest suite: **219 passed**.
- No VLM inference or new evaluation metric was run for Phase 24.
- No frozen benchmark, eligibility, baseline, gated, Phase 19–23 result artifact was modified.
