# Phase 24: Paper-Ready Results Statements

- The eligible cohort comprised 576 records from 96 studies, with 96 records in each of six benchmark conditions. The gate produced covered reports for 480 records (83.33%) and abstained for 96 (16.67%).
- Mean RadGraph-F1 was 0.196932 for baseline and 0.166094 for gated outputs across all 576 records (mean paired delta -0.030838). The two-sided paired Wilcoxon signed-rank test returned W=10532.5, raw p=6.21056e-10, and Holm-adjusted p=1.24211e-9 across the two primary endpoints.
- Mean CheXpert-F1 was 0.193885 for baseline and 0.191842 for gated outputs across all 576 records (mean paired delta -0.002042); the paired Wilcoxon test yielded W=2134 and Holm-adjusted p=0.842698.
- Among the 480 covered records, mean RadGraph-F1 was 0.195530 for baseline and 0.199313 for gated outputs (mean paired delta +0.003782; W=6906.5; Holm-adjusted p=0.119456). Mean CheXpert-F1 was 0.190528 and 0.196877 (mean paired delta +0.006349; W=556.5; Holm-adjusted p=0.576607). Neither covered-only comparison meets alpha=0.05 after Holm correction.
- Using the Phase 21 generated-candidate denominator, unsupported-claim counts/rates were 120/1,578 (7.6046%) for baseline, 106/1,460 (7.2603%) for gated all-record, and 106/1,460 (7.2603%) for gated covered-only outputs. This denominator definition is retained explicitly.
- At record level, 74 baseline reports and 64 gated outputs contained at least one unsupported claim. Exact two-sided McNemar testing on 18 discordant pairs (14 baseline-only; 4 gated-only) yielded p=0.030884. This is a separately reported secondary analysis and is not presented as a primary corrected endpoint.
- The insufficient condition had 96 abstentions. Their all-record metric scores are the existing Phase 20 abstention scores, not scores from newly generated reports.
- The analyzer classified all 96 evidentiary-incomplete records as sufficient; the gate did not intervene and gated reports matched baseline.
- Phase 23 reviewed 18 condition-level cases, three per condition. These are not 18 independent patients; repeated study IDs across perturbation conditions are expected.

The experiment does not show universal improvement. The all-record RadGraph-F1 decrease includes abstention scoring; covered-only increases are not statistically supported after the stated correction. Claim-level and report-overlap outcomes remain distinct.
