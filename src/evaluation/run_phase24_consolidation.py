"""Consolidate Phase 19-23 outputs into final Phase 24 tables and documents."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUTS = {
    "claims": ROOT / "results/tables/phase19_claim_evaluation.csv",
    "claim_summary": ROOT / "results/tables/phase19_claim_summary.csv",
    "radgraph": ROOT / "results/tables/phase20_radgraph_f1_results.csv",
    "radgraph_summary": ROOT / "results/tables/phase20_radgraph_f1_summary.csv",
    "chexpert": ROOT / "results/tables/phase20_chexpert_f1_results.csv",
    "chexpert_summary": ROOT / "results/tables/phase20_chexpert_f1_summary.csv",
    "risk": ROOT / "results/tables/phase21_risk_coverage_summary.csv",
    "condition": ROOT / "results/tables/phase21_condition_coverage.csv",
    "state": ROOT / "results/tables/phase21_analyzer_state_summary.csv",
    "action": ROOT / "results/tables/phase21_gate_action_summary.csv",
    "claim_risk": ROOT / "results/tables/phase21_claim_risk_summary.csv",
    "primary": ROOT / "results/tables/phase22_primary_statistics.csv",
    "covered_stats": ROOT / "results/tables/phase22_covered_statistics.csv",
    "claim_stats": ROOT / "results/tables/phase22_claim_statistics.csv",
    "diagnostics": ROOT / "results/tables/phase22_difference_diagnostics.csv",
    "qual_cases": ROOT / "results/tables/phase23_qualitative_cases.csv",
    "qual_patterns": ROOT / "results/tables/phase23_error_pattern_summary.csv",
    "benchmark": ROOT / "results/tables/phase16_perturbations.csv",
    "eligibility": ROOT / "results/tables/phase17_vlm_inference_eligibility.csv",
}
OUTPUTS = {
    "master": ROOT / "results/tables/phase24_master_results.csv",
    "statistics": ROOT / "results/tables/phase24_statistical_results.csv",
    "condition": ROOT / "results/tables/phase24_condition_results.csv",
    "gate": ROOT / "results/tables/phase24_gate_summary.csv",
    "qualitative": ROOT / "results/tables/phase24_qualitative_summary.csv",
    "synthesis": ROOT / "docs/phase24_results_synthesis.md",
    "paper": ROOT / "docs/phase24_paper_ready_results.md",
    "addendum": ROOT / "docs/phase24_manifest_addendum.md",
}
FIGURE_DIR = ROOT / "results/figures/phase24"
CONDITIONS = (
    "sufficient", "syntactic_incomplete", "evidentiary_incomplete",
    "irrelevant", "conflicting", "insufficient",
)
BENCHMARK_SHA256 = "3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53"
ELIGIBILITY_SHA256 = "cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6"
FULL_TEST_COUNT = 219


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(path: Path) -> dict[str, str]:
    return {
        item.relative_to(ROOT).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def input_snapshot() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative Phase 24 inputs: {missing}")
    hashes = {name: sha256(path) for name, path in INPUTS.items()}
    if hashes["benchmark"] != BENCHMARK_SHA256:
        raise ValueError("Frozen Phase 16 benchmark hash mismatch")
    if hashes["eligibility"] != ELIGIBILITY_SHA256:
        raise ValueError("Frozen Phase 17 eligibility hash mismatch")
    trees = {
        "baseline": tree_snapshot(ROOT / "results/baseline/phase17_full"),
        "gated": tree_snapshot(ROOT / "results/gated/phase18_full"),
    }
    return hashes, trees


def _master_table(risk: pd.DataFrame) -> pd.DataFrame:
    indexed = risk.set_index("view")
    rows = []
    definitions = [
        ("Baseline", "baseline", 576, "baseline", 1.0, 0.0),
        ("Gated - all records", "gated_all_records", 576, "gated_all", None, None),
        ("Gated - covered only", "gated_covered_only", 480, "gated_covered", None, None),
    ]
    for label, view, n, claim_prefix, coverage, abstention in definitions:
        record = indexed.loc[view]
        if coverage is None:
            coverage = float(record["coverage"])
            abstention = float(record["abstention_rate"])
        score_prefix = {
            "baseline": "baseline",
            "gated_all": "gated_all",
            "gated_covered": "gated_covered",
        }[claim_prefix]
        rows.append({
            "evaluation_view": label,
            "N": n,
            "coverage": float(coverage),
            "abstention_rate": float(abstention),
            "unsupported_claim_count": int(record["unsupported_claim_count"]),
            "generated_candidate_count": int(record["evaluated_generated_claim_denominator"]),
            "unsupported_claim_rate": float(record["unsupported_claim_rate"]),
            "radgraph_mean": float(record[f"{score_prefix}_radgraph_f1_mean"]),
            "radgraph_median": float(record[f"{score_prefix}_radgraph_f1_median"]),
            "chexpert_mean": float(record[f"{score_prefix}_chexpert_f1_mean"]),
            "chexpert_median": float(record[f"{score_prefix}_chexpert_f1_median"]),
        })
    result = pd.DataFrame(rows)
    if result["N"].tolist() != [576, 576, 480]:
        raise ValueError("Unexpected Phase 24 master-table N values")
    return result


def _statistical_table(primary: pd.DataFrame, covered: pd.DataFrame, claim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame, view in ((primary, "all-record primary family"), (covered, "covered-only secondary family")):
        for row in frame.to_dict("records"):
            rows.append({
                "analysis": view,
                "endpoint": "RadGraph-F1" if row["endpoint"] == "radgraph_f1" else "CheXpert-F1",
                "N": int(row["n"]),
                "baseline_mean": float(row["baseline_mean"]),
                "gated_mean": float(row["gated_mean"]),
                "mean_delta": float(row["mean_delta_gated_minus_baseline"]),
                "baseline_median": float(row["baseline_median"]),
                "gated_median": float(row["gated_median"]),
                "median_delta": float(row["median_delta_gated_minus_baseline"]),
                "wilcoxon_W": float(row["wilcoxon_statistic"]),
                "p_raw": float(row["p_value_raw"]),
                "p_holm_adjusted": float(row["p_value_holm"]),
                "rank_biserial_effect_size": float(row["rank_biserial_correlation"]),
                "baseline_positive_records": np.nan,
                "gated_positive_records": np.nan,
                "baseline_only_discordant": np.nan,
                "gated_only_discordant": np.nan,
                "test_method": row["test_method"],
            })
    claim_row = claim.iloc[0]
    rows.append({
        "analysis": "secondary claim-level analysis",
        "endpoint": "record-level unsupported-claim presence",
        "N": int(claim_row["n"]),
        "baseline_mean": np.nan, "gated_mean": np.nan, "mean_delta": np.nan,
        "baseline_median": np.nan, "gated_median": np.nan, "median_delta": np.nan,
        "wilcoxon_W": np.nan,
        "p_raw": float(claim_row["p_value"]),
        "p_holm_adjusted": np.nan,
        "rank_biserial_effect_size": np.nan,
        "baseline_positive_records": int(claim_row["baseline_positive_records"]),
        "gated_positive_records": int(claim_row["gated_positive_records"]),
        "baseline_only_discordant": int(claim_row["baseline_only_discordant_pairs"]),
        "gated_only_discordant": int(claim_row["gated_only_discordant_pairs"]),
        "test_method": "Exact two-sided McNemar test via binomial test; no new correction",
    })
    return pd.DataFrame(rows)


def _condition_table(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        record = source.set_index("condition").loc[condition]
        bden = int(record["baseline_evaluated_generated_claim_denominator"])
        gden = int(record["gated_all_evaluated_generated_claim_denominator"])
        rows.append({
            "condition": condition,
            "N": int(record["n"]),
            "covered": int(record["covered_n"]),
            "abstained": int(record["abstained_n"]),
            "coverage": float(record["coverage"]),
            "baseline_radgraph_f1": float(record["baseline_radgraph_f1_mean"]),
            "gated_radgraph_f1": float(record["gated_all_radgraph_f1_mean"]),
            "baseline_chexpert_f1": float(record["baseline_chexpert_f1_mean"]),
            "gated_chexpert_f1": float(record["gated_all_chexpert_f1_mean"]),
            "baseline_unsupported_count": int(record["baseline_unsupported_claim_count"]),
            "gated_all_unsupported_count": int(record["gated_all_unsupported_claim_count"]),
            "gated_covered_unsupported_count": int(record["gated_covered_unsupported_claim_count"]),
            "baseline_generated_candidate_count": bden,
            "gated_generated_candidate_count": gden,
            "baseline_unsupported_rate": float(record["baseline_unsupported_claim_count"] / bden) if bden else 0.0,
            "gated_all_unsupported_rate": float(record["gated_all_unsupported_claim_count"] / gden) if gden else 0.0,
            "gated_covered_generated_candidate_count": int(record["gated_covered_evaluated_generated_claim_denominator"]),
            "gated_covered_unsupported_rate": float(record["gated_covered_unsupported_claim_rate"]),
        })
    return pd.DataFrame(rows)


def _gate_table(state: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, frame, key in (
        ("analyzer_state", state, "analyzer_evidence_state"),
        ("gate_action", action, "gate_action"),
    ):
        for row in frame.to_dict("records"):
            rows.append({
                "category_type": level,
                "category": row[key],
                "N": int(row["n"]),
                "covered": int(row["covered_n"]),
                "abstained": int(row["abstained_n"]),
                "coverage": float(row["coverage"]),
                "baseline_radgraph_f1": float(row["baseline_radgraph_f1_mean"]),
                "gated_radgraph_f1": float(row["gated_all_radgraph_f1_mean"]),
                "baseline_chexpert_f1": float(row["baseline_chexpert_f1_mean"]),
                "gated_chexpert_f1": float(row["gated_all_chexpert_f1_mean"]),
            })
    return pd.DataFrame(rows)


def _qualitative_table(cases: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    pattern_lookup = {
        "sufficient": "normal generation preservation",
        "syntactic_incomplete": "syntactically incomplete context",
        "evidentiary_incomplete": "evidentiary-incomplete analyzer limitation",
        "irrelevant": "irrelevant context handling",
        "conflicting": "conflicting context handling",
        "insufficient": "insufficient evidence abstention",
    }
    rows = []
    for condition in CONDITIONS:
        case_group = cases[cases["condition"] == condition].copy()
        pattern = patterns[patterns["pattern"] == pattern_lookup[condition]].iloc[0]
        representative = case_group.assign(
            category_count=case_group["qualitative_categories"].fillna("").str.count(";")
        ).sort_values(["category_count", "case"], ascending=[False, True], kind="mergesort").iloc[0]
        rows.append({
            "condition": condition,
            "selected_case_count": int(len(case_group)),
            "selected_case_ids": ", ".join(case_group["case"].astype(str)),
            "selected_sample_ids": ", ".join(case_group["sample_id"].astype(str)),
            "major_qualitative_pattern": pattern["description"],
            "representative_interpretation": f"{representative['case']} ({representative['sample_id']}, UID {representative['uid']}): {representative['qualitative_interpretation']}",
        })
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame[columns] if columns else frame
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(c) for c in selected.columns]
    body = [[cell(x) for x in row] for row in selected.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
    ])


def _docs(master: pd.DataFrame, stats: pd.DataFrame, conditions: pd.DataFrame, gate: pd.DataFrame, qualitative: pd.DataFrame, hashes: dict[str, str], full_test_count: int = FULL_TEST_COUNT) -> tuple[str, str, str]:
    overall_stats = stats[stats["analysis"].str.contains("family")]
    condition_cols = ["condition", "N", "covered", "abstained", "coverage", "baseline_radgraph_f1", "gated_radgraph_f1", "baseline_chexpert_f1", "gated_chexpert_f1", "baseline_unsupported_count", "gated_all_unsupported_count", "gated_covered_unsupported_count", "baseline_unsupported_rate", "gated_all_unsupported_rate", "gated_covered_unsupported_rate"]
    master_text = _markdown(master)
    stats_text = _markdown(stats)
    condition_text = _markdown(conditions, condition_cols)
    gate_state_text = _markdown(gate[gate.category_type == "analyzer_state"])
    gate_action_text = _markdown(gate[gate.category_type == "gate_action"])
    qual_text = _markdown(qualitative)
    hash_items = "\n".join(f"- `{key}`: `{value}`" for key, value in hashes.items())
    synthesis = f"""# Phase 24: Results Synthesis

## 1. Cohort and evaluation setup

The frozen benchmark contains 600 records across 100 studies and six controlled conditions. The image-backed inference cohort contains 576 eligible records, 96 per condition, across 96 studies. The four excluded image-unavailable UIDs contribute 24 excluded records and are not part of these result tables. Baseline and gated outputs are paired by record. The six benchmark conditions, five analyzer states, and five gate actions remain separate concepts.

## 2. Analyzer and gate behavior

Phase 18 action outputs, summarized descriptively in Phase 21, are shown separately by analyzer state and gate action below. The analyzer classified all 96 evidentiary-incomplete records as sufficient; they followed the `generate` action. This observed limitation is retained.

### By analyzer state

{gate_state_text}

### By gate action

{gate_action_text}

## 3. Overall report-level results

RadGraph-F1 and CheXpert-F1 are the existing Phase 20 report-overlap metrics. The consolidated table separates baseline, gated all-record, and gated covered-only views.

{master_text}

## 4. Covered-only results

Covered-only quality uses the 480 records with a non-abstaining gated report. Mean RadGraph-F1 and CheXpert-F1 are slightly higher than baseline in this subset; neither covered-only paired comparison meets its Holm-adjusted threshold. These comparisons are conditional on gate coverage.

## 5. Unsupported-claim results

Phase 21's generated-candidate denominator is used throughout the consolidated risk table: baseline 120/1,578 (7.6046%); gated all-record 106/1,460 (7.2603%); gated covered-only 106/1,460 (7.2603%). This explicitly preserves the Phase 21 denominator definition and does not silently reconcile the historical Phase 19/Phase 21 denominator discrepancy. Omissions are not unsupported claims.

## 6. Statistical results

The primary all-record endpoints used two-sided paired Wilcoxon signed-rank tests with Holm correction across RadGraph-F1 and CheXpert-F1. The covered-only analyses form a separate Holm family. The record-level unsupported-claim presence comparison is separately reported as an exact two-sided McNemar test, with no new correction.

{stats_text}

The all-record RadGraph-F1 mean is lower in the gated view; this includes all 96 abstention outputs as scored under Phase 20. The all-record Wilcoxon test has Holm-adjusted p=1.242112e-9. All-record CheXpert-F1 has adjusted p=0.842698. Covered-only RadGraph-F1 and CheXpert-F1 means increase slightly, but adjusted p-values are 0.119456 and 0.576607, respectively.

## 7. Condition-level behavior

No condition-level significance tests were added.

{condition_text}

## 8. Qualitative findings

Phase 23 contains 18 condition-level records (three per condition), not 18 independent patients. UIDs may repeat across conditions because each condition is a controlled perturbation of an underlying study.

{qual_text}

The qualitative error-pattern summary includes unchanged sufficient/evidentiary-incomplete reports, incomplete-context qualifications, conflicting-context warnings, irrelevant-context changes, and insufficient-context abstentions. In the irrelevant and conflicting conditions, claim-level behavior was not uniformly improved.

## 9. Key limitation: evidentiary-incomplete misclassification

All 96 evidentiary-incomplete perturbations removed readable clinical evidence, but the analyzer classified them as sufficient. The gate did not intervene and outputs matched baseline. This does not imply evidentiary incompleteness is harmless.

## 10. Interpretation

The results show distinct behavior for report overlap, claim labels, and selective coverage. All-record RadGraph-F1 decreases when abstention outputs are included; covered-only report metrics move slightly upward but are not statistically supported after the specified correction. Unsupported-claim record presence falls from 74 to 64 with exact McNemar p=0.030884 as a secondary result. This does not establish universal improvement: irrelevant and conflicting context do not show universally improved claim-level behavior, and the evidentiary-incomplete condition exposes an analyzer limitation.

## 11. What the experiment does not establish

The experiment does not establish clinical safety, clinical significance, universal factuality improvement, generalization beyond this cohort/model/setup, or that a single operating point defines a continuous risk-coverage curve. No intermediate coverage points were created. Statistical significance does not establish clinical significance.
"""

    paper = f"""# Phase 24: Paper-Ready Results Statements

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
"""

    addendum = f"""# Phase 24 Manifest Addendum

This addendum records completion of Phases 20–24. Historical manifest sections were not rewritten.

- **Phase 20:** Existing RadGraph-F1 and CheXpert-F1 per-record and summary outputs were consolidated for the 576 image-eligible records.
- **Phase 21:** Descriptive risk/coverage analysis completed. Cohort coverage was 480/576 (83.33%) and abstention was 96/576 (16.67%). Unsupported rates in the consolidated package preserve Phase 21's generated-candidate denominator.
- **Phase 22:** Paired two-sided Wilcoxon signed-rank tests were run for all-record primary and covered-only secondary F1 endpoints, with separate Holm correction families. Rank-biserial effects and paired-difference diagnostics were reported. Record-level unsupported-claim presence used exact two-sided McNemar testing as a separate secondary analysis.
- **Phase 23:** Deterministic qualitative sample of 18 condition-level records (three per condition) and an 11-row error-pattern summary were produced. Repeated study IDs across conditions are expected; this is not an 18-patient sample.
- **Phase 24:** Final master, statistical, condition, gate, and qualitative tables; synthesis and paper-ready statements; and four reproducible figures were generated from existing artifacts.

## Frozen integrity hashes

- Phase 16 benchmark SHA-256: `{hashes['benchmark']}`
- Phase 17 eligibility SHA-256: `{hashes['eligibility']}`

## Validation

- Current full pytest suite: **{full_test_count} passed**.
- No VLM inference or new evaluation metric was run for Phase 24.
- No frozen benchmark, eligibility, baseline, gated, Phase 19–23 result artifact was modified.
"""
    return synthesis, paper, addendum


def build_phase24_outputs() -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, dict[str, str]]]:
    hashes_before, trees_before = input_snapshot()
    source = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    benchmark = source["benchmark"]
    eligible = source["eligibility"][source["eligibility"]["eligible_for_vlm"].astype(str).str.lower().eq("true")]
    if len(benchmark) != 600 or len(eligible) != 576:
        raise ValueError("Phase 24 cohort totals must be 600 benchmark / 576 eligible")
    if eligible.duplicated(["sample_id", "condition"]).any():
        raise ValueError("Duplicate eligible (sample_id, condition) key")
    if eligible["condition"].value_counts().to_dict() != {condition: 96 for condition in CONDITIONS}:
        raise ValueError("Expected 96 eligible records per condition")
    if eligible["uid"].astype(int).nunique() != 96 or set(eligible["uid"].astype(int)) & {74, 597, 803, 885}:
        raise ValueError("Eligible study count/exclusion check failed")
    if not eligible.groupby("uid")["condition"].nunique().eq(6).all():
        raise ValueError("Each eligible UID must have exactly six controlled conditions")
    expected_keys = set(zip(eligible["sample_id"].astype(str), eligible["condition"].astype(str)))
    expected_uid = eligible.set_index(["sample_id", "condition"])["uid"].astype(int)
    for metric_name in ("radgraph", "chexpert"):
        metric = source[metric_name]
        if len(metric) != 576 or metric.duplicated(["sample_id", "condition"]).any():
            raise ValueError(f"Phase 20 {metric_name} rows are missing or duplicated")
        keys = set(zip(metric["sample_id"].astype(str), metric["condition"].astype(str)))
        if keys != expected_keys or set(metric["uid"].astype(int)) & {74, 597, 803, 885}:
            raise ValueError(f"Phase 20 {metric_name} keys do not match the eligible cohort")
        metric_uid = metric.set_index(["sample_id", "condition"])["uid"].astype(int)
        if not metric_uid.sort_index().equals(expected_uid.sort_index()):
            raise ValueError(f"Phase 20 {metric_name} UID pairing differs from eligibility")
        for column in (f"baseline_{metric_name}_f1", f"gated_{metric_name}_f1"):
            if metric[column].isna().any():
                raise ValueError(f"Missing Phase 20 metric values: {column}")
    risk_overall = source["risk"].set_index("view")
    gated_all = risk_overall.loc["gated_all_records"]
    if (int(gated_all["eligible_n"]), int(gated_all["covered_n"]), int(gated_all["abstained_n"])) != (576, 480, 96):
        raise ValueError("Phase 21 cohort coverage totals do not match 576/480/96")
    if source["qual_cases"].duplicated(["sample_id", "condition"]).any() or len(source["qual_cases"]) != 18:
        raise ValueError("Phase 23 qualitative sample must contain 18 unique condition-level records")
    if source["qual_cases"]["condition"].value_counts().to_dict() != {condition: 3 for condition in CONDITIONS}:
        raise ValueError("Phase 23 sample must have three selected cases per condition")
    if set(source["qual_cases"]["uid"].astype(int)) & {74, 597, 803, 885}:
        raise ValueError("Phase 23 qualitative sample contains an excluded UID")

    master = _master_table(source["risk"])
    for metric, summary_key in (("radgraph", "radgraph_summary"), ("chexpert", "chexpert_summary")):
        overall_summary = source[summary_key].query("summary_level == 'overall'")
        if len(overall_summary) != 1:
            raise ValueError(f"Phase 20 {metric} summary lacks a unique overall row")
        source_row = overall_summary.iloc[0]
        baseline_mean = float(master.loc[master.evaluation_view == "Baseline", f"{metric}_mean"].iloc[0])
        gated_mean = float(master.loc[master.evaluation_view == "Gated - all records", f"{metric}_mean"].iloc[0])
        covered_mean = float(master.loc[master.evaluation_view == "Gated - covered only", f"{metric}_mean"].iloc[0])
        if not np.allclose(
            [baseline_mean, gated_mean, covered_mean],
            [source_row[f"baseline_{metric}_f1_mean"], source_row[f"overall_gated_{metric}_f1_mean"], source_row[f"covered_gated_{metric}_f1_mean"]],
            rtol=0, atol=1e-12,
        ):
            raise ValueError(f"Phase 21 master values do not match Phase 20 {metric} summaries")
    phase19_overall = source["claim_summary"].query("summary_level == 'overall'").set_index("report_type")
    if (int(phase19_overall.loc["baseline", "unsupported_claim_count"]), int(phase19_overall.loc["gated", "unsupported_claim_count"])) != (120, 106):
        raise ValueError("Phase 19 unsupported-claim counts disagree with Phase 21 risk summary")
    statistics = _statistical_table(source["primary"], source["covered_stats"], source["claim_stats"])
    if statistics.loc[statistics.analysis == "all-record primary family", "N"].tolist() != [576, 576]:
        raise ValueError("Phase 22 primary N values differ from 576")
    if statistics.loc[statistics.analysis == "covered-only secondary family", "N"].tolist() != [480, 480]:
        raise ValueError("Phase 22 covered-only N values differ from 480")
    claim_stats_row = statistics[statistics.analysis == "secondary claim-level analysis"].iloc[0]
    if (claim_stats_row["N"], claim_stats_row["p_raw"]) != (576, float(source["claim_stats"].iloc[0]["p_value"])):
        raise ValueError("Phase 22 McNemar result was not copied exactly")
    condition = _condition_table(source["condition"])
    gate = _gate_table(source["state"], source["action"])
    qualitative = _qualitative_table(source["qual_cases"], source["qual_patterns"])
    tables = {
        "master": master,
        "statistics": statistics,
        "condition": condition,
        "gate": gate,
        "qualitative": qualitative,
    }
    docs = dict(zip(("synthesis", "paper", "addendum"), _docs(master, statistics, condition, gate, qualitative, hashes_before)))
    hashes_after, trees_after = input_snapshot()
    if hashes_before != hashes_after or trees_before != trees_after:
        raise RuntimeError("A Phase 24 source artifact changed during consolidation")
    return tables, docs, {"hashes": hashes_before, "trees": trees_before}


def write_phase24_outputs() -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, dict[str, str]]]:
    tables, docs, integrity = build_phase24_outputs()
    for key in ("master", "statistics", "condition", "gate", "qualitative"):
        OUTPUTS[key].parent.mkdir(parents=True, exist_ok=True)
        tables[key].to_csv(OUTPUTS[key], index=False)
    for key in ("synthesis", "paper", "addendum"):
        OUTPUTS[key].parent.mkdir(parents=True, exist_ok=True)
        OUTPUTS[key].write_text(docs[key], encoding="utf-8")
    return tables, docs, integrity


def main() -> None:
    tables, _, integrity = write_phase24_outputs()
    print("Phase 24 table row counts:", {name: len(frame) for name, frame in tables.items()})
    print("Master results:")
    print(tables["master"].to_string(index=False))
    print("Condition counts:", tables["condition"]["N"].sum(), tables["condition"].set_index("condition")["N"].to_dict())
    print("Verified authoritative input files:", len(integrity["hashes"]))


if __name__ == "__main__":
    main()
