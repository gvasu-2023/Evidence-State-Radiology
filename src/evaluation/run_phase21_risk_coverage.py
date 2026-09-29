"""Descriptive Phase 21 risk/coverage analysis over frozen Phases 17-20 artifacts."""

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

BENCHMARK_PATH = ROOT / "results/tables/phase16_perturbations.csv"
ELIGIBILITY_PATH = ROOT / "results/tables/phase17_vlm_inference_eligibility.csv"
BASELINE_PATH = ROOT / "results/baseline/phase17_full/baseline_results.csv"
GATED_PATH = ROOT / "results/gated/phase18_full/gated_results.csv"
CLAIM_DETAIL_PATH = ROOT / "results/tables/phase19_claim_evaluation.csv"
CLAIM_SUMMARY_PATH = ROOT / "results/tables/phase19_claim_summary.csv"
RADGRAPH_RESULTS_PATH = ROOT / "results/tables/phase20_radgraph_f1_results.csv"
RADGRAPH_SUMMARY_PATH = ROOT / "results/tables/phase20_radgraph_f1_summary.csv"
CHEXPERT_RESULTS_PATH = ROOT / "results/tables/phase20_chexpert_f1_results.csv"
CHEXPERT_SUMMARY_PATH = ROOT / "results/tables/phase20_chexpert_f1_summary.csv"

OUTPUTS = {
    "risk": ROOT / "results/tables/phase21_risk_coverage_summary.csv",
    "condition": ROOT / "results/tables/phase21_condition_coverage.csv",
    "state": ROOT / "results/tables/phase21_analyzer_state_summary.csv",
    "action": ROOT / "results/tables/phase21_gate_action_summary.csv",
    "claim": ROOT / "results/tables/phase21_claim_risk_summary.csv",
    "docs": ROOT / "docs/phase21_risk_coverage_analysis.md",
}

FROZEN_BENCHMARK_SHA256 = "3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53"
FROZEN_ELIGIBILITY_SHA256 = "cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6"
EXCLUDED_UIDS = {74, 597, 803, 885}
CONDITIONS = (
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
)
STATES = ("sufficient", "incomplete", "irrelevant", "conflicting", "insufficient")
ACTIONS = ("generate", "qualify", "discount_context", "qualify_or_abstain", "abstain")
UNSUPPORTED_LABELS = {"ungrounded_affirmation", "unsupported_affirmation", "contradiction"}
ALL_LABELS = (
    "supported",
    "omission",
    "extra_negation",
    "ungrounded_affirmation",
    "unsupported_affirmation",
    "contradiction",
    "unclear",
)
KEYS = ["sample_id", "uid", "condition"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _key_set(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    return {
        (str(row.sample_id), int(row.uid), str(row.condition))
        for row in frame.itertuples(index=False)
    }


def _load_and_join() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    files = {
        "benchmark": BENCHMARK_PATH,
        "eligibility": ELIGIBILITY_PATH,
        "baseline": BASELINE_PATH,
        "gated": GATED_PATH,
        "claim_detail": CLAIM_DETAIL_PATH,
        "claim_summary": CLAIM_SUMMARY_PATH,
        "radgraph": RADGRAPH_RESULTS_PATH,
        "radgraph_summary": RADGRAPH_SUMMARY_PATH,
        "chexpert": CHEXPERT_RESULTS_PATH,
        "chexpert_summary": CHEXPERT_SUMMARY_PATH,
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 21 {name} input is missing: {path}")
    hashes = {name: sha256(path) for name, path in files.items()}
    if hashes["benchmark"] != FROZEN_BENCHMARK_SHA256:
        raise ValueError("Frozen benchmark SHA-256 does not match the authoritative hash")
    if hashes["eligibility"] != FROZEN_ELIGIBILITY_SHA256:
        raise ValueError("Eligibility SHA-256 does not match the authoritative hash")

    benchmark = pd.read_csv(BENCHMARK_PATH)
    eligibility_all = pd.read_csv(ELIGIBILITY_PATH)
    eligible = eligibility_all[
        eligibility_all["eligible_for_vlm"].astype(str).str.lower() == "true"
    ].copy()
    baseline = pd.read_csv(BASELINE_PATH)
    gated = pd.read_csv(GATED_PATH)
    claims = pd.read_csv(CLAIM_DETAIL_PATH).fillna(
        {
            "finding": "",
            "baseline_generated_polarity": "NOT_MENTIONED",
            "gated_generated_polarity": "NOT_MENTIONED",
            "baseline_label": "",
            "gated_label": "",
        }
    )
    claim_summary = pd.read_csv(CLAIM_SUMMARY_PATH)
    radgraph = pd.read_csv(RADGRAPH_RESULTS_PATH)
    radgraph_summary = pd.read_csv(RADGRAPH_SUMMARY_PATH)
    chexpert = pd.read_csv(CHEXPERT_RESULTS_PATH)
    chexpert_summary = pd.read_csv(CHEXPERT_SUMMARY_PATH)

    for name, frame in (
        ("eligible", eligible),
        ("baseline", baseline),
        ("gated", gated),
        ("RadGraph", radgraph),
        ("CheXpert", chexpert),
    ):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate (sample_id, uid, condition) key in {name}")
    if len(benchmark) != 600 or len(eligible) != 576:
        raise ValueError(f"Expected benchmark/eligible sizes 600/576, got {len(benchmark)}/{len(eligible)}")
    if eligible["condition"].value_counts().to_dict() != {condition: 96 for condition in CONDITIONS}:
        raise ValueError("Eligible cohort does not contain 96 records per condition")
    if eligible["uid"].nunique() != 96 or set(eligible["uid"].astype(int)) & EXCLUDED_UIDS:
        raise ValueError("Eligible UID set/count failed Phase 21 integrity checks")
    eligible_uid_counts = eligible.groupby("uid")["condition"].nunique()
    if not (eligible_uid_counts == 6).all():
        raise ValueError("An eligible study does not have exactly six conditions")

    eligible_keys = _key_set(eligible)
    for name, frame in (("baseline", baseline), ("gated", gated), ("RadGraph", radgraph), ("CheXpert", chexpert)):
        if len(frame) != 576 or _key_set(frame) != eligible_keys:
            raise ValueError(f"{name} does not join one-to-one with all eligible records")

    # Phase 19 is a claim-level paired table; every record must appear at least once.
    phase19_keys = set(
        zip(claims["sample_id"].astype(str), claims["uid"].astype(int), claims["benchmark_condition"].astype(str))
    )
    expected_phase19_keys = {
        (sample_id, uid, condition)
        for sample_id, uid, condition in eligible_keys
    }
    if phase19_keys != expected_phase19_keys:
        raise ValueError("Phase 19 claim evaluation does not represent exactly all eligible records")
    if claims.duplicated(["sample_id", "uid", "benchmark_condition", "finding"]).any():
        raise ValueError("Duplicate Phase 19 claim-level key found")

    base = eligible[["sample_id", "uid", "condition", "image_path"]].merge(
        benchmark[["sample_id", "uid", "condition", "findings", "impression"]],
        how="left",
        on=KEYS,
        validate="one_to_one",
    )
    base = base.merge(
        gated[["sample_id", "uid", "condition", "evidence_state", "gate_action", "gated_report"]],
        how="left",
        on=KEYS,
        validate="one_to_one",
    ).rename(columns={"evidence_state": "analyzer_evidence_state"})
    base = base.merge(
        baseline[["sample_id", "uid", "condition", "generated_report"]],
        how="left",
        on=KEYS,
        validate="one_to_one",
    )
    base = base.merge(
        radgraph,
        how="left",
        on=KEYS,
        validate="one_to_one",
        suffixes=("", "_radgraph"),
    )
    # Remove duplicate metadata columns from metric tables, retaining the Phase 18 source.
    for column in ("analyzer_evidence_state_radgraph", "gate_action_radgraph", "is_gated_abstention", "gated_report_covered"):
        if column in base.columns:
            if column in {"is_gated_abstention", "gated_report_covered"}:
                continue
            base = base.drop(columns=[column])
    base = base.merge(
        chexpert[[
            "sample_id", "uid", "condition", "baseline_chexpert_f1", "gated_chexpert_f1"
        ]],
        how="left",
        on=KEYS,
        validate="one_to_one",
    )

    phase20_abstention = base["is_gated_abstention"].astype(bool)
    phase20_covered = base["gated_report_covered"].astype(bool)
    base["is_gated_abstention"] = base["gate_action"].astype(str) == "abstain"
    base["gated_report_covered"] = ~base["is_gated_abstention"]
    if len(base) != 576:
        raise RuntimeError(f"Join silently dropped or multiplied records: {len(base)} rows")
    action_abstention = base["gate_action"].astype(str) == "abstain"
    if not (phase20_abstention == action_abstention).all():
        raise ValueError("Gate-action abstention flags disagree with Phase 20 flags")
    if not (phase20_covered == ~action_abstention).all():
        raise ValueError("Gate-action coverage flags disagree with Phase 20 flags")
    if set(base["analyzer_evidence_state"].astype(str)) != set(STATES):
        raise ValueError("Analyzer state set differs from the expected five existing states")
    if set(base["gate_action"].astype(str)) != set(ACTIONS):
        raise ValueError("Gate action set differs from the existing five actions")

    # Build per-record claim counts from actual Phase 19 generated polarities.
    candidate_claims = claims[
        claims["finding"].astype(str).str.strip().ne("")
    ].copy()
    for side, polarity_column, label_column in (
        ("baseline", "baseline_generated_polarity", "baseline_label"),
        ("gated", "gated_generated_polarity", "gated_label"),
    ):
        actual = candidate_claims[candidate_claims[polarity_column].astype(str) != "NOT_MENTIONED"].copy()
        key_columns = ["sample_id", "uid", "benchmark_condition", "finding"]
        if actual.duplicated(key_columns).any():
            raise ValueError(f"Phase 19 has duplicate extracted {side} claims")
        actual["unsupported_indicator"] = actual[label_column].isin(UNSUPPORTED_LABELS).astype(int)
        aggregates = actual.groupby(["sample_id", "uid", "benchmark_condition"], as_index=False).agg(
            **{
                f"{side}_generated_claim_count": (polarity_column, "size"),
                f"{side}_unsupported_claim_count": ("unsupported_indicator", "sum"),
            }
        )
        omissions = (
            claims[claims[label_column].astype(str) == "omission"]
            .groupby(["sample_id", "uid", "benchmark_condition"], as_index=False)
            .size()
            .rename(columns={"size": f"{side}_omission_count"})
        )
        aggregates = aggregates.merge(
            omissions,
            on=["sample_id", "uid", "benchmark_condition"],
            how="outer",
            validate="one_to_one",
        )
        base = base.merge(
            aggregates,
            left_on=["sample_id", "uid", "condition"],
            right_on=["sample_id", "uid", "benchmark_condition"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["benchmark_condition"])
        for column in (
            f"{side}_generated_claim_count",
            f"{side}_unsupported_claim_count",
            f"{side}_omission_count",
        ):
            base[column] = base[column].fillna(0).astype(int)

    base["baseline_unsupported_claim_rate"] = np.divide(
        base["baseline_unsupported_claim_count"],
        base["baseline_generated_claim_count"],
        out=np.zeros(len(base), dtype=float),
        where=base["baseline_generated_claim_count"].to_numpy() != 0,
    )
    base["gated_unsupported_claim_rate"] = np.divide(
        base["gated_unsupported_claim_count"],
        base["gated_generated_claim_count"],
        out=np.zeros(len(base), dtype=float),
        where=base["gated_generated_claim_count"].to_numpy() != 0,
    )

    # Match Phase 19 actual generated candidate totals and unsupported counts.
    summary_overall = claim_summary[claim_summary["summary_level"] == "overall"].set_index("report_type")
    expected = {
        "baseline": (int(summary_overall.loc["baseline", "total_generated_candidates"]), int(summary_overall.loc["baseline", "unsupported_claim_count"])),
        "gated": (int(summary_overall.loc["gated", "total_generated_candidates"]), int(summary_overall.loc["gated", "unsupported_claim_count"])),
    }
    for side in ("baseline", "gated"):
        actual = (
            int(base[f"{side}_generated_claim_count"].sum()),
            int(base[f"{side}_unsupported_claim_count"].sum()),
        )
        if actual != expected[side]:
            raise ValueError(f"Actual Phase 19 {side} claim counts do not reconcile: {actual} != {expected[side]}")
        expected_omissions = int(summary_overall.loc[side, "omission"])
        if int(base[f"{side}_omission_count"].sum()) != expected_omissions:
            raise ValueError(f"Phase 19 {side} omission counts failed reconciliation")

    frames = {
        "records": base,
        "claims": claims,
        "benchmark": benchmark,
        "eligibility": eligible,
        "baseline": baseline,
        "gated": gated,
        "claim_summary": claim_summary,
        "radgraph_summary": radgraph_summary,
        "chexpert_summary": chexpert_summary,
    }
    return frames, hashes


def _claim_view_summary(records: pd.DataFrame, side: str) -> dict[str, Any]:
    count = int(records[f"{side}_generated_claim_count"].sum())
    unsupported = int(records[f"{side}_unsupported_claim_count"].sum())
    omissions = int(records[f"{side}_omission_count"].sum())
    return {
        "unsupported_claim_count": unsupported,
        "evaluated_generated_claim_denominator": count,
        "unsupported_claim_rate": unsupported / count if count else 0.0,
        "omission_count": omissions,
    }


def _quality_view_summary(records: pd.DataFrame) -> dict[str, Any]:
    covered = records[records["gated_report_covered"]]
    result: dict[str, Any] = {}
    for metric in ("radgraph_f1", "chexpert_f1"):
        base = records[f"baseline_{metric}"]
        gated_all = records[f"gated_{metric}"]
        gated_covered = covered[f"gated_{metric}"]
        result.update(
            {
                f"baseline_{metric}_mean": float(base.mean()),
                f"baseline_{metric}_median": float(base.median()),
                f"gated_all_{metric}_mean": float(gated_all.mean()),
                f"gated_all_{metric}_median": float(gated_all.median()),
                f"gated_covered_{metric}_mean": float(gated_covered.mean()) if len(covered) else np.nan,
                f"gated_covered_{metric}_median": float(gated_covered.median()) if len(covered) else np.nan,
                f"gated_all_minus_baseline_{metric}_mean_delta": float(gated_all.mean() - base.mean()),
                f"gated_covered_minus_baseline_{metric}_mean_delta": (
                    float(gated_covered.mean() - base.mean()) if len(covered) else np.nan
                ),
            }
        )
    return result


def _group_table(records: pd.DataFrame, group_column: str, group_name: str, values: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for value in values:
        subset = records[records[group_column].astype(str) == value]
        if subset.empty:
            raise ValueError(f"Missing {group_name} group: {value}")
        covered = subset[subset["gated_report_covered"]]
        row: dict[str, Any] = {
            group_name: value,
            "n": len(subset),
            "covered_n": len(covered),
            "abstained_n": int(subset["is_gated_abstention"].sum()),
            "coverage": len(covered) / len(subset),
            "abstention_rate": float(subset["is_gated_abstention"].mean()),
        }
        row.update(_quality_view_summary(subset))
        row.update({f"baseline_{k}": v for k, v in _claim_view_summary(subset, "baseline").items()})
        row.update({f"gated_all_{k}": v for k, v in _claim_view_summary(subset, "gated").items()})
        row.update({f"gated_covered_{k}": v for k, v in _claim_view_summary(covered, "gated").items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _risk_coverage_summary(records: pd.DataFrame) -> pd.DataFrame:
    total = len(records)
    covered = records[records["gated_report_covered"]]
    views = [
        ("baseline", records, "baseline"),
        ("gated_all_records", records, "gated"),
        ("gated_covered_only", covered, "gated"),
    ]
    rows = []
    base_quality = _quality_view_summary(records)
    for view_name, subset, side in views:
        row: dict[str, Any] = {
            "view": view_name,
            "eligible_n": total,
            "evaluated_n": len(subset),
            "covered_n": int(records["gated_report_covered"].sum()) if side == "gated" else total,
            "abstained_n": int(records["is_gated_abstention"].sum()) if side == "gated" else 0,
            "coverage": float(records["gated_report_covered"].mean()) if side == "gated" else 1.0,
            "abstention_rate": float(records["is_gated_abstention"].mean()) if side == "gated" else 0.0,
        }
        row.update(_claim_view_summary(subset, side))
        row.update(base_quality)
        if view_name == "gated_covered_only":
            for metric in ("radgraph_f1", "chexpert_f1"):
                row[f"reported_gated_{metric}_mean"] = float(subset[f"gated_{metric}"].mean()) if len(subset) else np.nan
                row[f"reported_gated_{metric}_median"] = float(subset[f"gated_{metric}"].median()) if len(subset) else np.nan
        elif view_name == "gated_all_records":
            for metric in ("radgraph_f1", "chexpert_f1"):
                row[f"reported_gated_{metric}_mean"] = float(subset[f"gated_{metric}"].mean()) if len(subset) else np.nan
                row[f"reported_gated_{metric}_median"] = float(subset[f"gated_{metric}"].median()) if len(subset) else np.nan
        else:
            for metric in ("radgraph_f1", "chexpert_f1"):
                row[f"reported_gated_{metric}_mean"] = np.nan
                row[f"reported_gated_{metric}_median"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _claim_risk_summary(records: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", records)]
    scopes.extend(("benchmark_condition", value, records[records["condition"] == value]) for value in CONDITIONS)
    scopes.extend(
        ("analyzer_evidence_state", value, records[records["analyzer_evidence_state"] == value])
        for value in STATES
    )
    scopes.extend(("gate_action", value, records[records["gate_action"] == value]) for value in ACTIONS)

    rows = []
    for level, value, subset in scopes:
        covered = subset[subset["gated_report_covered"]]
        view_specs = (
            ("baseline", subset, "baseline"),
            ("gated_all_records", subset, "gated"),
            ("gated_covered_only", covered, "gated"),
        )
        for view, selected, side in view_specs:
            summary = _claim_view_summary(selected, side)
            rows.append(
                {
                    "summary_level": level,
                    "scope_value": value,
                    "report_view": view,
                    "eligible_n": len(subset),
                    "evaluated_n": len(selected),
                    "covered_n": len(covered) if side == "gated" else len(subset),
                    "abstained_n": int(subset["is_gated_abstention"].sum()) if side == "gated" else 0,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _analysis_document(
    risk: pd.DataFrame,
    condition: pd.DataFrame,
    state: pd.DataFrame,
    action: pd.DataFrame,
    claim: pd.DataFrame,
) -> str:
    over = risk.set_index("view")
    baseline = over.loc["baseline"]
    gated_all = over.loc["gated_all_records"]
    gated_covered = over.loc["gated_covered_only"]
    coverage = float(gated_all["coverage"])
    lines = [
        "# Phase 21: Risk-Coverage and Selective Prediction Analysis",
        "",
        "## Objective and inputs",
        "",
        "This descriptive analysis joins the frozen Phase 16 benchmark, Phase 17 eligibility and baseline, Phase 18 gated results, Phase 19 claim evaluation, and Phase 20 RadGraph/CheXpert per-record scores. It covers only records marked `eligible_for_vlm == True`.",
        "",
        "References are the benchmark `findings` plus `impression`. Keys are joined on `(sample_id, uid, condition)` and verified against the required `sample_id + condition` pairing. No reports or upstream artifacts were changed.",
        "",
        "## Methodology",
        "",
        "Coverage is covered gated records divided by all eligible records. A record is covered when its recorded gate action is not `abstain`; abstention count comes from `gate_action == abstain` and is cross-checked against Phase 20 flags.",
        "",
        "All-record gated quality averages the existing Phase 20 per-record scores, including each abstention output exactly as Phase 20 scored it. Covered-only quality averages scores for covered records only. Abstention scores were not replaced or recoded.",
        "",
        "Unsupported claims retain the Phase 19 definition: `ungrounded_affirmation`, `unsupported_affirmation`, and `contradiction`. Omissions remain separate. The denominator counts actual generated candidate claims from Phase 19 generated polarities. Empty fallback rows and abstention reports with no generated claims add no claims to the denominator. A zero denominator yields rate 0, matching the existing zero-denominator convention.",
        "",
        "RadGraph-F1 and CheXpert-F1 are report-quality metrics under their existing Phase 15 scoring procedures. This report keeps coverage, unsupported-claim risk, and report-quality scores as separate quantities.",
        "",
        "## Overall results",
        "",
        f"Eligible records: {int(gated_all['eligible_n'])}. Gated covered: {int(gated_all['covered_n'])}; abstained: {int(gated_all['abstained_n'])}. Coverage: {coverage:.4%}; abstention rate: {float(gated_all['abstention_rate']):.4%}.",
        "",
        "| View | N evaluated | Coverage | Unsupported count / generated candidates | Unsupported rate | RadGraph-F1 mean / median | CheXpert-F1 mean / median |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Baseline | {int(baseline['evaluated_n'])} | 100.00% | {int(baseline['unsupported_claim_count'])} / {int(baseline['evaluated_generated_claim_denominator'])} | {float(baseline['unsupported_claim_rate']):.4%} | {float(baseline['baseline_radgraph_f1_mean']):.4f} / {float(baseline['baseline_radgraph_f1_median']):.4f} | {float(baseline['baseline_chexpert_f1_mean']):.4f} / {float(baseline['baseline_chexpert_f1_median']):.4f} |",
        f"| Gated, all eligible | {int(gated_all['evaluated_n'])} | {coverage:.2%} | {int(gated_all['unsupported_claim_count'])} / {int(gated_all['evaluated_generated_claim_denominator'])} | {float(gated_all['unsupported_claim_rate']):.4%} | {float(gated_all['gated_all_radgraph_f1_mean']):.4f} / {float(gated_all['gated_all_radgraph_f1_median']):.4f} | {float(gated_all['gated_all_chexpert_f1_mean']):.4f} / {float(gated_all['gated_all_chexpert_f1_median']):.4f} |",
        f"| Gated, covered only | {int(gated_covered['evaluated_n'])} | {coverage:.2%} cohort coverage | {int(gated_covered['unsupported_claim_count'])} / {int(gated_covered['evaluated_generated_claim_denominator'])} | {float(gated_covered['unsupported_claim_rate']):.4%} | {float(gated_covered['gated_covered_radgraph_f1_mean']):.4f} / {float(gated_covered['gated_covered_radgraph_f1_median']):.4f} | {float(gated_covered['gated_covered_chexpert_f1_mean']):.4f} / {float(gated_covered['gated_covered_chexpert_f1_median']):.4f} |",
        "",
        f"Mean score deltas (gated all minus baseline): RadGraph-F1 {float(gated_all['gated_all_minus_baseline_radgraph_f1_mean_delta']):+.4f}; CheXpert-F1 {float(gated_all['gated_all_minus_baseline_chexpert_f1_mean_delta']):+.4f}. Covered-only minus baseline: RadGraph-F1 {float(gated_covered['gated_covered_minus_baseline_radgraph_f1_mean_delta']):+.4f}; CheXpert-F1 {float(gated_covered['gated_covered_minus_baseline_chexpert_f1_mean_delta']):+.4f}.",
        "",
        "## Condition-level results",
        "",
        _markdown_table(condition, [
            "condition", "n", "covered_n", "abstained_n", "coverage",
            "baseline_unsupported_claim_rate", "gated_all_unsupported_claim_rate", "gated_covered_unsupported_claim_rate",
            "baseline_radgraph_f1_mean", "gated_all_radgraph_f1_mean", "gated_covered_radgraph_f1_mean",
            "baseline_chexpert_f1_mean", "gated_all_chexpert_f1_mean", "gated_covered_chexpert_f1_mean",
        ]),
        "",
        "## Analyzer-state results",
        "",
        "States below are taken from Phase 18 analyzer output, not inferred from benchmark conditions.",
        "",
        _markdown_table(state, [
            "analyzer_evidence_state", "n", "covered_n", "abstained_n", "coverage",
            "baseline_unsupported_claim_rate", "gated_all_unsupported_claim_rate", "gated_covered_unsupported_claim_rate",
            "baseline_radgraph_f1_mean", "gated_all_radgraph_f1_mean", "gated_covered_radgraph_f1_mean",
            "baseline_chexpert_f1_mean", "gated_all_chexpert_f1_mean", "gated_covered_chexpert_f1_mean",
        ]),
        "",
        "## Gate-action results",
        "",
        _markdown_table(action, [
            "gate_action", "n", "covered_n", "abstained_n", "coverage",
            "baseline_unsupported_claim_rate", "gated_all_unsupported_claim_rate", "gated_covered_unsupported_claim_rate",
            "baseline_radgraph_f1_mean", "gated_all_radgraph_f1_mean", "gated_covered_radgraph_f1_mean",
            "baseline_chexpert_f1_mean", "gated_all_chexpert_f1_mean", "gated_covered_chexpert_f1_mean",
        ]),
        "",
        "## Interpretation and limitations",
        "",
        f"1. The gate abstains on all 96 insufficient records, yielding coverage of {coverage:.2%}.",
        "2. Covered-only metrics describe reports the system chose to produce; all-record metrics include the abstention outputs as scored by Phase 20.",
        "3. The `evidentiary_incomplete` condition was classified as `sufficient` by the current analyzer and remains fully covered. This observed analyzer limitation is preserved, not corrected here.",
        "4. Only one deterministic operating point is evaluated. This is not a continuous risk-coverage curve.",
        "5. Results are descriptive. They do not establish statistical significance or causal effects.",
        "6. RadGraph-F1 and CheXpert-F1 measure overlap under their existing implementations; neither alone establishes clinical safety or overall factual reliability.",
        "7. Abstention has no generated claim candidates. Its claim denominator is zero and it contributes no artificial claims; zero-denominator unsupported rate is reported as zero.",
        "",
        "## Reproducibility",
        "",
        "Run ` .venv\\Scripts\\python.exe -m src.evaluation.run_phase21_risk_coverage` from the repository root, then run focused tests with ` .venv\\Scripts\\python.exe -m pytest tests\\test_phase21_risk_coverage.py -q`.",
        "Input hashes and full baseline/gated output-tree hashes are checked before and after analysis. The script stops on cohort, key, state, action, or count mismatches.",
        "",
        "Generated tables: `phase21_risk_coverage_summary.csv`, `phase21_condition_coverage.csv`, `phase21_analyzer_state_summary.csv`, `phase21_gate_action_summary.csv`, and `phase21_claim_risk_summary.csv`.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def fmt(value: Any, column: str) -> str:
        if pd.isna(value):
            return "—"
        if column == "coverage" or column.endswith("unsupported_claim_rate"):
            return f"{float(value):.2%}"
        if "radgraph_f1" in column or "chexpert_f1" in column:
            return f"{float(value):.4f}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(fmt(row[column], column) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, separator, *body])


def build_phase21_outputs() -> tuple[dict[str, pd.DataFrame], str, dict[str, str], dict[str, dict[str, str]]]:
    frames, hashes_before = _load_and_join()
    records = frames["records"]
    risk = _risk_coverage_summary(records)
    condition = _group_table(records, "condition", "condition", CONDITIONS)
    state = _group_table(records, "analyzer_evidence_state", "analyzer_evidence_state", STATES)
    action = _group_table(records, "gate_action", "gate_action", ACTIONS)
    claim = _claim_risk_summary(records)
    outputs = {"risk": risk, "condition": condition, "state": state, "action": action, "claim": claim}

    expected_coverage = risk.set_index("view").loc["gated_all_records"]
    if (
        int(expected_coverage["eligible_n"]) != 576
        or int(expected_coverage["covered_n"]) != 480
        or int(expected_coverage["abstained_n"]) != 96
        or not np.isclose(float(expected_coverage["coverage"]), 480 / 576)
    ):
        raise ValueError("Calculated Phase 21 coverage does not match the Phase 18 cohort behavior")
    if int(risk.set_index("view").loc["baseline", "evaluated_generated_claim_denominator"]) != 1578:
        raise ValueError("Baseline generated-claim denominator failed reconciliation with Phase 19")
    if int(risk.set_index("view").loc["gated_all_records", "evaluated_generated_claim_denominator"]) != 1460:
        raise ValueError("Gated generated-claim denominator failed reconciliation with Phase 19")

    docs = _analysis_document(risk, condition, state, action, claim)
    snapshots = {
        "benchmark": {BENCHMARK_PATH.name: sha256(BENCHMARK_PATH)},
        "eligibility": {ELIGIBILITY_PATH.name: sha256(ELIGIBILITY_PATH)},
        "baseline": tree_hashes(BASELINE_PATH.parent),
        "gated": tree_hashes(GATED_PATH.parent),
        "phase19": {
            CLAIM_DETAIL_PATH.name: sha256(CLAIM_DETAIL_PATH),
            CLAIM_SUMMARY_PATH.name: sha256(CLAIM_SUMMARY_PATH),
        },
        "phase20": {
            RADGRAPH_RESULTS_PATH.name: sha256(RADGRAPH_RESULTS_PATH),
            RADGRAPH_SUMMARY_PATH.name: sha256(RADGRAPH_SUMMARY_PATH),
            CHEXPERT_RESULTS_PATH.name: sha256(CHEXPERT_RESULTS_PATH),
            CHEXPERT_SUMMARY_PATH.name: sha256(CHEXPERT_SUMMARY_PATH),
        },
    }
    return outputs, docs, hashes_before, snapshots


def run_phase21() -> dict[str, pd.DataFrame]:
    outputs, docs, hashes_before, snapshots_before = build_phase21_outputs()
    for key, frame in outputs.items():
        frame.to_csv(OUTPUTS[key], index=False)
    OUTPUTS["docs"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS["docs"].write_text(docs, encoding="utf-8")

    hashes_after = {name: sha256(path) for name, path in {
        "benchmark": BENCHMARK_PATH,
        "eligibility": ELIGIBILITY_PATH,
        "baseline": BASELINE_PATH,
        "gated": GATED_PATH,
        "claim_detail": CLAIM_DETAIL_PATH,
        "claim_summary": CLAIM_SUMMARY_PATH,
        "radgraph": RADGRAPH_RESULTS_PATH,
        "radgraph_summary": RADGRAPH_SUMMARY_PATH,
        "chexpert": CHEXPERT_RESULTS_PATH,
        "chexpert_summary": CHEXPERT_SUMMARY_PATH,
    }.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("A Phase 21 input artifact changed during analysis")
    snapshots_after = {
        "benchmark": {BENCHMARK_PATH.name: sha256(BENCHMARK_PATH)},
        "eligibility": {ELIGIBILITY_PATH.name: sha256(ELIGIBILITY_PATH)},
        "baseline": tree_hashes(BASELINE_PATH.parent),
        "gated": tree_hashes(GATED_PATH.parent),
        "phase19": {
            CLAIM_DETAIL_PATH.name: sha256(CLAIM_DETAIL_PATH),
            CLAIM_SUMMARY_PATH.name: sha256(CLAIM_SUMMARY_PATH),
        },
        "phase20": {
            RADGRAPH_RESULTS_PATH.name: sha256(RADGRAPH_RESULTS_PATH),
            RADGRAPH_SUMMARY_PATH.name: sha256(RADGRAPH_SUMMARY_PATH),
            CHEXPERT_RESULTS_PATH.name: sha256(CHEXPERT_RESULTS_PATH),
            CHEXPERT_SUMMARY_PATH.name: sha256(CHEXPERT_SUMMARY_PATH),
        },
    }
    if snapshots_before != snapshots_after:
        raise RuntimeError("A frozen Phase 21 input tree changed during analysis")
    return outputs


def main() -> None:
    outputs = run_phase21()
    print(f"Phase 21 outputs written: {len(outputs)} CSVs + analysis documentation")
    print(outputs["risk"].to_string(index=False))
    print("No VLM inference, statistical significance testing, ECE, Brier, or continuous curve was run.")


if __name__ == "__main__":
    main()
