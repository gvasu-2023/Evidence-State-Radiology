"""Phase 20 cohort adapter for established Phase 15 RadGraph/CheXpert metrics."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_chexpert_f1 import (
    CHEXPERT_CATEGORIES,
    compute_chexpert_f1,
    extract_chexpert_vector,
)

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

EXCLUDED_UIDS = {74, 597, 803, 885}
CONDITIONS = (
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
)
METRICS = ("radgraph_f1", "chexpert_f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _uid(value: Any) -> int:
    return int(float(value))


def _keys(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    return {
        (str(row.sample_id), _uid(row.uid), str(row.condition))
        for row in frame.itertuples(index=False)
    }


def _load_cohort() -> pd.DataFrame:
    required_paths = (BENCHMARK_PATH, ELIGIBILITY_PATH, BASELINE_PATH, GATED_PATH)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 20 input is missing: {path}")

    benchmark = pd.read_csv(BENCHMARK_PATH)
    eligibility_all = pd.read_csv(ELIGIBILITY_PATH)
    eligibility = eligibility_all[
        eligibility_all["eligible_for_vlm"].astype(str).str.lower() == "true"
    ].copy()
    baseline = pd.read_csv(BASELINE_PATH)
    gated = pd.read_csv(GATED_PATH)

    for name, frame in (("eligibility", eligibility), ("baseline", baseline), ("gated", gated)):
        if frame.duplicated(["sample_id", "uid", "condition"]).any():
            raise ValueError(f"Duplicate {name} key found")
    expected_keys = _keys(eligibility)
    if len(eligibility) != 576 or _keys(baseline) != expected_keys or _keys(gated) != expected_keys:
        raise ValueError("Expected exact one-to-one eligibility, baseline, and gated cohorts of 576")
    if set(eligibility["uid"].astype(int)) & EXCLUDED_UIDS:
        raise ValueError("An image-unavailable UID appears in the eligible cohort")
    if eligibility["condition"].value_counts().to_dict() != {condition: 96 for condition in CONDITIONS}:
        raise ValueError("Eligible cohort condition balance is not 96 per condition")

    for frame in (benchmark, eligibility, baseline, gated):
        frame["uid"] = frame["uid"].astype(int)

    cohort = eligibility[["sample_id", "uid", "condition", "image_path"]].merge(
        benchmark[["sample_id", "uid", "condition", "findings", "impression"]],
        on=["sample_id", "uid", "condition"],
        how="left",
        validate="one_to_one",
    )
    cohort = cohort.merge(
        baseline[["sample_id", "uid", "condition", "generated_report"]],
        on=["sample_id", "uid", "condition"],
        how="left",
        validate="one_to_one",
    )
    cohort = cohort.merge(
        gated[["sample_id", "uid", "condition", "evidence_state", "gate_action", "gated_report"]],
        on=["sample_id", "uid", "condition"],
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "evidence_state": "analyzer_evidence_state",
        }
    )
    if len(cohort) != 576 or cohort[["findings", "impression", "generated_report", "gated_report"]].isna().any().any():
        raise ValueError("Missing reference text or generated report in joined Phase 20 cohort")

    # Inputs from all conditions for each study must preserve the same reference.
    reference_variants = cohort.groupby("uid")[["sample_id", "findings", "impression"]].nunique(dropna=False)
    if (reference_variants > 1).any().any():
        raise ValueError("Reference findings/impression vary by condition for a UID")
    if set(cohort["analyzer_evidence_state"].astype(str)) - {
        "sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"
    }:
        raise ValueError("Unknown analyzer evidence state in Phase 18 outputs")
    return cohort.sort_values(["sample_id", "condition"]).reset_index(drop=True)


def _reference_text(row: Any) -> str:
    findings = "" if pd.isna(row.findings) else str(row.findings)
    impression = "" if pd.isna(row.impression) else str(row.impression)
    return f"{findings} {impression}".strip()


def _radgraph_scores(refs: list[str], hyps: list[str], chunk_size: int = 16) -> tuple[list[float], list[float], list[float]]:
    """Call the existing Phase 15 F1RadGraph reward with one cached model instance."""
    # Import the established wrapper, including its Transformers compatibility patch.
    phase15 = importlib.import_module("src.evaluation.evaluate_radgraph_f1")
    scorer = phase15.F1RadGraph(reward_level="all", model_type="radgraph")
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for offset in range(0, len(refs), chunk_size):
        batch_refs = refs[offset : offset + chunk_size]
        batch_hyps = hyps[offset : offset + chunk_size]
        _, reward_list, _, _ = scorer(refs=batch_refs, hyps=batch_hyps)
        precisions.extend(float(value) for value in reward_list[0])
        recalls.extend(float(value) for value in reward_list[1])
        f1_scores.extend(float(value) for value in reward_list[2])
    if not (len(precisions) == len(recalls) == len(f1_scores) == len(refs)):
        raise RuntimeError("RadGraph returned a score count that differs from input records")
    return precisions, recalls, f1_scores


def _vector_string(vector: dict[str, Any]) -> str:
    return str({key: (value if pd.notna(value) else None) for key, value in vector.items()})


def _calculate_record_scores(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_reports = [_reference_text(row) for row in cohort.itertuples(index=False)]
    baseline_reports = cohort["generated_report"].astype(str).tolist()
    gated_reports = cohort["gated_report"].astype(str).tolist()

    base_rad_p, base_rad_r, base_rad_f1 = _radgraph_scores(reference_reports, baseline_reports)
    gated_rad_p, gated_rad_r, gated_rad_f1 = _radgraph_scores(reference_reports, gated_reports)

    records: list[dict[str, Any]] = []
    for index, row in enumerate(cohort.itertuples(index=False)):
        ref_vector = extract_chexpert_vector(reference_reports[index])
        baseline_vector = extract_chexpert_vector(baseline_reports[index])
        gated_vector = extract_chexpert_vector(gated_reports[index])
        baseline_chexpert = compute_chexpert_f1(ref_vector, baseline_vector)
        gated_chexpert = compute_chexpert_f1(ref_vector, gated_vector)
        is_abstention = str(row.gate_action) == "abstain"

        records.append(
            {
                "sample_id": str(row.sample_id),
                "uid": int(row.uid),
                "condition": str(row.condition),
                "analyzer_evidence_state": str(row.analyzer_evidence_state),
                "gate_action": str(row.gate_action),
                "is_gated_abstention": is_abstention,
                "gated_report_covered": not is_abstention,
                "baseline_radgraph_precision": round(base_rad_p[index], 4),
                "baseline_radgraph_recall": round(base_rad_r[index], 4),
                "baseline_radgraph_f1": round(base_rad_f1[index], 4),
                "gated_radgraph_precision": round(gated_rad_p[index], 4),
                "gated_radgraph_recall": round(gated_rad_r[index], 4),
                "gated_radgraph_f1": round(gated_rad_f1[index], 4),
                "delta_radgraph_f1": round(gated_rad_f1[index] - base_rad_f1[index], 4),
                "baseline_chexpert_f1": round(baseline_chexpert, 4),
                "gated_chexpert_f1": round(gated_chexpert, 4),
                "delta_chexpert_f1": round(gated_chexpert - baseline_chexpert, 4),
                "reference_chexpert_vector": _vector_string(ref_vector),
                "baseline_chexpert_vector": _vector_string(baseline_vector),
                "gated_chexpert_vector": _vector_string(gated_vector),
            }
        )
    results = pd.DataFrame(records)
    if len(results) != 576 or results.duplicated(["sample_id", "uid", "condition"]).any():
        raise RuntimeError("Phase 20 per-record results failed uniqueness/count validation")
    return results, cohort


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", results)]
    scopes.extend(
        ("benchmark_condition", condition, results[results["condition"] == condition])
        for condition in CONDITIONS
    )
    for state in sorted(results["analyzer_evidence_state"].unique()):
        scopes.append(
            ("analyzer_evidence_state", str(state), results[results["analyzer_evidence_state"] == state])
        )
    for action in sorted(results["gate_action"].unique()):
        scopes.append(("gate_action", str(action), results[results["gate_action"] == action]))

    rows: list[dict[str, Any]] = []
    for level, scope, subset in scopes:
        n = len(subset)
        covered = subset[subset["gated_report_covered"]]
        row: dict[str, Any] = {
            "summary_level": level,
            "scope_value": scope,
            "n_records": n,
            "n_gated_covered": len(covered),
            "n_gated_abstained": int(subset["is_gated_abstention"].sum()),
            "gated_coverage": len(covered) / n if n else np.nan,
        }
        for metric in METRICS:
            baseline = subset[f"baseline_{metric}"]
            gated_all = subset[f"gated_{metric}"]
            gated_covered = covered[f"gated_{metric}"]
            row[f"baseline_{metric}_mean"] = float(baseline.mean()) if n else np.nan
            row[f"baseline_{metric}_median"] = float(baseline.median()) if n else np.nan
            row[f"overall_gated_{metric}_mean"] = float(gated_all.mean()) if n else np.nan
            row[f"overall_gated_{metric}_median"] = float(gated_all.median()) if n else np.nan
            row[f"covered_gated_{metric}_mean"] = float(gated_covered.mean()) if len(covered) else np.nan
            row[f"covered_gated_{metric}_median"] = float(gated_covered.median()) if len(covered) else np.nan
            row[f"overall_gated_minus_baseline_{metric}_mean_delta"] = (
                float(gated_all.mean() - baseline.mean()) if n else np.nan
            )
            row[f"covered_gated_minus_baseline_{metric}_mean_delta"] = (
                float(gated_covered.mean() - baseline.mean()) if len(covered) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_phase20() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = (
        BENCHMARK_PATH,
        ELIGIBILITY_PATH,
        BASELINE_PATH,
        GATED_PATH,
        CLAIM_DETAIL_PATH,
        CLAIM_SUMMARY_PATH,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input missing: {path}")
    hashes_before = {path: sha256(path) for path in inputs}
    baseline_tree_before = tree_hashes(BASELINE_PATH.parent)
    gated_tree_before = tree_hashes(GATED_PATH.parent)

    cohort = _load_cohort()
    results, _ = _calculate_record_scores(cohort)
    summary = _summary(results)
    radgraph_summary = summary.drop(
        columns=[column for column in summary if "chexpert" in column]
    )
    chexpert_summary = summary.drop(
        columns=[column for column in summary if "radgraph" in column]
    )

    RADGRAPH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results[
        [
            "sample_id", "uid", "condition", "analyzer_evidence_state", "gate_action",
            "is_gated_abstention", "gated_report_covered", "baseline_radgraph_precision",
            "baseline_radgraph_recall", "baseline_radgraph_f1", "gated_radgraph_precision",
            "gated_radgraph_recall", "gated_radgraph_f1", "delta_radgraph_f1",
        ]
    ].to_csv(RADGRAPH_RESULTS_PATH, index=False)
    radgraph_summary.to_csv(RADGRAPH_SUMMARY_PATH, index=False)
    results[
        [
            "sample_id", "uid", "condition", "analyzer_evidence_state", "gate_action",
            "is_gated_abstention", "gated_report_covered", "baseline_chexpert_f1",
            "gated_chexpert_f1", "delta_chexpert_f1", "reference_chexpert_vector",
            "baseline_chexpert_vector", "gated_chexpert_vector",
        ]
    ].to_csv(CHEXPERT_RESULTS_PATH, index=False)
    chexpert_summary.to_csv(CHEXPERT_SUMMARY_PATH, index=False)

    if hashes_before != {path: sha256(path) for path in inputs}:
        raise RuntimeError("Frozen benchmark/cohort/report/claim-evaluation inputs changed")
    if baseline_tree_before != tree_hashes(BASELINE_PATH.parent):
        raise RuntimeError("Baseline files changed during Phase 20")
    if gated_tree_before != tree_hashes(GATED_PATH.parent):
        raise RuntimeError("Gated files changed during Phase 20")
    return results, radgraph_summary, chexpert_summary


def main() -> None:
    results, radgraph_summary, chexpert_summary = run_phase20()
    overall_rad = radgraph_summary.query("summary_level == 'overall'").iloc[0]
    overall_chex = chexpert_summary.query("summary_level == 'overall'").iloc[0]
    print(f"Evaluated records: {len(results)}")
    print(
        "Gated covered/abstained: "
        f"{int(results['gated_report_covered'].sum())}/"
        f"{int(results['is_gated_abstention'].sum())}"
    )
    print(
        "RadGraph-F1 mean baseline / gated-all / gated-covered: "
        f"{overall_rad['baseline_radgraph_f1_mean']:.4f} / "
        f"{overall_rad['overall_gated_radgraph_f1_mean']:.4f} / "
        f"{overall_rad['covered_gated_radgraph_f1_mean']:.4f}"
    )
    print(
        "CheXpert-F1 mean baseline / gated-all / gated-covered: "
        f"{overall_chex['baseline_chexpert_f1_mean']:.4f} / "
        f"{overall_chex['overall_gated_chexpert_f1_mean']:.4f} / "
        f"{overall_chex['covered_gated_chexpert_f1_mean']:.4f}"
    )
    print("No VLM inference or statistical significance testing was performed.")


if __name__ == "__main__":
    main()
