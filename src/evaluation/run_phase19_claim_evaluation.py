"""Phase 19 paired claim evaluation for the Phase 17/18 576-record cohort.

Uses the established claim-pattern extractor and conservative polarity labels.
This is limited-ontology claim comparison; it does not run RadGraph-F1,
CheXpert-F1, or statistical comparison.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.build_reference_candidates import extract_reference_candidates
from src.evaluation.claim_evaluator import load_ontology
from src.evaluation.claim_extractor import extract_generated_claim_candidates
from src.evaluation.claim_patterns import REFERENCE_PATTERNS
from src.evaluation.evaluate_reference_factuality import build_factuality_table

BENCHMARK_PATH = ROOT / "results/tables/phase16_perturbations.csv"
ELIGIBILITY_PATH = ROOT / "results/tables/phase17_vlm_inference_eligibility.csv"
BASELINE_PATH = ROOT / "results/baseline/phase17_full/baseline_results.csv"
GATED_PATH = ROOT / "results/gated/phase18_full/gated_results.csv"
ONTOLOGY_PATH = ROOT / "configs/claim_ontology.yaml"
DETAIL_PATH = ROOT / "results/tables/phase19_claim_evaluation.csv"
SUMMARY_PATH = ROOT / "results/tables/phase19_claim_summary.csv"

EXCLUDED_UIDS = {74, 597, 803, 885}
LABELS = (
    "supported",
    "omission",
    "extra_negation",
    "ungrounded_affirmation",
    "unsupported_affirmation",
    "contradiction",
    "unclear",
)
REQUIRED_CONDITIONS = {
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
}
UNSUPPORTED_LABELS = {
    "ungrounded_affirmation",
    "unsupported_affirmation",
    "contradiction",
}


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


def _key(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    return {
        (str(row.sample_id), _uid(row.uid), str(row.condition))
        for row in frame.itertuples(index=False)
    }


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = (BENCHMARK_PATH, ELIGIBILITY_PATH, BASELINE_PATH, GATED_PATH, ONTOLOGY_PATH)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 19 input is missing: {path}")

    benchmark = pd.read_csv(BENCHMARK_PATH)
    eligibility_all = pd.read_csv(ELIGIBILITY_PATH)
    eligibility = eligibility_all[
        eligibility_all["eligible_for_vlm"].astype(str).str.lower() == "true"
    ].copy()
    baseline = pd.read_csv(BASELINE_PATH)
    gated = pd.read_csv(GATED_PATH)

    if len(eligibility) != 576 or len(baseline) != 576 or len(gated) != 576:
        raise ValueError(
            f"Expected 576 eligible/baseline/gated rows; got "
            f"{len(eligibility)}/{len(baseline)}/{len(gated)}"
        )
    for name, frame in (("eligibility", eligibility), ("baseline", baseline), ("gated", gated)):
        if frame.duplicated(["sample_id", "uid", "condition"]).any():
            raise ValueError(f"{name} has duplicate record keys")

    eligible_keys = _key(eligibility)
    if _key(baseline) != eligible_keys or _key(gated) != eligible_keys:
        raise ValueError("Eligibility, baseline, and gated record keys do not match exactly")
    if set(eligibility["uid"].astype(int)) & EXCLUDED_UIDS:
        raise ValueError("Excluded image-unavailable UID appears in the eligible cohort")
    if eligibility["condition"].value_counts().to_dict() != {
        condition: 96 for condition in REQUIRED_CONDITIONS
    }:
        raise ValueError("Eligible cohort does not have 96 records per condition")

    benchmark = benchmark.copy()
    benchmark["uid"] = benchmark["uid"].astype(int)
    eligibility = eligibility.copy()
    eligibility["uid"] = eligibility["uid"].astype(int)
    baseline = baseline.copy()
    baseline["uid"] = baseline["uid"].astype(int)
    gated = gated.copy()
    gated["uid"] = gated["uid"].astype(int)

    eligible_benchmark = eligibility[["sample_id", "uid", "condition"]].merge(
        benchmark,
        on=["sample_id", "uid", "condition"],
        how="left",
        validate="one_to_one",
        suffixes=("_eligibility", "_benchmark"),
    )
    if eligible_benchmark["findings"].isna().any() or eligible_benchmark["impression"].isna().any():
        raise ValueError("Eligible record is missing benchmark reference text")

    # Ensure each study has one stable reference findings/impression source.
    reference_variants = eligible_benchmark.groupby("uid")[
        ["sample_id", "findings", "impression"]
    ].nunique(dropna=False)
    if (reference_variants > 1).any().any():
        raise ValueError("Reference findings/impression vary across conditions for a UID")

    for name, frame, report_column in (
        ("baseline", baseline, "generated_report"),
        ("gated", gated, "gated_report"),
    ):
        paired = eligibility[["sample_id", "uid", "condition", "image_path"]].merge(
            frame,
            on=["sample_id", "uid", "condition"],
            how="left",
            validate="one_to_one",
            suffixes=("_eligibility", f"_{name}"),
        )
        if paired[report_column].isna().any() or (paired[report_column].astype(str).str.len() == 0).any():
            raise ValueError(f"{name} has an empty report")
        if name == "baseline":
            benchmark_contexts = eligible_benchmark.set_index(
                ["sample_id", "uid", "condition"]
            )["perturbed_context"].to_dict()
            for row in frame.itertuples(index=False):
                key = (str(row.sample_id), int(row.uid), str(row.condition))
                if str(row.perturbed_context) != str(benchmark_contexts[key]):
                    raise ValueError(f"Baseline context does not match benchmark for {key}")
        else:
            baseline_by_key = baseline.set_index(["sample_id", "uid", "condition"])
            for row in frame.itertuples(index=False):
                key = (str(row.sample_id), int(row.uid), str(row.condition))
                base_context = str(baseline_by_key.loc[key, "perturbed_context"])
                if str(row.perturbed_context) != base_context:
                    raise ValueError(f"Gated context does not match baseline for {key}")
                if str(row.image_path) != str(baseline_by_key.loc[key, "image_path"]):
                    raise ValueError(f"Gated image path does not match baseline for {key}")

    # claim_ontology.yaml is the approved conflict-target alias map. Its targets
    # resolve into the canonical general candidate vocabulary used below.
    ontology = load_ontology()
    unknown_targets = {
        str(mapping["extractor_finding"]).strip().lower()
        for mapping in ontology.values()
        if str(mapping["extractor_finding"]).strip().lower() not in REFERENCE_PATTERNS
    }
    if unknown_targets:
        raise ValueError(f"Configured ontology targets are outside claim vocabulary: {sorted(unknown_targets)}")

    return eligibility, eligible_benchmark, baseline, gated, benchmark


def _extract_reference_claims(eligible_benchmark: pd.DataFrame) -> pd.DataFrame:
    studies = (
        eligible_benchmark[["sample_id", "findings", "impression"]]
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
    )
    return extract_reference_candidates(studies)


def _factuality_for_reports(
    report_frame: pd.DataFrame,
    report_column: str,
    reference_claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generation = report_frame[["sample_id", "condition", report_column]].rename(
        columns={report_column: "generated_report"}
    )
    claims = extract_generated_claim_candidates(generation)
    factuality = build_factuality_table(generation, reference_claims, claims)
    return claims, factuality


def build_phase19_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility, eligible_benchmark, baseline, gated, _ = _load_inputs()
    reference_claims = _extract_reference_claims(eligible_benchmark)
    baseline_claims, baseline_fact = _factuality_for_reports(
        baseline, "generated_report", reference_claims
    )
    gated_claims, gated_fact = _factuality_for_reports(
        gated, "gated_report", reference_claims
    )

    keys = ["sample_id", "condition", "finding"]
    baseline_fact = baseline_fact.rename(
        columns={
            "generated_polarity": "baseline_generated_polarity",
            "label": "baseline_label",
            "reference_polarity": "baseline_reference_polarity",
        }
    )
    gated_fact = gated_fact.rename(
        columns={
            "generated_polarity": "gated_generated_polarity",
            "label": "gated_label",
            "reference_polarity": "gated_reference_polarity",
        }
    )
    detail = baseline_fact.merge(
        gated_fact,
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    if not detail["baseline_reference_polarity"].fillna("").astype(str).eq(
        detail["gated_reference_polarity"].fillna("").astype(str)
    ).all():
        raise ValueError("Baseline and gated factuality tables used different reference polarities")
    detail["reference_polarity"] = detail["baseline_reference_polarity"].fillna(
        detail["gated_reference_polarity"]
    ).fillna("")
    detail["baseline_generated_polarity"] = detail["baseline_generated_polarity"].fillna(
        "NOT_MENTIONED"
    )
    detail["gated_generated_polarity"] = detail["gated_generated_polarity"].fillna(
        "NOT_MENTIONED"
    )
    # A generated-only finding on one side has no counterpart claim to label
    # on the other side. Keep that paired label blank instead of manufacturing
    # an omission (omission is reserved for absent reference-supported claims).
    detail["baseline_label"] = detail["baseline_label"].fillna("")
    detail["gated_label"] = detail["gated_label"].fillna("")
    metadata = gated[
        ["sample_id", "uid", "condition", "evidence_state", "gate_action"]
    ].rename(
        columns={
            "condition": "benchmark_condition",
            "evidence_state": "analyzer_evidence_state",
        }
    )
    detail = detail.merge(
        metadata,
        left_on=["sample_id", "condition"],
        right_on=["sample_id", "benchmark_condition"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["condition"])
    detail = detail[
        [
            "sample_id",
            "uid",
            "benchmark_condition",
            "analyzer_evidence_state",
            "gate_action",
            "finding",
            "reference_polarity",
            "baseline_generated_polarity",
            "baseline_label",
            "gated_generated_polarity",
            "gated_label",
        ]
    ].sort_values(["sample_id", "benchmark_condition", "finding"]).reset_index(drop=True)
    if detail.duplicated(
        ["sample_id", "uid", "benchmark_condition", "finding"]
    ).any():
        raise RuntimeError("Duplicate claim-level evaluation key found")

    # Record-level key coverage is checked separately from claim-level row count.
    expected_keys = _key(eligibility)
    for frame, label in ((baseline, "baseline"), (gated, "gated")):
        if _key(frame) != expected_keys:
            raise RuntimeError(f"{label} record evaluations do not match eligible keys")
    represented = set(
        zip(detail["sample_id"].astype(str), detail["benchmark_condition"].astype(str))
    )
    expected_pairs = set(zip(eligibility["sample_id"].astype(str), eligibility["condition"].astype(str)))
    if represented != expected_pairs:
        raise RuntimeError("Claim evaluation does not represent every eligible record")

    summary = _build_summary(
        eligibility,
        reference_claims,
        detail,
        baseline_claims,
        gated_claims,
    )
    return detail, summary


def _build_summary(
    eligibility: pd.DataFrame,
    reference_claims: pd.DataFrame,
    detail: pd.DataFrame,
    baseline_claims: pd.DataFrame,
    gated_claims: pd.DataFrame,
) -> pd.DataFrame:
    scopes: list[tuple[str, str, set[tuple[str, str]]]] = []
    all_pairs = set(zip(eligibility["sample_id"].astype(str), eligibility["condition"].astype(str)))
    scopes.append(("overall", "all", all_pairs))
    for condition in sorted(REQUIRED_CONDITIONS):
        pairs = set(
            zip(
                eligibility.loc[eligibility["condition"] == condition, "sample_id"].astype(str),
                eligibility.loc[eligibility["condition"] == condition, "condition"].astype(str),
            )
        )
        scopes.append(("benchmark_condition", condition, pairs))
    for state in sorted(eligibility.merge(
        detail[["sample_id", "benchmark_condition", "analyzer_evidence_state"]].drop_duplicates(),
        left_on=["sample_id", "condition"],
        right_on=["sample_id", "benchmark_condition"],
        how="left",
    )["analyzer_evidence_state"].dropna().unique()):
        pairs = set(
            zip(
                detail.loc[detail["analyzer_evidence_state"] == state, "sample_id"].astype(str),
                detail.loc[detail["analyzer_evidence_state"] == state, "benchmark_condition"].astype(str),
            )
        )
        scopes.append(("analyzer_evidence_state", str(state), pairs))

    rows: list[dict[str, Any]] = []
    meta_pairs = eligibility[["sample_id", "condition"]].astype(str).drop_duplicates()
    for scope_level, scope_value, scope_pairs in scopes:
        scoped_keys = {(sample, condition) for sample, condition in scope_pairs}
        scoped_detail = detail[
            detail.apply(
                lambda row: (str(row["sample_id"]), str(row["benchmark_condition"])) in scoped_keys,
                axis=1,
            )
        ]
        scoped_meta = meta_pairs[
            meta_pairs.apply(lambda row: (row["sample_id"], row["condition"]) in scoped_keys, axis=1)
        ]
        sample_ids = set(scoped_meta["sample_id"].astype(str))
        reference_count = int(reference_claims[reference_claims["sample_id"].astype(str).isin(sample_ids)].shape[0])
        reference_comparisons = int(
            scoped_detail[scoped_detail["reference_polarity"].astype(str) != ""].shape[0]
        )

        for report_type, claims, label_column in (
            ("baseline", baseline_claims, "baseline_label"),
            ("gated", gated_claims, "gated_label"),
        ):
            scoped_claims = claims[
                claims.apply(
                    lambda row: (str(row["sample_id"]), str(row["condition"])) in scoped_keys,
                    axis=1,
                )
            ]
            labels = scoped_detail[label_column].value_counts().to_dict()
            generated_total = len(scoped_claims)
            evaluated_generated = sum(
                int(labels.get(label, 0))
                for label in LABELS
                if label != "omission"
            )
            unsupported_count = sum(int(labels.get(label, 0)) for label in UNSUPPORTED_LABELS)
            row: dict[str, Any] = {
                "summary_level": scope_level,
                "scope_value": scope_value,
                "report_type": report_type,
                "record_count": len(scoped_meta),
                "total_reference_candidates": reference_count,
                "reference_candidate_comparisons": reference_comparisons,
                "total_generated_candidates": generated_total,
                "unsupported_claim_rate_denominator": evaluated_generated,
                "comparison_candidate_count": len(scoped_detail),
            }
            for label in LABELS:
                row[label] = int(labels.get(label, 0))
            row["unsupported_claim_count"] = unsupported_count
            row["unsupported_claim_rate"] = (
                unsupported_count / evaluated_generated if evaluated_generated else 0.0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_phase19() -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = (BENCHMARK_PATH, ELIGIBILITY_PATH, BASELINE_PATH, GATED_PATH, ONTOLOGY_PATH)
    hashes_before = {path: sha256(path) for path in inputs}
    baseline_tree_before = tree_hashes(BASELINE_PATH.parent)
    gated_tree_before = tree_hashes(GATED_PATH.parent)

    detail, summary = build_phase19_outputs()
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    hashes_after = {path: sha256(path) for path in inputs}
    if hashes_before != hashes_after:
        raise RuntimeError("Benchmark, eligibility, baseline, gated, or ontology input changed")
    if baseline_tree_before != tree_hashes(BASELINE_PATH.parent):
        raise RuntimeError("Baseline outputs changed during Phase 19 evaluation")
    if gated_tree_before != tree_hashes(GATED_PATH.parent):
        raise RuntimeError("Gated outputs changed during Phase 19 evaluation")

    return detail, summary


def main() -> None:
    detail, summary = run_phase19()
    print(f"Eligible report records: 576 (baseline and gated)")
    print(f"Claim-level paired comparison rows: {len(detail)}")
    print(f"Reference candidates (unique studies): {summary.iloc[0]['total_reference_candidates']}")
    print(f"Baseline generated candidates: {int(summary.iloc[0]['total_generated_candidates'])}")
    print(f"Gated generated candidates: {int(summary.iloc[1]['total_generated_candidates'])}")
    print(f"Detail: {DETAIL_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("RadGraph-F1, CheXpert-F1, and statistical comparison: not run")


if __name__ == "__main__":
    main()
