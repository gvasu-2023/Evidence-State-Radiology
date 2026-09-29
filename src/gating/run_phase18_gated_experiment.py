"""Apply the existing Phase 13B analyzer and gate to Phase 17 eligible records.

This module only transforms/reuses frozen baseline reports. It does not load a
VLM, extract claims, or run downstream evaluation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.context_completeness import parse_context_completeness
from src.gating.reliability_gate import ReliabilityGate

ELIGIBILITY_PATH = ROOT / "results/tables/phase17_vlm_inference_eligibility.csv"
BENCHMARK_PATH = ROOT / "results/tables/phase16_perturbations.csv"
BASELINE_PATH = ROOT / "results/baseline/phase17_full/baseline_results.csv"
OUTPUT_DIR = ROOT / "results/gated/phase18_full"
OUTPUT_CSV = OUTPUT_DIR / "gated_results.csv"
REQUIRED_CONDITIONS = {
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
}
EXCLUDED_UIDS = {74, 597, 803, 885}
ABSTENTION_REPORT = "[ABSTAIN: Insufficient evidence for report generation]"


def _key(row: pd.Series) -> tuple[str, str, str]:
    return str(row["sample_id"]), str(int(row["uid"])), str(row["condition"])


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _uid_token(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_tree_hashes() -> dict[str, str]:
    """Hash every frozen baseline output file for before/after integrity checks."""
    base_dir = BASELINE_PATH.parent
    return {
        path.relative_to(base_dir).as_posix(): sha256(path)
        for path in sorted(base_dir.rglob("*"))
        if path.is_file()
    }


def _load_and_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (ELIGIBILITY_PATH, BENCHMARK_PATH, BASELINE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 18 input is missing: {path}")

    eligibility = pd.read_csv(ELIGIBILITY_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)
    baseline = pd.read_csv(BASELINE_PATH)

    eligibility = eligibility[eligibility["eligible_for_vlm"].astype(str).str.lower() == "true"].copy()
    if len(eligibility) != 576:
        raise ValueError(f"Expected 576 eligible records, found {len(eligibility)}")
    if set(eligibility["condition"].astype(str)) != REQUIRED_CONDITIONS:
        raise ValueError("Eligible cohort condition labels do not match the six-state benchmark")
    if eligibility.duplicated(["sample_id", "uid", "condition"]).any():
        raise ValueError("Eligibility artifact contains duplicate eligible keys")
    if set(eligibility["uid"].astype(int)) & EXCLUDED_UIDS:
        raise ValueError("An image-unavailable UID is marked eligible")

    for frame_name, frame in (("benchmark", benchmark), ("baseline", baseline)):
        if frame.duplicated(["sample_id", "uid", "condition"]).any():
            raise ValueError(f"{frame_name} contains duplicate record keys")

    eligible_keys = {_key(row) for _, row in eligibility.iterrows()}
    benchmark_index = {_key(row): row for _, row in benchmark.iterrows()}
    baseline_index = {_key(row): row for _, row in baseline.iterrows()}
    if not eligible_keys.issubset(benchmark_index):
        raise ValueError("Eligible keys are missing from the frozen benchmark")
    if eligible_keys != set(baseline_index):
        missing = eligible_keys - set(baseline_index)
        extra = set(baseline_index) - eligible_keys
        raise ValueError(f"Baseline key mismatch: {len(missing)} missing, {len(extra)} extra")

    for _, eligibility_row in eligibility.iterrows():
        key = _key(eligibility_row)
        benchmark_row = benchmark_index[key]
        baseline_row = baseline_index[key]
        expected_context = _text(benchmark_row["perturbed_context"])
        if _text(baseline_row["perturbed_context"]) != expected_context:
            raise ValueError(f"Baseline context differs from benchmark for {key}")
        if _text(baseline_row["image_path"]) != _text(eligibility_row["image_path"]):
            raise ValueError(f"Baseline image path differs from eligibility for {key}")
        if expected_context != _text(eligibility_row.get("perturbed_context", expected_context)):
            raise ValueError(f"Eligibility context differs from benchmark for {key}")

        # Eligibility is authoritative, and recheck the recorded frontal image locally.
        if not bool(eligibility_row["image_available"]):
            raise ValueError(f"Eligible row lacks image_available for {key}")
        image_path = (ROOT / _text(eligibility_row["image_path"])).resolve()
        if "frontal" not in image_path.name.lower():
            raise ValueError(f"Eligible image path is not a frontal image for {key}: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Eligible frontal image is missing: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        if not bool(eligibility_row["image_readable"]) or not bool(eligibility_row["image_verify_ok"]):
            raise ValueError(f"Eligibility image integrity flags are false for {key}")

    return eligibility, benchmark, baseline


def _apply_existing_policy(
    action: str,
    sample_id: str,
    condition: str,
    baseline_report: str,
    baseline_lookup: dict[tuple[str, str], str],
) -> tuple[str, str]:
    """Phase 13B baseline-reuse report policy; never performs model inference."""
    if action == "generate":
        return baseline_report, condition
    if action == "discount_context":
        return baseline_lookup[(sample_id, "insufficient")], "insufficient"
    if action == "qualify":
        image_only_report = baseline_lookup[(sample_id, "insufficient")]
        return f"[QUALIFIED: Incomplete clinical context] {image_only_report}", "insufficient"
    if action == "qualify_or_abstain":
        return f"[WARNING: Clinical context conflict detected] {baseline_report}", condition
    if action == "abstain":
        return ABSTENTION_REPORT, "none"
    raise ValueError(f"Unsupported ReliabilityGate action: {action}")


def build_phase18_results() -> pd.DataFrame:
    eligibility, benchmark, baseline = _load_and_validate_inputs()
    benchmark_index = {_key(row): row for _, row in benchmark.iterrows()}
    baseline_lookup = {
        (str(row["sample_id"]), str(row["condition"])): _text(row["generated_report"])
        for _, row in baseline.iterrows()
    }
    analyzer = EvidenceStateAnalyzer()
    gate = ReliabilityGate()
    records: list[dict[str, Any]] = []

    for _, eligible_row in eligibility.iterrows():
        key = _key(eligible_row)
        row = benchmark_index[key]
        sample_id, _, condition = key
        baseline_row = baseline.loc[
            (baseline["sample_id"].astype(str) == sample_id)
            & (baseline["uid"].astype(int) == int(row["uid"]))
            & (baseline["condition"].astype(str) == condition)
        ].iloc[0]
        context = _text(row["perturbed_context"]).strip()
        baseline_report = _text(baseline_row["generated_report"])

        # Match Phase 13B assessment construction. Construction metadata supplies
        # relevance/consistency inputs; the condition label is never passed to the analyzer.
        donor_uid = _uid_token(row.get("source_context_uid"))
        target_uid = _uid_token(row.get("uid"))
        context_relevant = not (donor_uid and donor_uid != target_uid)
        conflict_method = _text(row.get("conflict_method")).strip()
        context_consistent = not bool(conflict_method)
        completeness = parse_context_completeness(context)
        assessment = EvidenceAssessment(
            image_available=True,
            context_available=bool(context),
            context_relevant=context_relevant,
            context_consistent=context_consistent,
            context_complete=completeness.is_complete,
            evidence_strength=1.0,
        )
        predicted_state = analyzer.classify(assessment)
        decision = gate.decide(predicted_state)
        gated_report, source_baseline_condition = _apply_existing_policy(
            decision.action,
            sample_id,
            condition,
            baseline_report,
            baseline_lookup,
        )
        output_file = f"{condition}/{sample_id}_{condition}.txt"
        records.append(
            {
                "sample_id": sample_id,
                "uid": int(row["uid"]),
                "condition": condition,
                "image_path": _text(eligible_row["image_path"]),
                "perturbed_context": context,
                "evidence_state": predicted_state.value,
                "gate_action": decision.action,
                "gate_reason": decision.reason,
                "context_complete": completeness.is_complete,
                "context_completeness_rule": completeness.rule_triggered,
                "context_available": bool(context),
                "context_relevant": context_relevant,
                "context_consistent": context_consistent,
                "evidence_strength": 1.0,
                "baseline_report": baseline_report,
                "gated_report": gated_report,
                "report_changed": gated_report != baseline_report,
                "baseline_report_length": len(baseline_report),
                "gated_report_length": len(gated_report),
                "source_baseline_condition": source_baseline_condition,
                "was_model_inference_run": False,
                "output_file": output_file,
            }
        )

    results = pd.DataFrame(records)
    if len(results) != 576 or results.duplicated(["sample_id", "uid", "condition"]).any():
        raise RuntimeError("Phase 18 results failed count/key uniqueness validation")
    counts = results["condition"].value_counts().to_dict()
    if counts != {condition: 96 for condition in REQUIRED_CONDITIONS}:
        raise RuntimeError(f"Unexpected Phase 18 condition balance: {counts}")
    return results


def write_phase18_outputs(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for row in results.itertuples(index=False):
        report_path = OUTPUT_DIR / row.output_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(row.gated_report, encoding="utf-8")
    results.to_csv(OUTPUT_CSV, index=False)

    # Read the artifacts back and verify exact serialization.
    written = pd.read_csv(OUTPUT_CSV)
    if len(written) != len(results):
        raise RuntimeError("Gated CSV row count changed after writing")
    for row in written.itertuples(index=False):
        report_path = OUTPUT_DIR / row.output_file
        if not report_path.is_file() or report_path.read_text(encoding="utf-8") != row.gated_report:
            raise RuntimeError(f"Gated report did not round-trip correctly: {row.output_file}")


def main() -> None:
    tracked_inputs = (BENCHMARK_PATH, ELIGIBILITY_PATH, BASELINE_PATH)
    hashes_before = {path: sha256(path) for path in tracked_inputs}
    baseline_files_before = baseline_tree_hashes()
    results = build_phase18_results()
    write_phase18_outputs(results)
    hashes_after = {path: sha256(path) for path in tracked_inputs}
    baseline_files_after = baseline_tree_hashes()
    if hashes_before != hashes_after or baseline_files_before != baseline_files_after:
        raise RuntimeError("A frozen Phase 17 input/baseline artifact changed during gated generation")

    print(f"Gated records: {len(results)}")
    print("Evidence-state distribution:")
    print(results["evidence_state"].value_counts().sort_index().to_string())
    print("Gate-action distribution:")
    print(results["gate_action"].value_counts().sort_index().to_string())
    print(f"Changed outputs: {int(results['report_changed'].sum())}")
    print(f"Unchanged outputs: {int((~results['report_changed']).sum())}")
    print("Additional VLM inference: 0")
    print(f"Summary: {OUTPUT_CSV}")
    print(f"Text reports: {len(results)} under {OUTPUT_DIR}")
    for path in tracked_inputs:
        print(f"SHA-256 {path.relative_to(ROOT)}: {hashes_after[path]}")


if __name__ == "__main__":
    main()
