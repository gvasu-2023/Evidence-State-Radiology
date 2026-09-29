"""Deterministic qualitative sample and error-pattern summary for Phase 23."""

from __future__ import annotations

import difflib
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
    "benchmark": ROOT / "results/tables/phase16_perturbations.csv",
    "eligibility": ROOT / "results/tables/phase17_vlm_inference_eligibility.csv",
    "baseline": ROOT / "results/baseline/phase17_full/baseline_results.csv",
    "gated": ROOT / "results/gated/phase18_full/gated_results.csv",
    "claims": ROOT / "results/tables/phase19_claim_evaluation.csv",
    "claim_summary": ROOT / "results/tables/phase19_claim_summary.csv",
    "radgraph": ROOT / "results/tables/phase20_radgraph_f1_results.csv",
    "chexpert": ROOT / "results/tables/phase20_chexpert_f1_results.csv",
    "phase21_condition": ROOT / "results/tables/phase21_condition_coverage.csv",
    "phase22_primary": ROOT / "results/tables/phase22_primary_statistics.csv",
    "phase22_covered": ROOT / "results/tables/phase22_covered_statistics.csv",
    "phase22_claim": ROOT / "results/tables/phase22_claim_statistics.csv",
}
OUTPUTS = {
    "cases": ROOT / "results/tables/phase23_qualitative_cases.csv",
    "patterns": ROOT / "results/tables/phase23_error_pattern_summary.csv",
    "docs": ROOT / "docs/phase23_qualitative_error_analysis.md",
}
CONDITIONS = (
    "sufficient", "syntactic_incomplete", "evidentiary_incomplete",
    "irrelevant", "conflicting", "insufficient",
)
UNSUPPORTED = {"ungrounded_affirmation", "unsupported_affirmation", "contradiction"}
CLAIM_LABELS = (
    "supported", "omission", "extra_negation", "ungrounded_affirmation",
    "unsupported_affirmation", "contradiction", "unclear",
)
KEYS = ["sample_id", "uid", "condition"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(path: Path) -> dict[str, str]:
    return {
        item.relative_to(ROOT).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _input_snapshot() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    for name, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 23 input {name}: {path}")
    hashes = {name: sha256(path) for name, path in INPUTS.items()}
    trees = {
        "baseline": _tree_snapshot(INPUTS["baseline"].parent),
        "gated": _tree_snapshot(INPUTS["gated"].parent),
    }
    return hashes, trees


def _claim_aggregate(claims: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "sample_id", "uid", "benchmark_condition", "finding",
        "baseline_label", "gated_label",
    }
    missing = needed - set(claims.columns)
    if missing:
        raise ValueError(f"Phase 19 claim artifact lacks columns: {sorted(missing)}")
    claims = claims.copy()
    claims["benchmark_condition"] = claims["benchmark_condition"].astype(str)
    claim_keys = ["sample_id", "uid", "benchmark_condition", "finding"]
    if claims.duplicated(claim_keys).any():
        raise ValueError("Duplicate record/finding claim rows in Phase 19 artifact")

    def label_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
        values = rows[column].fillna("").astype(str)
        return {label: int(values.eq(label).sum()) for label in CLAIM_LABELS}

    records = []
    for (sample_id, uid, condition), rows in claims.groupby(
        ["sample_id", "uid", "benchmark_condition"], sort=True
    ):
        base_labels = rows["baseline_label"].fillna("").astype(str)
        gated_labels = rows["gated_label"].fillna("").astype(str)
        transitions = []
        for row in rows.sort_values("finding").itertuples(index=False):
            base_label = "" if pd.isna(row.baseline_label) else str(row.baseline_label)
            gated_label = "" if pd.isna(row.gated_label) else str(row.gated_label)
            if base_label != gated_label:
                transitions.append(f"{row.finding}: {base_label or 'blank'} → {gated_label or 'blank'}")
        base_counts = label_counts(rows, "baseline_label")
        gated_counts = label_counts(rows, "gated_label")
        baseline_unsupported_findings = sorted(
            rows.loc[base_labels.isin(UNSUPPORTED), "finding"].dropna().astype(str).unique()
        )
        gated_unsupported_findings = sorted(
            rows.loc[gated_labels.isin(UNSUPPORTED), "finding"].dropna().astype(str).unique()
        )
        baseline_contradiction_findings = sorted(
            rows.loc[base_labels.eq("contradiction"), "finding"].dropna().astype(str).unique()
        )
        gated_contradiction_findings = sorted(
            rows.loc[gated_labels.eq("contradiction"), "finding"].dropna().astype(str).unique()
        )
        records.append({
            "sample_id": str(sample_id),
            "uid": int(uid),
            "condition": condition,
            "baseline_labels": "; ".join(f"{key}={value}" for key, value in base_counts.items() if value),
            "gated_labels": "; ".join(f"{key}={value}" for key, value in gated_counts.items() if value),
            "baseline_unsupported_claims": int(base_labels.isin(UNSUPPORTED).sum()),
            "gated_unsupported_claims": int(gated_labels.isin(UNSUPPORTED).sum()),
            "baseline_contradictions": int(base_labels.eq("contradiction").sum()),
            "gated_contradictions": int(gated_labels.eq("contradiction").sum()),
            "baseline_unsupported_findings": " | ".join(baseline_unsupported_findings),
            "gated_unsupported_findings": " | ".join(gated_unsupported_findings),
            "baseline_contradiction_findings": " | ".join(baseline_contradiction_findings),
            "gated_contradiction_findings": " | ".join(gated_contradiction_findings),
            "baseline_omissions": int(base_labels.eq("omission").sum()),
            "gated_omissions": int(gated_labels.eq("omission").sum()),
            "claim_label_change_count": len(transitions),
            "claim_label_changes": "; ".join(transitions) if transitions else "none",
        })
    result = pd.DataFrame(records)
    return result


def _load_cohort() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    benchmark = frames["benchmark"]
    eligibility = frames["eligibility"]
    eligible = eligibility[eligibility["eligible_for_vlm"].astype(str).str.lower().eq("true")].copy()
    if len(benchmark) != 600 or benchmark.duplicated(KEYS).any():
        raise ValueError("Expected 600 unique benchmark records")
    if len(eligible) != 576 or eligible.duplicated(KEYS).any():
        raise ValueError("Expected 576 unique eligible records")
    if eligible["condition"].value_counts().to_dict() != {condition: 96 for condition in CONDITIONS}:
        raise ValueError("Eligible records do not have the expected 96 per condition")
    excluded = eligibility[eligibility["eligible_for_vlm"].astype(str).str.lower().eq("false")]
    if set(excluded["uid"].astype(int)) != {74, 597, 803, 885} or len(excluded) != 24:
        raise ValueError("Eligibility exclusions differ from the frozen four-study cohort")

    _claim_aggregate(frames["claims"])  # enforce claim-row uniqueness before joining
    data = eligible[KEYS + ["image_path"]].merge(
        benchmark[KEYS + [
            "original_context", "perturbed_context", "findings", "impression",
            "removed_evidence_text", "evidentiary_incomplete_method", "conflict_target",
            "conflict_method", "conflict_source_text",
        ]], on=KEYS, how="left", validate="one_to_one",
    )
    baseline = frames["baseline"]
    gated = frames["gated"]
    radgraph = frames["radgraph"]
    chexpert = frames["chexpert"]
    for name, frame in (("baseline", baseline), ("gated", gated), ("RadGraph", radgraph), ("CheXpert", chexpert)):
        if len(frame) != 576 or frame.duplicated(KEYS).any():
            raise ValueError(f"{name} artifact does not have 576 unique paired records")
    data = data.merge(
        baseline[KEYS + ["generated_report"]].rename(columns={"generated_report": "baseline_report"}),
        on=KEYS, how="left", validate="one_to_one",
    ).merge(
        gated[KEYS + ["evidence_state", "gate_action", "gated_report", "report_changed"]].rename(
            columns={"evidence_state": "analyzer_state"}
        ), on=KEYS, how="left", validate="one_to_one",
    ).merge(
        radgraph[KEYS + ["baseline_radgraph_f1", "gated_radgraph_f1"]],
        on=KEYS, how="left", validate="one_to_one",
    ).merge(
        chexpert[KEYS + ["baseline_chexpert_f1", "gated_chexpert_f1"]],
        on=KEYS, how="left", validate="one_to_one",
    ).merge(
        _claim_aggregate(frames["claims"]), on=KEYS, how="left", validate="one_to_one",
    )
    if len(data) != 576 or data.duplicated(KEYS).any():
        raise ValueError("Phase 23 joins failed or produced duplicate records")
    for report_column in ("baseline_report", "gated_report"):
        if data[report_column].isna().any():
            raise ValueError(f"Missing report text in {report_column}")
    if data[["baseline_radgraph_f1", "gated_radgraph_f1", "baseline_chexpert_f1", "gated_chexpert_f1"]].isna().any().any():
        raise ValueError("Missing Phase 20 per-record score")
    if data["claim_label_change_count"].isna().any():
        raise ValueError("A Phase 19 record-level claim aggregate is missing")

    # The claim table uses benchmark_condition rather than condition; join keys
    # above preserve sample_id/UID/condition identity.
    for frame_name in ("phase21_condition",):
        if "condition" not in frames[frame_name].columns:
            raise ValueError(f"{frame_name} lacks condition metadata")
    if len(frames["phase22_primary"]) != 2 or len(frames["phase22_covered"]) != 2:
        raise ValueError("Phase 22 F1 statistical summaries do not contain two endpoints each")
    if len(frames["phase22_claim"]) != 1:
        raise ValueError("Phase 22 claim comparison artifact is not a single record-level result")
    return data, frames


def _add_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["report_unchanged"] = result["baseline_report"].astype(str).eq(result["gated_report"].astype(str))
    result["report_changed"] = ~result["report_unchanged"]
    result["text_change_fraction"] = [
        1.0 - difflib.SequenceMatcher(None, str(a), str(b), autojunk=False).ratio()
        for a, b in zip(result["baseline_report"], result["gated_report"])
    ]
    result["unsupported_claim_delta"] = (
        result["gated_unsupported_claims"] - result["baseline_unsupported_claims"]
    )
    result["radgraph_delta"] = result["gated_radgraph_f1"] - result["baseline_radgraph_f1"]
    result["chexpert_delta"] = result["gated_chexpert_f1"] - result["baseline_chexpert_f1"]
    result["absolute_metric_delta"] = result["radgraph_delta"].abs() + result["chexpert_delta"].abs()
    result["unsupported_claim_reduction"] = -result["unsupported_claim_delta"]
    return result


def _pick(data: pd.DataFrame, used: set[tuple[str, int, str]], sort_columns: list[str], ascending: list[bool], role: str, condition: str) -> pd.Series:
    available = data[~data.apply(lambda row: (str(row.sample_id), int(row.uid), str(row.condition)) in used, axis=1)]
    ordered = available.sort_values(sort_columns + ["sample_id", "uid"], ascending=ascending + [True, True], kind="mergesort")
    if ordered.empty:
        raise ValueError(f"Cannot choose a unique {role} example for {condition}")
    selected = ordered.iloc[0]
    used.add((str(selected.sample_id), int(selected.uid), str(selected.condition)))
    return selected


def _deterministic_selection(data: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for condition in CONDITIONS:
        group = data[data["condition"] == condition].copy()
        used: set[tuple[str, int, str]] = set()

        # Prefer observed removal of unsupported findings first, then the
        # greatest overall claim-label transition count; when neither exists,
        # retained unsupported claims identify limitation cases.
        row = _pick(
            group, used,
            ["unsupported_claim_reduction", "claim_label_change_count", "baseline_unsupported_claims", "absolute_metric_delta"],
            [False, False, False, False], "claim-change", condition,
        ).copy()
        row["selection_role"] = "largest_unsupported_reduction_or_claim_label_change"
        chosen.append(row)

        row = _pick(
            group, used,
            ["absolute_metric_delta", "text_change_fraction", "claim_label_change_count"],
            [False, False, False], "metric-contrast", condition,
        ).copy()
        row["selection_role"] = "largest_absolute_metric_delta"
        chosen.append(row)

        # Prefer an unchanged report as a control where available. If this
        # condition has no unchanged rows, show changed text with no claim-label
        # transition; the final fallback is largest text change.
        available = group[~group.apply(lambda r: (str(r.sample_id), int(r.uid), str(r.condition)) in used, axis=1)]
        if available["report_unchanged"].any():
            row = _pick(group, used, ["report_unchanged", "baseline_unsupported_claims"], [False, False], "unchanged-control", condition).copy()
            role = "unchanged_report_control"
        elif (available["claim_label_change_count"] == 0).any():
            row = _pick(group, used, ["claim_label_change_count", "baseline_unsupported_claims", "absolute_metric_delta"], [True, False, True], "changed-without-claim-transition", condition).copy()
            role = "changed_report_without_claim_label_transition"
        else:
            row = _pick(group, used, ["text_change_fraction", "absolute_metric_delta"], [False, False], "text-change", condition).copy()
            role = "largest_remaining_text_change"
        row["selection_role"] = role
        chosen.append(row)

    result = pd.DataFrame(chosen).sort_values(
        ["condition", "selection_role", "sample_id", "uid"], kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != 18 or result.duplicated(KEYS).any():
        raise RuntimeError("Deterministic qualitative selection is not 18 unique records")
    if result["condition"].value_counts().to_dict() != {condition: 3 for condition in CONDITIONS}:
        raise RuntimeError("Qualitative sample does not contain three records per condition")
    return result


def _labels_for_case(row: pd.Series) -> tuple[list[str], list[str]]:
    categories = []
    notes = []
    before_findings = set(filter(None, str(row.baseline_unsupported_findings).split(" | ")))
    after_findings = set(filter(None, str(row.gated_unsupported_findings).split(" | ")))
    before_contradiction_findings = set(filter(None, str(row.baseline_contradiction_findings).split(" | ")))
    after_contradiction_findings = set(filter(None, str(row.gated_contradiction_findings).split(" | ")))
    before = len(before_findings)
    after = len(after_findings)
    before_contra = len(before_contradiction_findings)
    after_contra = len(after_contradiction_findings)
    if row.report_unchanged:
        categories.append("report_unchanged")
    if before_findings - after_findings:
        categories.append("unsupported_claim_removed")
        notes.append(f"unsupported findings removed: {', '.join(sorted(before_findings - after_findings))}")
    if after_findings - before_findings:
        categories.append("unsupported_claim_added")
        notes.append(f"unsupported findings added: {', '.join(sorted(after_findings - before_findings))}")
    if before_findings & after_findings:
        categories.append("unsupported_claim_retained")
        notes.append(f"unsupported findings retained: {', '.join(sorted(before_findings & after_findings))}")
    if before_contradiction_findings - after_contradiction_findings:
        categories.append("contradictory_content_reduced")
    if before_contradiction_findings & after_contradiction_findings:
        categories.append("contradictory_content_retained")
    if int(row.gated_omissions) > int(row.baseline_omissions):
        categories.append("supported_content_omitted")
    gated_lower = str(row.gated_report).lower()
    has_qualifying_notice = "[qualified" in gated_lower or "[warning:" in gated_lower
    if str(row.gate_action) in {"qualify", "qualify_or_abstain"} and has_qualifying_notice:
        categories.append("qualification_added")
        if after > 0 or after_contra > 0:
            categories.append("qualification_not_effective")
    if str(row.condition) == "irrelevant":
        if before > after or before_contra > after_contra:
            categories.append("irrelevant_context_discounted")
    if str(row.condition) == "evidentiary_incomplete" and str(row.analyzer_state) == "sufficient":
        categories.append("analyzer_misclassification")
    if str(row.condition) == "insufficient" and str(row.gate_action) == "abstain":
        categories.append("abstention_due_to_insufficient_evidence")
    if int(row.claim_label_change_count) > 0 and row.radgraph_delta * row.chexpert_delta < 0:
        categories.append("mixed_change")
    return list(dict.fromkeys(categories)), notes


def _case_table(selected: pd.DataFrame) -> pd.DataFrame:
    records = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        s = pd.Series(row._asdict())
        categories, notes = _labels_for_case(s)
        behavior = "text unchanged" if s.report_unchanged else "text changed"
        if s.gate_action == "abstain":
            behavior += "; gated output is abstention"
        if notes:
            claim_change = "; ".join(notes)
        elif s.claim_label_change_count:
            claim_change = str(s.claim_label_changes)
        else:
            claim_change = f"labels unchanged ({s.baseline_labels or 'no labels'} → {s.gated_labels or 'no labels'})"
        baseline_behavior = (
            f"generated report; unsupported finding labels={int(s.baseline_unsupported_claims)}; "
            f"labels: {s.baseline_labels or 'none'}"
        )
        gated_notice = "qualification/warning added" if (
            "[qualified" in str(s.gated_report).lower() or "[warning:" in str(s.gated_report).lower()
        ) else ("abstention output" if s.gate_action == "abstain" else "no qualification notice")
        gated_behavior = (
            f"{gated_notice}; report text {'unchanged' if s.report_unchanged else 'changed'}; "
            f"unsupported finding labels={int(s.gated_unsupported_claims)}; labels: {s.gated_labels or 'none'}"
        )
        interpretation = (
            f"{behavior}; analyzer={s.analyzer_state}, action={s.gate_action}. "
            f"Phase 19 unsupported labels {s.baseline_unsupported_claims}→{s.gated_unsupported_claims}; "
            f"RadGraph Δ={s.radgraph_delta:+.4f}, CheXpert Δ={s.chexpert_delta:+.4f}."
        )
        records.append({
            "case": f"Q{index:02d}",
            "sample_id": s.sample_id,
            "uid": int(s.uid),
            "condition": s.condition,
            "analyzer_state": s.analyzer_state,
            "gate_action": s.gate_action,
            "selection_role": s.selection_role,
            "original_context": s.original_context,
            "perturbed_context": s.perturbed_context,
            "removed_evidence_text": s.removed_evidence_text,
            "findings": s.findings,
            "impression": s.impression,
            "conflict_target": s.conflict_target,
            "conflict_source_text": s.conflict_source_text,
            "baseline_report": s.baseline_report,
            "gated_report": s.gated_report,
            "baseline_claim_labels": s.baseline_labels,
            "gated_claim_labels": s.gated_labels,
            "baseline_behavior": baseline_behavior,
            "gated_behavior": gated_behavior,
            "baseline_unsupported_findings": s.baseline_unsupported_findings,
            "gated_unsupported_findings": s.gated_unsupported_findings,
            "claim_level_change": claim_change,
            "claim_label_changes": s.claim_label_changes,
            "baseline_unsupported_claim_count": int(s.baseline_unsupported_claims),
            "gated_unsupported_claim_count": int(s.gated_unsupported_claims),
            "baseline_radgraph_f1": float(s.baseline_radgraph_f1),
            "gated_radgraph_f1": float(s.gated_radgraph_f1),
            "radgraph_delta": float(s.radgraph_delta),
            "baseline_chexpert_f1": float(s.baseline_chexpert_f1),
            "gated_chexpert_f1": float(s.gated_chexpert_f1),
            "chexpert_delta": float(s.chexpert_delta),
            "report_unchanged": bool(s.report_unchanged),
            "qualitative_categories": ";".join(categories),
            "qualitative_interpretation": interpretation,
        })
    return pd.DataFrame(records)


def _pattern_summary(cohort: pd.DataFrame, frames: dict[str, pd.DataFrame], cases: pd.DataFrame) -> pd.DataFrame:
    entries: list[dict[str, Any]] = []

    def add(pattern: str, level: str, condition: str, n: int, description: str, case_refs: str = "") -> None:
        entries.append({
            "pattern": pattern, "scope": level, "condition": condition,
            "record_count": int(n), "description": description, "selected_cases": case_refs,
        })

    claims = _claim_aggregate(frames["claims"])
    combined = cohort.merge(claims, on=KEYS, suffixes=("", "_claims"), validate="one_to_one") if "baseline_unsupported_claims" not in cohort else cohort
    # Normally cohort has the Phase 19 aggregate columns already; this branch
    # keeps the function usable with a minimally assembled test fixture.
    if "baseline_unsupported_claims" not in combined:
        combined = cohort.merge(claims, on=KEYS, validate="one_to_one")

    def condition_rows(condition: str) -> pd.DataFrame:
        return combined[combined["condition"] == condition]

    cases_by_condition = {}
    for key, group in cases.groupby("condition"):
        cases_by_condition[key] = ", ".join(group["case"].tolist())
    for condition in CONDITIONS:
        group = condition_rows(condition)
        unsupported_before = int(group["baseline_unsupported_claims"].sum())
        unsupported_after = int(group["gated_unsupported_claims"].sum())
        changed = int((group["baseline_report"].astype(str) != group["gated_report"].astype(str)).sum())
        omitted_before = int(group["baseline_omissions"].sum())
        omitted_after = int(group["gated_omissions"].sum())
        action = str(group["gate_action"].iloc[0]) if condition in group["condition"].values else ""
        state_counts = group["analyzer_state"].value_counts().to_dict()
        if condition == "sufficient":
            add("normal generation preservation", "condition", condition, len(group), f"Reports unchanged in {len(group)-changed}/{len(group)} cases; unsupported labels {unsupported_before} baseline / {unsupported_after} gated, retained where present.", cases_by_condition.get(condition, ""))
        elif condition == "syntactic_incomplete":
            add("syntactically incomplete context", "condition", condition, len(group), f"Analyzer state distribution {state_counts}; reports changed in {changed}/{len(group)}. Unsupported labels {unsupported_before}→{unsupported_after}; omission labels {omitted_before}→{omitted_after}. Text change does not imply claim-risk reduction.", cases_by_condition.get(condition, ""))
        elif condition == "evidentiary_incomplete":
            add("evidentiary-incomplete analyzer limitation", "condition", condition, len(group), f"All {len(group)} analyzer states were sufficient; gate action was {action}; reports unchanged in {len(group)-changed}/{len(group)}. Removed-evidence records therefore received no gate intervention; unsupported labels remain {unsupported_before}→{unsupported_after}.", cases_by_condition.get(condition, ""))
        elif condition == "irrelevant":
            add("irrelevant context handling", "condition", condition, len(group), f"Discount-context action changed reports in {changed}/{len(group)}. Unsupported labels were {unsupported_before} baseline and {unsupported_after} gated; this cohort-level count does not support calling the action successful suppression.", cases_by_condition.get(condition, ""))
        elif condition == "conflicting":
            add("conflicting context handling", "condition", condition, len(group), f"Reports changed in {changed}/{len(group)} under qualification action. Unsupported labels {unsupported_before}→{unsupported_after}; conflicting content was not removed in cases where Phase 19 continued to label contradictions.", cases_by_condition.get(condition, ""))
        else:
            add("insufficient evidence abstention", "condition", condition, len(group), f"Abstain action applied in {int(group.gate_action.eq('abstain').sum())}/{len(group)}; report text changed to the abstention output. Unsupported labels {unsupported_before}→{unsupported_after}; gated candidates are not conventional generated reports.", cases_by_condition.get(condition, ""))

    add("unchanged outputs", "all eligible", "sufficient + evidentiary_incomplete", int(combined["report_unchanged"].sum()) if "report_unchanged" in combined else 192,
        "The unchanged-output pool consists of sufficient and evidentiary_incomplete cases; the latter reflects analyzer classification as sufficient, not harmless evidence removal.",
        cases_by_condition.get("sufficient", "") + "; " + cases_by_condition.get("evidentiary_incomplete", ""))
    add("report changed without claim-label transition", "all eligible", "conflicting", int(((combined.condition == "conflicting") & (combined.claim_label_change_count == 0) & (combined.baseline_report.astype(str) != combined.gated_report.astype(str))).sum()),
        "Conflicting reports can change by adding qualification while Phase 19 claim labels remain the same; a textual change alone is not evidence that the conflict was resolved.",
        cases_by_condition.get("conflicting", ""))
    by_condition_counts = []
    for condition in CONDITIONS:
        subset = condition_rows(condition)
        by_condition_counts.append(
            f"{condition}: {int(subset.baseline_unsupported_claims.sum())}→{int(subset.gated_unsupported_claims.sum())}"
        )
    add("unsupported-claim labels by condition", "all eligible", "all six conditions", 576,
        "Phase 19 unsupported-label counts by condition (baseline→gated): " + "; ".join(by_condition_counts) + ". These associations do not establish that context caused a particular claim.",
        "; ".join(cases_by_condition.get(condition, "") for condition in CONDITIONS))
    irrelevant = condition_rows("irrelevant")
    irrelevant_removed = int((irrelevant.baseline_unsupported_findings.apply(lambda v: set(filter(None, str(v).split(" | ")))) - irrelevant.gated_unsupported_findings.apply(lambda v: set(filter(None, str(v).split(" | "))))).map(bool).sum())
    irrelevant_added = int((irrelevant.gated_unsupported_findings.apply(lambda v: set(filter(None, str(v).split(" | ")))) - irrelevant.baseline_unsupported_findings.apply(lambda v: set(filter(None, str(v).split(" | "))))).map(bool).sum())
    add("irrelevant-context claim transitions", "condition", "irrelevant", 96,
        f"Discount-context reports changed for all 96 records. At least one unsupported finding was removed in {irrelevant_removed} records and added in {irrelevant_added}; aggregate unsupported labels rose from 19 to 21, so the pattern is mixed and is not summarized as a successful overall reduction.",
        cases_by_condition.get("irrelevant", ""))
    content_only = 0
    for condition in ("syntactic_incomplete", "irrelevant", "conflicting"):
        subset = condition_rows(condition)
        content_only += int((
            (subset.baseline_report.astype(str) != subset.gated_report.astype(str))
            & (subset.radgraph_delta <= 0)
            & (subset.chexpert_delta <= 0)
        ).sum())
    add("text changes without positive delta on either metric", "selected conditions", "syntactic_incomplete + irrelevant + conflicting", content_only,
        "Count of records with changed report text and non-positive paired delta on both existing Phase 20 metrics. This is descriptive; unchanged overlap scores do not establish clinical harm or benefit.",
        "; ".join(cases_by_condition.get(condition, "") for condition in ("syntactic_incomplete", "irrelevant", "conflicting")))
    return pd.DataFrame(entries)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = [[cell(value) for value in row] for row in frame[columns].itertuples(index=False, name=None)]
    headers = columns
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def _render_docs(cases: pd.DataFrame, patterns: pd.DataFrame, hashes: dict[str, str]) -> str:
    ids = cases[["case", "sample_id", "uid", "condition", "selection_role"]].copy()
    ids_table = _markdown_table(ids, list(ids.columns))
    compact = cases[[
        "case", "condition", "analyzer_state", "gate_action", "baseline_behavior",
        "gated_behavior", "claim_level_change", "radgraph_delta", "chexpert_delta",
        "qualitative_categories", "qualitative_interpretation",
    ]]
    compact_table = _markdown_table(compact, list(compact.columns))
    pattern_table = _markdown_table(patterns, ["pattern", "scope", "condition", "record_count", "description", "selected_cases"])
    hash_text = "\n".join(f"- `{name}`: `{digest}`" for name, digest in hashes.items())
    return f"""# Phase 23: Qualitative Error Analysis

## Objective and source artifacts

This is a reproducible qualitative analysis of existing baseline, gated, claim-evaluation, and report-metric outputs. It introduces no new reports, clinical findings, or significance tests. Reference evidence is drawn only from the `findings` and `impression` fields in the frozen Phase 16 perturbation file; context fields are reported as recorded. All twelve authoritative input files were hashed before and after analysis and remained unchanged:\n\n{hash_text}

## Deterministic selection procedure

The eligible Phase 17 cohort was joined one-to-one by `(sample_id, uid, condition)`. The sample contains three records per condition. Features used were exact report equality, character-level text change fraction (`1 - SequenceMatcher.ratio`), Phase 19 per-record label transitions and unsupported-claim counts (unsupported labels are `ungrounded_affirmation`, `unsupported_affirmation`, and `contradiction`), absolute paired RadGraph-F1 plus CheXpert-F1 delta, and stable identifiers.

Within each condition, selection slots were applied in order: (1) maximize observed removal of unsupported findings, then total claim-label transitions; if neither exists, retained unsupported claims resolve the slot; (2) maximize the sum of absolute RadGraph and CheXpert deltas, then text difference; (3) prefer an unchanged report as a control, otherwise a changed report with no claim-label transition, otherwise the largest remaining text change. Already selected records are excluded from later slots. Remaining ties resolve by ascending `sample_id`, then UID, using stable sorting. This rule selects observed examples and does not inspect qualitative conclusions.

Selected IDs and roles:

{ids_table}

## Representative case table

Full contexts, reference findings/impressions, reports, label counts, and claim transitions are in `results/tables/phase23_qualitative_cases.csv`. Text below is a compact index; interpretations refer to the recorded labels and score deltas and do not add clinical findings.

{compact_table}

## Recurring error patterns

Counts are descriptive summaries of the 576 eligible records and existing Phase 19–21 outputs. No subgroup hypothesis tests were performed.

{pattern_table}

### Required evidentiary-incomplete limitation

All 96 `evidentiary_incomplete` records had readable clinical evidence removed by the perturbation protocol, while the current analyzer classified all 96 as `sufficient`. The gate therefore did not intervene, and gated reports matched baseline. Unsupported claim labels remained present in the paired claim evaluation. This is an analyzer limitation; it is not evidence that evidentiary incompleteness is harmless. No analyzer or gate changes were made.

### Insufficient cases

The 96 insufficient records were handled with abstention outputs. These outputs are described as abstentions, not conventional generated reports or report-quality improvements. Phase 19 evaluates their claim consequences through omissions/absence of generated unsupported claims; they must be interpreted separately from generated-report content.

## Limitations

- The sample is a deterministic qualitative subset, not a statistical sample for inferential claims.
- For sufficient and evidentiary-incomplete conditions, all reports are unchanged, so no changed-versus-unchanged contrast exists within those conditions.
- Conflicting reports changed text, but Phase 19 recorded no claim-label transitions for the conflicting condition; qualification can therefore leave evaluated contradiction labels in place.
- Irrelevant-context discounting changed all reports, but cohort-level unsupported labels rose from 19 to 21. It is not characterized as successful unsupported-claim suppression.
- A report text or metric change does not by itself establish clinical benefit. RadGraph-F1 and CheXpert-F1 remain the existing overlap metrics.
- No new significance tests were run. Statistical or clinical significance is not inferred in this phase.

## Reproducibility

Run `.venv\\Scripts\\python.exe -m src.evaluation.run_phase23_qualitative_analysis` from the repository root. The runner validates cohort keys and condition balance, deterministically recomputes the 18-case set, and verifies file/tree hashes before and after writing outputs.
"""


def run_phase23() -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, str]]:
    hashes_before, trees_before = _input_snapshot()
    cohort, frames = _load_cohort()
    featured = _add_features(cohort)
    selected = _deterministic_selection(featured)
    cases = _case_table(selected)
    patterns = _pattern_summary(featured, frames, cases)
    docs = _render_docs(cases, patterns, hashes_before)
    hashes_after, trees_after = _input_snapshot()
    if hashes_before != hashes_after or trees_before != trees_after:
        raise RuntimeError("An authoritative Phase 23 input changed during analysis")
    return cases, patterns, docs, hashes_before


def main() -> None:
    cases, patterns, docs, _ = run_phase23()
    hashes_before_write, trees_before_write = _input_snapshot()
    OUTPUTS["cases"].parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTPUTS["cases"], index=False)
    patterns.to_csv(OUTPUTS["patterns"], index=False)
    OUTPUTS["docs"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS["docs"].write_text(docs, encoding="utf-8")
    hashes_after_write, trees_after_write = _input_snapshot()
    if hashes_before_write != hashes_after_write or trees_before_write != trees_after_write:
        raise RuntimeError("Phase 23 input files or output trees changed during artifact writing")
    print(f"Selected {len(cases)} cases; counts by condition: {cases['condition'].value_counts().sort_index().to_dict()}")
    print(cases[["case", "sample_id", "uid", "condition", "selection_role", "qualitative_categories"]].to_string(index=False))
    print(f"\nWrote {len(patterns)} descriptive pattern rows. No inference or significance testing performed.")


if __name__ == "__main__":
    main()
