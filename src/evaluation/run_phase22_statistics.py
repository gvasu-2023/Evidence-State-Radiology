"""Paired Phase 22 statistical analysis over frozen Phase 17-21 artifacts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ELIGIBILITY_PATH = ROOT / "results/tables/phase17_vlm_inference_eligibility.csv"
BASELINE_PATH = ROOT / "results/baseline/phase17_full/baseline_results.csv"
GATED_PATH = ROOT / "results/gated/phase18_full/gated_results.csv"
CLAIM_PATH = ROOT / "results/tables/phase19_claim_evaluation.csv"
RADGRAPH_PATH = ROOT / "results/tables/phase20_radgraph_f1_results.csv"
CHEXPERT_PATH = ROOT / "results/tables/phase20_chexpert_f1_results.csv"
PHASE21_SUMMARY_PATH = ROOT / "results/tables/phase21_risk_coverage_summary.csv"
PHASE21_CONDITION_PATH = ROOT / "results/tables/phase21_condition_coverage.csv"
BENCHMARK_PATH = ROOT / "results/tables/phase16_perturbations.csv"

OUTPUTS = {
    "primary": ROOT / "results/tables/phase22_primary_statistics.csv",
    "covered": ROOT / "results/tables/phase22_covered_statistics.csv",
    "claim": ROOT / "results/tables/phase22_claim_statistics.csv",
    "diagnostics": ROOT / "results/tables/phase22_difference_diagnostics.csv",
    "docs": ROOT / "docs/phase22_statistical_analysis.md",
}

CONDITIONS = (
    "sufficient", "syntactic_incomplete", "evidentiary_incomplete",
    "irrelevant", "conflicting", "insufficient",
)
EXCLUDED_UIDS = {74, 597, 803, 885}
UNSUPPORTED_LABELS = {
    "ungrounded_affirmation", "unsupported_affirmation", "contradiction",
}
KEYS = ["sample_id", "condition"]
BENCHMARK_SHA256 = "3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53"
ELIGIBILITY_SHA256 = "cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6"
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def _tree_snapshot(path: Path) -> dict[str, str]:
    return {
        item.relative_to(ROOT).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _check_unique(frame: pd.DataFrame, name: str, keys: list[str] = KEYS) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate keys: {keys}")


def _load_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, dict[str, str]]]:
    files = [
        ELIGIBILITY_PATH, BASELINE_PATH, GATED_PATH, CLAIM_PATH,
        RADGRAPH_PATH, CHEXPERT_PATH, PHASE21_SUMMARY_PATH,
        PHASE21_CONDITION_PATH, BENCHMARK_PATH,
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen Phase 22 input is missing: {path}")

    file_hashes = _file_snapshot(files)
    if file_hashes[str(ELIGIBILITY_PATH.relative_to(ROOT))] != ELIGIBILITY_SHA256:
        raise ValueError("Phase 17 eligibility SHA-256 differs from its frozen hash")
    if file_hashes[str(BENCHMARK_PATH.relative_to(ROOT))] != BENCHMARK_SHA256:
        raise ValueError("Frozen Phase 16 benchmark SHA-256 differs from its authoritative hash")
    tree_hashes = {
        "baseline": _tree_snapshot(BASELINE_PATH.parent),
        "gated": _tree_snapshot(GATED_PATH.parent),
    }

    eligibility_all = pd.read_csv(ELIGIBILITY_PATH)
    _require_columns(eligibility_all, ["sample_id", "uid", "condition", "eligible_for_vlm"], "eligibility")
    eligibility = eligibility_all[
        eligibility_all["eligible_for_vlm"].astype(str).str.lower().eq("true")
    ].copy()
    baseline = pd.read_csv(BASELINE_PATH)
    gated = pd.read_csv(GATED_PATH)
    claims = pd.read_csv(CLAIM_PATH)
    radgraph = pd.read_csv(RADGRAPH_PATH)
    chexpert = pd.read_csv(CHEXPERT_PATH)
    phase21_summary = pd.read_csv(PHASE21_SUMMARY_PATH)
    phase21_conditions = pd.read_csv(PHASE21_CONDITION_PATH)

    for name, frame in (
        ("eligible cohort", eligibility), ("baseline", baseline), ("gated", gated),
        ("RadGraph", radgraph), ("CheXpert", chexpert),
    ):
        _require_columns(frame, ["sample_id", "uid", "condition"], name)
        _check_unique(frame, name)
    if len(eligibility) != 576 or eligibility["condition"].value_counts().to_dict() != {c: 96 for c in CONDITIONS}:
        raise ValueError("Expected 576 eligible records with 96 per condition")
    if set(pd.to_numeric(eligibility["uid"], errors="raise").astype(int)) & EXCLUDED_UIDS:
        raise ValueError("An excluded image-unavailable UID appears in the eligible cohort")
    expected_keys = set(zip(eligibility["sample_id"].astype(str), eligibility["condition"].astype(str)))
    for name, frame in (("baseline", baseline), ("gated", gated), ("RadGraph", radgraph), ("CheXpert", chexpert)):
        frame_keys = set(zip(frame["sample_id"].astype(str), frame["condition"].astype(str)))
        if len(frame) != 576 or frame_keys != expected_keys:
            raise ValueError(f"{name} records are not a one-to-one pairing with eligibility")
        uid_by_key = frame.set_index(KEYS)["uid"].astype(int)
        eligible_uid_by_key = eligibility.set_index(KEYS)["uid"].astype(int)
        if not uid_by_key.sort_index().equals(eligible_uid_by_key.sort_index()):
            raise ValueError(f"{name} UID values disagree with eligibility pairing")

    _require_columns(radgraph, ["baseline_radgraph_f1", "gated_radgraph_f1", "gated_report_covered"], "RadGraph")
    _require_columns(chexpert, ["baseline_chexpert_f1", "gated_chexpert_f1", "gated_report_covered"], "CheXpert")
    scores = radgraph[KEYS + ["uid", "baseline_radgraph_f1", "gated_radgraph_f1", "gated_report_covered"]].merge(
        chexpert[KEYS + ["baseline_chexpert_f1", "gated_chexpert_f1", "gated_report_covered"]],
        on=KEYS, how="inner", validate="one_to_one", suffixes=("_radgraph", "_chexpert"),
    )
    for column in (
        "baseline_radgraph_f1", "gated_radgraph_f1",
        "baseline_chexpert_f1", "gated_chexpert_f1",
    ):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
        if scores[column].isna().any() or not np.isfinite(scores[column]).all():
            raise ValueError(f"Missing or non-finite values found in {column}")
    if not scores["gated_report_covered_radgraph"].astype(bool).equals(
        scores["gated_report_covered_chexpert"].astype(bool)
    ):
        raise ValueError("Phase 20 RadGraph and CheXpert coverage flags disagree")
    scores["gated_report_covered"] = scores["gated_report_covered_radgraph"].astype(bool)
    if int(scores["gated_report_covered"].sum()) != 480:
        raise ValueError("Expected 480 covered records from Phase 20")

    _require_columns(phase21_summary, ["view", "eligible_n", "covered_n", "abstained_n"], "Phase 21 overall summary")
    overall = phase21_summary.set_index("view")
    if "gated_all_records" not in overall.index:
        raise ValueError("Phase 21 summary lacks gated_all_records")
    phase21_all = overall.loc["gated_all_records"]
    if (int(phase21_all["eligible_n"]), int(phase21_all["covered_n"]), int(phase21_all["abstained_n"])) != (576, 480, 96):
        raise ValueError("Phase 21 overall coverage summary disagrees with expected cohort")
    _require_columns(phase21_conditions, ["condition", "n", "covered_n", "abstained_n"], "Phase 21 condition summary")
    if phase21_conditions.set_index("condition")["n"].to_dict() != {c: 96 for c in CONDITIONS}:
        raise ValueError("Phase 21 condition summary does not contain 96 records per condition")

    _require_columns(claims, [
        "sample_id", "uid", "benchmark_condition", "finding", "baseline_label", "gated_label",
    ], "Phase 19 claim evaluation")
    claim_key = ["sample_id", "uid", "benchmark_condition", "finding"]
    if claims.duplicated(claim_key).any():
        raise ValueError("Phase 19 claim table contains duplicate record/finding keys")
    claim_record_keys = set(zip(claims["sample_id"].astype(str), claims["benchmark_condition"].astype(str)))
    if claim_record_keys != expected_keys:
        raise ValueError("Phase 19 claim table does not represent exactly the eligible record keys")
    claim_uid_by_key = (
        claims[["sample_id", "benchmark_condition", "uid"]]
        .drop_duplicates()
        .set_index(["sample_id", "benchmark_condition"])["uid"]
        .astype(int)
    )
    eligible_uid_by_key = eligibility.set_index(KEYS)["uid"].astype(int)
    if not claim_uid_by_key.sort_index().equals(eligible_uid_by_key.sort_index()):
        raise ValueError("Phase 19 UID values disagree with eligibility pairing")
    if not claims["baseline_label"].dropna().isin({
        "supported", "omission", "extra_negation", "ungrounded_affirmation", "unsupported_affirmation", "contradiction", "unclear",
    }).all() or not claims["gated_label"].dropna().isin({
        "supported", "omission", "extra_negation", "ungrounded_affirmation", "unsupported_affirmation", "contradiction", "unclear",
    }).all():
        raise ValueError("Phase 19 contains unexpected evaluation labels")

    _require_columns(baseline, ["generated_report"], "baseline")
    _require_columns(gated, ["gated_report", "gate_action"], "gated")
    if baseline["generated_report"].isna().any() or gated["gated_report"].isna().any():
        raise ValueError("A baseline or gated report is missing")

    frames = {
        "eligibility": eligibility,
        "baseline": baseline,
        "gated": gated,
        "claims": claims,
        "scores": scores,
        "phase21_summary": phase21_summary,
        "phase21_conditions": phase21_conditions,
    }
    return frames, file_hashes, tree_hashes


def _paired_effect_and_test(differences: np.ndarray) -> tuple[float, float, float]:
    """Return SciPy Wilcoxon W, approximate two-sided p, and paired rank-biserial r."""
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0.0
    test = wilcoxon(
        differences,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="approx",
    )
    ranks = rankdata(np.abs(nonzero), method="average")
    w_positive = float(ranks[nonzero > 0].sum())
    w_negative = float(ranks[nonzero < 0].sum())
    denominator = w_positive + w_negative
    rank_biserial = (w_positive - w_negative) / denominator if denominator else 0.0
    return float(test.statistic), float(test.pvalue), float(rank_biserial)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in original input order."""
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        [(count - rank) * p_values[index] for rank, index in enumerate(order)]
    )
    result = np.empty(count, dtype=float)
    for sorted_index, original_index in enumerate(order):
        result[original_index] = min(1.0, float(adjusted_sorted[sorted_index]))
    return result.tolist()


def _test_rows(scores: pd.DataFrame, cohort: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for metric in ("radgraph_f1", "chexpert_f1"):
        baseline = scores[f"baseline_{metric}"].to_numpy(dtype=float)
        gated = scores[f"gated_{metric}"].to_numpy(dtype=float)
        differences = gated - baseline
        statistic, p_value, effect = _paired_effect_and_test(differences)
        rows.append({
            "cohort": cohort,
            "endpoint": metric,
            "n": len(scores),
            "baseline_mean": float(np.mean(baseline)),
            "gated_mean": float(np.mean(gated)),
            "mean_delta_gated_minus_baseline": float(np.mean(differences)),
            "baseline_median": float(np.median(baseline)),
            "gated_median": float(np.median(gated)),
            "median_delta_gated_minus_baseline": float(np.median(differences)),
            "wilcoxon_statistic": statistic,
            "p_value_raw": p_value,
            "holm_family": cohort,
            "rank_biserial_correlation": effect,
            "alpha": ALPHA,
            "significant_after_holm": False,
            "test_method": "Wilcoxon signed-rank, two-sided, zero_method=wilcox, method=approx, no continuity correction",
        })
        q1, q3 = np.quantile(differences, [0.25, 0.75])
        diagnostics.append({
            "cohort": cohort,
            "endpoint": metric,
            "n": len(differences),
            "zero_differences": int(np.count_nonzero(differences == 0)),
            "positive_differences": int(np.count_nonzero(differences > 0)),
            "negative_differences": int(np.count_nonzero(differences < 0)),
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "difference_q1": float(q1),
            "difference_q3": float(q3),
            "difference_iqr": float(q3 - q1),
        })
    adjusted = _holm_adjust([row["p_value_raw"] for row in rows])
    for row, corrected in zip(rows, adjusted):
        row["p_value_holm"] = corrected
        row["significant_after_holm"] = bool(corrected < ALPHA)
    return rows, diagnostics


def _claim_statistics(claims: pd.DataFrame, expected_keys: set[tuple[str, str]]) -> pd.DataFrame:
    indicators = claims.copy()
    indicators = indicators.rename(columns={"benchmark_condition": "condition"})
    indicators["baseline_has_unsupported_claim"] = indicators["baseline_label"].isin(UNSUPPORTED_LABELS)
    indicators["gated_has_unsupported_claim"] = indicators["gated_label"].isin(UNSUPPORTED_LABELS)
    per_record = indicators.groupby(KEYS, as_index=False).agg(
        uid=("uid", "first"),
        baseline_has_unsupported_claim=("baseline_has_unsupported_claim", "any"),
        gated_has_unsupported_claim=("gated_has_unsupported_claim", "any"),
    )
    record_keys = set(zip(per_record["sample_id"].astype(str), per_record["condition"].astype(str)))
    if record_keys != expected_keys or len(per_record) != 576:
        raise ValueError("Phase 19 could not support unambiguous record-level paired binary outcomes")
    a = per_record["baseline_has_unsupported_claim"].astype(bool)
    b = per_record["gated_has_unsupported_claim"].astype(bool)
    baseline_positive = int(a.sum())
    gated_positive = int(b.sum())
    baseline_only = int((a & ~b).sum())
    gated_only = int((~a & b).sum())
    discordant = baseline_only + gated_only
    p_value = float(binomtest(min(baseline_only, gated_only), discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return pd.DataFrame([{
        "analysis": "record_level_unsupported_claim_presence",
        "n": len(per_record),
        "baseline_positive_records": baseline_positive,
        "gated_positive_records": gated_positive,
        "baseline_only_discordant_pairs": baseline_only,
        "gated_only_discordant_pairs": gated_only,
        "discordant_pairs": discordant,
        "test": "Exact McNemar via two-sided binomial test on discordant pairs, null probability 0.5",
        "p_value": p_value,
        "analysis_family": "secondary, reported separately; no Holm correction",
    }])


def _render_docs(primary: pd.DataFrame, covered: pd.DataFrame, claims: pd.DataFrame, diagnostics: pd.DataFrame, hashes: dict[str, str]) -> str:
    def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
        selected = frame[columns] if columns is not None else frame

        def cell(value: Any) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6g}"
            return str(value).replace("|", "\\|").replace("\n", " ")

        headers = [str(column) for column in selected.columns]
        rows = [[cell(value) for value in row] for row in selected.itertuples(index=False, name=None)]
        return "\n".join([
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ])

    def stat_table(frame: pd.DataFrame) -> str:
        cols = ["endpoint", "n", "baseline_mean", "gated_mean", "mean_delta_gated_minus_baseline", "baseline_median", "gated_median", "median_delta_gated_minus_baseline", "wilcoxon_statistic", "p_value_raw", "p_value_holm", "rank_biserial_correlation", "significant_after_holm"]
        return markdown_table(frame, cols)

    diag_text = markdown_table(diagnostics)
    claim_text = markdown_table(claims)
    hashes_text = "\n".join(f"- `{name}`: `{value}`" for name, value in hashes.items())
    return f"""# Phase 22: Statistical Analysis of Baseline vs Gated Results

## Objective and frozen inputs

This phase performs paired statistical analyses of the existing eligible Phase 17–20 records. It does not recompute reports or scores. Frozen benchmark, eligibility, baseline, gated, Phase 19, Phase 20, and Phase 21 artifacts were read as specified. SHA-256 values before analysis:\n\n{hashes_text}

## Pairing and integrity

The analysis includes 576 eligible records, 96 per benchmark condition. Baseline and gated scores were paired by `(sample_id, condition)` and UID equality was checked. Each paired score table had one row per eligible key; there were no excluded UIDs and no missing/non-finite RadGraph-F1 or CheXpert-F1 values. Phase 20 coverage flags agreed across both metrics; the covered-only subset contains 480 records.

## Primary all-record analysis

Endpoints are RadGraph-F1 and CheXpert-F1 for all 576 records. For each record, delta is gated minus baseline. Two-sided Wilcoxon signed-rank tests were used because the observations are paired and differences need not be normally distributed. Zeros were handled with SciPy `zero_method="wilcox"`; the normal approximation with tie correction (`method="approx"`) and no continuity correction was used. Holm correction was applied across the two primary endpoint p-values. Alpha is 0.05.

{stat_table(primary)}

## Secondary covered-only analysis

The same paired comparisons were repeated only for the 480 records with a covered gated report. The two covered-only p-values form a separate Holm family.

{stat_table(covered)}

## Paired difference diagnostics

Counts and distribution summaries are computed on gated-minus-baseline paired differences. No normality test was used to select the test.

{diag_text}

## Paired effect size

The paired rank-biserial correlation is reported for each Wilcoxon comparison. It is calculated from nonzero paired differences as `(W_positive - W_negative) / (W_positive + W_negative)`, where positive and negative rank sums use average ranks of absolute differences. The sign therefore follows gated-minus-baseline direction. This is a descriptive numerical effect size and is not given a qualitative magnitude label.

## Secondary record-level unsupported-claim comparison

Phase 19 supports one binary outcome per report: an indicator is positive if that record contains at least one claim labeled `ungrounded_affirmation`, `unsupported_affirmation`, or `contradiction`. Other labels do not count. Claim rows were aggregated within record before testing; individual claims were not treated as independent observations. The result uses exact two-sided McNemar testing, implemented as a binomial test on the discordant pairs under probability 0.5. It is reported separately and is not Holm-adjusted with the F1 endpoint families.

{claim_text}

## Limitations and interpretation

- The primary family is the two all-record F1 endpoints; the secondary family is the two covered-only endpoints. Holm correction is performed independently within each family.
- No tests were run separately by condition, analyzer state, or gate action. Phase 21 remains the descriptive subgroup analysis.
- A corrected p-value below 0.05 is the prespecified criterion for statistical significance. Statistical significance does not establish clinical significance.
- The covered-only comparison conditions on the gate's coverage decision and applies only to the 480 covered records.
- The all-record metric retains Phase 20's existing abstention scoring; scores were not recoded for this analysis.
- The analysis does not create intermediate thresholds, alter metrics, or exclude unfavorable records.

## Reproducibility

Run `.venv\\Scripts\\python.exe -m src.evaluation.run_phase22_statistics` from the repository root. Focused tests: `.venv\\Scripts\\python.exe -m pytest tests\\test_phase22_statistics.py -q`. The runner checks cohort integrity and snapshots the listed input files and baseline/gated output trees before and after writing Phase 22 outputs.
"""


def run_phase22() -> tuple[dict[str, pd.DataFrame], str, dict[str, str]]:
    frames, before_files, before_trees = _load_inputs()
    expected_keys = set(zip(frames["eligibility"]["sample_id"].astype(str), frames["eligibility"]["condition"].astype(str)))
    scores = frames["scores"].copy()
    full = scores
    covered = scores[scores["gated_report_covered"]].copy()
    if len(full) != 576 or len(covered) != 480:
        raise ValueError("Unexpected all-record or covered-only sample count")

    primary_rows, all_diagnostics = _test_rows(full, "all_records_primary")
    covered_rows, covered_diagnostics = _test_rows(covered, "covered_only_secondary")
    primary = pd.DataFrame(primary_rows)
    covered_table = pd.DataFrame(covered_rows)
    diagnostics = pd.DataFrame(all_diagnostics + covered_diagnostics)
    claim_table = _claim_statistics(frames["claims"], expected_keys)
    docs = _render_docs(primary, covered_table, claim_table, diagnostics, before_files)

    outputs = {
        "primary": primary,
        "covered": covered_table,
        "claim": claim_table,
        "diagnostics": diagnostics,
    }
    if before_files != _file_snapshot([Path(ROOT / Path(key)) for key in before_files]):
        raise RuntimeError("A Phase 22 input file changed while analysis was running")
    if before_trees["baseline"] != _tree_snapshot(BASELINE_PATH.parent):
        raise RuntimeError("Baseline output tree changed during Phase 22")
    if before_trees["gated"] != _tree_snapshot(GATED_PATH.parent):
        raise RuntimeError("Gated output tree changed during Phase 22")
    return outputs, docs, before_files


def main() -> None:
    outputs, docs, hashes = run_phase22()
    for key, path_key in (("primary", "primary"), ("covered", "covered"), ("claim", "claim"), ("diagnostics", "diagnostics")):
        OUTPUTS[path_key].parent.mkdir(parents=True, exist_ok=True)
        outputs[key].to_csv(OUTPUTS[path_key], index=False)
    OUTPUTS["docs"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS["docs"].write_text(docs, encoding="utf-8")
    print("All-record primary analysis:")
    print(outputs["primary"].to_string(index=False))
    print("\nCovered-only secondary analysis:")
    print(outputs["covered"].to_string(index=False))
    print("\nRecord-level claim McNemar analysis:")
    print(outputs["claim"].to_string(index=False))
    print(f"\nInput integrity verified for {len(hashes)} files and baseline/gated output trees.")
    print("No VLM inference or subgroup significance testing was performed.")


if __name__ == "__main__":
    main()
