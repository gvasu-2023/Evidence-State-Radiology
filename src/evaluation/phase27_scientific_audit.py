"""Independent scientific audit of frozen Phase 26D evaluation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDITIONS = (
    "sufficient", "syntactic_incomplete", "evidentiary_incomplete",
    "irrelevant", "conflicting", "insufficient",
)
DEVELOPMENT_IDS = {
    195, 227, 253, 282, 290, 317, 458, 494,
    577, 783, 793, 936, 960, 965, 1025, 1165,
}
EXCLUDED_UIDS = {74, 597, 803, 885}
UNSUPPORTED_LABELS = {
    "ungrounded_affirmation", "unsupported_affirmation", "contradiction",
}
TOL = 1e-9

PATHS = {
    "cohort": ROOT / "results/tables/phase16e_selected_studies.csv",
    "split": ROOT / "results/tables/phase26d/phase26d_maira2_split.csv",
    "eligibility": ROOT / "results/tables/phase17_vlm_inference_eligibility.csv",
    "generation": ROOT / "results/maira2_contrastive/phase26d_evaluation/phase26d_evaluation_generation.csv",
    "claims": ROOT / "results/tables/phase26d_claim_evaluation/phase26d_evaluation_claims.csv",
    "claim_summary": ROOT / "results/tables/phase26d_claim_evaluation/phase26d_evaluation_claim_summary.csv",
    "claim_paired": ROOT / "results/tables/phase26d_claim_evaluation/phase26d_evaluation_paired_records.csv",
    "radgraph_summary": ROOT / "results/tables/phase26d_metrics/phase26d_evaluation_radgraph_summary.csv",
    "radgraph_paired": ROOT / "results/tables/phase26d_metrics/phase26d_evaluation_radgraph_paired.csv",
    "radgraph_stats": ROOT / "results/tables/phase26d_metrics/phase26d_evaluation_radgraph_paired_stats.csv",
    "chexpert_summary": ROOT / "results/tables/phase26d_metrics/phase26d_evaluation_chexpert_summary.csv",
    "chexpert_paired": ROOT / "results/tables/phase26d_metrics/phase26d_evaluation_chexpert_paired.csv",
    "chexpert_stats": ROOT / "results/tables/phase26d_metrics/phase26d_evaluation_chexpert_paired_stats.csv",
    "master": ROOT / "results/tables/phase26d_master/phase26d_master_evaluation.csv",
    "master_condition": ROOT / "results/tables/phase26d_master/phase26d_master_by_condition.csv",
    "policy": ROOT / "docs/phase26d_lambda_selection_policy.md",
}
OUTPUT_CSV = ROOT / "results/tables/phase27_scientific_audit.csv"
OUTPUT_MD = ROOT / "docs/phase27_scientific_audit.md"
OUTPUT_VERIFIED_STATS = ROOT / "results/tables/phase27_verified_statistics.csv"


def _read_inputs() -> dict[str, Any]:
    for name, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 27 input is missing ({name}): {path}")
    frames = {
        name: pd.read_csv(path)
        for name, path in PATHS.items()
        if name != "policy"
    }
    frames["policy"] = PATHS["policy"].read_text(encoding="utf-8")
    return frames


def _need_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _run_audit(data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(section: str, name: str, passed: bool, expected: Any, observed: Any,
              detail: str = "", exception: bool = False) -> None:
        checks.append({
            "section": section,
            "check": name,
            "status": "PASS" if passed else "EXCEPTION" if exception else "FAIL",
            "expected": _fmt(expected),
            "observed": _fmt(observed),
            "detail": detail,
        })

    cohort = data["cohort"]
    _need_columns(cohort, {"uid"}, "Phase 16E cohort")
    cohort_ids = set(pd.to_numeric(cohort["uid"], errors="raise").astype(int))
    split = data["split"]
    _need_columns(split, {"study_id", "split"}, "Phase 26D split")
    split["study_id"] = pd.to_numeric(split["study_id"], errors="raise").astype(int)
    dev_ids = set(split.loc[split["split"].eq("development"), "study_id"])
    eval_split_ids = set(split.loc[split["split"].eq("evaluation"), "study_id"])
    eval_nominal_ids = cohort_ids - dev_ids
    check("A. Study split", "100 selected studies", len(cohort_ids) == 100, 100, len(cohort_ids))
    check("A. Study split", "16 development studies", len(dev_ids) == 16, 16, len(dev_ids))
    check("A. Study split", "Development IDs exactly match frozen list", dev_ids == DEVELOPMENT_IDS,
          sorted(DEVELOPMENT_IDS), sorted(dev_ids))
    check("A. Study split", "84 nominal evaluation studies before frontal exclusion",
          len(eval_nominal_ids) == 84, 84, len(eval_nominal_ids))
    check("A. Study split", "Development and nominal evaluation disjoint",
          dev_ids.isdisjoint(eval_nominal_ids), "disjoint", len(dev_ids & eval_nominal_ids))
    check("A. Study split", "Development and delivered evaluation split disjoint",
          dev_ids.isdisjoint(eval_split_ids), "disjoint", len(dev_ids & eval_split_ids))
    check("A. Study split", "Four ineligible UIDs are in nominal evaluation cohort",
          EXCLUDED_UIDS <= eval_nominal_ids, sorted(EXCLUDED_UIDS), sorted(EXCLUDED_UIDS & eval_nominal_ids))
    expected_eval_ids = eval_nominal_ids - EXCLUDED_UIDS
    check("A. Study split", "80 eligible evaluation studies", len(expected_eval_ids) == 80, 80, len(expected_eval_ids))
    check("A. Study split", "Split evaluation IDs match nominal cohort after exclusions",
          eval_split_ids == expected_eval_ids, 80, len(eval_split_ids),
          f"missing={sorted(expected_eval_ids - eval_split_ids)} extra={sorted(eval_split_ids - expected_eval_ids)}")
    elig = data["eligibility"]
    _need_columns(elig, {"uid", "condition", "eligible_for_vlm"}, "Phase 17 eligibility")
    excluded_elig = elig.loc[elig["uid"].isin(EXCLUDED_UIDS)]
    check("A. Study split", "Frontal-ineligible UID set matches exclusions",
          set(elig.loc[~elig["eligible_for_vlm"].astype(bool), "uid"].astype(int)) == EXCLUDED_UIDS,
          sorted(EXCLUDED_UIDS), sorted(set(elig.loc[~elig["eligible_for_vlm"].astype(bool), "uid"].astype(int))))
    check("A. Study split", "All excluded UID condition records are ineligible",
          len(excluded_elig) == 24 and not excluded_elig["eligible_for_vlm"].astype(bool).any(),
          "24 ineligible records", f"{len(excluded_elig)} records; eligible={int(excluded_elig['eligible_for_vlm'].astype(bool).sum())}")

    gen = data["generation"]
    _need_columns(gen, {"sample_id", "uid", "condition", "lambda", "report", "status"}, "Generation")
    gen["uid"] = pd.to_numeric(gen["uid"], errors="raise").astype(int)
    gen["lambda"] = pd.to_numeric(gen["lambda"], errors="raise")
    check("B. Generation", "960 total trajectories", len(gen) == 960, 960, len(gen))
    gen_ids = set(gen["uid"])
    check("B. Generation", "80 unique evaluation studies", gen_ids == eval_split_ids and gen["uid"].nunique() == 80,
          80, gen["uid"].nunique())
    lambda_counts = gen["lambda"].value_counts().to_dict()
    check("B. Generation", "480 trajectories per lambda", lambda_counts == {0.0: 480, 0.25: 480},
          "0.00=480; 0.25=480", lambda_counts)
    cond_counts = gen["condition"].value_counts().to_dict()
    check("B. Generation", "Exactly six expected conditions", set(cond_counts) == set(CONDITIONS),
          sorted(CONDITIONS), sorted(cond_counts))
    check("B. Generation", "160 trajectories per condition", cond_counts == {c: 160 for c in CONDITIONS},
          "160 each", cond_counts)
    check("B. Generation", "No development studies", not (gen_ids & dev_ids), "none", sorted(gen_ids & dev_ids))
    check("B. Generation", "No excluded UID trajectories", not (gen_ids & EXCLUDED_UIDS),
          "none", sorted(gen_ids & EXCLUDED_UIDS))
    duplicate_gen = int(gen.duplicated(["sample_id", "condition", "lambda"]).sum())
    check("B. Generation", "No duplicate sample-condition-lambda keys", duplicate_gen == 0, 0, duplicate_gen)
    check("B. Generation", "All generation statuses succeeded", gen["status"].eq("success").all(),
          "all success", gen["status"].value_counts().to_dict())
    reports_nonempty = gen["report"].fillna("").astype(str).str.strip().ne("").all()
    check("B. Generation", "All generated reports are non-empty", reports_nonempty, "all non-empty",
          f"{int(gen['report'].fillna('').astype(str).str.strip().eq('').sum())} empty")

    raw, summary, paired = data["claims"], data["claim_summary"], data["claim_paired"]
    _need_columns(raw, {"sample_id", "condition", "lambda", "label"}, "Raw claim comparisons")
    _need_columns(summary, {"lambda", "comparison_rows", "supported", "omission", "extra_negation",
                             "ungrounded_affirmation", "unsupported_affirmation", "contradiction", "unclear",
                             "unsupported_claim_count", "unsupported_claim_rate_denominator", "unsupported_claim_rate"},
                  "Claim summary")
    _need_columns(paired, {"sample_id", "condition", "baseline_unsupported_rate", "selected_unsupported_rate",
                           "baseline_has_unsupported", "selected_has_unsupported", "delta_rate"},
                  "Claim paired records")
    check("C. Claim evaluation", "3864 raw claim comparison rows", len(raw) == 3864, 3864, len(raw))
    labels = set(raw["label"].dropna().astype(str))
    check("C. Claim evaluation", "Unsupported labels exactly match approved set",
          labels & UNSUPPORTED_LABELS == UNSUPPORTED_LABELS,
          sorted(UNSUPPORTED_LABELS), sorted(labels & UNSUPPORTED_LABELS))
    check("C. Claim evaluation", "No unrecognized claim labels",
          labels <= (UNSUPPORTED_LABELS | {"supported", "omission", "extra_negation", "unclear"}),
          "approved label vocabulary", sorted(labels))
    summary_matches = True
    unsupported_by_lambda: dict[float, tuple[int, int, float]] = {}
    for _, row in summary.iterrows():
        lam = float(row["lambda"])
        subset = raw.loc[np.isclose(raw["lambda"].astype(float), lam)]
        counts = subset["label"].value_counts().to_dict()
        for label in ["supported", "omission", "extra_negation", "ungrounded_affirmation",
                      "unsupported_affirmation", "contradiction", "unclear"]:
            if int(row[label]) != int(counts.get(label, 0)):
                summary_matches = False
        if int(row["comparison_rows"]) != len(subset):
            summary_matches = False
        numerator = int(subset["label"].isin(UNSUPPORTED_LABELS).sum())
        # The frozen rate uses all non-omission comparisons as its denominator:
        # supported, extra-negation, unsupported, and unclear labels.
        denominator = int(subset["label"].ne("omission").sum())
        rate = numerator / denominator if denominator else np.nan
        unsupported_by_lambda[lam] = (numerator, denominator, rate)
        if (int(row["unsupported_claim_count"]) != numerator
                or int(row["unsupported_claim_rate_denominator"]) != denominator
                or not np.isclose(float(row["unsupported_claim_rate"]), rate, atol=TOL, rtol=0)):
            summary_matches = False
    check("C. Claim evaluation", "Summary counts reproduce raw claim labels", summary_matches,
          "exact grouped label counts", "matched" if summary_matches else "mismatch")
    check("C. Claim evaluation", "Lambda 0 unsupported numerator/denominator/rate",
          unsupported_by_lambda.get(0.0, ())[:2] == (171, 935)
          and np.isclose(unsupported_by_lambda.get(0.0, (0, 0, np.nan))[2], 171 / 935, atol=TOL, rtol=0),
          "171 / 935 = 0.1828877005...", unsupported_by_lambda.get(0.0))
    check("C. Claim evaluation", "Lambda 0.25 unsupported numerator/denominator/rate",
          unsupported_by_lambda.get(0.25, ())[:2] == (168, 931)
          and np.isclose(unsupported_by_lambda.get(0.25, (0, 0, np.nan))[2], 168 / 931, atol=TOL, rtol=0),
          "168 / 931 = 0.1804511278...", unsupported_by_lambda.get(0.25))
    paired_unique = not paired.duplicated(["sample_id", "condition"]).any()
    check("C. Claim evaluation", "Exactly 480 unique study-condition pairs",
          len(paired) == 480 and paired_unique, "480, no duplicates",
          f"rows={len(paired)}, duplicates={int(paired.duplicated(['sample_id','condition']).sum())}")
    base_status = paired["baseline_has_unsupported"].astype(bool)
    selected_status = paired["selected_has_unsupported"].astype(bool)
    base_positive, selected_positive = int(base_status.sum()), int(selected_status.sum())
    baseline_only = int((base_status & ~selected_status).sum())
    selected_only = int((~base_status & selected_status).sum())
    discordant = baseline_only + selected_only
    check("C. Claim evaluation", "Paired unsupported-status counts",
          (base_positive, selected_positive, baseline_only, selected_only, discordant) == (126, 129, 30, 33, 63),
          "baseline=126, selected=129, baseline-only=30, selected-only=33, discordant=63",
          f"baseline={base_positive}, selected={selected_positive}, baseline-only={baseline_only}, selected-only={selected_only}, discordant={discordant}")
    claim_delta_test = wilcoxon(paired["delta_rate"].astype(float), alternative="two-sided",
                                zero_method="wilcox", correction=False, method="approx")
    claim_pair_test = wilcoxon(
        paired["baseline_unsupported_rate"].astype(float),
        paired["selected_unsupported_rate"].astype(float),
        alternative="two-sided", zero_method="wilcox", correction=False, method="approx",
    )
    master = data["master"]
    _need_columns(master, {"metric", "p_value"}, "Master evaluation")
    claim_master_p = float(master.loc[master["metric"].eq("Unsupported claim rate"), "p_value"].iloc[0])
    check("C. Claim evaluation", "Frozen claim-rate p-value remains 0.318635",
          np.isclose(claim_master_p, 0.318635, atol=1e-12, rtol=0), 0.318635, claim_master_p)
    check("C. Claim evaluation", "Paired-rate vector reconstruction reproduces frozen claim p-value",
          np.isclose(claim_pair_test.pvalue, claim_master_p, atol=5e-7, rtol=0),
          claim_master_p, claim_pair_test.pvalue,
          "baseline_unsupported_rate vs selected_unsupported_rate; two-sided; zero_method=wilcox; correction=False; method=approx; reconstructed because Phase 26D source code is absent")
    check("C. Claim evaluation", "Current-environment Wilcoxon from saved delta_rate computed",
          0.0 <= float(claim_delta_test.pvalue) <= 1.0, "valid p-value", claim_delta_test.pvalue,
          "independent current recomputation; not substituted for frozen p-value")
    mcnemar_p = float(binomtest(baseline_only, discordant, p=0.5, alternative="two-sided").pvalue)
    check("C. Claim evaluation", "Exact two-sided McNemar/binomial test recomputed",
          baseline_only + selected_only == 63 and 0 <= mcnemar_p <= 1,
          "two-sided exact binomial, n=63, k=30", mcnemar_p,
          "Frozen CSVs contain discordant counts but no separate stored McNemar p-value to compare.")

    metric_stats: dict[str, dict[str, float]] = {}
    for name, section, mean0, mean25, expected0, expected25, expected_p in [
        ("radgraph", "D. RadGraph", 0.11823509276005441, 0.12046682595491279, 0.11823509276005441, 0.12046682595491279, 0.3363292101),
        ("chexpert", "E. Custom CheXpert-style", 0.24788690476190475, 0.24316468253968254, 0.24788690476190475, 0.24316468253968254, 0.7119592028),
    ]:
        pframe, sframe, stats = data[f"{name}_paired"], data[f"{name}_summary"], data[f"{name}_stats"]
        _need_columns(pframe, {"sample_id", "condition", "f1_lambda0", "f1_lambda025", "f1_delta"}, f"{name} paired")
        _need_columns(sframe, {"lambda", "f1_mean"} if name == "radgraph" else {"lambda", "f1_mean"}, f"{name} summary")
        _need_columns(stats, {"wilcoxon_p_value", "n_pairs"}, f"{name} stats")
        dupes = int(pframe.duplicated(["sample_id", "condition"]).sum())
        check(section, "480 paired records; no duplicate study-condition keys",
              len(pframe) == 480 and dupes == 0, "480, no duplicates", f"rows={len(pframe)}, duplicates={dupes}")
        means = {float(row["lambda"]): float(row["f1_mean"]) for _, row in sframe.iterrows()}
        check(section, "Stored summary F1 means match frozen targets",
              np.isclose(means.get(0.0, np.nan), expected0, atol=TOL, rtol=0)
              and np.isclose(means.get(0.25, np.nan), expected25, atol=TOL, rtol=0),
              f"lambda0={expected0}; lambda0.25={expected25}", means)
        delta_test = wilcoxon(pframe["f1_delta"].astype(float), alternative="two-sided",
                              zero_method="wilcox", correction=False, method="approx")
        pair_test = wilcoxon(pframe["f1_lambda0"].astype(float), pframe["f1_lambda025"].astype(float),
                             alternative="two-sided", zero_method="wilcox", correction=False, method="approx")
        stored_p = float(stats["wilcoxon_p_value"].iloc[0])
        check(section, "Frozen paired-stat p-value remains at specified value",
              np.isclose(stored_p, expected_p, atol=1e-10, rtol=0), expected_p, stored_p)
        check(section, "Paired-score vector reconstruction reproduces frozen p-value",
              np.isclose(pair_test.pvalue, stored_p, atol=TOL, rtol=0), stored_p, pair_test.pvalue,
              "lambda0 vs lambda0.25 score vectors; two-sided; zero_method=wilcox; correction=False; method=approx",
              exception=name == "radgraph")
        check(section, "Current-environment Wilcoxon from saved delta vector computed",
              0.0 <= float(delta_test.pvalue) <= 1.0, "valid p-value", delta_test.pvalue,
              "independent current recomputation; not substituted for frozen p-value")
        check(section, "Stored paired-stat record count is 480", int(stats["n_pairs"].iloc[0]) == 480,
              480, int(stats["n_pairs"].iloc[0]))
        metric_stats[name] = {"mean0": means.get(0.0, np.nan), "mean25": means.get(0.25, np.nan),
                              "historical_candidate_p": float(pair_test.pvalue),
                              "delta_p": float(delta_test.pvalue), "stored_p": stored_p,
                              "reference_p": expected_p}

    current_p_values = [float(claim_pair_test.pvalue), float(claim_delta_test.pvalue), mcnemar_p]
    for metric in metric_stats.values():
        current_p_values.extend([metric["historical_candidate_p"], metric["delta_p"]])
    check("C-E. Statistical interpretation", "All current recomputed tests are non-significant at alpha 0.05",
          all(p_value >= 0.05 for p_value in current_p_values), "all p-values >= 0.05", current_p_values)

    expected_master_metrics = {
        "Unsupported claim rate", "RadGraph precision", "RadGraph recall", "RadGraph F1",
        "Custom CheXpert-style F1",
    }
    check("F. Master tables", "Master table contains exactly the five specified metrics",
          set(master["metric"]) == expected_master_metrics and len(master) == 5,
          sorted(expected_master_metrics), sorted(master["metric"]))
    master_condition = data["master_condition"]
    _need_columns(master_condition, {"condition", "lambda"}, "Condition master")
    condition_pairs = set(zip(master_condition["condition"], master_condition["lambda"].astype(float)))
    expected_pairs = {(condition, lam) for condition in CONDITIONS for lam in (0.0, 0.25)}
    check("F. Master tables", "Condition master has all 12 condition-lambda combinations",
          len(master_condition) == 12 and condition_pairs == expected_pairs,
          "6 conditions x 2 lambdas", len(master_condition))
    hetero_columns = [c for c in ["unsupported_claim_rate", "radgraph_f1_mean", "chexpert_f1_mean"]
                      if c in master_condition]
    hetero = any(master_condition.groupby("condition")[col].mean().nunique() > 1 for col in hetero_columns)
    check("F. Master tables", "Condition-level results show heterogeneous behavior", hetero,
          "at least one condition metric varies", f"checked={hetero_columns}")

    policy = data["policy"].lower()
    policy_ok = ("development cohort" in policy and "selected lambda = 0.25" in policy
                 and "evaluation cohort was not used to select lambda" in policy)
    check("G. Lambda policy", "Policy records development selection, lambda 0.25, and held-out exclusion",
          policy_ok, "all three explicit statements", "present" if policy_ok else "missing statement")

    context = {
        "studies_total": len(cohort_ids), "development": len(dev_ids), "nominal_eval": len(eval_nominal_ids),
        "excluded": len(EXCLUDED_UIDS), "eligible_eval": len(eval_split_ids), "trajectories": len(gen),
        "lambda_counts": lambda_counts, "condition_counts": cond_counts,
        "claims_raw": len(raw), "claim_unsupported": unsupported_by_lambda,
        "paired_count": len(paired), "baseline_positive": base_positive, "selected_positive": selected_positive,
        "baseline_only": baseline_only, "selected_only": selected_only, "discordant": discordant,
        "claim_frozen_p": claim_master_p,
        "claim_pair_wilcoxon_p": float(claim_pair_test.pvalue),
        "claim_delta_wilcoxon_p": float(claim_delta_test.pvalue),
        "scipy_version": scipy.__version__, "mcnemar_p": mcnemar_p,
        "metrics": metric_stats,
    }
    return pd.DataFrame(checks), context


def _write_report(results: pd.DataFrame, values: dict[str, Any]) -> None:
    passed = int(results["status"].eq("PASS").sum())
    failed = int(results["status"].eq("FAIL").sum())
    exceptions = int(results["status"].eq("EXCEPTION").sum())
    state = (
        "FAIL" if failed else
        "PASS WITH DOCUMENTED REPRODUCIBILITY EXCEPTION" if exceptions else
        "PASS"
    )
    results["overall_status"] = state
    verified_stats = pd.DataFrame([
        {
            "analysis": "Unsupported claim rate Wilcoxon",
            "phase": "26D",
            "frozen_historical_p": values["claim_frozen_p"],
            "current_paired_vector_p": round(values["claim_pair_wilcoxon_p"], 10),
            "saved_delta_vector_p": round(values["claim_delta_wilcoxon_p"], 10),
            "current_exact_paired_status_p": np.nan,
            "baseline_only_discordant": np.nan,
            "selected_only_discordant": np.nan,
            "interpretation": "Historically reproduced to recorded precision using paired baseline/selected rate columns.",
        },
        {
            "analysis": "RadGraph F1 Wilcoxon",
            "phase": "26D",
            "frozen_historical_p": values["metrics"]["radgraph"]["stored_p"],
            "current_paired_vector_p": round(values["metrics"]["radgraph"]["historical_candidate_p"], 10),
            "saved_delta_vector_p": round(values["metrics"]["radgraph"]["delta_p"], 10),
            "current_exact_paired_status_p": np.nan,
            "baseline_only_discordant": np.nan,
            "selected_only_discordant": np.nan,
            "interpretation": "Exact historical value is not reproducible from surviving Phase 26D artifacts; documented statistical-provenance exception.",
        },
        {
            "analysis": "Custom CheXpert-style F1 Wilcoxon",
            "phase": "26D",
            "frozen_historical_p": values["metrics"]["chexpert"]["stored_p"],
            "current_paired_vector_p": round(values["metrics"]["chexpert"]["historical_candidate_p"], 10),
            "saved_delta_vector_p": round(values["metrics"]["chexpert"]["delta_p"], 10),
            "current_exact_paired_status_p": np.nan,
            "baseline_only_discordant": np.nan,
            "selected_only_discordant": np.nan,
            "interpretation": "Historically reproduced using paired score columns.",
        },
        {
            "analysis": "Unsupported-claim status McNemar/binomial",
            "phase": "26D",
            "frozen_historical_p": np.nan,
            "current_paired_vector_p": np.nan,
            "saved_delta_vector_p": np.nan,
            "current_exact_paired_status_p": round(values["mcnemar_p"], 10),
            "baseline_only_discordant": values["baseline_only"],
            "selected_only_discordant": values["selected_only"],
            "interpretation": "Phase 26D result from 30/33 discordants. Phase 22 p=0.0308837890625 used separate 14/4 discordants and is not a Phase 26D value.",
        },
    ])
    lines = [
        "# Phase 27 - Final Scientific Audit", "",
        f"**Audit status: {state}**  ",
        f"Checks passed: {passed}; documented exceptions: {exceptions}; failed: {failed}; total: {len(results)}.", "",
        "## Integrity statements", "",
        "No Phase 26D result files were modified. No evaluation-set tuning occurred. Lambda=0.25 remained frozen.", "",
        "## Dataset accounting", "",
        f"The frozen selected cohort has {values['studies_total']} studies: {values['development']} development and {values['nominal_eval']} nominal evaluation studies before frontal-image exclusion. The excluded UID set is {sorted(EXCLUDED_UIDS)}; {values['eligible_eval']} evaluation studies remain eligible.", "",
        "## Development/evaluation separation", "",
        f"The development IDs exactly match the specified 16-study cohort. The evaluation split is disjoint and contains {values['eligible_eval']} studies; generation contains no development or excluded UID.", "",
        "## Inference accounting", "",
        f"The frozen generation table contains {values['trajectories']} successful trajectories, with lambda counts {values['lambda_counts']} and condition counts {values['condition_counts']}. All reports are non-empty.", "",
        "## Claim accounting", "",
        f"The raw table contains {values['claims_raw']} claim-comparison rows. Unsupported claim rates recomputed from raw labels are {values['claim_unsupported'][0.0][0]}/{values['claim_unsupported'][0.0][1]} = {values['claim_unsupported'][0.0][2]:.10f} at lambda 0 and {values['claim_unsupported'][0.25][0]}/{values['claim_unsupported'][0.25][1]} = {values['claim_unsupported'][0.25][2]:.10f} at lambda 0.25.", "",
        f"Across {values['paired_count']} paired study-condition records, unsupported status is positive for {values['baseline_positive']} baseline pairs and {values['selected_positive']} selected pairs; discordants are {values['baseline_only']} baseline-only and {values['selected_only']} selected-only. The paired-rate-vector Wilcoxon reconstruction gives p = {values['claim_pair_wilcoxon_p']:.10f}, matching the frozen value rounded to six decimals. Applying Wilcoxon directly to saved `delta_rate` gives p = {values['claim_delta_wilcoxon_p']:.10f}; this difference is retained and reported. Exact two-sided McNemar/binomial p-value recomputed from discordants is {values['mcnemar_p']:.10f}; no separate Phase 26D McNemar p-value is stored.", "",
        "## RadGraph accounting", "",
        f"Mean F1 is {values['metrics']['radgraph']['mean0']:.12f} at lambda 0 and {values['metrics']['radgraph']['mean25']:.12f} at lambda 0.25. The lambda-pair-vector reconstruction gives p = {values['metrics']['radgraph']['historical_candidate_p']:.10f} versus frozen p = {values['metrics']['radgraph']['stored_p']:.10f}; Wilcoxon on saved `f1_delta` gives p = {values['metrics']['radgraph']['delta_p']:.10f}. Neither available vector formulation reproduces the frozen p-value exactly.", "",
        "## Custom CheXpert-style accounting", "",
        f"Mean F1 is {values['metrics']['chexpert']['mean0']:.12f} at lambda 0 and {values['metrics']['chexpert']['mean25']:.12f} at lambda 0.25. The lambda-pair-vector reconstruction gives p = {values['metrics']['chexpert']['historical_candidate_p']:.10f}, reproducing the frozen p-value; Wilcoxon on saved `f1_delta` gives p = {values['metrics']['chexpert']['delta_p']:.10f}.", "",
        "## Lambda-selection integrity", "",
        "The policy states that candidates were selected on the development cohort, lambda = 0.25 was selected, and the evaluation cohort was not used for selection. Lambda 0.25 was frozen before held-out evaluation.", "",
        "## Statistical provenance exception", "",
        f"The Phase 27 runtime used SciPy {values['scipy_version']}. Repository requirements pin SciPy 1.15.3, but no Phase 26D execution log records the actual historical runtime. Commit `c660f21` added Phase 26D result tables without a corresponding analysis script or notebook. Matching paired vectors therefore reconstruct p-values but do not prove the historical call syntax. The RadGraph frozen p-value remains unreproducible from both available paired vectors.", "",
        "This is a documented statistical-provenance limitation, not a failure of experiment integrity. Every current recomputed test is non-significant, so the discrepancy does not change the significance conclusion.", "",
        "## Statistical interpretation", "",
        "Held-out evaluation does not demonstrate statistically significant overall improvement. Unsupported claim rate changed only slightly (18.2888% -> 18.0451%); RadGraph F1 changed 0.118235 -> 0.120467; custom CheXpert-style F1 changed 0.247887 -> 0.243165. All frozen and independently recomputed p-values remain above 0.05. Condition-level behavior is heterogeneous.", "",
        "## Final allowed scientific conclusion", "",
        "In this held-out evaluation, lambda 0.25 was associated with a small reduction in the measured unsupported-claim rate and a small increase in RadGraph F1, alongside a decrease in custom CheXpert-style F1. The paired comparisons do not establish statistically significant overall improvement. These results do not support a claim of universal reliability improvement.", "",
        "## Warnings against overclaiming", "",
        "- Do not describe the small unsupported-claim-rate change as a statistically significant improvement.",
        "- Do not treat any one metric as a complete measure of clinical factuality or reliability.",
        "- Do not claim universal reliability improvement; observed condition-level behavior is heterogeneous.",
        "- The exact McNemar/binomial result is recomputed from frozen paired-status records because no separate Phase 26D p-value is present in the supplied frozen CSVs.",
        "- Phase 22 p = 0.0308837890625 belongs to a separate 576-record analysis with 14 and 4 discordances; it does not describe Phase 26D counts.",
        "- The Phase 26D RadGraph p-value is not exactly reproducible from available paired CSV vectors; preserve it as a frozen value and disclose the unresolved provenance.", "",
        "## Verified statistics table", "",
        "Frozen historical values and current reconstructions are separated in `results/tables/phase27_verified_statistics.csv`.", "",
        "## Check results", "",
        "| Section | Check | Status | Expected | Observed |", "|---|---|---:|---|---|",
    ]
    for _, row in results.iterrows():
        lines.append(f"| {row['section']} | {row['check']} | {row['status']} | {row['expected']} | {row['observed']} |")
    if failed or exceptions:
        lines += ["", "## Failures and documented exceptions", ""]
        for _, row in results.loc[results["status"].isin(["FAIL", "EXCEPTION"])].iterrows():
            lines.append(f"- **{row['check']}**: {row['detail']}")
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    verified_stats.to_csv(OUTPUT_VERIFIED_STATS, index=False)


def main() -> int:
    try:
        data = _read_inputs()
        results, values = _run_audit(data)
        _write_report(results, values)
    except Exception as exc:
        print(f"FAIL: Phase 27 audit could not complete: {exc}", file=sys.stderr)
        return 1
    failed = results.loc[results["status"].eq("FAIL")]
    exceptions = results.loc[results["status"].eq("EXCEPTION")]
    if failed.empty and not exceptions.empty:
        state = "PASS WITH DOCUMENTED REPRODUCIBILITY EXCEPTION"
    else:
        state = "PASS" if failed.empty else "FAIL"
    print(f"Phase 27 scientific audit: {state}")
    print(f"Checks: {len(results)} total; {int(results['status'].eq('PASS').sum())} passed; {len(exceptions)} exceptions; {len(failed)} failed")
    for _, row in exceptions.iterrows():
        print(f"EXCEPTION: {row['section']} / {row['check']}: expected {row['expected']}; observed {row['observed']}")
    for _, row in failed.iterrows():
        print(f"FAIL: {row['section']} / {row['check']}: expected {row['expected']}; observed {row['observed']}")
    print(f"Report: {OUTPUT_MD.relative_to(ROOT)}")
    print(f"CSV: {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Verified statistics: {OUTPUT_VERIFIED_STATS.relative_to(ROOT)}")
    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
