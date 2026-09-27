"""
Phase 11: Case-Level Paired Reference-Grounded Evaluation and Statistical Analysis.

Calculates:
- Case-level metrics (results/tables/phase11_case_level_metrics.csv)
- Paired baseline vs. gated comparisons (results/tables/phase11_paired_comparison.csv)
- Case-level paired statistical tests with scope stratification (results/tables/phase11_statistical_tests.csv)
- Gate behavior and analyzer alignment (results/tables/phase11_gate_behavior.csv)

No new model inference is run.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.claim_extractor import extract_generated_claim_candidates
from src.evaluation.evaluate_reference_factuality import build_factuality_table

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_PATH = ROOT / "results/tables/baseline_generation_results.csv"
GATED_PATH = ROOT / "results/tables/gated_generation_results.csv"
REFERENCE_PATH = ROOT / "results/tables/reference_claim_candidates.csv"

CASE_METRICS_PATH = ROOT / "results/tables/phase11_case_level_metrics.csv"
PAIRED_COMP_PATH = ROOT / "results/tables/phase11_paired_comparison.csv"
STAT_TESTS_PATH = ROOT / "results/tables/phase11_statistical_tests.csv"
GATE_BEHAVIOR_PATH = ROOT / "results/tables/phase11_gate_behavior.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]

# Map label names to explicit column names
FACTUALITY_LABEL_COLS = {
    "supported": "has_supported_claim",
    "omission": "has_omission",
    "extra_negation": "has_extra_negation",
    "ungrounded_affirmation": "has_ungrounded_affirmation",
    "unsupported_affirmation": "has_unsupported_affirmation",
    "contradiction": "has_contradiction",
    "unclear": "has_unclear_claim",
}


def load_input_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [PERTURBATIONS_PATH, BASELINE_PATH, GATED_PATH, REFERENCE_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input path missing: {p}")

    pert_df = pd.read_csv(PERTURBATIONS_PATH)
    base_df = pd.read_csv(BASELINE_PATH)
    gated_df = pd.read_csv(GATED_PATH)
    ref_df = pd.read_csv(REFERENCE_PATH)

    return pert_df, base_df, gated_df, ref_df


# ==============================================================================
# 11B: Case-Level Metrics Construction
# ==============================================================================

def compute_case_level_metrics(
    base_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    ref_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 162 case-level metrics rows (81 baseline + 81 gated)."""

    # Baseline claims & factuality
    base_prep = base_df[["sample_id", "condition", "generated_report"]].copy()
    base_claims = extract_generated_claim_candidates(base_prep)
    base_fact = build_factuality_table(base_prep, ref_df, base_claims)

    # Gated claims & factuality
    gated_prep = gated_df.rename(columns={"experimental_condition": "condition"})[
        ["sample_id", "condition", "generated_report"]
    ].copy()
    gated_claims = extract_generated_claim_candidates(gated_prep)
    gated_fact = build_factuality_table(gated_prep, ref_df, gated_claims)

    rows = []

    for system_name, gen_df, claims_df, fact_df in [
        ("baseline", base_df, base_claims, base_fact),
        ("gated", gated_df.rename(columns={"experimental_condition": "condition"}), gated_claims, gated_fact),
    ]:
        for _, r in gen_df.iterrows():
            sid = str(r["sample_id"])
            cond = str(r["condition"])
            report_text = str(r["generated_report"])

            report_length_chars = len(report_text)

            # Claims count for this case
            case_claims = claims_df[
                (claims_df["sample_id"].astype(str) == sid)
                & (claims_df["condition"].astype(str) == cond)
            ]
            claim_count = len(case_claims)
            claim_count_per_report = float(claim_count)

            # Factuality labels for this case
            case_fact = fact_df[
                (fact_df["sample_id"].astype(str) == sid)
                & (fact_df["condition"].astype(str) == cond)
            ]

            case_labels = set(case_fact["label"].tolist()) if not case_fact.empty else set()

            rows.append(
                {
                    "sample_id": sid,
                    "condition": cond,
                    "system": system_name,
                    "report_length_chars": report_length_chars,
                    "claim_count": claim_count,
                    "claim_count_per_report": claim_count_per_report,
                    "has_supported_claim": "supported" in case_labels,
                    "has_omission": "omission" in case_labels,
                    "has_extra_negation": "extra_negation" in case_labels,
                    "has_ungrounded_affirmation": "ungrounded_affirmation" in case_labels,
                    "has_unsupported_affirmation": "unsupported_affirmation" in case_labels,
                    "has_contradiction": "contradiction" in case_labels,
                    "has_unclear_claim": "unclear" in case_labels,
                }
            )

    return pd.DataFrame(rows)


# ==============================================================================
# 11C: Paired Comparison Construction
# ==============================================================================

def compute_paired_comparison(case_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pair baseline and gated observations for each (sample_id, condition)."""

    base_subset = case_metrics_df[case_metrics_df["system"] == "baseline"].set_index(
        ["sample_id", "condition"]
    )
    gated_subset = case_metrics_df[case_metrics_df["system"] == "gated"].set_index(
        ["sample_id", "condition"]
    )

    paired_rows = []

    for key, b_row in base_subset.iterrows():
        sid, cond = key
        g_row = gated_subset.loc[key]

        b_len = int(b_row["report_length_chars"])
        g_len = int(g_row["report_length_chars"])
        d_len = g_len - b_len

        b_cnt = int(b_row["claim_count"])
        g_cnt = int(g_row["claim_count"])
        d_cnt = g_cnt - b_cnt

        b_cpr = float(b_row["claim_count_per_report"])
        g_cpr = float(g_row["claim_count_per_report"])
        d_cpr = g_cpr - b_cpr

        row = {
            "sample_id": sid,
            "condition": cond,
            "baseline_report_length_chars": b_len,
            "gated_report_length_chars": g_len,
            "delta_report_length_chars": d_len,
            "baseline_claim_count": b_cnt,
            "gated_claim_count": g_cnt,
            "delta_claim_count": d_cnt,
            "baseline_claim_count_per_report": b_cpr,
            "gated_claim_count_per_report": g_cpr,
            "delta_claim_count_per_report": d_cpr,
        }

        for lbl_key, col_name in FACTUALITY_LABEL_COLS.items():
            b_flag = bool(b_row[col_name])
            g_flag = bool(g_row[col_name])

            if b_flag and g_flag:
                change = "both_yes"
            elif not b_flag and not g_flag:
                change = "both_no"
            elif b_flag and not g_flag:
                change = "baseline_only"
            else:
                change = "gated_only"

            row[f"baseline_{col_name}"] = b_flag
            row[f"gated_{col_name}"] = g_flag
            row[f"change_type_{lbl_key}"] = change

        paired_rows.append(row)

    paired_df = pd.DataFrame(paired_rows)
    return paired_df


# ==============================================================================
# 11D: Statistical Analysis Construction (with Stratified Scopes)
# ==============================================================================

def run_statistical_tests(paired_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Wilcoxon signed-rank and McNemar exact tests on case-level paired observations.

    Stratifies analysis across explicit scopes:
        - overall_all_conditions (N=81)
        - non_irrelevant_conditions (N=61: sufficient, incomplete, conflicting, insufficient)
        - irrelevant_context_discount (N=20: irrelevant condition)
        - per-condition scopes (sufficient N=17, incomplete N=9, irrelevant N=20, conflicting N=15, insufficient N=20)
    """

    test_results = []

    subsets = [
        ("overall_all_conditions", paired_df),
        ("non_irrelevant_conditions", paired_df[paired_df["condition"] != "irrelevant"]),
        ("irrelevant_context_discount", paired_df[paired_df["condition"] == "irrelevant"]),
    ] + [
        (c, paired_df[paired_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    # Continuous variables
    cont_vars = [
        ("report_length_chars", "baseline_report_length_chars", "gated_report_length_chars"),
        ("claim_count", "baseline_claim_count", "gated_claim_count"),
        ("claim_count_per_report", "baseline_claim_count_per_report", "gated_claim_count_per_report"),
    ]

    for scope_name, subset in subsets:
        n_pairs = len(subset)

        for var_label, b_col, g_col in cont_vars:
            b_vals = subset[b_col].to_numpy()
            g_vals = subset[g_col].to_numpy()
            diffs = g_vals - b_vals

            # Check for zero variation
            if np.all(diffs == 0) or len(np.unique(diffs)) <= 1:
                notes_msg = (
                    "zero variation across non-irrelevant conditions"
                    if scope_name == "non_irrelevant_conditions"
                    else "zero variation in paired differences"
                )
                test_results.append(
                    {
                        "scope": scope_name,
                        "variable": var_label,
                        "test_type": "wilcoxon",
                        "n_pairs": n_pairs,
                        "baseline_to_gated_discordance_b": np.nan,
                        "gated_to_baseline_discordance_c": np.nan,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "test_status": "not_applicable",
                        "notes": notes_msg,
                    }
                )
            else:
                try:
                    res = stats.wilcoxon(b_vals, g_vals)
                    test_results.append(
                        {
                            "scope": scope_name,
                            "variable": var_label,
                            "test_type": "wilcoxon",
                            "n_pairs": n_pairs,
                            "baseline_to_gated_discordance_b": np.nan,
                            "gated_to_baseline_discordance_c": np.nan,
                            "statistic": float(res.statistic),
                            "p_value": float(res.pvalue),
                            "test_status": "executed",
                            "notes": f"Wilcoxon signed-rank test (mean diff={diffs.mean():.4f})",
                        }
                    )
                except Exception as e:
                    test_results.append(
                        {
                            "scope": scope_name,
                            "variable": var_label,
                            "test_type": "wilcoxon",
                            "n_pairs": n_pairs,
                            "baseline_to_gated_discordance_b": np.nan,
                            "gated_to_baseline_discordance_c": np.nan,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "test_status": "not_applicable",
                            "notes": f"test error: {str(e)}",
                        }
                    )

        # Binary McNemar outcomes
        for lbl_key, col_name in FACTUALITY_LABEL_COLS.items():
            b_col = f"baseline_{col_name}"
            g_col = f"gated_{col_name}"

            b_flags = subset[b_col].to_numpy()
            g_flags = subset[g_col].to_numpy()

            # b: baseline 1, gated 0; c: baseline 0, gated 1
            b_disc = int(np.sum(b_flags & (~g_flags)))
            c_disc = int(np.sum((~b_flags) & g_flags))
            total_disc = b_disc + c_disc

            if total_disc == 0:
                notes_msg = (
                    "zero discordant pairs across non-irrelevant conditions (b=0, c=0)"
                    if scope_name == "non_irrelevant_conditions"
                    else "zero discordant pairs (b=0, c=0)"
                )
                test_results.append(
                    {
                        "scope": scope_name,
                        "variable": col_name,
                        "test_type": "mcnemar",
                        "n_pairs": n_pairs,
                        "baseline_to_gated_discordance_b": 0,
                        "gated_to_baseline_discordance_c": 0,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "test_status": "not_applicable",
                        "notes": notes_msg,
                    }
                )
            else:
                # Exact binomial test for McNemar paired binary outcomes
                res = stats.binomtest(b_disc, total_disc, p=0.5)
                stat_val = min(b_disc, c_disc)
                test_results.append(
                    {
                        "scope": scope_name,
                        "variable": col_name,
                        "test_type": "mcnemar",
                        "n_pairs": n_pairs,
                        "baseline_to_gated_discordance_b": b_disc,
                        "gated_to_baseline_discordance_c": c_disc,
                        "statistic": float(stat_val),
                        "p_value": float(res.pvalue),
                        "test_status": "executed",
                        "notes": f"Exact binomial McNemar test (b={b_disc}, c={c_disc})",
                    }
                )

    return pd.DataFrame(test_results)


# ==============================================================================
# 11E: Gate Behavior Construction
# ==============================================================================

def compute_gate_behavior(gated_df: pd.DataFrame) -> pd.DataFrame:
    """Compute gate action distribution, rates, and analyzer alignment per condition and overall."""

    rows = []

    subsets = [("overall", gated_df)] + [
        (c, gated_df[gated_df["experimental_condition"] == c])
        for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        case_count = len(subset)

        pred_dist = str(subset["predicted_evidence_state"].value_counts().to_dict())
        action_dist = str(subset["gate_action"].value_counts().to_dict())

        gen_cnt = int((subset["gate_action"] == "generate").sum())
        qual_cnt = int((subset["gate_action"] == "qualify").sum())
        qual_abs_cnt = int((subset["gate_action"] == "qualify_or_abstain").sum())
        disc_cnt = int((subset["gate_action"] == "discount_context").sum())
        abs_cnt = int((subset["gate_action"] == "abstain").sum())

        gen_rate = gen_cnt / case_count if case_count else 0.0
        qual_rate = (qual_cnt + qual_abs_cnt) / case_count if case_count else 0.0
        disc_rate = disc_cnt / case_count if case_count else 0.0
        abs_rate = abs_cnt / case_count if case_count else 0.0

        if scope == "overall":
            align_matches = (
                subset["experimental_condition"] == subset["predicted_evidence_state"]
            ).sum()
        else:
            align_matches = (subset["predicted_evidence_state"] == scope).sum()

        alignment_rate = align_matches / case_count if case_count else 0.0

        rows.append(
            {
                "condition": scope,
                "case_count": case_count,
                "predicted_state_distribution": pred_dist,
                "gate_action_distribution": action_dist,
                "generate_count": gen_cnt,
                "qualify_count": qual_cnt,
                "qualify_or_abstain_count": qual_abs_cnt,
                "discount_context_count": disc_cnt,
                "abstain_count": abs_cnt,
                "generation_rate": gen_rate,
                "qualification_rate": qual_rate,
                "context_discount_rate": disc_rate,
                "abstention_rate": abs_rate,
                "analyzer_alignment_rate": alignment_rate,
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# Main Orchestration
# ==============================================================================

def main() -> None:
    pert_df, base_df, gated_df, ref_df = load_input_datasets()

    # 11B: Case Level Metrics
    case_metrics_df = compute_case_level_metrics(base_df, gated_df, ref_df)
    CASE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    case_metrics_df.to_csv(CASE_METRICS_PATH, index=False)

    # 11C: Paired Comparison
    paired_df = compute_paired_comparison(case_metrics_df)
    PAIRED_COMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(PAIRED_COMP_PATH, index=False)

    # 11D: Statistical Tests
    stat_df = run_statistical_tests(paired_df)
    STAT_TESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stat_df.to_csv(STAT_TESTS_PATH, index=False)

    # 11E: Gate Behavior
    gate_df = compute_gate_behavior(gated_df)
    GATE_BEHAVIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    gate_df.to_csv(GATE_BEHAVIOR_PATH, index=False)

    print("=" * 80)
    print("PHASE 11: PAIRED EVALUATION & STATISTICAL ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"1. Case-Level Metrics saved: {CASE_METRICS_PATH} (rows: {len(case_metrics_df)})")
    print(f"2. Paired Comparison saved: {PAIRED_COMP_PATH} (rows: {len(paired_df)})")
    print(f"3. Statistical Tests saved: {STAT_TESTS_PATH} (rows: {len(stat_df)})")
    print(f"4. Gate Behavior saved: {GATE_BEHAVIOR_PATH} (rows: {len(gate_df)})")

    print("\n--- Gate Behavior & Analyzer Alignment ---")
    print(gate_df[["condition", "case_count", "generate_count", "discount_context_count", "qualification_rate", "analyzer_alignment_rate"]].to_string(index=False))

    print("\n--- Executed Statistical Tests Summary ---")
    executed_stats = stat_df[stat_df["test_status"] == "executed"]
    if executed_stats.empty:
        print("No statistical tests were executed (all tests not_applicable due to zero discordance / zero variation).")
    else:
        print(executed_stats[["scope", "variable", "test_type", "n_pairs", "statistic", "p_value", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
