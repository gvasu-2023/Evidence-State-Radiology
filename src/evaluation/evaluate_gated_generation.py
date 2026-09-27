"""
Paired reference-grounded evaluation of the current prototype's gated generation policy.

Compares Phase 10 gated generation results (results/tables/gated_generation_results.csv)
against Phase 8 baseline results (results/tables/evidence_state_breakdown_analysis.csv)
and reference claim candidates.

Outputs:
    results/tables/gated_factuality_comparison.csv

Terminology: paired reference-grounded evaluation of the current prototype's
gated generation policy.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.claim_extractor import extract_generated_claim_candidates
from src.evaluation.evaluate_reference_factuality import (
    build_factuality_table,
    summarize_by_condition,
)

GATED_RESULTS_PATH = ROOT / "results/tables/gated_generation_results.csv"
BASELINE_BREAKDOWN_PATH = ROOT / "results/tables/evidence_state_breakdown_analysis.csv"
REFERENCE_PATH = ROOT / "results/tables/reference_claim_candidates.csv"
OUTPUT_PATH = ROOT / "results/tables/gated_factuality_comparison.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_input_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [GATED_RESULTS_PATH, BASELINE_BREAKDOWN_PATH, REFERENCE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required table missing: {path}")

    gated_df = pd.read_csv(GATED_RESULTS_PATH)
    baseline_breakdown_df = pd.read_csv(BASELINE_BREAKDOWN_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    return gated_df, baseline_breakdown_df, reference_df


def compute_gated_factuality_comparison(
    gated_df: pd.DataFrame,
    baseline_breakdown_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract claims from gated reports, evaluate reference factuality,
    and compute paired baseline-vs-gated descriptive comparisons.
    """
    # Prepare gated df for claim extraction
    gated_prep = gated_df.rename(columns={"experimental_condition": "condition"})[
        ["sample_id", "condition", "generated_report"]
    ].copy()

    # Extract claims from gated outputs
    gated_claims_df = extract_generated_claim_candidates(gated_prep)

    # Build reference factuality table for gated outputs
    gated_detail_df = build_factuality_table(
        gated_prep,
        reference_df,
        gated_claims_df,
    )

    gated_summary_df = summarize_by_condition(
        gated_detail_df,
        gated_prep,
    )

    # Map baseline breakdown by condition
    base_map = baseline_breakdown_df.set_index("condition").to_dict(orient="index")
    gated_sum_map = gated_summary_df.set_index("condition").to_dict(orient="index")

    rows = []

    for cond in REQUIRED_CONDITIONS:
        b_info = base_map[cond]
        g_info = gated_sum_map[cond]
        c_subset = gated_df[gated_df["experimental_condition"] == cond]

        case_count = int(b_info["case_count"])
        gated_action_counts = c_subset["gate_action"].value_counts().to_dict()

        abstain_count = int((c_subset["gate_action"] == "abstain").sum())
        qualify_count = int(
            (
                (c_subset["gate_action"] == "qualify")
                | (c_subset["gate_action"] == "qualify_or_abstain")
            ).sum()
        )
        discount_count = int(
            (c_subset["gate_action"] == "discount_context").sum()
        )

        abstention_rate = float(abstain_count / case_count) if case_count else 0.0
        qualification_rate = float(qualify_count / case_count) if case_count else 0.0
        context_discount_rate = float(discount_count / case_count) if case_count else 0.0

        comparison_row_count = int(g_info["comparison_row_count"])
        gated_claim_count = len(
            gated_claims_df[gated_claims_df["condition"] == cond]
        )
        gated_claim_density = float(gated_claim_count / case_count) if case_count else 0.0

        # Label counts
        sup_cnt = int(g_info.get("supported", 0))
        om_cnt = int(g_info.get("omission", 0))
        ext_cnt = int(g_info.get("extra_negation", 0))
        ung_cnt = int(g_info.get("ungrounded_affirmation", 0))
        unp_cnt = int(g_info.get("unsupported_affirmation", 0))
        con_cnt = int(g_info.get("contradiction", 0))
        unc_cnt = int(g_info.get("unclear", 0))

        # Proportions
        gated_sup_prop = sup_cnt / comparison_row_count if comparison_row_count else 0.0
        gated_om_prop = om_cnt / comparison_row_count if comparison_row_count else 0.0
        gated_ext_prop = ext_cnt / comparison_row_count if comparison_row_count else 0.0
        gated_ung_prop = ung_cnt / comparison_row_count if comparison_row_count else 0.0
        gated_unp_prop = unp_cnt / comparison_row_count if comparison_row_count else 0.0
        gated_con_prop = con_cnt / comparison_row_count if comparison_row_count else 0.0
        gated_unc_prop = unc_cnt / comparison_row_count if comparison_row_count else 0.0

        # Paired differences against baseline
        base_claim_count = int(b_info["claim_count"])
        base_claim_density = float(b_info["claim_density"])
        base_sup_prop = float(b_info["supported_proportion"])
        base_om_prop = float(b_info["omission_proportion"])
        base_ext_prop = float(b_info["extra_negation_proportion"])

        delta_claim_count = gated_claim_count - base_claim_count
        delta_claim_density = gated_claim_density - base_claim_density
        delta_supported_prop = gated_sup_prop - base_sup_prop
        delta_omission_prop = gated_om_prop - base_om_prop
        delta_extra_negation_prop = gated_ext_prop - base_ext_prop

        rows.append(
            {
                "condition": cond,
                "case_count": case_count,
                "gate_action_distribution": str(gated_action_counts),
                "abstention_rate": abstention_rate,
                "qualification_rate": qualification_rate,
                "context_discount_rate": context_discount_rate,
                "baseline_claim_count": base_claim_count,
                "gated_claim_count": gated_claim_count,
                "delta_claim_count": delta_claim_count,
                "baseline_claim_density": base_claim_density,
                "gated_claim_density": gated_claim_density,
                "delta_claim_density": delta_claim_density,
                "gated_supported_proportion": gated_sup_prop,
                "delta_supported_proportion": delta_supported_prop,
                "gated_omission_proportion": gated_om_prop,
                "delta_omission_proportion": delta_omission_prop,
                "gated_extra_negation_proportion": gated_ext_prop,
                "delta_extra_negation_proportion": delta_extra_negation_prop,
                "gated_ungrounded_aff_proportion": gated_ung_prop,
                "gated_unsupported_aff_proportion": gated_unp_prop,
                "gated_contradiction_proportion": gated_con_prop,
                "gated_unclear_proportion": gated_unc_prop,
            }
        )

    comparison_df = pd.DataFrame(rows)
    return comparison_df, gated_claims_df, gated_detail_df


def main() -> None:
    gated_df, baseline_breakdown_df, reference_df = load_input_tables()

    comparison_df, gated_claims_df, gated_detail_df = compute_gated_factuality_comparison(
        gated_df,
        baseline_breakdown_df,
        reference_df,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("PHASE 10: PAIRED REFERENCE-GROUNDED GATED EVALUATION")
    print("Terminology: paired reference-grounded evaluation of current prototype's gated policy")
    print("=" * 80)
    print(
        "Disclaimer: This evaluation measures descriptive baseline vs gated policy shifts.\n"
        "Comparison rows (N=365) are nested within 81 cases. No clinical efficacy claims are made."
    )
    print("=" * 80)

    display_cols = [
        "condition",
        "case_count",
        "gated_claim_count",
        "delta_claim_count",
        "gated_claim_density",
        "delta_claim_density",
        "gated_supported_proportion",
        "delta_supported_proportion",
        "gated_omission_proportion",
        "delta_omission_proportion",
        "gated_extra_negation_proportion",
        "delta_extra_negation_proportion",
    ]

    print(comparison_df[display_cols].to_string(index=False))
    print(f"\nSaved comparison summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
