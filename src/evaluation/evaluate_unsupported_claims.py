"""
Phase 15 Evaluation: Unsupported Claim Rate within Evaluated Claim Ontology.

Operational Definition:
    unsupported clinical claims = claims categorized as:
        - ungrounded_affirmation
        - unsupported_affirmation
        - contradiction

    unsupported_claim_rate = unsupported_claims / all evaluated generated clinical claims

Separate categories preserved:
    - omission
    - extra_negation
    - unclear

Scientific Integrity Constraint:
    The existing claim extractor has limited ontology coverage.
    Therefore, this metric MUST NOT be described as "overall hallucination rate."
    It is strictly reported as:
    "unsupported claim rate within the evaluated claim ontology."

Outputs:
    results/tables/unsupported_claim_rate.csv
    results/tables/unsupported_claim_summary.csv
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.claim_extractor import extract_generated_claim_candidates
from src.evaluation.evaluate_reference_factuality import (
    build_factuality_table,
    VALID_LABELS,
)

BASELINE_RESULTS_PATH = ROOT / "results/tables/baseline_generation_results.csv"
GATED_RESULTS_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"
REFERENCE_CLAIMS_PATH = ROOT / "results/tables/reference_claim_candidates.csv"
PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"

RESULTS_OUTPUT_PATH = ROOT / "results/tables/unsupported_claim_rate.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/unsupported_claim_summary.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_input_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [BASELINE_RESULTS_PATH, GATED_RESULTS_PATH, REFERENCE_CLAIMS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p}")

    base_df = pd.read_csv(BASELINE_RESULTS_PATH)
    gated_df = pd.read_csv(GATED_RESULTS_PATH)
    ref_claims_df = pd.read_csv(REFERENCE_CLAIMS_PATH)

    if len(base_df) != 81 or len(gated_df) != 81:
        raise ValueError("Expected 81 records in baseline and gated results.")

    return base_df, gated_df, ref_claims_df


def compute_claim_factuality_table(generation_df: pd.DataFrame, ref_claims_df: pd.DataFrame, report_col: str) -> pd.DataFrame:
    """
    Extract generated claims and build case-level reference factuality table.
    """
    prep_df = generation_df.rename(columns={report_col: "generated_report"})[
        ["sample_id", "condition", "generated_report"]
    ]
    gen_claims = extract_generated_claim_candidates(prep_df)
    factuality_df = build_factuality_table(prep_df, ref_claims_df, gen_claims)
    return factuality_df


def compute_record_metrics(fact_df: pd.DataFrame, sample_id: str, condition: str) -> Dict[str, Any]:
    case_df = fact_df[
        (fact_df["sample_id"].astype(str) == sample_id)
        & (fact_df["condition"].astype(str) == condition)
    ]

    counts = case_df["label"].value_counts().to_dict()

    supported = int(counts.get("supported", 0))
    unsupported_aff = int(counts.get("unsupported_affirmation", 0))
    ungrounded_aff = int(counts.get("ungrounded_affirmation", 0))
    contradiction = int(counts.get("contradiction", 0))
    omission = int(counts.get("omission", 0))
    extra_negation = int(counts.get("extra_negation", 0))
    unclear = int(counts.get("unclear", 0))

    unsupported_claims = unsupported_aff + ungrounded_aff + contradiction

    # Total generated clinical claims evaluated
    total_generated_claims = supported + unsupported_aff + ungrounded_aff + contradiction + extra_negation + unclear

    # Zero-denominator safe rates
    unsupported_rate = (unsupported_claims / total_generated_claims) if total_generated_claims > 0 else 0.0
    supported_rate = (supported / total_generated_claims) if total_generated_claims > 0 else 0.0
    ungrounded_rate = (ungrounded_aff / total_generated_claims) if total_generated_claims > 0 else 0.0
    extra_negation_rate = (extra_negation / total_generated_claims) if total_generated_claims > 0 else 0.0

    # Omission rate relative to total comparison findings for case
    total_findings = len(case_df)
    omission_rate = (omission / total_findings) if total_findings > 0 else 0.0

    return {
        "sample_id": sample_id,
        "condition": condition,
        "total_generated_claims": total_generated_claims,
        "unsupported_claims": unsupported_claims,
        "supported_claims": supported,
        "ungrounded_affirmations": ungrounded_aff,
        "unsupported_affirmations": unsupported_aff,
        "contradictions": contradiction,
        "omissions": omission,
        "extra_negations": extra_negation,
        "unclear_claims": unclear,
        "unsupported_claim_rate": unsupported_rate,
        "supported_claim_rate": supported_rate,
        "omission_rate": omission_rate,
        "extra_negation_rate": extra_negation_rate,
        "ungrounded_affirmation_rate": ungrounded_rate,
    }


def evaluate_unsupported_claim_rates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_df, gated_df, ref_claims_df = load_input_datasets()

    base_fact_df = compute_claim_factuality_table(base_df, ref_claims_df, "generated_report")
    gated_fact_df = compute_claim_factuality_table(gated_df, ref_claims_df, "gated_report")

    results_rows = []

    for _, row in base_df.iterrows():
        sid = str(row["sample_id"])
        cond = str(row["condition"])

        b_met = compute_record_metrics(base_fact_df, sid, cond)
        g_met = compute_record_metrics(gated_fact_df, sid, cond)

        delta_rate = g_met["unsupported_claim_rate"] - b_met["unsupported_claim_rate"]

        results_rows.append(
            {
                "sample_id": sid,
                "condition": cond,
                "baseline_claim_count": b_met["total_generated_claims"],
                "gated_claim_count": g_met["total_generated_claims"],
                "baseline_unsupported_claims": b_met["unsupported_claims"],
                "gated_unsupported_claims": g_met["unsupported_claims"],
                "baseline_unsupported_claim_rate": round(b_met["unsupported_claim_rate"], 4),
                "gated_unsupported_claim_rate": round(g_met["unsupported_claim_rate"], 4),
                "delta_unsupported_claim_rate": round(delta_rate, 4),
            }
        )

    results_df = pd.DataFrame(results_rows)

    # Compute summary breakdown by condition
    summary_rows = []
    subsets = [("overall", base_fact_df, gated_fact_df)] + [
        (c, base_fact_df[base_fact_df["condition"] == c], gated_fact_df[gated_fact_df["condition"] == c])
        for c in REQUIRED_CONDITIONS
    ]

    for scope, b_sub, g_sub in subsets:
        n_records = 81 if scope == "overall" else len(b_sub["sample_id"].unique())

        def calc_aggregate_rates(sub_df: pd.DataFrame) -> Dict[str, float]:
            counts = sub_df["label"].value_counts().to_dict()
            supp = int(counts.get("supported", 0))
            unsupp_aff = int(counts.get("unsupported_affirmation", 0))
            unground_aff = int(counts.get("ungrounded_affirmation", 0))
            contra = int(counts.get("contradiction", 0))
            omiss = int(counts.get("omission", 0))
            extra_neg = int(counts.get("extra_negation", 0))

            unsupp_total = unsupp_aff + unground_aff + contra
            gen_total = supp + unsupp_aff + unground_aff + contra + extra_neg + int(counts.get("unclear", 0))
            tot_rows = len(sub_df)

            return {
                "unsupported_rate": (unsupp_total / gen_total) if gen_total > 0 else 0.0,
                "supported_rate": (supp / gen_total) if gen_total > 0 else 0.0,
                "omission_rate": (omiss / tot_rows) if tot_rows > 0 else 0.0,
                "extra_negation_rate": (extra_neg / gen_total) if gen_total > 0 else 0.0,
                "ungrounded_rate": (unground_aff / gen_total) if gen_total > 0 else 0.0,
            }

        b_rates = calc_aggregate_rates(b_sub)
        g_rates = calc_aggregate_rates(g_sub)
        delta_unsupp = g_rates["unsupported_rate"] - b_rates["unsupported_rate"]

        summary_rows.append(
            {
                "condition": scope,
                "n_records": n_records,
                "baseline_unsupported_claim_rate": round(b_rates["unsupported_rate"], 4),
                "gated_unsupported_claim_rate": round(g_rates["unsupported_rate"], 4),
                "delta_unsupported_claim_rate": round(delta_unsupp, 4),
                "baseline_supported_claim_rate": round(b_rates["supported_rate"], 4),
                "gated_supported_claim_rate": round(g_rates["supported_rate"], 4),
                "baseline_omission_rate": round(b_rates["omission_rate"], 4),
                "gated_omission_rate": round(g_rates["omission_rate"], 4),
                "baseline_extra_negation_rate": round(b_rates["extra_negation_rate"], 4),
                "gated_extra_negation_rate": round(g_rates["extra_negation_rate"], 4),
                "baseline_ungrounded_affirmation_rate": round(b_rates["ungrounded_rate"], 4),
                "gated_ungrounded_affirmation_rate": round(g_rates["ungrounded_rate"], 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return results_df, summary_df


def main() -> None:
    print("=" * 70)
    print("UNSUPPORTED CLAIM RATE EVALUATION (EVALUATED ONTOLOGY)")
    print("=" * 70)

    results_df, summary_df = evaluate_unsupported_claim_rates()

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(f"Results saved to: {RESULTS_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
