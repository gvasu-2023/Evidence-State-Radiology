"""
Phase 15 Evaluation: Risk-Coverage & Selective Prediction Analysis.

Evaluates selective prediction operating points for baseline vs gated systems.

Coverage:
    coverage = accepted (non-abstained) evaluation cases / total evaluation cases

Risk:
    risk = unsupported claim rate among covered/generated outputs

Scientific Integrity Constraint:
    The current EvidenceStateAnalyzer and ReliabilityGate operate via deterministic rule-based
    categorical routing without a continuous confidence score.
    Arbitrary confidence scores MUST NOT be fabricated.
    Therefore, this script reports exact operating points (Coverage, Risk, Abstention Rate).
    AURC is omitted with explicit documentation that a continuous ranking signal is required.

Outputs:
    results/tables/risk_coverage_points.csv
    results/tables/risk_coverage_summary.csv
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UNSUPPORTED_CLAIMS_PATH = ROOT / "results/tables/unsupported_claim_rate.csv"
GATED_RESULTS_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"

POINTS_OUTPUT_PATH = ROOT / "results/tables/risk_coverage_points.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/risk_coverage_summary.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_input_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    for p in [UNSUPPORTED_CLAIMS_PATH, GATED_RESULTS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p}")

    unsupp_df = pd.read_csv(UNSUPPORTED_CLAIMS_PATH)
    gated_df = pd.read_csv(GATED_RESULTS_PATH)

    if len(unsupp_df) != 81 or len(gated_df) != 81:
        raise ValueError("Expected 81 records in input files.")

    return unsupp_df, gated_df


def evaluate_risk_coverage() -> Tuple[pd.DataFrame, pd.DataFrame]:
    unsupp_df, gated_df = load_input_data()

    # Map gated action by (sample_id, condition)
    action_lookup = {
        (str(r["sample_id"]), str(r["condition"])): str(r["gate_action"])
        for _, r in gated_df.iterrows()
    }

    points_rows = []

    for _, row in unsupp_df.iterrows():
        sid = str(row["sample_id"])
        cond = str(row["condition"])
        g_action = action_lookup[(sid, cond)]

        b_accepted = True
        g_accepted = (g_action != "abstain")

        b_risk = float(row["baseline_unsupported_claim_rate"])
        g_risk = float(row["gated_unsupported_claim_rate"])

        points_rows.append(
            {
                "sample_id": sid,
                "condition": cond,
                "baseline_accepted": b_accepted,
                "gated_accepted": g_accepted,
                "baseline_unsupported_claim_rate": round(b_risk, 4),
                "gated_unsupported_claim_rate": round(g_risk, 4),
                "gated_gate_action": g_action,
            }
        )

    points_df = pd.DataFrame(points_rows)

    # Compute summary operating points
    summary_rows = []

    subsets = [("overall", points_df)] + [
        (c, points_df[points_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        n_total = len(subset)

        # Baseline system operating point
        b_covered = int(subset["baseline_accepted"].sum())
        b_cov = b_covered / n_total if n_total > 0 else 0.0
        b_abst_rate = 1.0 - b_cov
        b_covered_subset = subset[subset["baseline_accepted"]]
        b_risk = float(b_covered_subset["baseline_unsupported_claim_rate"].mean()) if b_covered > 0 else 0.0

        summary_rows.append(
            {
                "system": "baseline",
                "condition": scope,
                "total_cases": n_total,
                "covered_cases": b_covered,
                "coverage": round(b_cov, 4),
                "abstention_rate": round(b_abst_rate, 4),
                "risk_unsupported_claim_rate_on_covered": round(b_risk, 4),
            }
        )

        # Gated system operating point
        g_covered = int(subset["gated_accepted"].sum())
        g_cov = g_covered / n_total if n_total > 0 else 0.0
        g_abst_rate = 1.0 - g_cov
        g_covered_subset = subset[subset["gated_accepted"]]
        g_risk = float(g_covered_subset["gated_unsupported_claim_rate"].mean()) if g_covered > 0 else 0.0

        summary_rows.append(
            {
                "system": "gated",
                "condition": scope,
                "total_cases": n_total,
                "covered_cases": g_covered,
                "coverage": round(g_cov, 4),
                "abstention_rate": round(g_abst_rate, 4),
                "risk_unsupported_claim_rate_on_covered": round(g_risk, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return points_df, summary_df


def main() -> None:
    print("=" * 70)
    print("RISK-COVERAGE & SELECTIVE PREDICTION EVALUATION")
    print("=" * 70)

    points_df, summary_df = evaluate_risk_coverage()

    POINTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    points_df.to_csv(POINTS_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(f"Points saved to: {POINTS_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
