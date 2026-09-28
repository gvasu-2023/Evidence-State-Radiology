"""
Phase 15 Evaluation: Abstention Precision, Recall, Abstention Rate & Coverage.

Operational Definitions for "Should Abstain" Ground-Truth:
    1. Primary Operational Reference (`ref_insufficient`):
       Ground-truth condition == `insufficient` (N=20 records).
       Missing clinical context where visual imaging evidence alone is inadequate for safe generation.

    2. Secondary Operational Reference (`ref_unsafe_insufficient_or_conflicting`):
       Ground-truth condition in {`insufficient`, `conflicting`} (N=35 records).
       Context is missing OR actively contradicts reference imaging findings.

Metrics Evaluated:
    - Abstention Precision = True Positive Abstentions / (True Positives + False Positives)
    - Abstention Recall = True Positive Abstentions / (True Positives + False Negatives)
    - Abstention F1 = 2 * Precision * Recall / (Precision + Recall)
    - Abstention Rate = Gated Abstentions / Total Evaluation Cases
    - Coverage = Accepted (Non-Abstained) Cases / Total Evaluation Cases

Outputs:
    results/tables/abstention_metrics.csv
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATED_RESULTS_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"
OUTPUT_PATH = ROOT / "results/tables/abstention_metrics.csv"


def load_gated_results() -> pd.DataFrame:
    if not GATED_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Required input file missing: {GATED_RESULTS_PATH}")
    gated_df = pd.read_csv(GATED_RESULTS_PATH)
    if len(gated_df) != 81:
        raise ValueError(f"Expected 81 gated evaluation rows, got {len(gated_df)}")
    return gated_df


def compute_abstention_metrics_for_ref(
    gated_df: pd.DataFrame,
    ref_name: str,
    target_conditions: List[str],
) -> Dict[str, Any]:
    n_total = len(gated_df)

    gated_df["is_ref_abstain"] = gated_df["condition"].isin(target_conditions)
    gated_df["is_gated_abstain"] = (gated_df["gate_action"] == "abstain")

    tp = int(((gated_df["is_gated_abstain"] == True) & (gated_df["is_ref_abstain"] == True)).sum())
    fp = int(((gated_df["is_gated_abstain"] == True) & (gated_df["is_ref_abstain"] == False)).sum())
    fn = int(((gated_df["is_gated_abstain"] == False) & (gated_df["is_ref_abstain"] == True)).sum())
    tn = int(((gated_df["is_gated_abstain"] == False) & (gated_df["is_ref_abstain"] == False)).sum())

    n_ref_abstain = tp + fn
    n_gated_abstain = tp + fp

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    abstention_rate = n_gated_abstain / n_total
    coverage = (n_total - n_gated_abstain) / n_total

    return {
        "operational_reference": ref_name,
        "n_total": n_total,
        "n_ref_abstain": n_ref_abstain,
        "n_gated_abstain": n_gated_abstain,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
        "abstention_rate": round(abstention_rate, 4),
        "coverage": round(coverage, 4),
    }


def evaluate_abstention() -> pd.DataFrame:
    gated_df = load_gated_results()

    # Ref 1: Insufficient only
    m1 = compute_abstention_metrics_for_ref(
        gated_df,
        ref_name="ref_insufficient",
        target_conditions=["insufficient"],
    )

    # Ref 2: Insufficient + Conflicting (Unsafe context)
    m2 = compute_abstention_metrics_for_ref(
        gated_df,
        ref_name="ref_unsafe_insufficient_or_conflicting",
        target_conditions=["insufficient", "conflicting"],
    )

    df = pd.DataFrame([m1, m2])
    return df


def main() -> None:
    print("=" * 70)
    print("ABSTENTION METRICS EVALUATION")
    print("=" * 70)

    metrics_df = evaluate_abstention()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Abstention metrics saved to: {OUTPUT_PATH}\n")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
