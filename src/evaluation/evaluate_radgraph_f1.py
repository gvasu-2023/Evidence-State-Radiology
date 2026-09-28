"""
Phase 15 Evaluation: RadGraph-F1 Clinical Content / Entity-Relation Evaluation.

Evaluates entity and relation overlap between generated radiology reports
(baseline and Phase 13B gated reports) and reference reports (reference findings + impression).

Uses the official `radgraph` PyPI package (F1RadGraph / DyGIE model).

Outputs:
    results/tables/radgraph_f1_results.csv
    results/tables/radgraph_f1_summary.csv

Scientific Integrity Note:
    RadGraph-F1 measures clinical entity and relation overlap against reference text.
    It is NOT a direct measure of hallucination rate, factuality accuracy, or safety.
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Monkeypatch transformers for radgraph compatibility with transformers >= 5.0
import transformers

if not hasattr(transformers.PreTrainedTokenizerBase, "encode_plus"):
    transformers.PreTrainedTokenizerBase.encode_plus = transformers.PreTrainedTokenizerBase.__call__

if not hasattr(transformers.PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
    transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

from radgraph import F1RadGraph

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_PATH = ROOT / "results/tables/baseline_generation_results.csv"
GATED_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"

RESULTS_OUTPUT_PATH = ROOT / "results/tables/radgraph_f1_results.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/radgraph_f1_summary.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def load_input_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [PERTURBATIONS_PATH, BASELINE_PATH, GATED_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")

    pert_df = pd.read_csv(PERTURBATIONS_PATH)
    base_df = pd.read_csv(BASELINE_PATH)
    gated_df = pd.read_csv(GATED_PATH)

    if len(pert_df) != 81 or len(base_df) != 81 or len(gated_df) != 81:
        raise ValueError("Expected exactly 81 records in input files.")

    return pert_df, base_df, gated_df


def compute_radgraph_scores(
    refs: List[str],
    hyps: List[str],
    model_type: str = "radgraph",
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute RadGraph precision, recall, and F1 for a list of reference and hypothesis pairs.
    """
    f1radgraph = F1RadGraph(reward_level="all", model_type=model_type)
    _, reward_list, _, _ = f1radgraph(refs=refs, hyps=hyps)

    precisions = [float(p) for p in reward_list[0]]
    recalls = [float(r) for r in reward_list[1]]
    f1_scores = [float(f) for f in reward_list[2]]

    return precisions, recalls, f1_scores


def evaluate_radgraph() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pert_df, base_df, gated_df = load_input_data()

    # Build reference reports (findings + impression)
    reference_reports = []
    sample_ids = []
    conditions = []
    baseline_reports = []
    gated_reports = []

    # Map baseline and gated by (sample_id, condition)
    base_lookup = {
        (str(r["sample_id"]), str(r["condition"])): str(r["generated_report"])
        for _, r in base_df.iterrows()
    }
    gated_lookup = {
        (str(r["sample_id"]), str(r["condition"])): str(r["gated_report"])
        for _, r in gated_df.iterrows()
    }

    for _, row in pert_df.iterrows():
        sid = str(row["sample_id"])
        cond = str(row["condition"])
        rf = str(row["reference_findings"]) if pd.notna(row["reference_findings"]) else ""
        ri = str(row["reference_impression"]) if pd.notna(row["reference_impression"]) else ""
        ref_text = f"{rf} {ri}".strip()

        sample_ids.append(sid)
        conditions.append(cond)
        reference_reports.append(ref_text)
        baseline_reports.append(base_lookup[(sid, cond)])
        gated_reports.append(gated_lookup[(sid, cond)])

    # Compute baseline RadGraph scores
    b_prec, b_rec, b_f1 = compute_radgraph_scores(reference_reports, baseline_reports)

    # Compute gated RadGraph scores
    g_prec, g_rec, g_f1 = compute_radgraph_scores(reference_reports, gated_reports)

    results_rows = []
    for i in range(len(sample_ids)):
        delta_f1 = g_f1[i] - b_f1[i]
        results_rows.append(
            {
                "sample_id": sample_ids[i],
                "condition": conditions[i],
                "baseline_radgraph_precision": round(b_prec[i], 4),
                "baseline_radgraph_recall": round(b_rec[i], 4),
                "baseline_radgraph_f1": round(b_f1[i], 4),
                "gated_radgraph_precision": round(g_prec[i], 4),
                "gated_radgraph_recall": round(g_rec[i], 4),
                "gated_radgraph_f1": round(g_f1[i], 4),
                "delta_radgraph_f1": round(delta_f1, 4),
            }
        )

    results_df = pd.DataFrame(results_rows)

    # Summary table
    summary_rows = []
    subsets = [("overall", results_df)] + [
        (c, results_df[results_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        n = len(subset)
        summary_rows.append(
            {
                "condition": scope,
                "n_records": n,
                "mean_baseline_radgraph_precision": round(float(subset["baseline_radgraph_precision"].mean()), 4),
                "mean_baseline_radgraph_recall": round(float(subset["baseline_radgraph_recall"].mean()), 4),
                "mean_baseline_radgraph_f1": round(float(subset["baseline_radgraph_f1"].mean()), 4),
                "mean_gated_radgraph_precision": round(float(subset["gated_radgraph_precision"].mean()), 4),
                "mean_gated_radgraph_recall": round(float(subset["gated_radgraph_recall"].mean()), 4),
                "mean_gated_radgraph_f1": round(float(subset["gated_radgraph_f1"].mean()), 4),
                "mean_delta_radgraph_f1": round(float(subset["delta_radgraph_f1"].mean()), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return results_df, summary_df


def main() -> None:
    print("=" * 70)
    print("RADGRAPH-F1 EVALUATION")
    print("=" * 70)

    results_df, summary_df = evaluate_radgraph()

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(f"Results saved to: {RESULTS_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
