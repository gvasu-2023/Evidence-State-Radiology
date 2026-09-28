"""
Phase 15 Master Evaluation Pipeline.

Executes all Phase 15 evaluation modules in sequence:
    1. RadGraph-F1 Clinical Content / Entity-Relation Evaluation
    2. CheXpert-F1 Clinical Finding Evaluation
    3. Unsupported Claim Rate (Evaluated Claim Ontology)
    4. Risk-Coverage & Selective Prediction Analysis
    5. Abstention Metrics (Precision, Recall, Rate & Coverage)

Generated Artifacts:
    - results/tables/radgraph_f1_results.csv
    - results/tables/radgraph_f1_summary.csv
    - results/tables/chexpert_f1_results.csv
    - results/tables/chexpert_f1_summary.csv
    - results/tables/unsupported_claim_rate.csv
    - results/tables/unsupported_claim_summary.csv
    - results/tables/risk_coverage_points.csv
    - results/tables/risk_coverage_summary.csv
    - results/tables/abstention_metrics.csv
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_radgraph_f1 import evaluate_radgraph, RESULTS_OUTPUT_PATH as RADGRAPH_RES_PATH, SUMMARY_OUTPUT_PATH as RADGRAPH_SUM_PATH
from src.evaluation.evaluate_chexpert_f1 import evaluate_chexpert, RESULTS_OUTPUT_PATH as CHEXPERT_RES_PATH, SUMMARY_OUTPUT_PATH as CHEXPERT_SUM_PATH
from src.evaluation.evaluate_unsupported_claims import evaluate_unsupported_claim_rates, RESULTS_OUTPUT_PATH as UNSUPP_RES_PATH, SUMMARY_OUTPUT_PATH as UNSUPP_SUM_PATH
from src.evaluation.evaluate_risk_coverage import evaluate_risk_coverage, POINTS_OUTPUT_PATH as RISK_PTS_PATH, SUMMARY_OUTPUT_PATH as RISK_SUM_PATH
from src.evaluation.evaluate_abstention_metrics import evaluate_abstention, OUTPUT_PATH as ABSTAIN_PATH


def main() -> None:
    print("=" * 80)
    print("PHASE 15 MASTER EVALUATION PIPELINE")
    print("=" * 80)

    # 1. RadGraph-F1
    print("\n--- 1. Evaluating RadGraph-F1 ---")
    rad_res_df, rad_sum_df = evaluate_radgraph()
    RADGRAPH_RES_PATH.parent.mkdir(parents=True, exist_ok=True)
    rad_res_df.to_csv(RADGRAPH_RES_PATH, index=False)
    rad_sum_df.to_csv(RADGRAPH_SUM_PATH, index=False)
    print(f"RadGraph-F1 Summary Saved: {RADGRAPH_SUM_PATH}")

    # 2. CheXpert-F1
    print("\n--- 2. Evaluating CheXpert-F1 ---")
    chex_res_df, chex_sum_df = evaluate_chexpert()
    CHEXPERT_RES_PATH.parent.mkdir(parents=True, exist_ok=True)
    chex_res_df.to_csv(CHEXPERT_RES_PATH, index=False)
    chex_sum_df.to_csv(CHEXPERT_SUM_PATH, index=False)
    print(f"CheXpert-F1 Summary Saved: {CHEXPERT_SUM_PATH}")

    # 3. Unsupported Claim Rate
    print("\n--- 3. Evaluating Unsupported Claim Rate ---")
    unsupp_res_df, unsupp_sum_df = evaluate_unsupported_claim_rates()
    UNSUPP_RES_PATH.parent.mkdir(parents=True, exist_ok=True)
    unsupp_res_df.to_csv(UNSUPP_RES_PATH, index=False)
    unsupp_sum_df.to_csv(UNSUPP_SUM_PATH, index=False)
    print(f"Unsupported Claim Summary Saved: {UNSUPP_SUM_PATH}")

    # 4. Risk-Coverage
    print("\n--- 4. Evaluating Risk-Coverage Operating Points ---")
    risk_pts_df, risk_sum_df = evaluate_risk_coverage()
    RISK_PTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    risk_pts_df.to_csv(RISK_PTS_PATH, index=False)
    risk_sum_df.to_csv(RISK_SUM_PATH, index=False)
    print(f"Risk-Coverage Summary Saved: {RISK_SUM_PATH}")

    # 5. Abstention Metrics
    print("\n--- 5. Evaluating Abstention Metrics ---")
    abstain_df = evaluate_abstention()
    ABSTAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    abstain_df.to_csv(ABSTAIN_PATH, index=False)
    print(f"Abstention Metrics Saved: {ABSTAIN_PATH}")

    print("\n" + "=" * 80)
    print("PHASE 15 EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
