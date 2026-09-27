"""
Run Phase 10 Gated Experiment across all 81 perturbation records.

Executes GatedGenerator over perturbations.csv using baseline results lookup.
Saves outputs to results/tables/gated_generation_results.csv.

Terminology: paired reference-grounded evaluation of the current prototype's
gated generation policy.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gating.gated_generator import GatedGenerator

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_RESULTS_PATH = ROOT / "results/tables/baseline_generation_results.csv"
OUTPUT_PATH = ROOT / "results/tables/gated_generation_results.csv"

REQUIRED_COLUMNS = [
    "sample_id",
    "experimental_condition",
    "predicted_evidence_state",
    "gate_action",
    "gate_reason",
    "output_policy",
    "source_baseline_condition",
    "generated_report",
    "was_model_inference_run",
    "baseline_pair_identifier",
]


def run_gated_experiment() -> pd.DataFrame:
    if not PERTURBATIONS_PATH.exists():
        raise FileNotFoundError(f"Perturbations CSV missing: {PERTURBATIONS_PATH}")
    if not BASELINE_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Baseline results CSV missing: {BASELINE_RESULTS_PATH}")

    perturbations_df = pd.read_csv(PERTURBATIONS_PATH)
    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)

    if len(perturbations_df) != 81:
        raise ValueError(f"Expected 81 perturbation records, got {len(perturbations_df)}")

    generator = GatedGenerator(baseline_df)

    records = [
        generator.process_record(row)
        for _, row in perturbations_df.iterrows()
    ]

    output_df = pd.DataFrame(records)

    # Validate output schema
    if list(output_df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Columns mismatch: {list(output_df.columns)}")

    if len(output_df) != 81:
        raise RuntimeError(f"Expected 81 gated output records, got {len(output_df)}")

    # Validate unique sample_id + experimental_condition pairs
    pairs = list(zip(output_df["sample_id"], output_df["experimental_condition"]))
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("Duplicate (sample_id, experimental_condition) pairs found in gated outputs")

    return output_df


def main() -> None:
    output_df = run_gated_experiment()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("PHASE 10: GATED EXPERIMENT GENERATION")
    print("Terminology: paired reference-grounded evaluation of current prototype's gated generation policy")
    print("=" * 80)
    print(f"Total gated output records: {len(output_df)}")
    print(f"Redundant CPU inference calls: {(output_df['was_model_inference_run'] == True).sum()} (All 81 outputs reused exact input-matched baseline results)")

    print("\nGated Output Records by Condition:")
    print(output_df["experimental_condition"].value_counts().to_string())

    print("\nGate Action Distribution by Condition:")
    action_counts = pd.crosstab(output_df["experimental_condition"], output_df["gate_action"])
    print(action_counts.to_string())

    print(f"\nSaved gated results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
