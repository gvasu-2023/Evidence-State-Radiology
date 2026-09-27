"""
Generalization and Robustness Audit of the Phase 12B Context Completeness Heuristic.

Audits assess_context_completeness() on a suite of dataset-observed and controlled
clinical contexts to quantify false-incomplete predictions and identify rule overfitting.

Outputs:
    results/tables/context_completeness_robustness_audit.csv

Terminology: Robustness audit of prototype heuristic; non-clinical evaluation.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_analyzer_phase12b import assess_context_completeness

OUTPUT_PATH = ROOT / "results/tables/context_completeness_robustness_audit.csv"

# Comprehensive audit dataset: combining dataset observations & plausible controlled examples
AUDIT_CASES = [
    # 1. Dataset Observations: Plausible Complete Contexts from IU X-Ray metadata
    ("Positive TB test", "complete", "dataset_observation"),
    ("Preop bariatric surgery.", "complete", "dataset_observation"),
    ("Evaluate for infection", "complete", "dataset_observation"),
    ("Preop lumbar surgery", "complete", "dataset_observation"),
    ("Chest pain.", "complete", "dataset_observation"),
    ("Dyspnea", "complete", "dataset_observation"),
    ("Chest and nasal congestion.", "complete", "dataset_observation"),
    ("Fatigue, weakness, anterior chest pain", "complete", "dataset_observation"),
    ("XXXX-year-old male, chest pain.", "complete", "dataset_observation"),
    ("XXXX-year-old female, chest pain", "complete", "dataset_observation"),
    ("Nausea, vomiting, preop for surgery", "complete", "dataset_observation"),
    ("XXXX, preop for abdominal aortic aneurysm repair", "complete", "dataset_observation"),
    ("XXXX-year-old with XXXX on XXXX. Dyspnea. History of mitral valve prolapse.", "complete", "dataset_observation"),
    ("Chest pain today. History of stent placement 7+ years ago.", "complete", "dataset_observation"),

    # 2. Dataset Observations: Actual 9 Incomplete Perturbation Cases
    ("nasal congestion.", "incomplete", "dataset_observation"),
    ("History of mitral valve prolapse.", "incomplete", "dataset_observation"),
    ("History of stent placement 7+ years ago.", "incomplete", "dataset_observation"),
    ("chest pain.", "incomplete", "dataset_observation"),
    ("anterior chest pain", "incomplete", "dataset_observation"),
    ("chest pain", "incomplete", "dataset_observation"),
    ("chest pain.", "incomplete", "dataset_observation"),
    ("pain", "incomplete", "dataset_observation"),
    ("preop for surgery", "incomplete", "dataset_observation"),

    # 3. Controlled Examples: Plausible Standalone Complete Clinical Contexts
    ("Chest pain.", "complete", "controlled_example"),
    ("Dyspnea.", "complete", "controlled_example"),
    ("Pain.", "complete", "controlled_example"),
    ("Preoperative evaluation.", "complete", "controlled_example"),
    ("History of mitral valve prolapse.", "complete", "controlled_example"),  # Test standalone history
    ("History of asthma.", "complete", "controlled_example"),                 # Test standalone history
    ("History of CHF.", "complete", "controlled_example"),                    # Test standalone history
    ("History of stent placement 7+ years ago.", "complete", "controlled_example"),  # Test standalone history
    ("Nausea and vomiting.", "complete", "controlled_example"),
    ("Cough.", "complete", "controlled_example"),
    ("Shortness of breath.", "complete", "controlled_example"),
]


def identify_triggered_rule(text: str) -> str:
    """Identify which heuristic rule in assess_context_completeness() triggered completeness=False."""
    if not text or pd.isna(text):
        return "empty_input"
    t = str(text).strip()
    if not t:
        return "empty_input"

    if t[0].islower():
        return "lowercase_initial"

    if t.startswith("History of") and not any(
        k in t for k in ["Chest", "Dyspnea", "Pain", "cough", "Shortness"]
    ):
        return "history_without_primary_indicator"

    if t.lower().startswith("preop for") or t.lower().startswith("preop surgery"):
        return "preop_prefix"

    return "none"


def run_robustness_audit() -> pd.DataFrame:
    records = []

    # De-duplicate audit cases while preserving ordering
    seen = set()
    unique_cases = []
    for text, exp, src in AUDIT_CASES:
        key = (text, exp, src)
        if key not in seen:
            seen.add(key)
            unique_cases.append((text, exp, src))

    for text, expected, src_type in unique_cases:
        pred_complete = assess_context_completeness(text)
        rule = identify_triggered_rule(text)

        # False incomplete: expected complete but predicted incomplete (False)
        false_incomplete = bool(expected == "complete" and not pred_complete)

        records.append(
            {
                "context_text": text,
                "expected_behavior": expected,
                "predicted_context_complete": pred_complete,
                "false_incomplete": false_incomplete,
                "rule_triggered": rule,
                "source_type": src_type,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    audit_df = run_robustness_audit()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    total_cases = len(audit_df)
    pred_complete = (audit_df["predicted_context_complete"] == True).sum()
    pred_incomplete = (audit_df["predicted_context_complete"] == False).sum()

    complete_expected = audit_df[audit_df["expected_behavior"] == "complete"]
    false_inc_df = audit_df[audit_df["false_incomplete"] == True]
    false_inc_count = len(false_inc_df)
    false_inc_rate = (
        float(false_inc_count / len(complete_expected))
        if len(complete_expected) > 0
        else 0.0
    )

    print("=" * 80)
    print("GENERALIZATION / ROBUSTNESS AUDIT OF CONTEXT COMPLETENESS HEURISTIC")
    print("=" * 80)
    print(f"Total Audit Contexts: {total_cases}")
    print(f"Classified Complete: {pred_complete}")
    print(f"Classified Incomplete: {pred_incomplete}")
    print(f"Plausible Complete Contexts Tested: {len(complete_expected)}")
    print(f"False-Incomplete Count: {false_inc_count}")
    print(f"False-Incomplete Rate: {false_inc_rate:.4f} ({false_inc_rate*100:.2f}%)")

    print("\n--- FALSE-INCOMPLETE CONTEXTS & RESPONSIBLE RULES ---")
    if false_inc_count == 0:
        print("None. All plausible complete contexts classified as complete.")
    else:
        print(
            false_inc_df[
                ["context_text", "expected_behavior", "rule_triggered", "source_type"]
            ].to_string(index=False)
        )

    print("\n--- RULE TRIGGERED BREAKDOWN (ALL AUDIT CASES) ---")
    print(audit_df["rule_triggered"].value_counts().to_string())

    print(f"\nAudit artifact saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
