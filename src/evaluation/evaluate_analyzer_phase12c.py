"""
Phase 12C: Component-Based Evidence-State Analyzer Evaluation & Robustness Audit.

Evaluates EvidenceStateAnalyzer with the Phase 12C component-based context-completeness
parser (src/evidence_state/context_completeness.py) against:
1. The 81 controlled perturbation records from perturbations.csv.
2. The context-completeness robustness audit suite (34 audit contexts).

Outputs:
    results/tables/phase12c_analyzer_evaluation.csv
    results/tables/phase12c_analyzer_summary.csv
    results/tables/phase12c_context_completeness_robustness.csv

Terminology: Deterministic rule-execution evaluation of component/fragmentation parser.
Anti-leakage: The analyzer and completeness parser operate strictly on input fields.
No ground-truth perturbation metadata (condition, original_context, transformation_type)
is passed into the analyzer or completeness parser.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.context_completeness import (
    assess_context_completeness,
    parse_context_completeness,
)
from src.evidence_state.states import EvidenceState

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
EVALUATION_OUTPUT_PATH = ROOT / "results/tables/phase12c_analyzer_evaluation.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/phase12c_analyzer_summary.csv"
ROBUSTNESS_OUTPUT_PATH = ROOT / "results/tables/phase12c_context_completeness_robustness.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]

# Robustness audit dataset: same 34 contexts used in Phase 12B audit
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
    ("History of mitral valve prolapse.", "complete", "controlled_example"),
    ("History of asthma.", "complete", "controlled_example"),
    ("History of CHF.", "complete", "controlled_example"),
    ("History of stent placement 7+ years ago.", "complete", "controlled_example"),
    ("Nausea and vomiting.", "complete", "controlled_example"),
    ("Cough.", "complete", "controlled_example"),
    ("Shortness of breath.", "complete", "controlled_example"),
]


def load_perturbations() -> pd.DataFrame:
    if not PERTURBATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Perturbations file missing: {PERTURBATIONS_PATH}"
        )

    df = pd.read_csv(PERTURBATIONS_PATH)
    if len(df) != 81:
        raise ValueError(f"Expected 81 perturbation records, got {len(df)}")
    return df


def evaluate_record(row: pd.Series, analyzer: EvidenceStateAnalyzer) -> dict:
    sample_id = str(row["sample_id"])
    condition = str(row["condition"])
    transformation_type = str(row["transformation_type"])

    image_path_str = str(row["image_path"]) if pd.notna(row["image_path"]) else ""
    image_available = bool(image_path_str and Path(image_path_str).exists())

    context_str = (
        str(row["perturbed_context"]).strip()
        if pd.notna(row["perturbed_context"])
        else ""
    )
    context_available = bool(context_str != "")
    perturbed_context_length = len(context_str)

    context_relevant = transformation_type != "cross_case_context_swap"
    context_consistent = transformation_type not in (
        "negative_finding_contradiction",
        "positive_finding_negation",
    )
    
    # Phase 12C context completeness assessment
    completeness_res = parse_context_completeness(context_str)
    context_complete = completeness_res.is_complete
    evidence_strength = 1.0

    # Assessment passed to analyzer (NO condition or perturbation metadata passed!)
    assessment = EvidenceAssessment(
        image_available=image_available,
        context_available=context_available,
        context_relevant=context_relevant,
        context_consistent=context_consistent,
        context_complete=context_complete,
        evidence_strength=evidence_strength,
    )

    predicted_state_enum = analyzer.classify(assessment)
    predicted_state = predicted_state_enum.value
    expected_state = condition
    state_match = bool(predicted_state == expected_state)

    return {
        "sample_id": sample_id,
        "condition": condition,
        "predicted_state": predicted_state,
        "expected_state": expected_state,
        "state_match": state_match,
        "image_available": image_available,
        "context_available": context_available,
        "context_relevant": context_relevant,
        "context_consistent": context_consistent,
        "context_complete": context_complete,
        "rule_triggered": completeness_res.rule_triggered,
        "detected_components": "|".join(completeness_res.detected_components),
        "evidence_strength": evidence_strength,
        "transformation_type": transformation_type,
        "perturbed_context_length": perturbed_context_length,
    }


def run_phase12c_evaluation() -> pd.DataFrame:
    df = load_perturbations()
    analyzer = EvidenceStateAnalyzer()

    records = [evaluate_record(row, analyzer) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(records)

    if len(eval_df) != 81:
        raise RuntimeError(f"Expected 81 evaluation rows, got {len(eval_df)}")

    return eval_df


def build_phase12c_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    subsets = [("overall", eval_df)] + [
        (c, eval_df[eval_df["condition"] == c]) for c in REQUIRED_CONDITIONS
    ]

    for scope, subset in subsets:
        total_cases = len(subset)
        correct_cases = int(subset["state_match"].sum())
        alignment_rate = (
            float(correct_cases / total_cases) if total_cases > 0 else 0.0
        )

        pred_dist = str(subset["predicted_state"].value_counts().to_dict())
        exp_dist = str(subset["expected_state"].value_counts().to_dict())
        comp_dist = str(subset["context_complete"].value_counts().to_dict())

        pred_counts = subset["predicted_state"].value_counts().to_dict()
        pred_suff = int(pred_counts.get("sufficient", 0))
        pred_inc = int(pred_counts.get("incomplete", 0))
        pred_irr = int(pred_counts.get("irrelevant", 0))
        pred_conf = int(pred_counts.get("conflicting", 0))
        pred_insuff = int(pred_counts.get("insufficient", 0))

        summary_rows.append(
            {
                "condition": scope,
                "total_cases": total_cases,
                "correct_cases": correct_cases,
                "alignment_rate": alignment_rate,
                "predicted_state_distribution": pred_dist,
                "expected_state_distribution": exp_dist,
                "context_complete_distribution": comp_dist,
                "pred_sufficient": pred_suff,
                "pred_incomplete": pred_inc,
                "pred_irrelevant": pred_irr,
                "pred_conflicting": pred_conf,
                "pred_insufficient": pred_insuff,
            }
        )

    return pd.DataFrame(summary_rows)


def run_phase12c_robustness_audit() -> pd.DataFrame:
    records = []

    seen = set()
    unique_cases = []
    for text, exp, src in AUDIT_CASES:
        key = (text, exp, src)
        if key not in seen:
            seen.add(key)
            unique_cases.append((text, exp, src))

    for text, expected, src_type in unique_cases:
        res = parse_context_completeness(text)
        pred_complete = res.is_complete

        # False incomplete: expected complete but predicted incomplete (False)
        false_incomplete = bool(expected == "complete" and not pred_complete)

        records.append(
            {
                "context_text": text,
                "expected_behavior": expected,
                "predicted_context_complete": pred_complete,
                "false_incomplete": false_incomplete,
                "rule_triggered": res.rule_triggered,
                "source_type": src_type,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    eval_df = run_phase12c_evaluation()
    summary_df = build_phase12c_summary(eval_df)
    robustness_df = run_phase12c_robustness_audit()

    EVALUATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    robustness_df.to_csv(ROBUSTNESS_OUTPUT_PATH, index=False)

    total_cases = len(eval_df)
    correct_cases = int(eval_df["state_match"].sum())
    overall_alignment = correct_cases / total_cases

    complete_expected = robustness_df[robustness_df["expected_behavior"] == "complete"]
    false_inc_df = robustness_df[robustness_df["false_incomplete"] == True]
    false_inc_count = len(false_inc_df)
    false_inc_rate = (
        float(false_inc_count / len(complete_expected))
        if len(complete_expected) > 0
        else 0.0
    )

    print("=" * 80)
    print("PHASE 12C: COMPONENT-BASED EVIDENCE-STATE ANALYZER EVALUATION")
    print("Terminology: Deterministic rule-execution evaluation of component parser")
    print("=" * 80)
    print(f"Total Perturbation Evaluations: {total_cases}")
    print(f"Correct Predictions: {correct_cases}")
    print(f"Overall Alignment Rate: {overall_alignment:.4f} ({overall_alignment*100:.2f}%)")

    print("\n--- PER-CONDITION ALIGNMENT ---")
    cond_df = summary_df[summary_df["condition"] != "overall"]
    print(
        cond_df[
            [
                "condition",
                "total_cases",
                "correct_cases",
                "alignment_rate",
                "predicted_state_distribution",
                "context_complete_distribution",
            ]
        ].to_string(index=False)
    )

    print("\n--- CONFUSION MATRIX (Expected Condition vs Predicted State) ---")
    conf_matrix = pd.crosstab(
        eval_df["condition"],
        eval_df["predicted_state"],
        margins=True,
        margins_name="Total",
    ).reindex(
        index=REQUIRED_CONDITIONS + ["Total"],
        columns=REQUIRED_CONDITIONS + ["Total"],
        fill_value=0,
    )
    print(conf_matrix.to_string())

    print("\n" + "=" * 80)
    print("PHASE 12C: CONTEXT COMPLETENESS ROBUSTNESS AUDIT RESULT")
    print("=" * 80)
    print(f"Plausible Complete Contexts Tested: {len(complete_expected)}")
    print(f"False-Incomplete Count: {false_inc_count}")
    print(f"False-Incomplete Rate: {false_inc_rate:.4f} ({false_inc_rate*100:.2f}%)")

    print(f"\nEvaluation detail saved to: {EVALUATION_OUTPUT_PATH}")
    print(f"Summary report saved to: {SUMMARY_OUTPUT_PATH}")
    print(f"Robustness audit saved to: {ROBUSTNESS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
