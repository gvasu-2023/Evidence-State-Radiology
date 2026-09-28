"""
Phase 15 Evaluation: CheXpert-F1 Clinical Finding Evaluation.

Evaluates clinical finding overlap across 14 Stanford CheXpert observation categories
between generated radiology reports (baseline and Phase 13B gated reports) and reference reports.

14 CheXpert Categories:
    1. Enlarged Cardiomediastinum
    2. Cardiomegaly
    3. Lung Opacity
    4. Lung Lesion
    5. Edema
    6. Consolidation
    7. Pneumonia
    8. Atelectasis
    9. Pneumothorax
    10. Pleural Effusion
    11. Pleural Other
    12. Fracture
    13. Support Devices
    14. No Finding

Label-Handling Policy (Explicitly Documented):
    - Positive (1): Explicit positive mention of condition.
    - Negative (0): Explicit negation of condition.
    - Uncertain (-1): Equivocal/uncertain mention (retained in label vectors for auditability).
    - Unmentioned (nan): Condition not mentioned in report text.
    - Metric calculation: Positive targets are compared against candidate positive predictions.
      Uncertain labels (-1) are NOT silently treated as positive or negative.

Outputs:
    results/tables/chexpert_f1_results.csv
    results/tables/chexpert_f1_summary.csv
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERTURBATIONS_PATH = ROOT / "data/processed/iu_xray/perturbations/perturbations.csv"
BASELINE_PATH = ROOT / "results/tables/baseline_generation_results.csv"
GATED_PATH = ROOT / "results/tables/phase13b_gated_generation_results.csv"

RESULTS_OUTPUT_PATH = ROOT / "results/tables/chexpert_f1_results.csv"
SUMMARY_OUTPUT_PATH = ROOT / "results/tables/chexpert_f1_summary.csv"

REQUIRED_CONDITIONS = [
    "sufficient",
    "incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]

CHEXPERT_CATEGORIES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

# Patterns for CheXpert labeling
CATEGORY_PATTERNS = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac silhouette is enlarged", "borderline enlarged cardiac silhouette", "borderline cardiomegaly"],
    "Enlarged Cardiomediastinum": ["enlarged mediastinum", "mediastinal widening", "widened mediastinum", "enlarged cardiomediastinum"],
    "Lung Opacity": ["airspace opacity", "air space opacity", "lung opacity", "pulmonary opacity", "opacities", "opacity", "airspace disease", "air space disease", "infiltrate", "infiltrates", "density", "densities"],
    "Lung Lesion": ["nodule", "nodules", "mass", "masses", "lung lesion", "granuloma", "calcified granuloma", "granulomatous disease"],
    "Edema": ["pulmonary edema", "edema", "vascular congestion", "pulmonary congestion", "chf"],
    "Consolidation": ["consolidation", "focal consolidation", "consolidations"],
    "Pneumonia": ["pneumonia", "infectious process"],
    "Atelectasis": ["atelectasis", "collapse", "basilar atelectasis"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces"],
    "Pleural Effusion": ["pleural effusion", "pleural effusions", "effusion", "effusions"],
    "Pleural Other": ["pleural thickening", "pleural plaque", "pleural scarring"],
    "Fracture": ["fracture", "fractures", "rib fracture", "displaced fracture"],
    "Support Devices": ["tube", "catheter", "line", "pacemaker", "stent", "picc", "hardware", "pigtail"],
    "No Finding": ["no acute cardiopulmonary abnormality", "no acute cardiopulmonary process", "lungs are clear", "no findings", "unremarkable chest", "no acute intrathoracic process"],
}

NEGATION_CUES = ["no ", "no evidence of ", "without ", "free of ", "negative for ", "cleared "]
UNCERTAIN_CUES = ["possible", "probable", "cannot be excluded", "cannot rule out", "evaluating for", "questionable", "borderline"]


def label_condition_in_text(text: str, category: str) -> int:
    """
    Assign label for a single CheXpert category in text:
        1: Positive
        0: Negative
        -1: Uncertain
        9: Unmentioned (used internally, mapped to nan)
    """
    text_lower = text.lower()
    patterns = CATEGORY_PATTERNS.get(category, [])

    matched = False
    matched_pos = -1
    for p in patterns:
        pos = text_lower.find(p)
        if pos != -1:
            matched = True
            matched_pos = pos
            break

    if not matched:
        return 9  # Unmentioned

    # Check negation
    prefix = text_lower[max(0, matched_pos - 30):matched_pos]
    for n in NEGATION_CUES:
        if n in prefix:
            return 0  # Negative

    # Check uncertainty
    for u in UNCERTAIN_CUES:
        if u in prefix or u in text_lower[matched_pos:matched_pos + len(patterns[0]) + 20]:
            return -1  # Uncertain

    return 1  # Positive


def extract_chexpert_vector(text: str) -> Dict[str, Any]:
    """
    Extract 14-condition CheXpert label vector from text.
    """
    vector = {}
    for cat in CHEXPERT_CATEGORIES:
        lbl = label_condition_in_text(text, cat)
        vector[cat] = lbl if lbl != 9 else np.nan
    return vector


def compute_chexpert_f1(ref_vector: Dict[str, Any], cand_vector: Dict[str, Any]) -> float:
    """
    Compute micro CheXpert-F1 between reference vector and candidate vector.
    
    Positive targets are conditions where ref_vector == 1.
    """
    tp = 0
    fp = 0
    fn = 0

    for cat in CHEXPERT_CATEGORIES:
        r_val = ref_vector[cat]
        c_val = cand_vector[cat]

        r_pos = (r_val == 1)
        c_pos = (c_val == 1)

        if r_pos and c_pos:
            tp += 1
        elif not r_pos and c_pos:
            fp += 1
        elif r_pos and not c_pos:
            fn += 1

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0  # Perfect agreement on absence of positive findings

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(f1)


def evaluate_chexpert() -> Tuple[pd.DataFrame, pd.DataFrame]:
    for p in [PERTURBATIONS_PATH, BASELINE_PATH, GATED_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    pert_df = pd.read_csv(PERTURBATIONS_PATH)
    base_df = pd.read_csv(BASELINE_PATH)
    gated_df = pd.read_csv(GATED_PATH)

    base_lookup = {
        (str(r["sample_id"]), str(r["condition"])): str(r["generated_report"])
        for _, r in base_df.iterrows()
    }
    gated_lookup = {
        (str(r["sample_id"]), str(r["condition"])): str(r["gated_report"])
        for _, r in gated_df.iterrows()
    }

    results_rows = []

    for _, row in pert_df.iterrows():
        sid = str(row["sample_id"])
        cond = str(row["condition"])

        rf = str(row["reference_findings"]) if pd.notna(row["reference_findings"]) else ""
        ri = str(row["reference_impression"]) if pd.notna(row["reference_impression"]) else ""
        ref_text = f"{rf} {ri}".strip()

        base_report = base_lookup[(sid, cond)]
        gated_report = gated_lookup[(sid, cond)]

        ref_vec = extract_chexpert_vector(ref_text)
        base_vec = extract_chexpert_vector(base_report)
        gated_vec = extract_chexpert_vector(gated_report)

        b_f1 = compute_chexpert_f1(ref_vec, base_vec)
        g_f1 = compute_chexpert_f1(ref_vec, gated_vec)
        delta_f1 = g_f1 - b_f1

        # Format label vectors as compact string representations for auditability
        ref_vec_str = str({k: (v if pd.notna(v) else None) for k, v in ref_vec.items()})
        base_vec_str = str({k: (v if pd.notna(v) else None) for k, v in base_vec.items()})
        gated_vec_str = str({k: (v if pd.notna(v) else None) for k, v in gated_vec.items()})

        results_rows.append(
            {
                "sample_id": sid,
                "condition": cond,
                "baseline_chexpert_f1": round(b_f1, 4),
                "gated_chexpert_f1": round(g_f1, 4),
                "delta_chexpert_f1": round(delta_f1, 4),
                "reference_chexpert_vector": ref_vec_str,
                "baseline_chexpert_vector": base_vec_str,
                "gated_chexpert_vector": gated_vec_str,
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
                "mean_baseline_chexpert_f1": round(float(subset["baseline_chexpert_f1"].mean()), 4),
                "mean_gated_chexpert_f1": round(float(subset["gated_chexpert_f1"].mean()), 4),
                "mean_delta_chexpert_f1": round(float(subset["delta_chexpert_f1"].mean()), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return results_df, summary_df


def main() -> None:
    print("=" * 70)
    print("CHEXPERT-F1 EVALUATION")
    print("=" * 70)

    results_df, summary_df = evaluate_chexpert()

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(f"Results saved to: {RESULTS_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
