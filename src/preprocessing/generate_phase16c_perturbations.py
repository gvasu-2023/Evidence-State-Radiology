"""
Phase 16C: Controlled Six-State Perturbation Generation for Expanded IU-Xray Benchmark.

Generates exactly six controlled evidence-state perturbation records for each of the 100
underlying IU-Xray studies selected in Phase 16B:

    1. sufficient
    2. syntactic_incomplete
    3. evidentiary_incomplete
    4. irrelevant
    5. conflicting
    6. insufficient

Target output: 100 x 6 = 600 total records.

Core Scientific Principles:
    - Modifies ONLY the clinical-context condition.
    - Underlying study, images, reference findings, and reference impression remain unchanged.
    - Deterministic execution using seed SEED = 20260928.

Outputs:
    results/tables/phase16_perturbations.csv
    data/processed/iu_xray/phase16/phase16_perturbations.csv
"""

import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.context_completeness import (
    assess_context_completeness,
    detect_clinical_components,
)

SELECTED_STUDIES_PATH = ROOT / "results/tables/phase16_selected_studies.csv"
OUTPUT_TABLE_PATH = ROOT / "results/tables/phase16_perturbations.csv"
OUTPUT_PROCESSED_DIR = ROOT / "data/processed/iu_xray/phase16"
OUTPUT_PROCESSED_CSV = OUTPUT_PROCESSED_DIR / "phase16_perturbations.csv"

SEED = 20260928

VALID_CONDITIONS = [
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
]


def generate_derangement(n: int, seed: int = SEED) -> List[int]:
    """
    Generates a deterministic derangement (permutation with no fixed points) of indices 0..n-1.
    Guarantees no index is mapped to itself.
    """
    rng = random.Random(seed)
    indices = list(range(n))
    while True:
        perm = indices.copy()
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def generate_syntactic_incomplete(context: str) -> Tuple[str, str, str]:
    """
    Generates a syntactically incomplete clinical context phrase.
    Returns: (perturbed_context, method, status)
    """
    ctx = str(context).strip()
    words = ctx.split()

    truncation_targets = [
        "of", "with", "for", "and", "or", "in", "to", "from",
        "status", "history", "hx", "eval", "rule",
    ]

    patterns = [
        (r"\b(history of|hx of|h/o)\b", 2),
        (r"\b(shortness of breath|chest pain and shortness of|sob and)\b", 3),
        (r"\b(eval for|evaluation for|rule out|r/o)\b", 2),
        (r"\b(preop|preoperative evaluation for)\b", 2),
        (r"\b(status post)\b", 2),
    ]

    for pat, _ in patterns:
        m = re.search(pat, ctx, re.IGNORECASE)
        if m:
            matched_text = ctx[: m.end()].strip()
            if not assess_context_completeness(matched_text):
                return matched_text, "pattern_truncation", "success"

    for i, w in enumerate(words):
        if w.lower().strip(".,;") in truncation_targets and i > 0:
            candidate = " ".join(words[: i + 1]).rstrip(".,;")
            if not assess_context_completeness(candidate):
                return candidate, "preposition_truncation", "success"

    for i in range(len(words) - 1, 0, -1):
        candidate = " ".join(words[:i]).rstrip(".,;")
        if not assess_context_completeness(candidate):
            return candidate, "right_truncation", "success"

    candidate = ctx.rstrip(".,;") + " and"
    if not assess_context_completeness(candidate):
        return candidate, "appended_conjunction_fallback", "success"

    return candidate, "syntactic_incomplete_generation_failed", "generation_failed"


def generate_evidentiary_incomplete(context: str) -> Tuple[str, str, str, str]:
    """
    Generates an evidentiary incomplete clinical context by removing diagnostic/clinical evidence.
    Returns: (perturbed_context, removed_evidence_text, method, status)
    """
    ctx = str(context).strip()
    ctx_clean = ctx.rstrip(" .?!")

    def is_demographic_only(text: str) -> bool:
        comp = detect_clinical_components(text)
        return set(comp) == {"demographic_information"}

    # 1. Multi-sentence contexts
    if "." in ctx_clean:
        sentences = [s.strip() for s in ctx_clean.split(".") if s.strip()]
        if len(sentences) >= 2:
            for i in range(len(sentences)):
                removed = sentences[i]
                if not is_demographic_only(removed):
                    remaining_parts = sentences[:i] + sentences[i + 1 :]
                    remaining = ". ".join(remaining_parts).strip()
                    if not remaining.endswith("."):
                        remaining += "."
                    if assess_context_completeness(remaining) and not is_demographic_only(remaining):
                        return remaining, removed, "sentence_clinical_removal", "success"

    # 2. Delimiters (semicolon, comma, and, &)
    delims = [";", ",", " and ", " & "]
    for delim in delims:
        if delim in ctx_clean:
            parts = [p.strip() for p in ctx_clean.split(delim) if p.strip()]
            if len(parts) >= 2:
                for i in range(len(parts)):
                    part_removed = parts[i]
                    if not is_demographic_only(part_removed):
                        remaining_parts = parts[:i] + parts[i + 1 :]
                        remaining = ", ".join(remaining_parts).strip()
                        if not remaining.endswith("."):
                            remaining += "."
                        if assess_context_completeness(remaining):
                            if is_demographic_only(remaining):
                                remaining = remaining.rstrip(".") + ", clinical evaluation."
                            return remaining, part_removed, "clause_clinical_removal", "success"

    # 3. Prepositional phrases
    prep_patterns = [
        (r"\bwith\s+(history\s+of\s+)?(.+)$", "with_phrase"),
        (r"\bstatus\s+post\s+(.+)$", "status_post_phrase"),
        (r"\bof\s+(.+)$", "of_phrase"),
        (r"\bfor\s+(.+)$", "for_phrase"),
        (r"\bprior\s+to\s+(.+)$", "prior_to_phrase"),
    ]
    for pattern, p_name in prep_patterns:
        m = re.search(pattern, ctx_clean, re.IGNORECASE)
        if m:
            removed_target = m.group(0).strip()
            prefix = ctx_clean[: m.start()].strip()
            if prefix and not is_demographic_only(prefix):
                remaining = prefix + "."
                if assess_context_completeness(remaining):
                    return remaining, removed_target, f"prepositional_phrase_removal_{p_name}", "success"
            elif prefix:
                remaining = prefix + ", clinical evaluation."
                if assess_context_completeness(remaining):
                    return remaining, removed_target, "prepositional_phrase_removal_with_general_indication", "success"

    # 4. Term generalizations
    specific_clinical_removals = [
        ("colon cancer", "Cancer evaluation."),
        ("testis cancer", "Cancer evaluation."),
        ("testicular cancer", "Cancer evaluation."),
        ("testicular carcinoma", "Carcinoma evaluation."),
        ("bone marrow transplant", "Preprocedure evaluation."),
        ("kidney transplant", "Transplant evaluation."),
        ("altered mental status", "Neurological evaluation."),
        ("positive ppd", "Infectious disease evaluation."),
        ("ppd", "Screening evaluation."),
        ("cardiac arrest", "Post-resuscitation evaluation."),
        ("dyspnea", "Respiratory evaluation."),
        ("sob", "Respiratory evaluation."),
        ("hemoptysis", "Respiratory evaluation."),
        ("pneumonia", "Respiratory evaluation."),
        ("copd", "Respiratory evaluation."),
        ("dvt", "Vascular evaluation."),
        ("dizziness", "Clinical evaluation."),
        ("pain", "Clinical evaluation."),
    ]

    for term, generalized in specific_clinical_removals:
        if term in ctx_clean.lower():
            return generalized, term, "evidenced_term_generalization", "success"

    words = ctx_clean.split()
    if len(words) == 1:
        return "Clinical evaluation.", words[0], "single_symptom_generalization", "success"

    return "Clinical evaluation.", ctx_clean, "fallback_generalization", "success"


def generate_conflict(findings: str, impression: str) -> Tuple[str, str, str, str]:
    """
    Generates a conservative clinical-context contradiction based on reference report evidence.
    Returns: (perturbed_context, conflict_target, conflict_source_text, status)
    """
    text = (str(findings) + " " + str(impression)).lower()

    if "pneumothorax" in text:
        if any(
            neg in text
            for neg in [
                "no pneumothorax",
                "no pneumothoraces",
                "without pneumothorax",
                "free of pneumothorax",
                "no evidence of pneumothorax",
                "negative for pneumothorax",
            ]
        ):
            return (
                "Patient presenting with sudden onset shortness of breath following trauma, evaluate acute pneumothorax.",
                "pneumothorax_absence",
                "no pneumothorax reported in reference findings/impression",
                "success",
            )
        else:
            return (
                "Routine follow-up chest radiograph, no trauma or pneumothorax suspected.",
                "pneumothorax_presence",
                "pneumothorax present in reference findings/impression",
                "success",
            )

    if "effusion" in text:
        if any(
            neg in text
            for neg in [
                "no pleural effusion",
                "no effusion",
                "without effusion",
                "free of effusion",
                "no evidence of effusion",
                "no pleural effusions",
            ]
        ):
            return (
                "Shortness of breath and orthopnea, evaluate for worsening pleural effusion.",
                "effusion_absence",
                "no pleural effusion reported in reference findings/impression",
                "success",
            )
        else:
            return (
                "Routine preoperative evaluation, no history of pleural effusion.",
                "effusion_presence",
                "pleural effusion present in reference findings/impression",
                "success",
            )

    if any(term in text for term in ["cardiomegaly", "enlarged heart", "enlarged cardiac silhouette"]):
        if any(neg in text for neg in ["no cardiomegaly", "without cardiomegaly"]):
            return (
                "History of severe cardiomegaly and congestive heart failure.",
                "cardiomegaly_absence",
                "no cardiomegaly reported in reference findings/impression",
                "success",
            )
        else:
            return (
                "Normal heart size, evaluate for non-cardiac chest pain.",
                "cardiomegaly_presence",
                "cardiomegaly present in reference findings/impression",
                "success",
            )

    if any(term in text for term in ["consolidation", "infiltrate", "pneumonia", "opacity", "opacities"]):
        if any(
            neg in text
            for neg in [
                "no consolidation",
                "no focal consolidation",
                "no focal opacity",
                "no focal opacities",
                "no infiltrate",
                "no infiltrates",
                "without opacity",
                "no pneumonia",
            ]
        ):
            return (
                "Fever, cough, and chills, evaluate for acute focal pneumonia and consolidation.",
                "consolidation_absence",
                "no focal consolidation reported in reference findings/impression",
                "success",
            )
        else:
            return (
                "Screening chest radiograph, clear lungs without focal consolidation.",
                "consolidation_presence",
                "consolidation/opacity present in reference findings/impression",
                "success",
            )

    if any(term in text for term in ["clear", "normal", "unremarkable", "no acute"]):
        return (
            "Patient with high fever and shortness of breath, evaluate acute multifocal pneumonia.",
            "normal_exam_contradiction",
            "reference report indicates clear/normal exam",
            "success",
        )

    return (
        "Lungs are clear without acute cardiopulmonary disease.",
        "unspecified_abnormal_contradiction",
        "reference report describes minor findings",
        "success",
    )


def generate_phase16c_perturbations(
    selected_studies_path: Path = SELECTED_STUDIES_PATH, seed: int = SEED
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not selected_studies_path.exists():
        raise FileNotFoundError(f"Missing input selected studies table: {selected_studies_path}")

    selected_df = pd.read_csv(selected_studies_path)
    if len(selected_df) != 100:
        raise ValueError(f"Expected exactly 100 selected studies, found {len(selected_df)}")

    # Deterministic derangement for irrelevant condition
    derangement = generate_derangement(len(selected_df), seed=seed)

    records = []

    for i, r in selected_df.iterrows():
        uid = r["uid"]
        sample_id = r["sample_id"]
        orig_ctx = str(r["clinical_context"])
        findings = str(r["findings"]) if pd.notna(r["findings"]) else ""
        impression = str(r["impression"]) if pd.notna(r["impression"]) else ""
        comparison = str(r["comparison"]) if pd.notna(r["comparison"]) else ""
        frontal = str(r["frontal_image"]) if pd.notna(r["frontal_image"]) and str(r["frontal_image"]).strip().lower() != "nan" else ""
        lateral = str(r["lateral_image"]) if pd.notna(r["lateral_image"]) and str(r["lateral_image"]).strip().lower() != "nan" else ""

        # 1. SUFFICIENT
        records.append({
            "sample_id": sample_id,
            "uid": uid,
            "condition": "sufficient",
            "evidence_state": "sufficient",
            "original_context": orig_ctx,
            "perturbed_context": orig_ctx,
            "frontal_image": frontal,
            "lateral_image": lateral,
            "findings": findings,
            "impression": impression,
            "comparison": comparison,
            "source_context_uid": uid,
            "source_context_sample_id": sample_id,
            "removed_evidence_text": "",
            "evidentiary_incomplete_method": "",
            "conflict_target": "",
            "conflict_method": "",
            "conflict_source_text": "",
            "generation_status": "success",
        })

        # 2. SYNTACTIC INCOMPLETE
        syn_ctx, syn_method, syn_status = generate_syntactic_incomplete(orig_ctx)
        records.append({
            "sample_id": sample_id,
            "uid": uid,
            "condition": "syntactic_incomplete",
            "evidence_state": "incomplete",
            "original_context": orig_ctx,
            "perturbed_context": syn_ctx,
            "frontal_image": frontal,
            "lateral_image": lateral,
            "findings": findings,
            "impression": impression,
            "comparison": comparison,
            "source_context_uid": uid,
            "source_context_sample_id": sample_id,
            "removed_evidence_text": "",
            "evidentiary_incomplete_method": syn_method,
            "conflict_target": "",
            "conflict_method": "",
            "conflict_source_text": "",
            "generation_status": syn_status,
        })

        # 3. EVIDENTIARY INCOMPLETE
        evi_ctx, evi_removed, evi_method, evi_status = generate_evidentiary_incomplete(orig_ctx)
        records.append({
            "sample_id": sample_id,
            "uid": uid,
            "condition": "evidentiary_incomplete",
            "evidence_state": "incomplete",
            "original_context": orig_ctx,
            "perturbed_context": evi_ctx,
            "frontal_image": frontal,
            "lateral_image": lateral,
            "findings": findings,
            "impression": impression,
            "comparison": comparison,
            "source_context_uid": uid,
            "source_context_sample_id": sample_id,
            "removed_evidence_text": evi_removed,
            "evidentiary_incomplete_method": evi_method,
            "conflict_target": "",
            "conflict_method": "",
            "conflict_source_text": "",
            "generation_status": evi_status,
        })

        # 4. IRRELEVANT
        donor_idx = derangement[i]
        donor_row = selected_df.iloc[donor_idx]
        donor_uid = donor_row["uid"]
        donor_sample_id = donor_row["sample_id"]
        donor_ctx = str(donor_row["clinical_context"])
        records.append({
            "sample_id": sample_id,
            "uid": uid,
            "condition": "irrelevant",
            "evidence_state": "irrelevant",
            "original_context": orig_ctx,
            "perturbed_context": donor_ctx,
            "frontal_image": frontal,
            "lateral_image": lateral,
            "findings": findings,
            "impression": impression,
            "comparison": comparison,
            "source_context_uid": donor_uid,
            "source_context_sample_id": donor_sample_id,
            "removed_evidence_text": "",
            "evidentiary_incomplete_method": "",
            "conflict_target": "",
            "conflict_method": "",
            "conflict_source_text": "",
            "generation_status": "success",
        })

        # 5. CONFLICTING
        conf_ctx, conf_target, conf_source, conf_status = generate_conflict(findings, impression)
        records.append({
            "sample_id": sample_id,
            "uid": uid,
            "condition": "conflicting",
            "evidence_state": "conflicting",
            "original_context": orig_ctx,
            "perturbed_context": conf_ctx,
            "frontal_image": frontal,
            "lateral_image": lateral,
            "findings": findings,
            "impression": impression,
            "comparison": comparison,
            "source_context_uid": uid,
            "source_context_sample_id": sample_id,
            "removed_evidence_text": "",
            "evidentiary_incomplete_method": "",
            "conflict_target": conf_target,
            "conflict_method": "reference_evidence_contradiction",
            "conflict_source_text": conf_source,
            "generation_status": conf_status,
        })

        # 6. INSUFFICIENT
        records.append({
            "sample_id": sample_id,
            "uid": uid,
            "condition": "insufficient",
            "evidence_state": "insufficient",
            "original_context": orig_ctx,
            "perturbed_context": "",
            "frontal_image": frontal,
            "lateral_image": lateral,
            "findings": findings,
            "impression": impression,
            "comparison": comparison,
            "source_context_uid": uid,
            "source_context_sample_id": sample_id,
            "removed_evidence_text": "",
            "evidentiary_incomplete_method": "",
            "conflict_target": "",
            "conflict_method": "",
            "conflict_source_text": "",
            "generation_status": "success",
        })

    result_df = pd.DataFrame(records)

    # Sort deterministically by uid and condition
    condition_order = {cond: idx for idx, cond in enumerate(VALID_CONDITIONS)}
    result_df["cond_order"] = result_df["condition"].map(condition_order)
    result_df["uid_int"] = result_df["uid"].astype(int)
    result_df = result_df.sort_values(["uid_int", "cond_order"]).reset_index(drop=True)
    result_df = result_df.drop(columns=["cond_order", "uid_int"])

    # Compute audit stats
    counts_by_condition = result_df["condition"].value_counts().to_dict()
    status_by_condition = (
        result_df.groupby(["condition", "generation_status"]).size().unstack(fill_value=0).to_dict()
    )

    complete_studies = 0
    for uid_val, group in result_df.groupby("uid"):
        if len(group) == 6 and (group["generation_status"] == "success").all():
            complete_studies += 1

    stats = {
        "total_selected_studies": len(selected_df),
        "total_perturbation_records": len(result_df),
        "counts_by_condition": counts_by_condition,
        "status_by_condition": status_by_condition,
        "complete_six_state_studies": complete_studies,
        "seed": seed,
    }

    return result_df, stats


def main() -> None:
    print("=" * 70)
    print("PHASE 16C: CONTROLLED SIX-STATE PERTURBATION GENERATION")
    print("=" * 70)

    result_df, stats = generate_phase16c_perturbations()

    # Save to both required locations
    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT_TABLE_PATH, index=False)
    result_df.to_csv(OUTPUT_PROCESSED_CSV, index=False)

    print(f"\nPerturbation table saved to:\n  - {OUTPUT_TABLE_PATH}\n  - {OUTPUT_PROCESSED_CSV}")
    print(f"Total Records Generated: {len(result_df)}")

    print("\n--- GENERATION AUDIT STATISTICS ---")
    print(f"Total underlying studies: {stats['total_selected_studies']}")
    print(f"Total perturbation records: {stats['total_perturbation_records']}")
    print(f"Complete 6-state studies: {stats['complete_six_state_studies']} / 100")
    print("\nCondition Breakdown:")
    for cond in VALID_CONDITIONS:
        cnt = stats["counts_by_condition"].get(cond, 0)
        print(f"  {cond:24s}: {cnt}")

    print("\nPhase 16C perturbation generation complete.")


if __name__ == "__main__":
    main()
