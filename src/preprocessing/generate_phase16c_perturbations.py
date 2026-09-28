"""
Phase 16C: Controlled Six-State Perturbation Generation for Expanded IU-Xray Benchmark.

Generates six controlled evidence-state perturbation slots for each of the 100
underlying IU-Xray studies selected in Phase 16E:

    1. sufficient
    2. syntactic_incomplete
    3. evidentiary_incomplete
    4. irrelevant
    5. conflicting
    6. insufficient

Target output: 100 x 6 = 600 total slots. A slot without a defensible perturbation
is retained with `generation_status=generation_failed` and an empty perturbed context.

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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_state.context_completeness import (
    assess_context_completeness,
    HISTORY_KEYWORDS,
    PROCEDURE_KEYWORDS,
    SYMPTOM_KEYWORDS,
)
from src.evaluation.claim_patterns import REFERENCE_PATTERNS, split_sentences

SELECTED_STUDIES_PATH = ROOT / "results/tables/phase16e_selected_studies.csv"
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

PLACEHOLDER_PATTERN = re.compile(r"\b[xX]{2,}\b")
DEMOGRAPHIC_PATTERN = re.compile(
    r"\b(?:xxxx|\d+)[ -]?year[ -]?old(?:\s+(?:male|female))?\b"
    r"|\b(?:male|female)\b|\bpatient\b",
    re.IGNORECASE,
)

CONFLICT_TARGETS = {
    "pneumothorax": {
        "aliases": REFERENCE_PATTERNS["pneumothorax"],
        "positive_context": "The patient has a pneumothorax.",
        "negative_context": "No pneumothorax is suspected.",
    },
    "pleural_effusion": {
        "aliases": REFERENCE_PATTERNS["pleural effusion"],
        "positive_context": "The patient has a pleural effusion.",
        "negative_context": "No pleural effusion is suspected.",
    },
    "cardiomegaly": {
        "aliases": REFERENCE_PATTERNS["cardiomegaly"] + [
            "enlarged heart", "enlarged cardiac silhouette", "heart size is enlarged",
        ],
        "positive_context": "Cardiomegaly is present.",
        "negative_context": "No cardiomegaly is reported.",
    },
    "consolidation": {
        "aliases": REFERENCE_PATTERNS["consolidation"],
        "positive_context": "Focal consolidation is present.",
        "negative_context": "No focal consolidation is present.",
    },
    "pneumonia": {
        "aliases": REFERENCE_PATTERNS["pneumonia"],
        "positive_context": "The patient has pneumonia.",
        "negative_context": "No pneumonia is present.",
    },
}

UNCERTAINTY_CUES = (
    "possible ", "possibly ", "questionable ", "cannot exclude", "may represent",
    "could represent", "suggestive of", "suspected ", "versus ",
)

DONOR_OVERLAP_VOCABULARY = {
    "pneumothorax": ("pneumothorax", "pneumothoraces"),
    "pleural_effusion": ("pleural effusion", "pleural effusions"),
    "cardiomegaly": (
        "cardiomegaly", "enlarged heart", "enlarged cardiac silhouette",
        "cardiac silhouette is enlarged", "heart size appears enlarged",
    ),
    "consolidation": ("consolidation", "infiltrate", "infiltrates", "pneumonia"),
    "hyperinflation": ("hyperinflated", "hyperinflation", "hyperexpanded", "copd", "emphysema"),
    "malignancy": ("cancer", "carcinoma", "metastatic", "malignancy", "tumor"),
    "fracture": ("fracture", "fractures"),
    "pulmonary_edema": ("pulmonary edema", "edema"),
    "fibrosis": ("fibrosis", "fibrotic"),
}


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


def _normalized_clinical_concepts(text: str) -> set[str]:
    text_lower = str(text).lower()
    concepts = set()
    for concept, aliases in DONOR_OVERLAP_VOCABULARY.items():
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text_lower) for alias in aliases):
            concepts.add(concept)
    return concepts


def _donor_overlaps_target(donor_context: str, findings: str, impression: str) -> bool:
    donor_concepts = _normalized_clinical_concepts(donor_context)
    reference = f"{findings} {impression}"
    reference_concepts = _normalized_clinical_concepts(reference)
    if donor_concepts & reference_concepts:
        return True
    if donor_concepts and re.search(
        r"\b(?:no acute (?:cardiopulmonary|pulmonary|intrathoracic)|without acute (?:cardiopulmonary|pulmonary))\b",
        reference,
        re.IGNORECASE,
    ):
        return True
    return False


def generate_irrelevant_assignment(selected_df: pd.DataFrame, seed: int = SEED) -> Dict[int, int]:
    """Find a deterministic one-to-one donor assignment with no obvious evidence overlap."""
    rows = list(selected_df.iterrows())
    preferences: Dict[int, List[int]] = {}
    for target_index, target_row in rows:
        target_uid = int(target_row["uid"])
        candidates = []
        for donor_index, donor_row in rows:
            if donor_index == target_index:
                continue
            if _donor_overlaps_target(
                str(donor_row["clinical_context"]),
                str(target_row["findings"]),
                str(target_row["impression"]),
            ):
                continue
            candidates.append(donor_index)
        random.Random(seed + target_uid).shuffle(candidates)
        preferences[target_index] = candidates

    donor_to_target: Dict[int, int] = {}

    def assign(target_index: int, seen_donors: set[int]) -> bool:
        for donor_index in preferences[target_index]:
            if donor_index in seen_donors:
                continue
            seen_donors.add(donor_index)
            previous = donor_to_target.get(donor_index)
            if previous is None or assign(previous, seen_donors):
                donor_to_target[donor_index] = target_index
                return True
        return False

    target_order = sorted(
        preferences,
        key=lambda idx: (len(preferences[idx]), int(selected_df.iloc[idx]["uid"])),
    )
    for target_index in target_order:
        if not assign(target_index, set()):
            break
    return {target_index: donor_index for donor_index, target_index in donor_to_target.items()}


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


def _readable_evidence(text: str) -> bool:
    """Return True only when text contains readable, non-demographic clinical evidence."""
    cleaned = DEMOGRAPHIC_PATTERN.sub(" ", str(text))
    cleaned = PLACEHOLDER_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\b(?:history of|hx of)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"[^a-zA-Z]+", " ", cleaned).lower().strip()
    if not cleaned or cleaned in {"history of", "hx of", "history", "known", "prior"}:
        return False

    meaningful_keywords = [
        keyword for keyword in (SYMPTOM_KEYWORDS + HISTORY_KEYWORDS + PROCEDURE_KEYWORDS)
        if keyword not in {"history of", "hx of", "prior", "known", "patient"}
    ]
    if any(keyword in cleaned for keyword in meaningful_keywords):
        return True

    extra_evidence_terms = (
        "trauma", "mva", "troponin", "lymphadenopathy", "adenopathy",
        "nodule", "nodules", "rales", "breath sounds", "oxygen",
        "desaturation", "hypoxia", "positive ppd", "tuberculosis",
        "infection", "hernia", "transplant", "malignancy", "metastatic",
        "carcinoma", "hemoptysis", "wheezing", "syncope", "collapse",
        "cardiac arrest", "home oxygen", "chest tube", "pneumothorax",
        "pleural effusion", "congestion", "weakness", "emesis",
        "pneumonia", "dvt", "sob", "copd", "sarcoidosis", "smoking",
        "not feeling well", "difficulty breathing", "breathing", "blood pressure",
        "chest pressure", "aml", "bmt workup", "workup", "lymphadenopathy",
        "bronchitis", "infection", "fever", "cough", "edema", "asthma",
        "altered mental status", "bone marrow transplant", "kidney transplant",
        "mental status changes", "flulike symptoms", "placement",
        "hiv", "infiltrate", "infiltrates",
    )
    return any(term in cleaned for term in extra_evidence_terms)


def _remove_placeholders(text: str) -> str:
    text = DEMOGRAPHIC_PATTERN.sub(" ", str(text))
    text = PLACEHOLDER_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:-")


def generate_evidentiary_incomplete(context: str) -> Tuple[str, str, str, str]:
    """
    Generates an evidentiary incomplete clinical context by removing diagnostic/clinical evidence.
    Returns: (perturbed_context, removed_evidence_text, method, status)
    """
    ctx = str(context).strip()
    ctx_clean = ctx.rstrip(" .?!")
    if not _readable_evidence(ctx_clean):
        return "", "", "no_readable_clinical_evidence", "generation_failed"

    # Prefer removal of a whole readable clause while retaining other readable evidence.
    parts = [
        p.strip()
        for p in re.split(r"\s*(?:[.;]|,|\band\b|&|\bwith\b)\s*", ctx_clean, flags=re.I)
        if p.strip()
    ]
    for index, removed in enumerate(parts):
        if not _readable_evidence(removed):
            continue
        remaining_parts = parts[:index] + parts[index + 1 :]
        remaining = ", ".join(remaining_parts).strip(" ,;:-")
        if remaining and _readable_evidence(remaining):
            if not remaining.endswith((".", "?", "!")):
                remaining += "."
            if assess_context_completeness(remaining):
                return remaining, removed, "clause_clinical_removal", "success"

    # A single readable concept may be generalized to a complete, less-specific indication.
    # The generic "Clinical evaluation." fallback is not a clinical indication and is
    # explicitly rejected. Placeholder-only text never reaches this branch because of
    # the evidence check above.
    generalizations = (
        (r"\b(?:shortness\s+of\s+breath|dyspnea|sob|difficulty breathing|breathing)\b", "Respiratory evaluation."),
        (r"\b(?:chest\s+)?(?:pain|pressure)\b", "Clinical evaluation."),
        (r"\b(?:copd|pneumonia|hemoptysis|cough|wheezing|rales|congestion|sarcoidosis|bronchitis)\b", "Respiratory evaluation."),
        (r"\b(?:metastatic\s+)?(?:colon|testis|testicular)\s+(?:cancer|carcinoma)\b", "Cancer evaluation."),
        (r"\b(?:cancer|carcinoma)\b", "Cancer evaluation."),
        (r"\b(?:kidney transplant|bone marrow transplant|transplant)\b", "Preprocedure evaluation."),
        (r"\b(?:positive ppd|ppd|tuberculosis)\b", "Infectious disease evaluation."),
        (r"\b(?:cardiac arrest|altered mental status|syncope|dizziness)\b", "Clinical evaluation."),
        (r"\b(?:dvt|troponin|hypoxia|desaturation|oxygen saturation|mva|trauma|infection|smoking|blood pressure|aml|bmt workup|lymphadenopathy|not feeling well)\b", "Clinical evaluation."),
        (r"\b(?:pleural effusion|pneumothorax|mental status changes|flulike symptoms|weakness|placement)\b", "Clinical evaluation."),
        (r"\b(?:preop|preoperative|pre-op)\b", "Preprocedure evaluation."),
    )
    readable_ctx = _remove_placeholders(ctx_clean)
    for pattern, generalized in generalizations:
        match = re.search(pattern, readable_ctx, re.I)
        if not match:
            continue
        removed = match.group(0).strip()
        if not _readable_evidence(removed) or not assess_context_completeness(generalized):
            continue
        if generalized.strip().casefold() == "clinical evaluation.":
            continue
        if generalized.lower().strip(".") == readable_ctx.lower().strip(" ."):
            continue
        return generalized, removed, "readable_concept_generalization", "success"

    return "", "", "no_safe_evidence_removal", "generation_failed"


def _explicit_polarity(text: str, target: str) -> Tuple[str, str]:
    """Return unambiguous explicit polarity and its supporting sentence for a finding."""
    config = CONFLICT_TARGETS[target]
    polarities = []
    evidence_sentences = []
    text_sentences = []
    for sentence in split_sentences(str(text).lower()):
        text_sentences.extend(part.strip() for part in re.split(r"[;:]", sentence) if part.strip())
    for sentence in text_sentences:
        for alias in config["aliases"]:
            for match in re.finditer(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", sentence):
                start = match.start()
                prefix = sentence[:start]
                # A contrast starts a new local polarity scope.
                prefix = re.split(r"\b(?:but|however|although|yet)\b", prefix)[-1]
                prefix = re.sub(r"\bno\s+change(?:s)?\s+(?:in|to)\b", " ", prefix)
                prior = prefix[-100:]
                if re.search(r"\b(?:history\s+of|prior\s+history\s+of)\s*$", prior):
                    continue
                if any(cue in prior for cue in UNCERTAINTY_CUES):
                    continue
                negated = bool(re.search(
                    r"\b(?:no|without|absent|negative\s+for|free\s+of|lacking)\b"
                    r"(?:[\w, /-]{0,90})$",
                    prior,
                ))
                after = sentence[match.end():match.end() + 45]
                negated = negated or bool(re.match(
                    r"\s*(?:is|was|are|were)?\s*(?:absent|not\s+(?:present|suspected|seen|identified|visualized|demonstrated|evident))\b",
                    after,
                ))
                polarities.append("NEGATED" if negated else "AFFIRMED")
                evidence_sentences.append(sentence.strip())
    if not polarities and target == "pneumonia":
        explicit_absence_patterns = (
            r"\bno\s+acute\s+(?:pulmonary|cardiopulmonary)\s+(?:disease|abnormality|process|findings?)\b",
            r"\b(?:lungs?\s+(?:(?:are|were|remain|appear)\s+)?(?:hypoinflated\s+but\s+)?clear|clear\s+lungs)\b",
            r"\blungs?\s+remain\s+clear\b",
        )
        for sentence in text_sentences:
            if any(re.search(pattern, sentence) for pattern in explicit_absence_patterns):
                polarities.append("NEGATED")
                evidence_sentences.append(sentence.strip())
    if not polarities and target == "cardiomegaly":
        explicit_normal_size_patterns = (
            r"\bheart\s+is\s+normal\s+in\s+size\b",
            r"\bheart\s+size\s+(?:is\s+)?within\s+normal\s+limits\b",
            r"\bheart\s+size\s+is\s+normal\b",
            r"\bcardiac\s+silhouette\s+is\s+normal\s+in\s+size\b",
            r"\bheart\s+and\s+mediastinum\s+normal\b",
        )
        for sentence in text_sentences:
            if any(re.search(pattern, sentence) for pattern in explicit_normal_size_patterns):
                polarities.append("NEGATED")
                evidence_sentences.append(sentence.strip())
    if not polarities or len(set(polarities)) != 1:
        return "AMBIGUOUS", ""
    return polarities[0], " | ".join(dict.fromkeys(evidence_sentences))


def _context_polarity(context: str, target: str) -> str:
    polarity, _ = _explicit_polarity(context, target)
    return polarity


def generate_conflict(findings: str, impression: str) -> Tuple[str, str, str, str]:
    """
    Generates a conservative clinical-context contradiction based on reference report evidence.
    Returns: (perturbed_context, conflict_target, conflict_source_text, status)
    """
    reference_text = f"{findings} {impression}"
    for target, config in CONFLICT_TARGETS.items():
        reference_polarity, evidence = _explicit_polarity(reference_text, target)
        if reference_polarity not in {"AFFIRMED", "NEGATED"}:
            continue
        generated_polarity = "NEGATED" if reference_polarity == "AFFIRMED" else "AFFIRMED"
        context_key = "negative_context" if generated_polarity == "NEGATED" else "positive_context"
        perturbed_context = config[context_key]
        if _context_polarity(perturbed_context, target) != generated_polarity:
            continue
        conflict_target = f"{target}_{'presence' if reference_polarity == 'AFFIRMED' else 'absence'}"
        return perturbed_context, conflict_target, evidence, "success"

    return "", "", "no_explicit_opposite_polarity_reference_evidence", "generation_failed"


def generate_phase16c_perturbations(
    selected_studies_path: Path = SELECTED_STUDIES_PATH, seed: int = SEED
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not selected_studies_path.exists():
        raise FileNotFoundError(f"Missing input selected studies table: {selected_studies_path}")

    selected_df = pd.read_csv(selected_studies_path)
    if len(selected_df) != 100:
        raise ValueError(f"Expected exactly 100 selected studies, found {len(selected_df)}")

    # A deterministic screened perfect matching avoids obvious target-evidence overlap.
    donor_assignment = generate_irrelevant_assignment(selected_df, seed=seed)

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
            "generation_failure_reason": "",
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
            "generation_failure_reason": "" if syn_status == "success" else "no_structural_truncation_found",
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
            "generation_failure_reason": "" if evi_status == "success" else evi_removed or evi_method,
        })

        # 4. IRRELEVANT
        donor_idx = donor_assignment.get(i)
        if donor_idx is not None:
            donor_row = selected_df.iloc[donor_idx]
            donor_uid = donor_row["uid"]
            donor_sample_id = donor_row["sample_id"]
            donor_ctx = str(donor_row["clinical_context"])
            irrel_status = "success"
            irrel_failure_reason = ""
        else:
            donor_uid = ""
            donor_sample_id = ""
            donor_ctx = ""
            irrel_status = "generation_failed"
            irrel_failure_reason = "no_screened_nonself_donor_available"
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
            "evidentiary_incomplete_method": "deterministic_screened_donor_assignment",
            "conflict_target": "",
            "conflict_method": "",
            "conflict_source_text": "",
            "generation_status": irrel_status,
            "generation_failure_reason": irrel_failure_reason,
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
            "conflict_method": "explicit_reference_polarity_contradiction" if conf_status == "success" else "",
            "conflict_source_text": conf_source if conf_status == "success" else "",
            "generation_status": conf_status,
            "generation_failure_reason": "" if conf_status == "success" else conf_source,
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
            "generation_failure_reason": "",
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
    methods_by_condition: Dict[str, Dict[str, int]] = {}
    for condition, group in result_df.groupby("condition"):
        method_column = "conflict_method" if condition == "conflicting" else "evidentiary_incomplete_method"
        methods = [value for value in group[method_column].astype(str) if value]
        methods_by_condition[condition] = dict(Counter(methods))
    failures_by_condition: Dict[str, Dict[str, int]] = {}
    failed_df = result_df[result_df["generation_status"] != "success"]
    for condition, group in failed_df.groupby("condition"):
        failures_by_condition[condition] = dict(Counter(group["generation_failure_reason"].astype(str)))

    complete_studies = 0
    for uid_val, group in result_df.groupby("uid"):
        if len(group) == 6 and (group["generation_status"] == "success").all():
            complete_studies += 1

    stats = {
        "total_selected_studies": len(selected_df),
        "total_perturbation_records": len(result_df),
        "counts_by_condition": counts_by_condition,
        "status_by_condition": status_by_condition,
        "methods_by_condition": methods_by_condition,
        "failures_by_condition": failures_by_condition,
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
    print("\nGeneration failures:")
    for cond, failures in stats["failures_by_condition"].items():
        print(f"  {cond}: {failures}")

    print("\nPhase 16C perturbation generation complete.")


if __name__ == "__main__":
    main()
