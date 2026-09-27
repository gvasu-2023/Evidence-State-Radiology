"""
Phase 12C: Component-Based Context Completeness Parser.

Classifies clinical context strings into complete or incomplete states by evaluating
observable structural fragmentation rather than clinical category or brevity.

Core Principles:
- Single clinically meaningful components (e.g. "History of CHF.", "Chest pain.") are COMPLETE.
- Lowercase initial characters do NOT imply incomplete.
- "History of..." prefixes do NOT automatically imply incomplete.
- Observable fragmentation signals (dangling conjunctions, dangling prepositions,
  trailing punctuation fragments) trigger INCOMPLETE.

Anti-leakage:
Operates strictly on the supplied input context string. Does NOT receive condition,
expected_state, original_context, transformation_type, or reference findings.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional
import pandas as pd


@dataclass
class ContextCompletenessResult:
    is_complete: bool
    rule_triggered: str
    detected_components: List[str] = field(default_factory=list)
    fragmentation_signals: List[str] = field(default_factory=list)


# Keywords for component recognition (descriptive metadata)
SYMPTOM_KEYWORDS = [
    "pain", "chest pain", "dyspnea", "shortness of breath", "cough", "fever",
    "congestion", "nausea", "vomiting", "fatigue", "weakness", "hypoxia",
    "hemoptysis", "wheezing", "chills", "dizziness", "swelling", "edema"
]

HISTORY_KEYWORDS = [
    "history of", "hx of", "prior", "known", "past medical history", "asthma",
    "chf", "mitral valve prolapse", "stent placement", "hypertension", "tb",
    "tuberculosis", "copd", "cad", "diabetes", "cancer"
]

PROCEDURE_KEYWORDS = [
    "preop", "preoperative", "post-op", "postoperative", "evaluation for surgery",
    "line placement", "clearance", "bariatric", "lumbar surgery", "aneurysm repair",
    "surgery", "procedure", "resection", "transplant"
]

DEMOGRAPHIC_KEYWORDS = [
    "year-old", "yo", "male", "female", "xxxx-year-old", "patient"
]

OTHER_INDICATION_KEYWORDS = [
    "routine", "screening", "evaluate for", "eval for", "follow-up", "check",
    "radiograph", "scan", "film"
]


def detect_clinical_components(text: str) -> List[str]:
    """Identify recognized clinical component types present in text for structured diagnostics."""
    t_lower = text.lower()
    components = []

    if any(k in t_lower for k in SYMPTOM_KEYWORDS):
        components.append("symptom_presenting_complaint")
    if any(k in t_lower for k in HISTORY_KEYWORDS):
        components.append("medical_history")
    if any(k in t_lower for k in PROCEDURE_KEYWORDS):
        components.append("procedure_preop_indication")
    if any(k in t_lower for k in DEMOGRAPHIC_KEYWORDS):
        components.append("demographic_information")
    if any(k in t_lower for k in OTHER_INDICATION_KEYWORDS):
        components.append("other_clinical_indication")

    if not components:
        components.append("unclassified_clinical_text")

    return components


def parse_context_completeness(context_text: Optional[str]) -> ContextCompletenessResult:
    """
    Transparent component and fragmentation parser for clinical context completeness.

    Args:
        context_text: Clinical context string supplied at inference time.

    Returns:
        ContextCompletenessResult containing boolean decision, rule triggered,
        detected components, and fragmentation signals.
    """
    # 1. Empty / null / whitespace input check
    if context_text is None or pd.isna(context_text):
        return ContextCompletenessResult(
            is_complete=False,
            rule_triggered="empty_input",
            detected_components=[],
            fragmentation_signals=["empty_input"],
        )

    text = str(context_text).strip()
    if not text:
        return ContextCompletenessResult(
            is_complete=False,
            rule_triggered="empty_input",
            detected_components=[],
            fragmentation_signals=["empty_input"],
        )

    components = detect_clinical_components(text)
    fragmentation_signals = []

    # 2. Check for trailing punctuation fragment (e.g. "chest pain," or "history of CHF -")
    if re.search(r'[,/-]\s*$', text):
        fragmentation_signals.append("trailing_punctuation_fragment")
        return ContextCompletenessResult(
            is_complete=False,
            rule_triggered="trailing_punctuation_fragment",
            detected_components=components,
            fragmentation_signals=fragmentation_signals,
        )

    # Clean trailing punctuation for word-boundary fragmentation checks
    clean_text = text.rstrip(" .?!").strip()
    clean_lower = clean_text.lower()

    # 3. Check for specific incomplete prepositional/verb prefixes with no object following
    incomplete_standalone_phrases = [
        "preop for",
        "eval for",
        "evaluation for",
        "history of",
        "hx of",
        "presents with",
        "presenting with",
        "evaluated for",
    ]
    if clean_lower in incomplete_standalone_phrases:
        fragmentation_signals.append("dangling_preposition")
        return ContextCompletenessResult(
            is_complete=False,
            rule_triggered="dangling_preposition",
            detected_components=components,
            fragmentation_signals=fragmentation_signals,
        )

    # 4. Check for dangling conjunction at the end of the phrase
    if re.search(r'\b(and|or|but)\s*$', clean_lower):
        fragmentation_signals.append("dangling_conjunction")
        return ContextCompletenessResult(
            is_complete=False,
            rule_triggered="dangling_conjunction",
            detected_components=components,
            fragmentation_signals=fragmentation_signals,
        )

    # 5. Check for dangling preposition at the end of the phrase
    if re.search(r'\b(for|of|on|to|in|at|by|from|about|with)\s*$', clean_lower):
        fragmentation_signals.append("dangling_preposition")
        return ContextCompletenessResult(
            is_complete=False,
            rule_triggered="dangling_preposition",
            detected_components=components,
            fragmentation_signals=fragmentation_signals,
        )

    # If no structural fragmentation is detected, the context is complete.
    return ContextCompletenessResult(
        is_complete=True,
        rule_triggered="none",
        detected_components=components,
        fragmentation_signals=[],
    )


def assess_context_completeness(context_text: Optional[str]) -> bool:
    """
    Convenience wrapper returning boolean context completeness.

    Args:
        context_text: Clinical context string supplied at inference time.

    Returns:
        bool: True if context is complete, False if incomplete/fragmented.
    """
    res = parse_context_completeness(context_text)
    return res.is_complete
