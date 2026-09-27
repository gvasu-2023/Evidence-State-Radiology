from dataclasses import dataclass

from .states import EvidenceState


@dataclass
class EvidenceAssessment:
    image_available: bool
    context_available: bool
    context_relevant: bool
    context_consistent: bool
    evidence_strength: float
    context_complete: bool = True


class EvidenceStateAnalyzer:
    """
    Evidence-state analyzer with explicit context-completeness assessment.

    Evaluates clinical and visual evidence assessments against a defined
    conceptual hierarchy:
        1. No image available -> INSUFFICIENT
        2. Image available + no clinical context -> INSUFFICIENT
        3. Context is irrelevant -> IRRELEVANT
        4. Context conflicts with available evidence -> CONFLICTING
        5. Context is relevant but incomplete -> INCOMPLETE
        6. Visual evidence strength is weak (< 0.5) -> INSUFFICIENT
        7. Otherwise -> SUFFICIENT
    """

    def classify(self, assessment: EvidenceAssessment) -> EvidenceState:

        if not assessment.image_available:
            return EvidenceState.INSUFFICIENT

        if not assessment.context_available:
            return EvidenceState.INSUFFICIENT

        if not assessment.context_relevant:
            return EvidenceState.IRRELEVANT

        if not assessment.context_consistent:
            return EvidenceState.CONFLICTING

        if not assessment.context_complete:
            return EvidenceState.INCOMPLETE

        if assessment.evidence_strength < 0.5:
            return EvidenceState.INSUFFICIENT

        return EvidenceState.SUFFICIENT