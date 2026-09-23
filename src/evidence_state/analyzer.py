from dataclasses import dataclass

from .states import EvidenceState


@dataclass
class EvidenceAssessment:
    image_available: bool
    context_available: bool
    context_relevant: bool
    context_consistent: bool
    evidence_strength: float


class EvidenceStateAnalyzer:
    """
    Initial evidence-state analyzer.

    This version uses explicit evidence assessments.
    The scoring/classification mechanism will be replaced or
    calibrated experimentally in later phases.
    """

    def classify(self, assessment: EvidenceAssessment) -> EvidenceState:

        if not assessment.image_available:
            return EvidenceState.INSUFFICIENT

        if not assessment.context_available:
            if assessment.evidence_strength < 0.5:
                return EvidenceState.INSUFFICIENT
            return EvidenceState.INCOMPLETE

        if not assessment.context_relevant:
            return EvidenceState.IRRELEVANT

        if not assessment.context_consistent:
            return EvidenceState.CONFLICTING

        if assessment.evidence_strength < 0.5:
            return EvidenceState.INSUFFICIENT

        return EvidenceState.SUFFICIENT