from src.evidence_state.analyzer import (
    EvidenceAssessment,
    EvidenceStateAnalyzer,
)
from src.evidence_state.states import EvidenceState


def test_sufficient_evidence():

    analyzer = EvidenceStateAnalyzer()

    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=True,
        context_consistent=True,
        evidence_strength=0.9,
    )

    assert analyzer.classify(assessment) == EvidenceState.SUFFICIENT


def test_missing_context():

    analyzer = EvidenceStateAnalyzer()

    assessment = EvidenceAssessment(
        image_available=True,
        context_available=False,
        context_relevant=False,
        context_consistent=False,
        evidence_strength=0.8,
    )

    assert analyzer.classify(assessment) == EvidenceState.INCOMPLETE


def test_conflicting_context():

    analyzer = EvidenceStateAnalyzer()

    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=True,
        context_consistent=False,
        evidence_strength=0.8,
    )

    assert analyzer.classify(assessment) == EvidenceState.CONFLICTING