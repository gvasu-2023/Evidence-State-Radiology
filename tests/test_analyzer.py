from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.states import EvidenceState


def test_image_unavailable_returns_insufficient():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=False,
        context_available=True,
        context_relevant=True,
        context_consistent=True,
        context_complete=True,
        evidence_strength=1.0,
    )
    assert analyzer.classify(assessment) == EvidenceState.INSUFFICIENT


def test_image_available_no_context_returns_insufficient():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=True,
        context_available=False,
        context_relevant=True,
        context_consistent=True,
        context_complete=False,
        evidence_strength=1.0,
    )
    assert analyzer.classify(assessment) == EvidenceState.INSUFFICIENT


def test_irrelevant_context_returns_irrelevant():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=False,
        context_consistent=True,
        context_complete=True,
        evidence_strength=1.0,
    )
    assert analyzer.classify(assessment) == EvidenceState.IRRELEVANT


def test_conflicting_context_returns_conflicting():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=True,
        context_consistent=False,
        context_complete=True,
        evidence_strength=1.0,
    )
    assert analyzer.classify(assessment) == EvidenceState.CONFLICTING


def test_relevant_incomplete_context_returns_incomplete():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=True,
        context_consistent=True,
        context_complete=False,
        evidence_strength=1.0,
    )
    assert analyzer.classify(assessment) == EvidenceState.INCOMPLETE


def test_relevant_complete_context_returns_sufficient():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=True,
        context_consistent=True,
        context_complete=True,
        evidence_strength=1.0,
    )
    assert analyzer.classify(assessment) == EvidenceState.SUFFICIENT


def test_weak_evidence_strength_returns_insufficient():
    analyzer = EvidenceStateAnalyzer()
    assessment = EvidenceAssessment(
        image_available=True,
        context_available=True,
        context_relevant=True,
        context_consistent=True,
        context_complete=True,
        evidence_strength=0.3,
    )
    assert analyzer.classify(assessment) == EvidenceState.INSUFFICIENT
