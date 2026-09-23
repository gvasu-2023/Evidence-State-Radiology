from src.evidence_state.states import EvidenceState
from src.gating.reliability_gate import ReliabilityGate


def test_sufficient_generates():

    gate = ReliabilityGate()

    decision = gate.decide(EvidenceState.SUFFICIENT)

    assert decision.action == "generate"


def test_insufficient_abstains():

    gate = ReliabilityGate()

    decision = gate.decide(EvidenceState.INSUFFICIENT)

    assert decision.action == "abstain"