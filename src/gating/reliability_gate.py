from dataclasses import dataclass

from src.evidence_state.states import EvidenceState


@dataclass
class GateDecision:
    action: str
    reason: str


class ReliabilityGate:
    """
    Maps evidence states to generation policies.

    The policy is an initial research prototype and must be
    evaluated experimentally.
    """

    def decide(self, state: EvidenceState) -> GateDecision:

        if state == EvidenceState.SUFFICIENT:
            return GateDecision(
                action="generate",
                reason="Evidence is sufficient and context is consistent."
            )

        if state == EvidenceState.INCOMPLETE:
            return GateDecision(
                action="qualify",
                reason="Relevant clinical context is incomplete."
            )

        if state == EvidenceState.IRRELEVANT:
            return GateDecision(
                action="discount_context",
                reason="Provided clinical context is not relevant to the imaging task."
            )

        if state == EvidenceState.CONFLICTING:
            return GateDecision(
                action="qualify_or_abstain",
                reason="Clinical context conflicts with available evidence."
            )

        return GateDecision(
            action="abstain",
            reason="Available evidence is insufficient for reliable generation."
        )