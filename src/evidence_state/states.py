from enum import Enum


class EvidenceState(str, Enum):
    SUFFICIENT = "sufficient"
    INCOMPLETE = "incomplete"
    IRRELEVANT = "irrelevant"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"