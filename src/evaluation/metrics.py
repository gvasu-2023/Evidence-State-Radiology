"""
Reference-grounded comparison labels.

These labels describe polarity agreement between generated-report
claim candidates and reference claim candidates.

They are not RadGraph-F1, CheXpert-F1, ECE, Brier, coverage,
abstention, hallucination rate, factuality F1, reliability score,
or overall accuracy.
"""


VALID_LABELS = {
    "supported",
    "contradiction",
    "omission",
    "unsupported_affirmation",
    "ungrounded_affirmation",
    "extra_negation",
    "unclear",
}


def assign_factuality_label(
    generated_polarity: str | None,
    reference_polarity: str | None,
) -> str:
    """
    Assign a conservative reference-grounded label.

    generated_polarity:
        AFFIRMED, NEGATED, MENTIONED_UNCLEAR, or NOT_MENTIONED.

    reference_polarity:
        AFFIRMED, NEGATED, MENTIONED_UNCLEAR, or None if no
        reference candidate exists for the finding.
    """

    generated = (
        None
        if generated_polarity is None
        else str(generated_polarity).strip()
    )

    reference = (
        None
        if reference_polarity is None
        else str(reference_polarity).strip()
    )

    if generated in {None, "", "NOT_MENTIONED"}:
        generated = "NOT_MENTIONED"

    if reference in {None, ""}:
        reference = None

    if generated == "MENTIONED_UNCLEAR":
        return "unclear"

    if generated == "AFFIRMED":
        if reference == "NEGATED":
            return "unsupported_affirmation"

        if reference == "AFFIRMED":
            return "supported"

        if reference is None:
            return "ungrounded_affirmation"

        return "unclear"

    if generated == "NEGATED":
        if reference == "AFFIRMED":
            return "contradiction"

        if reference == "NEGATED":
            return "supported"

        if reference is None:
            return "extra_negation"

        return "unclear"

    # generated == NOT_MENTIONED
    if reference is None:
        raise ValueError(
            "Cannot label a finding that is absent from both "
            "generated claims and reference candidates."
        )

    return "omission"
