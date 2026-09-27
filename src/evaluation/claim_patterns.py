"""
Shared claim-pattern and polarity helpers.

These helpers are used by both reference candidate extraction and
generated-report claim extraction. They must remain conservative:

Generic phrases such as 'lungs are clear' are NOT treated as
blanket negations of unmentioned findings.
"""


REFERENCE_PATTERNS = {
    "cardiomegaly": [
        "cardiomegaly",
        "borderline enlarged",
        "borderline enlargement",
        "cardiac silhouette is borderline enlarged",
    ],
    "consolidation": [
        "consolidation",
    ],
    "airspace opacity": [
        "air space opacity",
        "airspace opacity",
        "air space disease",
        "airspace disease",
        "right upper lobe opacity",
        "focal opacity",
        "focal opacities",
        "pulmonary opacity",
        "pulmonary opacities",
        "opacities",
        "opacity",
    ],
    "pleural effusion": [
        "pleural effusion",
        "pleural effusions",
        "effusion",
        "effusions",
    ],
    "pneumothorax": [
        "pneumothorax",
        "pneumothoraces",
    ],
    "pulmonary edema": [
        "pulmonary edema",
        "edema",
    ],
    "atelectasis": [
        "atelectasis",
    ],
    "emphysema": [
        "emphysema",
    ],
    "interstitial abnormality": [
        "interstitial",
        "interstitial fibrosis",
        "interstitial markings",
    ],
    "pleural thickening": [
        "pleural thickening",
    ],
    "hyperinflation": [
        "hyperinflation",
        "hyperexpanded",
    ],
    "granulomatous disease": [
        "granulomatous disease",
        "calcified granuloma",
    ],
    "degenerative change": [
        "degenerative change",
        "degenerative changes",
    ],
    "pneumonia": [
        "right upper lobe pneumonia",
        "pneumonia",
    ],
    "infiltrate": [
        "infiltrates",
        "infiltrate",
    ],
    "fracture": [
        "displaced fracture",
        "fractures",
        "fracture",
    ],
}



NEGATION_PATTERNS = [
    "no ",
    "no evidence of ",
    "there is no ",
    "there are no ",
    "without ",
    "without evidence of ",
    "free of ",
    "not ",
]


POSITIVE_PATTERNS = [
    "evidence of ",
    "there is ",
    "there are ",
    "shows ",
    "show ",
    "demonstrates ",
    "demonstrate ",
    "appears ",
]


def split_sentences(text: str) -> list[str]:
    """
    Deterministic sentence splitting for the current IU X-Ray
    development dataset.

    The source dataset contains some formatting artifacts,
    therefore a lightweight splitter is used instead of relying
    on external NLP packages.
    """

    text = text.replace("!", ".")
    text = text.replace("?", ".")

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    return sentences


def find_negation_scopes(sentence: str) -> list[tuple[int, int]]:
    """
    Identify explicit negation scopes within a sentence.

    Returns:
        List of (start, end) character positions.

    The scope starts immediately after a negation cue and extends
    to the end of the sentence.

    This is intentionally conservative and designed for the
    controlled IU X-Ray reference text used in this experiment.
    """

    sentence_lower = sentence.lower()

    scopes = []

    for cue in NEGATION_PATTERNS:
        search_start = 0

        while True:
            position = sentence_lower.find(
                cue,
                search_start,
            )

            if position == -1:
                break

            scope_start = position + len(cue)

            scopes.append(
                (
                    scope_start,
                    len(sentence_lower),
                )
            )

            search_start = position + 1

    return scopes


def is_negated_by_sentence_scope(
    sentence: str,
    start: int,
) -> bool:
    """
    Determine whether a matched finding is inside an explicit
    negation scope.

    Examples:

        No pneumothorax.
        No pneumothorax or pleural effusion.
        No focal consolidation.
        No definite pleural effusion.
        No typical findings of pulmonary edema.
        Without focal airspace disease.
        Free of focal airspace disease.
    """

    scopes = find_negation_scopes(sentence)

    for scope_start, scope_end in scopes:

        if (
            scope_start <= start < scope_end
        ):
            return True

    return False


def detect_polarity(
    sentence: str,
    start: int,
) -> str:
    """
    Determine the polarity of a matched finding.

    Priority:

    1. Explicit negation scope -> NEGATED
    2. Explicit positive cue -> AFFIRMED
    3. Conservative default -> AFFIRMED

    Generic phrases such as 'lungs are clear' are NOT treated
    as blanket negations.
    """

    sentence_lower = sentence.lower()

    # ---------------------------------------------------------
    # Explicit negation
    # ---------------------------------------------------------

    if is_negated_by_sentence_scope(
        sentence,
        start,
    ):
        return "NEGATED"

    # ---------------------------------------------------------
    # Explicit positive evidence
    # ---------------------------------------------------------

    prefix = sentence_lower[:start].strip()

    for cue in POSITIVE_PATTERNS:

        if prefix.endswith(cue):
            return "AFFIRMED"

    # ---------------------------------------------------------
    # Conservative default
    # ---------------------------------------------------------

    return "AFFIRMED"


def extract_raw_claim_records(text: str) -> list[dict]:
    """
    Extract pattern matches from a single text blob.

    Only one occurrence of a finding per sentence is retained.
    """

    records = []

    sentences = split_sentences(text)

    for sentence in sentences:

        sentence_lower = sentence.lower()

        for finding, patterns in (
            REFERENCE_PATTERNS.items()
        ):

            for pattern in patterns:

                start = sentence_lower.find(
                    pattern
                )

                if start == -1:
                    continue

                polarity = detect_polarity(
                    sentence,
                    start,
                )

                records.append(
                    {
                        "finding": finding,
                        "polarity": polarity,
                        "matched_pattern": pattern,
                        "source_sentence": sentence,
                    }
                )

                # Only one occurrence of a finding
                # per sentence is retained.
                break

    return records


def strip_generated_report_body(report: str) -> str:
    """
    Remove the reproduced clinical indication / prompt prefix.

    The baseline model often copies the supplied prompt before
    'findings:'. That reproduced context must not be counted as
    a generated clinical claim.

    The split matches the existing baseline conflict audit.
    """

    report = str(report).lower().strip()

    if "findings :" in report:
        report = report.split("findings :", 1)[1]

    elif "findings:" in report:
        report = report.split("findings:", 1)[1]

    elif "impression :" in report:
        report = report.split("impression :", 1)[1]

    elif "impression:" in report:
        report = report.split("impression:", 1)[1]

    return report.strip()
