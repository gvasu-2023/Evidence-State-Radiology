from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "data/processed/iu_xray/metadata/dataset.csv"
)

OUTPUT_PATH = Path(
    "results/tables/reference_claim_candidates.csv"
)


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
        "airspace disease",
    ],
    "pleural effusion": [
        "pleural effusion",
        "pleural effusions",
    ],
    "pneumothorax": [
        "pneumothorax",
        "pneumothoraces",
    ],
    "pulmonary edema": [
        "pulmonary edema",
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


def extract_reference_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract candidate reference findings from findings +
    impression text.
    """

    records = []

    for _, row in df.iterrows():

        sample_id = str(
            row["sample_id"]
        )

        findings = str(
            row["findings"]
        ).lower()

        impression = str(
            row["impression"]
        ).lower()

        text = (
            findings
            + " "
            + impression
        )

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
                            "sample_id": sample_id,
                            "finding": finding,
                            "polarity": polarity,
                            "matched_pattern": pattern,
                        }
                    )

                    # Only one occurrence of a finding
                    # per sentence is retained.
                    break

    output = pd.DataFrame(records)

    if output.empty:

        return pd.DataFrame(
            columns=[
                "sample_id",
                "finding",
                "polarity",
                "matched_pattern",
            ]
        )

    output = (
        output
        .drop_duplicates(
            subset=[
                "sample_id",
                "finding",
                "polarity",
            ]
        )
        .sort_values(
            [
                "sample_id",
                "finding",
            ]
        )
        .reset_index(drop=True)
    )

    return output


def validate_known_cases(
    output: pd.DataFrame,
) -> None:
    """
    Validate known reference cases from the actual IU X-Ray
    development dataset.

    These checks prevent an incorrect polarity extractor from
    silently becoming the reference ground truth.
    """

    expected = {
        # IU_1
        ("IU_1", "consolidation"): "NEGATED",
        ("IU_1", "pleural effusion"): "NEGATED",
        ("IU_1", "pneumothorax"): "NEGATED",
        ("IU_1", "pulmonary edema"): "NEGATED",

        # IU_10
        ("IU_10", "airspace opacity"): "NEGATED",
        ("IU_10", "pleural effusion"): "NEGATED",
        ("IU_10", "pneumothorax"): "NEGATED",

        # IU_11
        ("IU_11", "pleural effusion"): "NEGATED",
        ("IU_11", "pneumothorax"): "NEGATED",

        # IU_12
        ("IU_12", "pleural effusion"): "NEGATED",
        ("IU_12", "pneumothorax"): "NEGATED",

        # IU_13
        ("IU_13", "cardiomegaly"): "AFFIRMED",
        ("IU_13", "pleural effusion"): "NEGATED",
        ("IU_13", "pneumothorax"): "NEGATED",

        # IU_14
        ("IU_14", "consolidation"): "NEGATED",
        ("IU_14", "hyperinflation"): "AFFIRMED",
        ("IU_14", "interstitial abnormality"): "AFFIRMED",
        ("IU_14", "pleural effusion"): "NEGATED",
        ("IU_14", "pulmonary edema"): "NEGATED",

        # IU_15
        ("IU_15", "granulomatous disease"): "AFFIRMED",
        ("IU_15", "pleural effusion"): "NEGATED",
        ("IU_15", "pneumothorax"): "NEGATED",

        # IU_17
        ("IU_17", "consolidation"): "NEGATED",
        ("IU_17", "pleural effusion"): "NEGATED",
        ("IU_17", "pneumothorax"): "NEGATED",

        # IU_18
        ("IU_18", "consolidation"): "NEGATED",
        ("IU_18", "pleural effusion"): "NEGATED",
        ("IU_18", "pneumothorax"): "NEGATED",
        ("IU_18", "pulmonary edema"): "NEGATED",

        # IU_19
        ("IU_19", "airspace opacity"): "NEGATED",
        ("IU_19", "degenerative change"): "AFFIRMED",
        ("IU_19", "pleural effusion"): "NEGATED",
        ("IU_19", "pneumothorax"): "NEGATED",

        # IU_20
        ("IU_20", "degenerative change"): "AFFIRMED",
        ("IU_20", "pneumothorax"): "NEGATED",

        # IU_22
        ("IU_22", "airspace opacity"): "NEGATED",
        ("IU_22", "pleural effusion"): "NEGATED",
        ("IU_22", "pneumothorax"): "NEGATED",

        # IU_23
        ("IU_23", "airspace opacity"): "NEGATED",
        ("IU_23", "pleural effusion"): "NEGATED",
        ("IU_23", "pneumothorax"): "NEGATED",

        # IU_4
        ("IU_4", "emphysema"): "AFFIRMED",
        ("IU_4", "interstitial abnormality"): "AFFIRMED",
        ("IU_4", "pleural effusion"): "NEGATED",
        ("IU_4", "pneumothorax"): "NEGATED",

        # IU_5
        ("IU_5", "consolidation"): "NEGATED",
        ("IU_5", "hyperinflation"): "AFFIRMED",
        ("IU_5", "pleural effusion"): "NEGATED",
        ("IU_5", "pleural thickening"): "AFFIRMED",
        ("IU_5", "pneumothorax"): "NEGATED",

        # IU_6
        ("IU_6", "consolidation"): "NEGATED",
        ("IU_6", "degenerative change"): "AFFIRMED",
        ("IU_6", "pleural effusion"): "NEGATED",
        ("IU_6", "pneumothorax"): "NEGATED",

        # IU_7
        ("IU_7", "atelectasis"): "AFFIRMED",
        ("IU_7", "consolidation"): "NEGATED",
        ("IU_7", "pleural effusion"): "NEGATED",

        # IU_8
        ("IU_8", "airspace opacity"): "NEGATED",
        ("IU_8", "pleural effusion"): "NEGATED",
        ("IU_8", "pneumothorax"): "NEGATED",

        # IU_9
        ("IU_9", "consolidation"): "NEGATED",
        ("IU_9", "granulomatous disease"): "AFFIRMED",
        ("IU_9", "pleural effusion"): "NEGATED",
        ("IU_9", "pneumothorax"): "NEGATED",
    }

    failures = []

    for key, expected_polarity in expected.items():

        sample_id, finding = key

        matches = output[
            (output["sample_id"] == sample_id)
            & (output["finding"] == finding)
        ]

        if matches.empty:

            failures.append(
                (
                    sample_id,
                    finding,
                    expected_polarity,
                    "MISSING",
                )
            )

            continue

        actual_polarities = set(
            matches["polarity"]
        )

        if expected_polarity not in actual_polarities:

            failures.append(
                (
                    sample_id,
                    finding,
                    expected_polarity,
                    ", ".join(
                        sorted(actual_polarities)
                    ),
                )
            )

    if failures:

        print("\nREFERENCE VALIDATION FAILED")
        print("=" * 70)

        for (
            sample_id,
            finding,
            expected,
            actual,
        ) in failures:

            print(
                f"{sample_id:8s} | "
                f"{finding:25s} | "
                f"expected={expected:8s} | "
                f"actual={actual}"
            )

        raise RuntimeError(
            f"Reference validation failed for "
            f"{len(failures)} cases."
        )

    print(
        "\nREFERENCE VALIDATION PASSED"
    )

    print(
        f"Validated expectations: {len(expected)}"
    )


def main() -> None:

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    output = extract_reference_candidates(
        df
    )

    # ---------------------------------------------------------
    # Validate before writing the reference CSV.
    # ---------------------------------------------------------

    validate_known_cases(
        output
    )

    # ---------------------------------------------------------
    # Save output.
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    print("=" * 70)
    print(
        "REFERENCE CLAIM CANDIDATE EXTRACTION"
    )
    print("=" * 70)

    print(
        f"Cases: "
        f"{df['sample_id'].nunique()}"
    )

    print(
        f"Candidate claims: "
        f"{len(output)}"
    )

    print(
        "\nPolarity counts:"
    )

    if output.empty:

        print(
            "No candidate claims found."
        )

    else:

        print(
            output["polarity"]
            .value_counts()
            .to_string()
        )

    print(
        "\nCandidates:"
    )

    if output.empty:

        print(
            "No candidate claims found."
        )

    else:

        print(
            output.to_string(
                index=False
            )
        )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()