import pandas as pd

from src.evaluation.claim_extractor import (
    extract_generated_claim_candidates,
)
from src.evaluation.claim_patterns import (
    strip_generated_report_body,
)


def test_prompt_prefix_is_not_counted_as_generated_claim():

    report = (
        "clinical indication : pleural effusion is present. "
        "generate a radiology report for this chest x - ray. "
        "findings : the lungs are clear without consolidation. "
        "the heart is normal in size."
    )

    body = strip_generated_report_body(report)

    assert "pleural effusion is present" not in body
    assert "without consolidation" in body

    generation_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_TEST",
                "condition": "conflicting",
                "generated_report": report,
            }
        ]
    )

    claims = extract_generated_claim_candidates(
        generation_df
    )

    findings = set(claims["finding"].tolist())

    assert "pleural effusion" not in findings
    assert "consolidation" in findings

    consolidation = claims[
        claims["finding"] == "consolidation"
    ].iloc[0]

    assert consolidation["polarity"] == "NEGATED"


def test_findings_and_impression_duplicates_collapse():

    report = (
        "findings : there is no pneumothorax. "
        "impression : no pneumothorax."
    )

    generation_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_TEST",
                "condition": "sufficient",
                "generated_report": report,
            }
        ]
    )

    claims = extract_generated_claim_candidates(
        generation_df
    )

    pneumothorax = claims[
        claims["finding"] == "pneumothorax"
    ]

    assert len(pneumothorax) == 1
    assert pneumothorax.iloc[0]["polarity"] == "NEGATED"


def test_conflicting_polarities_are_unclear():

    report = (
        "findings : there is consolidation. "
        "impression : no consolidation is seen."
    )

    generation_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_TEST",
                "condition": "sufficient",
                "generated_report": report,
            }
        ]
    )

    claims = extract_generated_claim_candidates(
        generation_df
    )

    consolidation = claims[
        claims["finding"] == "consolidation"
    ]

    assert len(consolidation) == 1
    assert (
        consolidation.iloc[0]["polarity"]
        == "MENTIONED_UNCLEAR"
    )


def test_conditions_remain_separate_for_same_sample():

    generation_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_TEST",
                "condition": "sufficient",
                "generated_report": (
                    "findings : there is no pneumothorax."
                ),
            },
            {
                "sample_id": "IU_TEST",
                "condition": "conflicting",
                "generated_report": (
                    "findings : there is pneumothorax."
                ),
            },
        ]
    )

    claims = extract_generated_claim_candidates(
        generation_df
    )

    sufficient = claims[
        claims["condition"] == "sufficient"
    ].iloc[0]

    conflicting = claims[
        claims["condition"] == "conflicting"
    ].iloc[0]

    assert sufficient["polarity"] == "NEGATED"
    assert conflicting["polarity"] == "AFFIRMED"
