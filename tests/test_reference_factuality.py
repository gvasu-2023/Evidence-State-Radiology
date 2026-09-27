from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.claim_extractor import (
    extract_generated_claim_candidates,
)
from src.evaluation.evaluate_reference_factuality import (
    build_factuality_table,
)
from src.evaluation.metrics import assign_factuality_label


ROOT = Path(__file__).resolve().parents[1]

GENERATION_PATH = (
    ROOT / "results/tables/baseline_generation_results.csv"
)
REFERENCE_PATH = (
    ROOT / "results/tables/reference_claim_candidates.csv"
)
EXTRACTED_CLAIMS_PATH = (
    ROOT / "results/tables/extracted_claims.csv"
)
CONFLICT_EVAL_PATH = (
    ROOT / "results/tables/conflict_claim_evaluation.csv"
)


def test_unsupported_affirmation_label():

    assert assign_factuality_label(
        "AFFIRMED",
        "NEGATED",
    ) == "unsupported_affirmation"


def test_ungrounded_affirmation_label():

    assert assign_factuality_label(
        "AFFIRMED",
        None,
    ) == "ungrounded_affirmation"


def test_omission_label():

    assert assign_factuality_label(
        "NOT_MENTIONED",
        "AFFIRMED",
    ) == "omission"

    assert assign_factuality_label(
        "NOT_MENTIONED",
        "NEGATED",
    ) == "omission"


def test_supported_and_contradiction_and_extra_negation():

    assert assign_factuality_label(
        "NEGATED",
        "NEGATED",
    ) == "supported"

    assert assign_factuality_label(
        "AFFIRMED",
        "AFFIRMED",
    ) == "supported"

    assert assign_factuality_label(
        "NEGATED",
        "AFFIRMED",
    ) == "contradiction"

    assert assign_factuality_label(
        "NEGATED",
        None,
    ) == "extra_negation"


def test_unclear_generated_polarity():

    assert assign_factuality_label(
        "MENTIONED_UNCLEAR",
        "NEGATED",
    ) == "unclear"


def test_condition_aware_comparison_keeps_pairs_separate():

    generation_df = pd.DataFrame(
        [
            {"sample_id": "IU_X", "condition": "sufficient"},
            {"sample_id": "IU_X", "condition": "irrelevant"},
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_X",
                "finding": "pneumothorax",
                "polarity": "NEGATED",
                "matched_pattern": "pneumothorax",
            }
        ]
    )

    generated_claims_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_X",
                "condition": "sufficient",
                "finding": "pneumothorax",
                "polarity": "NEGATED",
            },
            {
                "sample_id": "IU_X",
                "condition": "irrelevant",
                "finding": "cardiomegaly",
                "polarity": "AFFIRMED",
            },
        ]
    )

    detail = build_factuality_table(
        generation_df,
        reference_df,
        generated_claims_df,
    )

    sufficient = detail[
        (detail["condition"] == "sufficient")
        & (detail["finding"] == "pneumothorax")
    ].iloc[0]

    irrelevant_ptx = detail[
        (detail["condition"] == "irrelevant")
        & (detail["finding"] == "pneumothorax")
    ].iloc[0]

    irrelevant_cardio = detail[
        (detail["condition"] == "irrelevant")
        & (detail["finding"] == "cardiomegaly")
    ].iloc[0]

    assert sufficient["label"] == "supported"
    assert irrelevant_ptx["label"] == "omission"
    assert (
        irrelevant_cardio["label"]
        == "ungrounded_affirmation"
    )


def test_unsupported_affirmation_in_table():

    generation_df = pd.DataFrame(
        [
            {"sample_id": "IU_X", "condition": "sufficient"},
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_X",
                "finding": "pleural effusion",
                "polarity": "NEGATED",
                "matched_pattern": "pleural effusion",
            }
        ]
    )

    generated_claims_df = pd.DataFrame(
        [
            {
                "sample_id": "IU_X",
                "condition": "sufficient",
                "finding": "pleural effusion",
                "polarity": "AFFIRMED",
            }
        ]
    )

    detail = build_factuality_table(
        generation_df,
        reference_df,
        generated_claims_df,
    )

    assert detail.iloc[0]["label"] == "unsupported_affirmation"


@pytest.mark.skipif(
    not GENERATION_PATH.exists()
    or not REFERENCE_PATH.exists(),
    reason="Required evaluation tables are not present.",
)
def test_all_generation_records_are_represented():

    generation_df = pd.read_csv(GENERATION_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    assert len(generation_df) == 81
    assert len(reference_df) == 73


    generated_claims_df = extract_generated_claim_candidates(
        generation_df
    )

    detail = build_factuality_table(
        generation_df,
        reference_df,
        generated_claims_df,
    )

    pairs = set(
        zip(
            detail["sample_id"].astype(str),
            detail["condition"].astype(str),
        )
    )

    expected = set(
        zip(
            generation_df["sample_id"].astype(str),
            generation_df["condition"].astype(str),
        )
    )

    assert pairs == expected
    assert len(expected) == 81
    assert ("IU_14", "irrelevant") in pairs
    assert ("IU_19", "irrelevant") in pairs


@pytest.mark.skipif(
    not EXTRACTED_CLAIMS_PATH.exists()
    or not CONFLICT_EVAL_PATH.exists(),
    reason="Phase 6 artifacts are not present.",
)
def test_phase6_artifacts_remain_intact():

    extracted = pd.read_csv(EXTRACTED_CLAIMS_PATH)
    conflict = pd.read_csv(CONFLICT_EVAL_PATH)

    assert len(extracted) == 230
    assert len(conflict) == 15

    counts = (
        conflict["evaluation"]
        .value_counts()
        .to_dict()
    )

    assert counts.get("NEGATED", 0) == 12
    assert counts.get("NOT_MENTIONED", 0) == 3
