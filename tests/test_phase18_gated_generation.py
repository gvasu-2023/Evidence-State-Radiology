import pandas as pd

from src.gating.run_phase18_gated_experiment import (
    ABSTENTION_REPORT,
    BENCHMARK_PATH,
    ELIGIBILITY_PATH,
    REQUIRED_CONDITIONS,
    _apply_existing_policy,
    build_phase18_results,
    sha256,
)


def test_phase18_builds_exact_eligible_cohort_without_inference():
    results = build_phase18_results()

    assert len(results) == 576
    assert not results.duplicated(["sample_id", "uid", "condition"]).any()
    assert set(results["condition"]) == REQUIRED_CONDITIONS
    assert results["condition"].value_counts().to_dict() == {
        condition: 96 for condition in REQUIRED_CONDITIONS
    }
    assert not set(results["uid"].astype(int)) & {74, 597, 803, 885}
    assert not results["was_model_inference_run"].any()
    assert results["evidence_state"].isin(
        {"sufficient", "incomplete", "irrelevant", "conflicting", "insufficient"}
    ).all()
    assert results["gate_action"].isin(
        {"generate", "qualify", "discount_context", "qualify_or_abstain", "abstain"}
    ).all()


def test_phase18_preserves_phase13b_baseline_reuse_policies():
    lookup = {("IU_1", "insufficient"): "image-only report"}
    assert _apply_existing_policy("generate", "IU_1", "sufficient", "baseline", lookup) == (
        "baseline",
        "sufficient",
    )
    assert _apply_existing_policy(
        "discount_context", "IU_1", "irrelevant", "context report", lookup
    ) == ("image-only report", "insufficient")
    assert _apply_existing_policy("qualify", "IU_1", "syntactic_incomplete", "x", lookup) == (
        "[QUALIFIED: Incomplete clinical context] image-only report",
        "insufficient",
    )
    assert _apply_existing_policy(
        "qualify_or_abstain", "IU_1", "conflicting", "baseline", lookup
    ) == ("[WARNING: Clinical context conflict detected] baseline", "conflicting")
    assert _apply_existing_policy("abstain", "IU_1", "insufficient", "x", lookup) == (
        ABSTENTION_REPORT,
        "none",
    )


def test_phase18_does_not_modify_frozen_cohort_inputs():
    assert sha256(BENCHMARK_PATH) == "3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53"
    assert sha256(ELIGIBILITY_PATH) == "cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6"
