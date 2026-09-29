from types import SimpleNamespace

import pandas as pd

from src.generation.run_baseline_experiment import (
    PHASE17_RESULT_COLUMNS,
    _is_valid_completed_result,
    _load_phase17_cohort,
)
from src.preprocessing.prepare_phase17_vlm_eligibility import EXCLUDED_UIDS


def test_phase17_runner_loads_only_the_576_eligible_rows_with_benchmark_context():
    cohort, hashes = _load_phase17_cohort()
    benchmark = pd.read_csv("results/tables/phase16_perturbations.csv")
    assert len(cohort) == 576
    assert cohort.uid.nunique() == 96
    assert not set(cohort.uid.astype(int)) & EXCLUDED_UIDS
    assert cohort.condition.value_counts().to_dict() == {
        "sufficient": 96,
        "syntactic_incomplete": 96,
        "evidentiary_incomplete": 96,
        "irrelevant": 96,
        "conflicting": 96,
        "insufficient": 96,
    }
    expected_contexts = benchmark.set_index(["sample_id", "uid", "condition"])["perturbed_context"]
    for row in cohort.itertuples(index=False):
        expected = expected_contexts.loc[(row.sample_id, row.uid, row.condition)]
        expected = "" if pd.isna(expected) else str(expected)
        assert row.perturbed_context == expected
    assert set(hashes) == {"benchmark", "selection", "eligibility"}


def test_resume_skips_only_a_matching_nonempty_result_with_readable_text(tmp_path):
    row = SimpleNamespace(
        sample_id="IU_32",
        uid=32,
        condition="sufficient",
        image_path="external_data/IU-Xray/images/IU_32_frontal.jpg",
        perturbed_context="Bilateral rib pain and shortness of breath.",
    )
    report = "A sample generated report."
    output_path = tmp_path / "sufficient" / "IU_32.txt"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        "MODEL: nathansutton/generate-cxr\n"
        "SAMPLE_ID: IU_32\nUID: 32\nCONDITION: sufficient\n"
        "IMAGE: external_data/IU-Xray/images/IU_32_frontal.jpg\n"
        "CLINICAL_CONTEXT: Bilateral rib pain and shortness of breath.\n"
        "GENERATED_REPORT:\n\nA sample generated report.\n",
        encoding="utf-8",
    )
    result = {
        "sample_id": "IU_32",
        "uid": 32,
        "condition": "sufficient",
        "image_path": row.image_path,
        "perturbed_context": row.perturbed_context,
        "generated_report": report,
        "nonempty": True,
        "report_length": len(report),
        "runtime_seconds": 1.0,
        "status": "success",
        "failure_reason": "",
        "output_file": "sufficient/IU_32.txt",
        "prompt": "",
        "model_name": "nathansutton/generate-cxr",
    }
    existing = pd.DataFrame([result], columns=PHASE17_RESULT_COLUMNS)
    assert _is_valid_completed_result(existing, row, tmp_path)
    existing.loc[0, "perturbed_context"] = "wrong context"
    assert not _is_valid_completed_result(existing, row, tmp_path)
