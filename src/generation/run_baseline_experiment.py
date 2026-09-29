from pathlib import Path
import hashlib
import os
import sys
import time

import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.generation.baseline import BaselineGenerator, GenerationInput, MODEL_NAME as PHASE17_MODEL_NAME
from src.preprocessing.prepare_phase17_vlm_eligibility import (
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_SELECTION_SHA256,
    EXCLUDED_UIDS,
    inspect_image,
    sha256_file,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "nathansutton/generate-cxr"

PERTURBATION_CSV = Path(
    "data/processed/iu_xray/perturbations/perturbations.csv"
)

IMAGE_DIR = Path(
    "data/processed/iu_xray/images"
)

OUTPUT_DIR = Path(
    "results/baseline"
)

RESULTS_CSV = Path(
    "results/tables/baseline_generation_results.csv"
)

MAX_NEW_TOKENS = 128

PHASE17_ROOT = Path(__file__).resolve().parents[2]
PHASE17_BENCHMARK = PHASE17_ROOT / "results/tables/phase16_perturbations.csv"
PHASE17_SELECTION = PHASE17_ROOT / "results/tables/phase16e_selected_studies.csv"
PHASE17_ELIGIBILITY = PHASE17_ROOT / "results/tables/phase17_vlm_inference_eligibility.csv"
PHASE17_OUTPUT_DIR = PHASE17_ROOT / "results/baseline/phase17_full"
PHASE17_RESULTS_CSV = PHASE17_OUTPUT_DIR / "baseline_results.csv"
PHASE17_CONDITIONS = {
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
}
PHASE17_RESULT_COLUMNS = [
    "sample_id", "uid", "condition", "image_path", "perturbed_context",
    "generated_report", "nonempty", "report_length", "runtime_seconds",
    "status", "failure_reason", "output_file", "prompt", "model_name",
]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _normal_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def _load_phase17_cohort():
    """Join the eligibility manifest to benchmark contexts without editing either input."""
    benchmark_hash = _sha256(PHASE17_BENCHMARK)
    selection_hash = _sha256(PHASE17_SELECTION)
    eligibility_hash = _sha256(PHASE17_ELIGIBILITY)
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256:
        raise ValueError(f"Frozen benchmark hash mismatch: {benchmark_hash}")
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(f"Frozen selection hash mismatch: {selection_hash}")

    benchmark = pd.read_csv(PHASE17_BENCHMARK)
    eligibility = pd.read_csv(PHASE17_ELIGIBILITY)
    if len(benchmark) != 600 or len(eligibility) != 600:
        raise ValueError("Expected exactly 600 frozen benchmark and eligibility rows")
    eligible = eligibility.loc[eligibility["eligible_for_vlm"].astype(bool)].copy()
    excluded = eligibility.loc[~eligibility["eligible_for_vlm"].astype(bool)].copy()
    if len(eligible) != 576 or len(excluded) != 24:
        raise ValueError(f"Expected 576 eligible and 24 excluded; got {len(eligible)} and {len(excluded)}")
    if eligible["condition"].value_counts().to_dict() != {condition: 96 for condition in PHASE17_CONDITIONS}:
        raise ValueError("Eligible cohort must contain 96 records per condition")
    if set(excluded["uid"].astype(int)) != EXCLUDED_UIDS or excluded.groupby("uid").size().to_dict() != {uid: 6 for uid in EXCLUDED_UIDS}:
        raise ValueError("Eligibility exclusions do not match the four approved image-unavailable UIDs")
    if eligible["uid"].astype(int).isin(EXCLUDED_UIDS).any():
        raise ValueError("An excluded UID appears in the VLM inference cohort")
    keys = ["sample_id", "uid", "condition"]
    if benchmark.duplicated(keys).any() or eligibility.duplicated(keys).any():
        raise ValueError("Duplicate benchmark keys found")

    joined = eligible.merge(
        benchmark[keys + ["frontal_image", "lateral_image", "perturbed_context"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_eligibility", "_benchmark"),
        indicator=True,
    )
    if len(joined) != 576 or not joined["_merge"].eq("both").all():
        raise ValueError("Eligibility rows could not be matched one-to-one to frozen benchmark rows")
    if not joined["image_path"].astype(str).eq(joined["frontal_image"].astype(str)).all():
        raise ValueError("Eligible manifest paths must match the frozen benchmark frontal_image references")
    if joined["image_path"].astype(str).str.lower().str.contains("_lateral.").any():
        raise ValueError("An eligible record references a lateral image")
    for relative_path in joined["image_path"].astype(str).drop_duplicates():
        image_path = PHASE17_ROOT / Path(relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Eligible frontal image is missing: {image_path}")
        image_check = inspect_image(image_path)
        if not image_check["integrity_ok"]:
            raise ValueError(f"Eligible frontal image failed integrity checks: {image_path}")

    # Use the benchmark's exact frontal reference where populated; for the four
    # blank references the manifest records the expected path solely for auditing.
    joined["perturbed_context"] = joined["perturbed_context"].map(_normal_text)
    joined["image_path"] = joined["image_path"].astype(str)
    joined = joined.sort_values(["sample_id", "condition"], kind="stable").reset_index(drop=True)
    hashes = {
        "benchmark": benchmark_hash,
        "selection": selection_hash,
        "eligibility": eligibility_hash,
    }
    return joined, hashes


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_results(results: pd.DataFrame) -> None:
    PHASE17_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = PHASE17_RESULTS_CSV.with_suffix(".csv.tmp")
    results.reindex(columns=PHASE17_RESULT_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, PHASE17_RESULTS_CSV)


def _read_existing_phase17_results() -> pd.DataFrame:
    if not PHASE17_RESULTS_CSV.exists():
        return pd.DataFrame(columns=PHASE17_RESULT_COLUMNS)
    existing = pd.read_csv(PHASE17_RESULTS_CSV, keep_default_na=False)
    missing = set(PHASE17_RESULT_COLUMNS) - set(existing.columns)
    if missing:
        raise ValueError(f"Existing results CSV is missing columns: {sorted(missing)}")
    if existing.duplicated(["sample_id", "condition"]).any():
        raise ValueError("Existing results CSV has duplicate sample_id/condition rows")
    eligible = pd.read_csv(PHASE17_ELIGIBILITY)
    allowed = set(
        zip(
            eligible.loc[eligible["eligible_for_vlm"].astype(bool), "sample_id"].astype(str),
            eligible.loc[eligible["eligible_for_vlm"].astype(bool), "condition"].astype(str),
        )
    )
    actual = set(zip(existing["sample_id"].astype(str), existing["condition"].astype(str)))
    if actual - allowed:
        raise ValueError("Existing baseline results contain records outside the approved eligibility cohort")
    return existing.reindex(columns=PHASE17_RESULT_COLUMNS)


def _is_valid_completed_result(existing: pd.DataFrame, row, output_dir: Path) -> bool:
    matches = existing.loc[
        existing["sample_id"].astype(str).eq(str(row.sample_id))
        & existing["condition"].astype(str).eq(str(row.condition))
    ]
    if len(matches) != 1:
        return False
    result = matches.iloc[0]
    if result["status"] != "success" or not str(result["generated_report"]).strip():
        return False
    if str(result["uid"]) != str(row.uid):
        return False
    if str(result["image_path"]) != str(row.image_path):
        return False
    if str(result["perturbed_context"]) != str(row.perturbed_context):
        return False
    try:
        output_file = Path(str(result["output_file"]))
        if not output_file.is_absolute():
            output_file = output_dir / output_file
        content = output_file.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    expected_report = str(result["generated_report"])
    return (
        f"SAMPLE_ID: {row.sample_id}\n" in content
        and f"UID: {row.uid}\n" in content
        and f"CONDITION: {row.condition}\n" in content
        and f"IMAGE: {row.image_path}\n" in content
        and f"CLINICAL_CONTEXT: {row.perturbed_context}\n" in content
        and f"GENERATED_REPORT:\n\n{expected_report}\n" in content
    )


def run_phase17_full() -> None:
    """Run or resume the 576-record image-backed Phase 17 baseline cohort."""
    cohort, input_hashes = _load_phase17_cohort()
    condition_counts = cohort["condition"].value_counts().sort_index().to_dict()
    print("Phase 17B baseline cohort preflight")
    print(f"eligible records: {len(cohort)}; studies: {cohort['uid'].nunique()}")
    print(f"condition counts: {condition_counts}")
    print(f"excluded UIDs: {sorted(EXCLUDED_UIDS)} (24 records)")
    print(f"benchmark SHA-256: {input_hashes['benchmark']}")
    print(f"selection SHA-256: {input_hashes['selection']}")
    print(f"eligibility SHA-256: {input_hashes['eligibility']}")
    print(f"output directory: {PHASE17_OUTPUT_DIR}")

    # Model and processor are loaded exactly once for this invocation.
    generator = BaselineGenerator()
    model_device = next(generator.model.parameters()).device
    if model_device.type != "cpu":
        raise RuntimeError(f"Phase 17B is CPU-only, but the baseline model loaded on {model_device}")
    print(f"model: {generator.model_name}; device: {model_device}; max_new_tokens: 128")
    results = _read_existing_phase17_results()
    resumed = sum(_is_valid_completed_result(results, row, PHASE17_OUTPUT_DIR) for row in cohort.itertuples(index=False))
    print(f"valid completed records to resume/skip: {resumed}")

    started = time.perf_counter()
    generated_count = 0
    skipped_count = 0
    failed_uids: list[int] = []

    for index, row in enumerate(cohort.itertuples(index=False), start=1):
        if _is_valid_completed_result(results, row, PHASE17_OUTPUT_DIR):
            skipped_count += 1
        else:
            output_path = PHASE17_OUTPUT_DIR / str(row.condition) / f"{row.sample_id}.txt"
            image_path = PHASE17_ROOT / Path(str(row.image_path))
            record_started = time.perf_counter()
            prompt = ""
            try:
                generated = generator.generate(
                    GenerationInput(image_path=image_path, clinical_context=row.perturbed_context)
                )
                report = str(generated.report)
                runtime = time.perf_counter() - record_started
                if not report.strip():
                    raise ValueError("empty_generated_report")
                # Match the baseline prompt construction for an auditable text record.
                prompt = (
                    f"Clinical indication: {row.perturbed_context.strip()} "
                    "Generate a radiology report for this chest X-ray."
                    if row.perturbed_context.strip()
                    else "Generate a radiology report for this chest X-ray."
                )
                text_output = (
                    f"MODEL: {generated.model_name}\n"
                    f"SAMPLE_ID: {row.sample_id}\n"
                    f"UID: {row.uid}\n"
                    f"CONDITION: {row.condition}\n"
                    f"IMAGE: {row.image_path}\n"
                    f"CLINICAL_CONTEXT: {row.perturbed_context}\n"
                    f"PROMPT: {prompt}\n\n"
                    f"GENERATED_REPORT:\n\n{report}\n"
                    f"INFERENCE_TIME_SECONDS: {runtime:.6f}\n"
                )
                _atomic_write_text(output_path, text_output)
                result = {
                    "sample_id": row.sample_id,
                    "uid": int(row.uid),
                    "condition": row.condition,
                    "image_path": row.image_path,
                    "perturbed_context": row.perturbed_context,
                    "generated_report": report,
                    "nonempty": True,
                    "report_length": len(report),
                    "runtime_seconds": runtime,
                    "status": "success",
                    "failure_reason": "",
                    "output_file": str(output_path.relative_to(PHASE17_OUTPUT_DIR)),
                    "prompt": prompt,
                    "model_name": generated.model_name,
                }
                key_mask = results["sample_id"].astype(str).eq(str(row.sample_id)) & results["condition"].astype(str).eq(str(row.condition))
                results = pd.concat([results.loc[~key_mask], pd.DataFrame([result])], ignore_index=True)
                _atomic_write_results(results)
                # The same checks used for resume must pass immediately after writing.
                if not _is_valid_completed_result(results, row, PHASE17_OUTPUT_DIR):
                    raise IOError(f"Saved output failed readback validation for UID {row.uid}")
                generated_count += 1
            except Exception as error:
                runtime = time.perf_counter() - record_started
                failed_uids.append(int(row.uid))
                failure = {
                    "sample_id": row.sample_id,
                    "uid": int(row.uid),
                    "condition": row.condition,
                    "image_path": row.image_path,
                    "perturbed_context": row.perturbed_context,
                    "generated_report": "",
                    "nonempty": False,
                    "report_length": 0,
                    "runtime_seconds": runtime,
                    "status": "failed",
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "output_file": "",
                    "prompt": prompt,
                    "model_name": PHASE17_MODEL_NAME,
                }
                key_mask = results["sample_id"].astype(str).eq(str(row.sample_id)) & results["condition"].astype(str).eq(str(row.condition))
                results = pd.concat([results.loc[~key_mask], pd.DataFrame([failure])], ignore_index=True)
                _atomic_write_results(results)
                raise RuntimeError(
                    f"Stopped after inference failure at UID {row.uid}, condition {row.condition}; "
                    "the failed record was persisted and no later record was processed."
                ) from error

        elapsed = time.perf_counter() - started
        completed = generated_count + skipped_count
        average = elapsed / generated_count if generated_count else 0.0
        remaining = max(0, len(cohort) - index)
        eta = average * remaining
        if index % 10 == 0 or index == len(cohort):
            print(
                f"progress {index}/{len(cohort)}; completed={completed}; skipped={skipped_count}; "
                f"failed={len(failed_uids)}; elapsed={elapsed:.1f}s; "
                f"avg_generated={average:.2f}s/record; ETA={eta:.1f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        f"Phase 17B complete: successful_new={generated_count}; skipped/resumed={skipped_count}; "
        f"failed={len(failed_uids)}; elapsed={elapsed:.1f}s; "
        f"average_new_generation={elapsed / generated_count if generated_count else 0:.2f}s"
    )
    if failed_uids:
        print(f"failed UIDs: {failed_uids}")

    if _sha256(PHASE17_BENCHMARK) != input_hashes["benchmark"]:
        raise RuntimeError("Frozen benchmark changed during inference")
    if _sha256(PHASE17_SELECTION) != input_hashes["selection"]:
        raise RuntimeError("Frozen selection changed during inference")
    if _sha256(PHASE17_ELIGIBILITY) != input_hashes["eligibility"]:
        raise RuntimeError("Eligibility artifact changed during inference")

    final_results = _read_existing_phase17_results()
    if len(final_results) != 576:
        raise RuntimeError(f"Expected exactly one result per eligible record; found {len(final_results)}")
    if final_results.duplicated(["sample_id", "condition"]).any():
        raise RuntimeError("Final result table contains duplicate benchmark keys")
    if set(final_results["uid"].astype(int)) & EXCLUDED_UIDS:
        raise RuntimeError("Final result table contains an excluded UID")
    if not final_results["status"].eq("success").all() or not final_results["nonempty"].astype(bool).all():
        raise RuntimeError("Final result table contains a failure or empty report")
    if final_results["condition"].value_counts().to_dict() != {condition: 96 for condition in PHASE17_CONDITIONS}:
        raise RuntimeError("Final results are not balanced across six conditions")


# ============================================================
# Model
# ============================================================

class CXRGenerator:

    def __init__(self):

        print("\nLoading processor...")

        self.processor = BlipProcessor.from_pretrained(
            MODEL_NAME
        )

        print("Loading model...")

        self.model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

        self.model.eval()

        print("Model loaded.")


    def generate(
        self,
        image_path: Path,
        clinical_context: str,
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        if clinical_context.strip():

            prompt = (
                "Clinical indication: "
                f"{clinical_context.strip()} "
                "Generate a radiology report for this chest X-ray."
            )

        else:

            prompt = (
                "Generate a radiology report "
                "for this chest X-ray."
            )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        start_time = time.perf_counter()

        with torch.no_grad():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        inference_time = (
            time.perf_counter() - start_time
        )

        report = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return (
            prompt,
            report,
            inference_time,
        )


# ============================================================
# Main experiment
# ============================================================

def main():

    if sys.argv[1:] == ["--phase17-full"]:
        run_phase17_full()
        return

    print("=" * 70)
    print("EVIDENCE-STATE RADIOLOGY")
    print("BASELINE CONTROLLED PERTURBATION EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not PERTURBATION_CSV.exists():

        raise FileNotFoundError(
            f"Perturbation dataset not found: "
            f"{PERTURBATION_CSV}"
        )

    df = pd.read_csv(
        PERTURBATION_CSV
    )

    print(
        f"\nPerturbation records: {len(df)}"
    )

    print("\nCondition counts:")

    print(
        df["condition"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Prepare output directories
    # --------------------------------------------------------

    for condition in df["condition"].unique():

        (
            OUTPUT_DIR / condition
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    generator = CXRGenerator()

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    completed_keys = set()

    if RESULTS_CSV.exists():

        existing = pd.read_csv(
            RESULTS_CSV
        )

        for _, row in existing.iterrows():

            completed_keys.add(
                (
                    row["sample_id"],
                    row["condition"],
                )
            )

        print(
            f"\nExisting completed records: "
            f"{len(completed_keys)}"
        )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    results = []

    for index, row in df.iterrows():

        sample_id = row["sample_id"]
        condition = row["condition"]

        key = (
            sample_id,
            condition,
        )

        if key in completed_keys:

            print(
                f"\n[{index + 1}/{len(df)}] "
                f"SKIPPING {sample_id} / {condition}"
            )

            continue

        print("\n" + "=" * 70)

        print(
            f"[{index + 1}/{len(df)}] "
            f"{sample_id} / {condition}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_name = row["image_path"]

        image_path = Path(
            image_name
        )

        if not image_path.is_absolute():

            image_path = Path(
                image_name
            )

            if not image_path.exists():

                image_path = (
                    IMAGE_DIR
                    / Path(image_name).name
                )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found for "
                f"{sample_id}: {image_path}"
            )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = row["perturbed_context"]

        if pd.isna(context):

            context = ""

        context = str(context)

        print(
            f"Context: "
            f"{context if context else '[EMPTY]'}"
        )

        print(
            f"Image: {image_path}"
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        prompt, report, inference_time = (
            generator.generate(
                image_path=image_path,
                clinical_context=context,
            )
        )

        print(
            f"\nInference time: "
            f"{inference_time:.2f} seconds"
        )

        print("\nGenerated report:")

        print(report)

        # ----------------------------------------------------
        # Save individual result
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / condition
            / f"{sample_id}.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                f"MODEL: {MODEL_NAME}\n"
            )

            file.write(
                f"SAMPLE_ID: {sample_id}\n"
            )

            file.write(
                f"CONDITION: {condition}\n"
            )

            file.write(
                f"IMAGE: {image_path}\n"
            )

            file.write(
                f"CLINICAL_CONTEXT: "
                f"{context}\n"
            )

            file.write(
                f"PROMPT: {prompt}\n\n"
            )

            file.write(
                "GENERATED_REPORT:\n\n"
            )

            file.write(report)

            file.write("\n")

            file.write(
                f"\nINFERENCE_TIME_SECONDS: "
                f"{inference_time:.4f}\n"
            )

        # ----------------------------------------------------
        # Structured result
        # ----------------------------------------------------

        result = {

            "sample_id":
                sample_id,

            "condition":
                condition,

            "image_path":
                str(image_path),

            "clinical_context":
                context,

            "reference_findings":
                row["reference_findings"],

            "reference_impression":
                row["reference_impression"],

            "prompt":
                prompt,

            "generated_report":
                report,

            "model_name":
                MODEL_NAME,

            "inference_time_seconds":
                inference_time,

        }

        results.append(result)

        # ----------------------------------------------------
        # Incremental save
        # ----------------------------------------------------

        new_results = pd.DataFrame(
            results
        )

        if RESULTS_CSV.exists():

            existing = pd.read_csv(
                RESULTS_CSV
            )

            combined = pd.concat(
                [
                    existing,
                    new_results,
                ],
                ignore_index=True,
            )

            combined = combined.drop_duplicates(
                subset=[
                    "sample_id",
                    "condition",
                ],
                keep="last",
            )

        else:

            combined = new_results

        combined.to_csv(
            RESULTS_CSV,
            index=False,
        )

        print(
            f"\nSaved: {output_file}"
        )

        print(
            f"Progress: "
            f"{len(combined)}/{len(df)}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENT COMPLETED")
    print("=" * 70)

    if RESULTS_CSV.exists():

        final_results = pd.read_csv(
            RESULTS_CSV
        )

        print(
            f"\nTotal results: "
            f"{len(final_results)}"
        )

        print("\nResults by condition:")

        print(
            final_results[
                "condition"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            f"\nResults CSV:"
            f"\n{RESULTS_CSV}"
        )


if __name__ == "__main__":
    main()
