from pathlib import Path
import time

import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


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