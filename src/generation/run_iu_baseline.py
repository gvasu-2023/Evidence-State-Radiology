from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


MODEL_NAME = "nathansutton/generate-cxr"

DATASET_CSV = Path(
    "data/processed/iu_xray/metadata/dataset.csv"
)

OUTPUT_DIR = Path(
    "results/qualitative/iu_baseline"
)

NUM_CASES = 3


def main():

    print("=" * 70)
    print("IU X-RAY BASELINE GENERATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------

    df = pd.read_csv(DATASET_CSV)

    print(f"\nDataset cases available: {len(df)}")
    print(f"Running baseline on first {NUM_CASES} cases.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading processor...")

    processor = BlipProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.eval()

    # --------------------------------------------------
    # Process cases
    # --------------------------------------------------

    for _, row in df.head(NUM_CASES).iterrows():

        sample_id = row["sample_id"]

        image_path = Path(
            row["frontal_image"]
        )

        clinical_context = str(
            row["clinical_context"]
        )

        print("\n" + "=" * 70)
        print(f"CASE: {sample_id}")
        print("=" * 70)

        print(f"Image: {image_path}")
        print(f"Clinical context: {clinical_context}")

        if not image_path.exists():

            print(
                f"WARNING: image not found: {image_path}"
            )

            continue

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # --------------------------------------------------
        # Build prompt
        # --------------------------------------------------

        prompt = (
            "indication: "
            + clinical_context
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        print("Generating report...")

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
            )

        report = processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # --------------------------------------------------
        # Display
        # --------------------------------------------------

        print("\nGENERATED REPORT:")
        print(report)

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"{sample_id}.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                f"MODEL: {MODEL_NAME}\n"
            )

            f.write(
                f"SAMPLE_ID: {sample_id}\n"
            )

            f.write(
                f"IMAGE: {image_path}\n"
            )

            f.write(
                f"CLINICAL_CONTEXT: "
                f"{clinical_context}\n\n"
            )

            f.write(
                "REFERENCE_FINDINGS:\n"
            )

            f.write(
                str(row["findings"])
            )

            f.write("\n\n")

            f.write(
                "REFERENCE_IMPRESSION:\n"
            )

            f.write(
                str(row["impression"])
            )

            f.write("\n\n")

            f.write(
                "GENERATED_REPORT:\n"
            )

            f.write(report)

            f.write("\n")

        print(
            f"Saved: {output_file}"
        )

    print("\n" + "=" * 70)
    print("IU BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()