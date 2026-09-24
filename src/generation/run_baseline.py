from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


MODEL_NAME = "nathansutton/generate-cxr"
IMAGE_PATH = Path("data/samples/images/CXR_001.png")


def main():

    print("=" * 70)
    print("EVIDENCE-STATE RADIOLOGY PROJECT")
    print("BASELINE VLM INFERENCE")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Check image
    # --------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    print(f"\nImage: {IMAGE_PATH}")
    print(
        f"Image size: "
        f"{IMAGE_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # --------------------------------------------------
    # 2. Load processor
    # --------------------------------------------------

    print("\nLoading processor...")

    processor = BlipProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------
    # 3. Load model
    # --------------------------------------------------

    print("Loading model...")

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.eval()

    # --------------------------------------------------
    # 4. Load image
    # --------------------------------------------------

    print("Loading image...")

    image = Image.open(IMAGE_PATH).convert("RGB")

    print(f"Image dimensions: {image.size}")

    # --------------------------------------------------
    # 5. Clinical prompt
    # --------------------------------------------------

    prompt = "indication: chest x-ray"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    # --------------------------------------------------
    # 6. Generate report
    # --------------------------------------------------

    print("\nRunning VLM inference...")
    print("This may take some time on CPU.")

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
        )

    # --------------------------------------------------
    # 7. Decode generated report
    # --------------------------------------------------

    report = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    # --------------------------------------------------
    # 8. Display result
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE VLM REPORT")
    print("=" * 70)

    print(report)

    print("=" * 70)

    # --------------------------------------------------
    # 9. Save result
    # --------------------------------------------------

    output_dir = Path("results/qualitative")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "baseline_CXR_001.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"MODEL: {MODEL_NAME}\n"
        )

        f.write(
            f"IMAGE: {IMAGE_PATH.name}\n"
        )

        f.write(
            f"PROMPT: {prompt}\n\n"
        )

        f.write(
            "GENERATED REPORT:\n\n"
        )

        f.write(report)

        f.write("\n")

    print(
        f"\nSaved result to: {output_file}"
    )


if __name__ == "__main__":
    main()