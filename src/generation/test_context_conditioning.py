from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


MODEL_NAME = "nathansutton/generate-cxr"
IMAGE_PATH = Path("data/processed/iu_xray/images/IU_1_frontal.jpg")


def generate_report(model, processor, image, prompt):
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
        )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


def main():

    print("=" * 70)
    print("CLINICAL CONTEXT CONDITIONING TEST")
    print("=" * 70)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    print(f"Image: {IMAGE_PATH}")

    print("\nLoading processor...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    prompts = {
        "image_only":
            "Generate a radiology report for this chest X-ray.",

        "normal_context":
            (
                "Clinical indication: "
                "Preoperative evaluation. "
                "Generate a radiology report for this chest X-ray."
            ),

        "abnormal_context":
            (
                "Clinical indication: "
                "Pleural effusion is present. "
                "Generate a radiology report for this chest X-ray."
            ),

        "conflicting_context":
            (
                "Clinical indication: "
                "Pneumothorax is present. "
                "Generate a radiology report for this chest X-ray."
            ),
    }

    output_dir = Path("results/qualitative/context_conditioning")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for name, prompt in prompts.items():

        print("\n" + "-" * 70)
        print(f"CONDITION: {name}")
        print(f"PROMPT: {prompt}")
        print("-" * 70)

        report = generate_report(
            model,
            processor,
            image,
            prompt,
        )

        print(report)

        results.append(
            f"CONDITION: {name}\n"
            f"PROMPT: {prompt}\n"
            f"REPORT:\n{report}\n"
        )

    output_file = (
        output_dir / "IU_1_context_conditioning.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n\n".join(results)
        )

    print("\n" + "=" * 70)
    print(f"Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()