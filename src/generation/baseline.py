from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


MODEL_NAME = "nathansutton/generate-cxr"


@dataclass
class GenerationInput:
    image_path: Path
    clinical_context: str = ""


@dataclass
class GenerationOutput:
    report: str
    model_name: str


class BaselineGenerator:
    """
    Initial CXR report-generation baseline.

    This model is used for pipeline validation.
    Clinical context is included in the BLIP text prompt.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        self.model.eval()

    def generate(self, inputs: GenerationInput) -> GenerationOutput:

        image = Image.open(inputs.image_path).convert("RGB")

        if inputs.clinical_context.strip():
            prompt = (
                "Clinical indication: "
                f"{inputs.clinical_context.strip()} "
                "Generate a radiology report for this chest X-ray."
            )
        else:
            prompt = "Generate a radiology report for this chest X-ray."

        model_inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128,
            )

        report = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return GenerationOutput(
            report=report,
            model_name=self.model_name,
        )
