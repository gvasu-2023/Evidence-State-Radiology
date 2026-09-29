from pathlib import Path

import pytest
import torch
from PIL import Image

from src.generation import baseline


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, images, text, return_tensors):
        self.calls.append(
            {
                "images": images,
                "text": text,
                "return_tensors": return_tensors,
            }
        )
        return {"input_ids": [[1, 2]]}

    def decode(self, output_ids, skip_special_tokens):
        assert output_ids == [1, 2]
        assert skip_special_tokens is True
        return "mock report"


class FakeModel:
    def __init__(self):
        self.eval_called = False
        self.generate_calls = []

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        assert not torch.is_grad_enabled()
        return [[1, 2]]


def configure_generator(monkeypatch):
    processor = FakeProcessor()
    model = FakeModel()
    calls = {}

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            calls["processor_model_name"] = model_name
            return processor

    class FakeBlipForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_model_name"] = model_name
            calls["model_kwargs"] = kwargs
            return model

    monkeypatch.setattr(baseline, "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(
        baseline,
        "BlipForConditionalGeneration",
        FakeBlipForConditionalGeneration,
    )

    return baseline.BaselineGenerator(), processor, model, calls


def test_baseline_loads_explicit_blip_conditional_generation(monkeypatch):
    generator, _, model, calls = configure_generator(monkeypatch)

    assert calls["processor_model_name"] == baseline.MODEL_NAME
    assert calls["model_model_name"] == baseline.MODEL_NAME
    assert calls["model_kwargs"] == {"torch_dtype": torch.float32}
    assert generator.model is model
    assert model.eval_called


@pytest.mark.parametrize(
    ("clinical_context", "expected_prompt"),
    [
        (
            "Chest pain",
            "Clinical indication: Chest pain Generate a radiology report for this chest X-ray.",
        ),
        (
            "",
            "Generate a radiology report for this chest X-ray.",
        ),
    ],
)
def test_generate_passes_context_to_blip_prompt(
    monkeypatch,
    tmp_path: Path,
    clinical_context,
    expected_prompt,
):
    generator, processor, model, _ = configure_generator(monkeypatch)
    image_path = tmp_path / "cxr.png"
    Image.new("L", (8, 8), color=64).save(image_path)

    result = generator.generate(
        baseline.GenerationInput(
            image_path=image_path,
            clinical_context=clinical_context,
        )
    )

    assert processor.calls[0]["text"] == expected_prompt
    assert processor.calls[0]["return_tensors"] == "pt"
    assert processor.calls[0]["images"].mode == "RGB"
    assert model.generate_calls[0]["max_new_tokens"] == 128
    assert result.report == "mock report"
    assert result.model_name == baseline.MODEL_NAME
