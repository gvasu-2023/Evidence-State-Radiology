"""Reusable MAIRA-2 input preparation and two-stream greedy decoding.

This module extracts the corrected inference path recorded in
``experiments/maira2/phase26d/phase26d_execution.ipynb``. It does not load a
model or download weights; callers provide an already-loaded MAIRA-2 processor
and model.
"""

from __future__ import annotations

from typing import Any


def prepare_maira_inputs(
    processor: Any,
    image: Any,
    context: Any,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Prepare the context-conditioned and image-only MAIRA-2 inputs.

    Both inputs use the notebook's ``format_and_preprocess_reporting_input``
    processor path and the same frontal image. The context stream receives the
    supplied text as its indication when non-null; the visual stream receives
    ``indication=None``. Inputs are moved to ``device``, which defaults to the
    notebook's CUDA device.
    """
    import pandas as pd

    indication = context if pd.notna(context) else None

    context_input = processor.format_and_preprocess_reporting_input(
        current_frontal=image,
        current_lateral=None,
        prior_frontal=None,
        indication=indication,
        technique=None,
        comparison=None,
        prior_report=None,
        return_tensors="pt",
        get_grounding=False,
    ).to(device)

    visual_input = processor.format_and_preprocess_reporting_input(
        current_frontal=image,
        current_lateral=None,
        prior_frontal=None,
        indication=None,
        technique=None,
        comparison=None,
        prior_report=None,
        return_tensors="pt",
        get_grounding=False,
    ).to(device)

    return context_input, visual_input


def two_stream_contrastive_decode(
    model: Any,
    context_cache: Any,
    visual_cache: Any,
    context_logits: Any,
    visual_logits: Any,
    lambda_value: float = 0.0,
    max_new_tokens: int = 128,
) -> Any:
    """Greedily decode with the notebook's corrected two-stream procedure.

    At each step the decoder uses
    ``context_logits - lambda_value * (context_logits - visual_logits)``.
    The chosen token is fed to both cached streams. Decoding stops when every
    stream row selects MAIRA-2's configured EOS token or ``max_new_tokens`` is
    reached. This follows the notebook's batch-one experiment behavior and
    returns generated token IDs only (not the prompt).
    """
    import torch

    eos_id = model.generation_config.eos_token_id
    generated_tokens = []

    adjusted_logits = context_logits - lambda_value * (
        context_logits - visual_logits
    )
    next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)
    generated_tokens.append(next_token)

    if eos_id is not None and torch.all(next_token == eos_id):
        return torch.cat(generated_tokens, dim=1)

    for _ in range(max_new_tokens - 1):
        context_outputs = model(
            input_ids=next_token,
            past_key_values=context_cache,
            use_cache=True,
        )
        visual_outputs = model(
            input_ids=next_token,
            past_key_values=visual_cache,
            use_cache=True,
        )

        context_cache = context_outputs.past_key_values
        visual_cache = visual_outputs.past_key_values
        context_logits = context_outputs.logits[:, -1, :]
        visual_logits = visual_outputs.logits[:, -1, :]

        adjusted_logits = context_logits - lambda_value * (
            context_logits - visual_logits
        )
        next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token)

        if eos_id is not None and torch.all(next_token == eos_id):
            break

    return torch.cat(generated_tokens, dim=1)


def generate_maira_trajectory(
    model: Any,
    processor: Any,
    image: Any,
    context: Any,
    lambda_value: float,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> tuple[str, int]:
    """Prepare inputs, run both cached streams, and return text and token count.

    ``model`` and ``processor`` must already be loaded by the caller. The
    function mirrors the notebook's trajectory generation: inference mode,
    cached context and visual forward passes, corrected EOS-terminating greedy
    decoding, and processor decoding with special tokens skipped.
    """
    import torch

    context_input, visual_input = prepare_maira_inputs(
        processor=processor,
        image=image,
        context=context,
        device=device,
    )

    with torch.inference_mode():
        context_forward = model(**context_input, use_cache=True)
        visual_forward = model(**visual_input, use_cache=True)

        output_tokens = two_stream_contrastive_decode(
            model=model,
            context_cache=context_forward.past_key_values,
            visual_cache=visual_forward.past_key_values,
            context_logits=context_forward.logits[:, -1, :],
            visual_logits=visual_forward.logits[:, -1, :],
            lambda_value=lambda_value,
            max_new_tokens=max_new_tokens,
        )

    text = processor.decode(output_tokens[0], skip_special_tokens=True)
    return text, int(output_tokens.shape[1])
