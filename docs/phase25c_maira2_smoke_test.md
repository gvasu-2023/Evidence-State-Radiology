# Phase 25C: MAIRA-2 Isolated Load and Interface Smoke Test

**Outcome: BLOCKED before environment creation, checkpoint download, or inference.** This host does not provide a CUDA GPU and cannot hold the official checkpoint in host memory. The files below record the attempted preflight; they do not report a model run.

## 1. Environment

Preflight was run in `C:\Evidence-State-Radiology` on branch `phase25b-maira2-feasibility`.

- Host OS: Windows; Python available in the project is 3.10.11.
- Existing project environment: PyTorch `2.14.0+cpu`; `torch.cuda.is_available()` returned `False`; `torch.version.cuda` is `None`.
- `nvidia-smi` and a CUDA device were not available in this environment.
- System RAM: 7.33 GB total, 0.72 GB available at preflight.
- No isolated MAIRA-2 environment was created. No packages were installed, and the existing `.venv` was not changed.

The official card lists a 7B F32 model, about 28 GB of parameter storage before runtime overhead, and Transformers `>=4.48,<4.52` (last tested at 4.51.3). The installed project Transformers version is outside that range; it was left untouched. This host cannot satisfy the GPU request or load the F32 checkpoint in RAM, so continuing with a CPU attempt would not be a valid substitute for the requested GPU smoke test.

## 2. Checkpoint and model loading

- Requested checkpoint: `microsoft/maira-2`.
- Processor load: **not attempted**.
- Model load: **not attempted**.
- Exact checkpoint revision/hash: not obtained.
- Model loading success: **not established**.

The run stopped at hardware preflight. No checkpoint files or model weights were downloaded.

## 3. Image and input configuration

The specified local image exists and passed a PIL open/verify check:

- Image ID: `IU_32_frontal.jpg` (UID 32)
- Path: `external_data/IU-Xray/images/IU_32_frontal.jpg`
- Format and dimensions: JPEG, 2051 × 2048
- Size: 214,452 bytes

The requested input configuration was recorded but not submitted to a MAIRA-2 processor:

```text
current_frontal = IU_32_frontal.jpg
current_lateral = None
prior_frontal = None
prior_report = None
indication = WEAKNESS OF MUSCLES; hx XXXX nodules; XXXX for changes in lungs
technique = None
comparison = None
get_grounding = False
```

No other clinical context was created. The official card documents `current_frontal`, `indication`, optional `current_lateral`, `prior_frontal`, `prior_report`, `technique`, and `comparison` fields. It documents frontal-only generation, but the requested combinations (indication only; indication plus technique; indication plus comparison without lateral) were **not empirically tested** because the processor was not loaded.

## 4. Generation and output

- Input preparation: **not run**.
- Generation success: **false / no generation attempted**.
- Generated findings: none.
- Output token count and generation time: not available.
- GPU name and GPU memory total/allocated/reserved: unavailable (no CUDA GPU).

The empty `generated_text` in the companion CSV means no report was generated; it is not an empty model response.

## 5. Token scores and logits

No MAIRA-2 generation output object was available for inspection. Consequently, `output_scores=True`, `return_dict_in_generate=True`, generated-token IDs, token-level scores, and logits access are **not verified for the checkpoint's custom remote model implementation**. The official card demonstrates `model.generate(...)`; that alone does not establish that score-return options work with this exact custom generation implementation. Custom decoding/contrastive decoding was not implemented or tested.

## 6. Six-state experiment compatibility

The documented processor interface appears to support keeping one frontal image fixed while changing only `indication`; the exact benchmark `perturbed_context` can be passed without mapping it to technique or comparison. However, context tokenization and end-to-end context propagation remain untested. No benchmark record was processed, and none of the six conditions or the cohort was run.

MAIRA-2's documented output is narrative Findings, not a separately generated Impression. The existing project evaluation reference combines Findings and Impression. This mismatch remains unresolved and must be addressed by the research owner before interpreting a later model comparison. No reference, evaluation method, or output was changed here.

## 7. Problems encountered and integrity

The execution environment has no CUDA runtime/GPU, and available RAM is far below the checkpoint's F32 parameter storage alone. Therefore the isolated CUDA environment, processor/model load, and one-report GPU inference could not be performed. The correct next action is to run this smoke test on the intended GPU workspace with an approved CUDA environment; do not substitute CPU inference on this host.

Verified frozen-file SHA-256 values after preflight:

- `results/tables/phase16_perturbations.csv`: `3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53`
- `results/tables/phase16e_selected_studies.csv`: `21144953c8d796a31c937405386d03d52cd933008f765befbc255b2c12d8cd0a`
- `results/tables/phase17_vlm_inference_eligibility.csv`: `cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6`

No benchmark, Phase 20 result, BLIP baseline, gated output, source, or evaluation script was modified. No package was installed, no checkpoint downloaded, and no inference was performed. Pytest was not run because no code or test file changed. The companion CSV was read back and validated as exactly one row with the requested columns.

## 8. Recommendation for next phase

Move this exact smoke test to a GPU workspace with adequate VRAM (the Phase 25B planning target was one A100 80 GB; upstream does not publish a minimum). In a new isolated environment, pin Transformers 4.51.3 and choose a compatible CUDA PyTorch wheel; record all installed versions. Then load the processor/model, test the requested frontal-only input and context-field combinations, inspect generation score outputs, and generate exactly one report. Keep all outputs isolated. Do not begin the 576-record cohort until that smoke test succeeds and the Findings-versus-Impression evaluation decision is resolved.

### Source

- [Official MAIRA-2 model card and loading/interface instructions](https://huggingface.co/microsoft/maira-2)
- [MAIRA-2 paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/06/2406.04449v1.pdf)
