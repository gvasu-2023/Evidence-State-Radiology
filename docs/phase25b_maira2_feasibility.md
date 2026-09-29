# Phase 25B: MAIRA-2 Feasibility

**Status:** Desk review and repository inspection only. No checkpoint was downloaded, no package was installed, and no inference was run. The Phase 16–24 benchmark, experiment outputs, and evaluation code were not changed.

## 1. Objective and scope

This phase determines whether the exact public Microsoft MAIRA-2 checkpoint can support a future controlled experiment with the frozen six-condition, image-backed Phase 17 cohort. It does not implement the model or authorize a new inference run.

Primary candidate: `microsoft/maira-2`. Backup: the third-party `HarsathV/llava-rad-iu-v3` adapter. Upstream sources were checked on 2026-09-29. Metrics below are reported with their original task/split and are not treated as directly comparable to Phase 20 metrics.

## 2. Existing pipeline and constraints

The current `BaselineGenerator` loads `nathansutton/generate-cxr` through `BlipForConditionalGeneration`, constructs a prompt from `GenerationInput.clinical_context`, and sends image plus prompt through the processor. The Phase 17 runner passes each authoritative `perturbed_context` as that input and saves a single generated-report string. The existing analyzer accepts an `EvidenceAssessment` and yields one of five states (`sufficient`, `incomplete`, `irrelevant`, `conflicting`, `insufficient`); the ReliabilityGate maps those states to the existing `generate`, `qualify`, `discount_context`, `qualify_or_abstain`, or `abstain` actions. These modules can surround a model adapter without changing their definitions.

The observed Phase 18 distribution is a material limitation for this research: all 96 `evidentiary_incomplete` cases were classified as `sufficient`. MAIRA-2 feasibility does not repair or justify changing that behavior. The analyzer's output must be recorded as observed if the model is later evaluated.

Phase 20 joins generated text against the same benchmark Findings and Impression concatenated as reference text. MAIRA-2's official output task is the **Findings section**, not a complete Findings-plus-Impression report. A wrapper can put its findings string into the existing `generated_report` field, but cannot honestly create an Impression. This is a task-format mismatch for strict apples-to-apples score comparison. Do not add a synthetic Impression or change the reference/evaluation protocol silently. A protocol decision is needed before treating its Phase 20 scores as directly comparable.

## 3. Official artifact and architecture

- Paper: Bannur et al., *MAIRA-2: Grounded Radiology Report Generation* (2024 preprint, arXiv:2406.04449).
- Exact checkpoint: [`microsoft/maira-2`](https://huggingface.co/microsoft/maira-2), gated by acceptance of research-use terms.
- Official distribution/code entry point: the Hugging Face model repository itself. It contains the model card and custom Transformers implementation files (`configuration_maira2.py`, `modeling_maira2.py`, and `processing_maira2.py`). The card demonstrates `AutoModelForCausalLM` and `AutoProcessor` with `trust_remote_code=True`. The paper links Microsoft’s [RadFact repository](https://github.com/microsoft/RadFact) for its metric; that is not the MAIRA-2 inference implementation. No separate official MAIRA-2 GitHub inference repository was identified in the official card/paper.
- Architecture: frozen RAD-DINO-MAIRA-2 image encoder, a trained projection layer, and fully fine-tuned Vicuna-7b-v1.5 language model. The card lists 7B parameters and F32 tensors. It is a custom-code vision-language causal generator, rather than the BLIP class used by this repository's baseline.
- Model files are split across six large safetensors shards; the checkpoint is multi-gigabyte and gated. No files were downloaded in this phase.

## 4. Training data and IU-Xray exposure

The official model card lists report-generation training data as MIMIC-CXR, PadChest, and private USMix. The official paper is explicit that IU-Xray was **not used to train MAIRA-2**: it describes using the entire IU-Xray dataset as external validation for Findings generation. Its dataset table lists no IU training examples and 3,306 IU test samples. Thus the source supports “not trained on IU-Xray” for the described MAIRA-2 training recipe; it does not prove absence of every possible cross-dataset duplicate in private or public source corpora.

The distinction matters for this project: the selected IU studies are not reported training data, but the full public IU dataset was already used for the paper's external evaluation. The cohort is therefore not a blind, previously untouched IU test set. This is prior public evaluation exposure, not evidence of training leakage. The paper does not provide a UID-level MAIRA evaluation manifest to establish whether each of our selected study identifiers was individually scored.

The project cohort comprises 96 image-backed IU studies, with six benchmark contexts per study (576 records). Preserve all benchmark rows and exact contexts. Do not describe model test-set exposure as absent.

## 5. Input and clinical-context compatibility

The official processor's reporting interface accepts `current_frontal`, optional `current_lateral`, optional prior frontal plus prior report, `indication`, `technique`, and `comparison`. A frontal image alone is a valid documented input; the lateral and prior inputs are optional. The model-card example uses PIL images and the processor's `format_and_preprocess_reporting_input` method, with `get_grounding=False` for narrative output. The processor owns image preprocessing; the exact effective dimensions should be read from the pinned processor code/configuration and recorded during later load validation. Do not invent a resize or send a lateral image in place of a missing frontal.

For this benchmark, pass the exact `perturbed_context` string into **`indication`**, retaining the same frontal image across the six rows for each UID. Do not rephrase it or map it into technique/comparison. For an empty context, pass the empty/absent indication using the official processor convention; do not add a filler phrase. This directly supports the core comparison of the same image under multiple clinical contexts.

The card warns that changing prompts may degrade performance because the model was not optimized for arbitrary user prompts. Therefore use the official reporting prompt/template and change only the indication value. Validate that the full exact text survives processor formatting and reaches model inputs. Context-conditioned generation is supported in interface terms; robust response to the benchmark's synthetic perturbations is an empirical question.

## 6. Output and evaluation compatibility

MAIRA-2's documented non-grounded report-generation output is narrative **Findings**. The optional grounded mode adds boxes and object tokens. It does not promise a separately generated Impression section. The current project stores one `generated_report` text and compares it to concatenated reference Findings + Impression (Phase 20). A thin adapter can preserve identifiers, contexts, outputs, and output-file schema, but cannot resolve this target mismatch.

Before any comparative 576-record run, the research owner must decide whether to (a) compare Findings output against Findings-only references under a formally approved, consistently applied protocol, or (b) retain the current full-reference methodology and label this model's output/score as task-mismatched. This feasibility phase does not change the existing methodology. Do not manufacture Impression text or splice reference text into model output.

## 7. Hardware and dependencies

The model card lists 7B F32 parameters (about 28 GB of weight storage at four bytes per parameter, before runtime buffers, activations, and image processing). It does not publish an inference VRAM minimum. Its example explicitly moves the model to CUDA. CPU placement may be technically possible with ample host RAM, but the F32 weights alone exceed ordinary workstation memory budgets and CPU throughput is not a realistic target for 576 records. Treat GPU inference as required for the planned cohort unless a measured, approved alternative demonstrates otherwise.

Official card setup lists Pillow, protobuf, sentencepiece, PyTorch, and `transformers>=4.48.0,<4.52`, last tested with Transformers 4.51.3. It does not pin a PyTorch version or document an official quantized configuration. Do not assume 4-bit/8-bit support or that quantization preserves outputs; no quantization path is proposed for the first run.

**Recommended isolated validation environment (proposal, not an upstream-tested recipe):** Linux x86_64, Python 3.10, one NVIDIA A100 80 GB, a CUDA-enabled PyTorch wheel explicitly verified with Transformers 4.51.3, and Transformers pinned to 4.51.3 with the remaining official card dependencies pinned in a lock file. The 80 GB GPU is a conservative feasibility target, not an asserted minimum; upstream documents A100 hardware in training/environment reporting but gives no inference VRAM minimum. The exact PyTorch/CUDA wheel must be selected and tested in a disposable environment before any model files are fetched. Do not retrofit the current project interpreter: the recorded project environment is Windows CPU-only with Transformers 5.17.0, outside MAIRA-2's documented Transformers range.

## 8. Decoding, Analyzer, and Gate integration

MAIRA-2 is loaded through a causal language-model API and generates autoregressively. Standard Transformers generation can expose per-step scores with `return_dict_in_generate=True, output_scores=True`, and a custom loop can in principle use model logits. However, the model relies on custom remote code and the accessible model card does not validate a custom contrastive-decoding implementation. Therefore:

- **Logits/custom decoding:** technically plausible, not yet verified for the pinned MAIRA custom class and multimodal input path. A later load-only/API probe must inspect `forward`/`generate` outputs without generating benchmark reports, then a synthetic tiny-input unit test should establish score shape and decoding hooks before any CCD experiment.
- **Contrastive decoding:** possible in principle if the model exposes compatible conditional/unconditional token logits and a suitable negative branch can be defined. No such branch or MAIRA-specific CCD correctness is established here; image/text handling, chat template, and memory cost are unresolved. Do not call it supported until validated.
- **Evidence-State Analyzer:** can run before the model call using current project code and preserve the state output. It consumes assessment flags rather than performing a clinically grounded image/text analysis itself; this is not made more capable by MAIRA-2.
- **ReliabilityGate:** can run after analysis and before report generation, routing/qualifying/abstaining according to its unchanged existing mapping. In the existing Phase 18 design, non-abstain actions can reuse model reports; a new MAIRA run would generate only where the unchanged architecture explicitly requires it. Any number of required generations must be counted before inference authorization.

## 9. Backup: LLaVA-Rad IU v3

`HarsathV/llava-rad-iu-v3` is a third-party LoRA adapter for Microsoft's LLaVA-Rad, not a Microsoft “v3” official checkpoint. Its public card reports fine-tuning on 2,837 Open-IU image-report pairs and a held-out IU test set of 500; reported RadGraph `rg_er` is 0.3503 and ROUGE-L 0.2794 versus the stated base results of 0.2877 and 0.2037. These are the adapter card's reported metrics, not Phase 20 RadGraph-F1 or an apples-to-apples result.

The configuration combines Vicuna-7b-v1.5, the BiomedCLIP-CXR 518 vision tower, a changed projector, and a LoRA adapter. The card describes loading in BF16 and reports 2x RTX 5060 Ti 16GB for its LoRA training setup; it does not establish an inference minimum. Microsoft LLaVA-Rad supports frontal image plus optional exam reason and generates Findings. The community v3 card links an evaluation repository and says the loader applies a config patch, projector, and adapter. The public metadata reviewed gives training count and held-out count but does **not** expose exact Open-I/IU study identifiers or a UID manifest sufficient to determine overlap with our selected UIDs. Since the adapter was trained on IU, it is not a suitable clean backup until exact train/test IDs or an independently auditable split are provided and checked.

## 10. Reproducibility and risks

1. The gated checkpoint requires acceptance of Microsoft's research-use terms; it is not currently a zero-friction public download. Terms prohibit clinical use.
2. The remote custom code and Transformers range require a pinned isolated environment; the project's current Windows/Transformers 5.17 environment is outside that range.
3. MAIRA-2 has seen external evaluation on all IU-Xray, so a new result must not be framed as a blind test-set generalization result.
4. Its official Findings-only target does not match the current Findings-plus-Impression reference string.
5. Its card warns arbitrary prompt changes may degrade performance. Exact indication insertion must be tested while keeping official prompt structure fixed.
6. No official minimum VRAM, quantized inference recipe, or validated 576-record runtime is documented.
7. Custom contrastive decoding, logits use, and integration with the project's unchanged generator interface remain untested.
8. The LLaVA-Rad IU v3 fallback has unresolved study-level training overlap and requires multiple components/custom loading.

## 11. Recommendation

**Conditional GO for a separate, user-authorized checkpoint-load feasibility step; NO-GO for cohort inference until output/reference compatibility is decided.** MAIRA-2 is the strongest technical match for the context intervention: its official documented interface accepts image plus indication, with an optional lateral view. Its training recipe excludes IU-Xray, although the full IU dataset was already used in external evaluation. It is preferable to the IU-trained LLaVA-Rad v3 adapter for a training-leakage-sensitive experiment, subject to the public-evaluation exposure qualification.

Do not download now. Before the download step, confirm access/terms and prepare the isolated GPU environment. The very first authorized test should load the exact checkpoint and processor once, report versions/device/memory/parameter count, and run no benchmark inference. Then validate processor handling with a non-benchmark or synthetic input, record exact indication tokenization and output parsing, inspect whether logits are exposed, and verify frontal-only operation. Only after that should the project owner decide whether the Findings-only mismatch can be handled without altering the frozen methodology.

## 12. Exact next-step validation tests

1. Confirm gated-model terms/access and pin the exact Hub revision/commit hash before download.
2. In a new isolated Linux GPU environment, validate Python 3.10, Transformers 4.51.3, the selected CUDA PyTorch wheel, processor dependencies, CUDA visibility, available VRAM, and clean import.
3. Load only MAIRA-2 and its processor; check config/model class, parameter count, CPU/GPU placement, eval mode, peak memory, and no missing/unexpected weight keys. No report generation.
4. Inspect the pinned processor/model code and create a unit test with a synthetic image and distinct short contexts to prove same-image/different-indication inputs produce differing tokenized prompt segments; do not use benchmark records or generate reports.
5. Verify frontal-only processor input, empty indication, handling of each optional metadata field, output slicing after prompt tokens, and decoding to a Findings string. Use a permitted non-benchmark fixture; report generation only after separate authorization.
6. Verify how to represent Findings-only text against the existing Phase 20 reference protocol. Stop until the study owner explicitly resolves this methodological incompatibility.
7. Load-only inspect generation score/logit access. Prototype CCD only in an isolated unit/synthetic test, without changing the ReliabilityGate, Analyzer, benchmark, or existing result files.
8. If approved after these gates, conduct a separately authorized small smoke test with preselected eligible benchmark cases, then review runtime and GPU memory before considering the full cohort.

## Sources

- [Official Microsoft MAIRA-2 model card/checkpoint](https://huggingface.co/microsoft/maira-2)
- [MAIRA-2 paper (Microsoft Research PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/06/2406.04449v1.pdf)
- [MAIRA-2 arXiv record](https://arxiv.org/abs/2406.04449)
- [Official RAD-DINO-MAIRA-2 encoder card](https://huggingface.co/microsoft/rad-dino-maira-2)
- [Microsoft RadFact implementation](https://github.com/microsoft/RadFact) (metric implementation, not MAIRA-2 generation code)
- [Microsoft LLaVA-Rad implementation](https://github.com/microsoft/LLaVA-Rad)
- [LLaVA-Rad IU v3 adapter card](https://huggingface.co/HarsathV/llava-rad-iu-v3)
- [LLaVA-Rad IU v3 public loader/evaluation repository](https://github.com/vijayakumarharsath/llava-rad-iu)

## Validation record

This phase created only this document and the structured table linked below. No source code, tests, benchmark, eligibility file, model output, or evaluation output was modified. No package or model checkpoint was downloaded; no inference was executed. The current full test suite was not run because there was no code change. The CSV structure was checked by reading it back and verifying its required columns and row count.
