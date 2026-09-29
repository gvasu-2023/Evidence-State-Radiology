# Phase 25A: Model and Architecture Feasibility Study

**Status:** Literature and repository feasibility review only. No model was downloaded or run. The frozen Phase 16–24 benchmark and results remain outside the scope of this review.

## 1. Objective

This review compares publicly documented radiology report-generation candidates before selecting an implementation target. It considers the fixed research question: how incomplete or conflicting clinical context affects report factuality, and whether the existing Evidence-State Analyzer and Reliability Gate can manage that context. It does not rank models by reported score alone.

Evidence was taken from the repository’s Phase 24 documents and source, candidate papers, official/public code repositories, and model cards as available on 2026-09-29. A reported metric remains attached to its source, split, task, and definition; it is not treated as directly comparable with this project’s Phase 20 metrics unless those details match.

## 2. Existing baseline limitations

The current baseline is `nathansutton/generate-cxr`, a BLIP image-to-text checkpoint fine-tuned on MIMIC-CXR. The Hub model card identifies 0.5B parameters and shows a processor call that accepts both an image and indication text. The project’s `BaselineGenerator` uses `BlipForConditionalGeneration`, sends `GenerationInput.clinical_context` in the BLIP text input, and generates with `max_new_tokens=128`. Its current runner processes the Phase 17 image-backed cohort on CPU. This makes it an already validated, relatively compact project baseline, not a large vision-language foundation model. [Model card](https://huggingface.co/nathansutton/generate-cxr)

Its known constraints for a new architecture study are:

- The underlying training data are MIMIC-CXR, not IU-Xray; transfer to the frozen IU-Xray cohort is external-domain evaluation, but existing model-card evidence does not document a study-level split or overlap audit.
- The implementation uses ordinary `model.generate()` and does not currently provide a project-level contrastive decoding path or calibrated confidence outputs.
- The six-state analyzer’s observed Phase 18 behavior includes all 96 evidentiary-incomplete records being classified as `sufficient`. Phase 24 treats this as an observed limitation. A stronger backbone cannot fix that analyzer behavior by itself.
- The Phase 24 metric package shows that a model/gate change must be assessed on paired records and by condition/state, including coverage. A single pooled score would hide important differences.

## 3. Candidate model comparison

The accompanying CSV, [phase25a_model_feasibility.csv](../results/tables/phase25a_model_feasibility.csv), contains one structured row per configuration. It includes the model, paper/year, repository/checkpoint, training and test evidence, reported metrics, model/hardware estimate, text and image interface, decoder access, Analyzer/Gate fit, complexity, suitability, and outstanding verification.

### Candidate findings

| Candidate | Evidence-based assessment |
|---|---|
| **LLaVA-Rad IU v3 adapter** | The named `llava-rad-iu-v3` is a public third-party LoRA adapter on the official Microsoft LLaVA-Rad base, not an official Microsoft v3 release. Its model card says it was fine-tuned on 2,837 Open-IU image/report pairs and reports a 500-study held-out IU set, RadGraph `rg_er` 0.3503 and ROUGE-L 0.2794. The adapter, a re-trained projector, and a custom loader are required. The model accepts an optional exam reason; LLaVA-style generation exposes token scores/logits for custom decoding. The selected project UID list must be checked against the adapter’s exact training and test identifiers before any use. [Adapter card](https://huggingface.co/HarsathV/llava-rad-iu-v3), [official base repository](https://github.com/microsoft/LLaVA-Rad) |
| **RadHiera IU 3B** | MICCAI 2026 paper and public author repository report an IU-Xray checkpoint, `RadHiera-IU-3B`, and state use of the official IU split. The paper identifies Qwen2.5-VL-3B as its model, followed by supervised and GRPO fine-tuning. Reported IU values include BLEU-2 0.284, BERTScore 0.607, SembScore 0.626, RadGraph 0.313, 1/RadCliQ-v1 2.325, CheXbert-related F1 0.519, and consistency 0.915. **The paper table and repository summary differ slightly** (repository lists 1/RadCliQ-v1 2.2929 and F1/consistency 0.5191/0.9141); this discrepancy must be resolved against the pinned evaluation artifact before reproducing a number. The repository exposes LoRA weight archives and evaluation scripts. The exact archive’s base-model revision, IU study split identifiers, and a user-facing arbitrary-context interface still need validation. It has the most directly relevant recent IU report-generation evidence here, but should not be called SOTA from those numbers. [Paper](https://papers.miccai.org/miccai-2026/paper/1791_paper.pdf), [repository](https://github.com/xmed-lab/RadHiera) |
| **MediVLM supervised IU configuration** | Findings of EMNLP 2025 describes a detector, CLIP image encoder, ClinicalBERT text path, cross-attention fusion, and a GPT-2-based report decoder. The public project offers IU configuration/training/evaluation code; the reviewed public material does not establish a released ready-to-load IU checkpoint. The paper’s IU table reports RadGraph-F1 0.406, BERTScore 0.616, and RaTEScore 0.722, with BLEU/ROUGE figures elsewhere. These are paper-reported values under that study’s data and metric pipeline. Its inference interface is image-centric; an arbitrary perturbed indication is not documented as an inference input. It is therefore a weak fit for the current context perturbation experiment despite reported IU results. [Paper](https://aclanthology.org/2025.findings-emnlp.544/), [public code](https://github.com/sonai-commits/MediVLM) |
| **MAIRA-2 direct checkpoint** | The official Microsoft model is a 7B Vicuna-based multimodal model with an 87M Rad-DINO image encoder and four-layer adapter. The paper trained on MIMIC-CXR, PadChest, and private USMix; IU-Xray is explicitly described as an external evaluation set, with 3,306 findings-generation samples. On IU, it reports ROUGE-L 27.4, BLEU-4 11.7, CheXbert Macro F1-14 28.6, Micro F1-14 52.9, and RadFact logical precision/recall 71.1/67.3. It explicitly accepts indication, technique, comparison, and optional image/history inputs. The official model card gives a 7B size, requires research-only use terms, and currently specifies Transformers 4.48–4.51.3. It fits the clinical-context question and offers autoregressive logits, but needs a GPU for practical cohort inference and a compatible isolated environment. Note that the full IU set was already used in the paper’s external evaluation; this is not training leakage, but the selected project cases are not a wholly unseen public benchmark. [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/06/2406.04449v1.pdf), [official model card](https://huggingface.co/microsoft/maira-2) |
| **MAIRA-2 + CCD-compatible Libra route** | CCD is an inference-time method that uses expert-derived signals and token-level logits. Its public implementation lists a MAIRA-2-compatible Libra checkpoint (`X-iZhang/libra-maira-2`) and supports several LLaVA/Libra-family configurations. The repository cautions that its adapted Libra weights are for demonstration and that accurate evaluation should use original weights/configuration, especially the chat template. Therefore this is a promising architecture recipe, not yet a verified drop-in CCD run for the exact official Microsoft checkpoint. The paper reports up to 17% RadGraph-F1 improvement on MIMIC-CXR; that claim is not transferred to IU or to this project. [CCD paper](https://arxiv.org/abs/2509.23379), [CCD repository](https://github.com/X-iZhang/CCD) |
| **R2GenGPT** | A reproducible research implementation for frozen LLM report generation, with IU training scripts and public code. Its public README explicitly gives checkpoint download/testing instructions for MIMIC-CXR, not a ready IU checkpoint. The model uses a Vicuna-7B family language model, with the paper describing 5M tuned parameters (0.07% of total). The method is primarily image-conditioned; arbitrary clinical indication input is not a first-class documented interface. Published IU scores vary by split and protocol; they are not a clean basis for selection here. It is a useful reproducibility/control candidate, but less aligned to the context and post-hoc gating questions than MAIRA-2. [Paper](https://arxiv.org/abs/2309.09812), [official repository](https://github.com/wang-zhanyu/R2GenGPT) |
| **Current BLIP baseline (`nathansutton/generate-cxr`)** | Retain as the experimental anchor. It accepts image plus indication, is 0.5B parameters, has a public checkpoint, runs in the existing CPU pipeline, and has project-verified context propagation. Its simple generation path makes it a feasible decoder-modification control, although its report quality and IU transfer remain limited and its MIMIC training provenance should be documented. [Model card](https://huggingface.co/nathansutton/generate-cxr) |

The requested **LLaVA-Rad IU v3** and **CCD** entries refer to specific public configurations, not generic architecture labels. “MediVLM” here refers to the 2025 EMNLP paper/repository, not unrelated projects using the same name.

## 4. Dataset and leakage analysis

The frozen Phase 16 benchmark is derived from IU-Xray. The 100 selected studies are not documented in Phase 24 as a standard IU train/test partition. Therefore, any checkpoint trained or fine-tuned on IU-Xray requires a UID/study-level overlap audit against all selected studies before use:

- **Highest direct training overlap risk:** LLaVA-Rad IU v3 and RadHiera-IU, because the reviewed sources explicitly fine-tune on IU reports/images. A paper/model-card statement that a test split was held out is not sufficient to prove our selected 96 image-eligible UIDs are in that held-out portion.
- **MediVLM:** an IU supervised configuration is available in public code; if trained for this project, the exact split and UID partition must be frozen before any checkpoint can be considered. Its reported paper score does not prove that a locally reproduced checkpoint is free from overlap.
- **MAIRA-2 direct:** the paper describes IU as external validation and its training examples come from MIMIC-CXR, PadChest, and USMix. This supports no IU report-generation training in the stated recipe. However, the selected IU cohort is drawn from a dataset the paper evaluated in full. Its previous public exposure means the result should be described as an experiment on the frozen context-perturbation benchmark, not a novel blind IU test.
- **MIMIC-trained BLIP/R2GenGPT:** no IU training is stated for the cited checkpoint sources, but exact pretraining data, benchmark exposure, and any IU fine-tuning still need checkpoint-level verification. Do not infer absence of leakage from a model name or paper abstract.

The existing cohort also derives multiple context conditions per underlying study. Any later split/uncertainty analysis must group on underlying UID, not treat the six rows as six independent studies. The current frozen design must not be altered to accommodate a candidate checkpoint; if overlap prevents scientifically meaningful use, that is a research decision for the project owner.

## 5. Hardware feasibility

Hardware figures in the CSV are planning estimates unless a cited source specifies hardware. They are not measured requirements.

- **Current BLIP baseline:** 0.5B parameters; existing project has already executed its CPU pipeline. CPU is plausible and demonstrated here, though cohort runtime remains material.
- **MediVLM:** parameter total and memory profile are not reported in the reviewed paper/repository. The multi-component detector/encoder/text-fusion setup makes hardware needs dependent on obtaining the code/checkpoint and exact inference stack. Do not promise CPU suitability without a measured load-and-run test.
- **RadHiera 3B:** 3B language model plus vision stack/adapters. A GPU with roughly 12–24 GB VRAM is a planning range for inference in reduced precision, not an official requirement; weight archive and chosen resolution determine actual use. CPU execution is theoretically possible with large RAM, but likely too slow for 576 paired conditions.
- **7B LLaVA-Rad / MAIRA-2 / R2GenGPT:** at least roughly 14 GB for 7B weights in BF16 alone, before image activations, KV cache, and runtime overhead. Plan for a 24 GB-class GPU or larger; memory optimization/quantization would need validation and can change outputs. Official LLaVA-Rad documentation reports V100/A100 testing. MAIRA-2's public model card documents 7B F32 weights and a Transformers 4.48–<4.52 range. CPU use may technically load with substantial RAM but is not a realistic throughput target for a 576×6-style cohort.
- **CCD:** adds expert-model computation and a custom decoding pass. Plan for GPU execution; its repository recommends CUDA-compatible hardware. The actual memory budget is the combined MLLM and expert model(s), not only the base checkpoint.

No hardware has been benchmarked in this phase. GPU availability, RAM, model-loading peak memory, generation throughput, and disk space are exact next-step measurements.

## 6. Clinical-context compatibility

The context perturbation benchmark requires the exact `perturbed_context` to be accepted at inference for every condition. This is distinct from a model having seen clinical text during training.

- Strongest documented fit: MAIRA-2 accepts an Indication input and can jointly process text and image tokens. The official MAIRA prompt/template should be preserved, replacing only the indication with the benchmark's exact context.
- Good potential fit: official LLaVA-Rad accepts an optional exam reason. The IU v3 adapter's loader must be checked to ensure that reason reaches the checkpoint's expected chat prompt unchanged.
- Weak fit: MediVLM's text encoder participates in representation alignment/training; its public image-only inference function does not document arbitrary context input.
- Weak/unclear fit: RadHiera paper formulation is image-conditioned. The public evaluation input appears report-generation oriented; arbitrary indication conditioning and its effect on the trained prompt need direct inspection.
- Weak fit: R2GenGPT's project interface is image-to-report; context conditioning is not established in the inspected repository.
- Existing BLIP baseline already passes `clinical_context` through the project’s text prompt.

For every selected candidate, a context-propagation unit test must capture the final processor/chat-template input and compare it byte-for-byte to the source benchmark context after only the explicitly documented wrapper formatting. No silent use of original clinical context is acceptable.

## 7. Contrastive-decoding compatibility

CCD and other contrastive decoding methods change token selection by combining or adjusting next-token distributions; they need access to logits and a compatible generation loop. LLaVA-family autoregressive models and MAIRA-2 expose this capability in principle. The CCD repository already implements a dual-stage method for listed LLaVA/Libra-family checkpoints and expert-model signals. This is stronger evidence than a general claim that a model supports `generate()`.

For exact Microsoft MAIRA-2, compatibility is **plausible but unverified**: its LLaVA-derived model is autoregressive and has logits, while the public CCD port lists a Libra-converted MAIRA-2 checkpoint and warns about demonstration weights/template settings. An adapter to the exact official class would require verifying processor templates, cache behavior, image-token expansion, token-score alignment, and expert signal dimensions. That is an implementation experiment, not assumed compatibility.

LLaVA-Rad v3 and current BLIP also permit custom autoregressive token scoring, but no repository-provided CCD path for the exact v3 adapter was established. For RadHiera/Qwen2.5-VL, a custom decoder is possible in principle but not pre-integrated and is higher complexity. For MediVLM/R2GenGPT, custom sampling could be implemented because their decoders produce token distributions, but no out-of-box CCD support was identified.

## 8. Evidence-State framework compatibility

The repository’s `EvidenceStateAnalyzer` returns the existing five values (`sufficient`, `incomplete`, `irrelevant`, `conflicting`, `insufficient`); the frozen benchmark retains six conditions, including distinct syntactic and evidentiary incomplete conditions. `ReliabilityGate` maps those states to its existing actions. Both modules wrap generation and are model-agnostic by design. A candidate can sit behind the analyzer/gate if an adapter exposes a consistent interface:

`image + exact perturbed context -> candidate generation`, with the existing policy before generation and existing report transformation/abstention after generation.

No candidate integration should revise analyzer logic, gate rules, state definitions, or benchmark conditions. The known evidentiary-incomplete analyzer output must be retained and reported as an observed behavior. Candidate adapters should expose context in the model’s native expected field, support abstention without model invocation, and keep baseline/gated outputs paired by `(sample_id, uid, condition)`.

## 9. Implementation complexity

| Candidate | Relative effort | Main work |
|---|---|---|
| Existing BLIP baseline | Low | Keep as frozen anchor; an experimental decoder hook would still require a separately tested custom generation path. |
| LLaVA-Rad IU v3 | Medium-high | Separate compatible environment; load Vicuna base, config patch, projector and PEFT adapter; verify licensed model/data terms, prompts, UID split, and logits interception. |
| RadHiera IU 3B | High | Resolve base checkpoint/config and public archive; reproduce intended Qwen2.5-VL prompt and IU split; bridge to project runner; custom contrastive decoding likely needed. |
| MediVLM | High | Assemble multiple component checkpoints, resolve exact released code provenance, detector weights, supervised checkpoint availability, inference mode, and context conditioning. |
| MAIRA-2 direct | Medium-high | Obtain access under model-card terms, isolate supported Transformers version, implement official input template and frontal image path, adapt to existing runner, and test generation. |
| MAIRA-2 + CCD | High | All MAIRA-2 work plus exact-checkpoint CCD adapter, expert-model loading, custom logits path, and deterministic decode validation. |
| R2GenGPT | High | Legacy research stack, Vicuna/checkpoint handling, likely GPU, dataset preparation/training for IU, and context-interface adaptation. |

## 10. Reproducibility assessment

Most reproducible for the **current project interface** is its existing BLIP baseline: checkpoint, processor, prompt path, inference code, output, and tests are already present. It is not necessarily the strongest architecture.

Among new candidates, MAIRA-2 has the strongest combination of a first-party paper/model card, explicit IU external evaluation, documented multimodal indication input, an exposed public checkpoint (access terms apply), and a documented model architecture. Its reproducibility requires pinning source revision, model revision, Transformers version, prompt template, image preprocessing, precision, and decoding parameters. The complete IU dataset was already externally evaluated in its paper.

LLaVA-Rad IU v3 has unusually direct IU-task metrics and explicit training/test counts, but the adapter is third-party, its split must be identified, and its base/projector/adapter must be loaded together. RadHiera has recent paper/repository/checkpoint/metric artifacts, but small public footprint, research-code dependencies, and UID-level split proof are concerns. MediVLM has a peer-reviewed paper and code but no verified ready checkpoint in reviewed sources, an inference interface mismatch, and multiple required component weights. R2GenGPT is transparent open research code but old-stack/checkpoint/context-fit limitations are significant. CCD has source code and a paper, but its current documented MAIRA route is an adapted compatibility checkpoint, not proof that the exact official MAIRA-2 weights load identically.

The numerical reports in papers use different test rows, report sections, metric versions, normalization, labelers, checkpoint revisions, and bootstrap/aggregation conventions. The table deliberately does **not** rank rows by the highest metric. In particular, MediVLM's RadGraph-F1 and MAIRA-2's RadFact/ChenXbert values are not Phase 20 scores; RadHiera's repository summary is not automatically comparable to the project metric runner.

## 11. Primary candidate

**Primary architecture candidate for a future controlled feasibility validation: official MAIRA-2 direct checkpoint, initially without CCD.**

Reasons:

1. Its intended input includes both frontal image and clinical indication, matching the central benchmark manipulation rather than requiring context to be bolted on.
2. Its paper explicitly reports IU-Xray as external evaluation and documents no IU report-generation training in its training recipe, reducing direct-training-overlap risk versus IU-fine-tuned candidates.
3. It is a first-party model with a public paper, model card, and documented checkpoint/architecture.
4. Its autoregressive logits make a later, separately controlled decoding study technically feasible, while analyzer and gate remain external wrappers.
5. The recommendation does not depend on its reported scores being numerically superior; those scores are not directly comparable with the project’s Phase 20 evaluation.

This is a recommendation to validate the checkpoint, not authorization to install packages, download it, or run inference. The model’s full-IU external evaluation and current Transformers version requirement remain material limitations. Exact selected-UID mapping and prompt leakage/overlap should be documented before the experiment is treated as blind.

## 12. Backup candidate

**Backup: official Microsoft LLaVA-Rad base with the IU v3 adapter only if selected-UID overlap audit passes.** It directly accepts exam reason text and the public adapter reports held-out IU metrics; its LLaVA-style decoder is a good technical target for contrastive decoding. However, it was fine-tuned on IU-Xray, the adapter is third-party, and the exact held-out split and cohort UID overlap must be verified first. If any selected study appears in its training partition, it is not a valid clean candidate for this frozen cohort without a research-level decision.

RadHiera-IU-3B is a strong alternative for an IU-specific architecture comparison, but its own IU fine-tuning and unverified exact split identifiers make it a lower-priority backup under this cohort's leakage constraints.

## 13. Risks and unresolved questions

- Does the chosen MAIRA-2 checkpoint revision load in an isolated environment with the exact public model-card-supported Transformers version, and what memory/throughput does it require?
- Can every frozen benchmark context, including empty and syntactically/evidentially perturbed strings, be inserted into the official MAIRA prompt without normalization or truncation?
- Which of the 96 eligible study UIDs were included in the published MAIRA-2 IU external evaluation? The paper reports the whole IU findings set, so the benchmark is previously seen as an evaluation dataset even though it was not used for report-generation training.
- What are the exact train/validation/test UID lists for the LLaVA-Rad IU v3 and RadHiera checkpoints, and do any selected studies overlap their training sets?
- Does the v3 LoRA adapter preserve public license/access terms for the Vicuna base and projector?
- Is the current CCD method compatible with original MAIRA-2 beyond its listed Libra adapter, without changing the official chat template or generation semantics?
- Can CCD be evaluated separately from the Evidence-State Gate without conflating interventions? It would need a pre-registered factorial or sequential design decision; no methodology change is made here.
- MediVLM checkpoint provenance, component weight versions, context interface, and split details remain unresolved.
- A fixed six-state benchmark exposes that syntactic and evidentiary incompleteness are distinct perturbation conditions. Candidate prompt adaptation must not collapse these conditions or hide the Phase 18 analyzer finding.

## 14. Exact next-step validation tests

Run these as a separate, explicitly authorized follow-up before any cohort inference:

1. **Artifact/provenance check:** record exact model repository, immutable revision, model-card terms, weight filenames/sizes, base model, and processor/tokenizer revisions. Do not download until the checkpoint and scope are approved.
2. **UID leakage audit:** compare all 100 selected UIDs (including the 4 image-unavailable studies for provenance) to candidate train/validation/test UID manifests. For MAIRA-2, record the paper’s full-IU external evaluation overlap and do not call the 96-case set blind.
3. **Environment/load-only check:** in an isolated environment matching the official requirements, load MAIRA-2 once on the intended GPU, record peak VRAM/RAM and parameter count, move/eval check, and stop without generating text.
4. **Image preprocessor check:** process one approved frontal image only; verify RGB/tensor shape and exact original image reference. Never substitute a lateral view.
5. **Prompt/context propagation test:** use mocks or processor instrumentation to prove exact benchmark `perturbed_context` is passed through the official chat template for all six states. Test empty context distinctly. Do not generate reports in this test.
6. **Token-score hook check:** with a synthetic/mock forward output, confirm next-token logits, attention/cache shapes, and deterministic greedy decoding are exposed and that adding a no-op logits hook yields identical token IDs. No contrastive coefficients should be selected in this feasibility stage.
7. **Analyzer/gate interface tests:** confirm the existing analyzer and gate are unchanged; test all five states/actions, state/action metadata retention, abstention without generation, and pairing by `(sample_id, uid, condition)`.
8. **Tiny authorized end-to-end smoke test:** only after load-only and tests pass, request explicit authorization for one UID with one record per state; keep it isolated from frozen Phase 17 results.
9. **Metric compatibility audit:** predefine which project metrics and exact versions will be used; report RadFact, RadGraph-F1, CheXbert and project claim labels separately rather than equating them.
10. **Reproducibility record:** pin code/model revisions, environment, image processing, prompt, decoding and seeds; hash all frozen inputs before and after any future run.

## Sources

- Phase 24 project artifacts: [synthesis](phase24_results_synthesis.md), [paper-ready results](phase24_paper_ready_results.md), [manifest addendum](phase24_manifest_addendum.md), and [experiment manifest](paper_experiment_manifest.md).
- [LLaVA-Rad official paper](https://www.nature.com/articles/s41467-025-58344-x), [official repository](https://github.com/microsoft/LLaVA-Rad), and [IU v3 adapter card](https://huggingface.co/HarsathV/llava-rad-iu-v3).
- [RadHiera MICCAI 2026 paper](https://papers.miccai.org/miccai-2026/paper/1791_paper.pdf) and [author repository](https://github.com/xmed-lab/RadHiera).
- [MediVLM EMNLP 2025 paper](https://aclanthology.org/2025.findings-emnlp.544/) and [public repository](https://github.com/sonai-commits/MediVLM).
- [MAIRA-2 paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/06/2406.04449v1.pdf) and [official model card](https://huggingface.co/microsoft/maira-2).
- [CCD paper](https://arxiv.org/abs/2509.23379) and [implementation](https://github.com/X-iZhang/CCD).
- [R2GenGPT paper](https://arxiv.org/abs/2309.09812) and [official repository](https://github.com/wang-zhanyu/R2GenGPT).
- [Existing BLIP checkpoint card](https://huggingface.co/nathansutton/generate-cxr).
