# Phase 26D Execution Provenance

## Scope and frozen-artifact statement

This document records the newly available Phase 26D execution notebook and the reusable inference code extracted from it. No MAIRA-2 inference was run while preparing this record. No result table was regenerated, and the frozen outputs remain unchanged.

The Phase 26D result artifacts were generated during the documented Colab execution recorded in phase26d_execution.ipynb. The execution notebook was not included in the original c660f21 commit; this provenance checkpoint adds the execution record and extracted generation implementation without changing the frozen results.

The notebook was absent from commit `c660f21aba26f24c56e5a791fc9ce873db67e537` (the commit adds the frozen result artifacts). Its source and saved outputs are now present at `experiments/maira2/phase26d/phase26d_execution.ipynb`.

## Execution environment and model

The notebook uses Google Colab paths and Colab's `userdata` API. Its saved environment output records:

- Python 3.13.15.
- PyTorch 2.11.0+cu130.
- Transformers 4.51.3.
- CUDA available: true.
- GPU: NVIDIA Tesla T4, 14.56 GB reported VRAM.

The PyTorch build identifies CUDA 13.0 (`cu130`); the notebook does not separately record the CUDA driver/runtime version. It does not record the SciPy version used by the later statistical cells.

The model ID is `microsoft/maira-2`. The notebook loads the processor with `AutoProcessor.from_pretrained(..., trust_remote_code=True)` and the model with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, device_map={"": 0}, quantization_config=...)`. The quantization configuration is 4-bit NF4, float16 compute, and double quantization. The model is put in evaluation mode.

The Hugging Face token is retrieved with `google.colab.userdata.get("HF_TOKEN")`; the notebook prints only whether it is available. Inspection found no hard-coded token or credential-like literal. The model repository revision is not pinned. Saved Transformers output warns that custom MAIRA-2 remote-code files were downloaded and that the processor's `use_fast` setting was not pinned.

## Input benchmark, cohort, and image recovery

The input is `results/tables/phase16_perturbations.csv`, a 600-row, 100-study benchmark with six conditions: `sufficient`, `syntactic_incomplete`, `evidentiary_incomplete`, `irrelevant`, `conflicting`, and `insufficient`.

Four studies (UIDs 74, 597, 803, and 885) have no frontal image for any of their six condition rows. The notebook excludes these studies for MAIRA-2, leaving 96 eligible studies and 576 condition-level rows. Its saved split has 16 development studies (96 condition rows) and 80 held-out evaluation studies (480 condition rows), with no study overlap.

The notebook retrieves frontal image bytes from the `ykumards/open-i` Hugging Face dataset's five Parquet shards, using `uid` and `img_frontal`, and writes JPEGs under `external_data/IU-Xray/images/`. It first recovers UID 195 for an input/decoder check, then recovers and validates one image for each of the 80 evaluation studies. The saved integrity audit reports all 80 images recovered and valid.

## Development experiment and frozen lambda policy

The notebook records five development lambda candidates: 0.00, 0.10, 0.25, 0.40, and 0.50, across 16 studies and 96 condition-level records. It selects the smallest lambda satisfying all recorded constraints:

1. Reduce pooled unsupported-claim rate relative to lambda 0.00.
2. Retain at least 80% of the lambda 0.00 supported claims.
3. Increase omissions by no more than 25% relative to lambda 0.00.

The selected value is `lambda = 0.25`, chosen on development only. The evaluation set was not used for lambda selection.

The notebook embeds the five-row development summary directly in its lambda-selection cell. The raw development trajectories and a separate development inference output are not included in this notebook. Therefore the selection rule and recorded summary can be inspected, but the complete development inference/scoring pipeline cannot be reconstructed from this notebook alone.

## Held-out evaluation and generation procedure

The held-out cohort contains 80 studies by six conditions, or 480 study-condition records. The notebook generates two trajectories per record: lambda 0.00 and lambda 0.25, for 960 total trajectories. Its resume-safe key is `(sample_id, condition, lambda)`, and it checkpoints the generation CSV after every trajectory.

The saved log shows that generation resumed with 160 keys already present and generated the remaining 800. The final notebook audit and the frozen CSV both show:

- 960 rows across 80 studies.
- 480 rows at lambda 0.00 and 480 at lambda 0.25.
- 160 rows for each of the six conditions.
- 960 successful statuses, no empty reports, no duplicate keys, and no missing or extra trajectory keys.

The generation uses two processor inputs made with `format_and_preprocess_reporting_input`:

- Context stream: the frontal image and the condition's clinical context as `indication`.
- Visual stream: the same frontal image and `indication=None`.

Both streams are forwarded with cache enabled. At each greedy decoding step the notebook computes:

```text
adjusted_logits = context_logits - lambda * (context_logits - visual_logits)
```

It appends the selected token to both stream caches and stops at the model's configured EOS token or the 128-token maximum. The corrected EOS-terminating implementation is in `src/generation/maira2_contrastive.py`; the extraction does not load a model or add behavior beyond the notebook procedure.

## Lambda-zero verification

The notebook tests UID 195's conflicting-context example against official context-conditioned `model.generate`. An early decoder definition did not stop at EOS and failed exact equivalence when compared with the full official output. The later corrected decoder stops at EOS; the notebook then compares the official generated suffix (after removing prompt tokens) with the custom output. The saved corrected check reports exact token and decoded-text equivalence: 45 generated tokens, with EOS token ID 2 at position 44. This verifies one tested case, not every input.

## Frozen output artifacts and relationship to the notebook

The generation notebook writes `results/maira2_contrastive/phase26d_evaluation/phase26d_evaluation_generation.csv`. The downstream analysis cells read that file and save the following frozen tables:

- Claims: `phase26d_evaluation_claims.csv`, `phase26d_evaluation_claim_summary.csv`, `phase26d_evaluation_claim_summary_by_condition.csv`, and `phase26d_evaluation_paired_records.csv` under `results/tables/phase26d_claim_evaluation/`.
- RadGraph: `phase26d_evaluation_radgraph.csv`, `phase26d_evaluation_radgraph_summary.csv`, `phase26d_evaluation_radgraph_by_condition.csv`, `phase26d_evaluation_radgraph_paired.csv`, and `phase26d_evaluation_radgraph_paired_stats.csv` under `results/tables/phase26d_metrics/`.
- Custom CheXpert-style: `phase26d_evaluation_chexpert.csv`, `phase26d_evaluation_chexpert_summary.csv`, `phase26d_evaluation_chexpert_by_condition.csv`, `phase26d_evaluation_chexpert_paired.csv`, and `phase26d_evaluation_chexpert_paired_stats.csv` under `results/tables/phase26d_metrics/`.
- Synthesis: `phase26d_master_evaluation.csv`, `phase26d_paper_results_table.csv`, and `phase26d_master_by_condition.csv` under `results/tables/phase26d_master/`.
- Study split and selection: `phase26d_maira2_split.csv` under `results/tables/phase26d/`, plus `phase26d_lambda_selection.csv` and `docs/phase26d_lambda_selection_policy.md`.

The notebook contains the code that writes these artifacts and saved output audits/summaries. The generation CSV has 960 rows; claim detail has 3,864 rows and its paired study-condition table has 480 rows; RadGraph and custom CheXpert-style record tables each have 960 rows and their paired tables each have 480 rows. The frozen outputs were checked against the notebook's recorded dimensions, lambda/condition counts, statuses, and named output paths. The 20 tracked Phase 26D result CSVs match the current repository `HEAD` after normalizing Windows CRLF versus Git LF line endings; no Phase 26D numerical artifact was changed during this provenance work.

## Notebook corrections and known limitations

- An early split check used all 600 benchmark rows against a split that excluded four no-frontal-image studies, producing 24 missing split rows. A later cell applies the exclusion and passes the 576-row split audit.
- An early RadGraph cell treats the returned tuple as a dictionary and records a `TypeError`. Later cells inspect the tuple structure, preserve the lambda-zero scores, evaluate lambda 0.25, and save the final record and paired tables.
- The notebook includes historical Git staging/commit/push cells. The saved `git push` attempt failed because credentials were unavailable; the recorded local commit is `c660f21`. Do not execute the notebook top-to-bottom merely to inspect it.
- The notebook contains paired Wilcoxon code in its analysis cells, but does not record the SciPy version. The existing Phase 27 conclusion remains unchanged: the frozen historical RadGraph p-value `0.3363292101` remains a documented reproducibility exception because it is not reproduced exactly from the surviving paired-score artifacts. This checkpoint does not change that conclusion or any frozen p-value.
- `trust_remote_code=True` is used without a pinned model revision, so a future download may not resolve to identical remote code. Runtime package and driver details are not complete enough to guarantee bit-for-bit inference reproduction.
- Notebook cell execution-count metadata is empty even though saved outputs are present. The saved cell outputs provide the recorded logs and checks; they are not a fresh execution performed during this provenance work.

## Inspecting the record

Open `experiments/maira2/phase26d/phase26d_execution.ipynb` and inspect its source and saved outputs without using Run All. Read `src/generation/maira2_contrastive.py` for the extracted, reusable generation path. The lightweight tests validate imports and frozen artifacts without loading MAIRA-2 or accessing a GPU.
