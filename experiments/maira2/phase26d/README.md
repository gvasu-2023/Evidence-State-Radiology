# Phase 26D MAIRA-2 execution record

## Purpose

This directory preserves the execution notebook for the frozen Phase 26D MAIRA-2 contrastive-decoding experiment. It documents execution provenance; it is not a request to rerun inference or recreate the frozen paper numbers.

## Execution environment

The notebook's saved Colab output records Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.51.3, CUDA available, and an NVIDIA Tesla T4 GPU with 14.56 GB reported VRAM. MAIRA-2 was loaded from `microsoft/maira-2` using 4-bit NF4 quantization, float16 compute, double quantization, and `device_map={"": 0}`. See [the detailed provenance record](../../../docs/phase26d_execution_provenance.md) for limitations and method details.

## Notebook

`phase26d_execution.ipynb` contains the Colab setup, saved environment details, input/image recovery, a decoder check, the development lambda-selection table, resume-safe held-out generation, and downstream analysis code with saved outputs. The notebook has saved outputs but empty execution-count metadata. Inspect its cells without using Run All; it contains historical installation, data-download, and Git commands.

## Source implementation

`src/generation/maira2_contrastive.py` contains the reusable functions extracted from the corrected notebook generation path:

- `prepare_maira_inputs`: creates image-plus-context and image-only processor inputs.
- `two_stream_contrastive_decode`: performs cached two-stream greedy decoding and stops at EOS.
- `generate_maira_trajectory`: combines preparation, both forward passes, decoding, and text decoding.

The module does not load weights. Callers must explicitly provide an already-loaded model and processor.

## Input benchmark

The input is `results/tables/phase16_perturbations.csv`: six condition rows per study across the evidence conditions `sufficient`, `syntactic_incomplete`, `evidentiary_incomplete`, `irrelevant`, `conflicting`, and `insufficient`. Four studies without a frontal image were excluded from MAIRA-2, leaving 96 eligible studies.

## Development split

Sixteen studies (96 condition-level records) formed the development cohort. The notebook evaluates lambda candidates 0.00, 0.10, 0.25, 0.40, and 0.50 and selects the smallest candidate that reduces unsupported-claim rate, retains at least 80% of baseline supported claims, and limits omission increase to 25%. The selected value was lambda 0.25. Selection used development only.

The notebook stores the development summary values directly in its selection cell; separate raw development trajectories are not part of this record.

## Evaluation split

The held-out cohort contains 80 disjoint studies and 480 study-condition rows. It generated two trajectories per row, at lambda 0 and lambda 0.25, for 960 total trajectories. All 960 frozen generation rows have successful status: 480 per lambda and 160 per condition.

## Lambda policy

Lambda 0.25 was frozen using the development cohort before evaluation generation. No evaluation-set tuning was performed. The evaluation experiment compares the frozen baseline lambda 0.00 with the selected lambda 0.25.

## Output artifacts

The saved evaluation generation is `results/maira2_contrastive/phase26d_evaluation/phase26d_evaluation_generation.csv`. Claim outputs are in `results/tables/phase26d_claim_evaluation/`; RadGraph and custom CheXpert-style outputs are in `results/tables/phase26d_metrics/`; combined master and paper-facing tables are in `results/tables/phase26d_master/`. The study split is `results/tables/phase26d/phase26d_maira2_split.csv`.

## How to inspect or reproduce the procedure

To inspect the recorded procedure, read the notebook source and saved outputs, then compare them with the extracted module and frozen CSVs. Lightweight checks are available through `tests/test_phase26d_provenance.py`; they do not load MAIRA-2.

Historical inference should not be rerun merely to recreate the frozen paper numbers. Run a new, separately identified replication experiment only when that is explicitly intended, with its own output destination and recorded environment. Do not overwrite the frozen Phase 26D artifacts.
