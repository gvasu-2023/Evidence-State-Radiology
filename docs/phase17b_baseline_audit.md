# Phase 17B Baseline Output Audit

Read-only validation of the completed baseline inference outputs. No reports were regenerated or modified, and no report-quality evaluation was performed.

## Cohort integrity

- Baseline summary rows: **576**.
- Per-record report text files: **576**.
- One-to-one `(sample_id, condition)` correspondence with eligible records: **PASS**; no duplicate or missing keys.
- Excluded UIDs in results: **none** (74, 597, 803, and 885 absent).
- Result status: **576 success**; empty CSV reports: **0**; non-empty reports: **576**.
- Output text readback and CSV report match: **576/576 PASS**.
- Baseline image paths match both benchmark `frontal_image` and eligibility `image_path`: **576/576 PASS**.
- Perturbed contexts match frozen benchmark rows: **576/576 PASS**.
- Unique frontal images checked for existence, readability, PIL verification, and positive dimensions: **96/96 PASS**.
- Duplicate `(sample_id, condition)` keys: **0**.

## Condition counts and report lengths

| Condition | N | Characters min | max | mean | median | Words min | max | mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conflicting | 96 | 261 | 465 | 360.46 | 390.0 | 18 | 71 | 53.47 | 57.0 |
| evidentiary_incomplete | 96 | 234 | 538 | 339.94 | 322.0 | 15 | 74 | 51.58 | 50.0 |
| insufficient | 96 | 218 | 377 | 256.72 | 251.0 | 10 | 57 | 39.05 | 38.0 |
| irrelevant | 96 | 271 | 676 | 369.49 | 357.5 | 30 | 101 | 55.93 | 55.0 |
| sufficient | 96 | 271 | 571 | 360.24 | 341.5 | 16 | 87 | 55.28 | 53.0 |
| syntactic_incomplete | 96 | 246 | 526 | 336.20 | 319.0 | 17 | 80 | 51.96 | 51.0 |

Word count is whitespace-delimited (`str.split()`); character count is the Python string length of the stored `generated_report`. These are descriptive output-size statistics only.

## Exact duplicate reports

- Unique exact report strings: **422 / 576**.
- Duplicate report groups (exact, case-sensitive string equality): **37**.
- Rows participating in duplicate groups: **191**; excess duplicate rows beyond one representative per group: **154**.

| Condition | Duplicate groups | Rows in duplicate groups | Excess duplicate rows |
|---|---:|---:|---:|
| conflicting | 11 | 63 | 52 |
| evidentiary_incomplete | 1 | 2 | 1 |
| insufficient | 5 | 85 | 80 |
| irrelevant | 0 | 0 | 0 |
| sufficient | 0 | 0 | 0 |
| syntactic_incomplete | 3 | 7 | 4 |

Duplicate counts are exact-string matches. Per-condition counts group reports within each condition; they do not classify report quality.

## Input integrity

- `results/tables/phase16_perturbations.csv` SHA-256: `3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53`; expected hash matches.
- `results/tables/phase17_vlm_inference_eligibility.csv` SHA-256: `cf3f8a5c5cb16d663e4eb9f62c1bcdfca2b0f8eb55daab187ef766ccfc1c36a6`; expected hash matches.
- Both input files were read only and remain unchanged.

Detailed per-record checks are in `results/tables/phase17b_baseline_audit.csv`.

No VLM inference, gated generation, or downstream evaluation was run during this audit.
