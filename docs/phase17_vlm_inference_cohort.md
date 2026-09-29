# Phase 17 VLM Inference Eligibility Cohort

The frozen Phase 16 benchmark remains unchanged at **600 records across 100 studies**, with six records per study. This artifact identifies the image-backed subset that can be used for VLM inference.

| Cohort | Studies | Records |
|---|---:|---:|
| Frozen benchmark | 100 | 600 |
| Eligible for VLM inference | 96 | 576 |
| Excluded from VLM inference | 4 | 24 |

The excluded UIDs are **74, 597, 803, and 885**. Each contributes six benchmark records. Their required frontal images are unavailable locally. They are excluded solely with reason `frontal_image_unavailable`; no lateral or substitute images are used.

The eligible cohort is balanced across all six conditions: **96 records each** for `sufficient`, `syntactic_incomplete`, `evidentiary_incomplete`, `irrelevant`, `conflicting`, and `insufficient`.

The eligibility artifact is `results/tables/phase17_vlm_inference_eligibility.csv`. The benchmark stores its frontal reference in `frontal_image`; the artifact exposes this as `image_path`. For the four rows whose benchmark frontal reference is blank, `image_path` records the expected UID-specific frontal filename under the existing IU-Xray image convention so that the missing required file is explicit. The original blank value is preserved in `benchmark_frontal_image`.

For every eligible image, the preparation check confirmed that the referenced local file exists, is non-empty, can be opened and fully read by PIL, passes `PIL.Image.verify()`, and has positive dimensions. These checks are applied once per unique image path and then mapped to its six condition records.

The frozen inputs were verified by SHA-256:

- `results/tables/phase16_perturbations.csv`: `3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53`
- `results/tables/phase16e_selected_studies.csv`: `21144953c8d796a31c937405386d03d52cd933008f765befbc255b2c12d8cd0a`

This is an inference-eligibility manifest only. It does not remove rows from or otherwise revise the frozen benchmark, and it does not start VLM inference.
