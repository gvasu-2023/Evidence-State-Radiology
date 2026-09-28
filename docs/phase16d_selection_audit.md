# Phase 16D Study Selection Audit

## Decision

**Recommendation:** retain the 86 currently selected studies with successful six-state generation and replace the 14 evidentiary-incomplete failures with the 14 deterministic candidates listed below. The resulting selection contains exactly 100 studies and supports a complete six-state benchmark under the current generator. This is a selection recommendation only; Phase 16C files remain unchanged.

## Candidate pool and screening

- Phase 16A inventory: 3,851 studies; non-pilot usable candidate pool: **3,258**.
- Phase 16A usability filter, matching `select_phase16b_studies.py`: not in pilot; context, findings, and impression present; at least one frontal or lateral image.
- Current selection: **{len(current)}** studies, originally selected with seed `20260928` by sorted-UID pandas sampling.
- Current evidentiary-incomplete successes: **{len(preserved)}**; failures: **{len(failed)}**.
- Evidentiary-eligible candidates under the existing `generate_evidentiary_incomplete` helper: **2,193**; ineligible: **1,065**. Eligible candidates outside the current selection: **2,107**.
- Generator intersection: syntactic construction and parser rejection succeeded for 3,258/3,258; explicit-reference conflict construction succeeded for 3,141/3,258; all three individual generators (syntactic, evidentiary, conflict) succeeded for **2,119** candidates. Of these, **2,024** have a frontal image.
- A higher-priority replacement pool of **228** unused candidates meets all three generators, has frontal imaging, uses clause-level clinical evidence removal, and retains at least one non-generic clinical term from the existing context-parser vocabulary in the residual context.

### Evidentiary eligibility rule

This audit uses the current generator behavior rather than a new diagnosis ontology. A candidate is marked evidentiary eligible when the existing helper returns `success`, a non-empty `removed_evidence_text`, a non-empty residual context, and the existing completeness parser accepts that residual. The helper itself screens out demographics/placeholders and requires readable evidence using the parser keywords plus its explicit extra-term list. The CSV records the helper method, removed term, and generated residual for every candidate.

A successful helper result is a construction-level screen, not independent clinical adjudication. In particular, the current generator permits `readable_concept_generalization`, including generic outputs such as ?Clinical evaluation.? Those cases remain marked eligible to faithfully report current generator behavior. The replacement rule below prioritizes clause removals that retain a specific readable indication in the residual.

## Current selection

The audited current 100 UIDs are: 169, 181, 220, 253, 280, 282, 317, 327, 341, 343, 356, 366, 387, 388, 443, 455, 483, 503, 521, 547, 784, 817, 834, 884, 982, 995, 1025, 1087, 1095, 1166, 1264, 1273, 1360, 1369, 1383, 1388, 1422, 1491, 1518, 1519, 1556, 1568, 1625, 1630, 1667, 1676, 1683, 1689, 1710, 1730, 1796, 1854, 1864, 1894, 1916, 1994, 2147, 2165, 2203, 2207, 2216, 2270, 2328, 2339, 2351, 2397, 2445, 2480, 2535, 2561, 2589, 2610, 2684, 2724, 2822, 2828, 2967, 3023, 3024, 3077, 3120, 3143, 3165, 3171, 3211, 3216, 3414, 3417, 3508, 3515, 3526, 3588, 3676, 3687, 3796, 3818, 3834, 3958, 3977, 3994.

The 86 currently valid UIDs retained in the recommendation are: 169, 181, 220, 253, 280, 282, 317, 327, 341, 343, 356, 366, 387, 388, 443, 455, 483, 503, 521, 547, 784, 817, 834, 995, 1025, 1087, 1095, 1166, 1264, 1273, 1360, 1369, 1383, 1388, 1491, 1518, 1519, 1568, 1625, 1667, 1676, 1683, 1710, 1796, 1854, 1864, 1894, 1916, 2165, 2207, 2216, 2270, 2328, 2351, 2397, 2445, 2480, 2535, 2561, 2589, 2610, 2684, 2724, 2822, 2828, 2967, 3023, 3024, 3077, 3120, 3143, 3165, 3171, 3211, 3216, 3414, 3417, 3508, 3526, 3588, 3796, 3818, 3834, 3958, 3977, 3994. The existing Phase 16A eligibility allows either view; one retained study has lateral imaging without a frontal image. All 14 proposed replacements have frontal imaging.

## The 14 failed selected studies

| UID | Original context | Failure reason | Recommendation |
|---:|---|---|---|
| 884 | XXXX-year-old XXXX XXXX, persistent XXXX.. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 982 | XXXX-year-old female with chronic XXXX | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 1422 | XXXX-year-old male with XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 1556 | XXXX-year-old male with history of XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 1630 | XXXX for one XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 1689 | XXXX-year-old with XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 1730 | XXXX | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 1994 | XXXX and nonproductive XXXX x1 XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 2147 | XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 2203 | Persistent XXXX | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 2339 | Pre-op evaluation, XXXX surgery. | `no_safe_evidence_removal` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 3515 | XXXX-XXXX | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 3676 | XXXX-year-old male with XXXX. | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |
| 3687 | XXXX | `no_readable_clinical_evidence` | Remove from the revised selection; no safe evidentiary transformation under current helper. |

Thirteen failures return `no_readable_clinical_evidence`: their context is placeholder-only, demographic-only, or otherwise contains no readable clinical evidence recognized by the existing helper. UID 2339 returns `no_safe_evidence_removal`: ?Pre-op evaluation, XXXX surgery.? has readable preoperative evidence, but the helper cannot remove/generalize it while producing a defensible accepted residual. All 14 should be removed from this selection if the current generator remains fixed.

## Recommended replacements

The replacements are unused non-pilot Phase 16A candidates. Each has frontal imaging, passes the existing syntactic and conflict helpers, and succeeds in evidentiary construction by removing a readable clause while leaving a parser-complete residual with a specific existing-parser clinical term. Conflict targets are backed by the cited reference text.

| UID | Original context | Removed evidence | Residual context | Conflict target | Reference support |
|---:|---|---|---|---|---|
| 49 | XXXX-year-old with osteoarthritis of the hip scheduled for total hip replacement. Preoperative evaluation. | osteoarthritis of the hip scheduled for total hip replacement | XXXX-year-old, Preoperative evaluation. | pneumothorax_absence | there is no pleural effusion or pneumothorax |
| 53 | Dizziness, hypoxia. | Dizziness | hypoxia. | pneumothorax_absence | there is no pneumothorax |
| 67 | XXXX-year-old, MVA, chest pain | MVA | XXXX-year-old, chest pain. | pleural_effusion_absence | no focal infiltrate or effusion |
| 175 | Nausea and vomiting | Nausea | vomiting. | pneumothorax_absence | no pneumothorax |
| 185 | HYPERTENSION; preop hernia repair | HYPERTENSION | preop hernia repair. | pleural_effusion_absence | there is no acute infiltrate or pleural effusion |
| 195 | XXXX-year-old female with coughing and wheezing. | coughing | XXXX-year-old female, wheezing. | pneumothorax_absence | no pneumothorax or pleural effusion |
| 227 | XXXX-year-old male with dyspnea, chest pain | dyspnea | XXXX-year-old male, chest pain. | pneumothorax_absence | mild diffuse interstitial opacities bilaterally, predominantly in the bases, with no focal consolidation, pleural effusion, or pneumothoraces |
| 247 | Chest pain, dyspnea. | Chest pain | dyspnea. | pneumothorax_absence | no pneumothorax or large pleural effusion |
| 254 | XXXX-year-old female MVA, chest pain | XXXX-year-old female MVA | chest pain. | pneumonia_absence | lungs are clear \| no acute cardiopulmonary finding |
| 270 | Chest pain, renal failure. Anterior chest pain. Dehydrated, EtOH. | Chest pain | renal failure, Anterior chest pain, Dehydrated, EtOH. | pneumothorax_presence | the lungs are clear of focal airspace disease, pneumothorax, or pleural effusion |
| 275 | Chest pain, dyspnea | Chest pain | dyspnea. | pneumothorax_absence | there is no focal consolidation, pleural effusion, or pneumothoraces |
| 287 | Dyspnea, nausea and vomiting, XXXX | Dyspnea | nausea, vomiting, XXXX. | pneumothorax_absence | no pleural effusions or pneumothoraces |
| 290 | Occasional chest pain and shortness of breath. | Occasional chest pain | shortness of breath. | pneumonia_absence | the lungs are clear |
| 297 | Chest pain, dyspnea. | Chest pain | dyspnea. | pneumothorax_absence | there is no pneumothorax or pleural effusion |

## Deterministic selection procedure

1. Preserve every currently selected study whose six Phase 16C records succeeded (86 studies).
2. Start from the 3,258 Phase 16A usable non-pilot candidates and exclude all currently selected UIDs.
3. Filter by the existing evidentiary helper success and non-empty readable removed evidence; require syntactic helper success with parser-incomplete output; require explicit-reference conflict helper success; require frontal imaging.
4. As a replacement preference, retain only clause-removal cases whose generated residual contains a specific symptom/history/procedure term from the existing parser vocabulary. This yields 228 unused candidates.
5. Sort by integer UID ascending and take the first 14. This produces the replacement UIDs shown above without a random draw. The original selection seed `20260928` is unchanged in project records; it is not applied to this sorted top-up because no sampling is used.
6. Run the existing irrelevant donor assignment on the proposed 100-study set. It produces a complete one-to-one assignment for all 100 targets with 100 distinct donors, so no target is its own donor and the current overlap screen is satisfied. Insufficient context is constructed as empty by the existing generator; sufficient context retains original context.

### Replacement UIDs

49, 53, 67, 175, 185, 195, 227, 247, 254, 270, 275, 287, 290, 297

### Recommended final 100 UIDs

49, 53, 67, 169, 175, 181, 185, 195, 220, 227, 247, 253, 254, 270, 275, 280, 282, 287, 290, 297, 317, 327, 341, 343, 356, 366, 387, 388, 443, 455, 483, 503, 521, 547, 784, 817, 834, 995, 1025, 1087, 1095, 1166, 1264, 1273, 1360, 1369, 1383, 1388, 1491, 1518, 1519, 1568, 1625, 1667, 1676, 1683, 1710, 1796, 1854, 1864, 1894, 1916, 2165, 2207, 2216, 2270, 2328, 2351, 2397, 2445, 2480, 2535, 2561, 2589, 2610, 2684, 2724, 2822, 2828, 2967, 3023, 3024, 3077, 3120, 3143, 3165, 3171, 3211, 3216, 3414, 3417, 3508, 3526, 3588, 3796, 3818, 3834, 3958, 3977, 3994

## Expected result and scope

- Recommended selection size: **100** studies.
- Six-state construction capability: **100/100** by current deterministic generator/helper checks, including a complete 100-target irrelevant donor matching.
- At least 100 suitable studies exist: **yes**. The stricter unused replacement pool alone contains 228 candidates.
- No generator, benchmark, test, manifest, baseline, or gating files were modified. No VLM inference or tests were run. The only created outputs are this audit report and `results/tables/phase16d_candidate_audit.csv`.
- This audit assesses deterministic constructibility and input availability. It does not claim model performance or independent clinical validation of every synthetic context.
