# Evidence-State-Aware Radiology Report Generation

## Research Question

How does incomplete or conflicting clinical context affect the factual reliability of medical VLM-generated radiology reports, and can evidence-state-aware gating reduce unsupported clinical claims?

## Core Contribution

This project investigates an Evidence-State Analyzer and a model-agnostic Reliability Gate for medical vision-language report generation under varying evidence conditions.

## Evidence States

1. Sufficient
2. Incomplete
3. Irrelevant
4. Conflicting
5. Insufficient

## Experimental Pipeline

CXR Image + Clinical Context
→ Evidence Extraction
→ Evidence-State Analysis
→ Reliability Gate
→ Medical VLM
→ Claim Verification
→ Evaluation

## Evaluation

Primary metrics:

- RadGraph-F1
- Unsupported Clinical Claim Rate
- CheXpert-F1
- Expected Calibration Error
- Brier Score
- Coverage
- Abstention Precision/Recall
- Risk-Coverage

## Status

Research prototype — not intended for clinical deployment.