# Calibration Evaluation Future Work & Methodological Rationale

## 1. Executive Summary

This document establishes the scientific rationale for omitting **Expected Calibration Error (ECE)** and **Brier Score** from the current Phase 15 evaluation layer, and outlines the required methodology for future probabilistic calibration research on the Evidence-State-Aware Radiology framework.

---

## 2. Current Architecture Characterization

The current V1 research prototype consists of:
1. **Evidence-State Analyzer** ([`src/evidence_state/analyzer.py`](file:///C:/Evidence-State-Radiology/src/evidence_state/analyzer.py)): A deterministic rule-based classifier mapping discrete evidence assessments into five discrete states (`sufficient`, `incomplete`, `irrelevant`, `conflicting`, `insufficient`).
2. **Reliability Gate** ([`src/gating/reliability_gate.py`](file:///C:/Evidence-State-Radiology/src/gating/reliability_gate.py)): A deterministic policy lookup mapping discrete states to discrete reporting actions (`generate`, `qualify`, `discount_context`, `qualify_or_abstain`, `abstain`).

Because both components output deterministic discrete decisions without continuous scalar probabilities or confidence scores $P(\text{state} \mid x) \in [0, 1]$, probability-based calibration metrics cannot be computed directly from the frozen pipeline outputs.

---

## 3. Scientific Integrity Constraints

Per the project's **Scientific Integrity Rules**:
1. *Rule 1*: Never fabricate experimental results.
2. *Rule 5*: Never call a metric a calibration metric unless the implementation actually measures calibration of probabilistic predictions.

Inventing arbitrary confidence values (e.g. mapping `sufficient` $\rightarrow 1.0$ or using step heuristics) to artificially calculate ECE or Brier scores would produce mathematically invalid metrics that violate scientific integrity. Therefore, calibration metrics are strictly deferred to future probabilistic extensions.

---

## 4. Technical Requirements for Calibration Metrics

To implement valid ECE and Brier score evaluation in future iterations:

### A. Expected Calibration Error (ECE)
ECE measures the alignment between predicted model confidence $\hat{P}$ and observed empirical accuracy across $M$ probability bins:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|$$
*Prerequisites*:
- A continuous predicted confidence score $\hat{p}_i \in [0, 1]$ for each prediction instance $i$.
- Binary or multi-class empirical correctness labels $y_i \in \{0, 1\}$.

### B. Brier Score
Brier Score measures the mean squared error of probabilistic predictions:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (f_{ik} - o_{ik})^2$$
where $f_{ik}$ is the predicted probability for state $k$, and $o_{ik}$ is the one-hot indicator of the true state.
*Prerequisites*:
- A full probability distribution vector $[f_{i1}, f_{i2}, \dots, f_{iK}]$ over evidence states for each instance.

---

## 5. Roadmap for Probabilistic Extension

Future research will extend the Evidence-State Analyzer into a probabilistic framework:

1. **Probabilistic Evidence Classifier**: Train a soft evidence classifier (or Bayesian VLM head) that outputs soft state posteriors $P(S = s_k \mid I, C)$.
2. **Temperature Scaling & Isotonic Calibration**: Post-hoc recalibrate predicted state probabilities on a held-out validation set.
3. **Probabilistic Reliability Gating**: Threshold gating decisions dynamically on continuous state probabilities, producing continuous risk-coverage curves and enabling ECE and Brier score computation.
