# Phase 12C Design Document: Component-Based Context Completeness Parser

## 1. Phase 12B Problem Statement
In Phase 12B, a contextual completeness heuristic (`assess_context_completeness()`) was introduced to classify clinical context strings into `complete` or `incomplete` states. While Phase 12B achieved 81/81 (100%) alignment on the controlled development set of perturbation records (improving upon Phase 12A's 52/81 = 64.20%), subsequent analysis revealed that the Phase 12B heuristic relied on dataset-specific surface patterns. Specifically, Rule 2 assumed that any clinical context beginning with `"History of..."` without explicit primary indicator terms (such as `Chest`, `Dyspnea`, `Pain`, `cough`, `Shortness`) was an incomplete statement missing its primary presenting indication.

While this rule correctly identified the specific 9 truncated history perturbations present in the IU X-Ray perturbation benchmark, it created a severe brittle bias against legitimate standalone medical history indications.

## 2. Evidence from Robustness Audit
A dedicated robustness audit (`results/tables/context_completeness_robustness_audit.csv`) evaluated 34 contexts:
- **14 dataset-derived plausible complete contexts**: 14/14 predicted `complete` (0% false-incomplete).
- **9 actual incomplete perturbation contexts**: 9/9 predicted `incomplete` (100% detection rate).
- **11 controlled standalone complete examples**: 7/11 predicted `complete`, 4/11 predicted `incomplete` (36.36% error rate on standalone history examples).

Overall robustness audit summary across 25 plausible complete contexts:
- **Plausible complete contexts tested**: 25
- **False-incomplete predictions**: 4
- **False-incomplete rate**: 16.00%
- **Root cause**: All 4 failure cases (`"History of mitral valve prolapse."`, `"History of asthma."`, `"History of CHF."`, `"History of stent placement 7+ years ago."`) triggered the rule `history_without_primary_indicator`.

This empirical audit confirmed that the Phase 12B heuristic is dataset-format overfit. Consequently, the 100% Phase 12B result must remain explicitly labeled as a development-set result.

## 3. Why the Current History-Prefix Heuristic is Overfit
The Phase 12B Rule 2 implicitly encoded the assumption that a clinical context must contain *both* a presenting primary complaint *and* a secondary medical history entry. In actual clinical practice and in standard radiology requisition workflows:
1. Requisitions frequently state only a relevant medical history item (e.g., `"History of asthma."`, `"History of CHF."`) as the sole clinical indication for imaging.
2. A single, self-contained clinical statement containing a single valid medical history component is a fully formed, clinically complete context string.
3. Classifying `"History of asthma."` as incomplete merely because it lacks a secondary presenting symptom term like `"chest pain"` conflates *clinical brevity* with *textual completeness*.

Therefore, a complete context evaluator must NOT penalize contexts solely for starting with `"History of..."` or for describing a single clinical entity.

## 4. Proposed Clinical-Component Representation
Phase 12C introduces a transparent **Clinical-Component Representation**. Instead of relying on keyword matching for specific presenting symptoms, the parser breaks down the supplied clinical context string into five recognized clinical component types:

1. **Symptom / Presenting Complaint**: Explicit current signs or symptoms (e.g., `"Chest pain."`, `"Dyspnea."`, `"Cough and fever."`).
2. **Medical History**: Prior diagnoses, surgical histories, or chronic conditions (e.g., `"History of asthma."`, `"History of CHF."`, `"History of mitral valve prolapse."`).
3. **Procedure / Pre-operative Indication**: Pre-procedural screening, pre-op evaluations, or post-operative monitoring (e.g., `"Preoperative evaluation."`, `"Post-op line placement check."`).
4. **Demographic Information**: Patient demographic indicators embedded in the context string (e.g., `"72 year old female with dyspnea."`).
5. **Other Explicit Clinical Indication**: General clinical queries, routine checks, or clear indications (e.g., `"Routine screening."`, `"Evaluate for pneumonia."`).

### Component Definition & Integrity
A clinical context string is composed of one or more components:
$$\text{Context} = \{ C_1, C_2, \dots, C_k \}$$
where each component $C_i$ represents an intact grammatical/semantic clinical unit.

## 5. Proposed Completeness Decision Logic
Under Phase 12C, context completeness is determined by assessing **syntactic and semantic structural integrity (fragmentation)** rather than context category presence.

### Completeness Rule:
A clinical context string $S$ is classified as **`complete`** if:
1. It contains at least **one intact clinical component** ($k \ge 1$), AND
2. It exhibits **no structural fragmentation signals**.

A context string $S$ is classified as **`incomplete`** if it exhibits any of the following observable **fragmentation signals**:
- **Dangling Conjunction / Preposition**: Ends with an unclosed connecting word (e.g., `"History of"`, `"Patient presents with"`, `"Evaluated for"` without a target noun phrase).
- **Sentence Continuity Failure**: Begins with a lower-case initial character without an preceding main clause (e.g., `"cough and fever."`, `"chest pain for 3 days."`), indicating truncation at the start of a string.
- **Mid-phrase Truncation**: Contains incomplete grammatical structures or trailing fragment markers (e.g., `"Patient has history of asthma and"`).

### Crucial Invariance Principle:
- $\text{Count}(C) = 1 \implies \text{Potentially Complete}$ (e.g., `"Preoperative evaluation."`, `"History of CHF."`, `"Chest pain."` are all single-component complete contexts).
- $\text{Prefix} = \text{"History of"} \implies \text{NEVER automatically incomplete}$.

## 6. Anti-Leakage Constraints
To ensure strict scientific validity and prevent data leakage, the Phase 12C completeness parser MUST operate under the following isolation constraints:
1. **Inference-Time Input Only**: The parser receives ONLY the raw input context string `clinical_context`.
2. **Strictly Prohibited Inputs**: The parser MUST NOT receive, access, or inspect:
   - `condition` (experimental condition label, e.g., `incomplete`, `sufficient`)
   - `expected_state`
   - `original_context` (unperturbed context string)
   - `transformation_type`
   - `reference_findings` / `reference_impression`
3. **No Pairwise Ground-Truth Comparison**: The parser must evaluate completeness purely from the observable properties of the string itself, without comparing it against an unperturbed baseline string.

## 7. Scientific Claims Boundaries
To maintain rigorous scientific integrity (in accordance with project rules):
- **Prototype Status**: Phase 12C implements a rule-based deterministic clinical component parser.
- **What CAN be claimed**:
  - The Phase 12C parser evaluates structural integrity and clinical component completeness in context strings.
  - The parser removes dataset-format overfit caused by single-component history heuristics.
  - The parser achieves measurable robustness across both perturbation sets and unperturbed standalone clinical indications.
- **What CANNOT be claimed**:
  - Phase 12C does NOT constitute broad clinical validation by board-certified radiologists.
  - Phase 12C does NOT guarantee 100% generalization across unconstrained external electronic health record (EHR) text without further evaluation.
  - Phase 12C metrics must be described as *deterministic heuristic alignment* and *structural completeness accuracy*, NOT as medical factuality or clinical safety.

## 8. Proposed Phase 12C Tests
The Phase 12C implementation will be verified against a comprehensive test suite in `tests/test_context_completeness_parser.py`:

1. **Standalone Single-Component Complete Context Tests**:
   - Verify `"Chest pain."` $\to$ `complete`
   - Verify `"Dyspnea."` $\to$ `complete`
   - Verify `"History of mitral valve prolapse."` $\to$ `complete`
   - Verify `"History of asthma."` $\to$ `complete`
   - Verify `"History of CHF."` $\to$ `complete`
   - Verify `"History of stent placement 7+ years ago."` $\to$ `complete`
   - Verify `"Preoperative evaluation."` $\to$ `complete`
2. **Fragmented Incomplete Context Tests**:
   - Verify `"History of"` $\to$ `incomplete` (`dangling_preposition`)
   - Verify `"chest pain for 3 days."` $\to$ `incomplete` (`lowercase_initial`)
   - Verify `"Patient presents with"` $\to$ `incomplete` (`dangling_verb_phrase`)
3. **Multi-Component Context Tests**:
   - Verify `"72 year old female with history of CHF presenting with shortness of breath."` $\to$ `complete`
4. **Anti-Leakage Signature Test**:
   - Verify the parser function signature accepts only `(context_text: str)` and operates without external metadata.

## 9. Proposed Phase 12C Evaluation
Phase 12C evaluation will re-run the benchmark across both sets:
1. **Development Set (81 perturbation records)**: Evaluate alignment with controlled perturbation conditions.
2. **Robustness Audit Set (25 plausible complete + 9 incomplete contexts)**: Evaluate false-incomplete and false-complete error rates.

Target performance:
- Maintain high alignment ($\ge 95\%$) on controlled perturbations.
- Reduce robustness audit false-incomplete rate from 16.00% (4/25) to 0.00% (0/25).

## 10. Risks and Limitations
1. **Linguistic Diversity of Requisitions**: Short clinical context strings in real-world EHRs display wide variance (abbreviations, typos, ungrammatical fragments). A pure rule-based parser may encounter unhandled edge cases.
2. **Semantic Ambiguity of Multi-Word Terms**: Distinguishing between a dangling phrase (e.g., `"evaluated for"`) and a complete noun phrase (e.g., `"evaluated for surgery"`) requires explicit term pattern matching or light syntactic analysis.
3. **Dataset Feasibility**: The IU X-Ray dataset clinical indications are predominantly short sentences or phrases (averaging 5-15 words). The observable structure is simple enough for a transparent rule-based component parser, making an immediate large NLP/dependency parser unnecessary for Phase 12C.
