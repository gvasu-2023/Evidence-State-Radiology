# Phase 26D-7 — Lambda Selection Policy

## Purpose

Select the contrastive decoding strength using the development cohort
before evaluation-cohort generation.

## Development cohort

- 16 studies
- 96 condition-level records
- 6 evidence conditions
- Lambda candidates: 0.00, 0.10, 0.25, 0.40, 0.50

## Selection rule

1. Primary objective: reduce pooled unsupported claim rate relative to lambda=0.
2. Supported-claim retention must be >= 80% of lambda=0.
3. Omission increase must be <= 25% relative to lambda=0.
4. Among candidates satisfying all constraints, select the smallest lambda.

## Frozen selection

Selected lambda = 0.25

The evaluation cohort was not used to select lambda.
