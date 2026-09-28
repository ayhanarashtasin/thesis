---
name: rahc-data-evaluation
description: Build or review RAHC-LoRA task adapters, deterministic rewards, constraint data, isolated code execution, evaluation matrices, and statistical analysis. Use for data, prompts, rewards, benchmarks, evaluation, or result interpretation.
---

# RAHC-LoRA data and evaluation

Read the applicable parts of `project.md` sections 10 and 12-15 plus `architecture.md` sections 14-15 and 25 before changing data, rewards, or evaluation.

## Data roles are hard boundaries

The primary RL stream is GSM8K, ARC-Challenge, MBPP, and a versioned custom verifiable-constraints dataset. It uses three preselected orders. Official IFEval is evaluation-only and must never influence training, anchors, importance, reward development, hyperparameter tuning, or custom-prompt generation.

For every task, keep train, validation, anchor-candidate, and final-test data disjoint. Anchors may come only from the allowed training candidate partition. Pin dataset versions and preserve split IDs/content hashes, preprocessing, reward/validator versions, and generation metadata in manifests.

The custom constraints corpus must be versioned and validator-backed; separate template families across splits; record grammar/template/validator versions and seed; and check exact and normalized overlap against official IFEval. Do not revise it after inspecting IFEval results.

## Reward and task contract

Use the common task protocol, deterministic scoring, documented parsing-failure scores, and documented reward ranges (preferably `[0, 1]`). Preserve raw and normalized rewards. Unit-test hand-authored accepted, rejected, malformed, and normalization-equivalent examples.

MBPP or any generated code must run outside the trainer in an isolated environment with no network and strict CPU, memory, time, process, and filesystem limits. Never execute it in-process or accept an unavailable sandbox silently.

## Evaluation and claims

After every task, save the complete performance matrix `A[i, j]`, where each row is after training task `i` and each column is task `j`. Calculate final average forgetting from each old task's best previous score minus its final score, and calculate final average performance over all tasks. Report current-task learning with retention.

Keep the general-capability suite isolated from training and anchors: IFEval, fixed MMLU subset, HellaSwag, ARC-Easy, and held-out perplexity corpus as configured. Perplexity alone is insufficient.

For comparisons, pair observations by seed and task order. Derive tables, confidence intervals, effect sizes, and paired tests from raw artifacts; correct secondary ablations for false discovery rate; preserve and report negative or non-significant findings. Never report a final row alone as evidence of continual-learning performance.
