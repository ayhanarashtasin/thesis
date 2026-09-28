---
name: rahc-reproducible-experiments
description: Implement or review RAHC-LoRA configurations, baselines, checkpoints, artifacts, validation, sweeps, and result aggregation. Use for any reproducibility, experiment-control, run-state, or validation request.
---

# RAHC-LoRA reproducible experiments

Read `project.md` sections 11 and 13-23 and `architecture.md` sections 26-35 when this skill applies.

## Configuration and comparability

Use Hydra with typed validation. Resolve configuration once at run start, keep it immutable, and save it with each run and checkpoint. All scientific choices--model/tokenizer revisions, data revisions and splits, task order, seeds, generation, precision, reward definitions, optimizer/scheduler, method switches, and hardware context--must be explicit and recorded.

Method components must be independently configuration-controlled: anchor KL, parameter consolidation, reward-aware importance, factorized importance, conflict gate, and dual controller. Baselines and ablations share one trainer/evaluator and differ only through explicit configuration. Do not create hidden defaults or duplicate loops.

## State, artifacts, and resume

A resumable checkpoint includes LoRA/value-head state, optimizer/scheduler/scaler, task and optimizer progress, RNG states, reference factors, factorized importance, controller state, anchor memory and manifest, resolved configuration, data fingerprints, model/tokenizer revisions, code revision/dirty status, and dependency versions. Save atomically with a manifest and checksums. A missing retention state makes a RAHC-LoRA checkpoint non-resumable.

Write the artifact contract under `outputs/runs/<experiment>/<method>/<task_order>/<seed>/<run_id>/`: resolved config, manifests, structured metrics, performance matrix, general capability scores, layer diagnostics, memory/checkpoint manifests, logs, and summary. Local machine-readable artifacts are authoritative over external trackers.

## Validation and experiment gates

Validate from cheap to expensive: config validation, formatting/lint/type checks, unit tests, integration tests, deterministic smoke runs, then GPU smoke/pilot/main runs. An interrupted two-task run must resume within documented numeric tolerance.

Before claiming an improvement, run the required matched baselines and ablations. The pilot may proceed to the full study only if standard RL-LoRA has at least one prior-task loss of 10 percentage points, RAHC-LoRA reduces it, and current-task reward is within 5 percent of baseline. Otherwise redesign the benchmark or diagnose over-consolidation; do not claim success.

Long GPU, multi-seed, sweep, or paid/cloud launches require user approval after scope is reported. Preserve failed runs unless a verified implementation defect explains them.
