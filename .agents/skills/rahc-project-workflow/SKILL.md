---
name: rahc-project-workflow
description: Execute, review, or plan RAHC-LoRA repository work under its phased research-engineering workflow. Use for any task that changes or assesses this project; pair it with the specialized project skill for the affected subsystem.
---

# RAHC-LoRA project workflow

Apply the project-specific operating rules before doing RAHC-LoRA work.

## Establish scope and authority

1. Read `AGENTS.md` and identify any additionally matching project skills.
2. Treat `project.md` as authoritative for mathematics, data separation, experiment design, and done criteria; use `Goal.md` for phase execution and `architecture.md` for component and state boundaries.
3. Inspect the repository and tests to establish the earliest incomplete phase. Do not infer completion from filenames or scaffolding alone.
4. Keep the current request limited to that phase unless the user explicitly changes scope.

The primary method has one frozen backbone and one shared LoRA adapter. It measures both stability and plasticity. No result is a success merely because old-task performance is retained.

## Change contract

For a scientific or training-related change, include the smallest complete vertical slice:

- typed, validated configuration for every experimental choice;
- focused unit coverage for equations, state transitions, or invariants;
- integration coverage when training behaviour changes;
- structured metrics for new scientific state;
- checkpoint serialization and restoration for persistent state;
- configuration-only component toggles when the feature has an ablation.

Keep scripts and CLI entry points thin. Place reusable logic in `src/rahc_lora`, preserve the dependency direction in `architecture.md`, and avoid task-specific logic in generic orchestration.

## Required scientific behaviour

- Never leak final-test examples into training, anchors, importance, reward design, or tuning.
- Compare methods under matched backbone, LoRA configuration, rewards, rollout/step budget, task order, seed, generation settings, and memory budget where relevant.
- Fail loudly and diagnostically on non-finite scientific values; do not silently fall back or skip a batch.
- Run cheap validation before expensive validation. Do not start long GPU, cloud, or multi-seed work without the user's approval.
- Preserve failures and negative outcomes unless a verified implementation defect explains them.

## Report

For phase work, report: phase; files changed; scientific behaviour; validation commands and factual results; known limitations; recorded assumptions; next phase; and any required user decision.
