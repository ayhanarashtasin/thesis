# RAHC-LoRA Agent Instructions

## Required skill routing

These instructions apply to Codex, Claude, Agy, and any other coding agent working in this repository. Project skills live in `.agents/skills/`. A prompt does not need to name a skill: before planning, answering, reviewing, or editing, classify the request using the table below and read every matching `SKILL.md` in full. Do not perform the work until the matching skills have been read. If a required skill cannot be read, say so and follow the relevant source documents directly.

| Prompt concerns | Required skill |
| --- | --- |
| Any RAHC-LoRA implementation, design, bug fix, review, scaffold, or phase-status work | `.agents/skills/rahc-project-workflow/SKILL.md` |
| Effective LoRA weights, importance, factorization, penalties, anchor KL, gradient capture/conflict, or dual control | `.agents/skills/rahc-consolidation/SKILL.md` |
| GSM8K, ARC-Challenge, MBPP, constraints, datasets, prompts, rewards, generated-code execution, metrics, evaluation, or statistical analysis | `.agents/skills/rahc-data-evaluation/SKILL.md` |
| Hydra/configuration, baselines, ablations, checkpoints/resume, manifests, logging, sweeps, aggregation, tests, smoke/pilot/main runs, or reproducibility | `.agents/skills/rahc-reproducible-experiments/SKILL.md` |

For a request that spans rows, load all corresponding skills. `rahc-project-workflow` is mandatory for all project work. Do not substitute generic coding guidance for a matching project skill.

## Source authority

Use this precedence order when instructions conflict:

1. Current user instruction.
2. `project.md` - mathematical definitions, scientific rules, experiments, and acceptance criteria.
3. `Goal.md` - phased execution workflow.
4. `architecture.md` - component boundaries, lifecycle, state ownership, and operational detail.
5. Existing tests and documented decisions.
6. Existing implementation.

The repository filenames are case-sensitive on some systems; use the spelling above. If an equation remains ambiguous, select the simplest testable interpretation, record it in `docs/decisions/`, and disclose it.

## Non-negotiable project constraints

- Keep the frozen backbone and one shared LoRA policy; do not introduce per-task adapter growth for the primary method.
- Preserve task/train/validation/anchor/test isolation. Official IFEval is evaluation-only.
- Make method components and ablations configuration-controlled; do not create special training loops or source edits to enable an ablation.
- Any stateful scientific feature needs typed configuration and validation, tests, structured logging, and checkpoint/resume support.
- Do not hide fallbacks, silently skip invalid or non-finite values, invent results, or tune on final-test outcomes.
- Keep long GPU jobs, multi-seed sweeps, and paid/cloud work pending explicit user approval after reporting scope.
- Treat generated code as untrusted: execute it only in an isolated, network-disabled, resource-limited environment.
- Preserve unrelated work. Do not commit, push, expose secrets, or add raw datasets, model weights, or run outputs unless explicitly requested.

## Completion standard

Work in the earliest incomplete phase established from repository evidence. For a meaningful change, report the phase, changed files, scientific behaviour, validation commands/results, limitations, assumptions, and next phase. Do not call the project or an experiment successful without the evidence required by `project.md` and `Goal.md`.
