# Goal: Implement RAHC-LoRA Step by Step

## Execution prompt for Codex, Claude, Agy, or another capable coding agent

You are the lead research engineer responsible for implementing the RAHC-LoRA project in this repository.

Your task is to convert the specification in `Project.md` into a tested, reproducible, VS Code-ready research codebase. Work through the project in controlled phases. Do not skip directly to the full proposed method.

This file defines how you must execute the work. `Project.md` defines what the system must do, the scientific method, the repository architecture, the experiments, and the acceptance criteria.

---

## 1. Mission

Build a continual reinforcement learning framework for language models that can test whether RAHC-LoRA reduces catastrophic forgetting while preserving new-task learning.

The completed codebase must support:

1. A frozen language-model backbone with one shared LoRA policy.
2. Sequential training across multiple verifiable RL tasks.
3. Standard RL-LoRA and all required retention baselines.
4. Reward-aware importance in effective LoRA weight space.
5. Factorized row and column importance storage.
6. A fixed-size behavioural anchor memory.
7. Old-policy KL or distillation on anchor examples.
8. Layer-level gradient-conflict detection.
9. An adaptive KL retention controller.
10. Task-by-task evaluation, reproducibility, checkpointing, and statistical analysis.

The objective is not merely to make the code run. The objective is to produce scientifically valid evidence about catastrophic forgetting.

---

## 2. Source-of-truth order

Use this priority when instructions conflict:

1. User instructions given in the current conversation.
2. `Project.md`, especially its mathematical definitions and evaluation rules.
3. This `Goal.md` execution workflow.
4. Existing tests and documented architecture decisions.
5. Existing implementation details.

Do not silently reinterpret a mathematical definition. If an equation is ambiguous, implement the simplest testable interpretation, record the decision in `docs/decisions`, and clearly describe the assumption.

---

## 3. Non-negotiable rules

### 3.1 Scientific rules

- Do not invent experiment results.
- Do not claim reduced forgetting until the baseline demonstrates measurable forgetting.
- Do not use final test data for training, reward design, anchor selection, importance estimation, or hyperparameter tuning.
- Report current-task learning together with retention.
- Preserve failed runs and negative results unless failure was caused by a verified implementation error.
- Pair comparisons by task order and random seed.
- Keep compute, rollout count, model, adapter rank, rewards, and evaluation settings comparable between methods.
- Treat joint multi-task training only as an offline reference, not as a valid continual-learning competitor.
- A method that retains old tasks by failing to learn the current task is a failure.

### 3.2 Engineering rules

- Inspect the existing repository before writing code.
- Preserve unrelated user changes.
- Use the `src/rahc_lora` package layout specified in `Project.md`.
- Keep scripts thin. Scientific logic belongs in importable modules.
- Make all experimental decisions configurable.
- Add tests for every scientific equation and controller.
- Keep checkpoints fully resumable.
- Fail loudly on non-finite rewards, losses, gradients, importance values, or controller states.
- Do not hide a fallback, disable a feature silently, or catch an exception without reporting it.
- Do not hardcode model paths, dataset cache locations, GPU IDs, credentials, or task orders.
- Never commit secrets, raw private data, model weights, or large run outputs.
- Do not commit or push Git changes unless the user explicitly requests it.

### 3.3 Compute and safety rules

- Unit tests and tiny smoke tests may run automatically.
- Before launching a long GPU run, a multi-seed sweep, or a paid cloud job, report the estimated scope and obtain user approval when required.
- Generated code must execute only in an isolated environment with no network and strict time, memory, process, and filesystem limits.
- Never run generated code directly inside the training process.
- Never expose credentials or environment-variable values in logs.

---

## 4. Operating procedure

Repeat this workflow for every implementation phase.

### Step 1: Inspect

Before editing:

1. Read `Project.md` completely.
2. Read this file completely.
3. Check for repository-level instructions such as `AGENTS.md`, `CLAUDE.md`, or contributing guidelines.
4. Inspect the repository tree.
5. Inspect Git status without discarding existing work.
6. Identify implemented modules, tests, configurations, and unfinished work.
7. Determine the current implementation phase from evidence, not filenames alone.

### Step 2: Plan

Create a short implementation plan containing:

- Current phase.
- Exact files to add or modify.
- Interfaces being introduced.
- Tests to add first.
- Commands that will validate the work.
- Risks or assumptions.
- Phase exit condition.

Do not create a large plan covering every future code edit. Plan the current phase precisely and keep future phases at a higher level.

### Step 3: Implement

- Write the smallest complete vertical slice that satisfies the current phase.
- Prefer reusable modules over one-off scripts.
- Keep baseline and proposed methods inside the same training and evaluation infrastructure.
- Add configuration and validation together with the feature.
- Add checkpoint serialization for every stateful component.
- Add structured logging for every scientific quantity.

### Step 4: Validate

Run validation from cheapest to most expensive:

1. Static configuration validation.
2. Formatting and linting.
3. Type checks for changed modules.
4. Unit tests.
5. Integration tests.
6. Tiny deterministic smoke test.
7. GPU smoke test, only when relevant and available.
8. Pilot experiment, only after all earlier gates pass.

If a test fails, diagnose the cause. Do not weaken the assertion merely to obtain a passing result.

### Step 5: Report

At the end of every phase, report:

```text
Phase completed:
Files added or changed:
Scientific behaviour implemented:
Tests and commands run:
Results:
Known limitations:
Assumptions recorded:
Next phase:
User decision or resource needed, if any:
```

Update `docs/implementation-status.md` with the same factual status. Do not write experimental conclusions before results exist.

---

## 5. Initial repository assessment

On the first execution, determine which situation applies.

### Situation A: Empty or specification-only repository

Begin with Phase 0 and create the scaffold described in `Project.md`.

### Situation B: Partially implemented repository

Map existing files to the phases below. Run existing tests. Continue from the first phase whose exit condition is not fully satisfied.

### Situation C: Existing implementation with failing tests

Stabilize the earliest failing phase before adding features. Do not build later components on an unreliable baseline.

### Situation D: Existing full method without validated baseline

Pause feature development. Build the missing baseline, evaluation matrix, and ablation switches before trusting the full method.

---

## 6. Phase 0: Repository scaffold and development environment

### Objective

Create a clean, installable, testable Python project that opens and works correctly in VS Code.

### Implement

1. Create the `src/rahc_lora` package structure from `Project.md`.
2. Add `pyproject.toml` with Python 3.11 and separate runtime and development dependencies.
3. Configure Ruff for formatting and linting.
4. Configure mypy for project-owned modules.
5. Configure pytest and test discovery.
6. Add Hydra structured configuration and validation.
7. Add the initial CLI with a `doctor` command.
8. Add structured logging and run-manifest utilities.
9. Add reproducibility utilities for Python, NumPy, and PyTorch seeds.
10. Add `.gitignore`, `.env.example`, and pre-commit configuration.
11. Add `.vscode/settings.json`, `extensions.json`, `launch.json`, and `tasks.json`.
12. Add `README.md` with environment setup and the shortest smoke command.
13. Add `docs/implementation-status.md`.

### Doctor command requirements

The doctor command must report, without exposing secrets:

- Python version.
- Operating system.
- PyTorch version.
- CUDA availability.
- CUDA runtime and GPU names when available.
- BF16 support.
- Installed core dependency versions.
- Writable output and cache locations.
- Configuration loading status.

It must not fail merely because no GPU is present. It should clearly distinguish CPU-only development from GPU experiment readiness.

### Tests

- Package imports from the installed `src` layout.
- Root configuration composes successfully.
- Invalid configuration fails with an actionable error.
- Seed utility reproduces a known random sequence.
- Run manifest serializes and reloads.
- CLI help and doctor command run.

### Exit gate

Do not continue until:

```text
format check passes
lint passes
type check passes for scaffolded modules
unit tests pass
doctor command completes
VS Code uses the project virtual environment
```

---

## 7. Phase 1: Tasks, deterministic rewards, and evaluation

### Objective

Build the measurement system before the training system.

### Mandatory dataset allocation

Use these exact datasets and roles:

| Stage | Task | Dataset | Permitted use |
|---|---|---|---|
| HLoRA reproduction | General language | Fixed versioned subset of The Pile | Supervised reference only |
| HLoRA reproduction | Science | SciQ | Supervised reproduction only |
| HLoRA reproduction | Physical commonsense | PiQA | Supervised reproduction only |
| HLoRA reproduction | Medical | MedMCQA | Supervised reproduction only |
| Continual RL | Mathematics | GSM8K | RL train, validation, anchor candidates, held-out test |
| Continual RL | Science | ARC-Challenge | RL train, validation, anchor candidates, held-out test |
| Continual RL | Code | MBPP | RL train, validation, anchor candidates, held-out test |
| Continual RL | Instruction constraints | Custom Verifiable Constraint Training dataset | RL train, validation, anchor candidates, and held-out custom test |
| General evaluation | Instruction following | Official IFEval | Evaluation only |

Do not substitute SVAMP, SciQ, or IFEval into the main RL stream without a documented architecture decision and user approval. SciQ remains part of the supervised HLoRA reproduction stage.

### Implement

1. Define the `ContinualTask` protocol from `Project.md`.
2. Define the common reward interface.
3. Implement a task registry.
4. Implement the GSM8K mathematics task adapter.
5. Implement the ARC-Challenge science task adapter.
6. Implement the MBPP code task adapter and isolated test runner.
7. Implement the custom verifiable-constraint generator, task adapter, and deterministic validators.
8. Implement deterministic answer normalization.
9. Implement exact-match, multiple-choice, hidden-test, and constraint-fraction rewards.
10. Implement dataset split manifests and fingerprints.
11. Implement official IFEval as an evaluation-only adapter.
12. Implement the task-by-task performance matrix.
13. Implement average forgetting, final average performance, backward transfer, and forward transfer.
14. Add evaluation adapters for the fixed MMLU subset, HellaSwag, ARC-Easy, and held-out perplexity corpus.

### Data rules

- Pin dataset name and revision in configuration.
- Store split identifiers and hashes in manifests.
- Keep training, validation, anchor candidates, and final test examples disjoint.
- Reserve official IFEval entirely for evaluation.
- Do not copy or paraphrase official IFEval prompts when generating constraint-training prompts.
- Version the constraint grammar, templates, generation seed, validators, and prompt hashes.
- Create disjoint training, validation, anchor-candidate, and held-out custom test splits.
- Split the custom constraint dataset by prompt and template family across all splits.
- Check exact and normalized-text overlap between custom constraint prompts and official IFEval.
- Never download data at module import time.
- Make dataset access mockable in unit tests.
- Store only manifests in Git, not the raw datasets.

### Tests

- Hand-written examples for every reward parser.
- Equivalent numeric answers normalize identically when intended.
- Incorrect or unparsable responses receive the documented score.
- Train and test IDs do not overlap.
- Custom constraint prompts have no exact normalized-text overlap with official IFEval.
- Constraint validators correctly score single and combined constraints.
- Continual metrics match hand-calculated matrices.
- Repeated evaluation with fixed generation outputs is deterministic.

### Exit gate

A mocked policy must be evaluable across two tasks, producing a correct performance matrix and summary metrics without any RL training.

---

## 8. Phase 2: Reproducible standard RL-LoRA baseline

### Objective

Produce the ordinary continual RL baseline that establishes whether forgetting exists.

### Implement

1. Add the model and tokenizer factory.
2. Load a configurable causal language model with pinned revisions.
3. Freeze the backbone.
4. Attach one shared LoRA adapter using configurable rank, alpha, dropout, and target modules.
5. Implement or wrap a GRPO rollout and update loop behind project-owned interfaces.
6. Implement robust advantage normalization and clipping.
7. Connect task-specific deterministic rewards.
8. Train one task at a time without resetting the LoRA adapter.
9. Evaluate all learned tasks after every task boundary.
10. Save complete checkpoints using the contract in `Project.md`.
11. Implement resume from checkpoint.
12. Log rollout, reward, loss, KL, token, timing, and memory metrics.

### Design constraints

- External trainer libraries must be wrapped. Do not spread their APIs throughout the project.
- Generation settings must be configuration-controlled and identical during comparable evaluations.
- The baseline must not contain hidden retention logic.
- A CPU-compatible tiny mock or tiny model path must exist for tests.
- GPU-specific optimizations must not change scientific definitions.

### Tests

- Frozen backbone parameters do not change.
- Configured LoRA parameters do change after an update.
- Advantages are finite and normalized as specified.
- One RL update completes on a tiny deterministic setup.
- A two-task run creates the expected performance matrix.
- Checkpoint resume reproduces the uninterrupted run within documented tolerance.

### Exit gate

Complete a reproducible two-task smoke run. Then run a small pilot to determine whether the baseline produces measurable forgetting.

Do not implement the full proposed method until this baseline is trustworthy.

---

## 9. Phase 3: Direct HLoRA-to-RL baseline

### Objective

Create the strongest faithful direct transfer of HLoRA to RL, so the proposed method is compared against more than a weak baseline.

### Implement

1. Reproduce a manageable supervised HLoRA experiment using SciQ, PiQA, MedMCQA, and the fixed Pile subset.
2. Record exact dataset revisions, splits, preprocessing, optimizer, seeds, storage, and importance-computation time.
3. Add a dense importance reference implementation for small models.
4. Adapt the path-integral update to the actual RL loss.
5. Add static layer-level importance coefficients.
6. Save a task-boundary reference policy state.
7. Add the dense parameter regularization term to the common trainer.
8. Expose every assumption through configuration and documentation.
9. Keep this method separate from the proposed reward-aware effective-weight method.

### Tests

- Dense importance has the expected shape and nonnegative values.
- Zero importance produces zero regularization.
- No parameter movement produces zero or near-zero path contribution.
- Reference-state serialization and reload are exact.
- Disabling the baseline returns the standard RL-LoRA objective.

### Exit gate

The supervised reproduction result is documented, and the direct HLoRA-to-RL baseline completes the same two-task smoke experiment using the same trainer, rewards, evaluation, and rollout budget as standard RL-LoRA.

---

## 10. Phase 4: Reward-aware effective-weight importance

### Objective

Implement the first core RAHC-LoRA contribution and verify it numerically before adding memory or controllers.

### Implement

1. Identify every configured LoRA target layer.
2. Compute the effective update `DeltaW = scale * B @ A`.
3. Capture or derive effective-weight gradients from the actual RL objective.
4. Snapshot relevant LoRA factors immediately before the optimizer step.
5. Observe actual post-step effective-weight movement.
6. Implement the reward-aware importance equation in `Project.md`.
7. Apply nonnegative clipping, robust outlier clipping, and FP32 accumulation.
8. Log importance magnitude, sparsity, clipping rate, and layer distribution.
9. Save and restore the importance recorder state.

### Required invariance test

For nonzero scalar `k`:

```text
A_prime = k * A
B_prime = B / k
```

Verify that:

```text
B_prime @ A_prime == B @ A
```

and that effective-weight importance and consolidation behaviour remain equal within numeric tolerance.

### Sensitivity test

On a tiny controlled model:

1. Estimate importance.
2. Perturb high-importance directions.
3. Perturb matched low-importance directions.
4. Measure the old-task loss or behaviour change.

Do not expect perfect correlation, but record whether the estimator contains measurable sensitivity information.

### Exit gate

All invariance, finite-value, checkpoint, and sensitivity tests pass. The recorder can run during the baseline training loop without changing the loss when regularization is disabled.

---

## 11. Phase 5: Factorized importance and efficient penalty

### Objective

Replace persistent dense importance with bounded row and column factors while retaining a dense implementation only as a test oracle.

### Implement

1. Build the nonnegative rank-one factorization from row and column marginals.
2. Store `p_l` and `q_l` per protected layer.
3. Implement dense reconstruction for diagnostics only.
4. Implement the weighted effective-weight penalty.
5. Implement the low-rank Gram-matrix form from `Project.md`.
6. Avoid materializing dense effective-weight differences in the production path.
7. Record factorization error, storage, update time, and ranking correlation.
8. Add dense versus factorized configuration modes.

### Required numerical test

For small random matrices, compare:

```text
dense weighted Frobenius penalty
efficient low-rank Gram penalty
```

The results must agree within a documented floating-point tolerance.

### Exit gate

- Dense and efficient penalties agree.
- Stored importance is nonnegative.
- Persistent storage matches the expected order of `O + I` values per layer for rank one.
- The method completes a two-task smoke run without a dense persistent importance matrix.

---

## 12. Phase 6: Fixed anchor memory and behavioural retention

### Objective

Protect old policy behaviour with a fixed memory budget.

### Implement

1. Add the typed `AnchorRecord` schema.
2. Implement fixed-capacity storage.
3. Implement reservoir sampling for temporal fairness.
4. Add task balancing.
5. Add reward-stratified candidate selection.
6. Add optional diversity selection using embeddings or deterministic hashes.
7. Store compact top-k token distributions and residual probability mass.
8. Implement approximate forward KL or distillation loss.
9. Sample anchor minibatches independently of current-task batches.
10. Log anchor loss, anchor KL, task composition, difficulty composition, and memory bytes.
11. Serialize the complete memory and sampling state.

### Privacy and storage variant

Support a configuration that stores no raw historical prompts. It may use synthetic anchors or compressed statistics. Mark it as a separate experimental variant and do not imply equivalence to ordinary replay.

### Tests

- Capacity is never exceeded.
- Sampling is reproducible from a generator state.
- Rebalancing preserves configured task coverage.
- Top-k distributions are valid probabilities after reconstruction.
- Identical current and reference policies produce near-zero anchor loss.
- A deliberately changed policy produces higher anchor loss.
- Memory survives checkpoint save and resume.

### Exit gate

A controlled destructive update increases old-policy drift, and enabling anchor retention reduces that drift without causing non-finite training.

---

## 13. Phase 7: Conflict-aware layer protection and adaptive KL controller

### Objective

Protect layers only when current learning conflicts with old behaviour, and automatically regulate the behavioural drift budget.

### Implement

1. Compute current-task and anchor losses separately.
2. Capture per-layer effective gradients for both objectives.
3. Calculate epsilon-safe cosine similarity.
4. Set aligned or zero-norm gradients to zero conflict.
5. Combine conflict with layer importance mass.
6. Normalize and clip coefficients across layers.
7. Detach conflict coefficients from autograd.
8. Implement the dual update for `lambda_functional`.
9. Smooth observed anchor KL.
10. Clamp the multiplier and detect persistent saturation.
11. Add fixed-coefficient and static-layer ablation modes.
12. Log per-layer conflicts and controller trajectories.

### Tests

- Identical gradients produce zero conflict.
- Opposite gradients produce maximal positive conflict before clipping.
- Orthogonal gradients produce approximately zero conflict.
- Zero-norm gradients are handled safely.
- Coefficients are detached.
- The dual multiplier rises above the KL target.
- The dual multiplier falls or remains bounded below the target.
- State saves and resumes exactly.

### Exit gate

Synthetic gradient cases behave correctly, and a two-task smoke run shows stable controller values with no NaNs, saturation loops, or uncontrolled memory growth.

---

## 14. Phase 8: Pilot experiment

### Objective

Determine whether the benchmark is suitable and whether the full experiment is justified.

### Required methods

Run at least:

1. Standard RL-LoRA.
2. Balanced replay with the same anchor-memory budget.
3. Direct HLoRA-to-RL.
4. RAHC-LoRA.

### Pilot design

- Use the minimum viable model and response length.
- Use GSM8K, ARC-Challenge, MBPP, and the custom verifiable-constraint dataset.
- Keep official IFEval evaluation-only.
- Use at least two task orders.
- Use three seeds when affordable.
- Keep rollout, optimizer-step, generation, model, and evaluation settings matched.
- Evaluate after every task.
- Preserve the full performance matrix.

### Required diagnostics

- Old-task return.
- Current-task return.
- Average forgetting.
- Final average performance.
- Anchor KL.
- Gradient conflict by layer.
- Importance distribution by layer.
- Output length and diversity.
- Wall-clock time.
- Peak GPU memory.
- Persistent storage.

### Continuation decision

Apply the exact continuation rule from `Project.md`:

- At least one old task must lose 10 or more percentage points under standard RL-LoRA.
- RAHC-LoRA must reduce that loss.
- Current-task reward must remain within 5 percent of standard RL-LoRA.

If forgetting is too small, modify the benchmark before the main study. Do not declare the problem solved.

If retention improves but plasticity fails, diagnose over-consolidation before continuing.

### Exit gate

Create a factual pilot report with:

- Configuration hashes.
- Task orders and seeds.
- Full matrices.
- Uncertainty across runs.
- Failure logs.
- Recommendation to continue, simplify, redesign the benchmark, or stop.

Do not launch the full main experiment without presenting this decision.

---

## 15. Phase 9: Main experiment, baselines, and ablations

### Objective

Produce the evidence required for the thesis claim.

### Main design

- One primary 1B to 3B instruction model.
- Four exact task datasets: GSM8K, ARC-Challenge, MBPP, and custom verifiable constraints.
- Official IFEval reserved for evaluation.
- Three preselected task orders.
- Three random seeds.
- One fixed anchor-memory budget for headline comparisons.
- A second backbone only after the primary result is stable.

### Complete baselines

Implement every baseline required by `Project.md`. If a baseline is infeasible, document the exact reason and provide the closest defensible comparison without calling it equivalent.

### Complete ablations

Run every required ablation from `Project.md`, prioritizing:

1. Remove anchor KL.
2. Remove parameter consolidation.
3. Dense versus factorized importance.
4. Reward-aware versus ordinary importance.
5. Remove conflict gate.
6. Fixed versus adaptive KL coefficient.
7. Memory-size sweep.
8. LoRA-rank sweep.
9. Effective-weight versus factor-level penalty.

### Analysis

1. Aggregate results from raw run artifacts only.
2. Pair by seed and task order.
3. Calculate confidence intervals and effect sizes.
4. Run the predeclared paired statistical test.
5. Correct secondary comparisons for false discovery rate.
6. Produce tables and figures through versioned scripts.
7. Report efficiency and storage beside task performance.
8. Include negative and non-significant findings.

### Exit gate

Every headline result must be reproducible from:

```text
Git revision
resolved configuration
dataset manifests
run manifests
checkpoints
raw performance matrices
aggregation script
```

---

## 16. Debugging order

When a result looks wrong, investigate in this order:

1. Dataset split leakage.
2. Reward parsing and normalization.
3. Generation settings.
4. Frozen versus trainable parameters.
5. Loss signs and coefficient magnitudes.
6. Advantage normalization.
7. Gradient capture correctness.
8. Pre-step and post-step snapshot timing.
9. Effective LoRA scaling.
10. Checkpoint state completeness.
11. Evaluation determinism.
12. Distributed reduction and logging.

Do not first respond to an unexpected result by adding more regularization or tuning more hyperparameters.

---

## 17. Required validation invariants

The codebase must continuously enforce these invariants:

### Model invariants

- Backbone weights remain frozen.
- Only configured LoRA and optional value-head parameters are trainable.
- Effective LoRA weights include `alpha / rank` scaling exactly once.

### Importance invariants

- Persistent importance values are nonnegative.
- Effective-weight importance is invariant to equivalent LoRA factor rescaling.
- Importance accumulators remain finite.
- Disabled importance does not change the baseline objective.

### Memory invariants

- Anchor count and bytes remain within budget.
- No final test example enters memory.
- Stored probability distributions are valid.
- Sampling can be resumed reproducibly.

### Controller invariants

- Conflict coefficients are finite and detached.
- Functional multiplier is nonnegative and bounded.
- Zero-norm gradients do not create NaNs.

### Evaluation invariants

- Every task is evaluated after every completed task.
- Generation settings are identical across comparable methods.
- Full performance matrices are saved.
- Current-task performance and old-task retention are reported together.

---

## 18. Failure handling

### Implementation failure

If code fails:

1. Reproduce with the smallest deterministic case.
2. Add a failing test.
3. Fix the root cause.
4. Run nearby regression tests.
5. Document any changed assumption.

### Numerical failure

If NaNs or infinities occur:

1. Stop the affected optimizer step.
2. Save diagnostic values without secrets or private content.
3. Identify the first non-finite stage.
4. Verify reward, advantage, loss, gradient, optimizer update, importance, and controller states in order.
5. Do not silently skip the batch unless a documented recovery policy exists.

### Out-of-memory failure

Try in this order:

1. Confirm no unintended dense effective-weight tensor is retained.
2. Reduce rollout microbatch size.
3. Enable gradient checkpointing where scientifically neutral.
4. Reduce generation batch size.
5. Reduce response length for smoke testing.
6. Use chunked importance computation.
7. Reduce model size only for development, not silently for final comparisons.

### Scientific failure

If the method does not improve forgetting:

- Do not conceal the result.
- Verify benchmark difficulty and implementation correctness.
- Inspect anchor-only and parameter-only ablations.
- Test whether importance predicts sensitivity.
- Simplify the method if one component provides no consistent value.
- Preserve the negative result as a valid thesis outcome.

---

## 19. Definition of an acceptable pull request or change set

Every meaningful change set must include:

- A clear purpose tied to one project phase.
- Focused source changes.
- Configuration schema updates.
- Validation for invalid configuration.
- Unit tests.
- Integration coverage when training behaviour changes.
- Checkpoint support for new state.
- Logging for new scientific quantities.
- Documentation of assumptions.
- Exact validation commands and results.

Avoid mixing unrelated refactors with scientific changes.

---

## 20. Final deliverables

The completed project must contain:

1. Installable Python package.
2. VS Code development configuration.
3. Pinned and validated experiment configurations.
4. Dataset and split manifests.
5. Standard RL-LoRA baseline.
6. Direct HLoRA-to-RL baseline.
7. RAHC-LoRA implementation.
8. Required baselines and ablations.
9. Unit, integration, regression, and smoke tests.
10. Complete checkpoint and resume support.
11. Full task-by-task performance matrices.
12. Reproducible aggregation and statistical analysis.
13. Runtime, GPU memory, and storage measurements.
14. Method, dataset, experiment, and reproducibility documentation.
15. A factual final report with limitations and negative results.

---

## 21. Final completion check

Before declaring the project complete, verify every item in the `Definition of done` section of `Project.md`.

Then run, at minimum:

```bash
ruff format --check .
ruff check .
mypy src/rahc_lora
pytest -q
python -m rahc_lora.cli doctor
python -m rahc_lora.cli train experiment=smoke method=rl_lora
python -m rahc_lora.cli train experiment=smoke method=rahc_lora
python -m rahc_lora.cli evaluate checkpoint=<smoke-checkpoint>
```

Use the repository's actual documented commands if they differ after scaffolding. Update this file or the README when commands change.

Completion requires passing tests and traceable evidence. The presence of source files alone is not completion.

---

## 22. Start instruction

Begin now with the following actions:

1. Read `Project.md` fully.
2. Inspect the repository and Git status.
3. Identify the earliest incomplete phase.
4. Present a concise plan for that phase.
5. Implement it completely.
6. Run the required validation gates.
7. Report the factual result using the phase report format.
8. Continue to the next phase only when the current exit gate is satisfied.

Do not claim experimental success before running the required experiments. Do not skip the baseline. Do not bypass failed tests. Build the project as a sequence of independently verified scientific components.
