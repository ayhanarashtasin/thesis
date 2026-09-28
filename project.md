# RAHC-LoRA

## Reward-Aware Hierarchical Consolidation for Continual Reinforcement Learning of Language Models

**Document role:** Technical source of truth for implementation, experiments, and reproducibility  
**Primary environment:** VS Code  
**Language:** Python 3.11  
**Project type:** Research codebase and thesis experiment platform  
**Status:** Initial implementation specification

---

## 1. Project summary

RAHC-LoRA is a continual reinforcement learning method for language models. It is designed to let one LoRA policy learn a sequence of verifiable tasks while reducing catastrophic forgetting of earlier tasks and general model capabilities.

The method combines four protections:

1. Reward-aware importance estimation in effective LoRA weight space.
2. A fixed-size memory of old prompts and reference policy behaviour.
3. Layer-level gradient-conflict detection.
4. An adaptive KL controller that limits old-policy drift.

The final system must preserve old skills without preventing the model from learning the current task. Retention alone is not success.

---

## 2. Problem statement

Sequential RL fine-tuning can overwrite behaviours learned during previous tasks. For example, a model trained on mathematics, then science, then code may improve on code while its earlier mathematics performance falls sharply.

The project addresses the stability-plasticity problem:

- **Stability:** retain previous task performance and general capabilities.
- **Plasticity:** continue learning the current task effectively.
- **Efficiency:** avoid storing a full model, full replay dataset, or dense importance matrix for every task.

The exact problem is:

> Given a frozen language-model backbone and one shared LoRA policy trained on a sequence of RL tasks, reduce old-task forgetting while maintaining current-task learning under fixed memory and compute budgets.

---

## 3. Primary research question

> Can reward-aware parameter consolidation combined with old-policy behavioural constraints reduce catastrophic forgetting during sequential RL fine-tuning of language models without materially reducing new-task learning?

### 3.1 Secondary questions

1. Does advantage-aware importance outperform ordinary path-integral importance?
2. Is importance in effective LoRA weight space more reliable than importance assigned independently to the LoRA factors?
3. Does conflict-aware layer protection outperform static layer coefficients?
4. How much fixed anchor memory is required?
5. How much accuracy is lost by factorizing the importance matrix?
6. Does behavioural KL explain most of the retention benefit, or does parameter consolidation add a measurable improvement?
7. Does the method remain effective across task orders and random seeds?

---

## 4. Hypotheses and success criteria

### 4.1 Primary hypothesis

RAHC-LoRA will reduce average forgetting relative to standard RL-LoRA and a direct HLoRA-to-RL baseline while preserving competitive final average performance.

### 4.2 Co-primary outcomes

Both outcomes are required:

1. **Average forgetting:** lower is better.
2. **Final average task performance:** higher is better.

A method that preserves old tasks by failing to learn new tasks is not successful.

### 4.3 Pilot continuation rule

Continue to the full experiment if:

- At least one prior task loses 10 or more percentage points under the standard RL-LoRA baseline.
- RAHC-LoRA reduces that loss.
- Current-task reward falls by no more than 5 percent relative to standard RL-LoRA.

If the pilot does not produce measurable forgetting, increase task conflict, training duration, learning rate, or task-stream length before judging the method.

### 4.4 Thesis-level success

The thesis is successful if it produces a reproducible answer, including a negative result. Acceptable outcomes include:

- The full method outperforms all practical baselines.
- Behavioural KL explains most retention and parameter importance adds little.
- Factorized importance works with little or no anchor replay.
- The proposed importance estimator fails under policy-gradient noise, clearly identifying the boundary of HLoRA-style consolidation in RL.

---

## 5. Scope

### 5.1 Recommended main study

- One 1B to 3B instruction-tuned causal language model.
- Frozen base weights.
- One shared LoRA adapter trained across all tasks.
- GRPO as the default RL algorithm.
- Four sequential task families.
- Three task orders.
- Three random seeds per main condition.
- A fixed anchor-memory budget.
- Full evaluation after every task.

### 5.2 Minimum viable study

- One 0.5B to 1.5B model.
- Three task families.
- Two task orders.
- Three seeds.
- Four core baselines.
- Dense importance used only as a small-model reference.

### 5.3 Non-goals

- Training a foundation model from scratch.
- Full-parameter continual fine-tuning in the main study.
- Claiming that catastrophic forgetting is solved universally.
- Claiming state-of-the-art performance from a single task order.
- Storing an unlimited replay buffer.
- Using held-out test examples for training, memory selection, hyperparameter tuning, or importance estimation.
- Supporting every RL algorithm in the first implementation.

---

## 6. Conceptual architecture

```text
Current task data
    -> rollout engine
    -> policy responses
    -> deterministic reward adapters
    -> normalized advantages
    -> current RL objective

Frozen backbone + shared LoRA policy
    -> effective-weight tracker
    -> reward-aware importance recorder
    -> row and column importance factors

Fixed anchor memory
    -> stored prompts and compact reference distributions
    -> old-behaviour KL objective
    -> anchor gradient

Current RL gradient + anchor gradient + importance state
    -> layer conflict controller
    -> adaptive retention controller
    -> regularized LoRA optimizer update

After every task
    -> consolidate importance
    -> save reference LoRA state
    -> refresh fixed-size anchor memory
    -> evaluate every learned task and general-capability suite
```

---

## 7. Method definition

### 7.1 Combined objective

For current task `t`, optimize:

```text
L_total = L_RL_current
        + lambda_functional * L_anchor_KL
        + lambda_parameter * sum_l(c_l * R_l)
```

Where:

- `L_RL_current` is GRPO, PPO, or another policy-gradient objective.
- `L_anchor_KL` measures behavioural drift on stored old-task anchors.
- `R_l` is the effective-weight consolidation penalty for layer `l`.
- `c_l` is the detached conflict-aware layer coefficient.
- `lambda_functional` is controlled by an old-behaviour drift budget.
- `lambda_parameter` controls parameter consolidation globally.

All regularization terms must be independently switchable in configuration.

### 7.2 Effective LoRA weight

For a LoRA layer:

```text
DeltaW_l = scale_l * B_l @ A_l
scale_l = alpha_l / rank_l
```

Importance and consolidation are defined on `DeltaW_l`, not independently on `A_l` and `B_l`.

This is required because the transformation below leaves the effective LoRA update unchanged:

```text
A_l' = k * A_l
B_l' = B_l / k
B_l' @ A_l' = B_l @ A_l
```

The implementation must include a numerical test confirming that the effective-weight penalty is invariant to this rescaling.

### 7.3 Reward-aware importance

For optimizer step `s` and layer `l`:

```text
delta_step_l = DeltaW_l_after_step - DeltaW_l_before_step

raw_score_l = relu(
    -effective_gradient_l * delta_step_l
) / (delta_step_l^2 + epsilon)

score_l = beta * score_l
        + (1 - beta) * robust_clip(raw_score_l)
```

Requirements:

- The effective gradient comes from the actual RL loss using normalized, clipped advantages.
- The actual optimizer step is observed. Do not reconstruct a hypothetical SGD update.
- Negative contributions are clipped to zero.
- Accumulators run in FP32 even when model training uses BF16.
- Non-finite values must fail the step with a useful diagnostic.
- Score normalization and clipping thresholds must be logged.

### 7.4 Factorized importance

Persistent dense importance is not allowed in the final main method. Approximate the nonnegative importance matrix as:

```text
S_l ~= p_l @ q_l.T
```

Where:

- `p_l` is a nonnegative output-channel vector.
- `q_l` is a nonnegative input-channel vector.

The initial implementation should use a rank-one nonnegative approximation based on row and column marginals. Higher-rank nonnegative factorization is a publication extension, not an MVP requirement.

The factorization module must report:

- Dense-to-factorized reconstruction error during small-model tests.
- Storage in bytes.
- Time required to update the factors.
- Correlation between dense and factorized importance rankings.

### 7.5 Effective-weight consolidation penalty

At a task boundary, store the reference LoRA factors `A_ref` and `B_ref`. For the current factors:

```text
D_l = scale_l * (B_l @ A_l - B_ref_l @ A_ref_l)

R_l = || diag(sqrt(p_l))
         @ D_l
         @ diag(sqrt(q_l)) ||_F^2
```

The production implementation should compute this penalty through low-rank Gram matrices instead of materializing the dense `D_l` matrix.

One valid formulation is:

```text
X = [diag(sqrt(p)) @ B, -diag(sqrt(p)) @ B_ref]
Y = [A @ diag(sqrt(q)); A_ref @ diag(sqrt(q))]

R = || X @ Y ||_F^2
  = trace((X.T @ X) @ (Y @ Y.T))
```

Apply the LoRA scaling consistently. Unit tests must compare the efficient result against direct dense computation on small random matrices.

### 7.6 Anchor memory

The anchor memory has one global fixed budget. It must not grow linearly with the number of tasks.

Each anchor record contains:

```text
anchor_id
task_id
prompt_or_state
reference_completion, if used
reference_token_ids
reference_topk_token_ids
reference_topk_logprobs
reference_residual_probability_mass
reward_at_capture
return_or_score
selection_embedding_or_hash
capture_policy_version
metadata
```

Use compact top-k reference distributions or selected-token log probabilities. Do not store a full vocabulary distribution for every token.

Selection policy:

1. Maintain temporal fairness with reservoir sampling.
2. Maintain task balance.
3. Prefer diverse prompts using k-center or farthest-first selection when embeddings are available.
4. Preserve a mixture of easy, medium, and hard examples using reward strata.
5. Never select from the final test set.

Memory sizes for ablation:

```text
0, 128, 512, 2048 anchors
```

### 7.7 Behavioural retention loss

On an anchor minibatch, compare the current policy with the stored compact reference distribution.

Preferred order:

1. Approximate forward KL over stored top-k tokens plus residual mass.
2. Token-level distillation cross-entropy if stable KL is not available.
3. Sequence-level reference log-probability matching as a fallback.

Log both the training anchor loss and held-out old-task return. Low anchor KL alone does not prove that old-task behaviour is preserved.

### 7.8 Layer conflict coefficient

For each protected layer:

```text
conflict_l = max(
    0,
    -cosine(effective_grad_RL_l, effective_grad_anchor_l)
)

importance_mass_l = sqrt(sum(p_l) * sum(q_l))

c_l = normalize_across_layers(
    importance_mass_l * conflict_l
)
```

Requirements:

- Calculate conflict in effective-weight or functional-gradient space where feasible.
- Detach `c_l` before it weights the regularization term.
- Use epsilon-safe norms.
- Return zero conflict when either gradient norm is below the configured threshold.
- Log the layer conflict distribution.
- Provide an ablation using static layer coefficients.

### 7.9 Adaptive functional-retention controller

Maintain a nonnegative dual variable:

```text
lambda_functional = max(
    0,
    lambda_functional
    + dual_lr * (observed_anchor_KL - target_KL)
)
```

Requirements:

- Smooth the observed KL with an exponential moving average.
- Clamp the multiplier to a configured safe maximum.
- Log the raw KL, smoothed KL, target, and multiplier.
- Provide a fixed-coefficient ablation.
- Stop the run if the multiplier saturates and the KL remains above budget for a configured number of steps.

---

## 8. Training algorithm

```text
Input:
    frozen backbone
    shared LoRA policy
    ordered task stream D_1 ... D_T
    fixed memory budget B

Initialize:
    LoRA parameters
    optimizer and scheduler
    empty anchor memory
    zero factorized importance
    zero or configured functional multiplier

For each task t:
    load only the current task's training split
    snapshot the current consolidated LoRA reference

    For each RL update:
        collect responses or trajectories
        calculate deterministic rewards
        robustly normalize and clip advantages
        calculate current RL loss and effective gradients

        sample anchors if memory is nonempty
        calculate anchor KL and effective anchor gradients

        calculate detached layer conflicts
        calculate effective-weight consolidation penalties
        combine the losses

        snapshot pre-step LoRA factors
        run the optimizer step
        observe the actual effective-weight movement
        update transient importance statistics
        update the dual retention multiplier

        log learning, retention, stability, and systems metrics

    consolidate importance into row and column factors
    store the new reference LoRA factors
    select anchors from the completed task
    enforce the global memory budget
    evaluate all learned tasks and the general-capability suite
    save a complete resumable checkpoint

Output:
    one continual LoRA policy
    compact consolidation state
    fixed anchor memory
    complete performance matrix and experiment manifest
```

---

## 9. Repository architecture

```text
rahc-lora/
├── Project.md
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock                         # Commit when uv is used
├── Makefile                        # Optional convenience commands
├── .env.example                    # Names only, never real secrets
├── .gitignore
├── .pre-commit-config.yaml
├── .vscode/
│   ├── extensions.json
│   ├── settings.json
│   ├── launch.json
│   └── tasks.json
├── configs/
│   ├── config.yaml                 # Root Hydra configuration
│   ├── model/
│   │   ├── small.yaml
│   │   └── confirmation.yaml
│   ├── method/
│   │   ├── rl_lora.yaml
│   │   ├── replay.yaml
│   │   ├── ewc_lora.yaml
│   │   ├── hlora_rl.yaml
│   │   └── rahc_lora.yaml
│   ├── task/
│   │   ├── math.yaml
│   │   ├── science.yaml
│   │   ├── code.yaml
│   │   └── constraints.yaml
│   ├── task_stream/
│   │   ├── order_1.yaml
│   │   ├── order_2.yaml
│   │   └── order_3.yaml
│   ├── experiment/
│   │   ├── smoke.yaml
│   │   ├── pilot.yaml
│   │   ├── main.yaml
│   │   └── ablations.yaml
│   └── launcher/
│       ├── local.yaml
│       └── slurm.yaml
├── src/
│   └── rahc_lora/
│       ├── __init__.py
│       ├── cli.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   └── validation.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── factory.py
│       │   ├── lora_policy.py
│       │   ├── value_head.py
│       │   └── checkpoints.py
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── rollout_engine.py
│       │   ├── grpo_trainer.py
│       │   ├── advantages.py
│       │   ├── sampling.py
│       │   └── losses.py
│       ├── consolidation/
│       │   ├── __init__.py
│       │   ├── effective_weights.py
│       │   ├── gradient_capture.py
│       │   ├── importance.py
│       │   ├── factorization.py
│       │   ├── penalty.py
│       │   ├── conflict.py
│       │   ├── dual_controller.py
│       │   └── state.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── anchor_memory.py
│       │   ├── selection.py
│       │   ├── compact_logits.py
│       │   └── distillation.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── math_task.py
│       │   ├── science_task.py
│       │   ├── code_task.py
│       │   ├── constraint_task.py
│       │   └── constraint_generator.py
│       ├── rewards/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── exact_match.py
│       │   ├── multiple_choice.py
│       │   ├── code_tests.py
│       │   └── instruction_constraints.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   ├── performance_matrix.py
│       │   ├── continual_metrics.py
│       │   ├── general_capabilities.py
│       │   └── efficiency.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── continual_trainer.py
│       │   ├── task_boundary.py
│       │   ├── callbacks.py
│       │   └── resume.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── reproducibility.py
│           ├── distributed.py
│           ├── numerics.py
│           └── manifests.py
├── scripts/
│   ├── doctor.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_pilot.py
│   ├── run_main_matrix.py
│   ├── aggregate_results.py
│   └── profile_memory.py
├── tests/
│   ├── unit/
│   │   ├── test_effective_weights.py
│   │   ├── test_importance.py
│   │   ├── test_factorization.py
│   │   ├── test_penalty.py
│   │   ├── test_conflict.py
│   │   ├── test_dual_controller.py
│   │   ├── test_anchor_memory.py
│   │   └── test_continual_metrics.py
│   ├── integration/
│   │   ├── test_one_rl_update.py
│   │   ├── test_task_boundary.py
│   │   ├── test_checkpoint_resume.py
│   │   └── test_two_task_tiny_model.py
│   ├── regression/
│   │   ├── test_dense_factorized_agreement.py
│   │   └── test_reference_metrics.py
│   └── smoke/
│       └── test_smoke_experiment.py
├── docs/
│   ├── method.md
│   ├── datasets.md
│   ├── experiments.md
│   ├── reproducibility.md
│   ├── results_template.md
│   └── decisions/
│       └── 0001-effective-weight-importance.md
├── notebooks/
│   └── exploratory_analysis.ipynb
├── data/
│   ├── README.md
│   ├── raw/                         # Ignored by Git
│   ├── processed/                   # Ignored by Git
│   └── manifests/                   # Versioned metadata only
└── outputs/                         # Ignored by Git except schema docs
    ├── checkpoints/
    ├── runs/
    ├── evaluations/
    ├── figures/
    └── tables/
```

### 9.1 Folder rules

- `src/rahc_lora` contains reusable production code.
- `scripts` contains thin command entry points only.
- `configs` contains all experiment decisions that may affect results.
- `tests` mirrors the source architecture.
- `notebooks` is exploratory only. Final metrics must be produced by versioned scripts.
- `data/raw`, `data/processed`, and `outputs` are not committed.
- Dataset manifests, checksums, split definitions, and licenses are committed.
- Do not place scientific logic inside shell scripts, notebooks, or VS Code task files.

---

## 10. Module contracts

### 10.1 Task interface

Every task implements one common protocol:

```python
class ContinualTask(Protocol):
    task_id: str

    def load_train(self) -> Dataset: ...
    def load_validation(self) -> Dataset: ...
    def load_test(self) -> Dataset: ...
    def format_prompt(self, example: dict) -> str: ...
    def score_response(self, example: dict, response: str) -> float: ...
    def aggregate_metrics(self, records: list[dict]) -> dict[str, float]: ...
```

Rewards must be deterministic for identical inputs unless explicitly documented.

### 10.2 Importance recorder interface

```python
class ImportanceRecorder(Protocol):
    def begin_step(self, policy: nn.Module) -> None: ...
    def capture_effective_gradients(self, losses: dict[str, Tensor]) -> None: ...
    def end_step(self, policy: nn.Module) -> None: ...
    def consolidate(self) -> "ConsolidationState": ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

### 10.3 Anchor memory interface

```python
class AnchorMemory(Protocol):
    @property
    def capacity(self) -> int: ...

    def add_candidates(self, records: list["AnchorRecord"]) -> None: ...
    def sample(self, batch_size: int, generator: torch.Generator) -> list["AnchorRecord"]: ...
    def rebalance(self) -> None: ...
    def state_dict(self) -> dict: ...
```

### 10.4 Method component toggles

The trainer must support the following without code changes:

```text
use_anchor_kl
use_parameter_consolidation
use_reward_aware_importance
use_factorized_importance
use_conflict_gate
use_dual_controller
```

This is required for valid ablations.

---

## 11. Configuration design

Use Hydra and structured dataclass validation. The resolved configuration is immutable after a run begins and is saved with every checkpoint.

Example method configuration:

```yaml
method:
  name: rahc_lora

  lora:
    rank: 8
    alpha: 16
    dropout: 0.0
    target_modules: [q_proj, k_proj, v_proj, o_proj]

  importance:
    enabled: true
    reward_aware: true
    representation: factorized_rank1
    ema_beta: 0.99
    epsilon: 1.0e-8
    clip_quantile: 0.995
    accumulator_dtype: float32

  anchor_memory:
    enabled: true
    capacity: 512
    batch_size: 16
    top_k_logits: 32
    selection: balanced_diverse_reservoir

  conflict:
    enabled: true
    gradient_space: effective_weight
    zero_norm_threshold: 1.0e-12
    coefficient_clip: 5.0

  functional_controller:
    enabled: true
    initial_lambda: 0.01
    dual_lr: 0.001
    target_kl: 0.02
    ema_beta: 0.95
    max_lambda: 10.0

  parameter_regularization:
    lambda: 1.0
```

Every run must also resolve and save:

- Model identifier and exact revision.
- Tokenizer identifier and revision.
- Dataset identifiers and revisions.
- Task order.
- Train, validation, and test split hashes.
- Random seeds.
- Generation parameters.
- Reward definitions.
- Optimizer and scheduler.
- Precision mode.
- Hardware information.
- Dependency versions.

---

## 12. Task stream and rewards

### 12.1 HLoRA reproduction datasets

Before adapting HLoRA to RL, reproduce a manageable supervised version of the source method using the same dataset families as the paper:

| Role | Dataset | Use |
|---|---|---|
| General-language reference | Fixed, versioned subset of The Pile | General knowledge and perplexity reference |
| Science domain | SciQ | Supervised domain adaptation and retention measurement |
| Physical commonsense domain | PiQA | Supervised domain adaptation and retention measurement |
| Medical domain | MedMCQA | Supervised domain adaptation and retention measurement |

The reproduction stage is separate from the main continual-RL benchmark. Record the exact Pile subset, source revisions, splits, preprocessing, seeds, optimizer, and evaluation protocol.

### 12.2 Final continual-RL datasets

The primary continual-RL task stream uses these exact datasets and roles:

| Position | Family | Dataset | Training reward |
|---:|---|---|---|
| 1 | Arithmetic reasoning | GSM8K | Normalized exact final answer |
| 2 | Science reasoning | ARC-Challenge | Exact option match |
| 3 | Code generation | MBPP | Fraction of hidden unit tests passed |
| 4 | Instruction constraints | Custom Verifiable Constraint Training dataset | Fraction of constraints satisfied |

Primary order:

```text
GSM8K -> ARC-Challenge -> MBPP -> Custom Verifiable Constraints
```

Use at least three task orders selected before final experiments:

```text
Order 1: GSM8K -> ARC-Challenge -> MBPP -> Constraints
Order 2: MBPP -> GSM8K -> Constraints -> ARC-Challenge
Order 3: Constraints -> ARC-Challenge -> GSM8K -> MBPP
```

### 12.3 Custom Verifiable Constraint Training dataset

Do not train on official IFEval prompts. IFEval is an evaluation benchmark and remains fully held out.

Create a separate versioned training dataset with new prompts and deterministic validators. It may cover general constraint categories such as:

- Required keyword count.
- Exact number of bullet points.
- Forbidden words.
- Required ending phrase.
- Word-count interval.
- Required titled sections.
- Valid JSON with required fields.

Requirements:

- Do not copy or paraphrase official IFEval prompts.
- Generate prompts from a versioned grammar or template library.
- Store the generator version, random seed, constraint combination, validator version, and prompt hash.
- Create disjoint training, validation, anchor-candidate, and held-out custom test splits.
- Split by prompt and template family so near-duplicate templates do not cross any split boundary.
- Run exact-overlap and normalized-text-overlap checks against official IFEval.
- Score each response as satisfied constraints divided by total constraints.

Use the held-out custom test split for the Task 4 performance-matrix score. Use official IFEval as an additional external instruction-following generalization evaluation.

### 12.4 Reward rules

- Rewards must lie in a documented range, preferably `[0, 1]`.
- Keep raw and normalized rewards in the logs.
- Do not silently change reward functions during a run.
- Parsing failures receive a documented score.
- Code rewards must run in an isolated process or container with CPU, memory, file, network, and time limits.
- Never execute generated code directly in the training process.

### 12.5 Data separation

For every task maintain disjoint:

- Training set.
- Hyperparameter-validation set.
- Anchor-candidate subset drawn only from training data.
- Final test set.

The final test set is evaluation-only.

Additional rules:

- Anchor memory may use only the anchor-candidate portion of training data.
- Official GSM8K, ARC-Challenge, and MBPP test examples never enter training or anchor memory.
- Official IFEval is never used for RL training, reward development, importance estimation, anchor selection, or hyperparameter tuning.
- Evaluation results must not influence prompt generation for the custom constraint dataset.
- Public benchmarks that may have appeared in backbone pretraining are evaluated by change from the frozen starting model as well as absolute score.

### 12.6 Dataset manifests

Every dataset manifest records:

- Canonical dataset name and exact revision.
- License and citation.
- Original split names.
- Project split construction and seed.
- Example IDs and content hashes.
- Preprocessing and answer-normalization version.
- Reward or validator version.
- Exact membership of training, validation, anchor-candidate, and test splits.

---

## 13. Evaluation

### 13.1 Performance matrix

Let `A[i, j]` be test performance on task `j` after finishing training task `i`.

Save the complete matrix for every seed and task order. Do not report only the final row.

### 13.2 Primary metrics

Average forgetting after the final task `T`:

```text
forgetting_j = max(A[i, j] for i in j ... T-1) - A[T, j]
average_forgetting = mean(forgetting_j for j in 1 ... T-1)
```

Final average performance:

```text
final_average_performance = mean(A[T, j] for j in 1 ... T)
```

### 13.3 Additional metrics

- Backward transfer.
- Forward transfer.
- Current-task learning curve and sample efficiency.
- General-capability retention.
- Output diversity.
- Response length.
- Anchor KL.
- Parameter drift by layer.
- Gradient conflict by layer.
- Importance concentration by layer.
- Wall-clock time.
- Tokens per second.
- Peak CPU and GPU memory.
- Checkpoint and consolidation-state size.
- Anchor-memory size.

### 13.4 General-capability suite

Use a fixed evaluation suite that never contributes examples to importance estimation or anchor selection:

- Official IFEval for held-out verifiable instruction following.
- A fixed MMLU subset for general knowledge and reasoning.
- HellaSwag for commonsense completion.
- ARC-Easy for easier science capability.
- A fixed held-out corpus for language-model perplexity.
- Optional safety or truthfulness checks only when the base model has meaningful baseline performance.

Perplexity alone is not sufficient evidence of general-capability retention.
The official IFEval prompts remain evaluation-only for the entire project.

### 13.5 Statistical analysis

- Pre-register average forgetting as primary.
- Treat final average performance as co-primary.
- Pair methods by seed and task order.
- Report mean, median, standard deviation, 95 percent bootstrap confidence interval, and effect size.
- Use a paired permutation test or Wilcoxon signed-rank test when the sample is small or visibly non-normal.
- Correct the false discovery rate across secondary ablations.
- Report negative and non-significant results.

---

## 14. Required baselines

Implement and validate these baselines before claiming improvement:

1. Standard RL-LoRA without retention.
2. RL-LoRA with KL to the initial policy on current-task prompts.
3. Balanced replay with the same memory budget as RAHC-LoRA.
4. EWC-LoRA or SI-LoRA adapted to the policy loss.
5. Direct HLoRA-to-RL with dense importance and static layer weights.
6. O-LoRA or another subspace-based continual LoRA method.
7. Joint multi-task RL as an offline upper reference, clearly marked as violating the sequential-data constraint.

All comparisons must match:

- Backbone.
- LoRA rank and target modules.
- Training tokens or rollout count.
- Optimizer steps.
- Reward functions.
- Task order and seeds.
- Evaluation generation parameters.
- Memory budget where applicable.

---

## 15. Required ablations

| Ablation | Question answered |
|---|---|
| Remove anchor KL | Is parameter consolidation sufficient under policy shift? |
| Remove parameter consolidation | Does behavioural retention explain the result? |
| Dense vs factorized importance | What is lost through compression? |
| Reward-aware vs ordinary importance | Does advantage information improve importance? |
| Remove conflict gate | Does dynamic conflict control improve plasticity? |
| Static vs adaptive KL coefficient | Is the dual controller necessary? |
| Memory 0, 128, 512, 2048 | How much historical behaviour is needed? |
| LoRA rank 4, 8, 16 | Does adapter capacity change forgetting? |
| Effective-weight vs factor-level penalty | Does rescaling invariance matter in practice? |

Each component must be disabled through configuration, not by editing source code.

---

## 16. Checkpoint contract

Every resumable checkpoint contains:

```text
LoRA adapter state
optional value-head state
optimizer state
scheduler state
gradient-scaler state, if used
current task index
global optimizer step
rollout counter
random-number-generator states
reference LoRA factors
factorized importance state
dual-controller state
anchor-memory state and manifest
resolved configuration
dataset and split fingerprints
model and tokenizer revisions
Git commit hash and dirty-worktree flag
dependency versions
```

A checkpoint missing consolidation or anchor state is not resumable for this project.

The integration suite must verify that an interrupted two-task run resumes within documented numeric tolerance.

---

## 17. Logging and artifact contract

Each run writes to:

```text
outputs/runs/<experiment>/<method>/<task_order>/<seed>/<run_id>/
```

Required files:

```text
resolved_config.yaml
run_manifest.json
metrics.jsonl
system_metrics.jsonl
performance_matrix.csv
general_capabilities.csv
layer_importance.parquet
layer_conflicts.parquet
memory_manifest.json
checkpoint_manifest.json
stdout.log
summary.json
```

Weights & Biases may be used as an optional mirror, but local machine-readable artifacts remain authoritative.

Never log:

- API keys.
- Authentication tokens.
- Private dataset contents.
- Full prompts when privacy rules prohibit storage.
- Environment-variable values unrelated to reproducibility.

---

## 18. Development setup in VS Code

### 18.1 Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/doctor.py
pytest -q
```

### 18.2 Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/doctor.py
pytest -q
```

### 18.3 Recommended VS Code extensions

- Python.
- Pylance.
- Ruff.
- Jupyter.
- YAML.
- GitLens, optional.
- Remote SSH, optional for GPU servers.

### 18.4 VS Code workspace rules

- Select `.venv` as the workspace interpreter.
- Enable format on save with Ruff.
- Enable pytest discovery from `tests`.
- Use `src` layout. Do not add the repository root to `PYTHONPATH` manually.
- Store debug arguments in `.vscode/launch.json`.
- Store repeatable commands in `.vscode/tasks.json`.
- Do not store secrets in VS Code settings or launch configurations.

---

## 19. Core commands

The implemented CLI should support:

```bash
# Validate hardware, dependencies, dataset access, and configuration
python -m rahc_lora.cli doctor

# Prepare pinned dataset splits and manifests
python -m rahc_lora.cli prepare-data experiment=pilot

# Standard RL-LoRA baseline
python -m rahc_lora.cli train experiment=pilot method=rl_lora seed=1

# Direct HLoRA-to-RL baseline
python -m rahc_lora.cli train experiment=pilot method=hlora_rl seed=1

# Proposed method
python -m rahc_lora.cli train experiment=pilot method=rahc_lora seed=1

# Resume an interrupted run
python -m rahc_lora.cli train experiment=pilot method=rahc_lora resume=/absolute/checkpoint/path

# Evaluate one checkpoint on all learned tasks
python -m rahc_lora.cli evaluate checkpoint=/absolute/checkpoint/path

# Run the configured experiment matrix
python -m rahc_lora.cli sweep experiment=main

# Aggregate tables, confidence intervals, and figures
python -m rahc_lora.cli aggregate experiment=main
```

Exact syntax may change during scaffolding, but there must be one documented CLI and no duplicated experiment logic across scripts.

---

## 20. Dependency policy

Expected core libraries:

- PyTorch.
- Transformers.
- PEFT.
- TRL or a thin internally controlled GRPO implementation.
- Accelerate.
- Datasets.
- Hydra and OmegaConf.
- NumPy, pandas, scipy, and scikit-learn.
- safetensors.
- pytest.
- Ruff.
- mypy.

Rules:

- Pin exact versions in the lock file used for experiments.
- Record CUDA, driver, and GPU details in every run manifest.
- Do not rely on undocumented behaviour from trainer libraries.
- Wrap external trainer APIs behind project-owned interfaces.
- Add a regression test before upgrading a core dependency.
- Never silently fall back from BF16 to another precision mode.

---

## 21. Coding standards

### 21.1 General rules

- Use type hints for public functions and data models.
- Prefer dataclasses or typed configuration models over unstructured dictionaries.
- Keep files focused. Split a module before it exceeds roughly 500 lines unless cohesion justifies it.
- No scientific constants hidden inside functions.
- No mutable global training state.
- No silent exception handling.
- Raise actionable errors containing the component, task, step, and invalid value.
- Keep training and evaluation paths separate.
- Use structured logs rather than scattered `print` statements.
- Public scientific functions require docstrings with tensor shapes and units.

### 21.2 Tensor conventions

Document dimensions using:

```text
B = batch size
T = token length
H = hidden dimension
O = linear-layer output dimension
I = linear-layer input dimension
R = LoRA rank
```

Name tensors with semantic suffixes when ambiguity exists, such as:

```text
advantages_bt
logprobs_btv
importance_oi
lora_a_ri
lora_b_or
```

### 21.3 Numeric safety

- Importance and metric accumulators use FP32 or FP64 as configured.
- Check finite values after rewards, advantages, losses, gradients, importance updates, and controller updates.
- Use epsilon-safe division and cosine similarity.
- Clip only where specified in configuration.
- Log every clipping rate.
- Add deterministic synthetic tests for extreme values and zero gradients.

### 21.4 Research integrity

- Never tune on final test results.
- Never remove failed seeds without a documented technical reason.
- Do not change task order after viewing headline results.
- Preserve raw run artifacts.
- Mark post-hoc analysis as post-hoc.
- Report compute and memory overhead for every method.
- Distinguish statistical significance from practical significance.

---

## 22. Test strategy

### 22.1 Unit tests

Required before GPU training:

- Effective LoRA weight matches direct `scale * B @ A`.
- Rescaling `A` and `B` does not change effective-weight penalty.
- Efficient low-rank penalty matches dense penalty.
- Importance update clips negative contributions correctly.
- Factorization is nonnegative and preserves expected matrix mass.
- Conflict is zero for aligned gradients and positive for opposing gradients.
- Dual multiplier increases above the KL budget and decreases below it.
- Anchor memory never exceeds capacity.
- Anchor selection remains task balanced within configured tolerance.
- Continual metrics match hand-calculated examples.

### 22.2 Integration tests

- One complete RL optimizer update on a tiny model.
- One update with an anchor batch.
- One task-boundary consolidation.
- Two-task continual run on a tiny deterministic dataset.
- Checkpoint save and resume.
- Single-process and distributed metric agreement.

### 22.3 Regression tests

- Fixed synthetic inputs produce stable losses and metrics.
- Dense and factorized implementations stay within documented tolerance.
- Dependency upgrades do not silently change reward parsing or generation.

### 22.4 Smoke experiment

The smoke configuration must:

- Finish in less than 15 minutes on the development GPU.
- Use a tiny model or mock policy.
- Execute two tasks.
- Exercise anchor insertion, importance consolidation, conflict calculation, checkpointing, and evaluation.
- Produce the complete artifact contract.

---

## 23. Implementation phases

### Phase 0: Repository scaffold

Deliver:

- `src` package.
- Typed configuration.
- CLI skeleton.
- Logging and manifests.
- VS Code configuration.
- CI-ready linting and tests.

Exit condition:

- `doctor`, lint, type check, and unit test commands run successfully.

### Phase 1: Tasks, rewards, and evaluation

Deliver:

- Task protocol.
- GSM8K, ARC-Challenge, MBPP, and custom-constraint task adapters.
- Versioned custom-constraint prompt generator and deterministic validators.
- Deterministic reward adapters.
- Performance matrix and continual metrics.

Exit condition:

- Hand-checked examples receive correct rewards and metrics.

### Phase 2: Standard RL-LoRA baseline

Deliver:

- Frozen backbone and shared LoRA policy.
- GRPO training loop.
- Checkpoint and resume.
- Full evaluation after each task.

Exit condition:

- A two-task baseline run is reproducible across repeated launches with the same seed.

### Phase 3: Direct HLoRA-to-RL baseline

Deliver:

- Supervised HLoRA reproduction on SciQ, PiQA, MedMCQA, and the fixed Pile subset.
- Dense importance reference implementation.
- Static layer coefficients.
- Parameter regularization in RL training.

Exit condition:

- The supervised reproduction is documented, and the dense RL implementation passes synthetic correctness tests and completes the pilot stream.

### Phase 4: Effective-weight importance

Deliver:

- Effective gradient capture.

- Actual optimizer-step tracking.
- Reward-aware importance update.
- Rescaling-invariance tests.

Exit condition:

- Importance is numerically stable and correlates with measured parameter sensitivity in a small perturbation test.

### Phase 5: Factorized consolidation

Deliver:

- Rank-one nonnegative factorization.
- Efficient low-rank penalty.
- Dense comparison tools.

Exit condition:

- Efficient penalty matches dense reference within tolerance and lowers persistent storage as expected.

### Phase 6: Anchor memory and behavioural KL

Deliver:

- Fixed-size balanced memory.
- Compact reference distributions.
- Anchor KL or distillation loss.

Exit condition:

- Memory remains bounded and old-policy drift is measurable after a controlled destructive update.

### Phase 7: Conflict gate and dual controller

Deliver:

- Separate current and anchor gradient capture.
- Layer conflict coefficients.
- Adaptive KL multiplier.

Exit condition:

- Synthetic aligned and conflicting gradients trigger the expected controller behaviour.

### Phase 8: Pilot

Deliver:

- Standard RL-LoRA.
- Replay baseline.
- Direct HLoRA-to-RL.
- RAHC-LoRA.
- At least two task orders and three seeds.

Exit condition:

- Apply the continuation rule in Section 4.3 before launching the full matrix.

### Phase 9: Main experiment and ablations

Deliver:

- Three task orders.
- Three seeds.
- Required baselines.
- Required ablations.
- Statistical analysis.
- Efficiency comparison.

Exit condition:

- Every headline number is traceable to a run manifest and raw performance matrix.

---

## 24. Risks and controls

| Risk | Consequence | Control |
|---|---|---|
| Weak base performance | Little measurable forgetting | Pilot candidate models and choose tasks with headroom |
| Reward sparsity | Importance signal collapses | Partial credit, curriculum sampling, grouped rewards, robust normalization |
| Over-consolidation | Old tasks retained but new task not learned | Co-primary plasticity metric, conflict gate, adaptive KL budget |
| Anchor overfitting | Anchor KL looks good while task return falls | Diverse selection, held-out evaluation, memory-size ablation |
| Task-order artifact | One favourable order drives the result | Three preselected orders and paired reporting |
| Baseline mismatch | Proposed method receives more resources | Match steps, rollouts, rank, memory, generation, and hardware accounting |
| Dense transient memory | Out-of-memory during importance estimation | Chunked effective-weight calculations and low-rank penalty |
| Noisy conflict signal | Unstable layer coefficients | EMA, norm threshold, clipping, and static-coefficient ablation |
| Dual-controller instability | Lambda saturates or oscillates | Smoothed KL, bounded multiplier, smaller dual learning rate |
| Code reward escape | Host compromise or corrupted run | Isolated execution, no network, strict resource limits |
| Missing reproducible HLoRA code | Baseline ambiguity | Implement from equations and document every interpretation |

---

## 25. Decision rules

Apply these rules instead of changing the method based on one favourable run:

1. If anchor KL alone matches the full method across seeds and orders, simplify the contribution and report parameter consolidation as unnecessary in the tested setting.
2. If conflict gating is unstable or provides no consistent benefit, use factorized importance plus anchor KL as the main method.
3. If factorized importance loses substantial retention relative to dense importance, test a small factorization rank before abandoning compression.
4. If language-model RL cannot run reproducibly within available compute, validate the same scientific mechanism in a standard continual-control benchmark, then return to the LLM setting as an extension.
5. If no catastrophic forgetting occurs, redesign the stream. Do not claim success from the absence of a measurable problem.
6. If the method reduces forgetting but current-task performance falls materially, report over-consolidation rather than success.

---

## 26. Instructions for coding agents and contributors

Before changing code:

1. Read this file and the relevant configuration and tests.
2. Identify the scientific claim affected by the change.
3. Add or update a test before modifying a core equation.
4. Keep changes scoped to one component when possible.

For every feature:

- Add typed configuration.
- Add validation.
- Add unit tests.
- Add at least one integration path if training behaviour changes.
- Add logging for the new state.
- Ensure checkpoint save and load include the new state.
- Document the ablation switch.

Do not:

- Hide fallbacks.
- Change metric definitions without an architecture decision record.
- Add task-specific logic to the generic trainer.
- Duplicate training loops for baselines.
- Hardcode local paths, GPU IDs, dataset cache paths, or credentials.
- Commit model weights, raw datasets, secrets, or large run artifacts.
- refactor unrelated modules during an experiment-critical fix.

When assumptions are unclear, preserve the simplest testable interpretation and document it in `docs/decisions`.

---

## 27. Definition of done

The project is thesis-ready only when all items below are true:

- The baseline demonstrates measurable forgetting.
- RAHC-LoRA and every required baseline run through the same trainer and evaluator.
- Every component can be disabled through configuration.
- Dense and efficient penalty implementations agree in tests.
- Effective-weight rescaling invariance is verified.
- Anchor memory remains within the configured byte and record budgets.
- Checkpoints resume with consolidation, controller, and memory state intact.
- Results include at least three task orders and three seeds for the main comparison.
- Full task-by-task performance matrices are retained.
- General capabilities and current-task plasticity are reported.
- Runtime, GPU memory, and persistent storage are reported.
- Confidence intervals and paired statistical tests are included.
- Final figures and tables are generated from versioned scripts.
- The complete experiment can be traced through configuration, data manifests, Git revision, and run artifacts.
- Limitations and negative results are documented.

---

## 28. Initial implementation priority

Build in this order:

1. Evaluation matrix and deterministic task rewards.
2. Reproducible standard RL-LoRA baseline.
3. Direct HLoRA-to-RL baseline.
4. Dense effective-weight importance reference.
5. Factorized efficient penalty.
6. Fixed anchor memory and behavioural KL.
7. Conflict-aware layer controller.
8. Adaptive KL controller.
9. Full baselines and ablations.

Do not begin with the full RAHC-LoRA system. Each stage must have a working baseline, tests, and a measurable expected effect before the next mechanism is added.

---

## 29. Expected final thesis claim

Use a bounded claim supported by the actual experiment:

> Under the evaluated model scale, task sequences, and fixed memory budget, RAHC-LoRA reduced measured continual-RL forgetting while preserving new-task learning, with lower persistent importance storage than dense parameter-consolidation methods.

Do not claim that the method eliminates catastrophic forgetting in all language models or RL settings.
