# Standard RL-LoRA training and resume

Phase 2 supplies one project-owned sequential trainer for the no-retention baseline. It loads a
pinned causal language model, freezes the backbone, attaches one shared LoRA adapter, performs
grouped GRPO updates, evaluates all learned tasks after each boundary, and writes atomic resumable
checkpoints. The adapter is never reset between tasks. When enabled, the fixed general-capability
suite is evaluated once on the frozen starting policy and after every learned-task boundary; every
reported score is paired with its change from that immutable baseline.

The exact GRPO semantics are recorded in [ADR 0002](decisions/0002-phase2-grpo-reference.md).

## Reward diagnostics

Each `grpo_update` event records aggregate reward diagnostics alongside raw and normalized reward
statistics. `reward_diagnostics.parse_failure_count` counts responses the task scorer could not
parse. For isolated code rewards, `code_execution_count`, `code_tests_passed`, and
`code_tests_total` distinguish parse failures from code that ran but failed tests;
`code_timeout_count`, `code_output_limited_count`, and `code_nonzero_exit_count` report sandbox
outcomes. These counters do not store generated response text.

## Model profiles

| Profile | Purpose | Immutable revision |
| --- | --- | --- |
| `model=tiny` | CPU mechanics and deterministic tests only | internal model/tokenizer `1.0.0` |
| `model=small` | minimum pilot model, Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `model=confirmation` | 1.5B confirmation profile | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

External model and tokenizer revisions must be 40-character commit hashes. Remote model code is
disabled. Unsupported requested precision fails rather than silently falling back.

## Commands

Run the CPU mechanics smoke with explicit tiny-policy settings:

```powershell
python -m rahc_lora.cli train --run-id phase2-smoke `
  model=tiny method=rl_lora_tiny launcher.device=cpu `
  rollout.prompts_per_batch=1 rollout.group_size=4 `
  rollout.max_prompt_tokens=32 rollout.max_response_tokens=2 `
  general_capabilities.max_new_tokens=2 `
  experiment.evaluation_max_examples_per_task=2
```

Stop safely after the next task boundary:

```powershell
python -m rahc_lora.cli train --run-id interrupted --max-task-boundaries 1 <same-overrides>
```

Resume into a new run directory using exactly the same scientific configuration:

```powershell
python -m rahc_lora.cli train --run-id resumed `
  --resume outputs/runs/.../checkpoints/task-01-step-00000001 `
  <same-overrides>
```

Resume rejects changes to the resolved scientific configuration, model/tokenizer identity, dataset
fingerprints, or dependency versions. Operational run ID and resume path are not scientific config.

General-capability evaluation is disabled by default, and the run records that fact explicitly. To
enable it, set `general_capabilities.enabled=true` and point
`general_capabilities.ifeval_scorer_repository` at a local Google Research Git worktree checked out
at the configured commit. The built-in adapter verifies that revision and its four runtime source
files, then calls the official strict and loose functions. It also verifies the configured SHA-256
digest of the English NLTK `punkt_tab` resource before use. The runtime dependencies are in the
lock file; install `punkt_tab` in the configured `ifeval_nltk_data_dir` before the run. The
checkout and resource checks can be inspected with `rahc-lora doctor` using the same overrides.
IFEval, fixed-subject MMLU, HellaSwag, ARC-Easy, and WikiText perplexity remain evaluation-only.

For the default local cache path, fetch the NLTK resource during environment setup:

```powershell
python -m nltk.downloader -d .cache/rahc_lora/nltk_data punkt_tab
```

The configured tree digest is
`56e42ce8e87ce5653d8bc261e8100a2cc719fa51e3e66ef138dc54049733b4b6`. If the
downloaded data differs, doctor stops the run; supply the pinned resource and record its origin.
The scorer checkout must be at Google Research commit
`758b894eb02dc2a7097068031089a2803be147c6` with the four tracked runtime files present.
On Windows, the upstream monorepo contains filenames that prevent a full checkout; a sparse or
no-checkout clone with only `instruction_following_eval` runtime files is sufficient.

The initial 0.5B capable-model diagnostic command was:

```powershell
python -m rahc_lora.cli train --run-id pilot-two-task `
  experiment=pilot_two_task model=small method=rl_lora `
  launcher=single_gpu general_capabilities.enabled=true `
  general_capabilities.ifeval_scorer_repository=/path/to/google-research `
  general_capabilities.max_examples_per_capability=128 `
  general_capabilities.max_new_tokens=64
```

The initial plan was a single-seed, two-task pilot with 100 optimizer steps per task, two prompts per
rollout, four responses per prompt, at most 32 training-response tokens, and 128 held-out examples
per learned task at each boundary. Each capability uses at most 128 pinned examples per boundary
with at most 64 generated response tokens. The pinned 0.5B model runs in explicit BF16 and fails if
the selected GPU lacks BF16 support. It excludes MBPP, so Docker is not needed for this diagnostic.
This remains a long GPU job and must not be launched without user authorization.

The 0.5B execution used `rollout.max_response_tokens=128` instead of the predeclared 32-token budget.
Training-rollout probes had zero GSM8K reward at 32 and 64 tokens; 128 tokens produced the first
positive-reward group, so that budget was fixed before the full pilot. Evaluation used
`general_capabilities.max_new_tokens=64`. The 0.5B result and weak GSM8K learning are documented in
[implementation status](implementation-status.md).

### Confirmation pilot and duration extension

The 1.5B confirmation run `phase2-pilot-confirmation-gpu-128-20260924` used the same 100 updates per
task, 128-token training responses, task order, seed, and held-out example caps. Its GSM8K score was
5/128 after GSM8K and 6/128 after ARC-Challenge; ARC-Challenge reached 89/128. This is no measurable
forgetting, but the GSM8K task performance is too close to floor to estimate retention usefully.

The decision to extend training used training rollouts only: GSM8K had positive raw reward in 28/100
updates (mean raw batch reward 0.045), while ARC-Challenge had positive reward in 96/100 updates
(mean 0.49). Held-out scores remained evaluation-only and did not select training settings. The
duration diagnostic kept the model, response budget, seed, task order, and evaluation settings
fixed, and increased `experiment.max_steps_per_task` from 100 to 500 for both tasks.

```powershell
python -m rahc_lora.cli train --run-id phase2-pilot-duration-500-gpu-128-20260924 `
  experiment=pilot_two_task model=confirmation method=rl_lora task_stream=order_1 seed=1 `
  launcher=single_gpu general_capabilities.enabled=true `
  general_capabilities.ifeval_scorer_repository=.cache/upstream/google-research-ifeval-sparse `
  general_capabilities.ifeval_nltk_data_dir=.cache/rahc_lora/nltk_data `
  general_capabilities.max_examples_per_capability=128 `
  general_capabilities.max_new_tokens=64 experiment.max_steps_per_task=500 `
  experiment.task_limit=2 experiment.evaluation_max_examples_per_task=128 `
  rollout.max_response_tokens=128
```

Phase 3 remains gated on adequate earlier-task learning and a useful forgetting signal. If the
longer standard baseline does not reach the pilot criterion, redesign the task stream before
evaluating RAHC-LoRA. The confirmation and duration-pilot results and artifact validation are
recorded in [implementation status](implementation-status.md).

The 500-update duration run has now completed with exit code 0. Its matrix was:

```text
after GSM8K:         GSM8K 0.1875
after ARC-Challenge: GSM8K 0.171875, ARC-Challenge 0.7578125
```

This is 1.56 percentage points of average forgetting (2 fewer GSM8K answers out of 128), below the
10-point continuation criterion in `Project.md`; final average task performance is 0.46484375.
Training metrics had positive mean reward in 373/500 GSM8K batches and 478/500 ARC-Challenge batches,
but zero-gradient updates still occurred 160 and 345 times, respectively. The run is complete, but
the baseline still does not support a useful retention comparison. Choose a stronger task conflict
before the next pilot; do not start Phase 3 from this result. Both checkpoints passed file size/hash
validation, and the latest pointer references task 2. Full metrics and diagnostic capability changes
are recorded in [implementation status](implementation-status.md).

## Artifact contract

Runs are written below:

```text
outputs/runs/<experiment>/<method>/<task_order>/<seed>/<run_id>/
```

Each run contains the resolved configuration, schema-validated run manifest, dataset fingerprints,
learning and system JSONL metrics, hashed evaluation records, performance matrix, capability CSV
and capability-status record, baseline-empty layer diagnostics and memory manifest, checkpoint
index, stdout log, and final summary. Capability rows include strict/loose IFEval, MMLU, HellaSwag,
ARC-Easy, and perplexity scores, frozen baselines, signed changes, metric direction, and example
counts. Run manifests include model/tokenizer revisions, command overrides, task order, seed,
dependencies, Git state when available, and CPU/CUDA hardware details.

Each checkpoint is a new atomic directory. Its manifest records the size and SHA-256 hash of every
file. The checkpoint includes:

- LoRA adapter and explicit absent value head;
- optimizer, constant scheduler, and explicit absent gradient scaler;
- next task, global optimizer step, rollout count, policy version, general RNG state, and rollout
  generator state;
- complete performance matrix;
- explicit capability enablement and, when enabled, the frozen baseline plus complete boundary
  history;
- resolved configuration, run manifest, model/tokenizer identity, and dataset fingerprints;
- explicit empty reference, importance, controller, and anchor states for standard RL-LoRA.

No valid checkpoint is overwritten in place. The latest pointer changes only after validation and
atomic directory replacement succeed. Capability-aware checkpoints use schema version 2; earlier
Phase 2 development checkpoints are rejected rather than migrated silently.

## Reproducible environment

`requirements-lock-cpu-win-py312.txt` captures the exact Windows/Python 3.12 CPU environment used
for local validation. `environment-gpu-win-py311.yml` captures the current Conda Python 3.11 GPU
environment with Conda build pins, pip version pins, CUDA PyTorch wheel index, and driver note. The
project itself is installed separately in editable mode. Fresh environment recreation from this
export has not yet been tested.

## Current smoke interpretation

The source-backed two-task tiny smoke completed twice and resumed from its first-task checkpoint
identically. Both GSM8K and ARC-Challenge scores and rewards were zero; every prompt group had zero
reward variance and therefore produced a zero gradient, as specified. This is useful negative
evidence: the smoke proves wiring, artifacts, and determinism, but it does not prove the baseline can
learn or forget. A GPU pilot with a capable pinned instruction model is still required for the Phase
2 scientific exit gate.
