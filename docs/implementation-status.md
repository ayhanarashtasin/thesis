# Implementation status

## Current phase

Phase 2 - reproducible standard RL-LoRA baseline - is complete. The deterministic CPU/tiny-model
smokes, checkpoint/resume gates, and capable-model pilots establish an operational baseline. The
approved 1.5B MBPP -> GSM8K alternate-order pilot completed 1,000 optimizer updates and passed
held-out, capability, and checkpoint-integrity validation after recovery from its verified step-500
checkpoint. It measured no forgetting: MBPP rose from 0.049479 to 0.0625 after GSM8K. This and the
other tested streams remain below the 10-percentage-point continuation threshold, so Phase 8 must
select a stream with enough baseline headroom before a method pilot. No experimental success claim
is supported. Phase 3 is the earliest incomplete phase.

## Remaining phase map

| Phase | Status | Required exit evidence |
| --- | --- | --- |
| 2: Standard RL-LoRA baseline | Complete; reproducible smoke/resume gates and capable-model pilots passed; order_2 recovery artifacts verified | Baseline evidence complete; low-forgetting continuation rule must be revisited before Phase 8 |
| 3: Direct HLoRA-to-RL | Not started; next phase | Document supervised SciQ/PiQA/MedMCQA/fixed-Pile reproduction; dense importance and static-importance RL baseline passes the matched two-task smoke |
| 4: Reward-aware effective-weight importance | Not started; follows Phase 3 | Rescaling invariance, finite-value, checkpoint, and parameter-sensitivity tests pass |
| 5: Factorized consolidation | Not started; follows Phase 4 | Dense and low-rank penalties agree; storage is bounded and no dense persistent importance is used |
| 6: Anchor memory and behavioural KL | Not started; follows Phase 5 | Fixed-budget memory tests pass; controlled destructive update shows reduced old-policy drift |
| 7: Conflict gate and dual controller | Not started; follows Phase 6 | Synthetic gradient cases and two-task smoke show stable, resumable controller behavior |
| 8: Pilot | Not started; gated by Phases 3-7 | Standard, replay, HLoRA, and RAHC methods; at least two orders and three seeds when affordable; factual report applies the continuation rule |
| 9: Main experiment and ablations | Not started; gated by Phase 8 decision | Three orders and three seeds; required baselines/ablations, paired statistics, efficiency results, and traceable raw artifacts |

## Completed phases

### Phase 0

- Installable `src` package, immutable typed Hydra configuration, structured logging, manifests,
  environment doctor, test/lint/type tooling, and VS Code integration.

### Phase 1

- Exact four-task continual registry and evaluation-only general-capability suite.
- Pinned source revisions, deterministic isolated splits, exact manifests, and custom/IFEval
  overlap audit.
- Deterministic rewards, fail-closed MBPP Docker boundary, performance matrix, and continual
  metrics.
- Mocked two-task evaluation exit gate.

See [dataset protocol](datasets.md) and [ADR 0001](decisions/0001-task-split-construction.md).

## Phase 2 implemented

- Immutable pinned model/tokenizer profiles and a lazy Transformers/PEFT factory.
- Frozen causal backbone with exactly one shared configurable LoRA adapter.
- CPU-only internal byte tokenizer and tiny causal model using the same policy boundary.
- Project-owned grouped rollout records, FP32 population-variance advantage normalization,
  zero-variance handling, advantage clipping, and token-level clipped GRPO objective.
- Configuration validation that forbids all retention components in standard RL-LoRA.
- Sequential training without resetting the adapter and deterministic task-example cycling.
- Task-specific Phase 1 rewards connected directly to rollouts.
- Evaluation of every learned task after each boundary with explicit future missing cells.
- Configuration-controlled general-capability evaluation with an immutable frozen-policy baseline,
  a pinned official strict/loose IFEval adapter, fixed MMLU subjects, HellaSwag, ARC-Easy, and
  held-out WikiText perplexity. Absolute scores and signed baseline changes are recorded before
  training and after every task boundary.
- Structured reward, advantage, loss, sampled-KL, clipping, token, timing, throughput, CPU-memory,
  GPU-memory, diversity, response-length, and aggregate reward-diagnostic metrics. The latter
  separate parse failures from executed-code test, timeout, output-limit, and exit-code outcomes.
- Complete local run-artifact layout and exact CPU validation environment lock.
- New atomic checkpoint directories with file sizes/hashes, exact compatibility checks, complete RNG
  restoration, explicit baseline-empty retention state, complete capability history, and latest
  pointer update after validation.
- Unified `train` CLI with operational boundary stop and resume support.

See [training protocol](training.md) and
[ADR 0002](decisions/0002-phase2-grpo-reference.md).

## Scientific behavior

The standard baseline has no replay, anchor KL, current-prompt base KL, consolidation penalty,
importance state, conflict gate, or dual controller. Only LoRA tensors enter the optimizer. Sparse
reward groups with no within-group variation receive zero advantages and no update; the event is
logged rather than hidden behind a fallback.

The source-backed CPU smoke used GSM8K then ARC-Challenge with one update per task, four responses
per prompt, two generated response tokens, and two held-out examples per boundary. It completed in
under 10 seconds per run and produced:

```text
after GSM8K:        [0.0, missing]
after ARC-Challenge:[0.0, 0.0]
```

Both rollout groups had all-zero deterministic rewards, zero advantages, and zero gradients. The
adapter therefore remained unchanged. This is not retention success: the intentionally tiny model
and budget did not learn either task.

The synthetic-reward integration path separately proves a real optimizer update: LoRA parameters
change while every frozen parameter remains bit-identical.

A tiny CUDA smoke using Conda `gpu-env` completed the same two-task, one-update-per-task path on the
RTX 4070 Ti SUPER. Its matrix was also `[0.0, missing]`, then `[0.0, 0.0]`; it verifies device and
trainer wiring, not learning or retention.

### Capable-model pilot: Qwen2.5-0.5B-Instruct

Run `phase2-pilot-gpu-128-20260924` completed on 2026-09-24 with standard RL-LoRA, seed 1, task order
GSM8K then ARC-Challenge, 100 updates per task, 128 training-response tokens, and 128 held-out
examples per task and capability. Evaluation generation remained capped at 64 tokens. The response
budget was raised from the predeclared 32-token setting after training-rollout probes produced no
GSM8K reward signal at 32 or 64 tokens and the first positive-reward group at 128 tokens.

The final matrix was:

```text
after GSM8K:         GSM8K 0.015625
after ARC-Challenge: GSM8K 0.0078125, ARC-Challenge 0.4609375
```

This gives 0.0078125 average forgetting (0.78 percentage points) and 0.234375 final average task
performance for this single order and seed. GSM8K accuracy was only 2/128 after its own training and
1/128 after ARC-Challenge, so the old-task loss is not a useful catastrophic-forgetting estimate.
The pilot does not meet the project's 10-percentage-point continuation threshold and makes no method
success claim. General-capability changes from the frozen policy after ARC-Challenge were IFEval
strict/loose -0.015625 each, MMLU +0.1171875, HellaSwag +0.015625, ARC-Easy +0.2421875, and
perplexity +0.1746553 (higher is worse). These are single-run diagnostic measurements.

Both task-boundary checkpoints and the final matrix were written. All files in the final checkpoint
manifest and both checkpoint-manifest references passed SHA-256 verification. The summary records
`experimental_success_claimed: false`. Artifacts are under
`outputs/runs/pilot_two_task/rl_lora/order_1/1/phase2-pilot-gpu-128-20260924/`.

### Confirmation pilot: Qwen2.5-1.5B-Instruct

Run `phase2-pilot-confirmation-gpu-128-20260924` completed on 2026-09-24 with the pinned
Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, standard RL-LoRA,
seed 1, GSM8K then ARC-Challenge, 100 updates per task, 128 training-response tokens, and 128
held-out examples per task and capability. Evaluation generations were capped at 64 tokens.

The final matrix was:

```text
after GSM8K:         GSM8K 0.0390625
after ARC-Challenge: GSM8K 0.046875, ARC-Challenge 0.6953125
```

GSM8K improved from 5/128 to 6/128 after ARC-Challenge, so the computed average forgetting is
-0.0078125 and backward transfer is +0.0078125. This is no measurable forgetting, but the earlier
task is too close to floor to support a useful retention estimate. Final average task performance is
0.37109375. The result does not satisfy the pilot continuation rule for launching the full method;
`experimental_success_claimed` remains false.

Training metrics show that GSM8K had positive raw reward in 28/100 update batches, a mean raw batch
reward of 0.045, and 72 zero-gradient updates. ARC-Challenge had positive raw reward in 96/100
batches, a mean of 0.49, and 11 zero-gradient updates. The duration extension is based on this
sparse training signal. Held-out scores remain evaluation-only.

Changes from the frozen capability baseline after ARC-Challenge were IFEval strict/loose +0.0625
each, MMLU +0.234375, HellaSwag +0.0625, ARC-Easy +0.3203125, and perplexity +0.1089956 (higher is
worse). These are single-run diagnostic measurements.

Both task-boundary checkpoints, the final matrix, evaluation records, and capability metrics were
written. Every file listed in both checkpoint manifests passed size and SHA-256 validation, both
manifest references matched their hashes, and the latest-checkpoint pointer referenced task 2.
Artifacts are under
`outputs/runs/pilot_two_task/rl_lora/order_1/1/phase2-pilot-confirmation-gpu-128-20260924/`.

### Duration pilot: Qwen2.5-1.5B-Instruct, 500 updates per task

Run `phase2-pilot-duration-500-gpu-128-20260924` completed with exit code 0 on 2026-09-24/25. It
used the same pinned 1.5B model, standard RL-LoRA, seed, task order, 128-token training responses,
128 held-out examples, and 64-token evaluation generation cap as the confirmation pilot, increasing
training from 100 to 500 updates per task.

The final performance matrix was:

```text
after GSM8K:         GSM8K 0.1875
after ARC-Challenge: GSM8K 0.171875, ARC-Challenge 0.7578125
```

GSM8K scored 24/128 after its own training and 22/128 after ARC-Challenge. Average forgetting is
0.015625 (1.56 percentage points), backward transfer is -0.015625, and final average task
performance is 0.46484375. ARC-Challenge reached 97/128. The two-answer GSM8K drop is below the
project's 10-percentage-point continuation threshold and is too small to support a useful retention
comparison. `experimental_success_claimed` remains false.

Training-only rollout metrics show more reward signal after extending duration: GSM8K had positive
mean reward in 373/500 update batches (mean batch reward 0.30875), 160 zero-gradient updates, and at
least one zero-variance response group in 390 updates. ARC-Challenge had positive mean reward in
478/500 batches (mean 0.71525), 345 zero-gradient updates, and at least one zero-variance group in
473 updates. Positive batch reward therefore often still yielded no gradient update.

After ARC-Challenge, general-capability changes from the frozen baseline were IFEval strict +0.0546875,
IFEval loose +0.03125, MMLU +0.421875, HellaSwag +0.5546875, ARC-Easy +0.7578125, and perplexity
+0.6149424 (higher is worse). These are single-run diagnostics, not success evidence.

The run recorded all 384 held-out evaluation examples and both task-boundary checkpoints. All nine
files in each checkpoint passed size and SHA-256 checks; both manifest references matched, the
latest pointer referenced task 2, and the final matrix and summary agree. Artifacts are under
`outputs/runs/pilot_two_task/rl_lora/order_1/1/phase2-pilot-duration-500-gpu-128-20260924/`.

### Alternate-order pilot and verified recovery: Qwen2.5-1.5B-Instruct

The source run `phase2-pilot-order2-duration-500-gpu-128-20260925` used standard RL-LoRA, seed 1,
the predeclared MBPP -> GSM8K order, 500 updates per task, 128-token training responses, 128 held-out
examples per task, and the six-metric general-capability suite. It logged 918/1,000 updates before
stopping during GSM8K; its only resumable checkpoint is the verified task-1 boundary at global step
500. The source artifacts are preserved. Recovery run `phase2-pilot-order2-recovery-20260926` restored
that checkpoint with the matching resolved configuration and completed the remaining 500 GSM8K
updates and final evaluations.

Across the completed MBPP training task, 80/500 batches (16.0%) had positive mean reward and 73/500
(14.6%) had nonzero gradients. Positive-reward task steps were 57, 75, 93, 97, 104, 111, 120, 126,
134, 144, 146, 147, 149, 150, 151, 165, 174, 186, 204, 205, 212, 220, 222, 228, 252, 255, 257,
258, 259, 266, 273, 274, 282, 284, 288, 294, 296, 306, 309, 311, 312, 313, 316, 320, 327, 328,
329, 330, 336, 342, 345, 348, 350, 360, 363, 365, 367, 383, 384, 390, 420, 426, 435, 436, 438,
444, 450, 453, 458, 468, 471, 473, 474, 475, 480, 482, 489, 490, 492, and 498. Positive-reward
steps 205, 336, 367, 438, 450, 458, and 489 had zero gradient due to zero reward variance. These
training rollouts are not held-out learning evidence. The run predates per-response reward diagnostics,
so exact counts of rewarded completions are unavailable.

The pinned training split has 108 examples and each update samples two prompts, so one deterministic
pass takes 54 updates. Pass 1 had 0/54 positive batches; pass 2 had 5/54 (steps 57, 75, 93, 97, 104);
pass 3 had 10/54 (steps 111, 120, 126, 134, 144, 146, 147, 149, 150, 151); pass 4 had 6/54 (steps
165, 174, 186, 204, 205, 212), five with nonzero gradients; pass 5 completed at 9/54 (steps 220, 222,
228, 252, 255, 257, 258, 259, 266); pass 6 completed at 14/54 (steps 273, 274, 282, 284, 288, 294,
296, 306, 309, 311, 312, 313, 316, 320); pass 7 completed with 13/54 positive batches (steps 327,
328, 329, 330, 336, 342, 345, 348, 350, 360, 363, 365, and 367), twelve with nonzero gradients.
Pass 8 completed with 5/54 positive batches (task steps 383, 384, 390, 420, and 426); all five had nonzero
gradients. The corresponding first-pass
batches for positive steps 57, 75, 93, 97, and 104 had zero reward. Repeated training-prompt successes
are not evidence of held-out learning. All responses so far reached the 128-token cap; the prior
order_1 GSM8K/ARC-Challenge run did too, so cap saturation alone does not explain MBPP's sparse reward.
Pass 9 completed with 14/54 positive batches (steps 435, 436, 438, 444, 450, 453, 458, 468, 471,
473, 474, 475, 480, and 482); steps 438 and 458 had zero gradient. Pass 10 ended at 4/14 positive
batches (steps 489, 490, 492, and 498); step 489 had zero gradient.

The first 108 measured MBPP optimizer steps averaged 54.9 seconds. Updates 361-372 averaged 233.8
seconds (215.2-257.4) with 13.77 GiB mean process RSS. Updates 373-395 averaged 57.4 seconds with
1.89 GiB mean RSS. The slowdown recurred: updates 396-407 averaged 188.9 seconds (130.7-260.7) with
8.71 GiB mean RSS. During update 408, an orphaned PowerShell metrics reader started by earlier agent
monitoring was found using 22.5 GiB of host memory; free physical RAM was 0.34 GiB, with paging
peaking at 694 pages/sec and 46 page reads/sec. After verifying its command and parent, that reader
was stopped without touching the training process. Free RAM immediately rose to 22.13 GiB and was
20.26 GiB at the update 425 sample. Update 408, which overlapped the pressure, took 392.6 seconds;
the twenty-five completed steps after cleanup, updates 409-433, took 208.9, 170.7, 171.2, 172.4, 167.8,
173.8, 174.0, 173.7, 168.7, 170.2, 169.9, 171.6, 177.6, 173.1, 171.4, 173.1, 170.7, 169.3, 170.3, 166.5, 171.0, 134.2, 166.2, 168.3, and 164.1 seconds
(170.7 seconds on average). The reduction suggests
the orphaned reader contributed to the extreme slowdown, but the continuing elevated step time means
its cause is not fully established. At the update 425 resource sample, training-process working set
was 8.72 GiB and logged peak CUDA allocation was 11.62 GB. Source review found no growing
rollout-history collection in the training loop; MBPP responses are scored sequentially through
ephemeral Docker runs.

GSM8K steps 501-918 completed 418 optimizer updates in 27,782 seconds (7.72 optimizer hours), an
average of 66.46 seconds per update (range 50.2-162.6). The last update logged 0.5 mean reward and
zero gradient norm; this training batch is not held-out evidence. Replaying GSM8K from the verified
step-500 checkpoint requires 500 new optimizer updates. At the measured GSM8K pace, that is about
9.23 hours of optimizer time, excluding model/data startup and final held-out/capability evaluation.
All 500 GSM8K recovery updates completed; the latest seven update intervals averaged 54.18 seconds.
These reward and timing values are training diagnostics, not held-out learning evidence. The recovery
also completed final task-boundary and general-capability evaluation and wrote its final checkpoint,
matrix, evaluation records, capability history, and summary.

The sole checkpoint is `task-01-step-00000500`, with global step 500 and next task index 1. Its
manifest lists nine files; all sizes and SHA-256 hashes match, file membership matches, and the
latest-checkpoint pointer matches the manifest digest. The checkpoint identity still matches the run's
configuration hash and model revision; all recorded dataset-manifest hashes and dependency versions
match the current workspace and GPU environment. It preserves the MBPP matrix row (score 0.049479),
adapter, optimizer, RNG state, and capability history. The original run contains 128
MBPP held-out records and the six MBPP capability metrics. It has no GSM8K boundary records, final
matrix row, final capability metrics, summary.json, or task-2 checkpoint. Its top-level performance
matrix still has only the MBPP row.

The source run's process stopped after update 918; its 418 GSM8K updates after the step-500 checkpoint
were not saved and were replayed during recovery. Recovery restored
`general_capability_state_restored` at global step 500 with the frozen-start and MBPP capability rows,
then completed GSM8K through global step 1,000. Across the 500 replayed updates, 441 batches had
positive mean reward, 356 had a nonzero gradient, mean batch reward was 0.4320, and 405 batches
included a zero-variance group. The original source checkpoint still validates: its nine listed files
match their sizes and SHA-256 hashes, and its pointer matches digest
`62d525b27ab30eb312912253123d1f1da79fb3add2105a23d36e1b608457af29`.

The completed matrix is:

```text
after MBPP:  MBPP 0.049479
after GSM8K: MBPP 0.062500, GSM8K 0.265625
```

Average forgetting is -0.0130208 (-1.30 percentage points), backward transfer is +0.0130208, and
final average performance is 0.1640625. MBPP improved slightly after GSM8K; this stream has no
measurable forgetting. Since the first-task reward is bounded to [0, 1] and was 0.049479, its maximum
possible absolute loss is 4.95 percentage points, below the 10-point continuation criterion.
`experimental_success_claimed` is false.

The recovery contains 128 held-out evaluation records per task and 18 capability rows: six at frozen
start and six after each task. After GSM8K, capability changes from frozen were IFEval strict/loose
+0.015625 each, MMLU +0.0703125, HellaSwag +0.046875, ARC-Easy +0.3671875, and perplexity
+0.3817808 (higher is worse). These single-seed diagnostics do not establish method success.

The step-500 source and step-1,000 final checkpoints both passed file-membership, size, and SHA-256
validation; their pointers match manifest hashes `62d525b27ab30eb312912253123d1f1da79fb3add2105a23d36e1b608457af29`
and `c37cb5d39b37b857ebaa4525fa60705fa347631810d32e36ab48ed29a29b22ce`. Source and recovery run
manifests match on configuration hash `706aadf40a4546106b5f85b52ccc36f24c7a0a260b29178d39908db235d59364`,
model/tokenizer revisions, dataset fingerprints, dependencies, and hardware. The verified resume source
and final checkpoint are recorded in `recovery_provenance.json`. Artifacts are under
`outputs/runs/pilot_two_task/rl_lora/order_2/1/phase2-pilot-order2-recovery-20260926/`.

The run started before aggregate reward diagnostics were added to training metrics. Its existing
artifacts therefore cannot distinguish unparseable responses from executed code that failed tests.
New runs log those outcomes without storing generated response text. The original interrupted run
remains preserved alongside the completed recovery artifacts.

## Validation evidence

Validated locally on Windows 11 in both the earlier Python 3.12.3 CPU `.venv` and the current
Python 3.11.15 CUDA-enabled Conda `gpu-env`.

| Check | Result |
| --- | --- |
| Conda `gpu-env` | Python 3.11.15, torch `2.13.0+cu126`, CUDA 12.6, BF16 supported; pilot-profile doctor reports `gpu_experiment: true` |
| GPU environment export | `environment-gpu-win-py311.yml` records 52 Conda build pins and 115 pip entries; fresh recreation remains untested |
| `gpu-env` dependency and code checks | `pip check`, Ruff format/lint, mypy, and all 56 pytest tests passed |
| Tiny CUDA trainer smoke | Completed two task boundaries and wrote a final checkpoint with `launcher.device=cuda`; logged 17-18 MB peak GPU allocation per step, with no task learning observed |
| Pinned pilot model | Qwen2.5-0.5B-Instruct snapshot cached at its configured revision; project LoRA factory loaded it on CUDA with 1,081,344 trainable and 494,032,768 frozen parameters |
| Confirmation-model preflight | Pinned Qwen2.5-1.5B-Instruct loaded on CUDA and completed one RL update plus a two-example task boundary before the full confirmation pilot |
| Enabled pilot capability check | Doctor verified the pinned official IFEval checkout and Punkt resource with `general_capabilities.enabled=true` |
| MBPP isolation backend | Docker Desktop Linux engine 29.2.1 started; the configured sandbox passed a one-test generated-code check with no timeout or infrastructure error |
| Immutable small model revision | Official source revision `7ae557604adf67be50417f59c2c2f167def9a775` pinned |
| Immutable 1.5B revision | Official source revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` pinned |
| Configuration composition | Default, tiny, confirmation, and all method groups compose |
| Ruff formatting and lint | Passed |
| mypy | Passed across 51 source files |
| pytest | Passed; 56 tests (rechecked in the CPU `.venv` on 2026-09-25) |
| Pinned official IFEval adapter | Doctor ready on CPU; upstream strict/loose functions returned `(1, 1)` and `(0, 0)` for synthetic no-comma/comma responses |
| Two-task synthetic integration | One real update per task; frozen weights unchanged; LoRA changed |
| Direct interrupted resume | Final adapter and matrix match uninterrupted within `1e-7` absolute tolerance |
| Capability baseline/resume | Six metrics captured at frozen start and both boundaries; resumed CSV exactly matches uninterrupted |
| Source-backed CLI smoke repeat | Matrix, adapter tensors, learning events, and summary match exactly |
| Source-backed CLI resume | First-boundary interruption plus resume matches uninterrupted artifacts exactly |
| Checkpoint validation | Both task-boundary manifests, file memberships, sizes, and hashes pass for the 0.5B and 1.5B pilots; the order_2 source and recovered final checkpoints also pass |
| 0.5B capable-model pilot | 200 GRPO updates completed; both held-out task boundaries and all six capability metrics recorded; checkpoint hashes pass; forgetting 0.78 pp with GSM8K near floor |
| 1.5B confirmation pilot | 200 GRPO updates completed; both held-out task boundaries and all six capability metrics recorded; checkpoint hashes pass; no forgetting was measurable because GSM8K remained near floor |
| 1.5B duration pilot | 1,000 GRPO updates completed (500/task); held-out matrix and capability history recorded; both checkpoint manifests and all 18 checkpoint files pass hash/size validation; forgetting 1.56 pp, below the 10 pp pilot continuation criterion |
| Phase 2 reward diagnostics | Unit coverage distinguishes parse failures, executed tests, timeouts, output limits, and nonzero exits; integration coverage verifies nested counters reach `metrics.jsonl` |
| Interrupted order_2 source run and recovery | Source stopped at 918/1,000; recovery restored the verified step-500 checkpoint and completed the 1,000-update stream plus final evaluation |
| order_2 final artifacts and provenance | 256 held-out task records, 18 capability rows, final matrix/summary, source/final checkpoint hashes, and verified recovery provenance sidecar |

## Known limitations and pending gate

- VS Code now defaults to Conda `gpu-env`, which has a working CUDA build. The older `.venv` still
  has CPU-only PyTorch. The GPU environment export is recorded, but fresh recreation has not been
  tested. Checkpoints save at task boundaries; interruption during a task loses its unsaved optimizer
  progress. Recovery therefore replays GSM8K from the verified step-500 checkpoint.
- The pinned Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct models are cached and load on CUDA. The
  1.5B confirmation pilot completed without OOM, though GPU memory reached 16,016 MiB used with
  roughly 48 MiB free during evaluation. The locally stored Hugging Face login was verified against
  both pinned Qwen model revisions and configured public datasets; no new credential is needed.
- The Docker Desktop Linux engine is available and the configured MBPP sandbox passed a small
  execution check. Both MBPP and GSM8K held-out boundary evaluations completed for order_2. Its low
  MBPP baseline score limits the maximum possible loss to 4.95 percentage points, so this stream
  cannot meet the 10-point continuation rule.
- Capability evaluation requires a local checkout of the pinned Google Research IFEval source and
  the pinned NLTK Punkt resource. Both are present in the local cache and pass the built-in
  adapter's verification when the scorer repository is explicitly configured.
- Git is initialized locally, but there is no initial commit, configured author name/email, or
  remote. Manifests cannot record a commit hash until an authorized initial commit exists.
- The CPU smoke evaluates configured subsets and is not headline scientific evidence.

## Next action

Proceed with Phase 3: implement and validate the direct HLoRA-to-RL baseline, including the documented
supervised reproduction and matched two-task smoke. Before Phase 8, select a task stream with enough
standard-baseline headroom to test the 10-point continuation rule; the current MBPP -> GSM8K pilot
cannot establish that signal.
