# ADR 0002: Phase 2 GRPO reference semantics

- Status: Accepted for the standard RL-LoRA baseline
- Date: 2026-09-22
- Author: RAHC-LoRA project

## Context

The project requires a reproducible GRPO baseline, but the source documents intentionally leave
several implementation details open: the variance estimator for group normalization, behavior for
constant-reward groups, the exact clipped objective reduction, and the meaning of the logged policy
KL. Those choices affect gradients and must not remain library defaults.

## Decision

The Phase 2 reference uses a project-owned rollout and update loop with these definitions:

1. Keep all `K` responses for one prompt contiguous through generation, reward calculation,
   normalization, and optimization.
2. Normalize deterministic rewards within each prompt group using the population standard
   deviation (`correction=0`) in FP32.
3. If group standard deviation is at or below configured `advantage_epsilon`, assign exactly zero
   advantage to the entire group and count it as a zero-variance group.
4. Clip normalized advantages symmetrically to configured `advantage_clip` and log the clipping
   fraction.
5. Store the rollout policy's log-probability for every generated response token.
6. Recompute current log-probabilities and optimize the token-mean clipped surrogate:

   ```text
   ratio_bt = exp(new_logprob_bt - old_logprob_bt)
   surrogate_bt = min(
       ratio_bt * advantage_b,
       clip(ratio_bt, 1-epsilon, 1+epsilon) * advantage_b,
   )
   policy_loss = -mean(surrogate_bt over valid response tokens)
   ```

7. Use one optimizer update per rollout in Phase 2. The frozen backbone is never an optimizer
   parameter; only the single shared LoRA adapter is updated and it is not reset at task boundaries.
8. Log `mean(old_logprob - new_logprob)` over sampled valid tokens as `sampled_policy_kl`. It is a
   diagnostic of update size, not an anchor KL or retention penalty.
9. Standard RL-LoRA has no anchor memory, replay, base-policy KL, parameter consolidation,
   importance, conflict gate, or dual controller. Configuration validation rejects those hidden
   components when `method.name=rl_lora`.

## Alternatives considered

- Sample standard deviation: rejected because its small-group scaling differs and is undefined for
  `K=1`; configuration already requires `K>1`.
- Divide by `std + epsilon` for constant groups: rejected because numerical noise could create an
  arbitrary gradient when all rewards are identical.
- Sequence-level ratios: rejected for the reference because response lengths vary and the intended
  implementation stores token-level log-probabilities and masks.
- Delegate GRPO semantics to TRL: rejected because dependency upgrades could silently change the
  scientific objective. Transformers and PEFT remain behind project-owned policy interfaces.
- Add reference-policy KL to the standard baseline: rejected because it would turn baseline 1 into
  a different required baseline. The initial-policy/current-prompt KL baseline is implemented in a
  later phase through configuration, not hidden here.

## Scientific consequences

- Sparse deterministic rewards can create zero-variance groups and therefore intentionally produce
  no update. This is logged rather than hidden or replaced with a heuristic reward.
- Rollout group size, epsilon, both clipping thresholds, generation, optimizer, and scheduler are
  immutable configuration and part of run identity.
- The tiny CPU smoke validates mechanics and reproducibility but cannot establish learnability or
  forgetting on GSM8K or ARC-Challenge.
- Pilot evidence must report current-task learning together with prior-task change. An all-zero
  matrix is a failed/underpowered benchmark outcome, not retention success.

## Engineering consequences

- Rollout records contain task/example IDs, token IDs, old log-probabilities, sampling metadata,
  raw and normalized rewards, advantages, masks, and policy version.
- Every step fails before the optimizer mutation if rewards, advantages, loss, or gradients are
  non-finite.
- Checkpoints restore adapter, optimizer, scheduler, task/step/rollout counters, explicit rollout
  generator, general RNG states, performance matrix, configuration, identities, and explicit empty
  retention state for this baseline.

## Validation

- Unit tests cover finite normalization, constant groups, invalid rewards, clipped-loss backward,
  frozen parameters, and complete grouped sampling.
- A deterministic synthetic-reward integration run proves that LoRA parameters change while all
  frozen parameters remain bit-identical.
- Interrupted two-task training resumes within the configured `1e-7` absolute tolerance and matches
  the uninterrupted adapter and performance matrix.
- Two source-backed CLI smoke runs match in adapter tensors, learning events, summary, and matrix.
