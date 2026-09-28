---
name: rahc-consolidation
description: "Implement or review RAHC-LoRA retention mechanics: effective LoRA weights, reward-aware importance, factorized penalties, anchor KL, conflict gating, and dual control. Use whenever these equations, tensors, states, or their trainer integration are involved."
---

# RAHC-LoRA consolidation mechanics

Read the relevant sections of `project.md` (7.1-7.9), `architecture.md` (11-24), and their associated tests before changing consolidation code. `project.md` is the mathematical authority.

## Core invariants

- Protect the effective update, `DeltaW = (alpha / rank) * B @ A`, not `A` and `B` independently. Apply the scale exactly once.
- The combined objective is `L_RL_current + lambda_functional * L_anchor_KL + lambda_parameter * sum_l(c_l * R_l)`. Reference distributions, reference LoRA factors, importance factors, and detached conflict coefficients must not receive gradients.
- On the first task, every retention term is zero. With no historical state, the method must agree with standard RL-LoRA.
- Persistent state is bounded: factorized importance, one reference LoRA state, fixed-capacity anchors, and compact controller state. Dense importance may be a small-model oracle only, never persistent main-method state.

## Update and boundary discipline

For an optimizer step, calculate the current RL objective; optionally calculate anchor KL; capture detached current and anchor effective gradients; compute detached conflicts and differentiable penalties; snapshot pre-step factors; optimize; observe the actual post-step effective-weight movement; then update importance and the dual controller. Do not infer an SGD-like movement in place of the observed optimizer step.

At a task boundary: finalize and combine importance; store current LoRA factors as reference; add/rebalance eligible anchors under the global budget; evaluate all learned tasks and fixed capabilities; then atomically checkpoint complete state.

## Mechanism-specific rules

- Importance uses the normalized/clipped-advantage RL gradient, `relu(-gradient * actual_delta) / (actual_delta^2 + epsilon)`, robust clipping, and FP32 accumulation. Fail on non-finite values.
- For rank-one importance, preserve nonnegative row and column mass. Document and test normalization. Production penalties use the low-rank Gram form; compare it with dense weighted Frobenius computation on small tensors.
- Anchor KL uses stored top-k reference tokens plus residual mass. Only current-policy probabilities are differentiable. Anchor records are selected from allowed training candidates and remain under count and byte budgets.
- Conflict is positive only for opposing effective gradients; zero-norm and aligned gradients yield zero. Normalize, clip, and detach coefficients.
- The functional multiplier uses smoothed KL, remains nonnegative and bounded, and terminates with a diagnostic after configured persistent saturation.

## Tests required by a change

Choose the affected tests, but do not omit: effective-weight rescaling invariance; dense-versus-efficient penalty agreement; nonnegative/finite importance; zero first-task retention; valid compact distributions and KL behaviour; gradient conflict edge cases; controller direction and state round-trip; and checkpoint persistence for changed state.

Use `dense_reference` only as an explicitly selected small-model numerical oracle and `chunked_hooks` for memory-bounded main runs. Never silently switch modes.
