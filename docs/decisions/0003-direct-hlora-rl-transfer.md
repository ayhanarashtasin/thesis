# ADR 0003: Direct HLoRA transfer to the shared RL-LoRA trainer

## Status

Accepted for the Phase 3 reference baseline.

## Context

The project requires a direct HLoRA-to-RL baseline but the source method is supervised. Song et al.,
"How to Alleviate Catastrophic Forgetting in LLMs Finetuning? Hierarchical Layer-Wise and
Element-Wise Regularization," arXiv:2501.13669v2, records element-wise importance from a supervised
general-domain task, applies an importance-weighted parameter penalty on later domain tasks, and
weights each layer by the softmax of its importance norm. Its experiments use Pile as the general
reference and SciQ, PiQA, and MedMCQA as domain tasks.

The project transfers this baseline to policy-gradient training. The optimizer is AdamW with clipping,
so the original paper's closed-form SGD update for the LoRA product does not describe the actual
parameter movement. Importance must use the gradient of the real RL loss and the observed effective
LoRA movement.

## Decision

For each configured LoRA target layer, let `U_l = scale_l * B_l @ A_l` be its effective adapter
update. During the first task, capture the gradient `G_l = d L_RL / d U_l` at the target linear layer
and measure the actual post-optimizer movement `dU_l = U_l(after) - U_l(before)`. Accumulate the
Synaptic-Intelligence path contribution:

```text
omega_l += -G_l * dU_l
```

At the task boundary, calculate the dense nonnegative importance and static layer coefficients:

```text
Omega_l = clamp_min(omega_l, 0) / ((U_l(end) - U_l(start)) ** 2 + epsilon)
alpha_l = softmax_l(||Omega_l||_2)
```

For the next task, add the parameter penalty to the ordinary clipped RL objective:

```text
L = L_RL + lambda * sum_l alpha_l * sum_ij Omega_l[i,j] *
                          (U_l(current)[i,j] - U_l(reference)[i,j]) ** 2
```

The reference is the shared LoRA policy at the latest task boundary. Importance is accumulated
across completed tasks; layer coefficients are computed at each boundary and remain fixed during
the next task. The current task's unregularized RL gradient contributes to the next boundary's
importance. The method adds no task-specific adapter.

The effective-weight gradient is captured from the LoRA target layer's actual input after its
configured dropout and the gradient at its output. This yields the exact gradient of the dense
effective update for the forward pass, without reconstructing it from factor gradients. The path
contribution uses that pre-clipping objective gradient and the actual post-clipping, post-AdamW
movement. Negative accumulated contributions are clamped to zero at task consolidation, matching the
required nonnegative importance state.

The checkpoint stores the shared reference LoRA factors, dense importance tensors, layer
coefficients, and cumulative diagnostic counters. Reference effective updates are recomputed from the
stored factors and the pinned LoRA scaling configuration. This keeps the reference state compact
while retaining dense importance for the explicit Phase 3 reference baseline.

## Consequences

- The method is a documented HLoRA/SI-style RL adaptation, not a claim to reproduce the paper's
  supervised results or optimizer exactly.
- Dense importance has `O x I` values per LoRA target layer and is restricted to the HLoRA reference
  method and small-model diagnostics. The later RAHC-LoRA method must use bounded factorized state.
- Importance is based on the clipped GRPO policy-loss gradient, not raw reward. Reward-aware
  importance remains a later Phase 4 component.
- The supervised reproduction remains a separate experiment with its own immutable data, split,
  optimizer, seed, storage, and runtime record.

## Validation

The implementation must test dense tensor shapes and nonnegativity, exact zero penalty for zero
importance, zero path contribution for zero movement, exact reference-state round trips, checkpoint
resume, and equivalence of the disabled HLoRA configuration to the standard RL-LoRA objective.

## Source

- Song et al. (2025), arXiv:2501.13669v2: https://arxiv.org/abs/2501.13669v2
- Project Phase 3 requirements: `Goal.md`, Section 9; `project.md`, Phase 3 exit condition.
