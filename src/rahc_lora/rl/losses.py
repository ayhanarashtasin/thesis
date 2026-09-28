"""Project-owned clipped GRPO policy objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PolicyLossResult:
    """Scalar policy loss and detached rollout diagnostics."""

    loss: Tensor
    sampled_policy_kl: float
    ratio_clip_fraction: float


def clipped_grpo_loss(
    new_logprobs_bt: Tensor,
    old_logprobs_bt: Tensor,
    advantages_b: Tensor,
    valid_mask_bt: Tensor,
    *,
    clip_epsilon: float,
) -> PolicyLossResult:
    """Return the token-mean clipped GRPO surrogate loss.

    Shapes are ``new_logprobs_bt=[B,T]``, ``old_logprobs_bt=[B,T]``,
    ``advantages_b=[B]``, and ``valid_mask_bt=[B,T]``. The sampled policy KL is
    the valid-token mean of ``old_logprob - new_logprob`` and is diagnostic only.
    """

    if new_logprobs_bt.shape != old_logprobs_bt.shape:
        raise ValueError("New and old policy log-probabilities must have equal shape")
    if new_logprobs_bt.ndim != 2 or valid_mask_bt.shape != new_logprobs_bt.shape:
        raise ValueError("Policy log-probabilities and mask must have shape [B, T]")
    if advantages_b.shape != (new_logprobs_bt.shape[0],):
        raise ValueError("Advantages must have shape [B]")
    if not 0 < clip_epsilon < 1 or not math.isfinite(clip_epsilon):
        raise ValueError("Policy clip epsilon must be finite and in (0, 1)")
    mask = valid_mask_bt.to(dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("GRPO policy loss requires at least one valid response token")
    for name, tensor in (
        ("new_logprobs", new_logprobs_bt),
        ("old_logprobs", old_logprobs_bt),
        ("advantages", advantages_b),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"GRPO {name} contain non-finite values")
    log_ratio = new_logprobs_bt - old_logprobs_bt
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    advantage_bt = advantages_b[:, None].expand_as(ratio)
    surrogate = torch.minimum(ratio * advantage_bt, clipped_ratio * advantage_bt)
    loss = -surrogate.masked_select(mask).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("GRPO policy loss is non-finite")
    with torch.no_grad():
        sampled_kl = (old_logprobs_bt - new_logprobs_bt).masked_select(mask).mean()
        clip_fraction = (ratio != clipped_ratio).masked_select(mask).to(torch.float32).mean()
    return PolicyLossResult(
        loss=loss,
        sampled_policy_kl=float(sampled_kl.item()),
        ratio_clip_fraction=float(clip_fraction.item()),
    )
