"""Finite, epsilon-safe within-group GRPO advantage normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdvantageResult:
    """Normalized advantages ``[G, K]`` and clipping diagnostics."""

    values_gk: Tensor
    zero_variance_groups: int
    clipped_fraction: float


def normalize_group_advantages(
    rewards_gk: Tensor,
    *,
    epsilon: float,
    clip: float,
) -> AdvantageResult:
    """Normalize rewards within each group using population standard deviation.

    ``G`` is the number of prompts and ``K`` is the number of responses per prompt.
    Groups with standard deviation at or below ``epsilon`` receive exactly zero
    advantages. The returned tensor is FP32 and finite or the function fails.
    """

    if rewards_gk.ndim != 2 or rewards_gk.shape[1] <= 1:
        raise ValueError("GRPO rewards must have shape [G, K] with K > 1")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("Advantage epsilon must be finite and positive")
    if clip <= 0 or not math.isfinite(clip):
        raise ValueError("Advantage clip must be finite and positive")
    rewards = rewards_gk.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("GRPO rewards contain non-finite values")
    means = rewards.mean(dim=1, keepdim=True)
    scales = rewards.std(dim=1, correction=0, keepdim=True)
    zero_variance = scales <= epsilon
    safe_scales = torch.where(zero_variance, torch.ones_like(scales), scales)
    unbounded = (rewards - means) / safe_scales
    unbounded = torch.where(zero_variance, torch.zeros_like(unbounded), unbounded)
    values = unbounded.clamp(min=-clip, max=clip)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("GRPO advantage normalization produced non-finite values")
    clipped = (values != unbounded).to(dtype=torch.float32).mean().item()
    return AdvantageResult(
        values_gk=values,
        zero_variance_groups=int(zero_variance.sum().item()),
        clipped_fraction=float(clipped),
    )
