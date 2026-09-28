"""Rollout and reinforcement-learning components."""

from rahc_lora.rl.advantages import AdvantageResult, normalize_group_advantages
from rahc_lora.rl.rollout import (
    RewardDiagnostics,
    RolloutBatch,
    RolloutRecord,
    SampledResponse,
)

__all__ = [
    "AdvantageResult",
    "RewardDiagnostics",
    "RolloutBatch",
    "RolloutRecord",
    "SampledResponse",
    "normalize_group_advantages",
]
