"""Common deterministic reward result and scorer contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RewardResult:
    """A deterministic task reward with parsing and diagnostic metadata."""

    raw_reward: float
    normalized_reward: float
    parse_succeeded: bool
    parsed_response: str | None
    reward_version: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("raw_reward", self.raw_reward),
            ("normalized_reward", self.normalized_reward),
        ):
            if not math.isfinite(value):
                raise ValueError(f"RewardResult.{name} must be finite, received {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"RewardResult.{name} must be in [0, 1], received {value!r}")
        if not self.reward_version:
            raise ValueError("RewardResult.reward_version must not be empty")
        if self.parse_succeeded and self.parsed_response is None:
            raise ValueError("A successful parse requires parsed_response")


class RewardScorer(Protocol):
    """Project-owned deterministic reward interface."""

    reward_version: str

    def score(self, reference: Any, response: str) -> RewardResult:
        """Score one response deterministically in the documented range [0, 1]."""
