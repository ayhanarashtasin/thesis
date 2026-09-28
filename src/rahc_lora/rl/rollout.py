"""Typed rollout records that preserve prompt groups and reward provenance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rahc_lora.rewards.base import RewardResult


@dataclass(frozen=True)
class RewardDiagnostics:
    """Aggregate parse and isolated-code outcomes without retaining response text."""

    parse_failure_count: int = 0
    code_execution_count: int = 0
    code_tests_passed: int = 0
    code_tests_total: int = 0
    code_timeout_count: int = 0
    code_output_limited_count: int = 0
    code_nonzero_exit_count: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.parse_failure_count,
            self.code_execution_count,
            self.code_tests_passed,
            self.code_tests_total,
            self.code_timeout_count,
            self.code_output_limited_count,
            self.code_nonzero_exit_count,
        )
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Reward diagnostic counts must be nonnegative integers")
        if self.code_tests_passed > self.code_tests_total:
            raise ValueError("Passed code tests cannot exceed the total test count")

    @classmethod
    def from_results(cls, results: Sequence[RewardResult]) -> RewardDiagnostics:
        """Summarize only validated scalar reward metadata for one rollout batch."""

        parse_failures = 0
        executions = 0
        tests_passed = 0
        tests_total = 0
        timeouts = 0
        output_limits = 0
        nonzero_exits = 0

        for result in results:
            if not result.parse_succeeded:
                parse_failures += 1

            details = result.details
            has_passed = "passed" in details
            has_total = "total" in details
            if has_passed != has_total:
                raise ValueError("Code reward diagnostics must contain both passed and total")
            if not result.parse_succeeded or not has_passed:
                continue

            passed = details["passed"]
            total = details["total"]
            if (
                type(passed) is not int
                or type(total) is not int
                or total <= 0
                or passed < 0
                or passed > total
            ):
                raise ValueError("Code reward passed/total details must be valid integers")
            executions += 1
            tests_passed += passed
            tests_total += total

            timed_out = details.get("timed_out", False)
            output_limited = details.get("output_limited", False)
            if type(timed_out) is not bool or type(output_limited) is not bool:
                raise ValueError("Code reward timeout/output-limit details must be booleans")
            timeouts += int(timed_out)
            output_limits += int(output_limited)

            exit_code = details.get("exit_code", 0)
            if type(exit_code) is not int:
                raise ValueError("Code reward exit_code detail must be an integer")
            nonzero_exits += int(exit_code != 0)

        return cls(
            parse_failure_count=parse_failures,
            code_execution_count=executions,
            code_tests_passed=tests_passed,
            code_tests_total=tests_total,
            code_timeout_count=timeouts,
            code_output_limited_count=output_limits,
            code_nonzero_exit_count=nonzero_exits,
        )

    def as_dict(self) -> dict[str, int]:
        """Return the JSON-safe scalar counters written to structured metrics."""

        return {
            "parse_failure_count": self.parse_failure_count,
            "code_execution_count": self.code_execution_count,
            "code_tests_passed": self.code_tests_passed,
            "code_tests_total": self.code_tests_total,
            "code_timeout_count": self.code_timeout_count,
            "code_output_limited_count": self.code_output_limited_count,
            "code_nonzero_exit_count": self.code_nonzero_exit_count,
        }


@dataclass(frozen=True)
class SampledResponse:
    """One policy sample before task reward calculation."""

    prompt: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_text: str
    old_policy_logprobs: tuple[float, ...]
    sampling_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.prompt_token_ids or not self.response_token_ids:
            raise ValueError("SampledResponse token sequences must not be empty")
        if len(self.response_token_ids) != len(self.old_policy_logprobs):
            raise ValueError("Each response token requires one old-policy log-probability")
        if not all(math.isfinite(value) for value in self.old_policy_logprobs):
            raise ValueError("SampledResponse old-policy log-probabilities must be finite")


@dataclass(frozen=True)
class RolloutRecord:
    """One scored response in a contiguous GRPO prompt group."""

    task_id: str
    example_id: str
    prompt_group: int
    sample_index: int
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_text: str
    old_policy_logprobs: tuple[float, ...]
    sampling_metadata: dict[str, Any]
    raw_reward: float
    normalized_reward: float
    advantage: float
    valid_token_mask: tuple[bool, ...]
    policy_version: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.example_id:
            raise ValueError("RolloutRecord task and example IDs must not be empty")
        if self.prompt_group < 0 or self.sample_index < 0 or self.policy_version < 0:
            raise ValueError("Rollout indices and policy version must be nonnegative")
        response_length = len(self.response_token_ids)
        if response_length == 0:
            raise ValueError("RolloutRecord response tokens must not be empty")
        if len(self.old_policy_logprobs) != response_length:
            raise ValueError("RolloutRecord old log-probabilities have the wrong length")
        if len(self.valid_token_mask) != response_length or not all(self.valid_token_mask):
            raise ValueError("RolloutRecord must mark every stored response token as valid")
        values = (
            *self.old_policy_logprobs,
            self.raw_reward,
            self.normalized_reward,
            self.advantage,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("RolloutRecord contains a non-finite scientific value")
        if not 0 <= self.raw_reward <= 1 or not 0 <= self.normalized_reward <= 1:
            raise ValueError("Rollout rewards must be in [0, 1]")


@dataclass(frozen=True)
class RolloutBatch:
    """A complete set of prompt groups that may not be split during an update."""

    records: tuple[RolloutRecord, ...]
    group_size: int
    zero_variance_groups: int = 0
    reward_versions: tuple[str, ...] = field(default_factory=tuple)
    reward_diagnostics: RewardDiagnostics = field(default_factory=RewardDiagnostics)

    def __post_init__(self) -> None:
        if self.group_size <= 1:
            raise ValueError("GRPO rollout group_size must be greater than one")
        if not self.records or len(self.records) % self.group_size:
            raise ValueError("RolloutBatch must contain complete prompt groups")
        expected_groups = len(self.records) // self.group_size
        groups: dict[int, list[RolloutRecord]] = {}
        for record in self.records:
            groups.setdefault(record.prompt_group, []).append(record)
        if set(groups) != set(range(expected_groups)):
            raise ValueError("RolloutBatch prompt groups must be contiguous from zero")
        for group, records in groups.items():
            if len(records) != self.group_size:
                raise ValueError(f"Rollout prompt group {group} has the wrong size")
            if {record.sample_index for record in records} != set(range(self.group_size)):
                raise ValueError(f"Rollout prompt group {group} has invalid sample indices")
            if len({record.example_id for record in records}) != 1:
                raise ValueError(f"Rollout prompt group {group} mixes examples")
        if not 0 <= self.zero_variance_groups <= expected_groups:
            raise ValueError("RolloutBatch zero-variance group count is invalid")

    @property
    def response_token_count(self) -> int:
        return sum(len(record.response_token_ids) for record in self.records)
