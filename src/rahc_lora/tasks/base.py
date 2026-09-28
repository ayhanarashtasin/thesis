"""Common continual-task protocol and immutable split records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from rahc_lora.rewards.base import RewardResult


class DatasetRole(StrEnum):
    """Scientifically distinct data roles that may never be silently mixed."""

    TRAIN = "train"
    VALIDATION = "validation"
    ANCHOR_CANDIDATE = "anchor_candidate"
    TEST = "test"
    EVALUATION_ONLY = "evaluation_only"


def canonical_content_hash(fields: dict[str, Any]) -> str:
    """Hash a JSON-serializable example deterministically."""

    try:
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Task example fields must be JSON serializable: {error}") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskExample:
    """One example with an explicit, immutable scientific role."""

    task_id: str
    example_id: str
    role: DatasetRole
    fields: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.example_id:
            raise ValueError("TaskExample task_id and example_id must not be empty")
        canonical_content_hash(self.fields)

    @property
    def content_sha256(self) -> str:
        return canonical_content_hash(self.fields)


@dataclass(frozen=True)
class TaskSplits:
    """Disjoint train, validation, anchor-candidate, and final-test partitions."""

    train: tuple[TaskExample, ...]
    validation: tuple[TaskExample, ...]
    anchor_candidates: tuple[TaskExample, ...]
    test: tuple[TaskExample, ...]

    def __post_init__(self) -> None:
        partitions = {
            DatasetRole.TRAIN: self.train,
            DatasetRole.VALIDATION: self.validation,
            DatasetRole.ANCHOR_CANDIDATE: self.anchor_candidates,
            DatasetRole.TEST: self.test,
        }
        seen_ids: dict[str, DatasetRole] = {}
        seen_hashes: dict[str, DatasetRole] = {}
        for expected_role, examples in partitions.items():
            for example in examples:
                if example.role is not expected_role:
                    raise ValueError(
                        f"Example {example.example_id!r} has role {example.role.value!r}; "
                        f"expected {expected_role.value!r}"
                    )
                previous_role = seen_ids.get(example.example_id)
                if previous_role is not None:
                    raise ValueError(
                        f"Example ID {example.example_id!r} overlaps roles "
                        f"{previous_role.value!r} and {expected_role.value!r}"
                    )
                previous_hash_role = seen_hashes.get(example.content_sha256)
                if previous_hash_role is not None:
                    raise ValueError(
                        f"Example content {example.content_sha256} overlaps roles "
                        f"{previous_hash_role.value!r} and {expected_role.value!r}"
                    )
                seen_ids[example.example_id] = expected_role
                seen_hashes[example.content_sha256] = expected_role

    def for_role(self, role: DatasetRole) -> tuple[TaskExample, ...]:
        """Return the exact configured partition for a trainable task role."""

        mapping = {
            DatasetRole.TRAIN: self.train,
            DatasetRole.VALIDATION: self.validation,
            DatasetRole.ANCHOR_CANDIDATE: self.anchor_candidates,
            DatasetRole.TEST: self.test,
        }
        if role not in mapping:
            raise ValueError(f"TaskSplits does not expose role {role.value!r}")
        return mapping[role]


class ContinualTask(Protocol):
    """Common task contract used by training and evaluation orchestration."""

    task_id: str

    def load_train(self) -> Sequence[TaskExample]: ...

    def load_validation(self) -> Sequence[TaskExample]: ...

    def load_anchor_candidates(self) -> Sequence[TaskExample]: ...

    def load_test(self) -> Sequence[TaskExample]: ...

    def format_prompt(self, example: TaskExample) -> str: ...

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult: ...

    def score_response(self, example: TaskExample, response: str) -> float: ...

    def aggregate_metrics(self, records: Sequence[RewardResult]) -> dict[str, float]: ...


class BaseContinualTask:
    """Role-safe split access and common reward aggregation."""

    task_id: str

    def __init__(self, task_id: str, splits: TaskSplits) -> None:
        self.task_id = task_id
        self._splits = splits
        task_ids = {
            example.task_id
            for role in (
                DatasetRole.TRAIN,
                DatasetRole.VALIDATION,
                DatasetRole.ANCHOR_CANDIDATE,
                DatasetRole.TEST,
            )
            for example in splits.for_role(role)
        }
        if task_ids and task_ids != {task_id}:
            raise ValueError(f"Task splits contain unexpected task IDs: {sorted(task_ids)}")

    def load_train(self) -> Sequence[TaskExample]:
        return self._splits.train

    def load_validation(self) -> Sequence[TaskExample]:
        return self._splits.validation

    def load_anchor_candidates(self) -> Sequence[TaskExample]:
        return self._splits.anchor_candidates

    def load_test(self) -> Sequence[TaskExample]:
        return self._splits.test

    def score_response(self, example: TaskExample, response: str) -> float:
        return self.evaluate_response(example, response).normalized_reward

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        raise NotImplementedError

    def aggregate_metrics(self, records: Sequence[RewardResult]) -> dict[str, float]:
        if not records:
            raise ValueError(f"Cannot aggregate empty reward records for task {self.task_id!r}")
        return {
            "mean_reward": sum(record.normalized_reward for record in records) / len(records),
            "parse_success_rate": sum(record.parse_succeeded for record in records) / len(records),
            "count": float(len(records)),
        }
