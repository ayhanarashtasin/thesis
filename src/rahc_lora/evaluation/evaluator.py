"""Read-only deterministic policy evaluation and task-boundary matrix updates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rahc_lora.evaluation.performance_matrix import PerformanceMatrix
from rahc_lora.rewards.base import RewardResult
from rahc_lora.tasks.base import ContinualTask


@dataclass(frozen=True)
class EvaluationGenerationConfig:
    """Fixed generation controls shared by comparable evaluations."""

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("Evaluation max_new_tokens must be positive")
        if self.do_sample or self.temperature != 0.0:
            raise ValueError("Phase 1 deterministic evaluation requires greedy generation")
        if not 0 < self.top_p <= 1:
            raise ValueError("Evaluation top_p must be in (0, 1]")


class EvaluationPolicy(Protocol):
    """Read-only batch generation interface used by task evaluators."""

    def generate(
        self, prompts: Sequence[str], generation: EvaluationGenerationConfig
    ) -> Sequence[str]:
        """Generate exactly one response per prompt without updating policy state."""


@dataclass(frozen=True)
class EvaluationRecord:
    """Privacy-conscious per-example evaluation artifact."""

    task_id: str
    example_id: str
    prompt_sha256: str
    response_sha256: str
    reward: RewardResult


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Aggregate score and all hashed per-example reward records."""

    task_id: str
    mean_reward: float
    metrics: dict[str, float]
    records: tuple[EvaluationRecord, ...]


def evaluate_task(
    task: ContinualTask,
    policy: EvaluationPolicy,
    generation: EvaluationGenerationConfig,
    *,
    max_examples: int | None = None,
) -> TaskEvaluationResult:
    """Evaluate only the final-test role with fixed generation and no training-state mutation."""

    if max_examples is not None and max_examples <= 0:
        raise ValueError("Evaluation max_examples must be positive when provided")
    complete_examples = tuple(task.load_test())
    examples = complete_examples[:max_examples] if max_examples is not None else complete_examples
    if not examples:
        raise ValueError(f"Task {task.task_id!r} has no final-test examples")
    prompts = tuple(task.format_prompt(example) for example in examples)
    responses = tuple(policy.generate(prompts, generation))
    if len(responses) != len(prompts):
        raise ValueError(
            f"Policy returned {len(responses)} responses for {len(prompts)} prompts "
            f"on task {task.task_id!r}"
        )
    rewards = tuple(
        task.evaluate_response(example, response)
        for example, response in zip(examples, responses, strict=True)
    )
    records = tuple(
        EvaluationRecord(
            task_id=task.task_id,
            example_id=example.example_id,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            response_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest(),
            reward=reward,
        )
        for example, prompt, response, reward in zip(
            examples, prompts, responses, rewards, strict=True
        )
    )
    metrics = task.aggregate_metrics(rewards)
    return TaskEvaluationResult(
        task_id=task.task_id,
        mean_reward=metrics["mean_reward"],
        metrics=metrics,
        records=records,
    )


def evaluate_task_boundary(
    *,
    after_task_id: str,
    learned_tasks: Sequence[ContinualTask],
    policy: EvaluationPolicy,
    generation: EvaluationGenerationConfig,
    matrix: PerformanceMatrix,
    max_examples_per_task: int | None = None,
) -> tuple[TaskEvaluationResult, ...]:
    """Evaluate every learned task, then append exactly one complete boundary row."""

    results = tuple(
        evaluate_task(
            task,
            policy,
            generation,
            max_examples=max_examples_per_task,
        )
        for task in learned_tasks
    )
    if len({result.task_id for result in results}) != len(results):
        raise ValueError("Task-boundary evaluation received duplicate task IDs")
    matrix.append_row(
        after_task_id,
        {result.task_id: result.mean_reward for result in results},
    )
    return results
