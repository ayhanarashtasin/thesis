from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from rahc_lora.evaluation.continual_metrics import (
    average_forgetting,
    final_average_performance,
)
from rahc_lora.evaluation.evaluator import (
    EvaluationGenerationConfig,
    evaluate_task,
    evaluate_task_boundary,
)
from rahc_lora.evaluation.performance_matrix import PerformanceMatrix
from rahc_lora.tasks.base import DatasetRole, TaskExample, TaskSplits
from rahc_lora.tasks.math_task import GSM8KTask
from rahc_lora.tasks.science_task import ARCChallengeTask


def _splits(task_id: str, fields_for_role: dict[DatasetRole, dict[str, object]]) -> TaskSplits:
    def one(role: DatasetRole) -> tuple[TaskExample, ...]:
        return (
            TaskExample(
                task_id=task_id,
                example_id=f"{task_id}:{role.value}",
                role=role,
                fields=fields_for_role[role],
            ),
        )

    return TaskSplits(
        train=one(DatasetRole.TRAIN),
        validation=one(DatasetRole.VALIDATION),
        anchor_candidates=one(DatasetRole.ANCHOR_CANDIDATE),
        test=one(DatasetRole.TEST),
    )


def _math_task() -> GSM8KTask:
    fields = {
        role: {"question": f"What is 20 + 22? ({role.value})", "answer": "#### 42"}
        for role in (
            DatasetRole.TRAIN,
            DatasetRole.VALIDATION,
            DatasetRole.ANCHOR_CANDIDATE,
            DatasetRole.TEST,
        )
    }
    return GSM8KTask(_splits("gsm8k", fields))


def _science_task() -> ARCChallengeTask:
    fields = {
        role: {
            "question": f"Which is a planet? ({role.value})",
            "choices": {"label": ["A", "B"], "text": ["Venus", "Granite"]},
            "answerKey": "A",
        }
        for role in (
            DatasetRole.TRAIN,
            DatasetRole.VALIDATION,
            DatasetRole.ANCHOR_CANDIDATE,
            DatasetRole.TEST,
        )
    }
    return ARCChallengeTask(_splits("arc_challenge", fields))


@dataclass
class MappingPolicy:
    responses: dict[str, str]

    def generate(
        self, prompts: Sequence[str], generation: EvaluationGenerationConfig
    ) -> Sequence[str]:
        return [self.responses[prompt] for prompt in prompts]


def test_mock_policy_two_task_matrix_and_summary_are_correct_and_deterministic() -> None:
    math_task = _math_task()
    science_task = _science_task()
    generation = EvaluationGenerationConfig()
    math_prompt = math_task.format_prompt(math_task.load_test()[0])
    science_prompt = science_task.format_prompt(science_task.load_test()[0])

    first_policy = MappingPolicy({math_prompt: "Final answer: 42"})
    repeated_first = evaluate_task(math_task, first_policy, generation)
    assert evaluate_task(math_task, first_policy, generation) == repeated_first

    matrix = PerformanceMatrix(("gsm8k", "arc_challenge"))
    evaluate_task_boundary(
        after_task_id="gsm8k",
        learned_tasks=(math_task,),
        policy=first_policy,
        generation=generation,
        matrix=matrix,
    )
    assert matrix.values == ((1.0, None),)

    second_policy = MappingPolicy({math_prompt: "Final answer: 41", science_prompt: "Answer: A"})
    evaluate_task_boundary(
        after_task_id="arc_challenge",
        learned_tasks=(math_task, science_task),
        policy=second_policy,
        generation=generation,
        matrix=matrix,
    )
    assert matrix.values == ((1.0, None), (0.0, 1.0))
    assert average_forgetting(matrix) == pytest.approx(1.0)
    assert final_average_performance(matrix) == pytest.approx(0.5)
