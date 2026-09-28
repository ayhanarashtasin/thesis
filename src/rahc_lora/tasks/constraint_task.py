"""Custom verifiable-constraint continual-task adapter."""

from __future__ import annotations

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.instruction_constraints import ConstraintScorer, ConstraintSpec
from rahc_lora.tasks.base import BaseContinualTask, TaskExample, TaskSplits


def constraint_reference(example: TaskExample) -> tuple[ConstraintSpec, ...]:
    """Deserialize generated constraint specifications without accepting unknown shapes."""

    values = example.fields.get("constraints")
    if not isinstance(values, list):
        raise ValueError("Constraint example must contain a constraint list")
    constraints: list[ConstraintSpec] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"kind", "parameters"}:
            raise ValueError(f"Invalid constraint record: {value!r}")
        parameters = value["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("Constraint parameters must be an object")
        constraints.append(ConstraintSpec(kind=str(value["kind"]), parameters=parameters))
    return tuple(constraints)


class ConstraintTask(BaseContinualTask):
    """Prompt and satisfied-constraint-fraction scoring for the generated corpus."""

    def __init__(
        self, splits: TaskSplits, *, reward_version: str = "constraint-fraction-v1"
    ) -> None:
        super().__init__("constraints", splits)
        self._scorer = ConstraintScorer(reward_version=reward_version)

    def format_prompt(self, example: TaskExample) -> str:
        return str(example.fields["prompt"])

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        if example.task_id != self.task_id:
            raise ValueError(f"ConstraintTask received example for {example.task_id!r}")
        return self._scorer.score(constraint_reference(example), response)
