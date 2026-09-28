"""GSM8K continual-task adapter."""

from __future__ import annotations

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.exact_match import NumericExactMatchScorer
from rahc_lora.tasks.base import BaseContinualTask, TaskExample, TaskSplits


class GSM8KTask(BaseContinualTask):
    """Prompt and deterministic final-answer scoring for pinned GSM8K data."""

    def __init__(self, splits: TaskSplits, *, reward_version: str = "numeric-exact-v1") -> None:
        super().__init__("gsm8k", splits)
        self._scorer = NumericExactMatchScorer(reward_version=reward_version)

    def format_prompt(self, example: TaskExample) -> str:
        question = str(example.fields["question"]).strip()
        return f"Solve the problem. Give your final numeric answer clearly.\n\nProblem: {question}"

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        if example.task_id != self.task_id:
            raise ValueError(f"GSM8KTask received example for {example.task_id!r}")
        return self._scorer.score(str(example.fields["answer"]), response)
