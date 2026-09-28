"""MBPP continual-task adapter using an explicit isolated test runner."""

from __future__ import annotations

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.code_tests import CodeTestReference, CodeTestsScorer
from rahc_lora.tasks.base import BaseContinualTask, TaskExample, TaskSplits


class MBPPTask(BaseContinualTask):
    """Prompt and hidden-test-fraction scoring for pinned sanitized MBPP data."""

    def __init__(
        self,
        splits: TaskSplits,
        scorer: CodeTestsScorer,
    ) -> None:
        super().__init__("mbpp", splits)
        self._scorer = scorer

    def format_prompt(self, example: TaskExample) -> str:
        problem = str(example.fields.get("prompt", example.fields.get("text", ""))).strip()
        if not problem:
            raise ValueError(f"MBPP example {example.example_id!r} has no prompt text")
        return (
            "Write a complete Python solution. Return only Python code; do not include tests.\n\n"
            f"Problem: {problem}"
        )

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        if example.task_id != self.task_id:
            raise ValueError(f"MBPPTask received example for {example.task_id!r}")
        test_values = example.fields.get("challenge_test_list") or example.fields.get("test_list")
        if not isinstance(test_values, (list, tuple)):
            raise ValueError(f"MBPP example {example.example_id!r} has no hidden test list")
        return self._scorer.score(
            CodeTestReference(tuple(str(test) for test in test_values)), response
        )
