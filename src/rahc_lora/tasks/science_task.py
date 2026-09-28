"""ARC-Challenge continual-task adapter."""

from __future__ import annotations

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.multiple_choice import MultipleChoiceReference, MultipleChoiceScorer
from rahc_lora.tasks.base import BaseContinualTask, TaskExample, TaskSplits


def arc_reference(example: TaskExample) -> MultipleChoiceReference:
    """Convert an ARC record into the project multiple-choice contract."""

    choices = example.fields["choices"]
    if not isinstance(choices, dict):
        raise ValueError("ARC choices must be an object with label and text arrays")
    labels = tuple(str(value) for value in choices["label"])
    texts = tuple(str(value) for value in choices["text"])
    return MultipleChoiceReference(
        correct_label=str(example.fields["answerKey"]),
        labels=labels,
        texts=texts,
    )


class ARCChallengeTask(BaseContinualTask):
    """Prompt and exact option scoring for pinned ARC-Challenge data."""

    def __init__(self, splits: TaskSplits, *, reward_version: str = "option-exact-v1") -> None:
        super().__init__("arc_challenge", splits)
        self._scorer = MultipleChoiceScorer(reward_version=reward_version)

    def format_prompt(self, example: TaskExample) -> str:
        reference = arc_reference(example)
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(reference.labels, reference.texts, strict=True)
        )
        return (
            "Choose the correct option and end with its label.\n\n"
            f"Question: {str(example.fields['question']).strip()}\n{choices}"
        )

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        if example.task_id != self.task_id:
            raise ValueError(f"ARCChallengeTask received example for {example.task_id!r}")
        return self._scorer.score(arc_reference(example), response)
