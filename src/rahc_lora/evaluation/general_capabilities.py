"""Evaluation-only adapters for the fixed general-capability suite."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.multiple_choice import MultipleChoiceReference, MultipleChoiceScorer
from rahc_lora.tasks.base import DatasetRole, TaskExample
from rahc_lora.tasks.science_task import arc_reference


class EvaluationOnlyAccessError(RuntimeError):
    """Raised when evaluation-only data is requested through a training role."""


class EvaluationOnlyTask:
    """Capability adapter that intentionally exposes no training or anchor loader."""

    def __init__(self, task_id: str, examples: Sequence[TaskExample]) -> None:
        self.task_id = task_id
        self._examples = tuple(examples)
        for example in self._examples:
            if example.task_id != task_id or example.role is not DatasetRole.EVALUATION_ONLY:
                raise ValueError(
                    f"Capability {task_id!r} received non-evaluation example {example.example_id!r}"
                )

    def _deny(self, role: str) -> Sequence[TaskExample]:
        raise EvaluationOnlyAccessError(
            f"Capability dataset {self.task_id!r} is evaluation-only; {role} access is forbidden"
        )

    def load_train(self) -> Sequence[TaskExample]:
        return self._deny("training")

    def load_validation(self) -> Sequence[TaskExample]:
        return self._deny("validation/tuning")

    def load_anchor_candidates(self) -> Sequence[TaskExample]:
        return self._deny("anchor-candidate")

    def load_test(self) -> Sequence[TaskExample]:
        return self._examples

    def score_response(self, example: TaskExample, response: str) -> float:
        return self.evaluate_response(example, response).normalized_reward

    def aggregate_metrics(self, records: Sequence[RewardResult]) -> dict[str, float]:
        if not records:
            raise ValueError(f"Cannot aggregate empty records for capability {self.task_id!r}")
        return {
            "mean_reward": sum(record.normalized_reward for record in records) / len(records),
            "count": float(len(records)),
        }

    def format_prompt(self, example: TaskExample) -> str:
        raise NotImplementedError

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        raise NotImplementedError


class IFEvalOfficialScorer(Protocol):
    """Boundary to the version-matched official IFEval evaluator."""

    version: str

    def score(self, example: TaskExample, response: str) -> tuple[float, float]:
        """Return official strict and loose scores in [0, 1]."""


class IFEvalTask(EvaluationOnlyTask):
    """Official IFEval adapter that refuses any non-official scoring fallback."""

    def __init__(self, examples: Sequence[TaskExample], scorer: IFEvalOfficialScorer) -> None:
        super().__init__("ifeval", examples)
        self._scorer = scorer

    def format_prompt(self, example: TaskExample) -> str:
        return str(example.fields["prompt"])

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        strict, loose = self._scorer.score(example, response)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (strict, loose)):
            raise ValueError(f"Official IFEval scorer returned invalid scores: {(strict, loose)!r}")
        return RewardResult(
            raw_reward=strict,
            normalized_reward=strict,
            parse_succeeded=True,
            parsed_response="official_ifeval",
            reward_version=self._scorer.version,
            details={"strict": strict, "loose": loose},
        )


class MultipleChoiceCapabilityTask(EvaluationOnlyTask):
    """Shared exact-option adapter for MMLU, HellaSwag, and ARC-Easy."""

    def __init__(
        self,
        task_id: str,
        examples: Sequence[TaskExample],
        *,
        reward_version: str = "option-exact-v1",
    ) -> None:
        super().__init__(task_id, examples)
        self._scorer = MultipleChoiceScorer(reward_version=reward_version)

    def _reference(self, example: TaskExample) -> MultipleChoiceReference:
        if self.task_id == "arc_easy":
            return arc_reference(example)
        if self.task_id == "mmlu":
            texts = tuple(str(choice) for choice in example.fields["choices"])
            labels = tuple(chr(ord("A") + index) for index in range(len(texts)))
            answer = example.fields["answer"]
            correct = labels[int(answer)] if isinstance(answer, int) else str(answer)
            return MultipleChoiceReference(correct, labels, texts)
        if self.task_id == "hellaswag":
            texts = tuple(str(choice) for choice in example.fields["endings"])
            labels = tuple(str(index) for index in range(len(texts)))
            return MultipleChoiceReference(str(example.fields["label"]), labels, texts)
        raise ValueError(f"Unsupported multiple-choice capability: {self.task_id!r}")

    def format_prompt(self, example: TaskExample) -> str:
        reference = self._reference(example)
        if self.task_id == "hellaswag":
            stem = str(example.fields.get("ctx", example.fields.get("ctx_a", "")))
        else:
            stem = str(example.fields["question"])
        options = "\n".join(
            f"{label}. {text}"
            for label, text in zip(reference.labels, reference.texts, strict=True)
        )
        return f"Select the best completion and answer with its label.\n\n{stem}\n{options}"

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        return self._scorer.score(self._reference(example), response)


@dataclass(frozen=True)
class TokenNll:
    """Summed negative log-likelihood and contributing token count."""

    total_nll: float
    token_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_nll) or self.total_nll < 0:
            raise ValueError("Token negative log-likelihood must be finite and nonnegative")
        if self.token_count <= 0:
            raise ValueError("Token count must be positive")


class TokenNllPolicy(Protocol):
    """Read-only policy interface for held-out perplexity evaluation."""

    def token_nll(
        self, texts: Sequence[str], *, max_tokens: int | None = None
    ) -> Sequence[TokenNll]:
        """Return summed token NLL for each input text without updating state."""


class PerplexityCapability:
    """Evaluation-only held-out corpus perplexity adapter."""

    task_id = "perplexity"

    def __init__(self, examples: Sequence[TaskExample]) -> None:
        self._examples = tuple(examples)
        for example in self._examples:
            if example.task_id != self.task_id or example.role is not DatasetRole.EVALUATION_ONLY:
                raise ValueError("Perplexity corpus must be marked evaluation-only")

    def load_train(self) -> Sequence[TaskExample]:
        raise EvaluationOnlyAccessError("Perplexity corpus is evaluation-only")

    def load_anchor_candidates(self) -> Sequence[TaskExample]:
        raise EvaluationOnlyAccessError("Perplexity corpus cannot supply anchors")

    @property
    def example_count(self) -> int:
        return len(self._examples)

    def evaluate(self, policy: TokenNllPolicy, *, max_tokens: int | None = None) -> float:
        texts = [str(example.fields["text"]) for example in self._examples]
        records = tuple(policy.token_nll(texts, max_tokens=max_tokens))
        if len(records) != len(texts):
            raise ValueError("Perplexity policy returned the wrong number of records")
        total_nll = sum(record.total_nll for record in records)
        token_count = sum(record.token_count for record in records)
        try:
            perplexity = math.exp(total_nll / token_count)
        except OverflowError as error:
            raise ValueError("Held-out perplexity is non-finite") from error
        if not math.isfinite(perplexity):
            raise ValueError("Held-out perplexity is non-finite")
        return perplexity


@dataclass(frozen=True)
class CapabilityComparison:
    """Absolute capability score and change from the frozen starting policy."""

    task_id: str
    absolute_score: float
    frozen_baseline_score: float

    @property
    def change_from_frozen(self) -> float:
        return self.absolute_score - self.frozen_baseline_score
