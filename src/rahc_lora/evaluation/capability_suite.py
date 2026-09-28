"""Fixed general-capability evaluation with frozen-policy comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

from rahc_lora.config.schema import GeneralCapabilityConfig, TaskDataConfig
from rahc_lora.evaluation.evaluator import (
    EvaluationGenerationConfig,
    EvaluationPolicy,
    evaluate_task,
)
from rahc_lora.evaluation.general_capabilities import (
    IFEvalOfficialScorer,
    IFEvalTask,
    MultipleChoiceCapabilityTask,
    PerplexityCapability,
    TokenNllPolicy,
)
from rahc_lora.tasks.base import TaskExample
from rahc_lora.tasks.data import DatasetProvider, load_evaluation_only_examples

CAPABILITY_STATE_SCHEMA_VERSION = 1
CAPABILITY_METRICS = (
    "ifeval.strict",
    "ifeval.loose",
    "mmlu.accuracy",
    "hellaswag.accuracy",
    "arc_easy.accuracy",
    "perplexity.perplexity",
)


class CapabilityPolicy(EvaluationPolicy, TokenNllPolicy, Protocol):
    """Read-only policy surface required by the fixed capability suite."""


@dataclass(frozen=True)
class CapabilityMetricResult:
    """One absolute score paired with the frozen starting-policy baseline."""

    after_task: str
    capability: str
    metric: str
    absolute_score: float
    frozen_baseline_score: float
    change_from_frozen: float
    higher_is_better: bool
    example_count: int

    def __post_init__(self) -> None:
        if not self.after_task or not self.capability or not self.metric:
            raise ValueError("Capability result identities must be nonempty")
        values = (
            self.absolute_score,
            self.frozen_baseline_score,
            self.change_from_frozen,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Capability metric values must be finite")
        if type(self.higher_is_better) is not bool:
            raise ValueError("Capability metric direction must be Boolean")
        if type(self.example_count) is not int or self.example_count <= 0:
            raise ValueError("Capability metric example count must be positive")


class GeneralCapabilitySuite:
    """Evaluate five held-out capabilities and checkpoint the frozen baseline."""

    def __init__(
        self,
        *,
        ifeval: IFEvalTask,
        mmlu: MultipleChoiceCapabilityTask,
        hellaswag: MultipleChoiceCapabilityTask,
        arc_easy: MultipleChoiceCapabilityTask,
        perplexity: PerplexityCapability,
        generation: EvaluationGenerationConfig,
        max_perplexity_tokens: int,
    ) -> None:
        self._tasks = {
            "ifeval": ifeval,
            "mmlu": mmlu,
            "hellaswag": hellaswag,
            "arc_easy": arc_easy,
        }
        if tuple(self._tasks) != ("ifeval", "mmlu", "hellaswag", "arc_easy"):
            raise AssertionError("Capability task order is not canonical")
        self._perplexity = perplexity
        self._generation = generation
        self._max_perplexity_tokens = max_perplexity_tokens
        if max_perplexity_tokens <= 0:
            raise ValueError("Capability perplexity token limit must be positive")
        self._frozen_scores: dict[str, float] | None = None
        self._latest_scores: dict[str, float] | None = None
        self._latest_counts: dict[str, int] | None = None
        self._last_after_task: str | None = None
        self._history: list[CapabilityMetricResult] = []

    @property
    def has_frozen_baseline(self) -> bool:
        return self._frozen_scores is not None

    def _score(self, policy: CapabilityPolicy) -> tuple[dict[str, float], dict[str, int]]:
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for capability, task in self._tasks.items():
            result = evaluate_task(task, policy, self._generation)
            count = len(result.records)
            if capability == "ifeval":
                strict_values = [
                    float(record.reward.details["strict"]) for record in result.records
                ]
                loose_values = [float(record.reward.details["loose"]) for record in result.records]
                scores["ifeval.strict"] = sum(strict_values) / count
                scores["ifeval.loose"] = sum(loose_values) / count
                counts["ifeval.strict"] = count
                counts["ifeval.loose"] = count
            else:
                key = f"{capability}.accuracy"
                scores[key] = result.mean_reward
                counts[key] = count
        scores["perplexity.perplexity"] = self._perplexity.evaluate(
            policy, max_tokens=self._max_perplexity_tokens
        )
        counts["perplexity.perplexity"] = self._perplexity.example_count
        if tuple(scores) != CAPABILITY_METRICS or tuple(counts) != CAPABILITY_METRICS:
            raise AssertionError("Capability suite produced a noncanonical metric set")
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("Capability suite produced a non-finite metric")
        return scores, counts

    @staticmethod
    def _results(
        *,
        after_task: str,
        scores: dict[str, float],
        baselines: dict[str, float],
        counts: dict[str, int],
    ) -> tuple[CapabilityMetricResult, ...]:
        results: list[CapabilityMetricResult] = []
        for key in CAPABILITY_METRICS:
            capability, metric = key.split(".", maxsplit=1)
            current = scores[key]
            baseline = baselines[key]
            results.append(
                CapabilityMetricResult(
                    after_task=after_task,
                    capability=capability,
                    metric=metric,
                    absolute_score=current,
                    frozen_baseline_score=baseline,
                    change_from_frozen=current - baseline,
                    higher_is_better=capability != "perplexity",
                    example_count=counts[key],
                )
            )
        return tuple(results)

    def capture_frozen_baseline(
        self, policy: CapabilityPolicy
    ) -> tuple[CapabilityMetricResult, ...]:
        """Evaluate and freeze the starting policy exactly once before training."""

        if self._frozen_scores is not None:
            raise RuntimeError("Frozen capability baseline has already been captured")
        scores, counts = self._score(policy)
        self._frozen_scores = dict(scores)
        self._latest_scores = dict(scores)
        self._latest_counts = dict(counts)
        self._last_after_task = "frozen_start"
        results = self._results(
            after_task="frozen_start",
            scores=scores,
            baselines=scores,
            counts=counts,
        )
        self._history.extend(results)
        return results

    def evaluate_boundary(
        self,
        after_task: str,
        policy: CapabilityPolicy,
    ) -> tuple[CapabilityMetricResult, ...]:
        """Evaluate after one task and compare with the immutable frozen baseline."""

        if self._frozen_scores is None:
            raise RuntimeError("Capability evaluation requires a frozen starting baseline")
        scores, counts = self._score(policy)
        self._latest_scores = dict(scores)
        self._latest_counts = dict(counts)
        self._last_after_task = after_task
        results = self._results(
            after_task=after_task,
            scores=scores,
            baselines=self._frozen_scores,
            counts=counts,
        )
        self._history.extend(results)
        return results

    def history_results(self) -> tuple[CapabilityMetricResult, ...]:
        return tuple(self._history)

    def latest_results(self) -> tuple[CapabilityMetricResult, ...] | None:
        if (
            self._frozen_scores is None
            or self._latest_scores is None
            or self._latest_counts is None
            or self._last_after_task is None
        ):
            return None
        return self._results(
            after_task=self._last_after_task,
            scores=self._latest_scores,
            baselines=self._frozen_scores,
            counts=self._latest_counts,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_STATE_SCHEMA_VERSION,
            "frozen_scores": self._frozen_scores,
            "latest_scores": self._latest_scores,
            "latest_counts": self._latest_counts,
            "last_after_task": self._last_after_task,
            "history": [asdict(result) for result in self._history],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected_fields = {
            "schema_version",
            "frozen_scores",
            "latest_scores",
            "latest_counts",
            "last_after_task",
            "history",
        }
        if set(state) != expected_fields:
            raise ValueError("Capability checkpoint state fields are invalid")
        if state.get("schema_version") != CAPABILITY_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported general-capability state schema")
        frozen = state.get("frozen_scores")
        latest = state.get("latest_scores")
        counts = state.get("latest_counts")
        last_after_task = state.get("last_after_task")
        history_value = state.get("history")
        if not all(isinstance(value, dict) for value in (frozen, latest, counts)):
            raise ValueError("Capability checkpoint is missing evaluated metric dictionaries")
        frozen_dict = cast(dict[str, float], frozen)
        latest_dict = cast(dict[str, float], latest)
        counts_dict = cast(dict[str, int], counts)
        expected_metrics = set(CAPABILITY_METRICS)
        if not all(
            set(values) == expected_metrics for values in (frozen_dict, latest_dict, counts_dict)
        ):
            raise ValueError("Capability checkpoint metric membership is invalid")
        if not isinstance(last_after_task, str) or not last_after_task:
            raise ValueError("Capability checkpoint last boundary is invalid")
        if not isinstance(history_value, list) or not history_value:
            raise ValueError("Capability checkpoint history is missing")
        if not all(
            math.isfinite(float(value)) for value in (*frozen_dict.values(), *latest_dict.values())
        ):
            raise ValueError("Capability checkpoint contains non-finite scores")
        if not all(type(value) is int and value > 0 for value in counts_dict.values()):
            raise ValueError("Capability checkpoint contains invalid example counts")
        frozen_scores = {key: float(frozen_dict[key]) for key in CAPABILITY_METRICS}
        latest_scores = {key: float(latest_dict[key]) for key in CAPABILITY_METRICS}
        latest_counts = {key: counts_dict[key] for key in CAPABILITY_METRICS}
        try:
            history = [CapabilityMetricResult(**value) for value in history_value]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Capability checkpoint history is invalid: {error}") from error
        if len(history) % len(CAPABILITY_METRICS) or history[-1].after_task != last_after_task:
            raise ValueError("Capability checkpoint history boundary structure is invalid")
        boundary_names: list[str] = []
        baseline_counts: dict[str, int] = {}
        for offset in range(0, len(history), len(CAPABILITY_METRICS)):
            group = history[offset : offset + len(CAPABILITY_METRICS)]
            keys = tuple(f"{result.capability}.{result.metric}" for result in group)
            if keys != CAPABILITY_METRICS or len({result.after_task for result in group}) != 1:
                raise ValueError("Capability checkpoint history metric structure is invalid")
            boundary_name = group[0].after_task
            if boundary_name in boundary_names:
                raise ValueError("Capability checkpoint repeats a boundary")
            boundary_names.append(boundary_name)
            for key, result in zip(CAPABILITY_METRICS, group, strict=True):
                if (
                    result.frozen_baseline_score != frozen_scores[key]
                    or result.change_from_frozen
                    != result.absolute_score - result.frozen_baseline_score
                    or result.higher_is_better != (result.capability != "perplexity")
                ):
                    raise ValueError("Capability checkpoint history values are inconsistent")
                if offset == 0:
                    baseline_counts[key] = result.example_count
                    if result.absolute_score != frozen_scores[key]:
                        raise ValueError("Capability checkpoint frozen row is inconsistent")
                elif result.example_count != baseline_counts[key]:
                    raise ValueError(
                        "Capability checkpoint example counts changed across boundaries"
                    )
        if history[0].after_task != "frozen_start":
            raise ValueError("Capability checkpoint history lacks the frozen baseline")
        latest_group = history[-len(CAPABILITY_METRICS) :]
        for key, result in zip(CAPABILITY_METRICS, latest_group, strict=True):
            if (
                result.absolute_score != latest_scores[key]
                or result.example_count != latest_counts[key]
            ):
                raise ValueError("Capability checkpoint latest metrics are inconsistent")
        self._frozen_scores = frozen_scores
        self._latest_scores = latest_scores
        self._latest_counts = latest_counts
        self._last_after_task = last_after_task
        self._history = history


def _limited(examples: Sequence[TaskExample], maximum: int) -> tuple[TaskExample, ...]:
    selected = tuple(examples[:maximum]) if maximum else tuple(examples)
    if not selected:
        raise ValueError("Capability dataset selection must not be empty")
    return selected


def _load_examples(
    config: TaskDataConfig,
    provider: DatasetProvider,
    maximum: int,
) -> tuple[TaskExample, ...]:
    return _limited(load_evaluation_only_examples(config, provider), maximum)


def _select_mmlu_examples(
    examples: Sequence[TaskExample],
    subjects: Sequence[str],
    maximum: int,
) -> tuple[TaskExample, ...]:
    subject_examples: dict[str, list[TaskExample]] = {subject: [] for subject in subjects}
    for example in examples:
        subject = str(example.fields.get("subject"))
        if subject in subject_examples:
            subject_examples[subject].append(example)
    missing_subjects = [subject for subject in subjects if not subject_examples[subject]]
    if missing_subjects:
        raise ValueError(f"Pinned MMLU source is missing subjects: {missing_subjects}")
    if maximum == 0:
        return tuple(
            example
            for example in examples
            if str(example.fields.get("subject")) in subject_examples
        )
    selected: list[TaskExample] = []
    indices = {subject: 0 for subject in subjects}
    while len(selected) < maximum:
        added = False
        for subject in subjects:
            index = indices[subject]
            records = subject_examples[subject]
            if index < len(records):
                selected.append(records[index])
                indices[subject] += 1
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
    return _limited(selected, 0)


def build_general_capability_suite(
    config: GeneralCapabilityConfig,
    provider: DatasetProvider,
    scorer: IFEvalOfficialScorer,
) -> GeneralCapabilitySuite:
    """Build the fixed suite from held-out sources and one official scorer."""

    if scorer.version != config.ifeval_scorer_revision:
        raise ValueError("Official IFEval scorer revision does not match capability configuration")
    maximum = config.max_examples_per_capability
    ifeval_examples = _load_examples(config.ifeval, provider, maximum)
    mmlu_all = load_evaluation_only_examples(config.mmlu, provider)
    mmlu_examples = _select_mmlu_examples(mmlu_all, config.mmlu_subjects, maximum)
    hellaswag_examples = _load_examples(config.hellaswag, provider, maximum)
    arc_easy_examples = _load_examples(config.arc_easy, provider, maximum)
    perplexity_all = load_evaluation_only_examples(config.perplexity, provider)
    perplexity_examples = _limited(
        tuple(example for example in perplexity_all if str(example.fields.get("text", "")).strip()),
        maximum,
    )
    return GeneralCapabilitySuite(
        ifeval=IFEvalTask(ifeval_examples, scorer),
        mmlu=MultipleChoiceCapabilityTask(
            "mmlu", mmlu_examples, reward_version=config.mmlu.reward_version
        ),
        hellaswag=MultipleChoiceCapabilityTask(
            "hellaswag", hellaswag_examples, reward_version=config.hellaswag.reward_version
        ),
        arc_easy=MultipleChoiceCapabilityTask(
            "arc_easy", arc_easy_examples, reward_version=config.arc_easy.reward_version
        ),
        perplexity=PerplexityCapability(perplexity_examples),
        generation=EvaluationGenerationConfig(
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        max_perplexity_tokens=config.max_perplexity_tokens,
    )
