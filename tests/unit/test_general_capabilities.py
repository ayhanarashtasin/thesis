from __future__ import annotations

from collections.abc import Sequence

import pytest

from rahc_lora.evaluation.capability_suite import (
    CAPABILITY_METRICS,
    GeneralCapabilitySuite,
    _select_mmlu_examples,
)
from rahc_lora.evaluation.evaluator import EvaluationGenerationConfig
from rahc_lora.evaluation.general_capabilities import (
    EvaluationOnlyAccessError,
    IFEvalTask,
    MultipleChoiceCapabilityTask,
    PerplexityCapability,
    TokenNll,
)
from rahc_lora.tasks.base import DatasetRole, TaskExample


class FakeIFEvalScorer:
    version = "official-test-v1"

    def score(self, example: TaskExample, response: str) -> tuple[float, float]:
        return (1.0, 1.0) if response == "valid" else (0.0, 0.5)


class FakeNllPolicy:
    def token_nll(self, texts: Sequence[str], *, max_tokens: int | None = None) -> list[TokenNll]:
        del max_tokens
        return [TokenNll(total_nll=2.0, token_count=2) for _ in texts]


class FakeCapabilityPolicy:
    def __init__(self, *, correct: bool, nll_per_token: float) -> None:
        self.correct = correct
        self.nll_per_token = nll_per_token

    def generate(
        self, prompts: Sequence[str], generation: EvaluationGenerationConfig
    ) -> Sequence[str]:
        del generation
        correct_responses = {
            "IFEVAL": "valid",
            "MMLU": "B",
            "HELLA": "1",
            "ARC": "B",
        }
        return tuple(
            next(response for marker, response in correct_responses.items() if marker in prompt)
            if self.correct
            else "wrong"
            for prompt in prompts
        )

    def token_nll(
        self, texts: Sequence[str], *, max_tokens: int | None = None
    ) -> Sequence[TokenNll]:
        del max_tokens
        return tuple(TokenNll(total_nll=2 * self.nll_per_token, token_count=2) for _ in texts)


def _example(task_id: str, fields: dict[str, object]) -> TaskExample:
    return TaskExample(task_id, f"{task_id}:1", DatasetRole.EVALUATION_ONLY, fields)


def _suite() -> GeneralCapabilitySuite:
    return GeneralCapabilitySuite(
        ifeval=IFEvalTask((_example("ifeval", {"prompt": "IFEVAL"}),), FakeIFEvalScorer()),
        mmlu=MultipleChoiceCapabilityTask(
            "mmlu",
            (
                _example(
                    "mmlu",
                    {"question": "MMLU", "choices": ["a", "b"], "answer": 1},
                ),
            ),
        ),
        hellaswag=MultipleChoiceCapabilityTask(
            "hellaswag",
            (
                _example(
                    "hellaswag",
                    {"ctx": "HELLA", "endings": ["a", "b"], "label": "1"},
                ),
            ),
        ),
        arc_easy=MultipleChoiceCapabilityTask(
            "arc_easy",
            (
                _example(
                    "arc_easy",
                    {
                        "question": "ARC",
                        "choices": {"label": ["A", "B"], "text": ["a", "b"]},
                        "answerKey": "B",
                    },
                ),
            ),
        ),
        perplexity=PerplexityCapability((_example("perplexity", {"text": "held out text"}),)),
        generation=EvaluationGenerationConfig(max_new_tokens=2),
        max_perplexity_tokens=16,
    )


def test_ifeval_is_evaluation_only_and_uses_official_scorer_boundary() -> None:
    task = IFEvalTask((_example("ifeval", {"prompt": "held out"}),), FakeIFEvalScorer())
    with pytest.raises(EvaluationOnlyAccessError, match="training"):
        task.load_train()
    with pytest.raises(EvaluationOnlyAccessError, match="anchor"):
        task.load_anchor_candidates()
    result = task.evaluate_response(task.load_test()[0], "valid")
    assert result.normalized_reward == 1.0
    assert result.details["loose"] == 1.0


def test_mmlu_and_perplexity_capability_adapters() -> None:
    mmlu_example = _example(
        "mmlu",
        {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1},
    )
    mmlu = MultipleChoiceCapabilityTask("mmlu", (mmlu_example,))
    assert mmlu.evaluate_response(mmlu_example, "B").normalized_reward == 1.0

    perplexity_example = _example("perplexity", {"text": "held out text"})
    capability = PerplexityCapability((perplexity_example,))
    assert capability.evaluate(FakeNllPolicy()) == pytest.approx(2.718281828459045)
    with pytest.raises(EvaluationOnlyAccessError):
        capability.load_anchor_candidates()


def test_capability_suite_compares_every_boundary_with_frozen_policy() -> None:
    suite = _suite()
    baseline = suite.capture_frozen_baseline(FakeCapabilityPolicy(correct=True, nll_per_token=1.0))
    boundary = suite.evaluate_boundary(
        "gsm8k", FakeCapabilityPolicy(correct=False, nll_per_token=2.0)
    )

    assert tuple(f"{result.capability}.{result.metric}" for result in baseline) == (
        CAPABILITY_METRICS
    )
    assert all(result.change_from_frozen == 0 for result in baseline)
    changes = {
        f"{result.capability}.{result.metric}": result.change_from_frozen for result in boundary
    }
    assert changes["ifeval.strict"] == pytest.approx(-1.0)
    assert changes["ifeval.loose"] == pytest.approx(-0.5)
    assert changes["mmlu.accuracy"] == pytest.approx(-1.0)
    assert changes["hellaswag.accuracy"] == pytest.approx(-1.0)
    assert changes["arc_easy.accuracy"] == pytest.approx(-1.0)
    assert changes["perplexity.perplexity"] == pytest.approx(7.38905609893065 - 2.718281828459045)
    assert boundary[-1].higher_is_better is False

    restored = _suite()
    restored.load_state_dict(suite.state_dict())
    assert restored.history_results() == suite.history_results()
    assert restored.latest_results() == suite.latest_results()


def test_capability_checkpoint_rejects_inconsistent_history_without_partial_load() -> None:
    source = _suite()
    source.capture_frozen_baseline(FakeCapabilityPolicy(correct=True, nll_per_token=1.0))
    source.evaluate_boundary("gsm8k", FakeCapabilityPolicy(correct=False, nll_per_token=2.0))
    state = source.state_dict()
    state["history"][-1]["frozen_baseline_score"] += 1.0

    target = _suite()
    with pytest.raises(ValueError, match="history values are inconsistent"):
        target.load_state_dict(state)
    assert target.has_frozen_baseline is False


def test_perplexity_overflow_fails_as_nonfinite_metric() -> None:
    capability = PerplexityCapability((_example("perplexity", {"text": "held out text"}),))
    with pytest.raises(ValueError, match="non-finite"):
        capability.evaluate(FakeCapabilityPolicy(correct=True, nll_per_token=1_000.0))


def test_capped_mmlu_selection_round_robins_across_fixed_subjects() -> None:
    examples = tuple(
        _example("mmlu", {"subject": subject, "question": question})
        for subject, question in (
            ("subject_a", "a1"),
            ("subject_a", "a2"),
            ("subject_b", "b1"),
            ("subject_b", "b2"),
        )
    )
    selected = _select_mmlu_examples(examples, ("subject_a", "subject_b"), 2)
    assert [example.fields["question"] for example in selected] == ["a1", "b1"]
