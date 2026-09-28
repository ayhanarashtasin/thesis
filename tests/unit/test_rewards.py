from __future__ import annotations

import pytest

from rahc_lora.rewards.exact_match import NumericExactMatchScorer, TextExactMatchScorer
from rahc_lora.rewards.instruction_constraints import ConstraintScorer, ConstraintSpec
from rahc_lora.rewards.multiple_choice import (
    MultipleChoiceReference,
    MultipleChoiceScorer,
)
from rahc_lora.rewards.normalization import normalize_for_overlap, normalize_text


def test_numeric_reward_accepts_intended_equivalent_answers() -> None:
    scorer = NumericExactMatchScorer()
    reference = "The calculation gives #### 1,200"
    result = scorer.score(reference, "Reasoning omitted. Final answer: $1,200.00")
    assert result.normalized_reward == 1.0
    assert result.parsed_response == "1200"

    fractional = scorer.score("#### 0.5", "Answer: 1/2")
    assert fractional.normalized_reward == 1.0
    assert fractional.parsed_response == "1/2"


def test_numeric_reward_documents_incorrect_and_parse_failure_as_zero() -> None:
    scorer = NumericExactMatchScorer()
    incorrect = scorer.score("#### 42", "Final answer: 41")
    malformed = scorer.score("#### 42", "I cannot determine it.")
    assert incorrect.normalized_reward == 0.0
    assert incorrect.parse_succeeded
    assert malformed.normalized_reward == 0.0
    assert not malformed.parse_succeeded


def test_text_and_overlap_normalization_are_deterministic() -> None:
    assert normalize_text("  Café\nPLAN  ") == "café plan"
    assert normalize_for_overlap("Plan: A—B!") == "plan a b"
    assert TextExactMatchScorer().score("Alpha beta", " ALPHA\n beta ").normalized_reward == 1.0


def test_multiple_choice_accepts_label_or_exact_text_and_rejects_ambiguity() -> None:
    reference = MultipleChoiceReference(
        correct_label="B",
        labels=("A", "B", "C"),
        texts=("Mercury", "Venus", "Mars"),
    )
    scorer = MultipleChoiceScorer()
    assert scorer.score(reference, "Answer: B").normalized_reward == 1.0
    assert scorer.score(reference, "venus").normalized_reward == 1.0
    ambiguous = scorer.score(reference, "Answer: A, but final answer: B")
    assert ambiguous.normalized_reward == 0.0
    assert not ambiguous.parse_succeeded
    malformed = scorer.score(reference, "No option selected")
    assert malformed.normalized_reward == 0.0
    assert not malformed.parse_succeeded


def test_constraint_fraction_scores_single_and_combined_constraints() -> None:
    scorer = ConstraintScorer()
    constraints = (
        ConstraintSpec("bullet_count", {"count": 2}),
        ConstraintSpec("keyword_count", {"keyword": "verify", "count": 1}),
        ConstraintSpec("forbidden_words", {"words": ["always"]}),
        ConstraintSpec("ending_phrase", {"phrase": "Done."}),
    )
    accepted = scorer.score(constraints, "- verify inputs\n- Record outputs. Done.")
    assert accepted.normalized_reward == 1.0
    partial = scorer.score(constraints, "- verify inputs\n- always skip checks")
    assert partial.normalized_reward == pytest.approx(0.5)
    assert partial.details["satisfied"] == 2


def test_json_and_required_section_constraints() -> None:
    scorer = ConstraintScorer()
    json_result = scorer.score(
        (ConstraintSpec("json_fields", {"required_fields": ["name", "count"]}),),
        '{"name": "sample", "count": 2}',
    )
    assert json_result.normalized_reward == 1.0
    malformed = scorer.score(
        (ConstraintSpec("json_fields", {"required_fields": ["name"]}),),
        "{not json}",
    )
    assert malformed.normalized_reward == 0.0

    sections = scorer.score(
        (ConstraintSpec("required_sections", {"titles": ["Context", "Decision"]}),),
        "Context:\nBackground text.\nDecision:\nProceed carefully.",
    )
    assert sections.normalized_reward == 1.0
    plain_lines = scorer.score(
        (ConstraintSpec("required_sections", {"titles": ["Context", "Decision"]}),),
        "Context\nBackground text.\nDecision\nProceed carefully.",
    )
    assert plain_lines.normalized_reward == 0.0


def test_unknown_constraint_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown constraint"):
        ConstraintScorer().score((ConstraintSpec("unsupported"),), "response")
