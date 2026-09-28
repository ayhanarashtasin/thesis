from __future__ import annotations

import pytest

from rahc_lora.config.schema import ConstraintGeneratorConfig
from rahc_lora.tasks.base import DatasetRole
from rahc_lora.tasks.constraint_generator import (
    PromptOverlapError,
    all_constraint_examples,
    check_ifeval_prompt_overlap,
    generate_constraint_splits,
)


def _config() -> ConstraintGeneratorConfig:
    return ConstraintGeneratorConfig(
        seed=31,
        split_sizes={
            "train": 6,
            "validation": 4,
            "anchor_candidate": 4,
            "test": 4,
        },
    )


def test_constraint_generation_is_deterministic_and_family_disjoint() -> None:
    first = generate_constraint_splits(_config())
    second = generate_constraint_splits(_config())
    assert first == second
    families: dict[DatasetRole, set[str]] = {}
    for role in (
        DatasetRole.TRAIN,
        DatasetRole.VALIDATION,
        DatasetRole.ANCHOR_CANDIDATE,
        DatasetRole.TEST,
    ):
        families[role] = {
            str(example.metadata["template_family"]) for example in first.for_role(role)
        }
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(families.values())
        for right in list(families.values())[i + 1 :]
    )
    for example in all_constraint_examples(first):
        assert example.metadata["validator_version"] == "1.0.0"
        assert len(str(example.metadata["prompt_hash"])) == 64


def test_constraint_prompts_have_no_exact_or_normalized_ifeval_overlap() -> None:
    splits = generate_constraint_splits(_config())
    examples = all_constraint_examples(splits)
    report = check_ifeval_prompt_overlap(
        examples,
        ("An unrelated held-out official instruction.", "A second distinct prompt."),
    )
    assert report.exact_overlap_count == 0
    assert report.normalized_overlap_count == 0


def test_normalized_ifeval_overlap_is_rejected() -> None:
    examples = all_constraint_examples(generate_constraint_splits(_config()))
    prompt = str(examples[0].fields["prompt"])
    with pytest.raises(PromptOverlapError, match="normalized=1"):
        check_ifeval_prompt_overlap(examples, (prompt.upper(),))
