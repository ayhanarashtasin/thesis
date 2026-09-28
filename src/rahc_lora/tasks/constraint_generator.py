"""Deterministic, versioned custom constraint-dataset generation and overlap checks."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from rahc_lora.config.schema import ConstraintGeneratorConfig
from rahc_lora.rewards.instruction_constraints import ConstraintSpec
from rahc_lora.rewards.normalization import normalize_for_overlap, normalize_text
from rahc_lora.tasks.base import DatasetRole, TaskExample, TaskSplits


class PromptOverlapError(ValueError):
    """Raised when custom prompts overlap the held-out official IFEval corpus."""


@dataclass(frozen=True)
class PromptOverlapReport:
    """Exact and normalized cross-corpus overlap counts."""

    custom_prompt_count: int
    official_prompt_count: int
    exact_overlap_count: int
    normalized_overlap_count: int


TemplateBuilder = Callable[[int, random.Random], tuple[str, tuple[ConstraintSpec, ...]]]


def _topic_keyword_bullets(
    index: int, rng: random.Random
) -> tuple[str, tuple[ConstraintSpec, ...]]:
    topic = rng.choice(("gardens", "rivers", "libraries", "astronomy"))
    keyword = rng.choice(("evidence", "balance", "method", "signal"))
    count = 2 + index % 3
    prompt = (
        f"Write a compact field note about {topic}. Use exactly {count} dash-led bullet lines, "
        f"and include the standalone word '{keyword}' exactly once."
    )
    return prompt, (
        ConstraintSpec("bullet_count", {"count": count}),
        ConstraintSpec("keyword_count", {"keyword": keyword, "count": 1}),
    )


def _brief_forbidden_ending(
    index: int, rng: random.Random
) -> tuple[str, tuple[ConstraintSpec, ...]]:
    forbidden = rng.choice(("obvious", "simply", "always", "never"))
    ending = rng.choice(("Record complete.", "Review finished.", "Entry closed."))
    maximum = 28 + index % 5
    prompt = (
        f"Draft a short maintenance note of 12 to {maximum} words. Do not use the word "
        f"'{forbidden}'. End with the exact phrase '{ending}'"
    )
    return prompt, (
        ConstraintSpec("word_count", {"minimum": 12, "maximum": maximum}),
        ConstraintSpec("forbidden_words", {"words": [forbidden]}),
        ConstraintSpec("ending_phrase", {"phrase": ending}),
    )


def _summary_sections(index: int, rng: random.Random) -> tuple[str, tuple[ConstraintSpec, ...]]:
    subject = rng.choice(("rainwater", "urban shade", "seed storage", "night transit"))
    titles = ["Observation", "Action"] if index % 2 == 0 else ["Context", "Decision"]
    prompt = (
        f"Prepare a two-part memo about {subject}. Include titled sections named "
        f"'{titles[0]}' and '{titles[1]}', each written as a heading on its own line."
    )
    return prompt, (ConstraintSpec("required_sections", {"titles": titles}),)


def _json_profile(index: int, rng: random.Random) -> tuple[str, tuple[ConstraintSpec, ...]]:
    extra = rng.choice(("priority", "owner", "status"))
    fields = ["name", "summary", extra]
    prompt = (
        "Return only one valid JSON object describing a fictional project. "
        f"The object must contain the fields {', '.join(fields)}."
    )
    return prompt, (ConstraintSpec("json_fields", {"required_fields": fields}),)


def _steps_keyword(index: int, rng: random.Random) -> tuple[str, tuple[ConstraintSpec, ...]]:
    keyword = rng.choice(("verify", "measure", "compare", "document"))
    count = 3 + index % 2
    prompt = (
        f"Describe a small quality check in exactly {count} dash-led bullets. Include the word "
        f"'{keyword}' exactly twice across the response."
    )
    return prompt, (
        ConstraintSpec("bullet_count", {"count": count}),
        ConstraintSpec("keyword_count", {"keyword": keyword, "count": 2}),
    )


def _note_word_range(index: int, rng: random.Random) -> tuple[str, tuple[ConstraintSpec, ...]]:
    topic = rng.choice(("tool lending", "trail signs", "public clocks", "water meters"))
    forbidden = rng.choice(("excellent", "perfect", "easy"))
    minimum = 18 + index % 3
    prompt = (
        f"Write a neutral note about {topic} using {minimum} to 36 words. "
        f"The word '{forbidden}' must not appear."
    )
    return prompt, (
        ConstraintSpec("word_count", {"minimum": minimum, "maximum": 36}),
        ConstraintSpec("forbidden_words", {"words": [forbidden]}),
    )


def _recommendation_bullets(
    index: int, rng: random.Random
) -> tuple[str, tuple[ConstraintSpec, ...]]:
    subject = rng.choice(("a reading room", "a bike rack", "a notice board", "a bus shelter"))
    count = 2 + index % 3
    ending = "Plan noted."
    prompt = (
        f"Give recommendations for {subject} in exactly {count} dash-led bullets. "
        f"The final bullet must end with the exact phrase '{ending}'"
    )
    return prompt, (
        ConstraintSpec("bullet_count", {"count": count}),
        ConstraintSpec("ending_phrase", {"phrase": ending}),
    )


def _json_inventory(index: int, rng: random.Random) -> tuple[str, tuple[ConstraintSpec, ...]]:
    category = rng.choice(("tools", "books", "plants", "maps"))
    fields = ["category", "items", "count"]
    prompt = (
        f"Return only a valid JSON object representing a fictional {category} inventory. "
        f"Include all of these fields: {', '.join(fields)}."
    )
    return prompt, (ConstraintSpec("json_fields", {"required_fields": fields}),)


_FAMILIES_BY_ROLE: dict[DatasetRole, tuple[tuple[str, TemplateBuilder], ...]] = {
    DatasetRole.TRAIN: (
        ("topic_keyword_bullets", _topic_keyword_bullets),
        ("brief_forbidden_ending", _brief_forbidden_ending),
    ),
    DatasetRole.VALIDATION: (
        ("summary_sections", _summary_sections),
        ("json_profile", _json_profile),
    ),
    DatasetRole.ANCHOR_CANDIDATE: (
        ("steps_keyword", _steps_keyword),
        ("note_word_range", _note_word_range),
    ),
    DatasetRole.TEST: (
        ("recommendation_bullets", _recommendation_bullets),
        ("json_inventory", _json_inventory),
    ),
}


def _serialize_constraints(constraints: tuple[ConstraintSpec, ...]) -> list[dict[str, object]]:
    return [asdict(constraint) for constraint in constraints]


def generate_constraint_splits(config: ConstraintGeneratorConfig) -> TaskSplits:
    """Generate all four roles with template-family separation and deterministic metadata."""

    generated: dict[DatasetRole, tuple[TaskExample, ...]] = {}
    role_order = (
        DatasetRole.TRAIN,
        DatasetRole.VALIDATION,
        DatasetRole.ANCHOR_CANDIDATE,
        DatasetRole.TEST,
    )
    for role_index, role in enumerate(role_order):
        size = config.split_sizes[role.value]
        rng = random.Random(config.seed + role_index * 1_000_003)
        families = _FAMILIES_BY_ROLE[role]
        examples: list[TaskExample] = []
        for index in range(size):
            family_name, builder = families[index % len(families)]
            prompt, constraints = builder(index, rng)
            prompt = f"Scenario {role.value}-{index + 1}: {prompt}"
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            examples.append(
                TaskExample(
                    task_id="constraints",
                    example_id=f"constraints:{prompt_hash}",
                    role=role,
                    fields={
                        "prompt": prompt,
                        "constraints": _serialize_constraints(constraints),
                    },
                    metadata={
                        "dataset_version": config.dataset_version,
                        "generator_version": config.grammar_version,
                        "grammar_version": config.grammar_version,
                        "template_version": config.template_version,
                        "template_family": family_name,
                        "constraint_types": [constraint.kind for constraint in constraints],
                        "random_seed": config.seed,
                        "validator_version": config.validator_version,
                        "prompt_hash": prompt_hash,
                        "split": role.value,
                    },
                )
            )
        generated[role] = tuple(examples)

    splits = TaskSplits(
        train=generated[DatasetRole.TRAIN],
        validation=generated[DatasetRole.VALIDATION],
        anchor_candidates=generated[DatasetRole.ANCHOR_CANDIDATE],
        test=generated[DatasetRole.TEST],
    )
    family_roles: dict[str, DatasetRole] = {}
    for role in role_order:
        for example in splits.for_role(role):
            family = str(example.metadata["template_family"])
            previous = family_roles.setdefault(family, role)
            if previous is not role:
                raise AssertionError(
                    f"Template family {family!r} crosses {previous.value!r} and {role.value!r}"
                )
    return splits


def check_ifeval_prompt_overlap(
    custom_examples: tuple[TaskExample, ...], official_prompts: tuple[str, ...]
) -> PromptOverlapReport:
    """Reject exact or normalized-text overlap with official IFEval prompts."""

    custom_prompts = tuple(str(example.fields["prompt"]) for example in custom_examples)
    exact_custom = set(custom_prompts)
    exact_official = set(official_prompts)
    normalized_custom = {normalize_for_overlap(prompt) for prompt in custom_prompts}
    normalized_official = {normalize_for_overlap(prompt) for prompt in official_prompts}
    exact_count = len(exact_custom.intersection(exact_official))
    normalized_count = len(normalized_custom.intersection(normalized_official))
    report = PromptOverlapReport(
        custom_prompt_count=len(custom_prompts),
        official_prompt_count=len(official_prompts),
        exact_overlap_count=exact_count,
        normalized_overlap_count=normalized_count,
    )
    if exact_count or normalized_count:
        raise PromptOverlapError(
            "Custom constraint prompts overlap official IFEval: "
            f"exact={exact_count}, normalized={normalized_count}"
        )
    return report


def all_constraint_examples(splits: TaskSplits) -> tuple[TaskExample, ...]:
    """Return all generated examples for manifesting and overlap checks only."""

    return splits.train + splits.validation + splits.anchor_candidates + splits.test


def normalized_prompt_set(examples: tuple[TaskExample, ...]) -> set[str]:
    """Expose deterministic normalized prompt identities for tests and diagnostics."""

    return {normalize_text(str(example.fields["prompt"])) for example in examples}
