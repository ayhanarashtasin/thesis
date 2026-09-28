"""Deterministic parsing and exact scoring for labelled choices."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.normalization import normalize_text

_EXPLICIT_CHOICE_RE = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|=|:)?\s*[\[(]?([A-Z0-9])[\])]?",
    flags=re.IGNORECASE,
)
_BARE_CHOICE_RE = re.compile(r"^\s*[\[(]?([A-Z0-9])[\])]?\s*[.!]?\s*$", flags=re.IGNORECASE)


def parse_multiple_choice(
    response: str, labels: tuple[str, ...], texts: tuple[str, ...]
) -> str | None:
    """Parse one unambiguous option label or exact normalized option text."""

    canonical_labels = tuple(label.upper() for label in labels)
    explicit = [match.upper() for match in _EXPLICIT_CHOICE_RE.findall(response)]
    recognized = [match for match in explicit if match in canonical_labels]
    if recognized:
        return recognized[0] if len(set(recognized)) == 1 else None
    bare = _BARE_CHOICE_RE.fullmatch(response)
    if bare is not None:
        candidate = bare.group(1).upper()
        return candidate if candidate in canonical_labels else None
    normalized = normalize_text(response)
    matched = [
        label
        for label, text in zip(canonical_labels, texts, strict=True)
        if normalize_text(text) == normalized
    ]
    return matched[0] if len(matched) == 1 else None


@dataclass(frozen=True)
class MultipleChoiceReference:
    """One correct label and its complete labelled option set."""

    correct_label: str
    labels: tuple[str, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.labels or len(self.labels) != len(self.texts):
            raise ValueError("Multiple-choice labels and texts must be nonempty and equal length")
        canonical = tuple(label.upper() for label in self.labels)
        if len(set(canonical)) != len(canonical):
            raise ValueError("Multiple-choice labels must be unique")
        if self.correct_label.upper() not in canonical:
            raise ValueError("Correct label must appear in labels")


@dataclass(frozen=True)
class MultipleChoiceScorer:
    """Exact option-match scorer with a documented zero for parsing failure."""

    reward_version: str = "option-exact-v1"

    def score(self, reference: MultipleChoiceReference, response: str) -> RewardResult:
        parsed = parse_multiple_choice(response, reference.labels, reference.texts)
        expected = reference.correct_label.upper()
        reward = float(parsed is not None and parsed == expected)
        return RewardResult(
            raw_reward=reward,
            normalized_reward=reward,
            parse_succeeded=parsed is not None,
            parsed_response=parsed,
            reward_version=self.reward_version,
            details={"expected": expected},
        )
