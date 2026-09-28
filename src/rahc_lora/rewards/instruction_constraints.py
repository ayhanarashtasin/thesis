"""Versioned deterministic validators for the custom constraint dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.normalization import normalize_text

VALIDATOR_VERSION = "1.0.0"


@dataclass(frozen=True)
class ConstraintSpec:
    """Serializable constraint kind and parameters."""

    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)


def _words(value: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE)


def _whole_word_count(response: str, keyword: str) -> int:
    return len(
        re.findall(
            rf"(?<!\w){re.escape(normalize_text(keyword))}(?!\w)",
            normalize_text(response),
        )
    )


def validate_constraint(response: str, constraint: ConstraintSpec) -> bool:
    """Evaluate one supported constraint or fail on an unknown schema."""

    params = constraint.parameters
    if constraint.kind == "keyword_count":
        keyword = str(params["keyword"])
        expected = int(params["count"])
        return _whole_word_count(response, keyword) == expected
    if constraint.kind == "bullet_count":
        expected = int(params["count"])
        bullets = re.findall(r"^\s*[-*]\s+\S", response, flags=re.MULTILINE)
        return len(bullets) == expected
    if constraint.kind == "forbidden_words":
        forbidden = tuple(str(word) for word in params["words"])
        return all(_whole_word_count(response, word) == 0 for word in forbidden)
    if constraint.kind == "ending_phrase":
        phrase = normalize_text(str(params["phrase"]))
        return normalize_text(response).endswith(phrase)
    if constraint.kind == "word_count":
        count = len(_words(response))
        return int(params["minimum"]) <= count <= int(params["maximum"])
    if constraint.kind == "required_sections":
        required = {normalize_text(str(title)) for title in params["titles"]}
        matches = re.findall(
            r"^\s*(?:#{1,6}\s+([^:\n]+)|([^:\n]+):)\s*$",
            response,
            flags=re.MULTILINE,
        )
        headings = {normalize_text(markdown or colon) for markdown, colon in matches}
        return required.issubset(headings)
    if constraint.kind == "json_fields":
        try:
            value = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return False
        required_fields = {str(name) for name in params["required_fields"]}
        return isinstance(value, dict) and required_fields.issubset(value)
    raise ValueError(f"Unknown constraint kind: {constraint.kind!r}")


@dataclass(frozen=True)
class ConstraintScorer:
    """Score the satisfied fraction of a nonempty deterministic constraint set."""

    reward_version: str = "constraint-fraction-v1"

    def score(self, reference: tuple[ConstraintSpec, ...], response: str) -> RewardResult:
        if not reference:
            raise ValueError("Constraint reward requires at least one constraint")
        outcomes = tuple(validate_constraint(response, constraint) for constraint in reference)
        reward = sum(outcomes) / len(outcomes)
        return RewardResult(
            raw_reward=reward,
            normalized_reward=reward,
            parse_succeeded=bool(response.strip()),
            parsed_response=response if response.strip() else None,
            reward_version=self.reward_version,
            details={
                "satisfied": sum(outcomes),
                "total": len(outcomes),
                "outcomes": list(outcomes),
                "validator_version": VALIDATOR_VERSION,
            },
        )
