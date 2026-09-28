"""Deterministic normalized exact-match rewards."""

from __future__ import annotations

from dataclasses import dataclass

from rahc_lora.rewards.base import RewardResult
from rahc_lora.rewards.normalization import (
    extract_final_numeric_answer,
    normalize_numeric_token,
    normalize_text,
)


@dataclass(frozen=True)
class TextExactMatchScorer:
    """Case-folded NFKC exact match with whitespace normalization."""

    reward_version: str = "text-exact-v1"

    def score(self, reference: str, response: str) -> RewardResult:
        expected = normalize_text(reference)
        parsed = normalize_text(response)
        parse_succeeded = bool(parsed)
        reward = float(parse_succeeded and parsed == expected)
        return RewardResult(
            raw_reward=reward,
            normalized_reward=reward,
            parse_succeeded=parse_succeeded,
            parsed_response=parsed if parse_succeeded else None,
            reward_version=self.reward_version,
            details={"expected": expected},
        )


@dataclass(frozen=True)
class NumericExactMatchScorer:
    """Exact rational-number comparison for GSM8K-style final answers."""

    reward_version: str = "numeric-exact-v1"

    def score(self, reference: str, response: str) -> RewardResult:
        expected = extract_final_numeric_answer(reference)
        if expected is None:
            expected = normalize_numeric_token(reference)
        if expected is None:
            raise ValueError(f"Reference numeric answer is unparsable: {reference!r}")
        parsed = extract_final_numeric_answer(response)
        reward = float(parsed is not None and parsed == expected)
        return RewardResult(
            raw_reward=reward,
            normalized_reward=reward,
            parse_succeeded=parsed is not None,
            parsed_response=parsed,
            reward_version=self.reward_version,
            details={"expected": expected},
        )
