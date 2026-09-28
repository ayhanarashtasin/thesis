"""Versioned deterministic response and overlap normalization."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

NORMALIZATION_VERSION = "text-nfkc-casefold-v1"
NUMERIC_NORMALIZATION_VERSION = "numeric-fraction-v1"

_SPACE_RE = re.compile(r"\s+")
_OVERLAP_TOKEN_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+]?(?:[\u0024\u00a3\u20ac]\s*)?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?(?![\w.])"
)
_FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)?\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def normalize_text(value: str) -> str:
    """Normalize Unicode, whitespace, and case for exact textual comparison."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return _SPACE_RE.sub(" ", normalized).strip().casefold()


def normalize_for_overlap(value: str) -> str:
    """Normalize text aggressively for cross-corpus prompt-overlap checks."""

    normalized = normalize_text(value)
    return _SPACE_RE.sub(" ", _OVERLAP_TOKEN_RE.sub(" ", normalized)).strip()


def _fraction_from_token(token: str) -> Fraction | None:
    cleaned = token.strip().replace(",", "")
    for currency in ("\u0024", "\u00a3", "\u20ac"):
        cleaned = cleaned.replace(currency, "")
    cleaned = _SPACE_RE.sub("", cleaned)
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            denominator_decimal = Decimal(denominator)
            if denominator_decimal == 0:
                return None
            value = Fraction(Decimal(numerator)) / Fraction(denominator_decimal)
        else:
            value = Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value / 100 if is_percent else value


def normalize_numeric_token(token: str) -> str | None:
    """Return an exact canonical fraction string for one numeric token."""

    value = _fraction_from_token(token)
    if value is None:
        return None
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def extract_final_numeric_answer(response: str) -> str | None:
    """Extract a conventional final numeric answer, returning ``None`` on parse failure."""

    normalized = unicodedata.normalize("NFKC", response).strip()
    if not normalized:
        return None
    marked_segment: str | None = None
    if "####" in normalized:
        marked_segment = normalized.rsplit("####", maxsplit=1)[1]
    else:
        matches = list(_FINAL_MARKER_RE.finditer(normalized))
        if matches:
            marked_segment = matches[-1].group(1)
    search_text = marked_segment if marked_segment is not None else normalized
    tokens = _NUMERIC_TOKEN_RE.findall(search_text)
    if not tokens:
        return None
    return normalize_numeric_token(tokens[-1])
