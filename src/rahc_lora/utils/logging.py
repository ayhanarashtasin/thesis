"""Local structured JSON Lines logging for scientific and systems events."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

EVENT_CATEGORIES = frozenset(
    {"learning", "retention", "consolidation", "conflict", "system", "reproducibility"}
)


def _assert_finite(value: Any, path: str = "values") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite structured metric at {path}: {value!r}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


class JsonlLogger:
    """Append validated structured events to an authoritative local JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = path.open("a", encoding="utf-8", newline="\n")

    def log(
        self,
        *,
        category: str,
        event: str,
        values: Mapping[str, Any],
        step: int | None = None,
    ) -> None:
        """Write one event after category and finite-value validation."""

        if category not in EVENT_CATEGORIES:
            allowed = ", ".join(sorted(EVENT_CATEGORIES))
            raise ValueError(f"Unknown event category {category!r}; expected one of: {allowed}")
        if not event:
            raise ValueError("Structured log event name must not be empty")
        if step is not None and step < 0:
            raise ValueError(f"Structured log step must be nonnegative, received {step}")
        _assert_finite(values)
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "category": category,
            "event": event,
            "step": step,
            "values": dict(values),
        }
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        """Flush and close the log stream."""

        self._stream.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
