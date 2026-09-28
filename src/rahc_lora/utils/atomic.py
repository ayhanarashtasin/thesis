"""Small atomic local-artifact helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, payload: str) -> None:
    """Write UTF-8 text and atomically replace one local artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    """Write deterministic JSON through :func:`write_text_atomic`."""

    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
