"""Complete task-by-task performance matrix with explicit missing future cells."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerformanceMatrix:
    """Matrix ``A[i, j]`` after training task ``i`` and evaluating task ``j``."""

    task_ids: tuple[str, ...]
    _rows: list[list[float | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("PerformanceMatrix task IDs must be nonempty and unique")
        for index, row in enumerate(self._rows):
            self._validate_row(index, row)

    @property
    def completed_rows(self) -> int:
        return len(self._rows)

    @property
    def values(self) -> tuple[tuple[float | None, ...], ...]:
        return tuple(tuple(row) for row in self._rows)

    def _validate_row(self, row_index: int, row: list[float | None]) -> None:
        if len(row) != len(self.task_ids):
            raise ValueError("Performance matrix row has the wrong number of task columns")
        for column_index, value in enumerate(row):
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"Performance matrix value [{row_index}, {column_index}] is non-finite"
                )
            if column_index <= row_index and value is None:
                raise ValueError(
                    f"Learned task cell [{row_index}, {column_index}] may not be missing"
                )

    def append_row(self, after_task_id: str, scores: dict[str, float]) -> None:
        """Append the next task boundary row; future cells remain ``None`` unless supplied."""

        row_index = len(self._rows)
        if row_index >= len(self.task_ids):
            raise ValueError("Performance matrix already has all task rows")
        expected_task_id = self.task_ids[row_index]
        if after_task_id != expected_task_id:
            raise ValueError(
                f"Next row must follow task {expected_task_id!r}, received {after_task_id!r}"
            )
        unknown = sorted(set(scores).difference(self.task_ids))
        if unknown:
            raise ValueError(f"Performance row contains unknown tasks: {unknown}")
        missing_learned = [
            task_id for task_id in self.task_ids[: row_index + 1] if task_id not in scores
        ]
        if missing_learned:
            raise ValueError(f"Performance row is missing learned tasks: {missing_learned}")
        row = [scores.get(task_id) for task_id in self.task_ids]
        self._validate_row(row_index, row)
        self._rows.append(row)

    def get(self, row: int, column: int) -> float | None:
        return self._rows[row][column]

    def clone(self) -> PerformanceMatrix:
        """Return an independent validated copy for transactional boundary evaluation."""

        return PerformanceMatrix(self.task_ids, [list(row) for row in self._rows])

    def write_csv(self, path: Path) -> None:
        """Atomically save all cells, writing future missing values as empty fields."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["after_task", *self.task_ids])
                for index, row in enumerate(self._rows):
                    writer.writerow(
                        [
                            self.task_ids[index],
                            *("" if value is None else format(value, ".17g") for value in row),
                        ]
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()

    @classmethod
    def read_csv(cls, path: Path) -> PerformanceMatrix:
        """Load and validate a matrix saved by :meth:`write_csv`."""

        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows or not rows[0] or rows[0][0] != "after_task":
            raise ValueError("Performance matrix CSV has an invalid header")
        task_ids = tuple(rows[0][1:])
        matrix = cls(task_ids)
        for index, row in enumerate(rows[1:]):
            if len(row) != len(task_ids) + 1 or row[0] != task_ids[index]:
                raise ValueError(f"Performance matrix CSV row {index} is malformed")
            scores = {
                task_id: float(value)
                for task_id, value in zip(task_ids, row[1:], strict=True)
                if value != ""
            }
            matrix.append_row(row[0], scores)
        return matrix
