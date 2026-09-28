"""Continual-learning metrics derived only from full performance matrices."""

from __future__ import annotations

from dataclasses import dataclass

from rahc_lora.evaluation.performance_matrix import PerformanceMatrix


def _require_complete(matrix: PerformanceMatrix) -> None:
    if matrix.completed_rows != len(matrix.task_ids):
        raise ValueError(
            f"Final continual metrics require {len(matrix.task_ids)} rows; "
            f"received {matrix.completed_rows}"
        )


def average_forgetting(matrix: PerformanceMatrix) -> float:
    """Mean prior best minus final score over all tasks except the final task."""

    _require_complete(matrix)
    if len(matrix.task_ids) == 1:
        return 0.0
    final_row = len(matrix.task_ids) - 1
    forgetting: list[float] = []
    for task_index in range(final_row):
        prior_values = [
            matrix.get(row_index, task_index) for row_index in range(task_index, final_row)
        ]
        if any(value is None for value in prior_values):
            raise ValueError(f"Missing historical score for task {matrix.task_ids[task_index]!r}")
        final_value = matrix.get(final_row, task_index)
        if final_value is None:
            raise ValueError(f"Missing final score for task {matrix.task_ids[task_index]!r}")
        best_previous = max(float(value) for value in prior_values if value is not None)
        forgetting.append(best_previous - final_value)
    return sum(forgetting) / len(forgetting)


def final_average_performance(matrix: PerformanceMatrix) -> float:
    """Mean final-row performance over every task."""

    _require_complete(matrix)
    final_row = matrix.completed_rows - 1
    values = [matrix.get(final_row, index) for index in range(len(matrix.task_ids))]
    if any(value is None for value in values):
        raise ValueError("Final performance row is incomplete")
    return sum(float(value) for value in values if value is not None) / len(values)


def backward_transfer(matrix: PerformanceMatrix) -> float:
    """Mean final-score change relative to each old task's score when learned."""

    _require_complete(matrix)
    if len(matrix.task_ids) == 1:
        return 0.0
    final_row = matrix.completed_rows - 1
    changes: list[float] = []
    for task_index in range(final_row):
        learned_score = matrix.get(task_index, task_index)
        final_score = matrix.get(final_row, task_index)
        if learned_score is None or final_score is None:
            raise ValueError("Backward transfer requires diagonal and final scores")
        changes.append(final_score - learned_score)
    return sum(changes) / len(changes)


def forward_transfer(matrix: PerformanceMatrix, frozen_baselines: dict[str, float]) -> float:
    """Mean pre-training score change for tasks 2..T relative to the frozen policy."""

    _require_complete(matrix)
    if len(matrix.task_ids) == 1:
        return 0.0
    changes: list[float] = []
    for task_index in range(1, len(matrix.task_ids)):
        task_id = matrix.task_ids[task_index]
        if task_id not in frozen_baselines:
            raise ValueError(f"Missing frozen baseline for task {task_id!r}")
        before_learning = matrix.get(task_index - 1, task_index)
        if before_learning is None:
            raise ValueError(
                f"Forward transfer requires pre-learning cell for task {task_id!r}; "
                "future cells were not evaluated"
            )
        changes.append(before_learning - frozen_baselines[task_id])
    return sum(changes) / len(changes)


@dataclass(frozen=True)
class ContinualMetricSummary:
    """Co-primary and transfer metrics for one complete task stream."""

    average_forgetting: float
    final_average_performance: float
    backward_transfer: float
    forward_transfer: float | None


def summarize_continual_metrics(
    matrix: PerformanceMatrix,
    *,
    frozen_baselines: dict[str, float] | None = None,
) -> ContinualMetricSummary:
    """Calculate all supported metrics without inventing unavailable forward-transfer cells."""

    return ContinualMetricSummary(
        average_forgetting=average_forgetting(matrix),
        final_average_performance=final_average_performance(matrix),
        backward_transfer=backward_transfer(matrix),
        forward_transfer=(
            forward_transfer(matrix, frozen_baselines) if frozen_baselines is not None else None
        ),
    )
