from __future__ import annotations

from pathlib import Path

import pytest

from rahc_lora.evaluation.continual_metrics import (
    average_forgetting,
    backward_transfer,
    final_average_performance,
    forward_transfer,
    summarize_continual_metrics,
)
from rahc_lora.evaluation.performance_matrix import PerformanceMatrix


def _complete_matrix() -> PerformanceMatrix:
    matrix = PerformanceMatrix(("t1", "t2", "t3"))
    matrix.append_row("t1", {"t1": 0.8, "t2": 0.3, "t3": 0.2})
    matrix.append_row("t2", {"t1": 0.6, "t2": 0.7, "t3": 0.4})
    matrix.append_row("t3", {"t1": 0.5, "t2": 0.65, "t3": 0.9})
    return matrix


def test_continual_metrics_match_hand_calculation() -> None:
    matrix = _complete_matrix()
    assert average_forgetting(matrix) == pytest.approx(0.175)
    assert final_average_performance(matrix) == pytest.approx((0.5 + 0.65 + 0.9) / 3)
    assert backward_transfer(matrix) == pytest.approx(-0.175)
    assert forward_transfer(matrix, {"t2": 0.2, "t3": 0.1}) == pytest.approx(0.2)
    summary = summarize_continual_metrics(matrix, frozen_baselines={"t2": 0.2, "t3": 0.1})
    assert summary.average_forgetting == pytest.approx(0.175)
    assert summary.forward_transfer == pytest.approx(0.2)


def test_future_cells_remain_explicit_missing_and_csv_round_trips(tmp_path: Path) -> None:
    matrix = PerformanceMatrix(("t1", "t2"))
    matrix.append_row("t1", {"t1": 0.75})
    assert matrix.get(0, 1) is None
    path = tmp_path / "performance_matrix.csv"
    matrix.write_csv(path)
    assert ",\n" in path.read_text(encoding="utf-8")
    assert PerformanceMatrix.read_csv(path).values == matrix.values


def test_metrics_reject_incomplete_matrix_and_missing_forward_cells() -> None:
    incomplete = PerformanceMatrix(("t1", "t2"))
    incomplete.append_row("t1", {"t1": 0.5})
    with pytest.raises(ValueError, match="require 2 rows"):
        average_forgetting(incomplete)

    complete = PerformanceMatrix(("t1", "t2"))
    complete.append_row("t1", {"t1": 0.5})
    complete.append_row("t2", {"t1": 0.4, "t2": 0.8})
    with pytest.raises(ValueError, match="future cells were not evaluated"):
        forward_transfer(complete, {"t2": 0.2})
