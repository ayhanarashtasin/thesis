"""Unit tests for Phase 3: DensePathImportance for HLoRA-RL."""

from __future__ import annotations

import torch
import pytest

from rahc_lora.consolidation.hlora import DensePathImportance


def test_dense_path_importance_zero_movement_produces_zero_path() -> None:
    importance = DensePathImportance(epsilon=1e-8)
    initial_weights = {"q_proj": torch.randn(16, 16)}
    importance.begin_task(initial_weights)

    grad = {"q_proj": torch.randn(16, 16)}
    same_weights = {"q_proj": initial_weights["q_proj"].clone()}
    diag = importance.observe_step(
        effective_gradients=grad,
        effective_before=same_weights,
        effective_after=same_weights,
    )
    assert diag["q_proj"]["movement_l2"] == 0.0
    assert diag["q_proj"]["path_contribution_sum"] == 0.0

    ref_factors = {"base_model.model.layers.0.self_attn.q_proj.lora_A": torch.randn(4, 16)}
    cons_diag = importance.consolidate(
        final_effective_weights=same_weights,
        reference_lora_factors=ref_factors,
    )
    assert cons_diag.task_steps == 1
    assert cons_diag.clamped_negative_elements == 0
    assert torch.all(importance.importance["q_proj"] == 0.0)


def test_dense_path_importance_penalty_with_reference() -> None:
    importance = DensePathImportance(epsilon=1e-8)
    initial_weights = {"q_proj": torch.zeros(4, 4)}
    importance.begin_task(initial_weights)

    grad = {"q_proj": -torch.ones(4, 4)}
    before = {"q_proj": torch.zeros(4, 4)}
    after = {"q_proj": torch.ones(4, 4)}
    importance.observe_step(
        effective_gradients=grad,
        effective_before=before,
        effective_after=after,
    )
    ref_factors = {"adapter": torch.zeros(2, 4)}
    importance.consolidate(
        final_effective_weights=after,
        reference_lora_factors=ref_factors,
    )
    assert torch.all(importance.importance["q_proj"] > 0)

    # If current == reference, penalty is zero
    ref_effective = {"q_proj": torch.ones(4, 4)}
    current_effective = {"q_proj": torch.ones(4, 4)}
    penalty_zero = importance.penalty_with_reference(current_effective, ref_effective)
    assert torch.isclose(penalty_zero, torch.tensor(0.0))

    # If current moves away from reference, penalty is positive
    current_moved = {"q_proj": torch.zeros(4, 4)}
    penalty_moved = importance.penalty_with_reference(current_moved, ref_effective)
    assert penalty_moved.item() > 0.0


def test_dense_path_importance_serialization_round_trip() -> None:
    importance = DensePathImportance(epsilon=1e-5)
    w = {"q_proj": torch.randn(8, 8)}
    importance.begin_task(w)
    importance.observe_step(
        effective_gradients={"q_proj": torch.randn(8, 8)},
        effective_before=w,
        effective_after=w,
    )
    importance.consolidate(
        final_effective_weights=w,
        reference_lora_factors={"a": torch.randn(2, 8)},
    )
    state = importance.state_dict()

    restored = DensePathImportance(epsilon=1e-5)
    restored.load_state_dict(
        state,
        expected_effective_shapes={"q_proj": (8, 8)},
        expected_factor_shapes={"a": (2, 8)},
    )
    assert restored.epsilon == importance.epsilon
    assert restored.layer_coefficients == importance.layer_coefficients
    assert torch.equal(restored.importance["q_proj"], importance.importance["q_proj"])
    assert torch.equal(
        restored.reference_lora_factors["a"], importance.reference_lora_factors["a"]
    )
