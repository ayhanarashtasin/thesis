"""Dense path-integral importance and layer-weighted regularization for HLoRA-RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConsolidationDiagnostics:
    """JSON-safe summary of one task-boundary importance update."""

    task_steps: int
    clamped_negative_elements: int
    negative_path_mass: float
    importance_storage_bytes: int
    layers: dict[str, dict[str, float | int]]


class DensePathImportance:
    """Accumulate dense SI-style importance over effective LoRA updates.

    Persistent tensors are held on CPU. Step gradients and movements are detached and
    transferred to CPU before accumulation so the reference implementation does not keep
    dense importance state in the model's GPU memory.
    """

    SCHEMA_VERSION = 1

    def __init__(self, *, epsilon: float) -> None:
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Dense path-importance epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.importance: dict[str, Tensor] = {}
        self.reference_lora_factors: dict[str, Tensor] = {}
        self.layer_coefficients: dict[str, float] = {}
        self._task_start: dict[str, Tensor] | None = None
        self._last_effective: dict[str, Tensor] | None = None
        self._path_integral: dict[str, Tensor] | None = None
        self._task_steps = 0

    @staticmethod
    def _cpu_float_weights(weights: dict[str, Tensor], *, label: str) -> dict[str, Tensor]:
        if not weights:
            raise ValueError(f"{label} must contain at least one LoRA layer")
        output: dict[str, Tensor] = {}
        for name, value in weights.items():
            if not name or not isinstance(value, Tensor) or value.ndim != 2:
                raise ValueError(f"{label} entry {name!r} must be a rank-two tensor")
            detached = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if not bool(torch.isfinite(detached).all()):
                raise ValueError(f"{label} contains non-finite values for {name!r}")
            output[name] = detached.clone()
        return output

    def begin_task(self, effective_weights: dict[str, Tensor]) -> None:
        """Start a task path from the current shared policy."""

        if self._task_start is not None:
            raise RuntimeError("Dense importance already has an active task")
        start = self._cpu_float_weights(effective_weights, label="Task-start effective weights")
        expected = set(self.importance) if self.importance else set(start)
        if set(start) != expected:
            raise ValueError("Task-start LoRA layer names disagree with consolidated importance")
        for name, weight in start.items():
            if name in self.importance and self.importance[name].shape != weight.shape:
                raise ValueError(f"Task-start effective-weight shape changed for {name!r}")
        self._task_start = start
        self._last_effective = {name: value.clone() for name, value in start.items()}
        self._path_integral = {name: torch.zeros_like(value) for name, value in start.items()}
        self._task_steps = 0

    def observe_step(
        self,
        *,
        effective_gradients: dict[str, Tensor],
        effective_before: dict[str, Tensor],
        effective_after: dict[str, Tensor],
    ) -> dict[str, dict[str, float]]:
        """Accumulate `-gradient * observed effective-weight movement` for one update."""

        if self._task_start is None or self._last_effective is None or self._path_integral is None:
            raise RuntimeError("begin_task must be called before observing importance")
        expected = set(self._task_start)
        if (
            set(effective_gradients) != expected
            or set(effective_before) != expected
            or set(effective_after) != expected
        ):
            raise ValueError("Importance step tensors must contain the same LoRA layer names")
        diagnostics: dict[str, dict[str, float]] = {}
        for name in sorted(expected):
            gradient = effective_gradients[name].detach().to(device="cpu", dtype=torch.float32)
            before = effective_before[name].detach().to(device="cpu", dtype=torch.float32)
            after = effective_after[name].detach().to(device="cpu", dtype=torch.float32)
            if gradient.shape != self._task_start[name].shape:
                raise ValueError(f"Effective gradient shape changed for {name!r}")
            if before.shape != gradient.shape or after.shape != gradient.shape:
                raise ValueError(f"Effective movement shape changed for {name!r}")
            if not all(bool(torch.isfinite(value).all()) for value in (gradient, before, after)):
                raise ValueError(f"Non-finite path-integral input for {name!r}")
            if not torch.allclose(before, self._last_effective[name], rtol=1e-5, atol=1e-7):
                raise ValueError(f"Observed effective-weight path is discontinuous for {name!r}")
            movement = after - before
            contribution = -gradient * movement
            self._path_integral[name].add_(contribution)
            diagnostics[name] = {
                "gradient_l2": float(torch.linalg.vector_norm(gradient).item()),
                "movement_l2": float(torch.linalg.vector_norm(movement).item()),
                "path_contribution_sum": float(
                    contribution.sum(dtype=torch.float64).item()
                ),
            }
            self._last_effective[name] = after.contiguous().clone()
        self._task_steps += 1
        return diagnostics

    def consolidate(
        self,
        *,
        final_effective_weights: dict[str, Tensor],
        reference_lora_factors: dict[str, Tensor],
    ) -> ConsolidationDiagnostics:
        """Normalize task contributions, merge importance, and refresh the reference."""

        if self._task_start is None or self._last_effective is None or self._path_integral is None:
            raise RuntimeError("No active task path to consolidate")
        final = self._cpu_float_weights(
            final_effective_weights, label="Final effective weights"
        )
        factors: dict[str, Tensor] = {}
        for name, value in reference_lora_factors.items():
            if not name or not isinstance(value, Tensor):
                raise ValueError("Reference LoRA factors must be a named tensor mapping")
            detached = value.detach().to(device="cpu").contiguous().clone()
            if not bool(torch.isfinite(detached).all()):
                raise ValueError(f"Reference LoRA factor {name!r} is non-finite")
            factors[name] = detached
        if not factors:
            raise ValueError("Reference LoRA factors cannot be empty")
        if set(final) != set(self._task_start):
            raise ValueError("Final effective-weight layer names changed during the task")

        clamped_negative_elements = 0
        negative_path_mass = 0.0
        merged: dict[str, Tensor] = {}
        layer_norms: dict[str, Tensor] = {}
        for name in sorted(final):
            if not torch.allclose(final[name], self._last_effective[name], rtol=1e-5, atol=1e-7):
                raise ValueError(f"Final effective weights disagree with the observed path for {name!r}")
            path = self._path_integral[name]
            negative = path < 0
            clamped_negative_elements += int(negative.sum().item())
            negative_path_mass += float((-path[negative]).sum(dtype=torch.float64).item())
            displacement = final[name] - self._task_start[name]
            task_importance = path.clamp_min(0) / (displacement.square() + self.epsilon)
            old = self.importance.get(name)
            total = task_importance if old is None else old + task_importance
            if not bool(torch.isfinite(total).all()) or bool((total < 0).any()):
                raise ValueError(f"Consolidated importance is invalid for {name!r}")
            merged[name] = total.contiguous()
            layer_norms[name] = torch.linalg.vector_norm(total)

        norm_values = torch.stack([layer_norms[name] for name in sorted(layer_norms)])
        coefficients = torch.softmax(norm_values, dim=0)
        if not bool(torch.isfinite(coefficients).all()):
            raise ValueError("Static HLoRA layer coefficients are non-finite")
        self.importance = merged
        self.reference_lora_factors = factors
        self.layer_coefficients = {
            name: float(coefficients[index].item())
            for index, name in enumerate(sorted(layer_norms))
        }
        bytes_used = sum(value.numel() * value.element_size() for value in merged.values())
        layers = {
            name: {
                "importance_l2": float(layer_norms[name].item()),
                "coefficient": self.layer_coefficients[name],
                "elements": int(merged[name].numel()),
                "nonzero_fraction": float((merged[name] > 0).float().mean().item()),
                "storage_bytes": int(merged[name].numel() * merged[name].element_size()),
            }
            for name in sorted(merged)
        }
        diagnostics = ConsolidationDiagnostics(
            task_steps=self._task_steps,
            clamped_negative_elements=clamped_negative_elements,
            negative_path_mass=negative_path_mass,
            importance_storage_bytes=bytes_used,
            layers=layers,
        )
        self._task_start = None
        self._last_effective = None
        self._path_integral = None
        self._task_steps = 0
        return diagnostics

    def penalty_with_reference(
        self,
        current_effective_weights: dict[str, Tensor],
        reference_effective_weights: dict[str, Tensor],
    ) -> Tensor:
        """Return the penalty with reference effective weights computed by the policy."""

        if not self.importance or not self.reference_lora_factors:
            if not current_effective_weights:
                raise ValueError("Current effective weights cannot be empty")
            return next(iter(current_effective_weights.values())).sum() * 0.0
        if (
            set(current_effective_weights) != set(self.importance)
            or set(reference_effective_weights) != set(self.importance)
            or set(self.layer_coefficients) != set(self.importance)
        ):
            raise ValueError("HLoRA penalty layer names disagree with consolidated state")

        loss: Tensor | None = None
        for name, current in current_effective_weights.items():
            importance = self.importance[name].to(device=current.device, dtype=torch.float32)
            reference = reference_effective_weights[name].to(
                device=current.device, dtype=torch.float32
            )
            current_float = current.to(torch.float32)
            if current_float.shape != importance.shape or reference.shape != importance.shape:
                raise ValueError(f"Penalty tensor shape mismatch for {name!r}")
            term = self.layer_coefficients[name] * (
                importance * (current_float - reference).square()
            ).sum()
            loss = term if loss is None else loss + term
        if loss is None or not bool(torch.isfinite(loss).all()):
            raise ValueError("HLoRA parameter penalty is non-finite")
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return boundary state suitable for a trusted torch checkpoint."""

        if self._task_start is not None:
            raise RuntimeError("HLoRA checkpoints are supported only at task boundaries")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "epsilon": self.epsilon,
            "importance": {name: value.clone() for name, value in self.importance.items()},
            "reference_lora_factors": {
                name: value.clone() for name, value in self.reference_lora_factors.items()
            },
            "layer_coefficients": dict(self.layer_coefficients),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        expected_effective_shapes: dict[str, tuple[int, ...]],
        expected_factor_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        """Load and validate state at a completed task boundary."""

        required = {
            "schema_version",
            "epsilon",
            "importance",
            "reference_lora_factors",
            "layer_coefficients",
        }
        if set(state) != required or state["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("Unsupported HLoRA importance state")
        if float(state["epsilon"]) != self.epsilon:
            raise ValueError("Checkpoint importance epsilon disagrees with configuration")
        raw_importance = state["importance"]
        raw_factors = state["reference_lora_factors"]
        raw_coefficients = state["layer_coefficients"]
        if not isinstance(raw_importance, dict) or set(raw_importance) != set(
            expected_effective_shapes
        ):
            raise ValueError("Checkpoint dense-importance layers disagree with policy")
        if not isinstance(raw_factors, dict) or set(raw_factors) != set(expected_factor_shapes):
            raise ValueError("Checkpoint reference-factor keys disagree with policy")
        if not isinstance(raw_coefficients, dict) or set(raw_coefficients) != set(
            expected_effective_shapes
        ):
            raise ValueError("Checkpoint layer coefficients disagree with policy")
        importance: dict[str, Tensor] = {}
        for name, shape in expected_effective_shapes.items():
            value = raw_importance[name]
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Checkpoint importance shape mismatch for {name!r}")
            dense = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if not bool(torch.isfinite(dense).all()) or bool((dense < 0).any()):
                raise ValueError(f"Checkpoint importance is invalid for {name!r}")
            importance[name] = dense
        factors: dict[str, Tensor] = {}
        for name, shape in expected_factor_shapes.items():
            value = raw_factors[name]
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Checkpoint reference-factor shape mismatch for {name!r}")
            factor = value.detach().to(device="cpu").contiguous().clone()
            if not bool(torch.isfinite(factor).all()):
                raise ValueError(f"Checkpoint reference factor is non-finite for {name!r}")
            factors[name] = factor
        coefficients = {str(name): float(value) for name, value in raw_coefficients.items()}
        if any(not math.isfinite(value) or value < 0 for value in coefficients.values()):
            raise ValueError("Checkpoint layer coefficients must be finite and nonnegative")
        if not math.isclose(sum(coefficients.values()), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Checkpoint layer coefficients must sum to one")
        self.importance = importance
        self.reference_lora_factors = factors
        self.layer_coefficients = coefficients
