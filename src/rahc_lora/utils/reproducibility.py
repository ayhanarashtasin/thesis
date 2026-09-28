"""Random-seed utilities for deterministic local and experiment execution."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedReport:
    """Components seeded by :func:`seed_everything`."""

    seed: int
    python: bool
    numpy: bool
    torch: bool
    cuda: bool
    deterministic_algorithms: bool


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> SeedReport:
    """Seed Python, NumPy, PyTorch CPU, and all available CUDA generators."""

    if seed < 0:
        raise ValueError(f"Seed must be nonnegative, received {seed}")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    return SeedReport(
        seed=seed,
        python=True,
        numpy=True,
        torch=True,
        cuda=cuda_available,
        deterministic_algorithms=deterministic_algorithms,
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and available CUDA RNG states."""

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    import numpy as np
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"RNG state is missing required entries: {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG state because CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
