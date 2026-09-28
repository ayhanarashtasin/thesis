# RAHC-LoRA

RAHC-LoRA is a research codebase for studying catastrophic forgetting in sequential
reinforcement learning with one frozen language-model backbone and one shared LoRA policy.
The repository is being implemented phase by phase according to `Goal.md`; no experimental
claims exist yet.

## Development setup

Python 3.11 is the reference interpreter. Python 3.12 is supported for local scaffold
validation.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The exact Windows/Python 3.12 CPU validation environment is recorded in
`requirements-lock-cpu-win-py312.txt`. The current Windows/Python 3.11 Conda GPU environment,
including its CUDA PyTorch build and driver, is recorded in `environment-gpu-win-py311.yml`.

On this Windows PC, VS Code defaults to the existing Conda `gpu-env` interpreter at
`C:/Users/user3/.conda/envs/gpu-env/python.exe`. It has Python 3.11 and CUDA-enabled PyTorch.
For a fresh `gpu-env` installation, activate the environment and install this package with
`python -m pip install -e ".[dev]"`, then check the GPU profile:

```powershell
python -m rahc_lora.cli doctor experiment=pilot_two_task model=small `
  method=rl_lora launcher=single_gpu
```

If VS Code previously remembered `.venv`, select `gpu-env` once with **Python: Select Interpreter**.
The CPU `.venv` remains available for comparison.

On Linux or macOS, create the environment with `python3.11 -m venv .venv` and activate it
with `source .venv/bin/activate`.

## Shortest checks

```powershell
python -m rahc_lora.cli doctor
ruff format --check .
ruff check .
mypy src/rahc_lora
pytest -q
```

The doctor command is safe on CPU-only machines. It reports CPU development readiness and
GPU experiment readiness separately, does not print environment-variable values, and does
not download models or datasets.

## Dataset preparation

Pinned dataset sources, deterministic split rules, and evaluation-only boundaries are documented in
`docs/datasets.md`. Prepare or reproduce the versioned manifests with:

```powershell
python -m rahc_lora.cli prepare-data
```

This command writes manifests and overlap evidence only. Raw datasets remain in the configured
external cache. MBPP reward execution requires the configured Docker isolation backend and fails
closed when that backend is unavailable.

## Standard RL-LoRA baseline

The Phase 2 trainer is available through the same CLI:

```powershell
python -m rahc_lora.cli train experiment=smoke method=rl_lora seed=1
```

Model, rollout, optimizer, evaluation, and checkpoint choices are Hydra configuration. See
`docs/training.md` for the CPU tiny path, artifact contract, interruption/resume workflow, and the
current pilot limitation.

## Current scope

Phases 0 and 1 are complete. Phase 2 has a tested standard RL-LoRA implementation, frozen-baseline
general-capability tracking, reproducible CPU smoke/resume evidence, and a tiny CUDA smoke in the
Conda `gpu-env`; its capable-model GPU pilot remains pending explicit approval. No experimental
success claim exists yet.
See `docs/implementation-status.md` for factual status and validation evidence.

# thesis
