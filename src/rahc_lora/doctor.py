"""Read-only environment diagnostics with safe writable-path probes."""

from __future__ import annotations

import importlib
import os
import platform
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rahc_lora.config import compose_config
from rahc_lora.evaluation.official_ifeval import GoogleResearchIFEvalScorer
from rahc_lora.utils.manifests import dependency_versions


def _probe_writable(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".rahc-write-probe-", dir=resolved)
        os.close(descriptor)
        Path(temporary_path).unlink()
    except OSError as error:
        return {"path": str(resolved), "writable": False, "error": str(error)}
    return {"path": str(resolved), "writable": True, "error": None}


def _torch_report() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_runtime": None,
            "gpu_names": [],
            "bf16_supported": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_names": gpu_names,
        "bf16_supported": bf16_supported,
    }


def collect_doctor_report(
    *, config_name: str = "config", overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Collect diagnostics without downloading data, loading a model, or exposing secrets."""

    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "requirement": ">=3.11,<3.13",
            "compatible": (3, 11) <= sys.version_info[:2] < (3, 13),
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": _torch_report(),
        "dependencies": dependency_versions(),
    }

    try:
        config = compose_config(config_name=config_name, overrides=overrides)
    except Exception as error:
        report["configuration"] = {
            "loaded": False,
            "config_name": config_name,
            "error": f"{type(error).__name__}: {error}",
        }
        report["paths"] = {}
        report["general_capabilities"] = {"enabled": None, "ready": False, "error": None}
        report["readiness"] = {"cpu_development": False, "gpu_experiment": False}
        return report

    output_probe = _probe_writable(Path(str(config.paths.output_dir)))
    cache_probe = _probe_writable(Path(str(config.paths.cache_dir)))
    report["configuration"] = {
        "loaded": True,
        "config_name": config_name,
        "method": str(config.method.name),
        "experiment": str(config.experiment.name),
        "task_order": str(config.task_stream.name),
        "error": None,
    }
    report["paths"] = {"output": output_probe, "cache": cache_probe}
    capability_ready = True
    capability_error: str | None = None
    capability_repository: str | None = None
    if config.general_capabilities.enabled:
        capability_repository = str(config.general_capabilities.ifeval_scorer_repository)
        try:
            scorer = GoogleResearchIFEvalScorer(
                Path(capability_repository),
                expected_revision=str(config.general_capabilities.ifeval_scorer_revision),
                deterministic_seed=int(config.general_capabilities.ifeval_scorer_seed),
                nltk_data_root=Path(str(config.general_capabilities.ifeval_nltk_data_dir)),
                expected_punkt_tab_sha256=str(config.general_capabilities.ifeval_punkt_tab_sha256),
            )
            capability_repository = scorer.repository_root
        except (ImportError, RuntimeError, ValueError) as error:
            capability_ready = False
            capability_error = f"{type(error).__name__}: {error}"
    report["general_capabilities"] = {
        "enabled": bool(config.general_capabilities.enabled),
        "ready": capability_ready,
        "repository": capability_repository,
        "revision": str(config.general_capabilities.ifeval_scorer_revision),
        "error": capability_error,
    }
    cpu_ready = bool(
        report["python"]["compatible"]
        and output_probe["writable"]
        and cache_probe["writable"]
        and capability_ready
    )
    precision = str(config.model.precision)
    precision_ready = precision != "bfloat16" or report["torch"]["bf16_supported"]
    report["readiness"] = {
        "cpu_development": cpu_ready,
        "gpu_experiment": bool(cpu_ready and report["torch"]["cuda_available"] and precision_ready),
    }
    return report


def doctor_succeeded(report: dict[str, Any]) -> bool:
    """Return whether configuration and CPU development prerequisites are valid."""

    return bool(report.get("readiness", {}).get("cpu_development", False))
