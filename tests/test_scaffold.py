from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rahc_lora
from rahc_lora.config import ConfigurationError, compose_config
from rahc_lora.doctor import collect_doctor_report, doctor_succeeded
from rahc_lora.utils.logging import JsonlLogger
from rahc_lora.utils.manifests import (
    build_run_manifest,
    read_run_manifest,
    write_run_manifest,
)
from rahc_lora.utils.reproducibility import seed_everything


def test_package_imports_from_src_layout() -> None:
    package_path = Path(rahc_lora.__file__).resolve()
    assert package_path.parts[-3:-1] == ("src", "rahc_lora")


def test_root_configuration_composes_and_is_immutable() -> None:
    config = compose_config()
    assert config.method.name == "rl_lora"
    assert config.task_stream.tasks == ["gsm8k", "arc_challenge", "mbpp", "constraints"]
    assert OmegaConf.is_readonly(config)
    with pytest.raises(Exception, match="read-only"):
        config.seed = 2


def test_rahc_configuration_composes() -> None:
    config = compose_config(overrides=["method=rahc_lora"])
    assert config.method.use_anchor_kl
    assert config.method.use_parameter_consolidation
    assert config.method.use_factorized_importance
    assert config.method.use_conflict_gate
    assert config.method.use_dual_controller


def test_invalid_configuration_has_actionable_field() -> None:
    with pytest.raises(ConfigurationError, match=r"method\.anchor_memory\.capacity.*nonnegative"):
        compose_config(overrides=["method=rahc_lora", "method.anchor_memory.capacity=-1"])


def test_seed_utility_reproduces_known_sequences() -> None:
    report = seed_everything(123)
    assert report.python and report.numpy and report.torch
    assert random.random() == pytest.approx(0.052363598850944326)
    assert float(np.random.random()) == pytest.approx(0.6964691855978616)
    assert float(torch.rand(1).item()) == pytest.approx(0.29611194133758545)

    seed_everything(123)
    assert random.random() == pytest.approx(0.052363598850944326)
    assert float(np.random.random()) == pytest.approx(0.6964691855978616)
    assert float(torch.rand(1).item()) == pytest.approx(0.29611194133758545)


def test_run_manifest_serializes_and_reloads(tmp_path: Path) -> None:
    manifest = build_run_manifest(compose_config(), repository=tmp_path)
    destination = tmp_path / "run_manifest.json"
    write_run_manifest(destination, manifest)
    assert read_run_manifest(destination) == manifest
    assert len(manifest.config_sha256) == 64


def test_structured_logger_rejects_nonfinite_values(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.jsonl"
    with JsonlLogger(destination) as logger:
        logger.log(category="learning", event="test_metric", values={"loss": 1.25}, step=0)
        with pytest.raises(ValueError, match="Non-finite"):
            logger.log(
                category="learning",
                event="bad_metric",
                values={"loss": math.inf},
                step=1,
            )
    records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert records[0]["values"] == {"loss": 1.25}


def test_doctor_is_cpu_safe_and_checks_paths(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    cache = tmp_path / "cache"
    report = collect_doctor_report(
        overrides=[f"paths.output_dir={output.as_posix()}", f"paths.cache_dir={cache.as_posix()}"]
    )
    assert report["configuration"]["loaded"]
    assert report["paths"]["output"]["writable"]
    assert report["paths"]["cache"]["writable"]
    assert isinstance(report["torch"]["cuda_available"], bool)
    assert doctor_succeeded(report)


def test_doctor_fails_closed_for_missing_enabled_ifeval_checkout(tmp_path: Path) -> None:
    report = collect_doctor_report(
        overrides=[
            "general_capabilities.enabled=true",
            (f"general_capabilities.ifeval_scorer_repository={(tmp_path / 'missing').as_posix()}"),
        ]
    )
    assert report["configuration"]["loaded"]
    assert report["general_capabilities"]["enabled"]
    assert not report["general_capabilities"]["ready"]
    assert not doctor_succeeded(report)


def test_cli_help_and_doctor_complete() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "rahc_lora.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_result.returncode == 0
    assert "doctor" in help_result.stdout
    assert "prepare-data" in help_result.stdout
    assert "train" in help_result.stdout

    doctor_result = subprocess.run(
        [sys.executable, "-m", "rahc_lora.cli", "doctor"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert doctor_result.returncode == 0, doctor_result.stderr
    report = json.loads(doctor_result.stdout)
    assert report["configuration"]["loaded"]
