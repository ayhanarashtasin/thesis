"""Single command-line interface for RAHC-LoRA workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from rahc_lora import __version__
from rahc_lora.config import compose_config
from rahc_lora.config.schema import RootConfig
from rahc_lora.doctor import collect_doctor_report, doctor_succeeded
from rahc_lora.evaluation.capability_suite import build_general_capability_suite
from rahc_lora.evaluation.official_ifeval import GoogleResearchIFEvalScorer
from rahc_lora.models.factory import build_policy
from rahc_lora.rewards.code_tests import DockerSandboxRunner
from rahc_lora.tasks.data import HuggingFaceDatasetProvider
from rahc_lora.tasks.preparation import prepare_data
from rahc_lora.tasks.registry import TaskBuildContext, build_task_catalog
from rahc_lora.training.artifacts import RunArtifacts
from rahc_lora.training.trainer import ContinualTrainer
from rahc_lora.utils.manifests import canonical_config_hash
from rahc_lora.utils.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""

    parser = argparse.ArgumentParser(
        prog="rahc-lora",
        description="RAHC-LoRA continual reinforcement learning research tools",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor",
        help="report environment, dependency, hardware, path, and configuration readiness",
    )
    doctor.add_argument(
        "--config-name",
        default="config",
        help="Hydra root configuration name (default: config)",
    )
    doctor.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides such as method=rahc_lora or experiment=smoke",
    )
    prepare = commands.add_parser(
        "prepare-data",
        help="write pinned split manifests and verify custom/IFEval prompt isolation",
    )
    prepare.add_argument(
        "--config-name",
        default="config",
        help="Hydra root configuration name (default: config)",
    )
    prepare.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides such as data.prepare_general_capabilities=false",
    )
    train = commands.add_parser(
        "train",
        help="train or resume one validated continual RL-LoRA run",
    )
    train.add_argument(
        "--config-name",
        default="config",
        help="Hydra root configuration name (default: config)",
    )
    train.add_argument(
        "--run-id",
        default=None,
        help="artifact run ID; defaults to UTC timestamp plus config hash",
    )
    train.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="absolute or relative path to a validated checkpoint directory",
    )
    train.add_argument(
        "--max-task-boundaries",
        type=int,
        default=None,
        help="operational interruption hook used to stop after N new task boundaries",
    )
    train.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides such as experiment=smoke method=rl_lora seed=1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return a process exit code."""

    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        report = collect_doctor_report(
            config_name=arguments.config_name,
            overrides=arguments.overrides,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if doctor_succeeded(report) else 1
    if arguments.command == "prepare-data":
        resolved = compose_config(
            config_name=arguments.config_name,
            overrides=arguments.overrides,
        )
        config = cast(RootConfig, OmegaConf.to_object(resolved))
        data_result = prepare_data(config)
        print(json.dumps(asdict(data_result), indent=2, sort_keys=True))
        return 0
    if arguments.command == "train":
        resolved = compose_config(
            config_name=arguments.config_name,
            overrides=arguments.overrides,
        )
        config = cast(RootConfig, OmegaConf.to_object(resolved))
        if config.method.name not in {"rl_lora", "hlora_rl", "rahc_lora"}:
            raise ValueError(
                f"Unsupported method: {config.method.name}; expected rl_lora, hlora_rl, or rahc_lora"
            )
        seed_everything(config.seed, deterministic_algorithms=config.experiment.deterministic)
        bundle = build_policy(
            config.model,
            config.method.lora,
            requested_device=config.launcher.device,
            cache_dir=config.paths.cache_dir,
        )
        provider = HuggingFaceDatasetProvider(cache_dir=config.paths.cache_dir)
        context = TaskBuildContext(
            provider=provider,
            constraint_generator=config.constraint_generator,
            code_sandbox=DockerSandboxRunner(config.code_sandbox),
        )
        catalog = build_task_catalog(config.tasks, context)
        tasks = tuple(
            catalog[task_id] for task_id in config.task_stream.tasks[: config.experiment.task_limit]
        )
        capability_suite = None
        if config.general_capabilities.enabled:
            scorer_repository = config.general_capabilities.ifeval_scorer_repository
            if scorer_repository is None:
                raise ValueError(
                    "Enabled capability evaluation requires an official IFEval repository"
                )
            scorer = GoogleResearchIFEvalScorer(
                Path(scorer_repository),
                expected_revision=config.general_capabilities.ifeval_scorer_revision,
                deterministic_seed=config.general_capabilities.ifeval_scorer_seed,
                nltk_data_root=Path(config.general_capabilities.ifeval_nltk_data_dir),
                expected_punkt_tab_sha256=(config.general_capabilities.ifeval_punkt_tab_sha256),
            )
            capability_suite = build_general_capability_suite(
                config.general_capabilities,
                provider,
                scorer,
            )
        run_id = arguments.run_id or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + canonical_config_hash(resolved)[:12]
        )
        with RunArtifacts(
            resolved,
            config,
            run_id=run_id,
            repository=Path.cwd(),
            command_overrides=arguments.overrides,
        ) as artifacts:
            trainer = ContinualTrainer(
                policy=bundle.policy,
                tasks=tasks,
                resolved_config=resolved,
                config=config,
                artifacts=artifacts,
                capability_suite=capability_suite,
            )
            if arguments.resume is not None:
                trainer.resume_from(arguments.resume)
            training_result = trainer.train(max_task_boundaries=arguments.max_task_boundaries)
            output = {
                "run_dir": str(artifacts.run_dir),
                "next_task_index": training_result.progress.next_task_index,
                "global_step": training_result.progress.global_step,
                "rollout_counter": training_result.progress.rollout_counter,
                "latest_checkpoint": (
                    str(training_result.latest_checkpoint)
                    if training_result.latest_checkpoint is not None
                    else None
                ),
                "performance_matrix": training_result.matrix.values,
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
