from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from rahc_lora.config import compose_config
from rahc_lora.config.schema import RootConfig
from rahc_lora.evaluation.capability_suite import GeneralCapabilitySuite
from rahc_lora.evaluation.evaluator import EvaluationGenerationConfig
from rahc_lora.evaluation.general_capabilities import (
    IFEvalTask,
    MultipleChoiceCapabilityTask,
    PerplexityCapability,
)
from rahc_lora.models.factory import build_policy
from rahc_lora.models.policy import TorchCausalPolicy
from rahc_lora.rewards.base import RewardResult
from rahc_lora.tasks.base import (
    BaseContinualTask,
    DatasetRole,
    TaskExample,
    TaskSplits,
)
from rahc_lora.training.artifacts import RunArtifacts
from rahc_lora.training.trainer import ContinualTrainer, TrainingResult
from rahc_lora.utils.reproducibility import seed_everything


class SyntheticRewardTask(BaseContinualTask):
    def __init__(self, task_id: str, offset: int) -> None:
        def examples(role: DatasetRole, count: int) -> tuple[TaskExample, ...]:
            return tuple(
                TaskExample(
                    task_id=task_id,
                    example_id=f"{task_id}:{role.value}:{index}",
                    role=role,
                    fields={"prompt": f"{task_id}-{role.value}-{index}"},
                )
                for index in range(count)
            )

        super().__init__(
            task_id,
            TaskSplits(
                train=examples(DatasetRole.TRAIN, 2),
                validation=examples(DatasetRole.VALIDATION, 1),
                anchor_candidates=examples(DatasetRole.ANCHOR_CANDIDATE, 1),
                test=examples(DatasetRole.TEST, 1),
            ),
        )
        self.offset = offset

    def format_prompt(self, example: TaskExample) -> str:
        return str(example.fields["prompt"])

    def evaluate_response(self, example: TaskExample, response: str) -> RewardResult:
        del example
        total = sum(ord(character) for character in response)
        reward = ((total + self.offset) % 251) / 250.0
        return RewardResult(
            raw_reward=reward,
            normalized_reward=reward,
            parse_succeeded=True,
            parsed_response=response,
            reward_version="synthetic-v1",
        )


class SyntheticIFEvalScorer:
    version = "synthetic-official-v1"

    def score(self, example: TaskExample, response: str) -> tuple[float, float]:
        del example
        score = float(bool(response))
        return score, score


def _configuration(tmp_path: Path) -> tuple[DictConfig, RootConfig]:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "synthetic.json").write_text("{}\n", encoding="utf-8")
    resolved = compose_config(
        overrides=[
            "model=tiny",
            "method=rl_lora_tiny",
            f"paths.output_dir={tmp_path.as_posix()}/outputs",
            f"data.manifest_dir={manifest_dir.as_posix()}",
            "launcher.device=cpu",
            "experiment.max_steps_per_task=1",
            "experiment.task_limit=2",
            "experiment.evaluation_max_examples_per_task=1",
            "rollout.prompts_per_batch=1",
            "rollout.group_size=4",
            "rollout.max_prompt_tokens=16",
            "rollout.max_response_tokens=2",
            "optimizer.learning_rate=0.05",
            "general_capabilities.enabled=true",
            "general_capabilities.ifeval_scorer_repository=synthetic-checkout",
            "general_capabilities.max_perplexity_tokens=16",
            "general_capabilities.max_new_tokens=2",
        ]
    )
    typed = OmegaConf.to_object(resolved)
    assert isinstance(typed, RootConfig)
    return resolved, typed


def _tasks() -> tuple[SyntheticRewardTask, SyntheticRewardTask]:
    return (
        SyntheticRewardTask("gsm8k", 0),
        SyntheticRewardTask("arc_challenge", 37),
    )


def _capability_example(task_id: str, fields: dict[str, object]) -> TaskExample:
    return TaskExample(
        task_id=task_id,
        example_id=f"{task_id}:capability:0",
        role=DatasetRole.EVALUATION_ONLY,
        fields=fields,
    )


def _capability_suite(config: RootConfig) -> GeneralCapabilitySuite:
    return GeneralCapabilitySuite(
        ifeval=IFEvalTask(
            (_capability_example("ifeval", {"prompt": "follow this instruction"}),),
            SyntheticIFEvalScorer(),
        ),
        mmlu=MultipleChoiceCapabilityTask(
            "mmlu",
            (
                _capability_example(
                    "mmlu",
                    {"question": "choose", "choices": ["x", "y"], "answer": 1},
                ),
            ),
        ),
        hellaswag=MultipleChoiceCapabilityTask(
            "hellaswag",
            (
                _capability_example(
                    "hellaswag",
                    {"ctx": "continue", "endings": ["x", "y"], "label": "1"},
                ),
            ),
        ),
        arc_easy=MultipleChoiceCapabilityTask(
            "arc_easy",
            (
                _capability_example(
                    "arc_easy",
                    {
                        "question": "choose",
                        "choices": {"label": ["A", "B"], "text": ["x", "y"]},
                        "answerKey": "B",
                    },
                ),
            ),
        ),
        perplexity=PerplexityCapability(
            (_capability_example("perplexity", {"text": "held out text"}),)
        ),
        generation=EvaluationGenerationConfig(
            max_new_tokens=config.general_capabilities.max_new_tokens,
            do_sample=config.general_capabilities.do_sample,
            temperature=config.general_capabilities.temperature,
            top_p=config.general_capabilities.top_p,
        ),
        max_perplexity_tokens=config.general_capabilities.max_perplexity_tokens,
    )


def _new_trainer(
    resolved: DictConfig,
    config: RootConfig,
    tasks: Sequence[SyntheticRewardTask],
    *,
    run_id: str,
) -> tuple[RunArtifacts, ContinualTrainer, TorchCausalPolicy]:
    seed_everything(config.seed, deterministic_algorithms=True)
    policy = build_policy(
        config.model,
        config.method.lora,
        requested_device="cpu",
    ).policy
    artifacts = RunArtifacts(
        resolved,
        config,
        run_id=run_id,
        repository=Path.cwd(),
    )
    trainer = ContinualTrainer(
        policy=policy,
        tasks=tasks,
        resolved_config=resolved,
        config=config,
        artifacts=artifacts,
        capability_suite=_capability_suite(config),
    )
    return artifacts, trainer, policy


def _finish(
    artifacts: RunArtifacts, trainer: ContinualTrainer
) -> tuple[TrainingResult, dict[str, torch.Tensor]]:
    try:
        result = trainer.train()
        return result, trainer.policy.adapter_state_dict()
    finally:
        artifacts.close()


def test_two_task_training_updates_only_lora_and_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    resolved, config = _configuration(tmp_path)
    tasks = _tasks()

    full_artifacts, full_trainer, full_policy = _new_trainer(resolved, config, tasks, run_id="full")
    frozen_before = full_policy.frozen_parameter_state()
    adapter_before = full_policy.adapter_state_dict()
    full_result, full_adapter = _finish(full_artifacts, full_trainer)
    assert full_result.matrix.completed_rows == 2
    assert full_result.progress.global_step == 2
    assert full_result.latest_checkpoint is not None
    metric_rows = [
        json.loads(line)
        for line in (full_artifacts.run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    update_rows = [row for row in metric_rows if row["event"] == "grpo_update"]
    assert len(update_rows) == 2
    assert all(
        row["values"]["reward_diagnostics"]
        == {
            "parse_failure_count": 0,
            "code_execution_count": 0,
            "code_tests_passed": 0,
            "code_tests_total": 0,
            "code_timeout_count": 0,
            "code_output_limited_count": 0,
            "code_nonzero_exit_count": 0,
        }
        for row in update_rows
    )
    assert full_policy.frozen_parameter_state().keys() == frozen_before.keys()
    for name, value in frozen_before.items():
        assert torch.equal(full_policy.frozen_parameter_state()[name], value)
    assert any(not torch.equal(adapter_before[name], full_adapter[name]) for name in full_adapter)

    partial_artifacts, partial_trainer, _ = _new_trainer(resolved, config, tasks, run_id="partial")
    try:
        partial = partial_trainer.train(max_task_boundaries=1)
        assert partial.latest_checkpoint is not None
        checkpoint = partial.latest_checkpoint
    finally:
        partial_artifacts.close()

    resumed_artifacts, resumed_trainer, _ = _new_trainer(resolved, config, tasks, run_id="resumed")
    try:
        resumed_trainer.resume_from(checkpoint)
        resumed_result = resumed_trainer.train()
        resumed_adapter = resumed_trainer.policy.adapter_state_dict()
    finally:
        resumed_artifacts.close()

    assert resumed_result.progress == full_result.progress
    assert resumed_result.matrix.values == full_result.matrix.values
    for name in full_adapter:
        assert torch.allclose(
            resumed_adapter[name],
            full_adapter[name],
            rtol=0.0,
            atol=config.checkpoint.numeric_resume_tolerance,
        )

    required_artifacts = {
        "resolved_config.yaml",
        "run_manifest.json",
        "metrics.jsonl",
        "system_metrics.jsonl",
        "performance_matrix.csv",
        "general_capabilities.csv",
        "general_capabilities_status.json",
        "layer_importance.parquet",
        "layer_conflicts.parquet",
        "memory_manifest.json",
        "checkpoint_manifest.json",
        "stdout.log",
        "summary.json",
    }
    assert required_artifacts.issubset({path.name for path in resumed_artifacts.run_dir.iterdir()})
    assert resumed_result.matrix.values[0][0] == pytest.approx(full_result.matrix.values[0][0])
    full_capabilities = (full_artifacts.run_dir / "general_capabilities.csv").read_text(
        encoding="utf-8"
    )
    resumed_capabilities = (resumed_artifacts.run_dir / "general_capabilities.csv").read_text(
        encoding="utf-8"
    )
    assert resumed_capabilities == full_capabilities
    assert len(full_capabilities.splitlines()) == 1 + 6 * 3
    assert (checkpoint / "capability_state.json").is_file()
