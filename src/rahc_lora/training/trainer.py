"""One sequential trainer shared by the standard RL-LoRA baseline and later methods."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch
from omegaconf import DictConfig
from torch import Tensor

from rahc_lora.consolidation.hlora import DensePathImportance
from rahc_lora.config.schema import RootConfig
from rahc_lora.evaluation.capability_suite import (
    CAPABILITY_METRICS,
    CapabilityMetricResult,
    GeneralCapabilitySuite,
)
from rahc_lora.evaluation.continual_metrics import summarize_continual_metrics
from rahc_lora.evaluation.evaluator import (
    EvaluationGenerationConfig,
    evaluate_task_boundary,
)
from rahc_lora.evaluation.performance_matrix import PerformanceMatrix
from rahc_lora.models.policy import (
    PolicyGenerationConfig,
    TrainableCausalPolicy,
    old_logprobs_tensor,
)
from rahc_lora.rl.advantages import normalize_group_advantages
from rahc_lora.rl.losses import clipped_grpo_loss
from rahc_lora.rl.rollout import (
    RewardDiagnostics,
    RolloutBatch,
    RolloutRecord,
    SampledResponse,
)
from rahc_lora.tasks.base import ContinualTask, TaskExample
from rahc_lora.training.artifacts import RunArtifacts
from rahc_lora.training.checkpoint import CheckpointManager, TrainingProgress


@dataclass(frozen=True)
class TrainingResult:
    """Current progress, matrix, and latest durable checkpoint."""

    progress: TrainingProgress
    matrix: PerformanceMatrix
    latest_checkpoint: Path | None


class ContinualTrainer:
    """Train one shared LoRA policy sequentially without hidden retention logic."""

    def __init__(
        self,
        *,
        policy: TrainableCausalPolicy,
        tasks: Sequence[ContinualTask],
        resolved_config: DictConfig,
        config: RootConfig,
        artifacts: RunArtifacts,
        capability_suite: GeneralCapabilitySuite | None = None,
    ) -> None:
        if config.method.name not in {"rl_lora", "hlora_rl"}:
            raise ValueError(
                "ContinualTrainer currently supports method=rl_lora or method=hlora_rl"
            )
        expected_ids = tuple(config.task_stream.tasks[: config.experiment.task_limit])
        if tuple(task.task_id for task in tasks) != expected_ids:
            raise ValueError(
                "Trainer tasks do not match configured stream prefix: "
                f"expected={expected_ids}, actual={tuple(task.task_id for task in tasks)}"
            )
        capability_enabled = config.general_capabilities.enabled
        if capability_enabled != (capability_suite is not None):
            raise ValueError(
                "Capability suite presence must exactly match general_capabilities.enabled"
            )
        self.policy = policy
        self.tasks = tuple(tasks)
        self.resolved_config = resolved_config
        self.config = config
        self.artifacts = artifacts
        self.capability_suite = capability_suite
        self.dense_path_importance = (
            DensePathImportance(epsilon=config.method.parameter_importance_epsilon)
            if config.method.name == "hlora_rl" and config.method.use_parameter_consolidation
            else None
        )
        self.progress = TrainingProgress()
        self.matrix = PerformanceMatrix(expected_ids)
        optimizer_config = config.optimizer
        self.optimizer = torch.optim.AdamW(
            policy.trainable_parameters(),
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.betas[0], optimizer_config.betas[1]),
            eps=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        generator_device = policy.device.type if policy.device.type in {"cpu", "cuda"} else "cpu"
        self.rollout_generator = torch.Generator(device=generator_device)
        self.rollout_generator.manual_seed(config.seed)
        self.checkpoints = CheckpointManager(
            run_dir=artifacts.run_dir,
            resolved_config=resolved_config,
            typed_config=config,
            dataset_fingerprints=artifacts.dataset_fingerprints,
        )
        self.latest_checkpoint: Path | None = None

    def resume_from(self, checkpoint: Path) -> None:
        """Restore a validated task-boundary checkpoint into this trainer."""

        if self.progress.next_task_index or self.progress.global_step:
            raise RuntimeError("Cannot resume a trainer that has already made progress")
        loaded = self.checkpoints.load(
            checkpoint,
            policy=self.policy,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            rollout_generator=self.rollout_generator,
        )
        if loaded.matrix.task_ids != self.matrix.task_ids:
            raise ValueError("Checkpoint performance-matrix task order is incompatible")
        capability_enabled = self.config.general_capabilities.enabled
        if loaded.capability_state.get("enabled") != capability_enabled:
            raise ValueError("Checkpoint capability enablement disagrees with configuration")
        if capability_enabled:
            if set(loaded.capability_state) != {"enabled", "suite"}:
                raise ValueError("Enabled checkpoint capability state has invalid fields")
            suite_state = loaded.capability_state["suite"]
            if not isinstance(suite_state, dict) or self.capability_suite is None:
                raise ValueError("Enabled checkpoint capability suite state is invalid")
            self.capability_suite.load_state_dict(suite_state)
            history = self.capability_suite.history_results()
            expected_history_rows = (loaded.progress.next_task_index + 1) * len(CAPABILITY_METRICS)
            expected_boundary = (
                "frozen_start"
                if loaded.progress.next_task_index == 0
                else self.tasks[loaded.progress.next_task_index - 1].task_id
            )
            latest = self.capability_suite.latest_results()
            if (
                len(history) != expected_history_rows
                or latest is None
                or latest[0].after_task != expected_boundary
            ):
                raise ValueError("Checkpoint capability history disagrees with task progress")
            self.artifacts.append_capabilities(history)
            self.artifacts.metrics.log(
                category="reproducibility",
                event="general_capability_state_restored",
                step=loaded.progress.global_step,
                values={
                    "history_rows": len(history),
                    "last_after_task": suite_state["last_after_task"],
                },
            )
        elif loaded.capability_state != {"enabled": False}:
            raise ValueError("Disabled checkpoint capability state must be explicit and empty")
        if self.dense_path_importance is not None:
            dense_state = loaded.retention_state.get("dense_path_importance")
            if not isinstance(dense_state, dict):
                raise ValueError("HLoRA resume checkpoint is missing dense path-importance state")
            current_effective = self.policy.named_effective_lora_weights(detach=True)
            current_factors = self.policy.named_trainable_parameters()
            self.dense_path_importance.load_state_dict(
                dense_state,
                expected_effective_shapes={
                    name: tuple(value.shape) for name, value in current_effective.items()
                },
                expected_factor_shapes={
                    name: tuple(value.shape) for name, value in current_factors.items()
                },
            )
        self.progress = loaded.progress
        self.matrix = loaded.matrix
        self.artifacts.write_matrix(self.matrix)

    def _record_capabilities(
        self,
        event: str,
        results: tuple[CapabilityMetricResult, ...],
    ) -> None:
        self.artifacts.append_capabilities(results)
        self.artifacts.metrics.log(
            category="retention",
            event=event,
            step=self.progress.global_step,
            values={
                "after_task": results[0].after_task,
                "metrics": {
                    f"{result.capability}.{result.metric}": {
                        "absolute_score": result.absolute_score,
                        "frozen_baseline_score": result.frozen_baseline_score,
                        "change_from_frozen": result.change_from_frozen,
                        "higher_is_better": result.higher_is_better,
                        "example_count": result.example_count,
                    }
                    for result in results
                },
            },
        )

    def _capability_state(self) -> dict[str, object]:
        if self.capability_suite is None:
            return {"enabled": False}
        if not self.capability_suite.has_frozen_baseline:
            raise RuntimeError("Cannot checkpoint capabilities before baseline capture")
        return {"enabled": True, "suite": self.capability_suite.state_dict()}

    def _retention_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "method": self.config.method.name,
            "reference_lora_factors": {},
            "factorized_importance": {},
            "dual_controller": {},
            "anchor_memory": {"enabled": False, "capacity": 0, "records": []},
        }
        if self.config.method.name == "hlora_rl":
            state["dense_path_importance"] = (
                self.dense_path_importance.state_dict()
                if self.dense_path_importance is not None
                else None
            )
        return state

    def _examples_for_step(
        self, examples: Sequence[TaskExample], local_step: int
    ) -> tuple[TaskExample, ...]:
        if not examples:
            raise ValueError("Continual task has no training examples")
        count = self.config.rollout.prompts_per_batch
        start = (local_step * count) % len(examples)
        return tuple(examples[(start + offset) % len(examples)] for offset in range(count))

    def _rollout(
        self,
        task: ContinualTask,
        examples: tuple[TaskExample, ...],
    ) -> tuple[RolloutBatch, tuple[SampledResponse, ...], Tensor, float]:
        rollout_config = self.config.rollout
        prompts = tuple(task.format_prompt(example) for example in examples)
        samples = self.policy.sample(
            prompts,
            group_size=rollout_config.group_size,
            generation=PolicyGenerationConfig(
                max_prompt_tokens=rollout_config.max_prompt_tokens,
                max_response_tokens=rollout_config.max_response_tokens,
                do_sample=rollout_config.do_sample,
                temperature=rollout_config.temperature,
                top_p=rollout_config.top_p,
            ),
            generator=self.rollout_generator,
        )
        reward_results = tuple(
            task.evaluate_response(
                examples[index // rollout_config.group_size], sample.response_text
            )
            for index, sample in enumerate(samples)
        )
        reward_diagnostics = RewardDiagnostics.from_results(reward_results)
        normalized_rewards = torch.tensor(
            [result.normalized_reward for result in reward_results],
            dtype=torch.float32,
            device=self.policy.device,
        ).reshape(len(examples), rollout_config.group_size)
        advantages = normalize_group_advantages(
            normalized_rewards,
            epsilon=rollout_config.advantage_epsilon,
            clip=rollout_config.advantage_clip,
        )
        flat_advantages = advantages.values_gk.reshape(-1)
        records = tuple(
            RolloutRecord(
                task_id=task.task_id,
                example_id=examples[index // rollout_config.group_size].example_id,
                prompt_group=index // rollout_config.group_size,
                sample_index=index % rollout_config.group_size,
                prompt_token_ids=sample.prompt_token_ids,
                response_token_ids=sample.response_token_ids,
                response_text=sample.response_text,
                old_policy_logprobs=sample.old_policy_logprobs,
                sampling_metadata=sample.sampling_metadata,
                raw_reward=reward_results[index].raw_reward,
                normalized_reward=reward_results[index].normalized_reward,
                advantage=float(flat_advantages[index].item()),
                valid_token_mask=(True,) * len(sample.response_token_ids),
                policy_version=self.policy.policy_version,
            )
            for index, sample in enumerate(samples)
        )
        batch = RolloutBatch(
            records=records,
            group_size=rollout_config.group_size,
            zero_variance_groups=advantages.zero_variance_groups,
            reward_versions=tuple(result.reward_version for result in reward_results),
            reward_diagnostics=reward_diagnostics,
        )
        return batch, samples, flat_advantages, advantages.clipped_fraction

    def _optimizer_update(
        self,
        batch: RolloutBatch,
        samples: tuple[SampledResponse, ...],
        advantages: Tensor,
    ) -> tuple[float, float, float, float, bool]:
        self.optimizer.zero_grad(set_to_none=True)
        capture_context = (
            self.policy.capture_effective_weight_gradients()
            if self.dense_path_importance is not None
            else nullcontext()
        )
        with capture_context as capture:
            new_logprobs, current_mask = self.policy.response_logprobs(samples)
            old_logprobs, old_mask = old_logprobs_tensor(samples, device=self.policy.device)
            if not torch.equal(current_mask, old_mask):
                raise ValueError("Current and stored rollout token masks disagree")
            loss_result = clipped_grpo_loss(
                new_logprobs,
                old_logprobs,
                advantages,
                current_mask,
                clip_epsilon=self.config.rollout.policy_clip_epsilon,
            )
            regularization_loss = loss_result.loss.new_zeros(())
            effective_before: dict[str, Tensor] | None = None
            effective_gradients: dict[str, Tensor] | None = None
            objective = loss_result.loss
            if self.dense_path_importance is not None:
                current_effective = self.policy.named_effective_lora_weights(
                    detach=not bool(self.dense_path_importance.importance)
                )
                effective_before = {
                    name: value.detach().clone() for name, value in current_effective.items()
                }
                if self.dense_path_importance.importance:
                    reference_effective = self.policy.named_effective_lora_weights(
                        adapter_state=self.dense_path_importance.reference_lora_factors,
                        detach=True,
                    )
                    regularization_loss = self.dense_path_importance.penalty_with_reference(
                        current_effective, reference_effective
                    )
                    objective = (
                        loss_result.loss
                        + self.config.method.parameter_regularization_lambda * regularization_loss
                    )
            objective.backward()
            if self.dense_path_importance is not None:
                assert capture is not None
                effective_gradients = capture.gradients()
        gradients = [
            parameter.grad
            for parameter in self.policy.trainable_parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("GRPO update produced no LoRA gradients")
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError(f"Non-finite LoRA gradient at global step {self.progress.global_step}")
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.policy.trainable_parameters(), self.config.optimizer.max_grad_norm
        )
        grad_norm = float(grad_norm_tensor.item())
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError(f"Non-finite gradient norm at step {self.progress.global_step}")
        clipped = grad_norm > self.config.optimizer.max_grad_norm
        self.optimizer.step()
        self.scheduler.step()
        self.policy.increment_policy_version()
        if self.dense_path_importance is not None:
            if effective_before is None or effective_gradients is None:
                raise RuntimeError("HLoRA update did not capture its effective-weight path")
            effective_after = self.policy.named_effective_lora_weights(detach=True)
            path_diagnostics = self.dense_path_importance.observe_step(
                effective_gradients=effective_gradients,
                effective_before=effective_before,
                effective_after=effective_after,
            )
            self.artifacts.metrics.log(
                category="retention",
                event="hlora_path_integral_update",
                step=self.progress.global_step,
                values={
                    "task_id": batch.records[0].task_id,
                    "rl_loss": float(loss_result.loss.detach().item()),
                    "regularization_loss": float(regularization_loss.detach().item()),
                    "regularization_lambda": self.config.method.parameter_regularization_lambda,
                    "combined_objective": float(objective.detach().item()),
                    "effective_layers": path_diagnostics,
                },
            )
        return (
            float(loss_result.loss.detach().item()),
            loss_result.sampled_policy_kl,
            loss_result.ratio_clip_fraction,
            grad_norm,
            clipped,
        )

    def _train_task(self, task: ContinualTask) -> None:
        examples = tuple(task.load_train())
        for local_step in range(self.config.experiment.max_steps_per_task):
            selected = self._examples_for_step(examples, local_step)
            step_start = time.perf_counter()
            batch, samples, advantages, advantage_clip_fraction = self._rollout(task, selected)
            loss, policy_kl, ratio_clip_fraction, grad_norm, grad_clipped = self._optimizer_update(
                batch, samples, advantages
            )
            self.progress.global_step += 1
            self.progress.rollout_counter += len(samples)
            elapsed = time.perf_counter() - step_start
            rewards = [record.normalized_reward for record in batch.records]
            raw_rewards = [record.raw_reward for record in batch.records]
            response_lengths = [len(record.response_token_ids) for record in batch.records]
            distinct_responses = len({record.response_text for record in batch.records})
            self.artifacts.metrics.log(
                category="learning",
                event="grpo_update",
                step=self.progress.global_step,
                values={
                    "task_id": task.task_id,
                    "task_step": local_step,
                    "policy_loss": loss,
                    "sampled_policy_kl": policy_kl,
                    "raw_reward_mean": sum(raw_rewards) / len(raw_rewards),
                    "normalized_reward_mean": sum(rewards) / len(rewards),
                    "reward_diagnostics": batch.reward_diagnostics.as_dict(),
                    "normalized_reward_min": min(rewards),
                    "normalized_reward_max": max(rewards),
                    "advantage_mean": float(advantages.mean().item()),
                    "advantage_std": float(advantages.std(correction=0).item()),
                    "advantage_clip_fraction": advantage_clip_fraction,
                    "zero_variance_groups": batch.zero_variance_groups,
                    "ratio_clip_fraction": ratio_clip_fraction,
                    "gradient_norm_before_clip": grad_norm,
                    "gradient_clipped": grad_clipped,
                    "response_tokens": batch.response_token_count,
                    "response_length_mean": sum(response_lengths) / len(response_lengths),
                    "response_distinct_fraction": distinct_responses / len(batch.records),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                },
            )
            process_rss = psutil.Process().memory_info().rss
            cuda_peak = (
                int(torch.cuda.max_memory_allocated(self.policy.device))
                if self.policy.device.type == "cuda"
                else 0
            )
            self.artifacts.system_metrics.log(
                category="system",
                event="optimizer_step",
                step=self.progress.global_step,
                values={
                    "task_id": task.task_id,
                    "step_seconds": elapsed,
                    "tokens_per_second": batch.response_token_count / elapsed,
                    "cpu_rss_bytes": process_rss,
                    "gpu_peak_allocated_bytes": cuda_peak,
                },
            )

    def _evaluate_boundary(self, task_index: int) -> None:
        generation_config = self.config.general_capabilities
        maximum = self.config.experiment.evaluation_max_examples_per_task
        candidate_matrix = self.matrix.clone()
        results = evaluate_task_boundary(
            after_task_id=self.tasks[task_index].task_id,
            learned_tasks=self.tasks[: task_index + 1],
            policy=self.policy,
            generation=EvaluationGenerationConfig(
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
            ),
            matrix=candidate_matrix,
            max_examples_per_task=maximum or None,
        )
        capability_results = (
            self.capability_suite.evaluate_boundary(
                self.tasks[task_index].task_id,
                self.policy,
            )
            if self.capability_suite is not None
            else None
        )
        self.matrix = candidate_matrix
        self.artifacts.append_evaluations(self.tasks[task_index].task_id, results)
        self.artifacts.write_matrix(self.matrix)
        self.artifacts.metrics.log(
            category="learning",
            event="task_boundary_evaluation",
            step=self.progress.global_step,
            values={
                "after_task": self.tasks[task_index].task_id,
                "scores": {result.task_id: result.mean_reward for result in results},
                "evaluated_examples": {result.task_id: len(result.records) for result in results},
            },
        )
        if capability_results is not None:
            self._record_capabilities("general_capability_boundary_evaluation", capability_results)

    def _save_checkpoint(self) -> None:
        checkpoint, checksum = self.checkpoints.save(
            policy=self.policy,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            progress=self.progress,
            matrix=self.matrix,
            rollout_generator=self.rollout_generator,
            capability_state=self._capability_state(),
            retention_state=self._retention_state(),
        )
        self.latest_checkpoint = checkpoint
        self.artifacts.register_checkpoint(checkpoint, checksum)

    def train(self, *, max_task_boundaries: int | None = None) -> TrainingResult:
        """Continue through configured tasks or stop after an operational boundary count."""

        if max_task_boundaries is not None and max_task_boundaries <= 0:
            raise ValueError("max_task_boundaries must be positive when provided")
        if self.capability_suite is not None and not self.capability_suite.has_frozen_baseline:
            baseline = self.capability_suite.capture_frozen_baseline(self.policy)
            self._record_capabilities("general_capability_frozen_baseline", baseline)
        invocation_boundaries = 0
        while self.progress.next_task_index < len(self.tasks):
            if max_task_boundaries is not None and invocation_boundaries >= max_task_boundaries:
                break
            task_index = self.progress.next_task_index
            if self.dense_path_importance is not None:
                self.dense_path_importance.begin_task(
                    self.policy.named_effective_lora_weights(detach=True)
                )
            self._train_task(self.tasks[task_index])
            if self.dense_path_importance is not None:
                diagnostics = self.dense_path_importance.consolidate(
                    final_effective_weights=self.policy.named_effective_lora_weights(detach=True),
                    reference_lora_factors=self.policy.adapter_state_dict(),
                )
                self.artifacts.metrics.log(
                    category="retention",
                    event="hlora_importance_consolidated",
                    step=self.progress.global_step,
                    values={"after_task": self.tasks[task_index].task_id, **asdict(diagnostics)},
                )
            self._evaluate_boundary(task_index)
            self.progress.next_task_index += 1
            invocation_boundaries += 1
            if self.config.checkpoint.enabled and (
                self.progress.next_task_index % self.config.checkpoint.save_every_tasks == 0
                or self.progress.next_task_index == len(self.tasks)
            ):
                self._save_checkpoint()
        if self.progress.next_task_index == len(self.tasks):
            summary = summarize_continual_metrics(self.matrix)
            self.artifacts.write_summary(
                {
                    "method": self.config.method.name,
                    "task_order": list(self.matrix.task_ids),
                    "seed": self.config.seed,
                    "global_step": self.progress.global_step,
                    "rollout_counter": self.progress.rollout_counter,
                    "continual_metrics": asdict(summary),
                    "general_capabilities": {
                        "enabled": self.capability_suite is not None,
                        "latest": (
                            [
                                asdict(result)
                                for result in self.capability_suite.latest_results() or ()
                            ]
                            if self.capability_suite is not None
                            else []
                        ),
                    },
                    "experimental_success_claimed": False,
                }
            )
        return TrainingResult(
            progress=self.progress,
            matrix=self.matrix,
            latest_checkpoint=self.latest_checkpoint,
        )
