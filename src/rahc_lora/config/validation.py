"""Cross-field validation for resolved RAHC-LoRA configurations."""

from __future__ import annotations

import math
import re

from omegaconf import DictConfig


class ConfigurationError(ValueError):
    """Raised when a scientific configuration is invalid or inconsistent."""


def _require(condition: bool, field: str, message: str) -> None:
    if not condition:
        raise ConfigurationError(f"Invalid configuration at '{field}': {message}")


def _finite(value: float, field: str) -> None:
    _require(math.isfinite(value), field, f"expected a finite value, received {value!r}")


def _validate_task_data(task: DictConfig, field: str, *, expected_evaluation_only: bool) -> None:
    _require(bool(task.task_id), f"{field}.task_id", "must not be empty")
    _require(bool(task.family), f"{field}.family", "must not be empty")
    _require(
        task.provider in {"huggingface", "generated"},
        f"{field}.provider",
        "must be huggingface or generated",
    )
    _require(bool(task.dataset_name), f"{field}.dataset_name", "must not be empty")
    _require(bool(task.revision), f"{field}.revision", "must not be empty")
    if task.provider == "huggingface":
        _require(
            re.fullmatch(r"[0-9a-f]{40}", task.revision) is not None,
            f"{field}.revision",
            "must be an immutable 40-character commit hash",
        )
    _require(bool(task.license), f"{field}.license", "must not be empty")
    _require(bool(task.citation), f"{field}.citation", "must not be empty")
    _require(bool(task.source_test_split), f"{field}.source_test_split", "must not be empty")
    _require(bool(task.id_fields), f"{field}.id_fields", "must not be empty")
    _require(task.split_seed >= 0, f"{field}.split_seed", "must be nonnegative")
    _require(
        0 <= task.validation_fraction < 1,
        f"{field}.validation_fraction",
        "must be in [0, 1)",
    )
    _require(
        0 <= task.anchor_fraction < 1,
        f"{field}.anchor_fraction",
        "must be in [0, 1)",
    )
    _require(
        task.validation_fraction + task.anchor_fraction < 1,
        field,
        "validation_fraction + anchor_fraction must be less than 1",
    )
    _require(
        bool(task.preprocessing_version),
        f"{field}.preprocessing_version",
        "must not be empty",
    )
    _require(bool(task.prompt_version), f"{field}.prompt_version", "must not be empty")
    _require(bool(task.reward_version), f"{field}.reward_version", "must not be empty")
    _require(
        task.evaluation_only is expected_evaluation_only,
        f"{field}.evaluation_only",
        f"must be {expected_evaluation_only}",
    )
    if expected_evaluation_only:
        _require(
            task.source_train_split is None,
            f"{field}.source_train_split",
            "must be null for evaluation-only data",
        )
        _require(
            task.source_validation_split is None,
            f"{field}.source_validation_split",
            "must be null for evaluation-only data",
        )
        _require(
            task.validation_fraction == 0 and task.anchor_fraction == 0,
            field,
            "evaluation-only data cannot allocate validation or anchor fractions",
        )
    else:
        _require(
            bool(task.source_train_split),
            f"{field}.source_train_split",
            "must not be empty for a trainable task",
        )
        if task.source_validation_split is not None:
            _require(
                task.validation_fraction == 0,
                f"{field}.validation_fraction",
                "must be zero when an official validation split is configured",
            )


def validate_config(config: DictConfig) -> None:
    """Reject invalid scientific choices before any run state is created."""

    _require(config.seed >= 0, "seed", "must be nonnegative")
    _require(bool(config.paths.output_dir), "paths.output_dir", "must not be empty")
    _require(bool(config.paths.cache_dir), "paths.cache_dir", "must not be empty")

    model = config.model
    _require(
        model.backend in {"transformers", "tiny"},
        "model.backend",
        "must be transformers or tiny",
    )
    _require(bool(model.identifier), "model.identifier", "must not be empty")
    _require(bool(model.revision), "model.revision", "must not be empty")
    _require(
        bool(model.tokenizer_identifier),
        "model.tokenizer_identifier",
        "must not be empty",
    )
    _require(
        bool(model.tokenizer_revision),
        "model.tokenizer_revision",
        "must not be empty",
    )
    if model.backend == "transformers":
        for field_name in ("revision", "tokenizer_revision"):
            _require(
                re.fullmatch(r"[0-9a-f]{40}", model[field_name]) is not None,
                f"model.{field_name}",
                "must be an immutable 40-character commit hash",
            )
        _require(
            not model.trust_remote_code,
            "model.trust_remote_code",
            "must remain false for the reference implementation",
        )
    _require(model.vocab_size > 0, "model.vocab_size", "must be positive")
    _require(model.hidden_size > 0, "model.hidden_size", "must be positive")
    _require(
        model.max_sequence_length > 1,
        "model.max_sequence_length",
        "must be greater than one",
    )
    _require(
        model.precision in {"float32", "float16", "bfloat16"},
        "model.precision",
        "must be one of float32, float16, or bfloat16",
    )

    expected_tasks = ("gsm8k", "arc_challenge", "mbpp", "constraints")
    for task_id in expected_tasks:
        task = config.tasks[task_id]
        _validate_task_data(task, f"tasks.{task_id}", expected_evaluation_only=False)
        _require(
            task.task_id == task_id,
            f"tasks.{task_id}.task_id",
            f"must equal {task_id!r}",
        )
    _require(
        config.tasks.constraints.provider == "generated",
        "tasks.constraints.provider",
        "must use the versioned local generator",
    )
    for task_id in expected_tasks[:-1]:
        _require(
            config.tasks[task_id].provider == "huggingface",
            f"tasks.{task_id}.provider",
            "must use the pinned benchmark source",
        )

    capability_ids = ("ifeval", "mmlu", "hellaswag", "arc_easy", "perplexity")
    for capability_id in capability_ids:
        capability = config.general_capabilities[capability_id]
        _validate_task_data(
            capability,
            f"general_capabilities.{capability_id}",
            expected_evaluation_only=True,
        )
        _require(
            capability.task_id == capability_id,
            f"general_capabilities.{capability_id}.task_id",
            f"must equal {capability_id!r}",
        )
    _require(
        config.general_capabilities.ifeval.dataset_name == "google/IFEval",
        "general_capabilities.ifeval.dataset_name",
        "official IFEval is the required evaluation-only source",
    )
    _require(
        config.general_capabilities.max_examples_per_capability >= 0,
        "general_capabilities.max_examples_per_capability",
        "must be nonnegative; zero means every configured held-out example",
    )
    _require(
        2 <= config.general_capabilities.max_perplexity_tokens <= model.max_sequence_length,
        "general_capabilities.max_perplexity_tokens",
        "must be at least two and not exceed model.max_sequence_length",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", config.general_capabilities.ifeval_scorer_revision)
        is not None,
        "general_capabilities.ifeval_scorer_revision",
        "must be an immutable 40-character Google Research revision",
    )
    _require(
        config.general_capabilities.ifeval_scorer_seed >= 0,
        "general_capabilities.ifeval_scorer_seed",
        "must be nonnegative",
    )
    _require(
        bool(config.general_capabilities.ifeval_nltk_data_dir),
        "general_capabilities.ifeval_nltk_data_dir",
        "must not be empty",
    )
    _require(
        re.fullmatch(
            r"[0-9a-f]{64}",
            config.general_capabilities.ifeval_punkt_tab_sha256,
        )
        is not None,
        "general_capabilities.ifeval_punkt_tab_sha256",
        "must be the pinned English punkt_tab SHA-256 digest",
    )
    if config.general_capabilities.enabled:
        _require(
            config.data.prepare_general_capabilities,
            "data.prepare_general_capabilities",
            "must be true when capability evaluation is enabled",
        )
        scorer_repository = config.general_capabilities.ifeval_scorer_repository
        _require(
            isinstance(scorer_repository, str) and bool(scorer_repository.strip()),
            "general_capabilities.ifeval_scorer_repository",
            "must identify the pinned Google Research Git worktree when enabled",
        )
    _require(
        bool(config.general_capabilities.mmlu_subjects),
        "general_capabilities.mmlu_subjects",
        "must not be empty",
    )
    _require(
        len(set(config.general_capabilities.mmlu_subjects))
        == len(config.general_capabilities.mmlu_subjects),
        "general_capabilities.mmlu_subjects",
        "must not contain duplicates",
    )
    _require(
        config.general_capabilities.max_examples_per_capability == 0
        or config.general_capabilities.max_examples_per_capability
        >= len(config.general_capabilities.mmlu_subjects),
        "general_capabilities.max_examples_per_capability",
        "must be zero or large enough to include every fixed MMLU subject",
    )
    _require(
        config.general_capabilities.max_new_tokens > 0,
        "general_capabilities.max_new_tokens",
        "must be positive",
    )
    _require(
        config.general_capabilities.max_new_tokens < model.max_sequence_length,
        "general_capabilities.max_new_tokens",
        "must be smaller than model.max_sequence_length",
    )
    _require(
        config.general_capabilities.do_sample is False,
        "general_capabilities.do_sample",
        "fixed evaluation must be deterministic",
    )
    _require(
        config.general_capabilities.temperature == 0,
        "general_capabilities.temperature",
        "fixed deterministic evaluation requires zero temperature",
    )
    _require(
        0 < config.general_capabilities.top_p <= 1,
        "general_capabilities.top_p",
        "must be in (0, 1]",
    )

    generator = config.constraint_generator
    for version_field in (
        "dataset_version",
        "grammar_version",
        "template_version",
        "validator_version",
    ):
        _require(
            bool(generator[version_field]),
            f"constraint_generator.{version_field}",
            "must not be empty",
        )
    _require(generator.seed >= 0, "constraint_generator.seed", "must be nonnegative")
    expected_roles = {"train", "validation", "anchor_candidate", "test"}
    _require(
        set(generator.split_sizes) == expected_roles,
        "constraint_generator.split_sizes",
        f"must define exactly {sorted(expected_roles)}",
    )
    for role, size in generator.split_sizes.items():
        _require(size > 0, f"constraint_generator.split_sizes.{role}", "must be positive")

    sandbox = config.code_sandbox
    _require(sandbox.backend == "docker", "code_sandbox.backend", "must be docker")
    _require(
        re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", sandbox.image) is not None,
        "code_sandbox.image",
        "must be pinned by sha256 digest",
    )
    _require(sandbox.timeout_seconds > 0, "code_sandbox.timeout_seconds", "must be positive")
    _require(sandbox.memory_mb > 0, "code_sandbox.memory_mb", "must be positive")
    _require(sandbox.cpus > 0, "code_sandbox.cpus", "must be positive")
    _require(sandbox.pids_limit > 0, "code_sandbox.pids_limit", "must be positive")
    _require(
        sandbox.output_limit_bytes > 0,
        "code_sandbox.output_limit_bytes",
        "must be positive",
    )
    _require(
        sandbox.network_disabled,
        "code_sandbox.network_disabled",
        "generated code must not have network access",
    )
    _require(
        sandbox.read_only_root,
        "code_sandbox.read_only_root",
        "generated code requires a read-only container root",
    )
    _require(bool(config.data.manifest_dir), "data.manifest_dir", "must not be empty")
    _require(
        config.data.fail_on_content_overlap,
        "data.fail_on_content_overlap",
        "must remain enabled to enforce scientific split isolation",
    )

    method = config.method
    lora = method.lora
    _require(lora.rank > 0, "method.lora.rank", "must be positive")
    _finite(float(lora.alpha), "method.lora.alpha")
    _require(lora.alpha > 0, "method.lora.alpha", "must be positive")
    _finite(float(lora.dropout), "method.lora.dropout")
    _require(0 <= lora.dropout < 1, "method.lora.dropout", "must be in [0, 1)")
    _require(bool(lora.target_modules), "method.lora.target_modules", "must not be empty")
    _require(
        len(set(lora.target_modules)) == len(lora.target_modules),
        "method.lora.target_modules",
        "must not contain duplicates",
    )

    importance = method.importance
    _finite(float(importance.ema_beta), "method.importance.ema_beta")
    _require(
        0 <= importance.ema_beta < 1,
        "method.importance.ema_beta",
        "must be in [0, 1)",
    )
    _finite(float(importance.epsilon), "method.importance.epsilon")
    _require(importance.epsilon > 0, "method.importance.epsilon", "must be positive")
    _require(
        0 < importance.clip_quantile <= 1,
        "method.importance.clip_quantile",
        "must be in (0, 1]",
    )

    memory = method.anchor_memory
    _require(memory.capacity >= 0, "method.anchor_memory.capacity", "must be nonnegative")
    _require(memory.batch_size > 0, "method.anchor_memory.batch_size", "must be positive")
    _require(memory.top_k_logits > 0, "method.anchor_memory.top_k_logits", "must be positive")
    _require(
        memory.top_k_logits <= model.vocab_size,
        "method.anchor_memory.top_k_logits",
        f"must not exceed model vocabulary size {model.vocab_size}",
    )
    if memory.enabled:
        _require(memory.capacity > 0, "method.anchor_memory.capacity", "must be positive")
    if method.use_anchor_kl:
        _require(memory.enabled, "method.use_anchor_kl", "requires anchor memory")
    if method.use_conflict_gate:
        _require(method.use_anchor_kl, "method.use_conflict_gate", "requires anchor KL")
    if method.use_dual_controller:
        _require(method.use_anchor_kl, "method.use_dual_controller", "requires anchor KL")
    if method.use_reward_aware_importance:
        _require(
            method.use_parameter_consolidation,
            "method.use_reward_aware_importance",
            "requires parameter consolidation",
        )
    if method.use_factorized_importance:
        _require(
            method.use_parameter_consolidation or method.diagnostic_factorized_importance,
            "method.use_factorized_importance",
            "requires parameter consolidation or explicit diagnostic use",
        )
    if method.name == "rl_lora":
        retention_switches = {
            "use_anchor_kl": method.use_anchor_kl,
            "use_parameter_consolidation": method.use_parameter_consolidation,
            "use_reward_aware_importance": method.use_reward_aware_importance,
            "use_factorized_importance": method.use_factorized_importance,
            "use_conflict_gate": method.use_conflict_gate,
            "use_dual_controller": method.use_dual_controller,
        }
        enabled = sorted(name for name, value in retention_switches.items() if value)
        _require(
            not enabled,
            "method",
            f"standard RL-LoRA cannot enable retention components: {enabled}",
        )
        _require(
            not memory.enabled and memory.capacity == 0,
            "method.anchor_memory",
            "standard RL-LoRA requires disabled zero-capacity anchor memory",
        )
        _require(
            method.parameter_regularization_lambda == 0,
            "method.parameter_regularization_lambda",
            "standard RL-LoRA cannot include a consolidation penalty",
        )

    conflict = method.conflict
    _require(
        conflict.zero_norm_threshold > 0,
        "method.conflict.zero_norm_threshold",
        "must be positive",
    )
    _require(
        conflict.coefficient_clip > 0,
        "method.conflict.coefficient_clip",
        "must be positive",
    )

    controller = method.functional_controller
    for field_name in ("initial_lambda", "dual_lr", "target_kl", "max_lambda"):
        _finite(float(controller[field_name]), f"method.functional_controller.{field_name}")
    _require(
        controller.initial_lambda >= 0,
        "method.functional_controller.initial_lambda",
        "must be nonnegative",
    )
    _require(controller.dual_lr > 0, "method.functional_controller.dual_lr", "must be positive")
    _require(
        controller.target_kl > 0,
        "method.functional_controller.target_kl",
        "must be positive",
    )
    _require(
        0 <= controller.ema_beta < 1,
        "method.functional_controller.ema_beta",
        "must be in [0, 1)",
    )
    _require(
        controller.max_lambda >= controller.initial_lambda,
        "method.functional_controller.max_lambda",
        "must be at least initial_lambda",
    )
    _require(
        controller.saturation_patience > 0,
        "method.functional_controller.saturation_patience",
        "must be positive",
    )
    _require(
        method.parameter_regularization_lambda >= 0,
        "method.parameter_regularization_lambda",
        "must be nonnegative",
    )
    _finite(
        float(method.parameter_importance_epsilon),
        "method.parameter_importance_epsilon",
    )
    _require(
        method.parameter_importance_epsilon > 0,
        "method.parameter_importance_epsilon",
        "must be positive",
    )
    if method.name == "hlora_rl":
        unsupported = {
            "use_anchor_kl": method.use_anchor_kl,
            "use_reward_aware_importance": method.use_reward_aware_importance,
            "use_factorized_importance": method.use_factorized_importance,
            "use_conflict_gate": method.use_conflict_gate,
            "use_dual_controller": method.use_dual_controller,
        }
        enabled = sorted(name for name, value in unsupported.items() if value)
        _require(
            not enabled,
            "method",
            f"direct HLoRA-RL cannot enable RAHC components: {enabled}",
        )
        _require(
            not memory.enabled and memory.capacity == 0,
            "method.anchor_memory",
            "direct HLoRA-RL requires disabled zero-capacity anchor memory",
        )

    _require(bool(config.task_stream.tasks), "task_stream.tasks", "must not be empty")
    _require(
        len(set(config.task_stream.tasks)) == len(config.task_stream.tasks),
        "task_stream.tasks",
        "must not contain duplicate tasks",
    )
    _require(
        set(config.task_stream.tasks) == set(expected_tasks),
        "task_stream.tasks",
        f"must contain exactly the primary continual tasks {list(expected_tasks)}",
    )
    _require(
        config.experiment.max_steps_per_task > 0,
        "experiment.max_steps_per_task",
        "must be positive",
    )
    _require(config.experiment.task_limit > 0, "experiment.task_limit", "must be positive")
    _require(
        config.experiment.task_limit <= len(config.task_stream.tasks),
        "experiment.task_limit",
        "must not exceed the configured task stream length",
    )
    _require(
        config.experiment.evaluation_max_examples_per_task >= 0,
        "experiment.evaluation_max_examples_per_task",
        "must be nonnegative; zero means the complete held-out split",
    )

    rollout = config.rollout
    _require(rollout.prompts_per_batch > 0, "rollout.prompts_per_batch", "must be positive")
    _require(rollout.group_size > 1, "rollout.group_size", "must be greater than one")
    _require(rollout.max_prompt_tokens > 0, "rollout.max_prompt_tokens", "must be positive")
    _require(
        rollout.max_response_tokens > 0,
        "rollout.max_response_tokens",
        "must be positive",
    )
    _require(
        rollout.max_prompt_tokens + rollout.max_response_tokens <= model.max_sequence_length,
        "rollout",
        "prompt and response token limits exceed model.max_sequence_length",
    )
    _finite(float(rollout.temperature), "rollout.temperature")
    if rollout.do_sample:
        _require(rollout.temperature > 0, "rollout.temperature", "must be positive for sampling")
    else:
        _require(
            rollout.temperature == 0,
            "rollout.temperature",
            "must be zero for greedy rollouts",
        )
    _require(0 < rollout.top_p <= 1, "rollout.top_p", "must be in (0, 1]")
    _require(
        rollout.advantage_epsilon > 0,
        "rollout.advantage_epsilon",
        "must be positive",
    )
    _require(
        rollout.advantage_clip > 0,
        "rollout.advantage_clip",
        "must be positive",
    )
    _require(
        0 < rollout.policy_clip_epsilon < 1,
        "rollout.policy_clip_epsilon",
        "must be in (0, 1)",
    )

    optimizer = config.optimizer
    _require(optimizer.name == "adamw", "optimizer.name", "must be adamw")
    _finite(float(optimizer.learning_rate), "optimizer.learning_rate")
    _require(optimizer.learning_rate > 0, "optimizer.learning_rate", "must be positive")
    _require(len(optimizer.betas) == 2, "optimizer.betas", "must contain exactly two values")
    for index, beta in enumerate(optimizer.betas):
        _finite(float(beta), f"optimizer.betas.{index}")
        _require(0 <= beta < 1, f"optimizer.betas.{index}", "must be in [0, 1)")
    _require(optimizer.epsilon > 0, "optimizer.epsilon", "must be positive")
    _require(optimizer.weight_decay >= 0, "optimizer.weight_decay", "must be nonnegative")
    _require(optimizer.max_grad_norm > 0, "optimizer.max_grad_norm", "must be positive")

    _require(config.scheduler.name == "constant", "scheduler.name", "must be constant")
    _require(config.scheduler.warmup_steps >= 0, "scheduler.warmup_steps", "must be nonnegative")
    _require(
        config.scheduler.warmup_steps == 0,
        "scheduler.warmup_steps",
        "constant scheduler does not use warmup",
    )
    _require(
        config.checkpoint.save_every_tasks > 0,
        "checkpoint.save_every_tasks",
        "must be positive",
    )
    _require(
        config.checkpoint.numeric_resume_tolerance > 0,
        "checkpoint.numeric_resume_tolerance",
        "must be positive",
    )
    _require(config.launcher.num_processes > 0, "launcher.num_processes", "must be positive")
