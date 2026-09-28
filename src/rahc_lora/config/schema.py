"""Structured configuration schema for experiment-affecting choices."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PathsConfig:
    """Local artifact and cache roots."""

    output_dir: str = "outputs"
    cache_dir: str = ".cache/rahc_lora"


@dataclass
class ModelConfig:
    """Pinned model and tokenizer selection."""

    backend: str = "transformers"
    identifier: str = "sshleifer/tiny-gpt2"
    revision: str = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
    tokenizer_identifier: str = "sshleifer/tiny-gpt2"
    tokenizer_revision: str = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
    precision: str = "float32"
    allow_precision_fallback: bool = False
    vocab_size: int = 50_257
    hidden_size: int = 64
    max_sequence_length: int = 256
    trust_remote_code: bool = False


@dataclass
class TaskDataConfig:
    """Pinned source, split, prompt, and reward definition for one task."""

    task_id: str = ""
    family: str = ""
    provider: str = "huggingface"
    dataset_name: str = ""
    dataset_subset: str | None = None
    revision: str = ""
    license: str = ""
    citation: str = ""
    source_train_split: str | None = "train"
    source_validation_split: str | None = None
    source_test_split: str = "test"
    id_fields: list[str] = field(default_factory=list)
    split_seed: int = 17
    validation_fraction: float = 0.1
    anchor_fraction: float = 0.1
    preprocessing_version: str = "1.0.0"
    prompt_version: str = "1.0.0"
    reward_version: str = "1.0.0"
    evaluation_only: bool = False


def _task_config(task_id: str) -> TaskDataConfig:
    return TaskDataConfig(task_id=task_id)


@dataclass
class TaskCatalogConfig:
    """Exact datasets used by the primary continual-RL stream."""

    gsm8k: TaskDataConfig = field(default_factory=lambda: _task_config("gsm8k"))
    arc_challenge: TaskDataConfig = field(default_factory=lambda: _task_config("arc_challenge"))
    mbpp: TaskDataConfig = field(default_factory=lambda: _task_config("mbpp"))
    constraints: TaskDataConfig = field(default_factory=lambda: _task_config("constraints"))


@dataclass
class GeneralCapabilityConfig:
    """Pinned evaluation-only datasets and fixed generation settings."""

    enabled: bool = False
    max_examples_per_capability: int = 0
    max_perplexity_tokens: int = 256
    ifeval_scorer_repository: str | None = None
    ifeval_scorer_revision: str = "758b894eb02dc2a7097068031089a2803be147c6"
    ifeval_scorer_seed: int = 0
    ifeval_nltk_data_dir: str = ".cache/rahc_lora/nltk_data"
    ifeval_punkt_tab_sha256: str = (
        "56e42ce8e87ce5653d8bc261e8100a2cc719fa51e3e66ef138dc54049733b4b6"
    )
    ifeval: TaskDataConfig = field(default_factory=lambda: _task_config("ifeval"))
    mmlu: TaskDataConfig = field(default_factory=lambda: _task_config("mmlu"))
    hellaswag: TaskDataConfig = field(default_factory=lambda: _task_config("hellaswag"))
    arc_easy: TaskDataConfig = field(default_factory=lambda: _task_config("arc_easy"))
    perplexity: TaskDataConfig = field(default_factory=lambda: _task_config("perplexity"))
    mmlu_subjects: list[str] = field(
        default_factory=lambda: [
            "abstract_algebra",
            "college_computer_science",
            "high_school_physics",
            "professional_medicine",
        ]
    )
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class ConstraintGeneratorConfig:
    """Versioned deterministic custom-constraint corpus definition."""

    dataset_version: str = "1.0.0"
    grammar_version: str = "1.0.0"
    template_version: str = "1.0.0"
    validator_version: str = "1.0.0"
    seed: int = 20260922
    split_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "train": 32,
            "validation": 16,
            "anchor_candidate": 16,
            "test": 16,
        }
    )


@dataclass
class CodeSandboxConfig:
    """Fail-closed generated-code execution boundary."""

    backend: str = "docker"
    image: str = (
        "python:3.11.10-slim-bookworm@"
        "sha256:840e180ebcc6e5c8efab209c43f5e40fd2af98cb49db5c7103c90539c56bb30e"
    )
    timeout_seconds: float = 5.0
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 32
    output_limit_bytes: int = 65_536
    network_disabled: bool = True
    read_only_root: bool = True


@dataclass
class DataConfig:
    """Versioned data-manifest output and split policy."""

    manifest_dir: str = "data/manifests"
    fail_on_content_overlap: bool = True
    prepare_general_capabilities: bool = True


@dataclass
class LoRAConfig:
    """Shared LoRA adapter configuration."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class ImportanceConfig:
    """Reward-aware importance numeric controls."""

    ema_beta: float = 0.99
    epsilon: float = 1.0e-8
    clip_quantile: float = 0.995
    accumulator_dtype: str = "float32"


@dataclass
class AnchorMemoryConfig:
    """Bounded behavioural-memory controls."""

    enabled: bool = False
    capacity: int = 0
    batch_size: int = 16
    top_k_logits: int = 32


@dataclass
class ConflictConfig:
    """Layer-conflict numeric controls."""

    zero_norm_threshold: float = 1.0e-12
    coefficient_clip: float = 5.0


@dataclass
class FunctionalControllerConfig:
    """Adaptive functional-retention controller settings."""

    initial_lambda: float = 0.0
    dual_lr: float = 0.001
    target_kl: float = 0.02
    ema_beta: float = 0.95
    max_lambda: float = 10.0
    saturation_patience: int = 100


@dataclass
class MethodConfig:
    """Configuration-only method composition and ablation switches."""

    name: str = "rl_lora"
    use_anchor_kl: bool = False
    use_parameter_consolidation: bool = False
    use_reward_aware_importance: bool = False
    use_factorized_importance: bool = False
    use_conflict_gate: bool = False
    use_dual_controller: bool = False
    diagnostic_factorized_importance: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    anchor_memory: AnchorMemoryConfig = field(default_factory=AnchorMemoryConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    functional_controller: FunctionalControllerConfig = field(
        default_factory=FunctionalControllerConfig
    )
    parameter_regularization_lambda: float = 0.0
    parameter_importance_epsilon: float = 1.0e-8


@dataclass
class TaskStreamConfig:
    """Predeclared continual-task ordering."""

    name: str = "order_1"
    tasks: list[str] = field(
        default_factory=lambda: ["gsm8k", "arc_challenge", "mbpp", "constraints"]
    )


@dataclass
class ExperimentConfig:
    """Training-budget controls for one experiment class."""

    name: str = "smoke"
    max_steps_per_task: int = 1
    task_limit: int = 2
    evaluation_max_examples_per_task: int = 0
    deterministic: bool = True


@dataclass
class RolloutConfig:
    """Project-owned GRPO rollout and clipped-objective controls."""

    prompts_per_batch: int = 2
    group_size: int = 4
    max_prompt_tokens: int = 128
    max_response_tokens: int = 32
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    advantage_epsilon: float = 1.0e-8
    advantage_clip: float = 5.0
    policy_clip_epsilon: float = 0.2


@dataclass
class OptimizerConfig:
    """Optimizer choices shared by every method using the continual trainer."""

    name: str = "adamw"
    learning_rate: float = 1.0e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class SchedulerConfig:
    """Explicit learning-rate schedule configuration."""

    name: str = "constant"
    warmup_steps: int = 0


@dataclass
class CheckpointConfig:
    """Atomic checkpoint and compatibility policy."""

    enabled: bool = True
    save_every_tasks: int = 1
    numeric_resume_tolerance: float = 1.0e-7


@dataclass
class LauncherConfig:
    """Execution topology configuration."""

    name: str = "local"
    device: str = "auto"
    num_processes: int = 1


@dataclass
class RootConfig:
    """Complete resolved run configuration."""

    seed: int = 1
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tasks: TaskCatalogConfig = field(default_factory=TaskCatalogConfig)
    general_capabilities: GeneralCapabilityConfig = field(default_factory=GeneralCapabilityConfig)
    constraint_generator: ConstraintGeneratorConfig = field(
        default_factory=ConstraintGeneratorConfig
    )
    code_sandbox: CodeSandboxConfig = field(default_factory=CodeSandboxConfig)
    data: DataConfig = field(default_factory=DataConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    task_stream: TaskStreamConfig = field(default_factory=TaskStreamConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
