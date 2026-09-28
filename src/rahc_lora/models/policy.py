"""Project-owned causal-policy wrapper independent of external trainer APIs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from rahc_lora.evaluation.evaluator import EvaluationGenerationConfig
from rahc_lora.evaluation.general_capabilities import TokenNll
from rahc_lora.rl.rollout import SampledResponse


class Tokenizer(Protocol):
    """Minimal tokenizer boundary needed by rollout and evaluation."""

    pad_token_id: int
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str, *, max_length: int) -> tuple[int, ...]: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...


@dataclass(frozen=True)
class PolicyGenerationConfig:
    """Sampling settings for grouped RL rollouts."""

    max_prompt_tokens: int
    max_response_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


@dataclass(frozen=True)
class _LoRALayer:
    """One active LoRA pair and its exact trainable-state keys."""

    name: str
    module: nn.Module
    dropout: nn.Module
    a_name: str
    b_name: str
    a: Tensor
    b: Tensor
    scaling: float


class EffectiveWeightGradientCapture:
    """Capture exact gradients with respect to dense LoRA updates for one backward pass."""

    def __init__(self, policy: TorchCausalPolicy) -> None:
        self._policy = policy
        self._gradients: dict[str, Tensor] = {}
        self._activations: dict[str, list[Tensor]] = defaultdict(list)
        self._handles: list[Any] = []
        self._layers = policy._named_lora_layers()

    def __enter__(self) -> EffectiveWeightGradientCapture:
        for layer in self._layers:
            self._handles.append(
                layer.dropout.register_forward_hook(
                    self._dropout_hook(layer.name)
                )
            )
            self._handles.append(
                layer.module.register_forward_hook(self._layer_hook(layer))
            )
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _dropout_hook(self, name: str) -> Any:
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise TypeError(f"LoRA dropout for {name!r} must return a tensor")
            self._activations[name].append(output.detach())

        return capture

    def _layer_hook(self, layer: _LoRALayer) -> Any:
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, Tensor):
                raise TypeError(f"LoRA target layer {layer.name!r} must return a tensor")
            activations = self._activations[layer.name]
            if not activations:
                raise RuntimeError(
                    f"LoRA target layer {layer.name!r} did not run its configured dropout"
                )
            activation = activations.pop()
            if not tensor.requires_grad:
                raise RuntimeError(
                    f"LoRA target layer {layer.name!r} output does not require gradients"
                )

            def capture_gradient(gradient: Tensor) -> Tensor:
                if gradient.shape[:-1] != activation.shape[:-1]:
                    raise ValueError(
                        f"LoRA activation/gradient leading shapes disagree for {layer.name!r}"
                    )
                if gradient.shape[-1] != layer.b.shape[0]:
                    raise ValueError(f"LoRA output width disagrees for {layer.name!r}")
                if activation.shape[-1] != layer.a.shape[1]:
                    raise ValueError(f"LoRA input width disagrees for {layer.name!r}")
                grad_rows = gradient.reshape(-1, gradient.shape[-1]).to(torch.float32)
                input_rows = activation.reshape(-1, activation.shape[-1]).to(torch.float32)
                effective_gradient = grad_rows.transpose(0, 1) @ input_rows
                effective_gradient.mul_(layer.scaling)
                if not bool(torch.isfinite(effective_gradient).all()):
                    raise ValueError(
                        f"Non-finite effective LoRA gradient for {layer.name!r}"
                    )
                old = self._gradients.get(layer.name)
                self._gradients[layer.name] = (
                    effective_gradient if old is None else old + effective_gradient
                )
                return gradient

            tensor.register_hook(capture_gradient)

        return capture

    def gradients(self) -> dict[str, Tensor]:
        """Return the captured gradients after backward has completed."""

        expected = {layer.name for layer in self._layers}
        if set(self._gradients) != expected:
            raise RuntimeError(
                "Effective LoRA gradient capture is incomplete; "
                f"missing={sorted(expected - set(self._gradients))}, "
                f"extra={sorted(set(self._gradients) - expected)}"
            )
        return {name: value.detach() for name, value in self._gradients.items()}


class TrainableCausalPolicy(Protocol):
    """Shared train/evaluate/checkpoint interface for one LoRA policy."""

    policy_version: int
    device: torch.device

    def sample(
        self,
        prompts: Sequence[str],
        *,
        group_size: int,
        generation: PolicyGenerationConfig,
        generator: torch.Generator,
    ) -> tuple[SampledResponse, ...]: ...

    def response_logprobs(self, responses: Sequence[SampledResponse]) -> tuple[Tensor, Tensor]: ...

    def generate(
        self, prompts: Sequence[str], generation: EvaluationGenerationConfig
    ) -> Sequence[str]: ...

    def token_nll(
        self, texts: Sequence[str], *, max_tokens: int | None = None
    ) -> Sequence[TokenNll]: ...

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]: ...

    def named_trainable_parameters(self) -> dict[str, nn.Parameter]: ...

    def named_effective_lora_weights(
        self,
        *,
        adapter_state: dict[str, Tensor] | None = None,
        detach: bool = False,
    ) -> dict[str, Tensor]: ...

    def capture_effective_weight_gradients(self) -> EffectiveWeightGradientCapture: ...

    def adapter_state_dict(self) -> dict[str, Tensor]: ...

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None: ...

    def increment_policy_version(self) -> None: ...


class HuggingFaceTokenizer:
    """Adapter around a pinned Transformers tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("Causal tokenizer must define an EOS token ID")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.vocab_size = len(tokenizer)

    def encode(self, text: str, *, max_length: int) -> tuple[int, ...]:
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        if not token_ids:
            token_ids = [self.eos_token_id]
        return tuple(int(token) for token in token_ids)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=True))


class TorchCausalPolicy:
    """Grouped generation and differentiable response scoring for a torch causal LM."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        *,
        device: torch.device,
        max_sequence_length: int,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.policy_version = 0
        trainable = self.named_trainable_parameters()
        if not trainable:
            raise ValueError("Causal policy has no trainable LoRA parameters")
        non_lora = sorted(name for name in trainable if "lora_" not in name.casefold())
        if non_lora:
            raise ValueError(f"Only LoRA parameters may be trainable: {non_lora}")

    def named_trainable_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def _named_lora_layers(self) -> tuple[_LoRALayer, ...]:
        trainable = self.named_trainable_parameters()
        names_by_parameter = {id(parameter): name for name, parameter in trainable.items()}
        layers: list[_LoRALayer] = []

        for layer_name, module in self.model.named_modules():
            a_container = getattr(module, "lora_A", None)
            b_container = getattr(module, "lora_B", None)
            if a_container is None or b_container is None:
                continue

            if isinstance(a_container, nn.Parameter) and isinstance(b_container, nn.Parameter):
                a = a_container
                b = b_container
                dropout = getattr(module, "dropout", None)
                scaling_value = getattr(module, "scaling", None)
            elif isinstance(a_container, nn.ModuleDict) and isinstance(
                b_container, nn.ModuleDict
            ):
                active = getattr(module, "active_adapters", None)
                if active is None:
                    active = getattr(module, "active_adapter", None)
                if isinstance(active, str):
                    active = [active]
                if not active and len(a_container) == len(b_container) == 1:
                    active = [next(iter(a_container.keys()))]
                if not isinstance(active, Sequence) or len(active) != 1:
                    raise ValueError(
                        f"Direct HLoRA requires one active shared adapter in {layer_name!r}"
                    )
                adapter_name = str(active[0])
                if adapter_name not in a_container or adapter_name not in b_container:
                    raise ValueError(
                        f"Active LoRA adapter {adapter_name!r} is missing in {layer_name!r}"
                    )
                a_module = a_container[adapter_name]
                b_module = b_container[adapter_name]
                a = getattr(a_module, "weight", None)
                b = getattr(b_module, "weight", None)
                dropout_container = getattr(module, "lora_dropout", None)
                dropout = (
                    dropout_container[adapter_name]
                    if isinstance(dropout_container, nn.ModuleDict)
                    else dropout_container
                )
                scaling_container = getattr(module, "scaling", None)
                scaling_value = (
                    scaling_container.get(adapter_name)
                    if isinstance(scaling_container, dict)
                    else scaling_container
                )
            else:
                raise TypeError(f"Unsupported LoRA layer representation at {layer_name!r}")

            if not isinstance(a, Tensor) or not isinstance(b, Tensor):
                raise TypeError(f"LoRA factors at {layer_name!r} must be tensors")
            if not isinstance(dropout, nn.Module):
                raise TypeError(f"LoRA dropout at {layer_name!r} must be a module")
            if not isinstance(scaling_value, (int, float)):
                raise TypeError(f"LoRA scaling at {layer_name!r} must be numeric")
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise ValueError(f"LoRA factor shapes are incompatible at {layer_name!r}")
            if id(a) not in names_by_parameter or id(b) not in names_by_parameter:
                raise ValueError(
                    f"LoRA factors at {layer_name!r} must both be trainable shared-policy state"
                )
            layers.append(
                _LoRALayer(
                    name=layer_name or "root",
                    module=module,
                    dropout=dropout,
                    a_name=names_by_parameter[id(a)],
                    b_name=names_by_parameter[id(b)],
                    a=a,
                    b=b,
                    scaling=float(scaling_value),
                )
            )

        if not layers:
            raise ValueError("Policy has no supported active LoRA layers")
        if len({layer.name for layer in layers}) != len(layers):
            raise ValueError("LoRA target layer names must be unique")
        return tuple(layers)

    def named_effective_lora_weights(
        self,
        *,
        adapter_state: dict[str, Tensor] | None = None,
        detach: bool = False,
    ) -> dict[str, Tensor]:
        """Compute scaled dense updates from the shared LoRA factors."""

        state = adapter_state or {}
        current_names = set(self.named_trainable_parameters())
        if state and set(state) != current_names:
            raise ValueError("Reference adapter state keys do not match the current policy")
        result: dict[str, Tensor] = {}
        for layer in self._named_lora_layers():
            a = state.get(layer.a_name, layer.a)
            b = state.get(layer.b_name, layer.b)
            a = a.to(device=layer.a.device, dtype=layer.a.dtype)
            b = b.to(device=layer.b.device, dtype=layer.b.dtype)
            effective = torch.matmul(b, a) * layer.scaling
            if detach:
                effective = effective.detach()
            result[layer.name] = effective
        return result

    def capture_effective_weight_gradients(self) -> EffectiveWeightGradientCapture:
        """Capture actual dense update gradients during one policy-loss backward pass."""

        return EffectiveWeightGradientCapture(self)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.named_trainable_parameters().values())

    @contextmanager
    def _evaluation_mode(self) -> Iterator[None]:
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    def _forward_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output if isinstance(output, Tensor) else getattr(output, "logits", None)
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise TypeError("Causal model must return logits with shape [B, T, V]")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Causal model produced non-finite logits")
        return logits

    @staticmethod
    def _top_p_logits(logits: Tensor, top_p: float) -> Tensor:
        if top_p >= 1:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = sorted_logits.masked_fill(remove, float("-inf"))
        restored = torch.empty_like(filtered)
        restored.scatter_(0, sorted_indices, filtered)
        return restored

    def _sample_one(
        self,
        prompt: str,
        generation: PolicyGenerationConfig,
        generator: torch.Generator,
    ) -> SampledResponse:
        prompt_ids = self.tokenizer.encode(prompt, max_length=generation.max_prompt_tokens)
        response_ids: list[int] = []
        old_logprobs: list[float] = []
        for _ in range(generation.max_response_tokens):
            sequence = (*prompt_ids, *response_ids)
            if len(sequence) >= self.max_sequence_length:
                break
            input_ids = torch.tensor(sequence, dtype=torch.long, device=self.device)[None, :]
            attention = torch.ones_like(input_ids)
            logits = self._forward_logits(input_ids, attention)[0, -1].to(torch.float32)
            if generation.do_sample:
                scaled = logits / generation.temperature
                scaled = self._top_p_logits(scaled, generation.top_p)
                probabilities = torch.softmax(scaled, dim=-1)
                token = int(torch.multinomial(probabilities, 1, generator=generator).item())
                logprob = torch.log_softmax(scaled, dim=-1)[token]
            else:
                token = int(torch.argmax(logits).item())
                logprob = torch.log_softmax(logits, dim=-1)[token]
            response_ids.append(token)
            old_logprobs.append(float(logprob.item()))
            if token == self.tokenizer.eos_token_id:
                break
        if not response_ids:
            raise RuntimeError("Policy generation produced no response tokens")
        response_tuple = tuple(response_ids)
        return SampledResponse(
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_tuple,
            response_text=self.tokenizer.decode(response_tuple),
            old_policy_logprobs=tuple(old_logprobs),
            sampling_metadata={
                "do_sample": generation.do_sample,
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "max_response_tokens": generation.max_response_tokens,
            },
        )

    def sample(
        self,
        prompts: Sequence[str],
        *,
        group_size: int,
        generation: PolicyGenerationConfig,
        generator: torch.Generator,
    ) -> tuple[SampledResponse, ...]:
        if not prompts or group_size <= 0:
            raise ValueError("Policy sampling requires prompts and a positive group size")
        samples: list[SampledResponse] = []
        with self._evaluation_mode(), torch.no_grad():
            for prompt in prompts:
                for _ in range(group_size):
                    samples.append(self._sample_one(prompt, generation, generator))
        return tuple(samples)

    def response_logprobs(self, responses: Sequence[SampledResponse]) -> tuple[Tensor, Tensor]:
        if not responses:
            raise ValueError("Response scoring requires at least one sequence")
        combined = [
            (*response.prompt_token_ids, *response.response_token_ids) for response in responses
        ]
        if any(len(sequence) > self.max_sequence_length for sequence in combined):
            raise ValueError("Response sequence exceeds the configured model limit")
        max_input_length = max(len(sequence) - 1 for sequence in combined)
        batch = torch.full(
            (len(combined), max_input_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.zeros_like(batch)
        for index, sequence in enumerate(combined):
            input_tokens = sequence[:-1]
            batch[index, : len(input_tokens)] = torch.tensor(input_tokens, device=self.device)
            attention[index, : len(input_tokens)] = 1
        logits = self._forward_logits(batch, attention).to(torch.float32)
        max_response_length = max(len(response.response_token_ids) for response in responses)
        logprobs = torch.zeros(
            (len(responses), max_response_length), dtype=torch.float32, device=self.device
        )
        mask = torch.zeros_like(logprobs, dtype=torch.bool)
        for index, response in enumerate(responses):
            start = len(response.prompt_token_ids) - 1
            length = len(response.response_token_ids)
            positions = logits[index, start : start + length]
            targets = torch.tensor(
                response.response_token_ids, dtype=torch.long, device=self.device
            )
            token_logprobs = torch.log_softmax(positions, dim=-1).gather(1, targets[:, None])[:, 0]
            logprobs[index, :length] = token_logprobs
            mask[index, :length] = True
        if not bool(torch.isfinite(logprobs).all()):
            raise ValueError("Response log-probabilities are non-finite")
        return logprobs, mask

    def generate(
        self, prompts: Sequence[str], generation: EvaluationGenerationConfig
    ) -> Sequence[str]:
        device_type = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        generator = torch.Generator(device=device_type)
        generator.manual_seed(0)
        settings = PolicyGenerationConfig(
            max_prompt_tokens=max(1, self.max_sequence_length - generation.max_new_tokens),
            max_response_tokens=generation.max_new_tokens,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            top_p=generation.top_p,
        )
        return tuple(
            sample.response_text
            for sample in self.sample(
                prompts,
                group_size=1,
                generation=settings,
                generator=generator,
            )
        )

    def token_nll(
        self, texts: Sequence[str], *, max_tokens: int | None = None
    ) -> Sequence[TokenNll]:
        """Calculate read-only token NLL records for held-out perplexity."""

        if not texts:
            raise ValueError("Perplexity evaluation requires at least one text")
        limit = self.max_sequence_length if max_tokens is None else max_tokens
        if not 2 <= limit <= self.max_sequence_length:
            raise ValueError(
                "Perplexity max_tokens must be at least two and not exceed the model limit"
            )
        records: list[TokenNll] = []
        with self._evaluation_mode(), torch.no_grad():
            for text in texts:
                token_ids = self.tokenizer.encode(text, max_length=limit)
                if len(token_ids) < 2:
                    raise ValueError("Perplexity text produced fewer than two tokens")
                input_ids = torch.tensor(token_ids[:-1], dtype=torch.long, device=self.device)[
                    None, :
                ]
                attention = torch.ones_like(input_ids)
                logits = self._forward_logits(input_ids, attention)[0].to(torch.float32)
                targets = torch.tensor(token_ids[1:], dtype=torch.long, device=self.device)
                token_logprobs = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
                total_nll = float((-token_logprobs).sum(dtype=torch.float64).item())
                records.append(TokenNll(total_nll=total_nll, token_count=len(targets)))
        return tuple(records)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_trainable_parameters().items()
        }

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None:
        parameters = self.named_trainable_parameters()
        if set(parameters) != set(state):
            raise ValueError(
                "Adapter checkpoint keys do not match policy; "
                f"missing={sorted(set(parameters).difference(state))}, "
                f"extra={sorted(set(state).difference(parameters))}"
            )
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if value.shape != parameter.shape:
                    raise ValueError(f"Adapter checkpoint shape mismatch for {name!r}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def increment_policy_version(self) -> None:
        self.policy_version += 1

    def frozen_parameter_state(self) -> dict[str, Tensor]:
        """Return frozen weights for integration assertions, never for checkpoints."""

        trainable_names = set(self.named_trainable_parameters())
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if name not in trainable_names
        }


def old_logprobs_tensor(
    responses: Sequence[SampledResponse], *, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Pad stored old-policy log-probabilities to ``[B,T]`` with a validity mask."""

    if not responses:
        raise ValueError("Old-policy log-probability conversion requires responses")
    max_length = max(len(response.old_policy_logprobs) for response in responses)
    values = torch.zeros((len(responses), max_length), dtype=torch.float32, device=device)
    mask = torch.zeros_like(values, dtype=torch.bool)
    for index, response in enumerate(responses):
        length = len(response.old_policy_logprobs)
        values[index, :length] = torch.tensor(response.old_policy_logprobs, device=device)
        mask[index, :length] = True
    return values, mask
