"""Version-checked adapter for the official Google Research IFEval evaluator."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import random
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from rahc_lora.tasks.base import TaskExample

OFFICIAL_IFEVAL_PACKAGE = "instruction_following_eval"
OFFICIAL_IFEVAL_DEPENDENCIES = {
    "absl-py": "absl",
    "immutabledict": "immutabledict",
    "langdetect": "langdetect",
    "nltk": "nltk",
}
OFFICIAL_IFEVAL_SOURCE_FILES = (
    "instruction_following_eval/evaluation_lib.py",
    "instruction_following_eval/instructions.py",
    "instruction_following_eval/instructions_registry.py",
    "instruction_following_eval/instructions_util.py",
)
PUNKT_TAB_RESOURCE = "tokenizers/punkt_tab/english"


@dataclass(frozen=True)
class IFEvalCheckoutIdentity:
    """Verified official evaluator checkout identity."""

    repository_root: str
    revision: str


def _git_output(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Unable to inspect official IFEval checkout: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git failure"
        raise ValueError(f"Unable to inspect official IFEval checkout: {detail}")
    return result.stdout.strip()


def verify_ifeval_checkout(
    repository_root: Path,
    *,
    expected_revision: str,
) -> IFEvalCheckoutIdentity:
    """Require the exact clean Google Research source revision used by the run."""

    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("Official IFEval revision must be a 40-character commit hash")
    root = repository_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Official IFEval repository is not a directory: {root}")
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ValueError(
            "Official IFEval repository path must be the Git worktree root; "
            f"received {root}, detected {top_level}"
        )
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(
            "Official IFEval checkout revision mismatch; "
            f"expected {expected_revision}, received {revision}"
        )
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *OFFICIAL_IFEVAL_SOURCE_FILES,
    )
    if status:
        raise ValueError("Official IFEval runtime sources contain modified or untracked files")
    if (root / OFFICIAL_IFEVAL_PACKAGE / "__init__.py").exists():
        raise ValueError("Official IFEval checkout contains an unexpected package initializer")
    for relative_path in OFFICIAL_IFEVAL_SOURCE_FILES:
        _git_output(root, "ls-files", "--error-unmatch", "--", relative_path)
        if not (root / relative_path).is_file():
            raise ValueError(f"Official IFEval source file is not checked out: {relative_path}")
        expected_blob = _git_output(root, "rev-parse", f"HEAD:{relative_path}")
        actual_blob = _git_output(root, "hash-object", "--", relative_path)
        if actual_blob != expected_blob:
            raise ValueError(
                f"Official IFEval source bytes disagree with pinned commit: {relative_path}"
            )
    return IFEvalCheckoutIdentity(repository_root=str(root), revision=revision)


def missing_ifeval_dependencies() -> tuple[str, ...]:
    """Return missing distributions required by the pinned official evaluator."""

    return tuple(
        distribution
        for distribution, module in OFFICIAL_IFEVAL_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    )


def _resource_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"Official IFEval NLTK resource is empty: {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _require_nltk_punkt(nltk_data_root: Path, expected_sha256: str) -> tuple[ModuleType, str]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Official IFEval Punkt resource hash must be a SHA-256 digest")
    data_root = nltk_data_root.expanduser().resolve()
    resource_root = data_root / PUNKT_TAB_RESOURCE
    if not resource_root.is_dir():
        raise ValueError(f"Official IFEval NLTK resource is unavailable: {resource_root}")
    observed_sha256 = _resource_tree_sha256(resource_root)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Official IFEval Punkt resource hash mismatch; "
            f"expected {expected_sha256}, received {observed_sha256}"
        )
    nltk = importlib.import_module("nltk")
    data_path = getattr(getattr(nltk, "data", None), "path", None)
    if not isinstance(data_path, list):
        raise ValueError("Installed NLTK does not expose a mutable data path")
    root_text = str(data_root)
    inserted = root_text not in data_path
    if inserted:
        data_path.insert(0, root_text)
    try:
        nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError as error:
        raise ValueError(
            "Official IFEval requires the NLTK Punkt tokenizer data; "
            "install it before the run without downloading during evaluation"
        ) from error
    finally:
        if inserted:
            data_path.remove(root_text)
    return nltk, root_text


def _module_belongs_to(module: ModuleType, package_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        return Path(module_file).resolve().is_relative_to(package_root)
    module_paths = getattr(module, "__path__", ())
    return bool(module_paths) and all(
        Path(path).resolve().is_relative_to(package_root) for path in module_paths
    )


def _load_evaluation_lib(repository_root: Path) -> ModuleType:
    package_root = repository_root / OFFICIAL_IFEVAL_PACKAGE
    prefix = f"{OFFICIAL_IFEVAL_PACKAGE}."
    for name, module in tuple(sys.modules.items()):
        if (name == OFFICIAL_IFEVAL_PACKAGE or name.startswith(prefix)) and not _module_belongs_to(
            module, package_root
        ):
            raise RuntimeError(
                "A different instruction_following_eval package is already loaded; "
                "start a clean process for the configured official checkout"
            )
    module_name = f"{OFFICIAL_IFEVAL_PACKAGE}.evaluation_lib"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    root_text = str(repository_root)
    sys.path.insert(0, root_text)
    try:
        loaded = importlib.import_module(module_name)
    finally:
        with suppress(ValueError):
            sys.path.remove(root_text)
    if not _module_belongs_to(loaded, package_root):
        raise RuntimeError("Loaded IFEval evaluator does not belong to the verified checkout")
    return loaded


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Official IFEval field {field!r} must be a sequence")
    return value


class GoogleResearchIFEvalScorer:
    """Run strict and loose prompt-level metrics through the pinned official code."""

    def __init__(
        self,
        repository_root: Path,
        *,
        expected_revision: str,
        deterministic_seed: int,
        nltk_data_root: Path,
        expected_punkt_tab_sha256: str,
    ) -> None:
        if deterministic_seed < 0:
            raise ValueError("Official IFEval deterministic seed must be nonnegative")
        identity = verify_ifeval_checkout(
            repository_root,
            expected_revision=expected_revision,
        )
        missing = missing_ifeval_dependencies()
        if missing:
            raise ValueError(
                "Official IFEval runtime dependencies are missing: " + ", ".join(missing)
            )
        nltk, nltk_data_root_text = _require_nltk_punkt(
            nltk_data_root,
            expected_punkt_tab_sha256,
        )
        root = Path(identity.repository_root)
        evaluation_lib = _load_evaluation_lib(root)
        input_example = getattr(evaluation_lib, "InputExample", None)
        strict = getattr(evaluation_lib, "test_instruction_following_strict", None)
        loose = getattr(evaluation_lib, "test_instruction_following_loose", None)
        if not all(callable(value) for value in (input_example, strict, loose)):
            raise ValueError("Pinned official IFEval evaluator has an unexpected API")
        langdetect = importlib.import_module("langdetect")
        detector_factory = getattr(langdetect, "DetectorFactory", None)
        if detector_factory is None or not hasattr(detector_factory, "seed"):
            raise ValueError("Installed langdetect does not expose DetectorFactory.seed")
        self.version = identity.revision
        self.repository_root = identity.repository_root
        self.deterministic_seed = deterministic_seed
        self.nltk_punkt_tab_sha256 = expected_punkt_tab_sha256
        self._input_example = cast(Callable[..., object], input_example)
        self._strict = cast(Callable[[object, Mapping[str, str]], object], strict)
        self._loose = cast(Callable[[object, Mapping[str, str]], object], loose)
        self._detector_factory = detector_factory
        self._nltk_data_path = nltk.data.path
        self._nltk_data_root_text = nltk_data_root_text

    def _official_input(self, example: TaskExample) -> object:
        fields = example.fields
        key = fields.get("key")
        if type(key) is not int:
            raise ValueError("Official IFEval field 'key' must be an integer")
        prompt = fields.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Official IFEval field 'prompt' must be a nonempty string")
        instruction_values = _sequence(fields.get("instruction_id_list"), "instruction_id_list")
        instruction_ids = [str(value) for value in instruction_values]
        if not instruction_ids or any(not value for value in instruction_ids):
            raise ValueError("Official IFEval instruction IDs must be nonempty")
        kwargs_values = _sequence(fields.get("kwargs"), "kwargs")
        kwargs: list[dict[str, Any]] = []
        for value in kwargs_values:
            if not isinstance(value, Mapping):
                raise ValueError("Official IFEval kwargs entries must be mappings")
            kwargs.append({str(name): argument for name, argument in value.items()})
        if len(kwargs) != len(instruction_ids):
            raise ValueError("Official IFEval instruction IDs and kwargs lengths disagree")
        return self._input_example(
            key=key,
            instruction_id_list=instruction_ids,
            prompt=prompt,
            kwargs=kwargs,
        )

    def score(self, example: TaskExample, response: str) -> tuple[float, float]:
        """Return official prompt-level strict and loose scores deterministically."""

        official_input = self._official_input(example)
        prompt = str(example.fields["prompt"])
        prompt_to_response = {prompt: response}
        random_state = random.getstate()
        previous_detector_seed = self._detector_factory.seed
        inserted_nltk_path = self._nltk_data_root_text not in self._nltk_data_path
        try:
            if inserted_nltk_path:
                self._nltk_data_path.insert(0, self._nltk_data_root_text)
            random.seed(self.deterministic_seed)
            self._detector_factory.seed = self.deterministic_seed
            strict_output = self._strict(official_input, prompt_to_response)
            loose_output = self._loose(official_input, prompt_to_response)
        finally:
            random.setstate(random_state)
            self._detector_factory.seed = previous_detector_seed
            if inserted_nltk_path:
                self._nltk_data_path.remove(self._nltk_data_root_text)
        strict = getattr(strict_output, "follow_all_instructions", None)
        loose = getattr(loose_output, "follow_all_instructions", None)
        if type(strict) is not bool or type(loose) is not bool:
            raise ValueError("Official IFEval evaluator returned an unexpected result")
        return float(strict), float(loose)
