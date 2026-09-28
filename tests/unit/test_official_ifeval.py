from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import rahc_lora.evaluation.official_ifeval as official_ifeval
from rahc_lora.evaluation.official_ifeval import (
    GoogleResearchIFEvalScorer,
    _require_nltk_punkt,
    _resource_tree_sha256,
    verify_ifeval_checkout,
)
from rahc_lora.tasks.base import DatasetRole, TaskExample


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=RAHC Test",
            "-c",
            "user.email=rahc-test@example.invalid",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _fake_checkout(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "google-research"
    package = repository / "instruction_following_eval"
    package.mkdir(parents=True)
    (package / "evaluation_lib.py").write_text(
        """from dataclasses import dataclass

@dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, object]]

@dataclass
class OutputExample:
    follow_all_instructions: bool

def test_instruction_following_strict(inp, prompt_to_response):
    return OutputExample(prompt_to_response[inp.prompt] == 'strict')

def test_instruction_following_loose(inp, prompt_to_response):
    return OutputExample(prompt_to_response[inp.prompt] in {'strict', 'loose'})
""",
        encoding="utf-8",
    )
    for name in ("instructions.py", "instructions_registry.py", "instructions_util.py"):
        (package / name).write_text("# synthetic official boundary\n", encoding="utf-8")
    _git(repository.parent, "init", repository.name)
    _git(repository, "add", "instruction_following_eval")
    _git(repository, "commit", "-m", "synthetic pinned IFEval")
    return repository, _git(repository, "rev-parse", "HEAD")


def _clear_official_modules() -> None:
    for name in tuple(sys.modules):
        if name == "instruction_following_eval" or name.startswith("instruction_following_eval."):
            del sys.modules[name]


def test_checkout_verification_requires_exact_revision_and_clean_sources(
    tmp_path: Path,
) -> None:
    repository, revision = _fake_checkout(tmp_path)
    identity = verify_ifeval_checkout(repository, expected_revision=revision)
    assert identity.revision == revision
    assert identity.repository_root == str(repository.resolve())

    with pytest.raises(ValueError, match="revision mismatch"):
        verify_ifeval_checkout(repository, expected_revision="0" * 40)
    (repository / "instruction_following_eval" / "evaluation_lib.py").write_text(
        "# modified\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="modified or untracked"):
        verify_ifeval_checkout(repository, expected_revision=revision)


def test_checkout_verification_detects_hidden_source_edits(tmp_path: Path) -> None:
    repository, revision = _fake_checkout(tmp_path)
    relative_path = "instruction_following_eval/evaluation_lib.py"
    _git(repository, "update-index", "--assume-unchanged", relative_path)
    with (repository / relative_path).open("a", encoding="utf-8") as stream:
        stream.write("# hidden modification\n")
    with pytest.raises(ValueError, match="source bytes disagree"):
        verify_ifeval_checkout(repository, expected_revision=revision)


def test_punkt_resource_rejects_unpinned_content(tmp_path: Path) -> None:
    resource = tmp_path / "tokenizers" / "punkt_tab" / "english"
    resource.mkdir(parents=True)
    (resource / "abbrev_types.txt").write_text("Dr\n", encoding="utf-8")
    assert len(_resource_tree_sha256(resource)) == 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _require_nltk_punkt(tmp_path, "0" * 64)


def test_official_scorer_uses_strict_and_loose_api_without_changing_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _fake_checkout(tmp_path)
    monkeypatch.setattr(official_ifeval, "missing_ifeval_dependencies", lambda: ())
    nltk_root = tmp_path / "nltk_data"
    fake_nltk = SimpleNamespace(data=SimpleNamespace(path=[]))
    monkeypatch.setattr(
        official_ifeval,
        "_require_nltk_punkt",
        lambda *_: (fake_nltk, str(nltk_root)),
    )
    langdetect = ModuleType("langdetect")

    class DetectorFactory:
        seed: int | None = None

    langdetect.DetectorFactory = DetectorFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langdetect", langdetect)
    _clear_official_modules()
    try:
        scorer = GoogleResearchIFEvalScorer(
            repository,
            expected_revision=revision,
            deterministic_seed=19,
            nltk_data_root=nltk_root,
            expected_punkt_tab_sha256="0" * 64,
        )
        example = TaskExample(
            task_id="ifeval",
            example_id="ifeval:synthetic",
            role=DatasetRole.EVALUATION_ONLY,
            fields={
                "key": 1,
                "instruction_id_list": ["synthetic:instruction"],
                "prompt": "Follow the instruction.",
                "kwargs": [{}],
            },
        )
        random.seed(123)
        state = random.getstate()
        assert scorer.score(example, "loose") == (0.0, 1.0)
        assert random.getstate() == state
        assert DetectorFactory.seed is None
    finally:
        _clear_official_modules()
