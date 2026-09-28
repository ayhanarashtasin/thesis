from __future__ import annotations

from pathlib import Path

import pytest

from rahc_lora.config.schema import CodeSandboxConfig
from rahc_lora.rewards.code_tests import (
    CodeTestReference,
    CodeTestsScorer,
    DockerSandboxRunner,
    SandboxExecution,
    SandboxUnavailableError,
    extract_python_source,
)


class FakeSandbox:
    def __init__(self, result: SandboxExecution) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def run(self, source: str, tests: tuple[str, ...]) -> SandboxExecution:
        self.calls.append((source, tests))
        return self.result


def test_code_reward_uses_only_explicit_sandbox_result() -> None:
    sandbox = FakeSandbox(
        SandboxExecution(
            passed=2,
            total=3,
            exit_code=0,
            timed_out=False,
            output_limited=False,
            stdout="",
            stderr="",
        )
    )
    scorer = CodeTestsScorer(sandbox)
    result = scorer.score(
        CodeTestReference(("assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3")),
        "```python\ndef f(x):\n    return x\n```",
    )
    assert result.normalized_reward == pytest.approx(2 / 3)
    assert sandbox.calls == [
        ("def f(x):\n    return x", ("assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3"))
    ]


def test_empty_or_malformed_code_response_scores_documented_zero_without_execution() -> None:
    sandbox = FakeSandbox(SandboxExecution(0, 1, 0, False, False, "", ""))
    result = CodeTestsScorer(sandbox).score(CodeTestReference(("assert True",)), "   ")
    assert result.normalized_reward == 0.0
    assert not result.parse_succeeded
    assert sandbox.calls == []
    assert extract_python_source("```\n```") is None


def test_docker_command_enforces_required_security_controls(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(CodeSandboxConfig())
    command = runner.build_command(tmp_path, "rahc-test")
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "--memory 256m" in joined
    assert "--pids-limit 32" in joined
    assert "readonly" in joined
    assert "@sha256:" in joined


def test_unavailable_docker_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rahc_lora.rewards.code_tests.shutil.which", lambda _: None)
    runner = DockerSandboxRunner(CodeSandboxConfig())
    with pytest.raises(SandboxUnavailableError, match="no unsafe subprocess fallback"):
        runner.ensure_available()
