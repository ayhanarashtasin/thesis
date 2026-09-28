"""Fail-closed, resource-limited Docker execution for untrusted generated code."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from rahc_lora.config.schema import CodeSandboxConfig
from rahc_lora.rewards.base import RewardResult

_RESULT_PREFIX = "__RAHC_MBPP_RESULT__="


class SandboxUnavailableError(RuntimeError):
    """Raised when the configured isolation backend cannot be enforced."""


class SandboxInfrastructureError(RuntimeError):
    """Raised when Docker fails independently of the submitted program."""


@dataclass(frozen=True)
class CodeTestReference:
    """Hidden assertions used to score one generated program."""

    tests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tests or any(not test.strip() for test in self.tests):
            raise ValueError("CodeTestReference requires nonempty test sources")


@dataclass(frozen=True)
class SandboxExecution:
    """Sanitized outcome returned across the code-execution boundary."""

    passed: int
    total: int
    exit_code: int
    timed_out: bool
    output_limited: bool
    stdout: str
    stderr: str

    @property
    def reward(self) -> float:
        return self.passed / self.total if self.total else 0.0


class CodeSandbox(Protocol):
    """Isolation backend used by the MBPP reward adapter."""

    def run(self, source: str, tests: tuple[str, ...]) -> SandboxExecution:
        """Execute untrusted source and return only bounded test diagnostics."""


def extract_python_source(response: str) -> str | None:
    """Extract a Python fenced block or accept a nonempty plain response."""

    stripped = response.strip()
    if not stripped:
        return None
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    candidates: list[str] = []
    for index in range(1, len(parts), 2):
        block = parts[index].strip()
        if block.casefold().startswith("python"):
            block = block[6:].lstrip("\r\n ")
        if block:
            candidates.append(block)
    return candidates[-1] if candidates else None


def _build_harness(source: str, tests: tuple[str, ...]) -> str:
    return f"""import json as _json

_source = {source!r}
_tests = {tests!r}
_namespace = {{"__name__": "__submission__"}}
_passed = 0
_load_error = None
try:
    exec(compile(_source, "<submission>", "exec"), _namespace, _namespace)
except BaseException as _error:
    _load_error = f"{{type(_error).__name__}}: {{_error}}"

if _load_error is None:
    for _test in _tests:
        try:
            exec(compile(_test, "<hidden-test>", "exec"), _namespace, _namespace)
        except BaseException:
            pass
        else:
            _passed += 1

print(
    {_RESULT_PREFIX!r}
    + _json.dumps(
        {{"passed": _passed, "total": len(_tests), "load_error": _load_error}},
        sort_keys=True,
    )
)
"""


class DockerSandboxRunner:
    """Run generated code in an ephemeral, networkless, read-only Docker container."""

    def __init__(self, config: CodeSandboxConfig) -> None:
        self.config = config

    def ensure_available(self) -> None:
        """Fail before scoring when Docker or the daemon is unavailable."""

        if shutil.which("docker") is None:
            raise SandboxUnavailableError(
                "MBPP scoring requires Docker, but the docker executable is unavailable; "
                "no unsafe subprocess fallback is permitted"
            )
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxUnavailableError(f"Docker availability check failed: {error}") from error
        if result.returncode != 0 or not result.stdout.strip():
            diagnostic = result.stderr.strip() or "Docker daemon did not report a server version"
            raise SandboxUnavailableError(f"Docker isolation backend is unavailable: {diagnostic}")

    def build_command(self, workspace: Path, container_name: str) -> list[str]:
        """Build the auditable Docker security boundary command."""

        resolved = workspace.resolve()
        if "," in str(resolved):
            raise SandboxInfrastructureError("Docker sandbox workspace path may not contain commas")
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            f"{self.config.memory_mb}m",
            "--memory-swap",
            f"{self.config.memory_mb}m",
            "--cpus",
            str(self.config.cpus),
            "--pids-limit",
            str(self.config.pids_limit),
            "--ulimit",
            f"cpu={max(1, math.ceil(self.config.timeout_seconds))}",
            "--ulimit",
            "nofile=64:64",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,source={resolved},target=/workspace,readonly",
            "--workdir",
            "/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            self.config.image,
            "python",
            "-I",
            "-B",
            "/workspace/harness.py",
        ]

    @staticmethod
    def _force_remove(container_name: str) -> None:
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            check=False,
            capture_output=True,
            timeout=10,
        )

    def _read_bounded_output(
        self,
        stream: BinaryIO,
        destination: bytearray,
        total: list[int],
        lock: threading.Lock,
        limit_reached: threading.Event,
        process: subprocess.Popen[bytes],
    ) -> None:
        while chunk := stream.read(4096):
            with lock:
                remaining = max(0, self.config.output_limit_bytes - total[0])
                destination.extend(chunk[:remaining])
                total[0] += len(chunk)
                if total[0] > self.config.output_limit_bytes:
                    limit_reached.set()
                    process.kill()
                    return

    def run(self, source: str, tests: tuple[str, ...]) -> SandboxExecution:
        """Execute code with no network and strict CPU, memory, process, file, and output limits."""

        if not source.strip():
            raise ValueError("Sandbox source must not be empty")
        if not tests:
            raise ValueError("Sandbox execution requires at least one hidden test")
        self.ensure_available()
        container_name = f"rahc-mbpp-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="rahc-mbpp-") as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "harness.py").write_text(
                _build_harness(source, tests), encoding="utf-8", newline="\n"
            )
            command = self.build_command(workspace, container_name)
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as error:
                raise SandboxInfrastructureError(
                    f"Failed to start Docker sandbox: {error}"
                ) from error
            if process.stdout is None or process.stderr is None:
                process.kill()
                raise SandboxInfrastructureError("Docker sandbox pipes were not created")

            stdout_bytes = bytearray()
            stderr_bytes = bytearray()
            total = [0]
            lock = threading.Lock()
            limit_reached = threading.Event()
            readers = [
                threading.Thread(
                    target=self._read_bounded_output,
                    args=(
                        stream,
                        destination,
                        total,
                        lock,
                        limit_reached,
                        process,
                    ),
                    daemon=True,
                )
                for stream, destination in (
                    (process.stdout, stdout_bytes),
                    (process.stderr, stderr_bytes),
                )
            ]
            for reader in readers:
                reader.start()
            timed_out = False
            try:
                exit_code = process.wait(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                exit_code = process.wait(timeout=5)
            finally:
                if timed_out or limit_reached.is_set():
                    self._force_remove(container_name)
            for reader in readers:
                reader.join(timeout=5)

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        passed = 0
        total_tests = len(tests)
        for line in stdout.splitlines():
            if not line.startswith(_RESULT_PREFIX):
                continue
            try:
                payload = json.loads(line.removeprefix(_RESULT_PREFIX))
                passed = int(payload["passed"])
                reported_total = int(payload["total"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise SandboxInfrastructureError(
                    f"Malformed sandbox result marker: {error}"
                ) from error
            if reported_total != total_tests or not 0 <= passed <= total_tests:
                raise SandboxInfrastructureError("Sandbox result counts violate the test contract")

        return SandboxExecution(
            passed=passed,
            total=total_tests,
            exit_code=exit_code,
            timed_out=timed_out,
            output_limited=limit_reached.is_set(),
            stdout=stdout,
            stderr=stderr,
        )


@dataclass(frozen=True)
class CodeTestsScorer:
    """Fraction-of-hidden-tests reward using only an explicit sandbox backend."""

    sandbox: CodeSandbox
    reward_version: str = "isolated-hidden-tests-v1"

    def score(self, reference: CodeTestReference, response: str) -> RewardResult:
        source = extract_python_source(response)
        if source is None:
            return RewardResult(
                raw_reward=0.0,
                normalized_reward=0.0,
                parse_succeeded=False,
                parsed_response=None,
                reward_version=self.reward_version,
                details={"passed": 0, "total": len(reference.tests), "parsing_failure": True},
            )
        execution = self.sandbox.run(source, reference.tests)
        return RewardResult(
            raw_reward=execution.reward,
            normalized_reward=execution.reward,
            parse_succeeded=True,
            parsed_response="python_source",
            reward_version=self.reward_version,
            details={
                "passed": execution.passed,
                "total": execution.total,
                "timed_out": execution.timed_out,
                "output_limited": execution.output_limited,
                "exit_code": execution.exit_code,
            },
        )
