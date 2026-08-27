import json
import subprocess
import tempfile
import time

from app.adapters.base import ModelResponse


class ClaudeCodeCLIAdapter:
    def __init__(
        self,
        model: str,
        binary: str = "claude",
        timeout: float = 120.0,
    ):
        self.model = model
        self.binary = binary
        self.timeout = timeout
        self.celery_queue = "subscription_cli"

    def generate(self, prompt: str) -> ModelResponse:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--dangerously-skip-permissions",
        ]

        with tempfile.TemporaryDirectory(prefix="claude-code-cli-arm-") as scratch_dir:
            start = time.perf_counter()
            try:
                result = subprocess.run(
                    cmd,
                    cwd=scratch_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Claude Code CLI binary '{self.binary}' not found on PATH"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _looks_unauthenticated(stderr):
                raise RuntimeError(f"Claude Code CLI is not authenticated: {stderr}")
            raise RuntimeError(f"Claude Code CLI exited with {result.returncode}: {stderr}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude Code CLI returned non-JSON output: {result.stdout[:500]!r}"
            ) from exc

        if data.get("is_error"):
            raise RuntimeError(f"Claude Code CLI reported an error: {data.get('result')}")

        usage = data.get("usage", {})
        return ModelResponse(
            text=data.get("result", ""),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            cost_estimate_usd=None,
            finish_reason=data.get("subtype"),
        )


def _looks_unauthenticated(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        phrase in lowered
        for phrase in ("not logged in", "not authenticated", "please log in", "claude login")
    )
