import subprocess
import tempfile
import time

from app.adapters.base import ModelResponse


class CodexCLIAdapter:
    def __init__(
        self,
        model: str,
        binary: str = "codex",
        timeout: float = 120.0,
    ):
        self.model = model
        self.binary = binary
        self.timeout = timeout
        self.celery_queue = "subscription_cli"

    def generate(self, prompt: str) -> ModelResponse:
        with tempfile.TemporaryDirectory(prefix="codex-cli-arm-") as scratch_dir:
            cmd = [
                self.binary,
                "exec",
                prompt,
                "--model",
                self.model,
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "never",
                "--cd",
                scratch_dir,
            ]
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
                    f"Codex CLI binary '{self.binary}' not found on PATH"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _looks_unauthenticated(stderr):
                raise RuntimeError(f"Codex CLI is not authenticated: {stderr}")
            raise RuntimeError(f"Codex CLI exited with {result.returncode}: {stderr}")

        return ModelResponse(
            text=result.stdout.strip(),
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            cost_estimate_usd=None,
            finish_reason=None,
        )


def _looks_unauthenticated(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        phrase in lowered
        for phrase in ("not logged in", "not authenticated", "please log in", "codex login")
    )
