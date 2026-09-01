import json
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from app.adapters.base import ModelResponse


class UsageLimitError(RuntimeError):
    """The Claude Code subscription seat hit its usage limit.

    Distinct from an ordinary failure: the caller should stop and retry
    after ``retry_at`` (a multi-hour subscription window), not record a
    failed result and move on. Subclasses ``RuntimeError`` so the Celery
    worker's ``is_retryable`` / ``is_rate_limited`` classification (which
    matches on ``"usage limit"`` in the message) keeps working unchanged.
    """

    def __init__(self, message: str, retry_at: datetime | None = None):
        super().__init__(message)
        self.retry_at = retry_at


# "Claude AI usage limit reached|1893456000" -- the trailing pipe + unix ts.
_USAGE_LIMIT_TS_RE = re.compile(r"usage limit reached\|(\d+)", re.IGNORECASE)
_USAGE_LIMIT_STDERR_PHRASES = (
    "usage limit",
    "rate limit",
    "too many requests",
    "5-hour limit",
    "weekly limit",
    "resource_exhausted",
)


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

        data = _try_parse_json(result.stdout)

        limit = _usage_limit(result.stdout, result.stderr, data)
        if limit is not None:
            raise limit

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _looks_unauthenticated(stderr):
                raise RuntimeError(f"Claude Code CLI is not authenticated: {stderr}")
            raise RuntimeError(f"Claude Code CLI exited with {result.returncode}: {stderr}")

        if data is None:
            raise RuntimeError(
                f"Claude Code CLI returned non-JSON output: {result.stdout[:500]!r}"
            )

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


def _try_parse_json(stdout: str) -> dict | None:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _usage_limit(stdout: str, stderr: str, data: dict | None) -> UsageLimitError | None:
    """Return a ``UsageLimitError`` if any signal says the seat is limited,
    else ``None``. A bare non-zero exit with empty stderr and no JSON is
    NOT a usage limit -- that heuristic belongs to backoff selection, not a
    multi-hour pause."""
    retry_at: datetime | None = None
    ts_match = _USAGE_LIMIT_TS_RE.search(stdout) or _USAGE_LIMIT_TS_RE.search(stderr)
    if ts_match:
        try:
            retry_at = datetime.fromtimestamp(int(ts_match.group(1)), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            retry_at = None

    if "usage limit reached" in stdout.lower():
        return UsageLimitError(
            f"Claude Code CLI usage limit reached: {stdout.strip()[:200]}", retry_at
        )

    if data is not None:
        status = str(data.get("api_error_status") or "").lower()
        subtype = str(data.get("subtype") or "").lower()
        result_text = str(data.get("result") or "")
        if "rate_limit" in status or status == "429":
            return UsageLimitError(
                f"Claude Code CLI reported a rate/usage limit (api_error_status={status!r})",
                retry_at,
            )
        if "limit" in subtype:
            return UsageLimitError(
                f"Claude Code CLI reported a usage limit (subtype={subtype!r})", retry_at
            )
        if "usage limit reached" in result_text.lower():
            return UsageLimitError(
                f"Claude Code CLI usage limit reached: {result_text.strip()[:200]}", retry_at
            )

    lowered_stderr = stderr.lower()
    if any(phrase in lowered_stderr for phrase in _USAGE_LIMIT_STDERR_PHRASES):
        return UsageLimitError(
            f"Claude Code CLI usage limit (stderr): {stderr.strip()[:200]}", retry_at
        )

    return None


def _looks_unauthenticated(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        phrase in lowered
        for phrase in ("not logged in", "not authenticated", "please log in", "claude login")
    )
