from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.adapters.base import ModelResponse
from app.judge.scorer import JudgeParseError
from app.tasks import worker

SUCCESS = ModelResponse(text="positive", latency_ms=10.0, prompt_tokens=5, completion_tokens=1, cost_estimate_usd=0.0001)


class FakeAdapter:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_sends_task_framed_prompt_not_bare_example_text(monkeypatch):
    adapter = FakeAdapter([SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    monkeypatch.setattr(worker, "_persist_run_result", AsyncMock())
    monkeypatch.setattr(worker, "run_judge_call", MagicMock())

    example_text = "Widgets Inc reported record profits ."
    worker.execute_call(run_id=1, example_id=2, example_text=example_text, arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1
    sent_prompt = adapter.prompts[0]
    # A bare, unframed sentence is ambiguous input -- different models guess
    # the implied task ("classify this sentence's sentiment") with different
    # reliability. The prompt sent to the model must say what to do, not
    # just forward the raw eval-example text.
    assert sent_prompt != example_text
    assert example_text in sent_prompt
    assert "sentiment" in sent_prompt.lower()


def test_succeeds_on_first_try(monkeypatch):
    adapter = FakeAdapter([SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    monkeypatch.setattr(worker, "run_judge_call", MagicMock())
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1
    assert sleep_calls == []
    persist_mock.assert_awaited_once()
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"
    assert kwargs["response"] is SUCCESS


def test_retries_then_succeeds(monkeypatch):
    adapter = FakeAdapter([RuntimeError("timeout"), RuntimeError("timeout"), SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    monkeypatch.setattr(worker, "run_judge_call", MagicMock())
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 3
    assert sleep_calls == [1.0, 2.0]
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"


def test_persists_failure_after_exhausting_retries(monkeypatch):
    adapter = FakeAdapter([RuntimeError("boom")] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(
        run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0, max_retries=3
    )

    assert adapter.calls == 4  # initial attempt + 3 retries
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "boom" in kwargs["error_message"]


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.parametrize(
    "exc, retryable",
    [
        (RuntimeError("No API key found in environment variable 'OPENAI_API_KEY'"), False),
        (RuntimeError("Claude Code CLI is not authenticated: not logged in"), False),
        (RuntimeError("Codex CLI is not authenticated: please run codex login"), False),
        (RuntimeError("Claude Code CLI binary 'claude' not found on PATH"), False),
        (_http_status_error(400), False),
        (_http_status_error(401), False),
        (_http_status_error(404), False),
        (_http_status_error(429), True),
        (_http_status_error(500), True),
        (_http_status_error(503), True),
        (httpx.ConnectError("network down"), True),
        (httpx.ReadTimeout("slow"), True),
        (RuntimeError("transient blip"), True),
        (JudgeParseError("bad format"), False),
    ],
)
def test_is_retryable_classification(exc, retryable):
    assert worker.is_retryable(exc) is retryable


def test_does_not_retry_missing_api_key(monkeypatch):
    exc = RuntimeError("No API key found in environment variable 'OPENAI_API_KEY'")
    adapter = FakeAdapter([exc] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1  # no retries, no backoff sleep
    assert sleep_calls == []
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "No API key" in kwargs["error_message"]


def test_does_not_retry_unauthenticated_subscription_cli(monkeypatch):
    exc = RuntimeError("Claude Code CLI is not authenticated: not logged in")
    adapter = FakeAdapter([exc] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1  # no retries, no backoff sleep
    assert sleep_calls == []
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "is not authenticated" in kwargs["error_message"]


def test_does_not_retry_4xx(monkeypatch):
    adapter = FakeAdapter([_http_status_error(400)] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 1
    assert sleep_calls == []
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"


def test_still_retries_429(monkeypatch):
    adapter = FakeAdapter([_http_status_error(429), _http_status_error(429), SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    monkeypatch.setattr(worker, "run_judge_call", MagicMock())
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert adapter.calls == 3
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"


def test_enqueues_judge_call_after_successful_persist(monkeypatch):
    adapter = FakeAdapter([SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock(return_value=42)
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    judge_task_mock = MagicMock()
    monkeypatch.setattr(worker, "run_judge_call", judge_task_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    judge_task_mock.delay.assert_called_once_with(run_result_id=42)


def test_does_not_enqueue_judge_call_on_failure(monkeypatch):
    adapter = FakeAdapter([RuntimeError("boom")] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    monkeypatch.setattr(worker, "_persist_run_result", AsyncMock())
    judge_task_mock = MagicMock()
    monkeypatch.setattr(worker, "run_judge_call", judge_task_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0, max_retries=3)

    judge_task_mock.delay.assert_not_called()


def test_db_failure_after_success_does_not_retry_the_model_call(monkeypatch):
    """A billed model call must never be repeated because the DB write failed."""
    adapter = FakeAdapter([SUCCESS, SUCCESS, SUCCESS, SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock(side_effect=OSError("postgres is down"))
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    with pytest.raises(OSError):
        worker.execute_call(
            run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0
        )

    assert adapter.calls == 1
    assert persist_mock.await_count == 1


def test_db_failure_on_terminal_persist_does_not_crash_the_task(monkeypatch, caplog):
    adapter = FakeAdapter([RuntimeError("boom")] * 4)
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    monkeypatch.setattr(worker, "_persist_run_result", AsyncMock(side_effect=OSError("postgres is down")))
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    with caplog.at_level("ERROR"):
        worker.execute_call(
            run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0
        )

    assert "Failed to persist failed RunResult" in caplog.text


def test_unknown_arm_persists_failed_row_without_retrying(monkeypatch):
    monkeypatch.setattr(worker, "load_arms", lambda path: {"other-arm": object()})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    assert sleep_calls == []
    persist_mock.assert_awaited_once()
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "Could not resolve arm 'fake-arm'" in kwargs["error_message"]


def test_malformed_arms_config_persists_failed_row(monkeypatch):
    def _boom(path):
        raise ValueError("malformed arms.yaml")

    monkeypatch.setattr(worker, "load_arms", _boom)
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)

    worker.execute_call(run_id=1, example_id=2, example_text="hi", arm_name="fake-arm", repeat_index=0)

    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "malformed arms.yaml" in kwargs["error_message"]
