from unittest.mock import AsyncMock

from app.adapters.base import ModelResponse
from app.tasks import worker

SUCCESS = ModelResponse(text="positive", latency_ms=10.0, prompt_tokens=5, completion_tokens=1, cost_estimate_usd=0.0001)


class FakeAdapter:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate(self, prompt):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_succeeds_on_first_try(monkeypatch):
    adapter = FakeAdapter([SUCCESS])
    monkeypatch.setattr(worker, "load_arms", lambda path: {"fake-arm": adapter})
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_run_result", persist_mock)
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
