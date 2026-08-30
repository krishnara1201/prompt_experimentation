from unittest.mock import AsyncMock

import httpx
import pytest

from app.adapters.base import ModelResponse
from app.tasks import worker

GOOD_RESPONSE = ModelResponse(
    text="SCORE: 5\nRATIONALE: Correctly identifies positive sentiment.",
    latency_ms=10.0,
    prompt_tokens=40,
    completion_tokens=8,
)
MALFORMED_RESPONSE = ModelResponse(text="not a score", latency_ms=5.0, prompt_tokens=10, completion_tokens=2)


class FakeJudgeAdapter:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate(self, prompt):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_scores_and_persists_on_success(monkeypatch):
    adapter = FakeJudgeAdapter([GOOD_RESPONSE])
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: adapter)
    monkeypatch.setattr(
        worker, "_load_run_result_for_judging", AsyncMock(return_value=("text", "positive", "The tone is positive."))
    )
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)

    worker.execute_judge_call(run_result_id=7)

    assert adapter.calls == 1
    persist_mock.assert_awaited_once()
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"
    assert kwargs["score"] == 5
    assert "positive sentiment" in kwargs["rationale"]


def test_run_result_not_found_persists_failure(monkeypatch):
    monkeypatch.setattr(worker, "_load_run_result_for_judging", AsyncMock(return_value=None))
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)

    worker.execute_judge_call(run_result_id=999)

    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"
    assert "not found" in kwargs["error_message"]


def test_malformed_judge_response_does_not_retry(monkeypatch):
    adapter = FakeJudgeAdapter([MALFORMED_RESPONSE] * 4)
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: adapter)
    monkeypatch.setattr(worker, "_load_run_result_for_judging", AsyncMock(return_value=("text", "positive", "hmm")))
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_judge_call(run_result_id=7)

    assert adapter.calls == 1
    assert sleep_calls == []
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "failed"


def test_retries_transient_judge_errors(monkeypatch):
    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(500, request=request)
    transient = httpx.HTTPStatusError("boom", request=request, response=response)

    adapter = FakeJudgeAdapter([transient, GOOD_RESPONSE])
    monkeypatch.setattr(worker, "load_judge_arm", lambda path: adapter)
    monkeypatch.setattr(worker, "_load_run_result_for_judging", AsyncMock(return_value=("text", "positive", "ok")))
    persist_mock = AsyncMock()
    monkeypatch.setattr(worker, "_persist_judge_result", persist_mock)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))

    worker.execute_judge_call(run_result_id=7)

    assert adapter.calls == 2
    # jittered exponential backoff: the single retry sleeps within [0.5, 1.0]
    assert len(sleep_calls) == 1
    assert 0.5 <= sleep_calls[0] <= 1.0
    _, kwargs = persist_mock.call_args
    assert kwargs["status"] == "completed"
