import httpx
import pytest

from app.judge.scorer import JudgeParseError
from app.tasks import worker


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.parametrize(
    "exc, rate_limited",
    [
        (_http_status_error(429), True),
        (_http_status_error(503), True),
        (_http_status_error(529), True),
        (_http_status_error(500), False),
        (_http_status_error(400), False),
        # Subscription CLI seat hits its usage limit -> non-zero exit, empty stderr.
        (RuntimeError("Claude Code CLI exited with 1: "), True),
        (RuntimeError("Codex CLI exited with 1: "), True),
        (RuntimeError("Anthropic API error: overloaded_error"), True),
        (RuntimeError("Error 429: Too Many Requests"), True),
        (RuntimeError("rate limit exceeded, retry later"), True),
        # Exit non-zero WITH a real detail message is not a rate limit.
        (RuntimeError("Claude Code CLI exited with 1: unexpected token in JSON"), False),
        (RuntimeError("transient blip"), False),
        (JudgeParseError("bad format"), False),
    ],
)
def test_is_rate_limited_classification(exc, rate_limited):
    assert worker.is_rate_limited(exc) is rate_limited


def test_rate_limited_errors_stay_retryable():
    # is_rate_limited only selects the backoff schedule; it must never turn an
    # error that is_retryable would retry into a non-retryable one.
    for exc in (
        _http_status_error(429),
        _http_status_error(503),
        RuntimeError("Claude Code CLI exited with 1: "),
    ):
        assert worker.is_retryable(exc) is True


def test_standard_backoff_grows_exponentially_under_max_jitter(monkeypatch):
    monkeypatch.setattr(worker.random, "uniform", lambda a, b: b)
    assert worker._backoff_seconds(1, rate_limited=False) == 1.0
    assert worker._backoff_seconds(2, rate_limited=False) == 2.0
    assert worker._backoff_seconds(3, rate_limited=False) == 4.0


def test_backoff_has_a_floor_from_equal_jitter(monkeypatch):
    monkeypatch.setattr(worker.random, "uniform", lambda a, b: a)
    # equal jitter -> the sleep is never below half the capped delay
    assert worker._backoff_seconds(2, rate_limited=False) == 1.0


def test_rate_limit_backoff_is_much_longer_than_standard(monkeypatch):
    monkeypatch.setattr(worker.random, "uniform", lambda a, b: a)
    assert worker._backoff_seconds(1, rate_limited=True) >= 4.0
    assert worker._backoff_seconds(1, rate_limited=True) > worker._backoff_seconds(
        1, rate_limited=False
    )


def test_rate_limit_backoff_is_capped(monkeypatch):
    monkeypatch.setattr(worker.random, "uniform", lambda a, b: b)
    assert (
        worker._backoff_seconds(20, rate_limited=True)
        == worker.RATE_LIMIT_BACKOFF_CAP_SECONDS
    )


class _Flaky:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_retry_gives_rate_limited_errors_more_attempts_than_standard(monkeypatch):
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    fn = _Flaky([_http_status_error(429)] * worker.RATE_LIMIT_MAX_RETRIES + ["ok"])
    assert worker._retry_model_call(fn, standard_max_retries=3) == "ok"
    assert fn.calls == worker.RATE_LIMIT_MAX_RETRIES + 1


def test_retry_reraises_after_exhausting_rate_limit_attempts(monkeypatch):
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    fn = _Flaky([_http_status_error(429)] * 50)
    with pytest.raises(httpx.HTTPStatusError):
        worker._retry_model_call(fn, standard_max_retries=3)
    assert fn.calls == worker.RATE_LIMIT_MAX_RETRIES + 1


def test_retry_stops_immediately_on_non_retryable(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleep_calls.append(s))
    fn = _Flaky([RuntimeError("No API key found in environment variable 'X'")] * 3)
    with pytest.raises(RuntimeError):
        worker._retry_model_call(fn, standard_max_retries=3)
    assert fn.calls == 1
    assert sleep_calls == []


def test_run_judge_call_carries_a_celery_rate_limit():
    assert worker.run_judge_call.rate_limit == worker.JUDGE_RATE_LIMIT
    assert worker.JUDGE_RATE_LIMIT.endswith("/m")
