import os
import time

import httpx

from app.adapters.base import ModelResponse


class OpenAICompatibleAdapter:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        price_per_1m_input: float | None = None,
        price_per_1m_output: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        extra_body: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.price_per_1m_input = price_per_1m_input
        self.price_per_1m_output = price_per_1m_output
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Extra request-body keys merged verbatim into every chat-completions
        # call — e.g. `reasoning_effort: none` to disable a thinking model's
        # native reasoning so an arm's prompt_template is the only reasoning
        # driver. Provider-specific; unknown keys are ignored by most.
        self.extra_body = extra_body or {}

    def generate(self, prompt: str) -> ModelResponse:
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if self.api_key_env and not api_key:
            raise RuntimeError(
                f"No API key found in environment variable '{self.api_key_env}'"
            )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_body)

        start = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        data = response.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        cost_estimate_usd = None
        if self.price_per_1m_input is not None and self.price_per_1m_output is not None:
            cost_estimate_usd = (
                prompt_tokens / 1_000_000 * self.price_per_1m_input
                + completion_tokens / 1_000_000 * self.price_per_1m_output
            )

        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_estimate_usd=cost_estimate_usd,
            finish_reason=finish_reason,
        )
