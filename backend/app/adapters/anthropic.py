import os
import time

import httpx

from app.adapters.base import ModelResponse

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter:
    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 1024,
        price_per_1m_input: float | None = None,
        price_per_1m_output: float | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.price_per_1m_input = price_per_1m_input
        self.price_per_1m_output = price_per_1m_output
        self.timeout = timeout

    def generate(self, prompt: str) -> ModelResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key found in environment variable '{self.api_key_env}'"
            )

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        start = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/messages",
            headers=headers,
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        data = response.json()

        text = data["content"][0]["text"]
        finish_reason = data.get("stop_reason")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

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
