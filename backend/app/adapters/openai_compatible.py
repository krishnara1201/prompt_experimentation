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
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.price_per_1m_input = price_per_1m_input
        self.price_per_1m_output = price_per_1m_output
        self.timeout = timeout

    def generate(self, prompt: str) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        data = response.json()

        text = data["choices"][0]["message"]["content"]
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
        )
