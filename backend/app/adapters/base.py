from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_estimate_usd: float | None = None
    finish_reason: str | None = None


class ModelAdapter(Protocol):
    def generate(self, prompt: str) -> ModelResponse: ...
