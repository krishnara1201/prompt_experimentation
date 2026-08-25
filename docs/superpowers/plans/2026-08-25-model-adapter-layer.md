# Model Adapter Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified adapter interface so local (Ollama) and hosted API models are interchangeable behind one code path, and prove it end-to-end by running a handful of prompts through a local arm and API arms.

**Architecture:** Two adapter implementations behind a shared `ModelAdapter` protocol — `OpenAICompatibleAdapter` (Ollama, OpenAI, and any other OpenAI-schema provider) and `AnthropicAdapter` (Claude's distinct schema). Arms are declared entirely in a YAML config (provider, model, API-key env-var name, optional pricing), never hardcoded in code, so adding/swapping a provider is a config edit.

**Tech Stack:** Python 3.12, `uv`, `httpx` (HTTP), `pyyaml` (config), `pytest` + `respx` (mocked HTTP tests).

**Spec:** `docs/superpowers/specs/2026-08-25-model-adapter-layer-design.md`

## Global Constraints

- Python 3.12, `uv` tooling — matches the sibling `experimentation_copilot` repo.
- No Postgres, Celery, or frontend in this phase — those are later build phases per root `CLAUDE.md`.
- `ModelResponse.cost_estimate_usd` is `None` whenever pricing isn't configured for an arm (e.g. the local Ollama arm).
- Arms are declared in `arms.yaml`, never hardcoded per-provider in code — adding a provider must be a config edit only.
- The Ollama end-to-end test must **skip** (not fail) when Ollama isn't reachable, so the suite runs without local setup.

---

### Task 1: Repo scaffolding

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/adapters/__init__.py`
- Create: `backend/app/config/__init__.py`
- Create: `backend/tests/__init__.py`
- Test: `backend/tests/test_scaffold.py`

**Interfaces:**
- Produces: a working `uv` project at `backend/` that later tasks add code into.

- [ ] **Step 1: Create the project files**

`backend/pyproject.toml`:
```toml
[project]
name = "prompt-experimentation-backend"
version = "0.1.0"
description = "Model adapter layer for the LLM prompt-experimentation platform"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "respx>=0.21",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

`backend/app/__init__.py`: empty file.
`backend/app/adapters/__init__.py`: empty file.
`backend/app/config/__init__.py`: empty file.
`backend/tests/__init__.py`: empty file.

- [ ] **Step 2: Write a sanity test**

`backend/tests/test_scaffold.py`:
```python
def test_scaffold_sanity():
    assert True
```

- [ ] **Step 3: Sync dependencies and run the test**

Run: `cd backend && uv sync && uv run pytest -v`
Expected: `test_scaffold_sanity PASSED`, 1 passed.

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml backend/app backend/tests backend/uv.lock
git commit -m "chore: scaffold backend uv project"
```

---

### Task 2: `ModelAdapter` protocol and `ModelResponse`

**Files:**
- Create: `backend/app/adapters/base.py`
- Test: `backend/tests/adapters/__init__.py`
- Test: `backend/tests/adapters/test_base.py`

**Interfaces:**
- Produces: `ModelResponse` dataclass (`text: str`, `latency_ms: float`, `prompt_tokens: int`, `completion_tokens: int`, `cost_estimate_usd: float | None = None`) and `ModelAdapter` protocol (`generate(self, prompt: str) -> ModelResponse`) — every later adapter task implements this protocol and returns this dataclass.

- [ ] **Step 1: Write the failing test**

`backend/tests/adapters/__init__.py`: empty file.

`backend/tests/adapters/test_base.py`:
```python
from app.adapters.base import ModelResponse


def test_model_response_defaults_cost_to_none():
    response = ModelResponse(
        text="hello",
        latency_ms=12.3,
        prompt_tokens=5,
        completion_tokens=2,
    )
    assert response.cost_estimate_usd is None


def test_model_response_holds_all_fields():
    response = ModelResponse(
        text="hello",
        latency_ms=12.3,
        prompt_tokens=5,
        completion_tokens=2,
        cost_estimate_usd=0.001,
    )
    assert response.text == "hello"
    assert response.latency_ms == 12.3
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 2
    assert response.cost_estimate_usd == 0.001
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/adapters/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.adapters.base'`

- [ ] **Step 3: Write the implementation**

`backend/app/adapters/base.py`:
```python
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_estimate_usd: float | None = None


class ModelAdapter(Protocol):
    def generate(self, prompt: str) -> ModelResponse: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/adapters/test_base.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/base.py backend/tests/adapters
git commit -m "feat: add ModelAdapter protocol and ModelResponse"
```

---

### Task 3: `OpenAICompatibleAdapter`

**Files:**
- Create: `backend/app/adapters/openai_compatible.py`
- Test: `backend/tests/adapters/test_openai_compatible.py`

**Interfaces:**
- Consumes: `ModelResponse` from `app.adapters.base` (Task 2).
- Produces: `OpenAICompatibleAdapter(base_url: str, model: str, api_key_env: str | None = None, price_per_1m_input: float | None = None, price_per_1m_output: float | None = None, timeout: float = 60.0)` with `.generate(prompt: str) -> ModelResponse`. Used directly by Ollama arms and by the config loader (Task 5) for any `adapter: openai_compatible` entry.

- [ ] **Step 1: Write the failing tests**

`backend/tests/adapters/test_openai_compatible.py`:
```python
import httpx
import pytest
import respx

from app.adapters.openai_compatible import OpenAICompatibleAdapter


@respx.mock
def test_generate_returns_text_and_token_counts():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Paris"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )
    )
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    response = adapter.generate("What is the capital of France?")

    assert response.text == "Paris"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 1
    assert response.latency_ms > 0


@respx.mock
def test_generate_cost_is_none_without_pricing_config():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Paris"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )
    )
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    response = adapter.generate("What is the capital of France?")

    assert response.cost_estimate_usd is None


@respx.mock
def test_generate_computes_cost_when_pricing_configured():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
            },
        )
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        price_per_1m_input=0.15,
        price_per_1m_output=0.60,
    )

    response = adapter.generate("hello")

    assert response.cost_estimate_usd == pytest.approx(0.75)


@respx.mock
def test_generate_sends_bearer_token_when_key_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}], "usage": {}},
        )
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )

    adapter.generate("hello")

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test123"


@respx.mock
def test_generate_omits_auth_header_without_api_key_env():
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}], "usage": {}},
        )
    )
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    adapter.generate("hello")

    assert "Authorization" not in route.calls.last.request.headers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/adapters/test_openai_compatible.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.adapters.openai_compatible'`

- [ ] **Step 3: Write the implementation**

`backend/app/adapters/openai_compatible.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/adapters/test_openai_compatible.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/openai_compatible.py backend/tests/adapters/test_openai_compatible.py
git commit -m "feat: add OpenAICompatibleAdapter for Ollama/OpenAI-schema providers"
```

---

### Task 4: `AnthropicAdapter`

**Files:**
- Create: `backend/app/adapters/anthropic.py`
- Test: `backend/tests/adapters/test_anthropic.py`

**Interfaces:**
- Consumes: `ModelResponse` from `app.adapters.base` (Task 2).
- Produces: `AnthropicAdapter(model: str, api_key_env: str = "ANTHROPIC_API_KEY", base_url: str = "https://api.anthropic.com/v1", max_tokens: int = 1024, price_per_1m_input: float | None = None, price_per_1m_output: float | None = None, timeout: float = 60.0)` with `.generate(prompt: str) -> ModelResponse`. Used by the config loader (Task 5) for any `adapter: anthropic` entry.

- [ ] **Step 1: Write the failing tests**

`backend/tests/adapters/test_anthropic.py`:
```python
import httpx
import pytest
import respx

from app.adapters.anthropic import AnthropicAdapter


@respx.mock
def test_generate_returns_text_and_token_counts(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Paris"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )
    )
    adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001")

    response = adapter.generate("What is the capital of France?")

    assert response.text == "Paris"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 1
    assert response.latency_ms > 0


def test_generate_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001")

    with pytest.raises(RuntimeError):
        adapter.generate("hello")


@respx.mock
def test_generate_sends_required_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "hi"}], "usage": {}},
        )
    )
    adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001")

    adapter.generate("hello")

    sent = route.calls.last.request.headers
    assert sent["x-api-key"] == "sk-ant-test"
    assert sent["anthropic-version"] == "2023-06-01"


@respx.mock
def test_generate_computes_cost_when_pricing_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            },
        )
    )
    adapter = AnthropicAdapter(
        model="claude-haiku-4-5-20251001",
        price_per_1m_input=1.00,
        price_per_1m_output=5.00,
    )

    response = adapter.generate("hello")

    assert response.cost_estimate_usd == pytest.approx(6.00)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/adapters/test_anthropic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.adapters.anthropic'`

- [ ] **Step 3: Write the implementation**

`backend/app/adapters/anthropic.py`:
```python
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
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/adapters/test_anthropic.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/adapters/anthropic.py backend/tests/adapters/test_anthropic.py
git commit -m "feat: add AnthropicAdapter"
```

---

### Task 5: Arm config loader

**Files:**
- Create: `backend/app/config/arms.py`
- Test: `backend/tests/config/__init__.py`
- Test: `backend/tests/config/test_arms.py`

**Interfaces:**
- Consumes: `OpenAICompatibleAdapter` (Task 3), `AnthropicAdapter` (Task 4).
- Produces: `load_arms(config_path: str) -> dict[str, ModelAdapter]` and `UnknownAdapterError(ValueError)`. Used by the demo script (Task 7).

- [ ] **Step 1: Write the failing tests**

`backend/tests/config/__init__.py`: empty file.

`backend/tests/config/test_arms.py`:
```python
import pytest

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.config.arms import UnknownAdapterError, load_arms

VALID_CONFIG = """
arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b

  - name: gpt-4o-mini
    adapter: openai_compatible
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    price_per_1m_input: 0.15
    price_per_1m_output: 0.60

  - name: claude-haiku
    adapter: anthropic
    model: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
    price_per_1m_input: 1.00
    price_per_1m_output: 5.00
"""

INVALID_CONFIG = """
arms:
  - name: mystery-arm
    adapter: telepathy
    model: mind-reader-v1
"""


def test_load_arms_builds_correct_adapter_types(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    assert set(arms.keys()) == {"qwen3-8b-local", "gpt-4o-mini", "claude-haiku"}
    assert isinstance(arms["qwen3-8b-local"], OpenAICompatibleAdapter)
    assert isinstance(arms["gpt-4o-mini"], OpenAICompatibleAdapter)
    assert isinstance(arms["claude-haiku"], AnthropicAdapter)


def test_load_arms_passes_config_fields_through(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(VALID_CONFIG)

    arms = load_arms(str(config_path))

    local = arms["qwen3-8b-local"]
    assert local.base_url == "http://localhost:11434/v1"
    assert local.model == "qwen3:8b"

    hosted = arms["gpt-4o-mini"]
    assert hosted.price_per_1m_input == 0.15
    assert hosted.price_per_1m_output == 0.60


def test_load_arms_raises_on_unknown_adapter_type(tmp_path):
    config_path = tmp_path / "arms.yaml"
    config_path.write_text(INVALID_CONFIG)

    with pytest.raises(UnknownAdapterError):
        load_arms(str(config_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/config/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config.arms'`

- [ ] **Step 3: Write the implementation**

`backend/app/config/arms.py`:
```python
import yaml

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.base import ModelAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter


class UnknownAdapterError(ValueError):
    pass


ADAPTER_TYPES = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}


def load_arms(config_path: str) -> dict[str, ModelAdapter]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    arms: dict[str, ModelAdapter] = {}
    for entry in raw["arms"]:
        entry = dict(entry)
        name = entry.pop("name")
        adapter_type = entry.pop("adapter")

        adapter_cls = ADAPTER_TYPES.get(adapter_type)
        if adapter_cls is None:
            raise UnknownAdapterError(
                f"Unknown adapter type '{adapter_type}' for arm '{name}'"
            )

        arms[name] = adapter_cls(**entry)

    return arms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/config/test_arms.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/arms.py backend/tests/config
git commit -m "feat: add config-driven arm loader"
```

---

### Task 6: Ollama end-to-end test

**Files:**
- Test: `backend/tests/adapters/test_ollama_e2e.py`

**Interfaces:**
- Consumes: `OpenAICompatibleAdapter` from `app.adapters.openai_compatible` (Task 3).

- [ ] **Step 1: Write the test**

`backend/tests/adapters/test_ollama_e2e.py`:
```python
import socket

import pytest

from app.adapters.openai_compatible import OpenAICompatibleAdapter


def _ollama_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running on localhost:11434")
def test_qwen3_generates_nonempty_text():
    adapter = OpenAICompatibleAdapter(base_url="http://localhost:11434/v1", model="qwen3:8b")

    response = adapter.generate("Reply with the single word: hello")

    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.latency_ms > 0
    assert response.cost_estimate_usd is None
```

- [ ] **Step 2: Run the test and confirm it behaves correctly for your environment**

Run: `cd backend && uv run pytest tests/adapters/test_ollama_e2e.py -v`
Expected: either `1 passed` (if Ollama is running locally with `qwen3:8b` pulled — run `ollama pull qwen3:8b` first if needed) or `1 skipped` (if Ollama isn't running). Both outcomes are correct; the test must never fail the suite when Ollama is simply absent.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/adapters/test_ollama_e2e.py
git commit -m "test: add Ollama end-to-end test (skips if unreachable)"
```

---

### Task 7: Demo script, arm config, and README

**Files:**
- Create: `backend/arms.yaml`
- Create: `backend/app/demo.py`
- Create: `backend/README.md`
- Create: `backend/.env.example`
- Test: `backend/tests/test_demo.py`

**Interfaces:**
- Consumes: `load_arms` from `app.config.arms` (Task 5), `ModelResponse` from `app.adapters.base` (Task 2).
- Produces: `format_row(arm_name: str, prompt: str, response: ModelResponse) -> str` and `main() -> None`, the Phase 1 "done" deliverable — running every configured arm over a handful of prompts and printing results side by side.

- [ ] **Step 1: Write the failing test for the pure formatting function**

`backend/tests/test_demo.py`:
```python
from app.adapters.base import ModelResponse
from app.demo import format_row


def test_format_row_shows_cost_when_known():
    response = ModelResponse(
        text="hi",
        latency_ms=123.4,
        prompt_tokens=10,
        completion_tokens=2,
        cost_estimate_usd=0.000015,
    )
    row = format_row("gpt-4o-mini", "hello", response)
    assert "gpt-4o-mini" in row
    assert "hi" in row
    assert "0.000015" in row


def test_format_row_shows_na_when_cost_unknown():
    response = ModelResponse(
        text="hi",
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        cost_estimate_usd=None,
    )
    row = format_row("qwen3-8b-local", "hello", response)
    assert "n/a" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_demo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.demo'`

- [ ] **Step 3: Write the arm config**

`backend/arms.yaml`:
```yaml
# Pricing figures below are illustrative — verify against each provider's
# current pricing page before using cost_estimate_usd for real comparisons.
arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen3:8b

  - name: gpt-4o-mini
    adapter: openai_compatible
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    price_per_1m_input: 0.15
    price_per_1m_output: 0.60

  - name: claude-haiku
    adapter: anthropic
    model: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
    price_per_1m_input: 1.00
    price_per_1m_output: 5.00
```

- [ ] **Step 4: Write the demo script**

`backend/app/demo.py`:
```python
from app.adapters.base import ModelResponse
from app.config.arms import load_arms

PROMPTS = [
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Is the following sentence positive, negative, or neutral: "
    "'The company's stock dropped sharply after the earnings call.'",
]


def format_row(arm_name: str, prompt: str, response: ModelResponse) -> str:
    cost = (
        f"${response.cost_estimate_usd:.6f}"
        if response.cost_estimate_usd is not None
        else "n/a"
    )
    return (
        f"[{arm_name}] prompt={prompt!r}\n"
        f"  text={response.text!r}\n"
        f"  latency_ms={response.latency_ms:.1f} "
        f"prompt_tokens={response.prompt_tokens} "
        f"completion_tokens={response.completion_tokens} "
        f"cost={cost}\n"
    )


def main() -> None:
    arms = load_arms("arms.yaml")
    for prompt in PROMPTS:
        for arm_name, adapter in arms.items():
            try:
                response = adapter.generate(prompt)
            except Exception as exc:
                print(f"[{arm_name}] prompt={prompt!r} FAILED: {exc}")
                continue
            print(format_row(arm_name, prompt, response))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_demo.py -v`
Expected: 2 passed.

- [ ] **Step 6: Write `.env.example` and `README.md`**

`backend/.env.example`:
```
# Copy to .env and fill in whichever keys you have. Arms whose key is
# missing will print FAILED in the demo output instead of crashing it.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

`backend/README.md`:
```markdown
# Backend — Model Adapter Layer

Phase 1 of the LLM prompt-experimentation platform: a unified adapter
interface so local (Ollama) and hosted API models are interchangeable
behind one code path. See `docs/superpowers/specs/2026-08-25-model-adapter-layer-design.md`
at the repo root for the design.

## Setup

```bash
uv sync
cp .env.example .env   # fill in whichever API keys you have; local Ollama needs none
```

Local model: `ollama pull qwen3:8b` (requires [Ollama](https://ollama.com) running).

## Run the demo

```bash
uv run python -m app.demo
```

Runs every arm in `arms.yaml` over a handful of prompts and prints
text/latency/tokens/cost side by side. Arms with no configured API key fail
gracefully (printed as `FAILED`) rather than crashing the run.

## Add or swap an arm

Edit `arms.yaml` — no code changes needed. `adapter: openai_compatible`
works for Ollama and any provider using the OpenAI chat-completions schema
(OpenAI, OpenRouter, Groq, Together, etc.); `adapter: anthropic` is for
Claude models.

## Tests

```bash
uv run pytest -v
```

The Ollama end-to-end test skips automatically if Ollama isn't running
locally.
```

- [ ] **Step 7: Run the full test suite**

Run: `cd backend && uv run pytest -v`
Expected: all tests pass (Ollama e2e test passes or skips depending on local setup).

- [ ] **Step 8: Commit**

```bash
git add backend/arms.yaml backend/app/demo.py backend/README.md backend/.env.example backend/tests/test_demo.py
git commit -m "feat: add demo script, arm config, and backend README"
```

---

## Definition of done

- `cd backend && uv run pytest -v` passes (Ollama e2e passes or skips).
- `cd backend && uv run python -m app.demo` runs the local Qwen3-8B arm (if Ollama is set up) and prints results for every arm in `arms.yaml`, with API arms failing gracefully when no key is set.
- Adding a new provider requires only an `arms.yaml` edit, no code changes — this is the "model-agnostic, bring your own key" requirement from the spec.
