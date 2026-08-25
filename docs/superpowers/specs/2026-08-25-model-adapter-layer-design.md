# Model Adapter Layer — Design Spec

Date: 2026-08-25
Status: Approved for implementation
Scope: Build phase 1 of the platform described in `CLAUDE.md` — "unify local
(Ollama) and API models behind one interface; get one local + one API arm
running end to end on a handful of prompts."

## Context

This is the first sub-project of a larger platform (see root `CLAUDE.md`).
The repository is currently empty aside from `CLAUDE.md`. Later phases
(orchestration, judge calibration, stats, dashboard) depend on this layer's
interface being stable, but are explicitly out of scope here.

Two decisions from `CLAUDE.md`'s "Open decisions" are resolved by this spec:

- **Hardware**: smaller GPU, <16GB VRAM. Local model is **Qwen3-8B** via
  Ollama, per the CLAUDE.md default for this hardware tier.
- **API model**: no key is available yet, and the user wants the platform to
  be model-agnostic with bring-your-own-key rather than hardcoded to a
  specific hosted provider. This changes the adapter design from "one
  adapter per named model" to "one adapter per API *schema*" — see below.

## Architecture

Two adapter implementations, not one per provider:

1. **`OpenAICompatibleAdapter`** — any provider speaking the OpenAI
   chat-completions schema: Ollama (local), OpenAI, and (for free, since the
   schema is shared) OpenRouter/Groq/Together/DeepSeek/etc. Configured with
   `base_url`, `model`, and an env-var name holding the API key (empty/absent
   for Ollama, which needs none).
2. **`AnthropicAdapter`** — Claude models, which use a different request/
   response schema. Configured with `model` and an env-var name holding the
   API key.

Both implement a shared `ModelAdapter` protocol:

```python
class ModelAdapter(Protocol):
    def generate(self, prompt: str) -> ModelResponse: ...

@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_estimate_usd: float | None  # None when cost can't be computed (e.g. local)
```

**Arms are config, not code.** Each arm (local or API) is declared in a small
YAML config naming its adapter type, model, and (for API arms) which env var
holds its key:

```yaml
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
  - name: claude-haiku
    adapter: anthropic
    model: claude-haiku-4-5
    api_key_env: ANTHROPIC_API_KEY
```

Swapping or adding a provider is a config edit; no code changes. This is the
mechanism that satisfies "model-agnostic, bring your own key."

## Data flow (phase 1)

A handful of hardcoded prompts run through every configured arm; each arm's
`ModelResponse` is printed side by side (text, latency, tokens, cost where
known). No persistence layer yet — that's phase 2 (Postgres).

## Testing

- **Local (Ollama) arm**: real end-to-end test — requires Ollama running
  locally with `qwen3:8b` pulled. Skipped automatically (not failed) if
  Ollama isn't reachable, so the suite still runs in CI/without local setup.
- **API arms**: unit tests against a mocked HTTP response for both adapter
  types, proving the request/response mapping is correct without needing a
  live API key. Real end-to-end testing happens once the user has a key —
  no code changes required to enable it, just setting the env var named in
  config.

## Repo scaffolding

- Python project using `uv` (matches tooling used in the sibling
  `experimentation_copilot` repo).
- No Postgres, Celery, or frontend yet — those arrive in later build phases
  per `CLAUDE.md`.
- Directories: `backend/app/adapters/` (the two adapter implementations +
  protocol), `backend/app/config/` (arm config loader), `backend/tests/`.

## Out of scope (later phases)

- Orchestration (Celery/Redis), Postgres persistence, judge layer, stats
  layer, dashboard — per `CLAUDE.md` build phases 2–6.
- Multi-user/auth handling of API keys — single-user, env-var-based keys are
  sufficient for this project's stated purpose (portfolio demonstration, not
  a multi-tenant product).
