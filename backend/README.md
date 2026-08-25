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
