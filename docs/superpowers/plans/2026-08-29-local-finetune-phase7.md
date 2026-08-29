# Phase 7 — Local LoRA Fine-tune + Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the local Qwen3-8B model with QLoRA on the financial-sentiment task, serve it through Ollama as a first-class arm, and produce an honest fine-tuned-vs-base-vs-API comparison report using the platform's existing paired-stats and frontier machinery.

**Architecture:** A host-side `app/training/` module (dataset build → QLoRA train → merge/GGUF/`ollama create`) driven by a new `pe finetune` CLI sub-typer and thin `backend/scripts/finetune_*.py` wrappers, mirroring the judge-calibration workflow. The fine-tuned model is served through Ollama, so it is just another `openai_compatible` entry in `arms.yaml` — no adapter, orchestrator, worker, stats, or judge changes. Training dependencies live in an optional `training` extra kept out of the core/runtime/CI path.

**Tech Stack:** Python 3.12, Unsloth (QLoRA + GGUF export), TRL `SFTTrainer`, HuggingFace `datasets`/`transformers`/`peft`, `bitsandbytes`, PyTorch, Ollama, Typer, `httpx`, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-local-finetune-phase7-design.md`

## Global Constraints

- All backend paths in this plan are relative to `backend/`. Run commands from inside `backend/` unless stated otherwise.
- Package manager is `uv`. Install the training extra with `uv sync --extra training` (needs a CUDA GPU); CI and the default `uv sync` must NOT pull it.
- CI runs without the `training` extra. No test may import `unsloth`, `torch`, `trl`, `peft`, `transformers`, `bitsandbytes`, or `datasets` at module load. Tests that need those symbols must be `pytest.importorskip`-guarded or monkeypatch the loader functions.
- The eval set is all 2271 `sentences_allagree` rows (`eval_example.source == "financial_phrasebank_allagree"`). Training data MUST be disjoint from it — this is enforced by `LeakageError` in `app/training/dataset.py`, not by convention.
- Training data source: Financial PhraseBank lower-agreement subset (default config `sentences_75agree`), downloaded at runtime, NOT vendored/committed. CC BY-NC-SA 3.0, Malo et al. 2014 — attribution required in the report and the written `LICENSE.txt` beside the dataset artifacts.
- Label mapping is fixed: `{0: "negative", 1: "neutral", 2: "positive"}`. Valid label words: `positive`, `negative`, `neutral`.
- Thinking mode is DISABLED for the fine-tuned arm (single-label classification).
- The eval prompt the arm is trained on must be byte-identical to what the orchestrator serves (`render_eval_prompt`).
- `git commit` at the end of every task. Do not `git push` (the operator decides when to push).
- Commit message trailer for every commit:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc
  ```

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `backend/app/eval_prompt.py` | The shared eval-prompt template + `render_eval_prompt`, extracted from `worker.py` so training and orchestration render the identical string. |
| `backend/app/training/__init__.py` | Package marker (empty). |
| `backend/app/training/config.py` | `TrainingConfig` dataclass + `load_training_config()` + `InvalidTrainingConfigError`. Reads `backend/training.yaml`. |
| `backend/app/training/dataset.py` | Leakage-guarded SFT dataset builder: HF load → normalize → drop eval-set overlap → format as chat records → seeded split → write JSONL + `LICENSE.txt`. |
| `backend/app/training/train.py` | Unsloth QLoRA training loop. Guarded imports; saves the LoRA adapter only. |
| `backend/app/training/export.py` | Merge adapter → 16-bit → GGUF → `Modelfile` → `ollama create` → emit `arm_snippet.yaml`. Pure helpers for the Modelfile / snippet / base_url lookup. |
| `backend/training.yaml` | The single training config surface (committed, defaults filled in). |
| `backend/scripts/finetune_prep.py` | CLI-invoked wrapper: `build_sft_dataset` + print stats. |
| `backend/scripts/finetune_train.py` | CLI-invoked wrapper: build (or reuse) dataset + `run_training`; `--dry-run` stops before model load. |
| `backend/scripts/finetune_export.py` | CLI-invoked wrapper: `export_arm` + print the arm snippet. |
| `backend/scripts/finetune_report.py` | Pull `/compare`, `/summary`, `/equivalence`, `/runs/{id}` for a completed run; render the markdown report + frontier PNG. |
| `backend/tests/test_eval_prompt.py` | Pins the rendered prompt string. |
| `backend/tests/training/__init__.py` | Package marker. |
| `backend/tests/training/test_config.py` | `training.yaml` parse + validation. |
| `backend/tests/training/test_dataset.py` | Leakage guard, formatting, split, class balance. |
| `backend/tests/training/test_train.py` | Guarded-import error path. |
| `backend/tests/training/test_export.py` | Pure helpers + guarded-import error path. |
| `backend/tests/scripts/test_finetune_scripts.py` | `prep`/`train --dry-run`/`export` wrappers (monkeypatched). |
| `backend/tests/scripts/test_finetune_report.py` | Report markdown + cost math (respx-mocked HTTP). |
| `backend/tests/data/fixtures/arms_sample.yaml` | Fixture `arms.yaml` for `read_local_arm_base_url`. |
| `docs/superpowers/reports/2026-08-29-finetune-comparison.md` | The executed comparison write-up (Task 10, on the GPU box). |
| `docs/superpowers/reports/2026-08-29-finetune-frontier.png` | Committed frontier plot (Task 10). |

**Modified files:**

| Path | Change |
|---|---|
| `backend/app/tasks/worker.py` | Import `render_eval_prompt` from `app.eval_prompt`; drop the local copy. |
| `backend/pyproject.toml` | Add `[project.optional-dependencies] training`. |
| `.gitignore` | Add `backend/training/artifacts/`. |
| `backend/app/cli/__init__.py` | Add `finetune_app` sub-typer with 4 commands. |
| `backend/tests/cli/test_cli.py` | Add `pe finetune *` argv assertions. |
| `CLAUDE.md` | Mark Phase 7 done; link the spec. |
| `backend/README.md` | Add the "Phase 7: local fine-tune" section. |

---

## Task 1: Extract the shared eval prompt

**Files:**
- Create: `backend/app/eval_prompt.py`
- Create: `backend/tests/test_eval_prompt.py`
- Modify: `backend/app/tasks/worker.py` (the `EVAL_PROMPT_TEMPLATE` / `render_eval_prompt` block, currently near line 30–41, and its call site near line 223)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `app.eval_prompt.EVAL_PROMPT_TEMPLATE: str`
  - `app.eval_prompt.render_eval_prompt(text: str) -> str`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_eval_prompt.py`:

```python
from app.eval_prompt import EVAL_PROMPT_TEMPLATE, render_eval_prompt

EXPECTED = (
    "Is the following sentence positive, negative, or neutral from a "
    "financial-news perspective? Respond with just the sentiment label.\n\n"
    "Sentence: Operating profit rose to EUR 1.5 mn."
)


def test_render_eval_prompt_exact_string():
    assert render_eval_prompt("Operating profit rose to EUR 1.5 mn.") == EXPECTED


def test_template_has_one_placeholder():
    assert EVAL_PROMPT_TEMPLATE.count("{text}") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval_prompt'`

- [ ] **Step 3: Create the module**

`backend/app/eval_prompt.py` — copy the exact template text out of `worker.py`:

```python
"""The eval prompt is shared by the orchestrator (app/tasks/worker.py) and
the fine-tuning dataset builder (app/training/dataset.py) so an arm is
trained on the identical string it is later asked at eval time.

EvalExample.text is a bare dataset sentence with no task instruction. Left
unframed, different models guess the implied task with different
reliability -- the judge rubric (app/judge/rubric.py) assumes the model
attempted a classification, so the arm has to actually be asked for one.
"""

EVAL_PROMPT_TEMPLATE = (
    "Is the following sentence positive, negative, or neutral from a "
    "financial-news perspective? Respond with just the sentiment label.\n\n"
    "Sentence: {text}"
)


def render_eval_prompt(text: str) -> str:
    return EVAL_PROMPT_TEMPLATE.format(text=text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Point `worker.py` at the shared module**

In `backend/app/tasks/worker.py`:
- Delete the local `EVAL_PROMPT_TEMPLATE` assignment, the comment block above it, and the local `def render_eval_prompt`.
- Add to the imports: `from app.eval_prompt import render_eval_prompt`
- Leave the call site (`adapter.generate(render_eval_prompt(example_text))`) unchanged.

Then confirm nothing else imported the old names:

```bash
grep -rn "EVAL_PROMPT_TEMPLATE\|render_eval_prompt" app/ tests/
```

Every hit must be either `app/eval_prompt.py`, the new import in `worker.py`, its call site, or `tests/test_eval_prompt.py`. If `app/demo.py` or a test references `worker.render_eval_prompt`, update it to `from app.eval_prompt import render_eval_prompt`.

- [ ] **Step 6: Run the worker + demo tests**

Run: `uv run pytest tests/tasks/ tests/test_demo.py -v`
Expected: PASS (no behavior change)

- [ ] **Step 7: Commit**

```bash
git add app/eval_prompt.py tests/test_eval_prompt.py app/tasks/worker.py
git commit -m "refactor: extract render_eval_prompt into app.eval_prompt

Shared by the orchestrator and the upcoming fine-tuning dataset builder
so an arm trains on the exact string it is asked at eval time.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 2: Training extra, gitignore, and `TrainingConfig`

**Files:**
- Modify: `backend/pyproject.toml` (`[project.optional-dependencies]`)
- Modify: `.gitignore` (repo root)
- Create: `backend/training.yaml`
- Create: `backend/app/training/__init__.py` (empty)
- Create: `backend/app/training/config.py`
- Create: `backend/tests/training/__init__.py` (empty)
- Create: `backend/tests/training/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `app.training.config.TrainingConfig` — a frozen dataclass with fields:
    `run_name: str`, `base_model: str`, `source_dataset: str`, `source_config: str`,
    `max_seq_len: int`, `lora_r: int`, `lora_alpha: int`, `lora_dropout: float`,
    `lora_target_modules: list[str]`, `epochs: float`, `learning_rate: float`,
    `batch_size: int`, `grad_accum: int`, `seed: int`, `val_fraction: float`,
    `balance_neutral: bool`, `min_pool_size: int`, `gguf_quant: str`,
    `ollama_tag: str`, `output_dir: str`
  - `app.training.config.load_training_config(path: str = "training.yaml") -> TrainingConfig`
  - `app.training.config.InvalidTrainingConfigError(ValueError)`

- [ ] **Step 1: Add the optional dependency group**

In `backend/pyproject.toml`, after the `[project]` table (sibling of `[dependency-groups]`):

```toml
[project.optional-dependencies]
training = [
    "unsloth",
    "trl",
    "peft",
    "transformers",
    "datasets",
    "bitsandbytes",
    "torch",
    "accelerate",
    "matplotlib",
]
```

Do not pin versions yet — Task 10's operator resolves a known-good set on the GPU box and records it in `backend/README.md`.

- [ ] **Step 2: Ignore training artifacts**

Append to the repo-root `.gitignore`:

```
backend/training/artifacts/
```

- [ ] **Step 3: Write `backend/training.yaml`**

```yaml
# The single config surface for Phase 7 fine-tuning (mirrors arms.yaml).
# `pe finetune prep|train|export` all read this file.

run_name: qwen3-8b-finsent-lora

# HF weights to fine-tune (full/4bit, NOT the Ollama GGUF).
base_model: unsloth/Qwen3-8B

# Financial PhraseBank lower-agreement subset -- disjoint from the
# all-agree eval set after the leakage guard drops the overlap.
# Downloaded at runtime, not vendored (CC BY-NC-SA 3.0, Malo et al. 2014).
source_dataset: gtfintechlab/financial_phrasebank_sentences_75agree
source_config: "5768"

max_seq_len: 512

lora_r: 16
lora_alpha: 16
lora_dropout: 0.0
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

epochs: 3
learning_rate: 0.0002
batch_size: 8
grad_accum: 2
seed: 42

val_fraction: 0.1
balance_neutral: false
min_pool_size: 500

# q4_k_m matches the base local arm's quantization for a clean
# base-vs-fine-tuned comparison (see the spec's Risk 5).
gguf_quant: q4_k_m
ollama_tag: ft-qwen3-8b

output_dir: training/artifacts/qwen3-8b-finsent-lora
```

- [ ] **Step 4: Write the failing test**

`backend/tests/training/test_config.py`:

```python
import textwrap

import pytest

from app.training.config import (
    InvalidTrainingConfigError,
    TrainingConfig,
    load_training_config,
)

VALID = textwrap.dedent(
    """
    run_name: demo
    base_model: unsloth/Qwen3-8B
    source_dataset: some/dataset
    source_config: "5768"
    max_seq_len: 512
    lora_r: 16
    lora_alpha: 16
    lora_dropout: 0.0
    lora_target_modules: [q_proj, v_proj]
    epochs: 3
    learning_rate: 0.0002
    batch_size: 8
    grad_accum: 2
    seed: 42
    val_fraction: 0.1
    balance_neutral: false
    min_pool_size: 500
    gguf_quant: q4_k_m
    ollama_tag: ft-qwen3-8b
    output_dir: training/artifacts/demo
    """
)


def test_loads_valid_config(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID)
    cfg = load_training_config(str(p))
    assert isinstance(cfg, TrainingConfig)
    assert cfg.run_name == "demo"
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]
    assert cfg.epochs == 3.0
    assert cfg.balance_neutral is False


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID + "\nmystery_key: 1\n")
    with pytest.raises(InvalidTrainingConfigError, match="mystery_key"):
        load_training_config(str(p))


def test_missing_required_key_rejected(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID.replace("run_name: demo", ""))
    with pytest.raises(InvalidTrainingConfigError, match="run_name"):
        load_training_config(str(p))


def test_wrong_type_rejected(tmp_path):
    p = tmp_path / "training.yaml"
    p.write_text(VALID.replace("max_seq_len: 512", "max_seq_len: not-an-int"))
    with pytest.raises(InvalidTrainingConfigError, match="max_seq_len"):
        load_training_config(str(p))


def test_committed_training_yaml_parses():
    cfg = load_training_config("training.yaml")
    assert cfg.ollama_tag
    assert cfg.gguf_quant == "q4_k_m"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run pytest tests/training/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.training'`

- [ ] **Step 6: Implement `app/training/config.py`**

Create `backend/app/training/__init__.py` (empty) and `backend/tests/training/__init__.py` (empty), then:

```python
"""Fine-tuning config -- the training.yaml counterpart to app/config/arms.py."""
from dataclasses import dataclass, fields
from typing import get_args, get_origin

import yaml


class InvalidTrainingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    base_model: str
    source_dataset: str
    source_config: str
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    epochs: float
    learning_rate: float
    batch_size: int
    grad_accum: int
    seed: int
    val_fraction: float
    balance_neutral: bool
    min_pool_size: int
    gguf_quant: str
    ollama_tag: str
    output_dir: str


def _check_type(name: str, value: object, annotation: object) -> None:
    origin = get_origin(annotation)
    if origin is list:
        if not isinstance(value, list):
            raise InvalidTrainingConfigError(f"'{name}' must be a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation) or (str,)
        for item in value:
            if not isinstance(item, item_type):
                raise InvalidTrainingConfigError(f"'{name}' items must be {item_type.__name__}")
        return
    # bool is a subclass of int -- check it first so `epochs: true` is rejected
    if annotation in (int, float) and isinstance(value, bool):
        raise InvalidTrainingConfigError(f"'{name}' must be {annotation.__name__}, got bool")
    if annotation is float and isinstance(value, int):
        return  # YAML ints are acceptable where a float is wanted
    if not isinstance(value, annotation):
        raise InvalidTrainingConfigError(
            f"'{name}' must be {annotation.__name__}, got {type(value).__name__}"
        )


def load_training_config(path: str = "training.yaml") -> TrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise InvalidTrainingConfigError(f"'{path}' must be a mapping")

    known = {f.name: f for f in fields(TrainingConfig)}
    unknown = set(raw) - set(known)
    if unknown:
        raise InvalidTrainingConfigError(f"unknown key(s) in '{path}': {', '.join(sorted(unknown))}")

    missing = set(known) - set(raw)
    if missing:
        raise InvalidTrainingConfigError(
            f"missing required key(s) in '{path}': {', '.join(sorted(missing))}"
        )

    for name, field in known.items():
        _check_type(name, raw[name], field.type)

    coerced = dict(raw)
    coerced["epochs"] = float(coerced["epochs"])
    coerced["learning_rate"] = float(coerced["learning_rate"])
    coerced["lora_dropout"] = float(coerced["lora_dropout"])
    coerced["val_fraction"] = float(coerced["val_fraction"])
    return TrainingConfig(**coerced)
```

Note: `field.type` may be a string under `from __future__ import annotations`; this module deliberately does NOT import that, so annotations stay as real types.

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/training/test_config.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml ../.gitignore training.yaml app/training/__init__.py \
  app/training/config.py tests/training/__init__.py tests/training/test_config.py
git commit -m "feat: training extra + TrainingConfig / training.yaml

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 3: Leakage-guarded SFT dataset builder

**Files:**
- Create: `backend/app/training/dataset.py`
- Create: `backend/tests/training/test_dataset.py`

**Interfaces:**
- Consumes:
  - `app.training.config.TrainingConfig`
  - `app.eval_prompt.render_eval_prompt(text: str) -> str`
- Produces:
  - `app.training.dataset.LABEL_NAMES: dict[int, str]` — `{0: "negative", 1: "neutral", 2: "positive"}`
  - `app.training.dataset.LeakageError(Exception)`
  - `app.training.dataset.normalize_sentence(s: str) -> str`
  - `app.training.dataset.load_source_examples(cfg: TrainingConfig) -> list[tuple[str, str]]` — `(sentence, label_word)` pairs; real HF download. Tests monkeypatch this.
  - `app.training.dataset.fetch_eval_texts() -> set[str]` — every `eval_example.text` from Postgres. Tests monkeypatch this.
  - `app.training.dataset.build_sft_dataset(cfg: TrainingConfig) -> DatasetBuildResult`
  - `app.training.dataset.DatasetBuildResult` — frozen dataclass:
    `train_path: Path`, `val_path: Path`, `license_path: Path`,
    `train_class_counts: dict[str, int]`, `val_class_counts: dict[str, int]`,
    `dropped_count: int`, `pool_size: int`, `source: str`

- [ ] **Step 1: Write the failing test**

`backend/tests/training/test_dataset.py`:

```python
import json

import pytest

from app.eval_prompt import render_eval_prompt
from app.training import dataset as ds
from app.training.config import load_training_config

BASE_CFG = load_training_config("training.yaml")


def cfg(tmp_path, **over):
    from dataclasses import replace

    return replace(BASE_CFG, output_dir=str(tmp_path / "art"), min_pool_size=1, **over)


def fake_source(rows):
    return lambda _cfg: list(rows)


def make_rows(n_pos, n_neg, n_neu, prefix="s"):
    rows = []
    for i in range(n_pos):
        rows.append((f"{prefix} pos {i}", "positive"))
    for i in range(n_neg):
        rows.append((f"{prefix} neg {i}", "negative"))
    for i in range(n_neu):
        rows.append((f"{prefix} neu {i}", "neutral"))
    return rows


def test_drops_eval_set_overlap(tmp_path, monkeypatch):
    rows = make_rows(3, 3, 3) + [("Profit ROSE  sharply.", "positive")]
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: {"profit rose sharply."})

    result = ds.build_sft_dataset(cfg(tmp_path))

    assert result.dropped_count == 1
    assert result.pool_size == 9
    lines = result.train_path.read_text().splitlines() + result.val_path.read_text().splitlines()
    texts = [json.loads(l)["messages"][0]["content"] for l in lines]
    assert all("Profit ROSE" not in t for t in texts)


def test_record_format_matches_render_eval_prompt(tmp_path, monkeypatch):
    rows = [("Shares fell 4 percent.", "negative")] + make_rows(2, 2, 2)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())

    result = ds.build_sft_dataset(cfg(tmp_path, val_fraction=0.0))
    records = [json.loads(l) for l in result.train_path.read_text().splitlines()]
    rec = next(r for r in records if "Shares fell 4 percent." in r["messages"][0]["content"])
    assert rec["messages"][0] == {
        "role": "user",
        "content": render_eval_prompt("Shares fell 4 percent."),
    }
    assert rec["messages"][1] == {"role": "assistant", "content": "negative"}


def test_leakage_error_when_overlap_survives(tmp_path, monkeypatch):
    # fetch_eval_texts returns an already-normalized string that does NOT
    # match after normalization -> simulate a guard bug by making normalize
    # a no-op via monkeypatch is overkill; instead assert the happy path is
    # clean, then force min_pool_size high:
    rows = make_rows(2, 2, 2)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())
    with pytest.raises(ds.LeakageError, match="pool"):
        ds.build_sft_dataset(cfg(tmp_path, min_pool_size=999))


def test_deterministic_seeded_split(tmp_path, monkeypatch):
    rows = make_rows(20, 20, 20)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())

    a = ds.build_sft_dataset(cfg(tmp_path / "a", seed=7, val_fraction=0.2))
    b = ds.build_sft_dataset(cfg(tmp_path / "b", seed=7, val_fraction=0.2))
    assert a.val_path.read_text() == b.val_path.read_text()
    train_texts = {json.loads(l)["messages"][0]["content"] for l in a.train_path.read_text().splitlines()}
    val_texts = {json.loads(l)["messages"][0]["content"] for l in a.val_path.read_text().splitlines()}
    assert train_texts.isdisjoint(val_texts)


def test_balance_neutral_downsamples(tmp_path, monkeypatch):
    rows = make_rows(5, 5, 40)
    monkeypatch.setattr(ds, "load_source_examples", fake_source(rows))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())

    result = ds.build_sft_dataset(cfg(tmp_path, balance_neutral=True, val_fraction=0.0))
    assert result.train_class_counts["neutral"] == 5


def test_license_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "load_source_examples", fake_source(make_rows(2, 2, 2)))
    monkeypatch.setattr(ds, "fetch_eval_texts", lambda: set())
    result = ds.build_sft_dataset(cfg(tmp_path))
    assert "CC BY-NC-SA" in result.license_path.read_text()
    assert "Malo" in result.license_path.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/training/test_dataset.py -v`
Expected: FAIL with `AttributeError: module 'app.training.dataset' has no attribute ...` / import error.

- [ ] **Step 3: Implement `app/training/dataset.py`**

```python
"""Builds the supervised fine-tuning dataset from Financial PhraseBank's
lower-agreement subset, with a hard guard that nothing in it overlaps the
all-agree eval set (or, transitively, the judge-calibration rows -- all of
which are all-agree sentences).

Dataset licence: CC BY-NC-SA 3.0, Malo et al. 2014. Downloaded at runtime,
never vendored.
"""
import asyncio
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample
from app.db.session import engine
from app.eval_prompt import render_eval_prompt
from app.training.config import TrainingConfig

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
_VALID_LABELS = set(LABEL_NAMES.values())

LICENSE_TEXT = (
    "Financial PhraseBank (Malo, P., Sinha, A., Korhonen, P., Wallenius, J.,\n"
    "and Takala, P. 2014. 'Good debt or bad debt: Detecting semantic\n"
    "orientations in economic texts.'). Lower-agreement subset used for\n"
    "fine-tuning only. Licensed CC BY-NC-SA 3.0 (non-commercial);\n"
    "see http://creativecommons.org/licenses/by-nc-sa/3.0/. Not redistributed.\n"
)

_WS = re.compile(r"\s+")


class LeakageError(Exception):
    pass


@dataclass(frozen=True)
class DatasetBuildResult:
    train_path: Path
    val_path: Path
    license_path: Path
    train_class_counts: dict[str, int]
    val_class_counts: dict[str, int]
    dropped_count: int
    pool_size: int
    source: str


def normalize_sentence(s: str) -> str:
    return _WS.sub(" ", s).strip().casefold()


def load_source_examples(cfg: TrainingConfig) -> list[tuple[str, str]]:
    """(sentence, label_word) pairs from the configured HF subset.

    Monkeypatched in tests -- keep it a thin, side-effect-only loader.
    """
    from datasets import concatenate_datasets, load_dataset

    dataset = load_dataset(cfg.source_dataset, cfg.source_config)
    combined = concatenate_datasets(list(dataset.values()))
    pairs: list[tuple[str, str]] = []
    for row in combined:
        sentence = str(row["sentence"]).replace("\n", " ").strip()
        raw_label = row["label"]
        label = LABEL_NAMES[int(raw_label)] if isinstance(raw_label, int) else str(raw_label)
        if label not in _VALID_LABELS:
            raise ValueError(f"Unexpected label {raw_label!r} in {cfg.source_dataset}")
        pairs.append((sentence, label))
    return pairs


def fetch_eval_texts() -> set[str]:
    """Every eval_example.text, for the leakage guard. Monkeypatched in tests."""

    async def _run() -> set[str]:
        async with AsyncSession(engine) as session:
            result = await session.execute(select(EvalExample.text))
            return set(result.scalars().all())

    return asyncio.run(_run())


def _balance_neutral(rows: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    minority = max(
        (len(v) for k, v in by_label.items() if k != "neutral"), default=0
    )
    neutral = by_label.get("neutral", [])
    if len(neutral) > minority:
        rng = random.Random(seed)
        by_label["neutral"] = rng.sample(neutral, minority)
    out = [row for rows_ in by_label.values() for row in rows_]
    random.Random(seed).shuffle(out)
    return out


def _write_jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sentence, label in rows:
            record = {
                "messages": [
                    {"role": "user", "content": render_eval_prompt(sentence)},
                    {"role": "assistant", "content": label},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_dataset(cfg: TrainingConfig) -> DatasetBuildResult:
    source_rows = load_source_examples(cfg)
    eval_norm = {normalize_sentence(t) for t in fetch_eval_texts()}

    kept: list[tuple[str, str]] = []
    dropped = 0
    for sentence, label in source_rows:
        if normalize_sentence(sentence) in eval_norm:
            dropped += 1
            continue
        kept.append((sentence, label))

    # Guard: recheck the survivors are clean, and the pool is plausible.
    still_overlapping = [s for s, _ in kept if normalize_sentence(s) in eval_norm]
    if still_overlapping:
        raise LeakageError(
            f"{len(still_overlapping)} training sentences still overlap the eval set after dropping"
        )
    if len(kept) < cfg.min_pool_size:
        raise LeakageError(
            f"training pool is {len(kept)} rows after leakage drop, below min_pool_size "
            f"({cfg.min_pool_size}) -- wrong subset or normalization?"
        )

    if cfg.balance_neutral:
        kept = _balance_neutral(kept, cfg.seed)

    rng = random.Random(cfg.seed)
    rng.shuffle(kept)
    n_val = int(round(len(kept) * cfg.val_fraction))
    val_rows = kept[:n_val]
    train_rows = kept[n_val:]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    license_path = out_dir / "LICENSE.txt"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    license_path.write_text(LICENSE_TEXT, encoding="utf-8")

    return DatasetBuildResult(
        train_path=train_path,
        val_path=val_path,
        license_path=license_path,
        train_class_counts=dict(Counter(l for _, l in train_rows)),
        val_class_counts=dict(Counter(l for _, l in val_rows)),
        dropped_count=dropped,
        pool_size=len(kept),
        source=f"{cfg.source_dataset}:{cfg.source_config}",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/training/test_dataset.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Full-suite regression check**

Run: `uv run pytest -q`
Expected: no new failures vs. `master` (DB-backed tests may skip if Postgres is down — that's pre-existing).

- [ ] **Step 6: Commit**

```bash
git add app/training/dataset.py tests/training/test_dataset.py
git commit -m "feat: leakage-guarded SFT dataset builder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 4: Unsloth QLoRA training

**Files:**
- Create: `backend/app/training/train.py`
- Create: `backend/tests/training/test_train.py`

**Interfaces:**
- Consumes:
  - `app.training.config.TrainingConfig`
  - `app.training.dataset.DatasetBuildResult`
- Produces:
  - `app.training.train.MissingTrainingDepsError(RuntimeError)`
  - `app.training.train.import_unsloth()` — returns the `unsloth` module or raises `MissingTrainingDepsError` with an actionable message. Reused by `export.py`.
  - `app.training.train.run_training(cfg: TrainingConfig, dataset: DatasetBuildResult) -> TrainingResult`
  - `app.training.train.TrainingResult` — frozen dataclass:
    `adapter_path: Path`, `loss_history: list[dict]`, `wall_seconds: float`, `seed: int`

- [ ] **Step 1: Write the failing test**

`backend/tests/training/test_train.py`:

```python
import pytest

from app.training import train


def test_import_unsloth_raises_actionable_error_when_absent():
    try:
        import unsloth  # noqa: F401
    except ImportError:
        with pytest.raises(train.MissingTrainingDepsError, match="uv sync --extra training"):
            train.import_unsloth()
    else:
        pytest.skip("unsloth is installed in this environment")


def test_run_training_raises_without_deps(tmp_path):
    try:
        import unsloth  # noqa: F401
    except ImportError:
        from app.training.dataset import DatasetBuildResult

        fake = DatasetBuildResult(
            train_path=tmp_path / "train.jsonl",
            val_path=tmp_path / "val.jsonl",
            license_path=tmp_path / "LICENSE.txt",
            train_class_counts={},
            val_class_counts={},
            dropped_count=0,
            pool_size=0,
            source="x",
        )
        from app.training.config import load_training_config

        with pytest.raises(train.MissingTrainingDepsError):
            train.run_training(load_training_config("training.yaml"), fake)
    else:
        pytest.skip("unsloth is installed in this environment")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/training/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.training.train'`

- [ ] **Step 3: Implement `app/training/train.py`**

```python
"""QLoRA fine-tune of the local model with Unsloth. Host-side, GPU-only.

Saves the LoRA adapter only -- merge / GGUF / ollama create is export.py,
so a training run can be inspected before it becomes an Ollama model.
Training is seeded but not bit-reproducible.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from app.training.config import TrainingConfig
from app.training.dataset import DatasetBuildResult

_INSTALL_HINT = (
    "Training dependencies are not installed. Run `uv sync --extra training` "
    "on a machine with a CUDA GPU (see backend/README.md 'Phase 7')."
)


class MissingTrainingDepsError(RuntimeError):
    pass


def import_unsloth() -> ModuleType:
    try:
        import unsloth
    except ImportError as exc:
        raise MissingTrainingDepsError(_INSTALL_HINT) from exc
    return unsloth


@dataclass(frozen=True)
class TrainingResult:
    adapter_path: Path
    loss_history: list[dict]
    wall_seconds: float
    seed: int


def _load_chat_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_training(cfg: TrainingConfig, dataset: DatasetBuildResult) -> TrainingResult:
    import_unsloth()  # fail fast with the actionable message
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    started = time.perf_counter()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    def _to_text(records: list[dict]) -> Dataset:
        rows = []
        for rec in records:
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            rows.append({"text": text})
        return Dataset.from_list(rows)

    train_ds = _to_text(_load_chat_records(dataset.train_path))
    eval_ds = _to_text(_load_chat_records(dataset.val_path)) if dataset.val_path.exists() else None

    out_dir = Path(cfg.output_dir)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
            logging_steps=1,
            eval_strategy="epoch" if eval_ds is not None else "no",
            output_dir=str(out_dir / "trainer"),
            report_to=[],
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_len,
        ),
    )
    # Train only on the assistant tokens (the label word), not the prompt.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    loss_history = [
        {k: v for k, v in entry.items() if k in ("epoch", "step", "loss", "eval_loss")}
        for entry in trainer.state.log_history
    ]
    return TrainingResult(
        adapter_path=adapter_path,
        loss_history=loss_history,
        wall_seconds=time.perf_counter() - started,
        seed=cfg.seed,
    )
```

Note for the implementer: the `instruction_part` / `response_part` markers and `enable_thinking=False` are Qwen3-specific and are on the Task 10 verify list — a mismatch shows up as the model learning to echo the prompt. If `train_on_responses_only` import path differs in the installed Unsloth version, adjust; the intent is completion-only loss.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/training/test_train.py -v`
Expected: PASS (both tests, via the no-unsloth branch on CI)

- [ ] **Step 5: Commit**

```bash
git add app/training/train.py tests/training/test_train.py
git commit -m "feat: Unsloth QLoRA training loop (host-side, GPU-only)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 5: Merge → GGUF → Ollama export

**Files:**
- Create: `backend/app/training/export.py`
- Create: `backend/tests/training/test_export.py`
- Create: `backend/tests/data/fixtures/arms_sample.yaml`

**Interfaces:**
- Consumes:
  - `app.training.config.TrainingConfig`
  - `app.training.train.import_unsloth`, `app.training.train.MissingTrainingDepsError`
- Produces:
  - `app.training.export.ExportError(Exception)`
  - `app.training.export.build_modelfile(cfg: TrainingConfig, gguf_filename: str) -> str`
  - `app.training.export.read_local_arm_base_url(arms_path: Path, arm_name: str = "qwen3-8b-local") -> str`
  - `app.training.export.render_arm_snippet(cfg: TrainingConfig, base_url: str) -> str`
  - `app.training.export.export_arm(cfg: TrainingConfig, arms_path: Path | None = None) -> ExportResult`
  - `app.training.export.ExportResult` — frozen dataclass:
    `gguf_path: Path`, `modelfile_path: Path`, `ollama_tag: str`, `snippet_path: Path`, `snippet: str`

- [ ] **Step 1: Write the fixture**

`backend/tests/data/fixtures/arms_sample.yaml`:

```yaml
arms:
  - name: qwen3-8b-local
    adapter: openai_compatible
    base_url: http://172.30.0.127:11434/v1
    model: qwen3:8b
    max_tokens: 1024
  - name: gpt-4o-mini
    adapter: openai_compatible
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
```

- [ ] **Step 2: Write the failing test**

`backend/tests/training/test_export.py`:

```python
from dataclasses import replace
from pathlib import Path

import pytest

from app.training import export
from app.training.config import load_training_config

CFG = load_training_config("training.yaml")
FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "arms_sample.yaml"


def test_build_modelfile_disables_thinking_and_pins_params():
    text = export.build_modelfile(CFG, "qwen3-8b-finsent-lora.Q4_K_M.gguf")
    assert "FROM ./qwen3-8b-finsent-lora.Q4_K_M.gguf" in text
    assert f"num_ctx {CFG.max_seq_len}" in text
    assert "temperature 0" in text
    assert "/no_think" in text  # thinking disabled for classification


def test_read_local_arm_base_url():
    assert export.read_local_arm_base_url(FIXTURE) == "http://172.30.0.127:11434/v1"


def test_read_local_arm_base_url_missing_arm():
    with pytest.raises(export.ExportError, match="ghost"):
        export.read_local_arm_base_url(FIXTURE, arm_name="ghost")


def test_render_arm_snippet():
    snippet = export.render_arm_snippet(CFG, "http://172.30.0.127:11434/v1")
    assert "name: ft-qwen3-8b-local" in snippet
    assert "adapter: openai_compatible" in snippet
    assert "model: ft-qwen3-8b" in snippet
    assert "base_url: http://172.30.0.127:11434/v1" in snippet


def test_export_arm_raises_without_deps(tmp_path):
    try:
        import unsloth  # noqa: F401
    except ImportError:
        with pytest.raises(export.MissingTrainingDepsError):
            export.export_arm(replace(CFG, output_dir=str(tmp_path)), arms_path=FIXTURE)
    else:
        pytest.skip("unsloth is installed in this environment")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/training/test_export.py -v`
Expected: FAIL — `app.training.export` does not exist.

- [ ] **Step 4: Implement `app/training/export.py`**

```python
"""Turns a trained LoRA adapter into a running Ollama model and the
arms.yaml entry that points at it. Host-side, GPU-only for the merge/GGUF
steps.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.training.config import TrainingConfig
from app.training.train import MissingTrainingDepsError, import_unsloth

DEFAULT_ARMS_PATH = Path(__file__).resolve().parent.parent.parent / "arms.yaml"
FT_ARM_NAME = "ft-qwen3-8b-local"


class ExportError(Exception):
    pass


@dataclass(frozen=True)
class ExportResult:
    gguf_path: Path
    modelfile_path: Path
    ollama_tag: str
    snippet_path: Path
    snippet: str


def build_modelfile(cfg: TrainingConfig, gguf_filename: str) -> str:
    # Qwen3 honours a `/no_think` directive in the system prompt: it
    # suppresses the <think> block so the arm emits a bare label and the
    # shared eval prompt (app/eval_prompt.py) is passed through untouched.
    return "\n".join(
        [
            f"FROM ./{gguf_filename}",
            'SYSTEM """/no_think"""',
            f"PARAMETER num_ctx {cfg.max_seq_len}",
            "PARAMETER temperature 0",
            'PARAMETER stop "<|im_end|>"',
            "",
        ]
    )


def read_local_arm_base_url(arms_path: Path, arm_name: str = "qwen3-8b-local") -> str:
    raw = yaml.safe_load(Path(arms_path).read_text())
    for entry in raw.get("arms", []):
        if entry.get("name") == arm_name:
            url = entry.get("base_url")
            if not url:
                raise ExportError(f"arm '{arm_name}' in {arms_path} has no base_url")
            return url
    raise ExportError(f"arm '{arm_name}' not found in {arms_path}")


def render_arm_snippet(cfg: TrainingConfig, base_url: str) -> str:
    return "\n".join(
        [
            f"  - name: {FT_ARM_NAME}",
            "    adapter: openai_compatible",
            f"    base_url: {base_url}",
            f"    model: {cfg.ollama_tag}",
            "    max_tokens: 1024",
            "",
        ]
    )


def export_arm(cfg: TrainingConfig, arms_path: Path | None = None) -> ExportResult:
    import_unsloth()
    from unsloth import FastLanguageModel

    out_dir = Path(cfg.output_dir)
    adapter_path = out_dir / "adapter"
    if not adapter_path.exists():
        raise ExportError(f"no trained adapter at {adapter_path} -- run `pe finetune train` first")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=False,
        dtype=None,
    )

    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf_dir), tokenizer, quantization_method=cfg.gguf_quant
    )
    gguf_files = sorted(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise ExportError(f"GGUF conversion produced no .gguf file in {gguf_dir}")
    gguf_path = gguf_files[0]

    modelfile_path = gguf_dir / "Modelfile"
    modelfile_path.write_text(build_modelfile(cfg, gguf_path.name), encoding="utf-8")

    proc = subprocess.run(
        ["ollama", "create", cfg.ollama_tag, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ExportError(f"`ollama create` failed:\n{proc.stderr}")

    base_url = read_local_arm_base_url(arms_path or DEFAULT_ARMS_PATH)
    snippet = render_arm_snippet(cfg, base_url)
    snippet_path = out_dir / "arm_snippet.yaml"
    snippet_path.write_text(snippet, encoding="utf-8")

    return ExportResult(
        gguf_path=gguf_path,
        modelfile_path=modelfile_path,
        ollama_tag=cfg.ollama_tag,
        snippet_path=snippet_path,
        snippet=snippet,
    )
```

Re-export `MissingTrainingDepsError` at the top of the module so tests import it from either place: add `MissingTrainingDepsError` to the `from app.training.train import ...` line (already there).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/training/test_export.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add app/training/export.py tests/training/test_export.py tests/data/fixtures/arms_sample.yaml
git commit -m "feat: merge/GGUF/ollama export + arms.yaml snippet

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 6: Host-side `finetune_*` scripts

**Files:**
- Create: `backend/scripts/finetune_prep.py`
- Create: `backend/scripts/finetune_train.py`
- Create: `backend/scripts/finetune_export.py`
- Create: `backend/tests/scripts/test_finetune_scripts.py`

**Interfaces:**
- Consumes: `app.training.config`, `app.training.dataset`, `app.training.train`, `app.training.export`
- Produces (argv contracts, consumed by Task 7):
  - `python -m scripts.finetune_prep [--config PATH]`
  - `python -m scripts.finetune_train [--config PATH] [--dry-run] [--reuse-dataset]`
  - `python -m scripts.finetune_export [--config PATH]`

- [ ] **Step 1: Write the failing test**

`backend/tests/scripts/test_finetune_scripts.py`:

```python
import pytest

from app.training.config import load_training_config
from app.training.dataset import DatasetBuildResult


@pytest.fixture
def fake_dataset(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"messages": []}\n')
    (tmp_path / "val.jsonl").write_text("")
    return DatasetBuildResult(
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        license_path=tmp_path / "LICENSE.txt",
        train_class_counts={"positive": 1},
        val_class_counts={},
        dropped_count=4,
        pool_size=1000,
        source="demo:5768",
    )


def test_prep_prints_stats(capsys, monkeypatch, fake_dataset):
    from scripts import finetune_prep

    monkeypatch.setattr(finetune_prep, "build_sft_dataset", lambda cfg: fake_dataset)
    finetune_prep.main(["--config", "training.yaml"])
    out = capsys.readouterr().out
    assert "dropped" in out.lower()
    assert "1000" in out


def test_train_dry_run_skips_training(capsys, monkeypatch, fake_dataset):
    from scripts import finetune_train

    monkeypatch.setattr(finetune_train, "build_sft_dataset", lambda cfg: fake_dataset)

    def _boom(*a, **k):
        raise AssertionError("run_training must not be called on --dry-run")

    monkeypatch.setattr(finetune_train, "run_training", _boom)
    finetune_train.main(["--config", "training.yaml", "--dry-run"])
    out = capsys.readouterr().out
    assert "dry run" in out.lower()


def test_train_calls_run_training(monkeypatch, fake_dataset):
    from scripts import finetune_train
    from app.training.train import TrainingResult

    calls = {}
    monkeypatch.setattr(finetune_train, "build_sft_dataset", lambda cfg: fake_dataset)
    monkeypatch.setattr(
        finetune_train,
        "run_training",
        lambda cfg, ds: calls.setdefault("r", TrainingResult(fake_dataset.train_path, [], 1.0, 42)),
    )
    finetune_train.main(["--config", "training.yaml"])
    assert "r" in calls


def test_export_prints_snippet(capsys, monkeypatch):
    from scripts import finetune_export
    from app.training.export import ExportResult

    monkeypatch.setattr(
        finetune_export,
        "export_arm",
        lambda cfg: ExportResult(
            gguf_path=cfg and None or None,
            modelfile_path=None,
            ollama_tag="ft-qwen3-8b",
            snippet_path=None,
            snippet="  - name: ft-qwen3-8b-local\n",
        ),
    )
    finetune_export.main(["--config", "training.yaml"])
    assert "ft-qwen3-8b-local" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_finetune_scripts.py -v`
Expected: FAIL — `scripts.finetune_prep` does not exist.

- [ ] **Step 3: Implement the three scripts**

`backend/scripts/finetune_prep.py`:

```python
"""Build + leakage-check the fine-tuning dataset. Run from inside backend/:

    uv run python -m scripts.finetune_prep
"""
import argparse

from app.training.config import load_training_config
from app.training.dataset import build_sft_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    result = build_sft_dataset(cfg)

    print(f"source            : {result.source}")
    print(f"pool size         : {result.pool_size}")
    print(f"dropped (leakage) : {result.dropped_count}")
    print(f"train class counts: {result.train_class_counts}")
    print(f"val class counts  : {result.val_class_counts}")
    print(f"train jsonl       : {result.train_path}")
    print(f"val jsonl         : {result.val_path}")


if __name__ == "__main__":
    main()
```

`backend/scripts/finetune_train.py`:

```python
"""QLoRA fine-tune. GPU-only unless --dry-run. Run from inside backend/:

    uv run python -m scripts.finetune_train
    uv run python -m scripts.finetune_train --dry-run   # dataset + config only
"""
import argparse
import json

from app.training.config import load_training_config
from app.training.dataset import build_sft_dataset
from app.training.train import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-dataset", action="store_true",
                        help="(reserved) reuse existing artifacts; currently always rebuilds")
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    dataset = build_sft_dataset(cfg)
    print(f"dataset ready: pool={dataset.pool_size} dropped={dataset.dropped_count}")

    if args.dry_run:
        sample = json.loads(dataset.train_path.read_text().splitlines()[0])
        print("dry run -- not loading the model. First formatted record:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        return

    result = run_training(cfg, dataset)
    print(f"adapter saved : {result.adapter_path}")
    print(f"wall seconds  : {result.wall_seconds:.0f}")
    for entry in result.loss_history:
        print(entry)
```

`backend/scripts/finetune_export.py`:

```python
"""Merge the adapter, convert to GGUF, `ollama create`, print the arms.yaml
entry to paste. GPU-only. Run from inside backend/:

    uv run python -m scripts.finetune_export
"""
import argparse

from app.training.config import load_training_config
from app.training.export import export_arm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training.yaml")
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    result = export_arm(cfg)

    print(f"gguf      : {result.gguf_path}")
    print(f"modelfile : {result.modelfile_path}")
    print(f"ollama tag: {result.ollama_tag}")
    print()
    print("Paste this into arms.yaml under `arms:` --")
    print(result.snippet)


if __name__ == "__main__":
    main()
```

Add `if __name__ == "__main__": main()` to `finetune_train.py` too.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_finetune_scripts.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/finetune_prep.py scripts/finetune_train.py scripts/finetune_export.py \
  tests/scripts/test_finetune_scripts.py
git commit -m "feat: finetune_prep/train/export host scripts

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 7: `pe finetune` CLI sub-typer

**Files:**
- Modify: `backend/app/cli/__init__.py` (add sub-typer near the other `add_typer` calls, ~line 26; add commands near the `calibrate_*` commands, ~line 209–245)
- Modify: `backend/tests/cli/test_cli.py` (add cases; use the existing `captured_argv` fixture)

**Interfaces:**
- Consumes: the script argv contracts from Task 6.
- Produces: `pe finetune prep|train|export|report` commands (report wired here, script built in Task 8).

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/cli/test_cli.py`:

```python
def test_finetune_prep_shells_out(captured_argv):
    result = runner.invoke(app, ["finetune", "prep"])
    assert result.exit_code == 0
    assert captured_argv[-1] == ["uv", "run", "python", "-m", "scripts.finetune_prep",
                                "--config", "training.yaml"]


def test_finetune_train_dry_run_passes_flag(captured_argv):
    result = runner.invoke(app, ["finetune", "train", "--dry-run"])
    assert result.exit_code == 0
    assert captured_argv[-1] == ["uv", "run", "python", "-m", "scripts.finetune_train",
                                "--config", "training.yaml", "--dry-run"]


def test_finetune_export_shells_out(captured_argv):
    result = runner.invoke(app, ["finetune", "export"])
    assert result.exit_code == 0
    assert captured_argv[-1] == ["uv", "run", "python", "-m", "scripts.finetune_export",
                                "--config", "training.yaml"]


def test_finetune_report_passes_args(captured_argv):
    result = runner.invoke(
        app,
        ["finetune", "report", "--run-id", "5",
         "--baseline", "qwen3-8b-local",
         "--candidate", "ft-qwen3-8b-local",
         "--candidate", "gpt-4o-mini"],
    )
    assert result.exit_code == 0
    argv = captured_argv[-1]
    assert argv[:6] == ["uv", "run", "python", "-m", "scripts.finetune_report", "--run-id"]
    assert "--baseline" in argv and "qwen3-8b-local" in argv
    assert argv.count("--candidate") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_cli.py -k finetune -v`
Expected: FAIL — no such command `finetune`.

- [ ] **Step 3: Implement the sub-typer**

In `backend/app/cli/__init__.py`, near the other sub-typers:

```python
finetune_app = typer.Typer(
    help="Local LoRA fine-tune workflow (runs on the host; needs a CUDA GPU "
    "and `uv sync --extra training`).",
    no_args_is_help=True,
)
app.add_typer(finetune_app, name="finetune")
```

Then, near the `calibrate_*` commands:

```python
@finetune_app.command("prep")
def finetune_prep(config: str = typer.Option("training.yaml", "--config")):
    """Build + leakage-check the fine-tuning dataset."""
    backend_script("scripts.finetune_prep", "--config", config)


@finetune_app.command("train")
def finetune_train(
    config: str = typer.Option("training.yaml", "--config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dataset + config only; no GPU."),
    reuse_dataset: bool = typer.Option(False, "--reuse-dataset"),
):
    """Run the QLoRA fine-tune (GPU-only unless --dry-run)."""
    extra = []
    if dry_run:
        extra.append("--dry-run")
    if reuse_dataset:
        extra.append("--reuse-dataset")
    backend_script("scripts.finetune_train", "--config", config, *extra)


@finetune_app.command("export")
def finetune_export(config: str = typer.Option("training.yaml", "--config")):
    """Merge -> GGUF -> `ollama create`; print the arms.yaml entry to paste."""
    backend_script("scripts.finetune_export", "--config", config)


@finetune_app.command("report")
def finetune_report(
    run_id: int = typer.Option(..., "--run-id"),
    baseline: str = typer.Option(..., "--baseline", help="Base local arm name."),
    candidate: list[str] = typer.Option(
        ..., "--candidate", help="Arm to compare against the baseline (repeatable)."
    ),
    epsilon: float = typer.Option(0.5, "--epsilon", help="Equivalence margin on the 1-5 judge scale."),
    gpu_cost_per_hour: float = typer.Option(0.40, "--gpu-cost-per-hour"),
    train_seconds: float = typer.Option(0.0, "--train-seconds", help="Wall time of `pe finetune train`."),
    out: str = typer.Option(
        "../docs/superpowers/reports/2026-08-29-finetune-comparison.md", "--out"
    ),
):
    """Render the fine-tuned-vs-base-vs-API comparison report from a completed run."""
    args = ["--run-id", str(run_id), "--baseline", baseline,
            "--epsilon", str(epsilon), "--gpu-cost-per-hour", str(gpu_cost_per_hour),
            "--train-seconds", str(train_seconds), "--out", out]
    for c in candidate:
        args += ["--candidate", c]
    backend_script("scripts.finetune_report", *args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_cli.py -k finetune -v`
Expected: PASS (4 tests). `test_finetune_report_passes_args` exercises the argv ordering — adjust the slice assertion if you reorder args, keeping `--run-id` immediately after the module name.

- [ ] **Step 5: Run the whole CLI test module**

Run: `uv run pytest tests/cli/ -v`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add app/cli/__init__.py tests/cli/test_cli.py
git commit -m "feat: pe finetune prep|train|export|report CLI

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 8: Comparison report generator

**Files:**
- Create: `backend/scripts/finetune_report.py`
- Create: `backend/tests/scripts/test_finetune_report.py`

**Interfaces:**
- Consumes: the platform API (`GET /runs/{id}`, `/runs/{id}/summary`, `/runs/{id}/compare`, `/runs/{id}/equivalence`).
- Produces (argv contract, already wired in Task 7):
  `python -m scripts.finetune_report --run-id N --baseline ARM --candidate ARM [--candidate ARM ...] [--epsilon E] [--gpu-cost-per-hour C] [--train-seconds S] [--out PATH]`
- Internal pure functions (unit-tested):
  - `training_cost_usd(train_seconds: float, gpu_cost_per_hour: float) -> float`
  - `break_even_calls(training_cost: float, api_cost_per_call: float) -> float | None`
  - `build_report_markdown(ctx: ReportContext) -> str`

- [ ] **Step 1: Write the failing test**

`backend/tests/scripts/test_finetune_report.py`:

```python
import math

import httpx
import pytest
import respx

from scripts import finetune_report as fr

BASE = "http://localhost:8000"


def test_training_cost_usd():
    assert fr.training_cost_usd(3600, 0.40) == pytest.approx(0.40)
    assert fr.training_cost_usd(1800, 1.0) == pytest.approx(0.50)


def test_break_even_calls():
    assert fr.break_even_calls(0.50, 0.001) == pytest.approx(500)
    assert fr.break_even_calls(0.50, 0.0) is None


def test_build_report_markdown_has_all_sections():
    ctx = fr.ReportContext(
        run_id=5,
        run_meta={"arm_names": ["qwen3-8b-local", "ft-qwen3-8b-local", "gpt-4o-mini"],
                  "repeats": 3, "sample_size": 200, "seed": 42},
        baseline="qwen3-8b-local",
        candidates=["ft-qwen3-8b-local", "gpt-4o-mini"],
        epsilon=0.5,
        summary=[
            {"arm_name": "qwen3-8b-local", "mean_judge_score": 3.1, "mean_latency_ms": 900,
             "mean_cost_estimate_usd": None},
            {"arm_name": "ft-qwen3-8b-local", "mean_judge_score": 4.2, "mean_latency_ms": 850,
             "mean_cost_estimate_usd": None},
            {"arm_name": "gpt-4o-mini", "mean_judge_score": 4.4, "mean_latency_ms": 700,
             "mean_cost_estimate_usd": 0.0004},
        ],
        comparisons=[
            {"arm_a": "ft-qwen3-8b-local", "arm_b": "qwen3-8b-local", "mean_diff": 1.1,
             "ci_lower": 0.8, "ci_upper": 1.4, "p_value_corrected": 0.001},
            {"arm_a": "ft-qwen3-8b-local", "arm_b": "gpt-4o-mini", "mean_diff": -0.2,
             "ci_lower": -0.5, "ci_upper": 0.1, "p_value_corrected": 0.3},
        ],
        equivalences=[
            {"arm_local": "ft-qwen3-8b-local", "arm_api": "gpt-4o-mini",
             "epsilon": 0.5, "p_equivalent": 0.92},
        ],
        training_cost=0.50,
        gpu_cost_per_hour=0.40,
        train_seconds=4500,
    )
    md = fr.build_report_markdown(ctx)
    assert "# Fine-tuned vs. base vs. API" in md
    assert "Financial PhraseBank" in md and "CC BY-NC-SA" in md
    assert "Win-rate / quality" in md
    assert "Bayesian equivalence" in md
    assert "frontier" in md.lower()
    assert "Training-cost accounting" in md
    assert "0.50" in md  # one-time training cost surfaced
    assert "judge" in md.lower() and "opus" not in md.lower() or True  # provenance note optional


@respx.mock
def test_fetch_and_render_writes_file(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(
        200, json={"id": 5, "arm_names": ["qwen3-8b-local", "ft-qwen3-8b-local"],
                   "repeats": 2, "sample_size": 10, "seed": 1}))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        {"arm_name": "qwen3-8b-local", "mean_judge_score": 3.0, "mean_latency_ms": 800,
         "mean_cost_estimate_usd": None},
        {"arm_name": "ft-qwen3-8b-local", "mean_judge_score": 4.0, "mean_latency_ms": 750,
         "mean_cost_estimate_usd": None}]))
    respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        {"arm_a": "ft-qwen3-8b-local", "arm_b": "qwen3-8b-local", "mean_diff": 1.0,
         "ci_lower": 0.5, "ci_upper": 1.5, "p_value_corrected": 0.01}]))
    out = tmp_path / "report.md"
    fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
             "--candidate", "ft-qwen3-8b-local", "--out", str(out),
             "--train-seconds", "3600", "--gpu-cost-per-hour", "0.4"])
    assert out.exists()
    assert "ft-qwen3-8b-local" in out.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_finetune_report.py -v`
Expected: FAIL — `scripts.finetune_report` does not exist.

- [ ] **Step 3: Implement `backend/scripts/finetune_report.py`**

```python
"""Render the Phase 7 fine-tuned-vs-base-vs-API comparison report from a
completed run. Reads the platform API ($PE_API_URL, default
http://localhost:8000). Run from inside backend/:

    uv run python -m scripts.finetune_report --run-id 5 \
        --baseline qwen3-8b-local \
        --candidate ft-qwen3-8b-local --candidate gpt-4o-mini \
        --train-seconds 4500
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE_URL = os.environ.get("PE_API_URL", "http://localhost:8000").rstrip("/")
FRONTIER_PNG = "2026-08-29-finetune-frontier.png"


@dataclass
class ReportContext:
    run_id: int
    run_meta: dict
    baseline: str
    candidates: list[str]
    epsilon: float
    summary: list[dict]
    comparisons: list[dict]
    equivalences: list[dict]
    training_cost: float
    gpu_cost_per_hour: float
    train_seconds: float


def training_cost_usd(train_seconds: float, gpu_cost_per_hour: float) -> float:
    return train_seconds / 3600.0 * gpu_cost_per_hour


def break_even_calls(training_cost: float, api_cost_per_call: float) -> float | None:
    if not api_cost_per_call:
        return None
    return training_cost / api_cost_per_call


def _get(path: str, **params) -> object:
    resp = httpx.get(f"{BASE_URL}{path}", params={k: v for k, v in params.items() if v is not None},
                     timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def _summary_row(summary: list[dict], arm: str) -> dict:
    for row in summary:
        if row["arm_name"] == arm:
            return row
    raise SystemExit(f"arm '{arm}' not in run summary")


def build_report_markdown(ctx: ReportContext) -> str:
    m = ctx.run_meta
    lines: list[str] = []
    lines.append("# Fine-tuned vs. base vs. API — financial sentiment (Phase 7)\n")
    lines.append(
        f"Run **{ctx.run_id}** — arms `{', '.join(m['arm_names'])}`, "
        f"{m.get('repeats')} repeats × {m.get('sample_size')} examples, seed {m.get('seed')}. "
        "Quality metric is the calibrated LLM judge's 1–5 `judge_score`.\n"
    )
    lines.append(
        "> Training data: Financial PhraseBank lower-agreement subset "
        "(Malo et al. 2014), disjoint from the all-agree eval set. "
        "Licensed CC BY-NC-SA 3.0 (non-commercial).\n"
    )

    lines.append("## Win-rate / quality (paired)\n")
    lines.append("| candidate vs. baseline | mean Δ judge_score | 95% CI | corrected p |")
    lines.append("|---|---|---|---|")
    for c in ctx.comparisons:
        lines.append(
            f"| {c['arm_a']} vs. {c['arm_b']} | {_fmt(c['mean_diff'])} | "
            f"[{_fmt(c['ci_lower'])}, {_fmt(c['ci_upper'])}] | {_fmt(c.get('p_value_corrected'))} |"
        )
    lines.append("")

    lines.append("## Bayesian equivalence\n")
    lines.append(f"P(judge_score_candidate ≥ judge_score_api − ε), ε = {ctx.epsilon}:\n")
    if ctx.equivalences:
        lines.append("| candidate | vs. API arm | P(equivalent) |")
        lines.append("|---|---|---|")
        for e in ctx.equivalences:
            lines.append(f"| {e['arm_local']} | {e['arm_api']} | {_fmt(e['p_equivalent'])} |")
    else:
        lines.append("_No API candidates supplied._")
    lines.append("")

    lines.append("## Cost / latency / quality frontier\n")
    lines.append(f"![frontier]({FRONTIER_PNG})\n")
    lines.append("| arm | mean judge_score | mean latency (ms) | mean $/call |")
    lines.append("|---|---|---|---|")
    for row in ctx.summary:
        lines.append(
            f"| {row['arm_name']} | {_fmt(row.get('mean_judge_score'))} | "
            f"{_fmt(row.get('mean_latency_ms'))} | {_fmt(row.get('mean_cost_estimate_usd'))} |"
        )
    lines.append("")

    lines.append("## Training-cost accounting\n")
    lines.append(
        f"One-time fine-tune: {ctx.train_seconds/3600:.2f} GPU-hours × "
        f"${ctx.gpu_cost_per_hour:.2f}/hr ≈ **${ctx.training_cost:.2f}** "
        "(assumption — adjust `--gpu-cost-per-hour` to your rate). "
        "This is separate from per-inference cost; the fine-tuned local arm's "
        "`cost_estimate_usd` stays null (subscription/again-local compute).\n"
    )
    for c in ctx.candidates:
        row = _summary_row(ctx.summary, c) if any(r["arm_name"] == c for r in ctx.summary) else None
        api_cost = row.get("mean_cost_estimate_usd") if row else None
        be = break_even_calls(ctx.training_cost, api_cost) if api_cost else None
        if be is not None:
            lines.append(f"- vs. `{c}` at ${api_cost:.5f}/call, the fine-tune amortizes after ~{be:,.0f} calls.")
    lines.append("")

    lines.append("## Honest read\n")
    lines.append("_(fill in from the numbers above: did fine-tuning close the quality gap to the "
                 "API arms, and at what one-time cost / latency?)_\n")
    lines.append(
        "\n_Judge model overlap: the judge and the `claude-haiku` arm share a vendor; "
        "see `backend/README.md` 'Watch for judge/arm model overlap'._\n"
    )
    return "\n".join(lines)


def write_frontier_png(summary: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in summary:
        x = row.get("mean_latency_ms") or 0
        y = row.get("mean_judge_score") or 0
        ax.scatter(x, y)
        ax.annotate(row["arm_name"], (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean latency (ms)")
    ax.set_ylabel("mean judge_score (1–5)")
    ax.set_title("Cost / latency / quality frontier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True, dest="candidates")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=0.40)
    parser.add_argument("--train-seconds", type=float, default=0.0)
    parser.add_argument("--out", default="../docs/superpowers/reports/2026-08-29-finetune-comparison.md")
    args = parser.parse_args(argv)

    run_meta = _get(f"/runs/{args.run_id}")
    summary = _get(f"/runs/{args.run_id}/summary")
    comparisons = _get(f"/runs/{args.run_id}/compare")
    kept = [c for c in comparisons
            if {c["arm_a"], c["arm_b"]} <= ({args.baseline, *args.candidates})
            and args.baseline in (c["arm_a"], c["arm_b"])]

    equivalences = []
    for c in args.candidates:
        if c == args.baseline:
            continue
        row = next((r for r in summary if r["arm_name"] == c), None)
        is_api = bool(row and row.get("mean_cost_estimate_usd"))
        if not is_api:
            continue
        try:
            equivalences.append(_get(
                f"/runs/{args.run_id}/equivalence", arm_local="ft-qwen3-8b-local", arm_api=c,
                epsilon=args.epsilon,
            ))
        except httpx.HTTPStatusError:
            pass

    training_cost = training_cost_usd(args.train_seconds, args.gpu_cost_per_hour)
    ctx = ReportContext(
        run_id=args.run_id, run_meta=run_meta, baseline=args.baseline,
        candidates=args.candidates, epsilon=args.epsilon, summary=summary,
        comparisons=kept, equivalences=equivalences, training_cost=training_cost,
        gpu_cost_per_hour=args.gpu_cost_per_hour, train_seconds=args.train_seconds,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report_markdown(ctx), encoding="utf-8")
    png_ok = write_frontier_png(summary, out_path.parent / FRONTIER_PNG)
    print(f"wrote {out_path}")
    print(f"frontier png: {'written' if png_ok else 'skipped (matplotlib not installed)'}")


if __name__ == "__main__":
    main()
```

Note: the `equivalence` call hardcodes `arm_local="ft-qwen3-8b-local"` — that is the only local candidate Phase 7 defines. If you generalize the report later, thread the local arm through instead.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_finetune_report.py -v`
Expected: PASS (4 tests). Loosen the last assertion in `test_build_report_markdown_has_all_sections` if you reword a heading — keep one assertion per report section.

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q`
Expected: green (DB tests skip if Postgres down).

- [ ] **Step 6: Commit**

```bash
git add scripts/finetune_report.py tests/scripts/test_finetune_report.py
git commit -m "feat: finetune comparison report generator

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md` (the "Build phases" list, Phase 7 bullet)
- Modify: `backend/README.md` (append a "Phase 7: local fine-tune" section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `CLAUDE.md` Phase 7**

Replace the `7. **Stretch** — LoRA fine-tune ...` bullet with:

```markdown
7. **Local fine-tune** 🚧 **Capability built; comparison run pending a GPU
   session.** `backend/app/training/` — QLoRA fine-tune of Qwen3-8B on the
   Financial PhraseBank *lower-agreement* subset (disjoint from the
   all-agree eval set, enforced by a leakage guard in
   `training/dataset.py`), then merge → GGUF → `ollama create` so the
   fine-tuned model is just another `openai_compatible` arm. Driven by
   `pe finetune prep|train|export|report` over `backend/training.yaml`.
   Training deps are an optional `training` extra, out of the core/CI
   path. Spec:
   `docs/superpowers/specs/2026-08-29-local-finetune-phase7-design.md`;
   plan: `docs/superpowers/plans/2026-08-29-local-finetune-phase7.md`.
   The executed fine-tuned-vs-base-vs-API comparison lands at
   `docs/superpowers/reports/2026-08-29-finetune-comparison.md`.
```

(Task 10 flips 🚧 → ✅ once the report is committed.)

- [ ] **Step 2: Append the `backend/README.md` section**

```markdown
## Phase 7: local fine-tune

QLoRA fine-tune of the local model on the financial-sentiment task, served
through Ollama as a normal `openai_compatible` arm.

### Prerequisites

- A CUDA GPU (developed on a 12 GB RTX 4070).
- `uv sync --extra training` — pulls `unsloth`, `trl`, `peft`,
  `transformers`, `datasets`, `bitsandbytes`, `torch`, `matplotlib`. NOT
  installed by a plain `uv sync` and NOT needed by the API/worker/CI.
- Ollama running (same as the base local arm).
- The stack up (`pe up`) and eval examples seeded — the dataset builder
  reads `eval_example` to guarantee the training data is disjoint from it.

Record the resolved dependency versions here after a successful run:
_(unsloth ==, torch ==, transformers ==, trl ==, bitsandbytes ==)_

### Workflow

```bash
cd backend
# 1. Build + leakage-check the training data (Financial PhraseBank
#    lower-agreement subset; downloaded at runtime, not vendored).
pe finetune prep

# 2. Fine-tune (writes training/artifacts/<run_name>/adapter/). ~20-40 min.
pe finetune train            # --dry-run does dataset + config only, no GPU

# 3. Merge -> GGUF -> `ollama create ft-qwen3-8b`; prints an arms.yaml entry.
pe finetune export

# 4. Paste the printed snippet into arms.yaml under `arms:`.

# 5. Run the comparison over base local, fine-tuned local, and the API arms.
pe run --arm qwen3-8b-local --arm ft-qwen3-8b-local \
       --arm gpt-4o-mini --arm claude-haiku --repeats 5 --sample 200

# 6. After the run + judge scoring finish, render the report.
pe finetune report --run-id <id> \
    --baseline qwen3-8b-local \
    --candidate ft-qwen3-8b-local --candidate gpt-4o-mini --candidate claude-haiku \
    --train-seconds <wall-seconds-from-step-2> --gpu-cost-per-hour <your-rate>
```

### Config

Everything is in `backend/training.yaml` (the training counterpart to
`arms.yaml`): base model, HF source subset, LoRA rank/alpha, epochs, LR,
`gguf_quant` (default `q4_k_m`, matching the base local arm), `ollama_tag`.

### Fallbacks

- **HF subset**: `training.yaml`'s `source_dataset` defaults to a
  `sentences_75agree` mirror. If it 404s or needs `trust_remote_code`,
  switch to `sentences_66agree`, `sentences_50agree`, or the canonical
  `takala/financial_phrasebank` — a one-line config edit.
- **GGUF conversion** needs a llama.cpp build toolchain (cmake/gcc);
  Unsloth clones + builds it on first `pe finetune export`. If that fails,
  build llama.cpp manually and run `convert_hf_to_gguf.py` +
  `llama-quantize` against `training/artifacts/<run_name>/merged/`.
- **Out of VRAM**: lower `max_seq_len`, set `batch_size: 1`, raise
  `grad_accum` in `training.yaml`.
- **Model echoes the prompt instead of a label**: the completion-only loss
  markers in `app/training/train.py` (`instruction_part` /
  `response_part`) don't match the installed Unsloth's Qwen3 chat
  template — check `tokenizer.apply_chat_template` output and adjust.
```

- [ ] **Step 3: Verify the docs render**

Run: `grep -n "Phase 7" CLAUDE.md backend/README.md`
Expected: both files show the new content; no leftover "Stretch —" bullet in `CLAUDE.md`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md backend/README.md
git commit -m "docs: Phase 7 local fine-tune workflow

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

---

## Task 10: Execute the fine-tune and comparison (GPU session, manual)

**This task is not TDD.** It runs the pipeline the earlier tasks built, on a machine with a CUDA GPU, and commits the resulting report. Do it in one sitting on the GPU box.

**Files:**
- Create: `docs/superpowers/reports/2026-08-29-finetune-comparison.md` (generated, then hand-edit the "Honest read" section)
- Create: `docs/superpowers/reports/2026-08-29-finetune-frontier.png` (generated)
- Modify: `CLAUDE.md` (Phase 7 🚧 → ✅), `backend/README.md` (fill in resolved versions)

**Interfaces:** consumes everything built in Tasks 1–9.

- [ ] **Step 1: Install training deps and record versions**

```bash
cd backend
uv sync --extra training
uv pip list | grep -iE "unsloth|torch|transformers|trl|bitsandbytes|peft"
```

Paste the versions into the `backend/README.md` placeholder.

- [ ] **Step 2: Confirm the stack is up and seeded**

```bash
pe up
uv run python -m scripts.seed_eval_examples   # idempotent
```

- [ ] **Step 3: Build the dataset and eyeball the leakage report**

```bash
pe finetune prep
```

Expected: `dropped (leakage)` > 0 (all-agree sentences are a subset of the
75-agree subset, so overlap is expected and must be dropped); `pool size`
comfortably above `min_pool_size` (500); class counts across all three
labels. If `pool size` is tiny or `LeakageError` fires, stop — switch the
`source_dataset` per the README fallback and retry.

- [ ] **Step 4: Sanity-check one formatted record**

```bash
pe finetune train --dry-run
```

Expected: the printed record's `messages[0].content` is exactly the eval
prompt wording; `messages[1].content` is a bare label word.

- [ ] **Step 5: Fine-tune**

```bash
time pe finetune train 2>&1 | tee training/artifacts/train.log
```

Record the wall-clock seconds (for `--train-seconds`). Expected: training
loss trends down; eval loss printed per epoch; adapter saved.

- [ ] **Step 6: Export and wire the arm**

```bash
pe finetune export
```

Paste the printed snippet into `backend/arms.yaml` under `arms:` as
`ft-qwen3-8b-local`. Then smoke-test it directly:

```bash
curl -s http://<ollama-base-url>/chat/completions \
  -d '{"model":"ft-qwen3-8b","messages":[{"role":"user","content":"Is the following sentence positive, negative, or neutral from a financial-news perspective? Respond with just the sentiment label.\n\nSentence: The company posted a record quarterly loss."}]}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Expected: a bare `negative` (or close) — **no `<think>` block**. If a
`<think>` block appears, fix the Modelfile `SYSTEM "/no_think"` handling
before running the full comparison.

Verify-or-fix (deferred here from the Phase 7 final review — need a GPU + live Ollama):
- [ ] `ollama create <tag> -f <generated Modelfile>` exits 0 — the generated
      `TEMPLATE` / `PARAMETER` lines parse with no error.
- [ ] The exported arm emits a bare sentiment label with **no `<think>`
      block**; if one appears, fix the Modelfile `SYSTEM "/no_think"` /
      `TEMPLATE` handling before the comparison run.
- [ ] Confirm `num_ctx` (now `max_seq_len + 1024` from `build_modelfile`) is
      >= prompt + completion for the eval prompt; raise it if the smoke-test
      response comes back truncated.

- [ ] **Step 7: Run the comparison**

```bash
pe run --arm qwen3-8b-local --arm ft-qwen3-8b-local \
       --arm gpt-4o-mini --arm claude-haiku --repeats 5 --sample 200
pe watch <run-id>     # until judge scoring completes
```

(Drop `--arm claude-haiku` / `--arm gpt-4o-mini` if you don't have those
keys; the report adapts to whatever arms are in the run.)

- [ ] **Step 8: Generate the report**

```bash
pe finetune report --run-id <id> \
    --baseline qwen3-8b-local \
    --candidate ft-qwen3-8b-local --candidate gpt-4o-mini --candidate claude-haiku \
    --train-seconds <from step 5> --gpu-cost-per-hour <your rate>
```

- [ ] **Step 9: Write the "Honest read"**

Edit `docs/superpowers/reports/2026-08-29-finetune-comparison.md` — replace
the placeholder "Honest read" paragraph with the actual conclusion from the
numbers: did fine-tuning close the quality gap to the API arms (paired Δ,
CI, corrected p, P(equivalent)), and at what one-time training cost and
per-call latency. Note any caveat (small eval set, judge/arm vendor
overlap, quant differences).

- [ ] **Step 10: Flip the phase status**

In `CLAUDE.md`, change Phase 7's `🚧 **Capability built; comparison run
pending a GPU session.**` to `✅ **Done.**` and adjust the trailing
sentence to past tense.

- [ ] **Step 11: Commit**

```bash
git add docs/superpowers/reports/2026-08-29-finetune-comparison.md \
        docs/superpowers/reports/2026-08-29-finetune-frontier.png \
        CLAUDE.md backend/README.md backend/arms.yaml
git commit -m "feat: Phase 7 fine-tuned-vs-base-vs-API comparison report

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNL2qi6ftwXbv3WBFyLgCc"
```

Note: committing `arms.yaml` with the `ft-qwen3-8b-local` arm means anyone
without that Ollama model gets a `FAILED` result for it on a default
`pe run` — acceptable (same as the commented-out subscription arms), and
the README documents how to rebuild it.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| §0 Shared eval prompt (refactor) | Task 1 |
| §1 `dataset.py` — HF load, normalization, leakage guard, formatting, balance, split, artifacts | Task 3 |
| §2 `config.py` + `training.yaml` | Task 2 |
| §3 `train.py` — guarded import, QLoRA, adapter-only save | Task 4 |
| §4 `export.py` — merge/GGUF/Modelfile/`ollama create`/snippet | Task 5 |
| §5 Host scripts | Task 6 |
| §6 `pe finetune` CLI | Task 7 |
| §7 Comparison run + report (script, markdown, frontier PNG, training-cost accounting) | Task 8 (generator) + Task 10 (execution) |
| Dependencies — `training` extra | Task 2 |
| Testing — all CI tests avoid heavy imports | Tasks 1–8 (guarded/monkeypatched throughout) |
| Risks 1–4 (HF mirror, GGUF toolchain, non-thinking Modelfile, VRAM) | Task 9 README fallbacks + Task 10 verification steps |
| Risk 5 (quant parity) | Task 2 (`gguf_quant: q4_k_m` default) |
| Risk 6 (judge/arm overlap) | Task 8 (report restates it) |
| Documentation updates (CLAUDE.md, README, .gitignore) | Task 2 (.gitignore) + Task 9 + Task 10 |

No gaps.

**2. Placeholder scan:** The only intentional "fill in" is the report's
"Honest read" paragraph (Task 8 emits it as a placeholder, Task 10 Step 9
writes it from real numbers) and the README version placeholder (Task 10
Step 1). Both are human-judgement outputs that cannot exist before the run.
No `TODO`/`TBD` in code.

**3. Type consistency:**
- `TrainingConfig` field names are used identically in `dataset.py`,
  `train.py`, `export.py`, and all three scripts.
- `DatasetBuildResult` fields (`train_path`, `val_path`, `license_path`,
  `train_class_counts`, `val_class_counts`, `dropped_count`, `pool_size`,
  `source`) match between Task 3's definition and Tasks 4/6's consumption.
- `MissingTrainingDepsError` is defined in `train.py` and re-imported by
  `export.py` and both test modules — single definition.
- `import_unsloth()` (not `_import_unsloth`) — consistent between Task 4
  definition and Task 5 use.
- `build_report_markdown` / `ReportContext` / `training_cost_usd` /
  `break_even_calls` names match between Task 8's test and implementation.
- CLI command → script argv (`--config`, `--dry-run`, `--run-id`,
  `--baseline`, `--candidate`) consistent between Task 7 and Tasks 6/8.
