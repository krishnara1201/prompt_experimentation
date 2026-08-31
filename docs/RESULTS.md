# Results

A walk-through of what this project actually measured, for someone who has
not seen the repo. The theme throughout: **compare LLM arms the way you'd run
an A/B test** — paired tests on the per-example difference, a Bayesian "is it
good enough" question, and a judge you've checked against humans — rather than
reading a single leaderboard score.

Every experiment below uses the same machinery: each arm sees the **same**
eval examples, every example is run **multiple times** per arm (LLMs are
non-deterministic), an LLM judge scores each output 1–5 against the gold
label, and the analysis runs on the **paired** per-example score difference
(hierarchical bootstrap CI + Wilcoxon signed-rank, Holm-corrected across arm
pairs) plus a PyMC Bayesian equivalence test.

---

## The question

Can a **local** model — an 8B model on a 12GB consumer GPU — stand in for a
hosted API model on a real classification task, once you account for quality,
latency, and cost honestly?

Three experiments chip at that question. A fourth leg — the actual
local-vs-hosted-API run — has **not landed** (see "The open gap" below).

---

## Experiment 1 — does a small fine-tune beat the base model?

**Setup.** Base `qwen3:8b` vs. a QLoRA fine-tune of it (rank 16, 3 epochs,
807s on an RTX 4070 12GB), on financial-sentiment classification (Financial
PhraseBank, 3 classes). The fine-tune's training data is the *lower*-agreement
slice of the dataset, held disjoint from the all-agree eval set by a leakage
guard. 50 examples × 3 repeats per arm.

**What it found.**

| | base | fine-tune |
|---|---|---|
| judge accuracy | 80.0% | **90.0%** |
| latency / call | 6,744 ms | **3,804 ms** (1.77× faster) |
| output tokens / call | 296.6 | **6.0** (~49× fewer) |

The paired Wilcoxon test puts the quality gain at **p = 0.031** after
correction, and the Bayesian posterior for the difference is **+0.30 with a
94% interval of [+0.07, +0.53] — entirely above zero**. So this is a real
quality *win*, not just "good enough."

The latency and token deltas share one mechanism: the base model runs with
chain-of-thought on and emits a few hundred tokens of reasoning per call; the
fine-tune was trained to emit just the label. Same hardware, same
quantization — ~1.8× the throughput at *higher* accuracy. One-time training
cost was about **$0.05** of GPU time.

Full report:
[`docs/superpowers/reports/2026-08-30-finetune-comparison.md`](superpowers/reports/2026-08-30-finetune-comparison.md).

---

## Experiment 2 — does "think step by step" actually help?

**Setup.** Same model (`qwen3:8b`) on both arms, native thinking off. The two
arms differ in **one thing**: the prompt. One says *"answer with only the
label"*; the other says *"reason step by step, then give the label."* Task is
AG News topic classification (4 classes). 120 examples × 2 repeats per arm.

**What it found.** Chain-of-thought "wins" the raw accuracy count — 79.1% vs
77.5%. **A leaderboard would stop there and call CoT better.** The paired test
does not:

- paired Wilcoxon corrected **p = 0.37** — no detectable difference
- Bayesian posterior **+0.06**, entire 94% interval inside ±0.2,
  **P(equivalent) = 1.00** at ε = 0.5
- meanwhile CoT costs **~40× the output tokens** (p ≈ 2e-21) and ~6× the
  per-call latency (uncontended)

The 1.6-point accuracy gap is noise. Because both prompts saw the same 120
snippets, the paired test can see that the gap is within-example jitter, not a
real effect — and it redirects attention to the cost axis, where the
difference is large and unambiguous. **Recommendation: the terse prompt.**

This is the clearest demonstration of why the project exists: same data, two
ways of reading it, opposite conclusions.

Full report:
[`docs/superpowers/reports/2026-08-31-prompt-ab-comparison.md`](superpowers/reports/2026-08-31-prompt-ab-comparison.md).

---

## Experiment 3 — can you trust the LLM judge?

Every number above rests on an LLM (`qwen3:8b`) scoring outputs 1–5 against
the gold label. Before trusting it, score a hand-labeled subset and check
agreement.

**Setup.** A `qwen3:8b` arm prompted to *explain* its call in prose ("The
sentiment is negative because…") rather than emit a bare label — so the judge
has to read a rationale, not string-match a token. 120 calls, all judged,
then a 50-row stratified subset hand-labeled **blind to the judge's score**.

**What it found.**

| n | Spearman r (1–5) | Cohen's κ (label right / wrong) | mean \|Δ\| |
|---|---|---|---|
| 50 | 0.729 | **1.000** | 0.06 |

- **On the decision that matters — is the label right? — judge and human
  agree on all 50/50 rows.**
- The exact 1–5 scores agree only moderately (Spearman 0.729) for one
  reason: the judge uses a **2-value vocabulary** — 5 (right) or 2 (wrong),
  never 3 or 4. The only 3 disagreements are all "judge said 5, human said 4"
  on responses that appended a redundant `Sentiment: <Label>` line — a minor
  format slip the judge ignored and a human docks a point for.

So the judge is a **trustworthy label-correctness classifier** on these
classification tasks (this run and the AG News run both hit κ = 1.00), but
**not a calibrated quality scorer** — three of the five rubric points are
dead letters. That's fine for what the paired stats need (`judge_score ≥ 4`
as an accuracy proxy); it would not be fine for a task that needs partial
credit.

Full report:
[`docs/superpowers/reports/2026-08-31-judge-calibration-financial.md`](superpowers/reports/2026-08-31-judge-calibration-financial.md).

---

## The open gap — no hosted-API arm has ever completed a run

This is the honest hole in the project. The headline question — *is the local
model good enough to replace a hosted API arm?* — needs a hosted-API arm in a
paired run, and that has not happened:

- A `gemini-flash` arm was added, but Google's free tier rate-limited **130
  of 150 calls** (HTTP 429). Only 20 calls across 8 examples landed — not
  enough for any paired test to converge.
- Running it for real needs a **paid API key** (~$5 of OpenAI / Anthropic /
  Gemini credit) or a deliberately throttled re-run at concurrency 1.

The entire pipeline that would consume that data is built and exercised — the
equivalence endpoint, the frontier plot, the paired tests — against the
base-vs-fine-tune and prompt-vs-prompt comparisons. The API leg is a config
edit and a run away; it just costs money this project hasn't spent.

Stated plainly: **this project rigorously compares two local models and two
prompts, and calibrates its judge. It does not yet have the
local-vs-hosted-API number it was built to produce.**

---

## What's reusable here

- **Paired stats for LLM evals** (`backend/app/stats/`) — bootstrap +
  Wilcoxon + Holm, a PyMC equivalence test, a power calculator. Drop-in for
  any arm-vs-arm comparison.
- **Task packs** (`backend/tasks/`) — a new eval is a `task.yaml` + a JSONL
  file, no code change. Financial sentiment and AG News ship.
- **Config-driven arms** (`backend/arms.yaml`) — Ollama, OpenAI-schema
  providers, Anthropic, and subscription-seat CLI sessions behind one
  interface. Prompts are arms too.
- **Judge calibration workflow** — `pe calibrate select | import | report`
  reports Spearman + Cohen's κ against a hand-labeled gold subset before you
  trust judge scores.
