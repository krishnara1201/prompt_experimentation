# Fine-tuned vs. base vs. API — financial sentiment (Phase 7, re-run)

Run **1914** (`completed_with_errors`) — arms `qwen3-8b-local` (base),
`ft-qwen3-8b-local` (QLoRA fine-tune), `gemini-flash` (API). 50 eval examples ×
3 repeats = 150 calls per arm, seed 11. Quality metric is the LLM judge's 1–5
`judge_score` (judge: local `qwen3:8b`).

This supersedes the run-1731 report (`2026-08-29-finetune-comparison.md`),
which had only 22 usable paired quality examples after a judge rate-limit
burst. This run judged **all 150 calls per local arm** and re-scored every
result with a single consistent judge — the base-vs-fine-tune comparison here
is the conclusive one. The API leg still did not land (see the Gemini caveat).

> Training data: Financial PhraseBank lower-agreement subset (Malo et al. 2014),
> disjoint from the all-agree eval set (leakage guard: 2259 overlapping rows
> dropped, 2582-row training pool). Licensed CC BY-NC-SA 3.0 (non-commercial).
> Fine-tune: QLoRA r=16, 3 epochs, 807s wall on an RTX 4070 12GB. Eval loss
> 0.0736 → 0.0668 → 0.0769 (mild epoch-3 overfit).

## Win-rate / quality (paired)

| candidate vs. baseline | mean Δ judge_score | 95% CI | corrected p |
|---|---|---|---|
| qwen3-8b-local vs. ft-qwen3-8b-local | **−0.30** | [−0.60, −0.06] | **0.031** |

The judge emitted only two values on this run — **5** (label correct) or **2**
(label wrong) — so `judge_score` is effectively a binary accuracy proxy.

| arm | correct / n | accuracy |
|---|---|---|
| qwen3-8b-local (base) | 120 / 150 | **80.0%** |
| ft-qwen3-8b-local | 135 / 150 | **90.0%** |
| gemini-flash | 16 / 20 | 80.0% (thin — see caveat) |

Paired Wilcoxon signed-rank on the 50 aggregated examples: p = 0.010, Holm-
corrected p = **0.031**. The fine-tune is a **significant quality win** over the
base on this run — not just directional as in run 1731.

## Bayesian equivalence

`metric=judge_score`, ε = 0.5 on the 1–5 scale, fine-tune vs. the base arm as
the reference (no API arm with enough data — see caveat):

| | posterior mean Δ (ft − base) | 94% CI | P(ft ≥ base − ε) |
|---|---|---|---|
| ft-qwen3-8b-local vs. qwen3-8b-local | **+0.30** | [+0.07, +0.53] | **1.00** |

The posterior sits **entirely above 0** — this run supports a real quality
*gain* from fine-tuning, not merely "good enough" equivalence (run 1731's
posterior barely crossed 0 at +0.14).

## Latency & output efficiency (paired, all 150 calls/arm)

| metric | base | fine-tune | paired mean Δ | 95% CI | p |
|---|---|---|---|---|---|
| latency / call | 6,744 ms | 3,804 ms | **−2,939 ms (1.77× faster)** | [−4,122, −1,721] | 1.6e-10 |
| completion tokens / call | 296.6 | 6.0 | **−290.6 (~49× fewer)** | [−325, −260] | <1e-15 |

Same story as run 1731, same mechanism: the base arm runs Qwen3 with thinking
on and emits a full reasoning trace; the fine-tune (bare-label targets, served
`/no_think`) emits just the label. Same hardware, same q4_k_m quant, ~1.8× the
throughput and a fraction of the output tokens — now at a *measurably higher*
accuracy, not just equal.

## Cost / latency / quality frontier

![frontier](2026-08-30-finetune-frontier.png)

| arm | n | mean judge_score | mean latency (ms) | mean $/call |
|---|---|---|---|---|
| qwen3-8b-local | 150 | 4.4 | 6,744 | — (local) |
| ft-qwen3-8b-local | 150 | 4.7 | 3,804 | — (local) |
| gemini-flash | 20 | 4.4 | 6,950¹ | 1.86e-05 |

¹ Gemini latency is inflated by rate-limit retry backoff; not a clean number.

## Training-cost accounting

One-time fine-tune: 807s = **0.224 GPU-h** × $0.20/hr ≈ **$0.045** (assumption —
adjust to your rate). Separate from per-inference cost; the fine-tuned local
arm's `cost_estimate_usd` stays null (local compute, no per-token price). With
~49× fewer output tokens per call it would also be the cheaper option on any
per-token-priced comparison — break-even against a metered API arm is trivial,
but can't be computed here without a working API arm.

## Honest read

- **Quality: the fine-tune wins.** +10 pts raw accuracy (90.0% vs 80.0%),
  paired Wilcoxon corrected p = 0.031, and a Bayesian posterior that clears
  zero (+0.30, 94% CI [+0.07, +0.53], P(equivalent) = 1.00). Run 1731 could
  only call this directional; the full judge sample here makes it significant.
- **Latency / output: unambiguous, as before.** 1.77× faster per call, ~49×
  fewer completion tokens, p ≤ 1.6e-10. For a local no-per-token-price arm the
  payoff is wall-clock throughput on the eval loop.
- **One-time cost:** ~$0.045 at $0.20/GPU-h. Trivially amortised.
- **Still no API comparison.** The `gemini-flash` arm was rate-limited to
  death — 130/150 calls failed with HTTP 429 (Google AI Studio free tier,
  ~15 req/min). Only 20 calls across 8 distinct examples landed; every paired
  test against Gemini is p = 1.0 after correction, and the paired Bayesian
  model does not converge on 8 examples. A real local-vs-API frontier needs a
  **paid** Gemini/OpenAI/Anthropic key, or a deliberately throttled re-run
  (concurrency 1, spaced calls).
- **Next re-run should** (a) use a metered API arm with a paid key at low
  concurrency, (b) optionally stop the fine-tune at epoch 2 given the epoch-3
  eval-loss uptick, (c) attach calibration labels so `judge_score` is a
  calibrated agreement figure, not an uncalibrated accuracy proxy.

## Run caveats

- **Judge consistency.** The run started with an `opus` / `claude_code_cli`
  judge; a subscription usage limit was hit after ~67 judge calls. The judge
  was switched to local `qwen3:8b` and **all 320 completed results were
  re-scored** from scratch with it (`backend/scripts/rejudge_run.py`), so every
  score in this report comes from one judge model.
- **Judge / arm model overlap.** The judge (`qwen3:8b`) is the *same base
  model* as `qwen3-8b-local` and the parent of the fine-tune. LLM-as-judge
  self-preference would bias toward the local arms over Gemini — but Gemini
  isn't the comparison that matters here, and base vs. fine-tune share the
  parent so relative bias between them is limited. Still: not a calibrated
  judge. See `backend/README.md` "Watch for judge/arm model overlap".
- **No calibration.** No human-labeled gold subset was attached to this run;
  judge agreement with human labels is not established. Treat `judge_score`
  as an uncalibrated accuracy proxy.
- **Infra.** A Docker Desktop outage interrupted the re-judge mid-way;
  recovered from the Postgres volume and the 10 stragglers were re-enqueued.
  Redis (no volume) lost its queue in the outage — accounted for in the
  re-enqueue.
