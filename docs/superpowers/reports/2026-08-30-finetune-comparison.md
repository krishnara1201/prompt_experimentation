# Deliverable 1 — QLoRA fine-tune vs. base Qwen3-8B on financial sentiment

**Question.** Does a small QLoRA fine-tune of the local model beat the base
model on the financial-sentiment eval — on quality, and on cost/latency — with
paired statistical rigor rather than a win-rate?

**Setup.** Run **1914** — arms `qwen3-8b-local` (base) and `ft-qwen3-8b-local`
(QLoRA fine-tune). 50 eval examples × 3 repeats = 150 calls per arm, seed 11.
Quality metric is the LLM judge's 1–5 `judge_score`; judge is local `qwen3:8b`,
and every completed result was scored by that one judge model. A `gemini-flash`
API arm was included but did not produce enough data to compare (footnote 1).

> **Training data.** Financial PhraseBank lower-agreement subset (Malo et al.
> 2014), disjoint from the all-agree eval set — leakage guard drops 2259
> overlapping rows, leaving a 2582-row training pool. Licensed CC BY-NC-SA 3.0
> (non-commercial). Fine-tune: QLoRA r=16, 3 epochs, 807 s wall on an RTX 4070
> 12 GB. Eval loss 0.0736 → 0.0668 → 0.0769 (mild epoch-3 overfit).

## Quality — the fine-tune wins (paired)

| candidate vs. baseline | mean Δ judge_score | 95% CI | corrected p |
|---|---|---|---|
| ft-qwen3-8b-local vs. qwen3-8b-local | **+0.30** | [+0.06, +0.60] | **0.031** |

Paired Wilcoxon signed-rank on the 50 aggregated examples: p = 0.010,
Holm-corrected **p = 0.031**. The judge emitted only two values on this run —
**5** (label correct) or **2** (label wrong) — so `judge_score` is effectively a
binary accuracy proxy:

| arm | correct / n | accuracy |
|---|---|---|
| qwen3-8b-local (base) | 120 / 150 | **80.0%** |
| ft-qwen3-8b-local | 135 / 150 | **90.0%** |

+10 points of raw accuracy, and the paired test clears significance after
multiple-comparison correction.

## Bayesian equivalence — actually a gain, not just "good enough"

`metric=judge_score`, ε = 0.5 on the 1–5 scale, base arm as the reference:

| | posterior mean Δ (ft − base) | 94% CI | P(ft ≥ base − ε) |
|---|---|---|---|
| ft-qwen3-8b-local vs. qwen3-8b-local | **+0.30** | [+0.07, +0.53] | **1.00** |

The posterior sits **entirely above 0** — this supports a real quality *gain*
from fine-tuning, not merely equivalence.

## Latency & output efficiency (paired, all 150 calls/arm)

| metric | base | fine-tune | paired mean Δ | 95% CI | p |
|---|---|---|---|---|---|
| latency / call | 6,744 ms | 3,804 ms | **−2,939 ms (1.77× faster)** | [−4,122, −1,721] | 1.6e-10 |
| completion tokens / call | 296.6 | 6.0 | **−290.6 (~49× fewer)** | [−325, −260] | <1e-15 |

Mechanism: the base arm runs Qwen3 with thinking on and emits a full reasoning
trace; the fine-tune (bare-label targets, served `/no_think`) emits just the
label. Same hardware, same q4_k_m quant — ~1.8× the throughput and a fraction
of the output tokens, now at *measurably higher* accuracy.

## Cost / latency / quality frontier

![frontier](2026-08-30-finetune-frontier.png)

| arm | n | mean judge_score | mean latency (ms) | mean $/call |
|---|---|---|---|---|
| qwen3-8b-local | 150 | 4.4 | 6,744 | — (local) |
| ft-qwen3-8b-local | 150 | 4.7 | 3,804 | — (local) |

**One-time training cost:** 807 s = 0.224 GPU-h × $0.20/hr ≈ **$0.045**
(assumption — adjust to your rate). Separate from per-inference cost; the
fine-tuned arm's `cost_estimate_usd` stays null (local compute, no per-token
price). With ~49× fewer output tokens per call it would also be the cheaper
option on any per-token-priced comparison.

## Honest read

- **Quality: the fine-tune wins.** 90.0% vs 80.0% accuracy, paired Wilcoxon
  corrected p = 0.031, Bayesian posterior +0.30 (94% CI [+0.07, +0.53])
  clearing zero.
- **Latency / output: unambiguous.** 1.77× faster per call, ~49× fewer
  completion tokens, p ≤ 1.6e-10. For a local no-per-token-price arm the payoff
  is wall-clock throughput on the eval loop.
- **One-time cost:** ~$0.045 at $0.20/GPU-h. Trivially amortised.
- **No hosted-API leg landed.** The headline "is a local model good enough to
  replace a hosted API arm?" question still needs a paid API key — see
  footnote 1 and `docs/RESULTS.md`.
- **Uncalibrated judge on this run.** No human-labeled gold subset was attached
  to run 1914, so `judge_score` here is an uncalibrated accuracy proxy. A
  separate calibration study on this task lands at
  `2026-08-31-judge-calibration-financial.md`. The judge (`qwen3:8b`) is also
  the same base model as both arms — self-preference bias is limited between
  base and fine-tune (shared parent) but not zero; see `backend/README.md`
  "Watch for judge/arm model overlap".

---

1. **Gemini API arm did not land.** The `gemini-flash` arm was rate-limited by
   Google AI Studio's free tier (~15 req/min): 130/150 calls returned HTTP 429,
   only 20 landed across 8 distinct examples, so every paired test against it is
   p = 1.0 after correction and the paired Bayesian model does not converge. A
   real local-vs-API frontier needs a paid Gemini/OpenAI/Anthropic key or a
   deliberately throttled re-run (concurrency 1, spaced calls). Separately, the
   run began under an `opus`/`claude_code_cli` judge that hit a subscription
   usage limit after ~67 calls; the judge was switched to local `qwen3:8b` and
   **all 320 completed results were re-scored from scratch** with it
   (`backend/scripts/rejudge_run.py`), so every score in this report comes from
   one judge model. A mid-run Docker Desktop outage was recovered from the
   Postgres volume; the 10 stragglers were re-enqueued (Redis has no volume and
   lost its queue — accounted for).

An earlier, thinner version of this comparison (run 1731, only 22 usable paired
quality examples) is archived at `archive/2026-08-29-finetune-comparison.md`.
