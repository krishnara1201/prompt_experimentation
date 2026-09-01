# Local vs. subscription-CLI hosted — Qwen3-8B vs. Claude Sonnet on financial sentiment

Run **1058** (`completed`) — arms `qwen3-8b-local` (Ollama, Qwen3-8B, native
thinking on) and `claude-code-sonnet` (the authenticated `claude` CLI under a
Max seat, `adapter: claude_code_cli`), **same 150 Financial PhraseBank
all-agree sentences**, 1 repeat = 150 calls per arm, **300 total, 0
model-call failures**. Both arms use the task's default eval prompt — a fair
paired prompt.

This is the run the project was built to produce: `CLAUDE.md` differentiator
#4 (a local-vs-hosted cost/latency/quality frontier) and the Phase 7
deliverable both needed a hosted arm that actually completes a paired run.
The metered `gemini-flash` attempt never did (free-tier 429s — see
`docs/RESULTS.md`). The subscription-CLI arm closes that gap at **no
per-token cost**: `cost_estimate_usd` is `null` for both arms — local
compute on one side, a flat-rate seat on the other.

> **How it ran.** `backend/scripts/serial_eval_run.py` — an in-process
> runner (no Celery/Redis), one call at a time, that **pauses cleanly when
> the Claude seat hits its usage limit and resumes after the window
> resets**. The 150 CLI calls were run in 5 batches of 30 (`--max-cli-calls
> 30`) to leave the shared seat headroom; the seat never actually limited.
> Judge: local `qwen3:8b` via `serial_judge_run.py`.

## Quality (paired)

| comparison | mean Δ `judge_score` | 95% CI | Wilcoxon W | corrected p |
|---|---|---|---|---|
| `qwen3-8b-local` − `claude-code-sonnet` | **−0.147** | [−0.32, +0.02] | 66.0 | **0.104** |

Paired hierarchical bootstrap + Wilcoxon signed-rank over the 150 examples,
Holm-corrected. **No statistically significant quality difference.** The
point estimate favours Claude by 0.15 on the 1–5 scale; the CI includes
zero.

| arm | correct / n | accuracy |
|---|---|---|
| `qwen3-8b-local` | 131 / 150 | **87.3%** |
| `claude-code-sonnet` | 138 / 150 | **92.0%** |

Binarised as `judge_score ≥ 4` = "label correct". The judge emitted almost
only **5** (correct) or **2** (wrong), with a single **1** — as in every
prior financial-sentiment run, `judge_score` here is effectively a binary
accuracy proxy, not a calibrated 1–5 score.

Discordant examples (one arm right, the other wrong): **Claude right / Qwen
wrong = 13**, **Qwen right / Claude wrong = 6**. 13 vs 6 is not significant
(binomial p ≈ 0.17) — consistent with the Wilcoxon result. Both arms wrong
on the same 6 examples.

## Bayesian equivalence

`metric=judge_score`, ε = 0.5 on the 1–5 scale, `qwen3-8b-local` as the
"local" side:

| | posterior mean Δ (qwen − claude) | 94% CI | P(qwen ≥ claude − ε) |
|---|---|---|---|
| `qwen3-8b-local` vs. `claude-code-sonnet` | **−0.147** | [−0.32, +0.03] | **1.00** |

The entire posterior sits well inside ε = 0.5. **For a half-rubric-point
margin the local model is "good enough" with probability 1.00.** At a
tighter ε = 0.2 the answer weakens (posterior mass near the −0.32 tail), so
the honest statement is: Qwen3-8B matches hosted Sonnet here *to within
about a fifth of the judge's binary step* — i.e. within a handful of
label-accuracy points — but a real ~5-point accuracy gap in Claude's favour
is not ruled out.

## Power

| observed effect (SD of paired diff) | achieved power @ n=150 | n for 80% power |
|---|---|---|
| −0.138 | **0.39** | **414** |

The "not significant" result is **underpowered**: at n=150 the study has
only a 39% chance of detecting an effect the size of the one observed. It
would take ~414 examples to confirm or rule out a −0.15 difference. So this
run establishes *practical* equivalence (the effect is small — bounded well
inside ε = 0.5) but not *statistical* equivalence at a tight margin.

## Latency

| arm | median | mean | max | notes |
|---|---|---|---|---|
| `qwen3-8b-local` | **3.8 s** | 88 s | 3.4 h (!) | native thinking on; ~282 completion tokens/call |
| `claude-code-sonnet` | **2.4 s** | 2.4 s | 4.5 s | ~37 completion tokens/call |

**Typical latency is comparable** (local 3.8 s vs hosted 2.4 s median). The
story is the **tail**: a handful of local calls stalled for minutes to
hours (max 3.4 h; mean 88 s is entirely outlier-driven) when Ollama came
under memory pressure on this 7.8 GB WSL box — the same stalls that forced a
judge re-run. The hosted arm's latency is tight (1.8–4.5 s, SD 0.5 s) with
no tail. This is a real reliability difference on constrained local
hardware, not just measurement noise — but it *is* hardware-specific: a
box that fits Qwen3-8B comfortably would not show it. The in-process runner
issues one call at a time, so unlike run 559 there is **no queue
contention** confounding the measurement.

## Cost

Both `null`. `qwen3-8b-local` is local compute (electricity + a GPU/CPU you
already own). `claude-code-sonnet` is a **flat-rate subscription seat** — no
per-call price, which is why the adapter refuses to invent a `$0` or an
amortised number. The CLI's own `usage` block reports only the user-turn
tokens (`prompt_tokens ≈ 2`), not its ~25k-token agent system prompt, so
those counts are informational, not a cost basis. On the frontier plot these
two arms belong to a **third category** alongside metered API arms — "no
per-token bill", accounted separately.

## Caveats

- **Judge is `qwen3:8b`** — the same base model as the `qwen3-8b-local` arm.
  LLM-as-judge self-preference would, if anything, **understate** Claude's
  lead, so the true quality gap may be a little wider than measured. See
  `backend/README.md` "Watch for judge/arm model overlap".
- **No human-labeled gold subset** was attached to run 1058, so
  `judge_score` is an uncalibrated accuracy proxy. Prior financial-sentiment
  calibration (run 707, `2026-08-31-judge-calibration-financial.md`) found
  the `qwen3:8b` judge agrees with human labels on *label-correctness*
  (κ = 1.00, n = 50) but is a 2-value scorer, not a calibrated 1–5 rater —
  which is exactly how it behaves here.
- **Underpowered** for a tight-margin equivalence claim (see Power).
- Latency tail is hardware-specific (7.8 GB WSL box, Ollama under memory
  pressure).

## Bottom line

On financial sentiment, **local Qwen3-8B is statistically indistinguishable
from hosted Claude Sonnet on quality** (paired Wilcoxon p = 0.10; Bayesian
P(within ±0.5) = 1.00), trailing by ~5 points of raw label-accuracy (87% vs
92%) that this sample size cannot confirm as real. Neither arm carries a
per-token bill. The hosted arm's advantage is **latency reliability** — no
multi-minute tail — not headline quality. For a batch financial-sentiment
workload where a slow call now and then is tolerable, the local model is a
defensible replacement; where p99 latency matters, it is not.
