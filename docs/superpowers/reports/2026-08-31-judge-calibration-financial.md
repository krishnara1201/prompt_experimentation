# Judge calibration — financial sentiment, free-text model outputs (Task A)

**Question.** The AG News calibration (run 559, κ = 1.00) was a near-trivial
4-way label match. Does the LLM judge still agree with a human when the
model's output is **free-text prose** ("The sentiment is negative because
…"), where the judge has to *read* a rationale and decide whether the stated
sentiment matches the gold label — not just string-match a token?

**Setup.** Run **707** — a single arm, `qwen3-fin-explain` (`qwen3:8b`,
`reasoning_effort: none`), prompted to *"write 1–2 sentences explaining your
reasoning, and state the sentiment in that explanation. Do not answer with
just the label."* 60 Financial PhraseBank examples × 2 repeats = 120 calls.
Every call was judged by the local `qwen3:8b` judge against the task rubric
(1–5). A 50-row stratified subset was then hand-labeled by the report author
**blind to the judge's score and rationale** (the `human_score` column was
filled from `model_output` + `gold_label` + the rubric only), and imported
via `pe calibrate import` / `report`.

> Author-labeled, not an independent panel. Labels were assigned against the
> Financial PhraseBank expert labels using the pack's rubric
> (`backend/tasks/financial_sentiment/task.yaml`); this measures
> judge-vs-careful-reader agreement, not judge-vs-consensus.

## Agreement

| n | Spearman r (1–5 score) | Cohen's κ (score ≥ 4 = "label correct") | mean \|Δ\| |
|---|---|---|---|
| 50 | **0.729** (p ≈ 1e-4) | **1.000** | **0.060** |

- **On the decision that matters — is the label right? — judge and human
  agree on all 50/50 rows** (κ = 1.00). Every response the human read as
  correctly-labeled, the judge scored ≥ 4; both wrong-label responses in the
  sample, the judge scored 2.
- **On the exact 1–5 score, agreement is only moderate (Spearman 0.729)**,
  and that gap is entirely one effect: the judge emits a **2-value
  vocabulary** — 5 (label right) or 2 (label wrong) — and *never* uses 3 or
  4. Across the full 120-call run: **113 fives, 7 twos, zero 1s/3s/4s.** The
  human, using the full rubric, assigned 3 fours. Those 3 rows are the only
  disagreements; mean absolute difference is 0.06.

## The one disagreement pattern: judge is lenient on minor format deviations

All 3 judge≠human rows are the **same failure mode**. The model produced
correct, well-reasoned prose but then tacked a redundant `   Sentiment:
<Label>` line onto the end — a partial slip back toward the "just the label"
format the prompt told it to avoid. The rubric's **4** is *"correctly
identifies the sentiment, but with minor clarity/formatting issues"*; the
human scored these **4**, the judge scored them **5** and its rationale
explicitly praised them as *"clear and direct"*.

| run_result | input (gold = neutral) | model output (trailing append **bolded**) | human | judge |
|---|---|---|---|---|
| 4199 | *"Its annual capacity is some 10,000 MW."* | "…without expressing any opinion or emotional tone. There is no indication of positive or negative sentiment… **Sentiment: Neutral**" | 4 | 5 |
| 4290 | *"BAVARIA Industriekapital AG's 2006 revenues were EUR 333 million, with an EBITDA of EUR 51 million."* | "…It simply states the revenue and EBITDA figures for a specific year, which are neutral in tone… **Sentiment: Neutral**" | 4 | 5 |
| 4495 | *"Cramo, headquartered in Vantaa, Finland, rents construction machinery and equipment…"* | "…No indication of success, failure, optimism, or pessimism regarding the company's financial status. **Sentiment: Neutral**" | 4 | 5 |

Judge rationale for 4199, verbatim: *"The model accurately identifies the
neutral sentiment, providing a clear and direct explanation of why the
statement is factual and devoid of emotional tone."* — it did not register
the format inconsistency at all.

**Patterns that did *not* appear** (checked for, absent in this run):

- *Judge harsher than human on a correct-but-verbose answer* — none. The
  judge scored every correct, on-topic rationale a 5 regardless of length.
- *Judge distracted by the rationale text into the wrong label* — none. In
  all 50 rows the judge's score tracked the sentiment the model actually
  stated, not stray words in the explanation.
- *Judge lenient on a hedged output* — n/a. `qwen3:8b` never actually
  hedged on this task; every output committed to one label.

## An observation outside the calibrated sample

Two of the 7 wrong-scored rows on the full run were **non-responsive** — the
model got a truncated input (`"NWC ANALYSIS :"`) and correctly said the
sentiment *"cannot be determined."* Gold label is `neutral`, so the judge
scored these **2** ("wrong sentiment, otherwise coherent"). A strict human
reading the rubric could argue **1** ("non-responsive"). These rows were not
in the hand-labeled 50, so this is flagged as an observation, not a
calibrated disagreement — but it suggests the judge's binary collapse also
swallows the 1-vs-2 distinction, not just 4-vs-5.

## Honest read

- **The judge is trustworthy as a label-correctness classifier on this
  task**, even on free-text outputs: κ = 1.00 vs. a careful human on 50 rows,
  zero label disagreements.
- **It is not a calibrated 1–5 scorer.** It uses two of the five rubric
  points. If a future task needs the judge to *grade quality* (partial
  credit, hedging, verbosity) rather than *check a label*, this judge would
  need a stronger model and a re-run of this workflow — the rubric's
  gradations are currently dead letters.
- **Both the AG News run and this one land at κ = 1.00 on the binary
  question.** For classification tasks where the model emits a defensible
  label, the LLM judge adds essentially no disagreement with ground truth —
  so `judge_score ≥ 4` is a sound accuracy proxy for the paired stats. The
  value of calibration here is the *negative* result: it rules out judge
  noise as a confound in the fine-tune and prompt-A/B comparisons.
- **Judge / arm model overlap still applies** — the judge (`qwen3:8b`) is
  the arm. For a label-match check on an easy task this is close to
  harmless; it would matter for a model-vs-model quality comparison. See
  `backend/README.md` "Watch for judge/arm model overlap".

## Reproduce

```bash
cd backend
# run 707 already exists; to redo from scratch:
uv run pe run -a qwen3-fin-explain -n 60 -r 2 --seed 7 -q
# wait for arm calls + judge (or judge serially if the box is RAM-tight):
uv run python -m scripts.serial_judge_run <run_id> --only-unjudged
uv run pe calibrate select --run-id <run_id> --n 50 --seed 5 --out /tmp/cal.json
# hand-fill each row's human_score in /tmp/cal.json, then:
uv run pe calibrate import --in /tmp/cal.json --labeled-by "you"
uv run pe calibrate report --run-id <run_id>
```
