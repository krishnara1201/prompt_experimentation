# Stats Layer — Design Spec

Date: 2026-08-27
Status: Approved for implementation
Scope: Build phase 4 of the platform described in `CLAUDE.md` — paired
significance testing, a Bayesian equivalence test, multiple-comparison
correction, and a sample-size/power calculator, exposed via a new API
router alongside the existing `runs` endpoints.

## Context

Phases 1-3 (model adapters, Celery orchestration, judge layer + calibration)
are done. `RunResult` rows carry `judge_score` (1-5 ordinal, judge-assigned),
`latency_ms`, `cost_estimate_usd`, `prompt_tokens`/`completion_tokens` per
`(run_id, example_id, arm_name, repeat_index)`. Nothing currently compares
arms statistically — this phase adds that layer.

The sibling project `experimentation_copilot` has a stats module
(`backend/app/stats/stat_analysis.py`) that this project's tech stack
section points to for conventions, but it is purely frequentist and built
around independent two-sample summary statistics (`p1, p2, n1, n2` or
`mean1, mean2, std1, std2, n1, n2`) — the opposite of differentiator #1
("paired comparisons, not independent samples"). It is **not** reused for
the actual test logic; only its `alpha`/`test_type` enum and CI-tuple
conventions carry over, adapted to a paired, array-based signature.

No frontend exists yet (Phase 5 is not started). This phase's endpoints are
the contract Phase 5's dashboard will consume, decided now rather than
guessed at later, per this project's precedent of designing each interface
against what actually exists.

## Architecture

```
[RunResult rows, status=completed] -> [aggregation.py: group by (example_id, arm_name) -> repeat lists]
                                    -> [paired_tests.py: hierarchical bootstrap CI + Wilcoxon + Holm-Bonferroni]
                                    -> [bayesian.py: PyMC paired-diff model -> P(mu >= -epsilon)]
                                    -> [power.py: pilot variance -> required n / achieved power]
                                    -> [api/routes/stats.py: GET /runs/{id}/compare | /equivalence | /power]
```

### Data eligibility

An `(example_id, arm_name)` cell is eligible for a repeat list if it has at
least one `RunResult` with `status == "completed"`. For `metric ==
"judge_score"` specifically, a repeat additionally requires `judge_status
== "completed"` (a completed generation with a still-pending or failed judge
score contributes to latency/cost stats but not quality stats). An example
is eligible for a *pairwise comparison* only if both arms in the pair have
at least one eligible repeat for that example; excluded examples are
counted and reported as `n_excluded`, never silently dropped from the
response.

### Module layout

```
backend/app/stats/
  __init__.py
  aggregation.py   — RunResult rows -> {(example_id, arm_name): [repeat values]} for a given metric
  paired_tests.py  — hierarchical paired bootstrap CI, Wilcoxon signed-rank, Holm-Bonferroni correction
  bayesian.py      — PyMC paired-difference model, region-of-practical-equivalence posterior
  power.py         — sample-size / achieved-power estimate from pilot (run) variance
backend/app/api/routes/stats.py   — new router, mounted in app/main.py next to runs.py
```

### `aggregation.py`

`load_metric_by_example(session, run_id, metric, arm_names) -> dict[(example_id, arm_name), list[float]]`

Generic over `metric` (any of `judge_score`, `latency_ms`,
`cost_estimate_usd`, or any other numeric `RunResult` column) — one function
serves quality, latency, and cost comparisons rather than a judge_score-
specific path. Applies the eligibility rule above, queries only the arms
requested, and returns repeat lists (not pre-averaged), since the repeat
list is what the hierarchical bootstrap needs.

### `paired_tests.py`

`compare_pair(repeats_by_cell, arm_a, arm_b, examples) -> PairedComparisonResult`

For one arm pair:

- **Point estimate**: mean of per-example paired differences. Each
  example's per-arm "value" used for the point estimate is the mean of its
  own repeat list (repeat-averaged), giving a single stable number to
  report alongside the CI.
- **Hierarchical paired bootstrap CI** (default `B = 10,000` replicates,
  95% CI, overridable): each replicate (1) resamples examples with
  replacement, (2) for each resampled example resamples one repeat value
  per arm with replacement from that example's own repeat list, (3) takes
  the mean of the resulting paired diffs. The 2.5th/97.5th percentiles of
  the `B` replicate means form the CI. This is what carries within-arm
  (across-repeat) variance into the result, per differentiator #5 — a flat
  average-then-bootstrap would discard it.
- **Wilcoxon signed-rank**: `scipy.stats.wilcoxon` on the per-example
  paired diffs computed from repeat-averaged values. Wilcoxon has no
  natural hierarchical-resampling variant, so this stays at the
  repeat-averaged level; the bootstrap CI above is the mechanism that
  reflects repeat-level noise, and this simplification is intentional, not
  an oversight.
- Raises (surfaced as a 422 from the API layer) if fewer than 5 eligible
  paired examples exist for a pair — Wilcoxon and a bootstrap CI are not
  meaningful below that, and a silently-computed CI on 2 examples would be
  misleading.

`correct_pairwise_pvalues(results: list[PairedComparisonResult]) -> list[PairedComparisonResult]`

Holm-Bonferroni step-down correction across a family of pairwise
comparisons (sort p-values ascending, compare `p_(i)` to `alpha /
(m - i + 1)`, propagate rejection down). Implemented directly from sorted
p-values — no new dependency. Applied whenever `/compare` returns more than
one pair in a single response; a single explicit pair returns its raw
p-value uncorrected (nothing to correct against).

### `bayesian.py`

`equivalence_probability(diffs: list[float], epsilon: float) -> EquivalenceResult`

- Input `diffs` are per-example, repeat-averaged paired differences
  (`value_local_i - value_api_i`) — same repeat-averaging rationale as
  Wilcoxon; a full repeat-level hierarchical Bayesian model is possible but
  is meaningfully more complex for a single-parameter paired-difference
  question, so this spec deliberately stays at the per-example level.
- Model: `d_i ~ Normal(mu, sigma)`, with weakly-informative priors scaled
  off the data itself rather than a hardcoded per-metric range — this is
  what keeps the function generic across `judge_score` (bounded, diffs in
  `[-4, 4]`), `latency_ms`, and `cost_estimate_usd` (both unbounded) without
  per-metric special-casing. Concretely: `scale = max(std(diffs), 1e-6)`
  (the floor guards the degenerate all-identical-diffs case), then
  `mu ~ Normal(0, 10 * scale)`, `sigma ~ HalfNormal(10 * scale)` — a prior
  an order of magnitude wider than the data's own spread, weak enough to be
  dominated by the likelihood on any real sample. Sampled with PyMC's
  default NUTS sampler; short chains are adequate for a single parameter of
  interest.
- Output: `P(mu >= -epsilon)` from the posterior draws over `mu`, plus the
  posterior mean and a 95% credible interval on `mu`. This is literally
  "is local at least as good as API within margin epsilon" from
  differentiator #3.
- `epsilon` has no default anywhere in the stack — it is a judgment call
  about "good enough," not a statistical default, and the API layer
  rejects a request that omits it.

### `power.py`

`estimate_sample_size(pilot_diffs: list[float], effect_size: float | None, power: float, alpha: float) -> PowerResult`

- Computes pilot mean/std of paired diffs (repeat-averaged per example,
  same convention as above) from an existing run, used as the variance
  estimate.
- `effect_size` (`delta`) defaults to the pilot's own observed mean diff
  when not given.
- Required `n` via the standard paired normal-approximation formula:
  `n = ((z_{alpha/2} + z_power)^2 * sigma_d^2) / delta^2`, using
  `scipy.stats.norm.ppf` for the z-quantiles. No new dependency.
- Also reports **achieved power at the run's actual n** (solving the same
  relationship for power given the observed `n`), so a completed run can be
  checked retrospectively, not just used to plan a future one.

### API surface (`backend/app/api/routes/stats.py`)

- `GET /runs/{run_id}/compare?metric=&arm_a=&arm_b=&bootstrap_samples=`
  Frequentist paired bootstrap CI + Wilcoxon. `arm_a`/`arm_b` optional; if
  either is omitted, returns all pairwise comparisons among the run's arms
  for that metric, with Holm-Bonferroni correction applied across the
  returned family. `bootstrap_samples` optional override of `B` (default
  10,000).
- `GET /runs/{run_id}/equivalence?metric=&arm_local=&arm_api=&epsilon=`
  Bayesian equivalence for one specific directional pair. All four params
  required — no defaulted arm pair or epsilon.
- `GET /runs/{run_id}/power?metric=&arm_a=&arm_b=&power=&alpha=&effect_size=`
  Sample-size/power estimate using the run as pilot data. `power` defaults
  to 0.8, `alpha` to 0.05, `effect_size` to the pilot's observed mean diff.

All three routes 404 if `run_id` doesn't exist, and 422 if the requested
metric/arm combination has fewer than 5 eligible paired examples (see
`paired_tests.py` above) — mirrors the existing 400/404 conventions in
`runs.py`.

### New dependency

`pymc` (pulls in `pytensor` transitively). Everything else (bootstrap,
Wilcoxon, Holm-Bonferroni, the power formula) stays on `scipy`/`numpy`,
already present.

### Testing

Synthetic `RunResult` fixtures (small in-memory sets, no need for a live
Postgres per test):

- Bootstrap CI recovers the known mean/spread of a synthetic paired
  distribution.
- Wilcoxon output cross-checked against a direct `scipy.stats.wilcoxon`
  call on the same repeat-averaged data.
- Holm-Bonferroni correction cross-checked against a hand-worked example
  with a known set of p-values.
- PyMC equivalence sanity checks: identical arms -> `P(mu >= -epsilon)` ~
  1; disjoint/opposite arms -> ~ 0. Kept fast with few chains/draws — these
  are correctness sanity checks, not posterior-quality benchmarks.
- Power formula cross-checked against a closed-form textbook sample-size
  example.
- API-level tests for `/compare`, `/equivalence`, `/power`: 404 on missing
  run, 422 on too-few-paired-examples, correction only applied when more
  than one pair is returned.

## Non-goals (this phase)

- No dashboard/frontend consumption of these endpoints — that's Phase 5,
  designed against this contract once it exists.
- No repeat-level (fully hierarchical) Bayesian model — documented above as
  a deliberate simplification, not a gap to fill later in this phase.
- No FDR/Benjamini-Hochberg alternative to Holm-Bonferroni — Holm's
  family-wise control is the standard default for a small number of arm
  pairs, and this project isn't running enough simultaneous comparisons to
  need FDR's looser control for power.
