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
