"""Render the Phase 7 fine-tuned-vs-base-vs-API comparison report from a
completed run. Reads the platform API ($PE_API_URL, default
http://localhost:8000). Run from inside backend/:

    uv run python -m scripts.finetune_report --run-id 5 \
        --baseline qwen3-8b-local \
        --candidate ft-qwen3-8b-local --candidate gpt-4o-mini \
        --finetuned ft-qwen3-8b-local --train-seconds 4500
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE_URL = os.environ.get("PE_API_URL", "http://localhost:8000").rstrip("/")
FRONTIER_PNG = "2026-08-29-finetune-frontier.png"

# Mirrors app/cli/__init__.py:_TERMINAL -- the stats endpoints only mean
# something once a run has stopped moving.
_TERMINAL = {"completed", "completed_with_errors"}


@dataclass
class ReportContext:
    run_id: int
    status: str
    arm_names: list[str]
    baseline: str
    candidates: list[str]
    finetuned: str
    epsilon: float
    summary: list[dict]
    comparisons: list[dict]
    equivalences: list[dict]
    training_cost: float
    gpu_cost_per_hour: float
    train_seconds: float
    frontier_png: str | None


def training_cost_usd(train_seconds: float, gpu_cost_per_hour: float) -> float:
    return train_seconds / 3600.0 * gpu_cost_per_hour


def break_even_calls(training_cost: float, api_cost_per_call: float) -> float | None:
    if not api_cost_per_call:
        return None
    return training_cost / api_cost_per_call


def _get(path: str, **params) -> object:
    resp = httpx.get(
        f"{BASE_URL}{path}",
        params={k: v for k, v in params.items() if v is not None},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def _summary_row(summary: list[dict], arm: str) -> dict | None:
    return next((row for row in summary if row["arm_name"] == arm), None)


def build_report_markdown(ctx: ReportContext) -> str:
    lines: list[str] = []
    lines.append("# Fine-tuned vs. base vs. API — financial sentiment (Phase 7)\n")
    lines.append(
        f"Run **{ctx.run_id}** (`{ctx.status}`) — arms `{', '.join(ctx.arm_names)}`. "
        "N per arm is the **n** column in the frontier table below; the run's "
        "`repeats` / `sample_size` / `seed` are not exposed by the API. "
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
    lines.append(
        f"P(judge_score[fine-tuned local] ≥ judge_score[API] − ε), ε = {ctx.epsilon}:\n"
    )
    if ctx.equivalences:
        lines.append("| fine-tuned local | vs. API arm | P(equivalent) |")
        lines.append("|---|---|---|")
        for e in ctx.equivalences:
            lines.append(f"| {e['arm_local']} | {e['arm_api']} | {_fmt(e['p_equivalent'])} |")
    else:
        lines.append("_No API candidate with per-call cost, or equivalence was skipped for lack of data._")
    lines.append("")

    lines.append("## Cost / latency / quality frontier\n")
    if ctx.frontier_png:
        lines.append(f"![frontier]({ctx.frontier_png})\n")
    lines.append("| arm | n | mean judge_score | mean latency (ms) | mean $/call |")
    lines.append("|---|---|---|---|---|")
    for row in ctx.summary:
        lines.append(
            f"| {row['arm_name']} | {_fmt(row.get('n'))} | {_fmt(row.get('mean_judge_score'))} | "
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
        row = _summary_row(ctx.summary, c)
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
    max_cost = max((row.get("mean_cost_estimate_usd") or 0.0 for row in summary), default=0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in summary:
        x = row.get("mean_latency_ms") or 0
        y = row.get("mean_judge_score") or 0
        cost = row.get("mean_cost_estimate_usd")
        # Marker area encodes per-call cost; null-cost (local) arms get a
        # small fixed marker (spec §7: "marker size/label = cost").
        if cost and max_cost:
            size = 40.0 + 360.0 * (cost / max_cost)
            label = f"{row['arm_name']} (${cost:.5f}/call)"
        else:
            size = 40.0
            label = f"{row['arm_name']} ($0/call)"
        ax.scatter(x, y, s=size)
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean latency (ms)")
    ax.set_ylabel("mean judge_score (1–5)")
    ax.set_title("Cost / latency / quality frontier (marker size = $/call)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True, dest="candidates")
    parser.add_argument(
        "--finetuned",
        default="ft-qwen3-8b-local",
        help="Name of the fine-tuned local arm (the arm_local side of the equivalence test).",
    )
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=0.40)
    parser.add_argument("--train-seconds", type=float, default=0.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Render the report even if the run has not reached a terminal state.",
    )
    parser.add_argument("--out", default="../docs/superpowers/reports/2026-08-29-finetune-comparison.md")
    args = parser.parse_args(argv)

    status_meta = _get(f"/runs/{args.run_id}")
    status = str(status_meta.get("status", "unknown"))
    if status not in _TERMINAL and not args.force:
        raise SystemExit(
            f"run {args.run_id} is '{status}', not a terminal state "
            f"({', '.join(sorted(_TERMINAL))}) -- pass --force to report anyway"
        )

    summary = _get(f"/runs/{args.run_id}/summary")
    if not summary:
        raise SystemExit(f"run {args.run_id} has no per-arm summary rows yet")
    arm_names = [row["arm_name"] for row in summary]
    if args.baseline not in arm_names:
        raise SystemExit(
            f"baseline arm '{args.baseline}' is not in run {args.run_id} (arms: {', '.join(arm_names)})"
        )

    comparisons = _get(f"/runs/{args.run_id}/compare", metric="judge_score")
    kept = [c for c in comparisons
            if {c["arm_a"], c["arm_b"]} <= ({args.baseline, *args.candidates})
            and args.baseline in (c["arm_a"], c["arm_b"])]

    equivalences: list[dict] = []
    if args.finetuned not in arm_names:
        print(f"skipped equivalence: fine-tuned arm '{args.finetuned}' is not in run {args.run_id}")
    else:
        for c in args.candidates:
            if c in (args.baseline, args.finetuned):
                continue
            row = _summary_row(summary, c)
            is_api = bool(row and row.get("mean_cost_estimate_usd"))
            if not is_api:
                continue
            try:
                equivalences.append(_get(
                    f"/runs/{args.run_id}/equivalence",
                    metric="judge_score", arm_local=args.finetuned, arm_api=c,
                    epsilon=args.epsilon,
                ))
            except httpx.HTTPStatusError as exc:
                # 422 from this endpoint means "not enough paired data" (or a
                # bad metric/arm, which we control) -- skip that candidate and
                # say why. Anything else is a real failure; let it propagate.
                if exc.response.status_code == 422:
                    print(f"skipped equivalence for {c}: {_detail(exc.response)}")
                    continue
                raise

    training_cost = training_cost_usd(args.train_seconds, args.gpu_cost_per_hour)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_ok = write_frontier_png(summary, out_path.parent / FRONTIER_PNG)

    ctx = ReportContext(
        run_id=args.run_id, status=status, arm_names=arm_names,
        baseline=args.baseline, candidates=args.candidates, finetuned=args.finetuned,
        epsilon=args.epsilon, summary=summary, comparisons=kept,
        equivalences=equivalences, training_cost=training_cost,
        gpu_cost_per_hour=args.gpu_cost_per_hour, train_seconds=args.train_seconds,
        frontier_png=FRONTIER_PNG if png_ok else None,
    )
    out_path.write_text(build_report_markdown(ctx), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"frontier png: {'written' if png_ok else 'skipped (matplotlib not installed)'}")


if __name__ == "__main__":
    main()
