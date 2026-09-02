"""`pe` — a single entrypoint for driving the prompt-experimentation platform.

Stack lifecycle and seeding shell out to `docker compose`; run and stats
commands talk to the API over HTTP (`$PE_API_URL`, default
http://localhost:8000); calibration commands run the host scripts.
"""
import time

import httpx
import typer

from app.cli import _render
from app.cli._api import api_get, api_post, base_url
from app.cli._shell import backend_script, compose

app = typer.Typer(
    help="Drive the LLM prompt-experimentation platform.",
    no_args_is_help=True,
    add_completion=False,
)
stats_app = typer.Typer(help="Paired stats over a completed run.", no_args_is_help=True)
calibrate_app = typer.Typer(
    help="Judge calibration workflow (runs on the host; needs a local .env).",
    no_args_is_help=True,
)
finetune_app = typer.Typer(
    help="Local LoRA fine-tune workflow (runs on the host; needs a CUDA GPU "
    "and `uv sync --extra training`).",
    no_args_is_help=True,
)
app.add_typer(stats_app, name="stats")
app.add_typer(calibrate_app, name="calibrate")
app.add_typer(finetune_app, name="finetune")

_TERMINAL = {"completed", "completed_with_errors"}


# --- stack lifecycle -------------------------------------------------------


@app.command()
def up(wait: bool = typer.Option(True, help="Wait for containers to be healthy, then for the API.")):
    """Build and start the whole stack (postgres, redis, api, worker, frontend)."""
    args = ["up", "-d", "--build"]
    if wait:
        args.append("--wait")
    compose(*args)
    if wait:
        _wait_for_api()


@app.command()
def down(volumes: bool = typer.Option(False, "--volumes", "-v", help="Also drop the postgres volume.")):
    """Stop the stack."""
    compose("down", *(["-v"] if volumes else []))


@app.command()
def logs(
    service: str = typer.Argument(None, help="Limit to one service (api, worker, ...)."),
    follow: bool = typer.Option(False, "--follow", "-f"),
):
    """Tail container logs."""
    compose("logs", *(["-f"] if follow else []), *([service] if service else []))


@app.command()
def seed(
    task: str = typer.Option("financial_sentiment", "--task", help="Task pack to seed."),
):
    """Seed the eval dataset (idempotent) via a one-off migrate container."""
    compose(
        "run", "--rm", "migrate", "uv", "run", "python", "-m",
        "scripts.seed_eval_examples", "--task", task,
    )


@app.command()
def tasks():
    """List configured task packs (active marker + seeded row counts)."""
    rows = api_get("/tasks")
    _render.table(
        [
            {"name": r["name"], "active": "*" if r["active"] else "", "seeded": r["seeded_count"]}
            for r in rows
        ],
        columns=["name", "active", "seeded"],
    )


def _wait_for_api(timeout: float = 60.0) -> None:
    typer.echo("Waiting for the API ...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url()}/health", timeout=5.0).is_success:
                typer.secho("API is up.", fg=typer.colors.GREEN)
                return
        except httpx.RequestError:
            pass
        time.sleep(2.0)
    typer.secho("API did not come up in time; check `pe logs api`.", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(1)


# --- run lifecycle -------------------------------------------------------


@app.command()
def arms():
    """List the arms configured in arms.yaml."""
    _render.table(api_get("/arms"), columns=["name", "adapter", "model"])


@app.command()
def run(
    sample: int = typer.Option(None, "--sample", "-n", help="Sample size (default: whole dataset)."),
    repeats: int = typer.Option(1, "--repeats", "-r", help="Repeats per example per arm."),
    seed: int = typer.Option(None, "--seed", help="RNG seed for the sample."),
    arm: list[str] = typer.Option(None, "--arm", "-a", help="Arm name; repeatable (default: all arms except subscription-CLI)."),
    task: str = typer.Option(None, "--task", help="Task pack to run against (default: the active task in arms.yaml)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print only the new run id."),
):
    """Start a run."""
    body = {"repeats": repeats}
    if sample is not None:
        body["sample_size"] = sample
    if seed is not None:
        body["seed"] = seed
    if arm:
        body["arms"] = arm
    if task is not None:
        body["task"] = task
    result = api_post("/runs", body)
    if quiet:
        typer.echo(result["run_id"])
    else:
        _render.kv(result)


@app.command()
def status(run_id: int):
    """Show the current status of a run."""
    _render.kv(api_get(f"/runs/{run_id}"))


@app.command()
def watch(run_id: int, interval: float = typer.Option(3.0, "--interval", help="Poll interval, seconds.")):
    """Poll a run until it finishes; exit non-zero if any call failed."""
    while True:
        state = api_get(f"/runs/{run_id}")
        done = state["completed"] + state["failed"]
        typer.echo(
            f"\r{state['status']:<22} {done}/{state['total_calls']} "
            f"(failed {state['failed']})",
            nl=False,
        )
        if state["status"] in _TERMINAL:
            typer.echo()
            raise typer.Exit(1 if state["failed"] else 0)
        time.sleep(interval)


@app.command()
def results(
    run_id: int,
    limit: int = typer.Option(100, "--limit"),
    offset: int = typer.Option(0, "--offset"),
):
    """Show per-call result rows for a run."""
    rows = api_get(f"/runs/{run_id}/results", limit=limit, offset=offset)
    _render.table(
        rows,
        columns=["id", "arm_name", "status", "judge_score", "latency_ms", "cost_estimate_usd"],
    )


# --- stats --------------------------------------------------------------


@stats_app.command("compare")
def stats_compare(run_id: int, metric: str = typer.Option("judge_score", "--metric", "-m")):
    """Pairwise paired comparison across every arm pair."""
    rows = api_get(f"/runs/{run_id}/compare", metric=metric)
    _render.table(
        rows,
        columns=["arm_a", "arm_b", "n_examples", "mean_diff", "ci_lower", "ci_upper", "p_value_corrected"],
    )


@stats_app.command("equivalence")
def stats_equivalence(
    run_id: int,
    local: str = typer.Option(..., "--local", help="Local arm name."),
    api: str = typer.Option(..., "--api", help="Reference (API) arm name."),
    metric: str = typer.Option("judge_score", "--metric", "-m"),
    eps: float = typer.Option(None, "--eps", help="Equivalence margin."),
):
    """Bayesian P(local >= api - eps) for judge_score."""
    _render.kv(
        api_get(
            f"/runs/{run_id}/equivalence",
            arm_local=local,
            arm_api=api,
            metric=metric,
            epsilon=eps,
        )
    )


@stats_app.command("power")
def stats_power(
    run_id: int,
    arm_a: str = typer.Option(..., "--arm-a"),
    arm_b: str = typer.Option(..., "--arm-b"),
    metric: str = typer.Option("judge_score", "--metric", "-m"),
    power: float = typer.Option(0.8, "--power"),
    alpha: float = typer.Option(0.05, "--alpha"),
):
    """Required sample size to detect the observed effect at the target power."""
    _render.kv(
        api_get(
            f"/runs/{run_id}/power",
            arm_a=arm_a,
            arm_b=arm_b,
            metric=metric,
            power=power,
            alpha=alpha,
        )
    )


# --- calibration ------------------------------------------------------


@calibrate_app.command("select")
def calibrate_select(
    run_id: int = typer.Option(..., "--run-id"),
    out: str = typer.Option(..., "--out", help="Path for the sample JSON to hand-label."),
    n: int = typer.Option(40, "--n"),
    seed: int = typer.Option(None, "--seed"),
):
    """Write a stratified sample of judged results for human labeling."""
    extra = ["--seed", str(seed)] if seed is not None else []
    backend_script(
        "scripts.select_calibration_sample",
        "--run-id", str(run_id), "--n", str(n), "--out", out, *extra,
    )


@calibrate_app.command("import")
def calibrate_import(
    in_path: str = typer.Option(..., "--in", help="The labeled sample JSON."),
    labeled_by: str = typer.Option(..., "--labeled-by"),
):
    """Import hand-entered human_score values from the sample JSON."""
    backend_script(
        "scripts.import_calibration_labels", "--in", in_path, "--labeled-by", labeled_by
    )


@calibrate_app.command("report")
def calibrate_report(run_id: int = typer.Option(..., "--run-id")):
    """Print judge/human agreement (Spearman, Cohen's kappa) for a run."""
    backend_script("scripts.calibration_report", "--run-id", str(run_id))


# --- finetune -----------------------------------------------


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
    finetuned: str = typer.Option(
        "ft-qwen3-8b-local", "--finetuned",
        help="Name of the fine-tuned local arm (the local side of the equivalence test).",
    ),
    epsilon: float = typer.Option(0.5, "--epsilon", help="Equivalence margin on the 1-5 judge scale."),
    gpu_cost_per_hour: float = typer.Option(0.40, "--gpu-cost-per-hour"),
    train_seconds: float = typer.Option(0.0, "--train-seconds", help="Wall time of `pe finetune train`."),
    force: bool = typer.Option(
        False, "--force", help="Render even if the run has not reached a terminal state."
    ),
    out: str = typer.Option(
        "../docs/reports/2026-08-30-finetune-comparison.md", "--out"
    ),
):
    """Render the fine-tuned-vs-base-vs-API comparison report from a completed run."""
    args = ["--run-id", str(run_id), "--baseline", baseline, "--finetuned", finetuned,
            "--epsilon", str(epsilon), "--gpu-cost-per-hour", str(gpu_cost_per_hour),
            "--train-seconds", str(train_seconds), "--out", out]
    if force:
        args.append("--force")
    for c in candidate:
        args += ["--candidate", c]
    backend_script("scripts.finetune_report", *args)
