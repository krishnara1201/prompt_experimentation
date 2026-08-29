"""Contract tests for scripts/finetune_report.py.

Every respx mock is built from the REAL API Pydantic models
(RunStatusResponse, ArmSummaryResponse, PairedComparisonResponse,
EquivalenceResponse) so the script can't drift from the endpoints again,
and the outgoing requests are asserted to carry the required query params.
"""
import httpx
import pytest
import respx

from app.api.routes.runs import RunStatusResponse
from app.api.routes.stats import (
    ArmSummaryResponse,
    EquivalenceResponse,
    PairedComparisonResponse,
)
from scripts import finetune_report as fr

BASE = "http://localhost:8000"


# --- model-backed mock builders ---------------------------------------


def run_status(run_id=5, status="completed", total=20):
    return RunStatusResponse(
        run_id=run_id, status=status, total_calls=total,
        completed=total, failed=0, pending=0,
    ).model_dump(mode="json")


def arm_summary(arm_name, n=20, judge=3.5, latency=800.0, cost=None):
    return ArmSummaryResponse(
        arm_name=arm_name, n=n, mean_judge_score=judge, mean_latency_ms=latency,
        mean_cost_estimate_usd=cost, mean_prompt_tokens=42.0, mean_completion_tokens=3.0,
    ).model_dump(mode="json")


def comparison(arm_a, arm_b, mean_diff=1.0, p_corr=0.01):
    return PairedComparisonResponse(
        arm_a=arm_a, arm_b=arm_b, metric="judge_score", n_examples=20, n_excluded=0,
        mean_diff=mean_diff, ci_lower=mean_diff - 0.3, ci_upper=mean_diff + 0.3,
        wilcoxon_statistic=10.0, p_value=p_corr, p_value_corrected=p_corr,
    ).model_dump(mode="json")


def equivalence(arm_local, arm_api, p_equivalent=0.9, epsilon=0.5):
    return EquivalenceResponse(
        arm_local=arm_local, arm_api=arm_api, metric="judge_score", epsilon=epsilon,
        n_examples=20, n_excluded=0, posterior_mean=-0.1, ci_lower=-0.4, ci_upper=0.2,
        p_equivalent=p_equivalent,
    ).model_dump(mode="json")


# --- pure functions --------------------------------------------------


def test_training_cost_usd():
    assert fr.training_cost_usd(3600, 0.40) == pytest.approx(0.40)
    assert fr.training_cost_usd(1800, 1.0) == pytest.approx(0.50)


def test_break_even_calls():
    assert fr.break_even_calls(0.50, 0.001) == pytest.approx(500)
    assert fr.break_even_calls(0.50, 0.0) is None


def test_build_report_markdown_has_all_sections():
    ctx = fr.ReportContext(
        run_id=5,
        status="completed",
        arm_names=["qwen3-8b-local", "ft-qwen3-8b-local", "gpt-4o-mini"],
        baseline="qwen3-8b-local",
        candidates=["ft-qwen3-8b-local", "gpt-4o-mini"],
        finetuned="ft-qwen3-8b-local",
        epsilon=0.5,
        summary=[
            {"arm_name": "qwen3-8b-local", "n": 20, "mean_judge_score": 3.1,
             "mean_latency_ms": 900, "mean_cost_estimate_usd": None},
            {"arm_name": "ft-qwen3-8b-local", "n": 20, "mean_judge_score": 4.2,
             "mean_latency_ms": 850, "mean_cost_estimate_usd": None},
            {"arm_name": "gpt-4o-mini", "n": 20, "mean_judge_score": 4.4,
             "mean_latency_ms": 700, "mean_cost_estimate_usd": 0.0004},
        ],
        comparisons=[
            {"arm_a": "ft-qwen3-8b-local", "arm_b": "qwen3-8b-local", "mean_diff": 1.1,
             "ci_lower": 0.8, "ci_upper": 1.4, "p_value_corrected": 0.001},
        ],
        equivalences=[
            {"arm_local": "ft-qwen3-8b-local", "arm_api": "gpt-4o-mini",
             "epsilon": 0.5, "p_equivalent": 0.92},
        ],
        training_cost=0.50,
        gpu_cost_per_hour=0.40,
        train_seconds=4500,
        frontier_png=None,
    )
    md = fr.build_report_markdown(ctx)
    assert "# Fine-tuned vs. base vs. API" in md
    assert "(`completed`)" in md  # run status surfaced
    assert "not exposed by the API" in md  # truthful about repeats/sample_size/seed
    assert "Financial PhraseBank" in md and "CC BY-NC-SA" in md
    assert "Win-rate / quality" in md
    assert "Bayesian equivalence" in md
    assert "frontier" in md.lower()
    assert "Training-cost accounting" in md
    assert "0.50" in md  # one-time training cost surfaced
    assert "judge" in md.lower()


def test_frontier_image_line_only_when_png_written():
    base = dict(
        run_id=5, status="completed", arm_names=["a"], baseline="a", candidates=["a"],
        finetuned="ft-qwen3-8b-local", epsilon=0.5,
        summary=[{"arm_name": "a", "n": 5, "mean_judge_score": 3.0,
                  "mean_latency_ms": 100, "mean_cost_estimate_usd": None}],
        comparisons=[], equivalences=[], training_cost=0.0, gpu_cost_per_hour=0.4,
        train_seconds=0.0,
    )
    assert "![frontier]" not in fr.build_report_markdown(fr.ReportContext(**base, frontier_png=None))
    assert "![frontier](f.png)" in fr.build_report_markdown(fr.ReportContext(**base, frontier_png="f.png"))


# --- endpoint contract ---------------------------------------------


@respx.mock
def test_render_writes_file_and_sends_metric(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(200, json=run_status()))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        arm_summary("qwen3-8b-local"), arm_summary("ft-qwen3-8b-local", judge=4.0)]))
    compare_route = respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        comparison("ft-qwen3-8b-local", "qwen3-8b-local")]))

    out = tmp_path / "report.md"
    fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
             "--candidate", "ft-qwen3-8b-local", "--out", str(out),
             "--train-seconds", "3600", "--gpu-cost-per-hour", "0.4"])

    assert out.exists()
    assert "ft-qwen3-8b-local" in out.read_text()
    # the /compare endpoint has a REQUIRED metric param (no default) -- a
    # missing one would 422 the whole report.
    assert compare_route.calls.last.request.url.params["metric"] == "judge_score"


@respx.mock
def test_equivalence_call_carries_metric_and_finetuned_arm(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(200, json=run_status()))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        arm_summary("qwen3-8b-local"),
        arm_summary("ft-local", judge=4.0),
        arm_summary("gpt-4o-mini", judge=4.3, cost=0.0004)]))
    respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        comparison("ft-local", "qwen3-8b-local")]))
    eq_route = respx.get(f"{BASE}/runs/5/equivalence").mock(return_value=httpx.Response(
        200, json=equivalence("ft-local", "gpt-4o-mini")))

    out = tmp_path / "r.md"
    fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
             "--candidate", "ft-local", "--candidate", "gpt-4o-mini",
             "--finetuned", "ft-local", "--out", str(out)])

    params = eq_route.calls.last.request.url.params
    assert params["metric"] == "judge_score"
    assert params["arm_local"] == "ft-local"
    assert params["arm_api"] == "gpt-4o-mini"
    assert "P(equivalent)" in out.read_text()


@respx.mock
def test_equivalence_422_is_skipped_with_reason(tmp_path, capsys):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(200, json=run_status()))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        arm_summary("qwen3-8b-local"),
        arm_summary("ft-local", judge=4.0),
        arm_summary("gpt-4o-mini", judge=4.3, cost=0.0004)]))
    respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        comparison("ft-local", "qwen3-8b-local")]))
    respx.get(f"{BASE}/runs/5/equivalence").mock(return_value=httpx.Response(
        422, json={"detail": "need at least 8 paired examples, got 3"}))

    out = tmp_path / "r.md"
    fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
             "--candidate", "ft-local", "--candidate", "gpt-4o-mini",
             "--finetuned", "ft-local", "--out", str(out)])

    printed = capsys.readouterr().out
    assert "skipped equivalence for gpt-4o-mini" in printed
    assert "8 paired examples" in printed


@respx.mock
def test_equivalence_non_4xx_propagates(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(200, json=run_status()))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        arm_summary("qwen3-8b-local"),
        arm_summary("ft-local", judge=4.0),
        arm_summary("gpt-4o-mini", judge=4.3, cost=0.0004)]))
    respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        comparison("ft-local", "qwen3-8b-local")]))
    respx.get(f"{BASE}/runs/5/equivalence").mock(return_value=httpx.Response(500, text="boom"))

    with pytest.raises(httpx.HTTPStatusError):
        fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
                 "--candidate", "ft-local", "--candidate", "gpt-4o-mini",
                 "--finetuned", "ft-local", "--out", str(tmp_path / "r.md")])


@respx.mock
def test_non_terminal_run_refused_without_force(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(
        200, json=run_status(status="running")))
    with pytest.raises(SystemExit, match="running"):
        fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
                 "--candidate", "ft-qwen3-8b-local", "--out", str(tmp_path / "r.md")])


@respx.mock
def test_non_terminal_run_allowed_with_force(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(
        200, json=run_status(status="running")))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        arm_summary("qwen3-8b-local"), arm_summary("ft-qwen3-8b-local", judge=4.0)]))
    respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        comparison("ft-qwen3-8b-local", "qwen3-8b-local")]))

    out = tmp_path / "r.md"
    fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
             "--candidate", "ft-qwen3-8b-local", "--force", "--out", str(out)])
    assert out.exists()
    assert "(`running`)" in out.read_text()
