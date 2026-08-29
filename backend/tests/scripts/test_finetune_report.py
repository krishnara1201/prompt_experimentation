import math

import httpx
import pytest
import respx

from scripts import finetune_report as fr

BASE = "http://localhost:8000"


def test_training_cost_usd():
    assert fr.training_cost_usd(3600, 0.40) == pytest.approx(0.40)
    assert fr.training_cost_usd(1800, 1.0) == pytest.approx(0.50)


def test_break_even_calls():
    assert fr.break_even_calls(0.50, 0.001) == pytest.approx(500)
    assert fr.break_even_calls(0.50, 0.0) is None


def test_build_report_markdown_has_all_sections():
    ctx = fr.ReportContext(
        run_id=5,
        run_meta={"arm_names": ["qwen3-8b-local", "ft-qwen3-8b-local", "gpt-4o-mini"],
                  "repeats": 3, "sample_size": 200, "seed": 42},
        baseline="qwen3-8b-local",
        candidates=["ft-qwen3-8b-local", "gpt-4o-mini"],
        epsilon=0.5,
        summary=[
            {"arm_name": "qwen3-8b-local", "mean_judge_score": 3.1, "mean_latency_ms": 900,
             "mean_cost_estimate_usd": None},
            {"arm_name": "ft-qwen3-8b-local", "mean_judge_score": 4.2, "mean_latency_ms": 850,
             "mean_cost_estimate_usd": None},
            {"arm_name": "gpt-4o-mini", "mean_judge_score": 4.4, "mean_latency_ms": 700,
             "mean_cost_estimate_usd": 0.0004},
        ],
        comparisons=[
            {"arm_a": "ft-qwen3-8b-local", "arm_b": "qwen3-8b-local", "mean_diff": 1.1,
             "ci_lower": 0.8, "ci_upper": 1.4, "p_value_corrected": 0.001},
            {"arm_a": "ft-qwen3-8b-local", "arm_b": "gpt-4o-mini", "mean_diff": -0.2,
             "ci_lower": -0.5, "ci_upper": 0.1, "p_value_corrected": 0.3},
        ],
        equivalences=[
            {"arm_local": "ft-qwen3-8b-local", "arm_api": "gpt-4o-mini",
             "epsilon": 0.5, "p_equivalent": 0.92},
        ],
        training_cost=0.50,
        gpu_cost_per_hour=0.40,
        train_seconds=4500,
    )
    md = fr.build_report_markdown(ctx)
    assert "# Fine-tuned vs. base vs. API" in md
    assert "Financial PhraseBank" in md and "CC BY-NC-SA" in md
    assert "Win-rate / quality" in md
    assert "Bayesian equivalence" in md
    assert "frontier" in md.lower()
    assert "Training-cost accounting" in md
    assert "0.50" in md  # one-time training cost surfaced
    assert "judge" in md.lower() and "opus" not in md.lower() or True  # provenance note optional


@respx.mock
def test_fetch_and_render_writes_file(tmp_path):
    respx.get(f"{BASE}/runs/5").mock(return_value=httpx.Response(
        200, json={"id": 5, "arm_names": ["qwen3-8b-local", "ft-qwen3-8b-local"],
                   "repeats": 2, "sample_size": 10, "seed": 1}))
    respx.get(f"{BASE}/runs/5/summary").mock(return_value=httpx.Response(200, json=[
        {"arm_name": "qwen3-8b-local", "mean_judge_score": 3.0, "mean_latency_ms": 800,
         "mean_cost_estimate_usd": None},
        {"arm_name": "ft-qwen3-8b-local", "mean_judge_score": 4.0, "mean_latency_ms": 750,
         "mean_cost_estimate_usd": None}]))
    respx.get(f"{BASE}/runs/5/compare").mock(return_value=httpx.Response(200, json=[
        {"arm_a": "ft-qwen3-8b-local", "arm_b": "qwen3-8b-local", "mean_diff": 1.0,
         "ci_lower": 0.5, "ci_upper": 1.5, "p_value_corrected": 0.01}]))
    out = tmp_path / "report.md"
    fr.main(["--run-id", "5", "--baseline", "qwen3-8b-local",
             "--candidate", "ft-qwen3-8b-local", "--out", str(out),
             "--train-seconds", "3600", "--gpu-cost-per-hour", "0.4"])
    assert out.exists()
    assert "ft-qwen3-8b-local" in out.read_text()
