import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from app.cli import app
from app.cli import _shell

runner = CliRunner()
BASE = "http://localhost:8000"


@pytest.fixture
def captured_argv(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, *, cwd):
        calls.append(argv)
        return 0

    monkeypatch.setattr(_shell, "_run", fake_run)
    return calls


# --- HTTP-backed commands ------------------------------------------------


@respx.mock
def test_run_posts_expected_body():
    route = respx.post(f"{BASE}/runs").mock(
        return_value=httpx.Response(200, json={"run_id": 7, "status": "pending", "total_calls": 30})
    )
    result = runner.invoke(
        app, ["run", "--sample", "5", "--repeats", "3", "--seed", "42", "--arm", "a", "--arm", "b"]
    )
    assert result.exit_code == 0
    assert json.loads(route.calls.last.request.content) == {
        "repeats": 3,
        "sample_size": 5,
        "seed": 42,
        "arms": ["a", "b"],
    }
    assert "7" in result.stdout


@respx.mock
def test_run_quiet_prints_only_id():
    respx.post(f"{BASE}/runs").mock(
        return_value=httpx.Response(200, json={"run_id": 12, "status": "pending", "total_calls": 4})
    )
    result = runner.invoke(app, ["run", "-q"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "12"


@respx.mock
def test_stats_compare_renders_rows():
    respx.get(f"{BASE}/runs/3/compare").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "arm_a": "local",
                    "arm_b": "api",
                    "metric": "judge_score",
                    "n_examples": 20,
                    "n_excluded": 0,
                    "mean_diff": -0.15,
                    "ci_lower": -0.4,
                    "ci_upper": 0.1,
                    "wilcoxon_statistic": 12.0,
                    "p_value": 0.3,
                    "p_value_corrected": 0.3,
                }
            ],
        )
    )
    result = runner.invoke(app, ["stats", "compare", "3"])
    assert result.exit_code == 0
    assert "local" in result.stdout and "api" in result.stdout


@respx.mock
def test_api_error_prints_detail_and_exits_1():
    respx.post(f"{BASE}/runs").mock(
        return_value=httpx.Response(400, json={"detail": "Unknown arm(s): nope"})
    )
    result = runner.invoke(app, ["run", "--arm", "nope"])
    assert result.exit_code == 1
    assert "Unknown arm(s): nope" in result.output


@respx.mock
def test_connection_error_hints_to_bring_stack_up():
    respx.get(f"{BASE}/runs/1").mock(side_effect=httpx.ConnectError("refused"))
    result = runner.invoke(app, ["status", "1"])
    assert result.exit_code == 1
    assert "pe up" in result.output


@respx.mock
def test_watch_exits_1_when_calls_failed():
    respx.get(f"{BASE}/runs/9").mock(
        return_value=httpx.Response(
            200,
            json={
                "run_id": 9,
                "status": "completed_with_errors",
                "total_calls": 10,
                "completed": 8,
                "failed": 2,
                "pending": 0,
            },
        )
    )
    result = runner.invoke(app, ["watch", "9"])
    assert result.exit_code == 1


# --- shell-backed commands ---------------------------------------------


def test_seed_runs_one_off_migrate_container(captured_argv):
    result = runner.invoke(app, ["seed"])
    assert result.exit_code == 0
    assert captured_argv == [
        ["docker", "compose", "run", "--rm", "migrate", "uv", "run", "python", "-m", "scripts.seed_eval_examples"]
    ]


def test_up_wait_builds_expected_compose_argv(captured_argv, monkeypatch):
    monkeypatch.setattr("app.cli._wait_for_api", lambda *a, **k: None)
    result = runner.invoke(app, ["up"])
    assert result.exit_code == 0
    assert captured_argv[0] == ["docker", "compose", "up", "-d", "--build", "--wait"]


def test_calibrate_select_invokes_host_script(captured_argv):
    result = runner.invoke(
        app, ["calibrate", "select", "--run-id", "1", "--out", "s.json", "--n", "10", "--seed", "5"]
    )
    assert result.exit_code == 0
    assert captured_argv == [
        [
            "uv", "run", "python", "-m", "scripts.select_calibration_sample",
            "--run-id", "1", "--n", "10", "--out", "s.json", "--seed", "5",
        ]
    ]


def test_repo_root_contains_compose_file():
    from app.cli._shell import repo_root

    assert (repo_root() / "docker-compose.yml").is_file()
