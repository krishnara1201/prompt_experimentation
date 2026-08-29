"""Subprocess helpers for the `pe` CLI.

`_run` is a single module-level indirection so tests can assert the argv
of a `docker compose` / `uv run` invocation without spawning anything.
"""
import subprocess
from functools import lru_cache
from pathlib import Path

import typer

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Nearest ancestor of the backend package that holds docker-compose.yml."""
    for candidate in [BACKEND_DIR, *BACKEND_DIR.parents]:
        if (candidate / "docker-compose.yml").is_file():
            return candidate
    # Fall back to the backend dir's parent rather than guessing further.
    return BACKEND_DIR.parent


def _run(argv: list[str], *, cwd: Path) -> int:
    """Run `argv` in `cwd`, streaming output. Returns the exit code."""
    return subprocess.run(argv, cwd=cwd).returncode


def compose(*args: str) -> None:
    """`docker compose <args>` from the repo root; exits the CLI on failure."""
    code = _run(["docker", "compose", *args], cwd=repo_root())
    if code != 0:
        raise typer.Exit(code)


def backend_script(module: str, *args: str) -> None:
    """`uv run python -m <module> <args>` from backend/; exits on failure."""
    code = _run(["uv", "run", "python", "-m", module, *args], cwd=BACKEND_DIR)
    if code != 0:
        raise typer.Exit(code)
