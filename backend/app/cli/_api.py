"""HTTP helpers for the `pe` CLI — a thin wrapper over the platform API."""
import os

import httpx
import typer

DEFAULT_BASE_URL = "http://localhost:8000"
# The stats endpoints (/compare bootstrap, /equivalence PyMC sampling) can run
# well past a typical HTTP timeout on a full run. Default generously; override
# with PE_API_TIMEOUT for a slow host.
DEFAULT_TIMEOUT_SECONDS = 120.0


def base_url() -> str:
    return os.environ.get("PE_API_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout() -> float:
    return float(os.environ.get("PE_API_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))


def _request(method: str, path: str, **kwargs) -> object:
    url = f"{base_url()}{path}"
    try:
        response = httpx.request(method, url, timeout=_timeout(), **kwargs)
    except httpx.RequestError as exc:
        typer.secho(
            f"Cannot reach the API at {base_url()} ({exc}). "
            "Is the stack up? Try: pe up",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from exc

    if response.is_success:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    typer.secho(f"API error {response.status_code}: {detail}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def api_get(path: str, **params) -> object:
    clean = {k: v for k, v in params.items() if v is not None}
    return _request("GET", path, params=clean)


def api_post(path: str, json: dict) -> object:
    return _request("POST", path, json=json)
