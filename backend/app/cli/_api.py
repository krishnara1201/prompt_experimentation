"""HTTP helpers for the `pe` CLI — a thin wrapper over the platform API."""
import os

import httpx
import typer

DEFAULT_BASE_URL = "http://localhost:8000"


def base_url() -> str:
    return os.environ.get("PE_API_URL", DEFAULT_BASE_URL).rstrip("/")


def _request(method: str, path: str, **kwargs) -> object:
    url = f"{base_url()}{path}"
    try:
        response = httpx.request(method, url, timeout=30.0, **kwargs)
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
