"""Plain-text rendering helpers — no Rich, just aligned columns."""
import typer


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def kv(data: dict) -> None:
    """Print a dict as aligned `key: value` lines."""
    if not data:
        typer.echo("(no data)")
        return
    width = max(len(k) for k in data)
    for key, value in data.items():
        typer.echo(f"{key.rjust(width)} : {_fmt(value)}")


def table(rows: list[dict], columns: list[str] | None = None) -> None:
    """Print a list of dicts as an aligned table."""
    if not rows:
        typer.echo("(no rows)")
        return
    columns = columns or list(rows[0].keys())
    cells = [[_fmt(row.get(col)) for col in columns] for row in rows]
    widths = [max(len(col), *(len(r[i]) for r in cells)) for i, col in enumerate(columns)]
    typer.echo("  ".join(col.ljust(widths[i]) for i, col in enumerate(columns)))
    typer.echo("  ".join("-" * widths[i] for i in range(len(columns))))
    for row in cells:
        typer.echo("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))))
