"""Judge a run's RunResults serially, in-process — no Celery/Redis/worker.

A low-memory fallback for hosts where the Ollama judge model + the full
Docker stack don't fit in RAM together. Needs only Postgres reachable at
DATABASE_URL and the arms.yaml `judge:` adapter reachable. Idempotent:
overwrites judge_score/rationale for every row it touches.

    uv run python -m scripts.serial_judge_run <run_id> [--only-unjudged]
"""
import argparse
import asyncio

from sqlalchemy import text

from app.config.arms import load_judge_arm
from app.config.tasks import load_task
from app.db.session import engine
from app.judge.scorer import score_output
from app.tasks.worker import ARMS_PATH


async def _rows(run_id: int, only_unjudged: bool) -> tuple[str, list[tuple[int, str, str, str]]]:
    async with engine.begin() as conn:
        task_name = (
            await conn.execute(text("SELECT task FROM run WHERE id = :r"), {"r": run_id})
        ).scalar_one()
        q = (
            "SELECT rr.id, e.text, e.gold_label, rr.output_text "
            "FROM run_result rr JOIN eval_example e ON rr.example_id = e.id "
            "WHERE rr.run_id = :r AND rr.status = 'completed'"
        )
        if only_unjudged:
            q += " AND (rr.judge_status IS NULL OR rr.judge_status <> 'completed')"
        q += " ORDER BY rr.id"
        rows = (await conn.execute(text(q), {"r": run_id})).all()
    return task_name, [tuple(row) for row in rows]


async def _persist(run_result_id: int, status: str, score: int | None, rationale: str | None, err: str | None) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE run_result SET judge_status = :s, judge_score = :sc, "
                "judge_rationale = :ra, judge_error_message = :e WHERE id = :i"
            ),
            {"s": status, "sc": score, "ra": rationale, "e": err, "i": run_result_id},
        )


async def main(run_id: int, only_unjudged: bool) -> None:
    task_name, rows = await _rows(run_id, only_unjudged)
    task_cfg = load_task(task_name)
    judge = load_judge_arm(str(ARMS_PATH))
    print(f"judging {len(rows)} rows for run {run_id} (task={task_name})")
    ok = fail = 0
    for i, (rr_id, input_text, gold_label, model_output) in enumerate(rows, 1):
        try:
            res = score_output(
                judge, input_text, gold_label, model_output,
                rubric_template=task_cfg.rubric, description=task_cfg.description,
            )
            await _persist(rr_id, "completed", res.score, res.rationale, None)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - best-effort, record and move on
            await _persist(rr_id, "failed", None, None, str(exc))
            fail += 1
            print(f"  [{i}/{len(rows)}] rr={rr_id} FAILED: {exc!r}")
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}  ok={ok} fail={fail}")
    print(f"done: ok={ok} fail={fail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", type=int)
    ap.add_argument("--only-unjudged", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.run_id, args.only_unjudged))
