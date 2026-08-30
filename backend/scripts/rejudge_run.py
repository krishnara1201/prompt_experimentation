"""One-off: re-enqueue judge scoring for every completed RunResult in a run.

Re-scores a run with the *current* arms.yaml judge after it was partly judged
by a different judge model. `run_judge_call` overwrites the existing
judge_score/rationale, so this is idempotent.

    uv run python scripts/rejudge_run.py <run_id>
"""

import asyncio
import sys

from sqlalchemy import text

from app.db.session import engine
from app.tasks.worker import run_judge_call


async def _reset_and_fetch(run_id: int) -> list[int]:
    async with engine.begin() as conn:
        ids = (
            await conn.execute(
                text(
                    "SELECT id FROM run_result "
                    "WHERE run_id = :rid AND status = 'completed' ORDER BY id"
                ),
                {"rid": run_id},
            )
        ).scalars().all()
        await conn.execute(
            text(
                "UPDATE run_result SET judge_status = 'pending', judge_score = NULL, "
                "judge_rationale = NULL, judge_error_message = NULL "
                "WHERE run_id = :rid AND status = 'completed'"
            ),
            {"rid": run_id},
        )
    return list(ids)


def main(run_id: int) -> None:
    ids = asyncio.run(_reset_and_fetch(run_id))
    for result_id in ids:
        run_judge_call.apply_async(kwargs={"run_result_id": result_id}, queue="celery")
    print(f"re-enqueued {len(ids)} judge calls for run {run_id}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
