"""Imports hand-filled human_score values from a calibration sample JSON
file (produced by select_calibration_sample.py) into judge_calibration_label.
Idempotent: re-running upserts by run_result_id rather than duplicating.

Run from inside backend/:
uv run python -m scripts.import_calibration_labels --in calibration_sample.json --labeled-by you@example.com
"""
import argparse
import asyncio
import json
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import JudgeCalibrationLabel
from app.db.session import engine


class CalibrationImportError(ValueError):
    pass


def _validate(rows: list[dict]) -> None:
    for row in rows:
        score = row.get("human_score")
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise CalibrationImportError(
                f"run_result_id {row.get('run_result_id')} has invalid human_score: {score!r}"
            )


async def import_labels(rows: list[dict], labeled_by: str) -> int:
    _validate(rows)
    upserted = 0
    async with AsyncSession(engine) as session:
        for row in rows:
            result = await session.execute(
                select(JudgeCalibrationLabel).where(
                    JudgeCalibrationLabel.run_result_id == row["run_result_id"]
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.human_score = row["human_score"]
                existing.labeled_by = labeled_by
                existing.notes = row.get("notes")
                session.add(existing)
            else:
                session.add(
                    JudgeCalibrationLabel(
                        run_result_id=row["run_result_id"],
                        human_score=row["human_score"],
                        labeled_by=labeled_by,
                        notes=row.get("notes"),
                    )
                )
            upserted += 1
        await session.commit()
    return upserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--labeled-by", required=True)
    args = parser.parse_args()

    rows = json.loads(args.in_path.read_text())
    count = asyncio.run(import_labels(rows, args.labeled_by))
    print(f"Upserted {count} calibration labels.")


if __name__ == "__main__":
    main()
