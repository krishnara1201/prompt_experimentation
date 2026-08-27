"""Computes judge/human agreement for a run's calibration sample. Run this
and read the result before trusting judge_score on the rest of the run.

Run from inside backend/:
uv run python -m scripts.calibration_report --run-id 1
"""
import argparse
import asyncio

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import JudgeCalibrationLabel, RunResult
from app.db.session import engine
from app.judge.calibration import calibration_report


async def _fetch_pairs(run_id: int) -> list[tuple[float, int]]:
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(RunResult.judge_score, JudgeCalibrationLabel.human_score)
            .join(JudgeCalibrationLabel, JudgeCalibrationLabel.run_result_id == RunResult.id)
            .where(RunResult.run_id == run_id)
        )
        return [(judge_score, human_score) for judge_score, human_score in result.all()]


async def build_report(run_id: int) -> dict:
    pairs = await _fetch_pairs(run_id)
    return calibration_report(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    report = asyncio.run(build_report(args.run_id))
    print(f"n = {report['n']}")
    print(f"Spearman r = {report['spearman_r']:.3f} (p = {report['spearman_p']:.3f})")
    print(f"Cohen's kappa (score>=4 as correct) = {report['cohens_kappa']:.3f}")
    print(f"Mean absolute difference = {report['mean_abs_diff']:.3f}")


if __name__ == "__main__":
    main()
