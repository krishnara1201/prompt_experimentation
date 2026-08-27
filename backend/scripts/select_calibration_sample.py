"""Selects a stratified sample of judged RunResults for human calibration
labeling. Writes a JSON file — fill in each row's `human_score` by hand,
then run import_calibration_labels.py.

Run from inside backend/:
uv run python -m scripts.select_calibration_sample --run-id 1 --n 40 --out calibration_sample.json
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import EvalExample, RunResult
from app.db.session import engine


async def _fetch_judged_rows(run_id: int) -> list[dict]:
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(RunResult, EvalExample)
            .join(EvalExample, RunResult.example_id == EvalExample.id)
            .where(RunResult.run_id == run_id, RunResult.judge_status == "completed")
            .order_by(RunResult.id)
        )
        rows = []
        for run_result, example in result.all():
            rows.append(
                {
                    "run_result_id": run_result.id,
                    "arm_name": run_result.arm_name,
                    "input_text": example.text,
                    "gold_label": example.gold_label,
                    "model_output": run_result.output_text,
                    "judge_score": run_result.judge_score,
                    "judge_rationale": run_result.judge_rationale,
                    "human_score": None,
                }
            )
        return rows


def stratified_sample(rows: list[dict], n: int, seed: int | None = None) -> list[dict]:
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["run_result_id"])

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["arm_name"], row["gold_label"])
        buckets.setdefault(key, []).append(row)

    ordered_keys = sorted(buckets.keys())
    for key in ordered_keys:
        rng.shuffle(buckets[key])

    quota = max(1, n // len(ordered_keys))
    selected: list[dict] = []
    leftovers: list[dict] = []
    for key in ordered_keys:
        bucket = buckets[key]
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    return sorted(selected, key=lambda r: r["run_result_id"])


async def select_sample(run_id: int, n: int, seed: int | None = None) -> list[dict]:
    rows = await _fetch_judged_rows(run_id)
    return stratified_sample(rows, n, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sample = asyncio.run(select_sample(args.run_id, args.n, seed=args.seed))
    args.out.write_text(json.dumps(sample, indent=2))
    print(f"Wrote {len(sample)} rows to {args.out}")


if __name__ == "__main__":
    main()
