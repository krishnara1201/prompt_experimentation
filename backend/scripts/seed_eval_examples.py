"""Loads a task pack's eval dataset into the eval_example table.

Task-driven: ``--task <name>`` picks a pack under ``backend/tasks/`` (default
``financial_sentiment`` — the vendored Financial PhraseBank subset). Safe to
re-run: skips sentences that already exist for the pack's ``source``.

Run from inside backend/: uv run python -m scripts.seed_eval_examples [--task <name>]
"""
import argparse
import asyncio

from dotenv import load_dotenv
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.tasks import DEFAULT_TASK, load_task
from app.data.loader import load_task_examples
from app.db.models import EvalExample
from app.db.session import engine

load_dotenv()


async def seed(task_name: str = DEFAULT_TASK, tasks_dir=None) -> tuple[int, str]:
    task = load_task(task_name, tasks_dir=tasks_dir)
    examples = load_task_examples(task)
    inserted = 0
    async with AsyncSession(engine) as session:
        existing_result = await session.execute(
            select(EvalExample.text).where(EvalExample.source == task.source)
        )
        existing_texts = set(existing_result.scalars().all())

        for example in examples:
            if example.text in existing_texts:
                continue
            session.add(
                EvalExample(
                    text=example.text,
                    gold_label=example.gold_label,
                    source=task.source,
                )
            )
            inserted += 1
        await session.commit()
    return inserted, task.source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task pack to seed.")
    args = parser.parse_args()
    inserted, source = asyncio.run(seed(args.task))
    print(
        f"Inserted {inserted} new eval examples (task={args.task}, source={source})."
    )


if __name__ == "__main__":
    main()
