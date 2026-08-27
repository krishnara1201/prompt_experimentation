"""Loads the vendored Financial PhraseBank file into the eval_example
table. Safe to re-run: skips sentences that already exist for this source.

Run from inside backend/: uv run python -m scripts.seed_eval_examples
"""
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.data.financial_phrasebank import load_examples
from app.db.models import EvalExample
from app.db.session import engine

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "financial_phrasebank" / "sentences_allagree.txt"
SOURCE = "financial_phrasebank_allagree"


async def seed() -> int:
    examples = load_examples(DATA_PATH)
    inserted = 0
    async with AsyncSession(engine) as session:
        existing_result = await session.execute(
            select(EvalExample.text).where(EvalExample.source == SOURCE)
        )
        existing_texts = set(existing_result.scalars().all())

        for example in examples:
            if example.text in existing_texts:
                continue
            session.add(EvalExample(text=example.text, gold_label=example.label, source=SOURCE))
            inserted += 1
        await session.commit()
    return inserted


def main() -> None:
    inserted = asyncio.run(seed())
    print(f"Inserted {inserted} new eval examples (source={SOURCE}).")


if __name__ == "__main__":
    main()
