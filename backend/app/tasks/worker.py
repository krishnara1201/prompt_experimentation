import asyncio
import os
from pathlib import Path
import time

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel.ext.asyncio.session import AsyncSession

from app.adapters.base import ModelResponse
from app.config.arms import load_arms
from app.db.models import RunResult
from app.db.session import DATABASE_URL

load_dotenv()

ARMS_PATH = Path(__file__).resolve().parent.parent.parent / "arms.yaml"

celery_app = Celery(
    "worker",
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_BACKEND_URL"),
    broker_connection_retry_on_startup=True,
)


async def _persist_run_result(
    *,
    run_id: int,
    example_id: int,
    arm_name: str,
    repeat_index: int,
    celery_task_id: str | None,
    status: str,
    response: ModelResponse | None = None,
    error_message: str | None = None,
) -> None:
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            session.add(
                RunResult(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    celery_task_id=celery_task_id,
                    status=status,
                    output_text=response.text if response else None,
                    latency_ms=response.latency_ms if response else None,
                    prompt_tokens=response.prompt_tokens if response else None,
                    completion_tokens=response.completion_tokens if response else None,
                    cost_estimate_usd=response.cost_estimate_usd if response else None,
                    error_message=error_message,
                )
            )
            await session.commit()
    finally:
        await worker_engine.dispose()


def execute_call(
    *,
    run_id: int,
    example_id: int,
    example_text: str,
    arm_name: str,
    repeat_index: int,
    celery_task_id: str | None = None,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.0,
) -> None:
    arms = load_arms(str(ARMS_PATH))
    adapter = arms[arm_name]

    attempt = 0
    last_exc: Exception | None = None
    while attempt <= max_retries:
        try:
            response = adapter.generate(example_text)
            asyncio.run(
                _persist_run_result(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    celery_task_id=celery_task_id,
                    status="completed",
                    response=response,
                )
            )
            return
        except Exception as exc:
            last_exc = exc
            attempt += 1
            if attempt <= max_retries:
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))

    asyncio.run(
        _persist_run_result(
            run_id=run_id,
            example_id=example_id,
            arm_name=arm_name,
            repeat_index=repeat_index,
            celery_task_id=celery_task_id,
            status="failed",
            error_message=str(last_exc),
        )
    )


@celery_app.task(bind=True)
def run_single_call(self, run_id: int, example_id: int, example_text: str, arm_name: str, repeat_index: int) -> None:
    execute_call(
        run_id=run_id,
        example_id=example_id,
        example_text=example_text,
        arm_name=arm_name,
        repeat_index=repeat_index,
        celery_task_id=self.request.id,
    )
