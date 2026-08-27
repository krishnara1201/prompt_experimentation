import asyncio
import logging
import os
from pathlib import Path
import time

import httpx
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

logger = logging.getLogger(__name__)

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


def is_retryable(exc: Exception) -> bool:
    """False for errors that cannot possibly succeed on a retry.

    A missing API key raises RuntimeError from the adapters
    (`app/adapters/openai_compatible.py`, `app/adapters/anthropic.py`), and
    a 4xx response raises httpx.HTTPStatusError via raise_for_status(). Both
    are permanent — retrying only burns backoff sleep. 429 is the exception:
    rate limiting is transient, so it stays retryable, as do network errors,
    timeouts and 5xx.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return True
        return not (400 <= status < 500)
    if isinstance(exc, RuntimeError) and "No API key found in environment variable" in str(exc):
        return False
    return True


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
    def _persist_failure(error_message: str) -> None:
        # A dead DB here must not crash the Celery task — log and move on.
        try:
            asyncio.run(
                _persist_run_result(
                    run_id=run_id,
                    example_id=example_id,
                    arm_name=arm_name,
                    repeat_index=repeat_index,
                    celery_task_id=celery_task_id,
                    status="failed",
                    error_message=error_message,
                )
            )
        except Exception:
            logger.error(
                "Failed to persist failed RunResult (run_id=%s example_id=%s arm=%s repeat=%s): %s",
                run_id,
                example_id,
                arm_name,
                repeat_index,
                error_message,
                exc_info=True,
            )

    # Config loading is outside the retry loop on purpose: a bad arm name or
    # malformed arms.yaml will never succeed on retry. But it must still
    # produce a persisted failed row, or the run's derived status can never
    # reach total_calls.
    try:
        arms = load_arms(str(ARMS_PATH))
        adapter = arms[arm_name]
    except Exception as exc:
        logger.error(
            "Could not resolve arm (run_id=%s example_id=%s arm=%s repeat=%s): %s",
            run_id,
            example_id,
            arm_name,
            repeat_index,
            exc,
            exc_info=True,
        )
        _persist_failure(f"Could not resolve arm '{arm_name}': {exc!r}")
        return

    # Only the model call is retried. Persisting a successful response is
    # deliberately outside this loop: a DB failure after a billed model call
    # must never trigger another billed model call.
    attempt = 0
    last_exc: Exception | None = None
    response: ModelResponse | None = None
    while attempt <= max_retries:
        try:
            response = adapter.generate(example_text)
            break
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc):
                logger.warning(
                    "Non-retryable error (run_id=%s example_id=%s arm=%s repeat=%s): %s",
                    run_id,
                    example_id,
                    arm_name,
                    repeat_index,
                    exc,
                )
                break
            attempt += 1
            if attempt <= max_retries:
                logger.warning(
                    "Model call failed, retrying (attempt %s/%s, run_id=%s example_id=%s arm=%s repeat=%s): %s",
                    attempt,
                    max_retries,
                    run_id,
                    example_id,
                    arm_name,
                    repeat_index,
                    exc,
                )
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))

    if response is None:
        logger.error(
            "Model call gave up (run_id=%s example_id=%s arm=%s repeat=%s): %s",
            run_id,
            example_id,
            arm_name,
            repeat_index,
            last_exc,
        )
        _persist_failure(str(last_exc))
        return

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
