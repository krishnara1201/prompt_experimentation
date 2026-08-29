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
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.adapters.base import ModelResponse
from app.config.arms import load_arms, load_judge_arm
from app.db.models import EvalExample, RunResult
from app.db.session import DATABASE_URL
from app.eval_prompt import render_eval_prompt
from app.judge.scorer import JudgeParseError, JudgeResult, score_output

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
) -> int | None:
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            run_result = RunResult(
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
            session.add(run_result)
            await session.commit()
            await session.refresh(run_result)
            return run_result.id
    finally:
        await worker_engine.dispose()


async def _load_run_result_for_judging(run_result_id: int) -> tuple[str, str, str] | None:
    """Returns (input_text, gold_label, model_output), or None if the row is missing."""
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            result = await session.execute(
                select(EvalExample.text, EvalExample.gold_label, RunResult.output_text)
                .join(RunResult, RunResult.example_id == EvalExample.id)
                .where(RunResult.id == run_result_id)
            )
            row = result.first()
            return tuple(row) if row else None
    finally:
        await worker_engine.dispose()


async def _persist_judge_result(
    *,
    run_result_id: int,
    celery_task_id: str | None,
    status: str,
    score: int | None = None,
    rationale: str | None = None,
    error_message: str | None = None,
) -> None:
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(worker_engine) as session:
            run_result = await session.get(RunResult, run_result_id)
            if run_result is None:
                logger.error("Cannot persist judge result: RunResult %s not found", run_result_id)
                return
            run_result.judge_status = status
            run_result.judge_score = score
            run_result.judge_rationale = rationale
            run_result.judge_error_message = error_message
            run_result.judge_celery_task_id = celery_task_id
            session.add(run_result)
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
    timeouts and 5xx. An unauthenticated subscription CLI
    (`app/adapters/claude_code_cli.py`, `app/adapters/codex_cli.py`) is the
    same kind of permanent failure as a missing API key, and so is a missing
    CLI binary (raised as RuntimeError with "not found on PATH" from the
    same two adapters) — no retry will make the binary appear on PATH.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return True
        return not (400 <= status < 500)
    if isinstance(exc, RuntimeError) and "No API key found in environment variable" in str(exc):
        return False
    if isinstance(exc, RuntimeError) and "is not authenticated" in str(exc):
        return False
    if isinstance(exc, RuntimeError) and "not found on PATH" in str(exc):
        return False
    if isinstance(exc, JudgeParseError):
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
            response = adapter.generate(render_eval_prompt(example_text))
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

    result_id = asyncio.run(
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
    # A subscription-CLI judge (celery_queue="subscription_cli") only has a
    # CLI binary/authenticated session on its own dedicated worker, same as
    # a subscription-CLI eval arm -- .delay() would enqueue to the default
    # queue and hang forever with nothing eligible to consume it.
    judge_queue = getattr(load_judge_arm(str(ARMS_PATH)), "celery_queue", "celery")
    run_judge_call.apply_async(kwargs={"run_result_id": result_id}, queue=judge_queue)


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


def execute_judge_call(
    *,
    run_result_id: int,
    celery_task_id: str | None = None,
    max_retries: int = 3,
    backoff_base_seconds: float = 1.0,
) -> None:
    def _persist_failure(error_message: str) -> None:
        try:
            asyncio.run(
                _persist_judge_result(
                    run_result_id=run_result_id,
                    celery_task_id=celery_task_id,
                    status="failed",
                    error_message=error_message,
                )
            )
        except Exception:
            logger.error(
                "Failed to persist failed judge result (run_result_id=%s): %s",
                run_result_id,
                error_message,
                exc_info=True,
            )

    try:
        loaded = asyncio.run(_load_run_result_for_judging(run_result_id))
    except Exception as exc:
        logger.error(
            "Could not load RunResult for judging (run_result_id=%s): %s", run_result_id, exc, exc_info=True
        )
        _persist_failure(f"Could not load RunResult: {exc!r}")
        return

    if loaded is None:
        logger.error("RunResult %s not found for judging", run_result_id)
        _persist_failure(f"RunResult {run_result_id} not found")
        return

    input_text, gold_label, model_output = loaded

    try:
        judge_adapter = load_judge_arm(str(ARMS_PATH))
    except Exception as exc:
        logger.error("Could not resolve judge arm (run_result_id=%s): %s", run_result_id, exc, exc_info=True)
        _persist_failure(f"Could not resolve judge arm: {exc!r}")
        return

    attempt = 0
    last_exc: Exception | None = None
    judge_result: JudgeResult | None = None
    while attempt <= max_retries:
        try:
            judge_result = score_output(judge_adapter, input_text, gold_label, model_output)
            break
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc):
                logger.warning("Non-retryable judge error (run_result_id=%s): %s", run_result_id, exc)
                break
            attempt += 1
            if attempt <= max_retries:
                logger.warning(
                    "Judge call failed, retrying (attempt %s/%s, run_result_id=%s): %s",
                    attempt,
                    max_retries,
                    run_result_id,
                    exc,
                )
                time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))

    if judge_result is None:
        logger.error("Judge call gave up (run_result_id=%s): %s", run_result_id, last_exc)
        _persist_failure(str(last_exc))
        return

    asyncio.run(
        _persist_judge_result(
            run_result_id=run_result_id,
            celery_task_id=celery_task_id,
            status="completed",
            score=judge_result.score,
            rationale=judge_result.rationale,
        )
    )


@celery_app.task(bind=True)
def run_judge_call(self, run_result_id: int) -> None:
    execute_judge_call(run_result_id=run_result_id, celery_task_id=self.request.id)
