from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EvalExample(SQLModel, table=True):
    __tablename__ = "eval_example"

    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    gold_label: str
    source: str


class Run(SQLModel, table=True):
    __tablename__ = "run"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)
    arm_names: list[str] = Field(sa_column=Column(JSON, nullable=False))
    sample_size: Optional[int] = Field(default=None)
    repeats: int
    seed: Optional[int] = Field(default=None)
    total_calls: int


class RunResult(SQLModel, table=True):
    __tablename__ = "run_result"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    example_id: int = Field(foreign_key="eval_example.id")
    arm_name: str
    repeat_index: int
    output_text: Optional[str] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None)
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)
    cost_estimate_usd: Optional[float] = Field(default=None)
    judge_score: Optional[float] = Field(default=None)
    judge_rationale: Optional[str] = Field(default=None)
    judge_status: str = Field(default="pending")
    judge_error_message: Optional[str] = Field(default=None)
    judge_celery_task_id: Optional[str] = Field(default=None)
    status: str
    error_message: Optional[str] = Field(default=None)
    celery_task_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)


class JudgeCalibrationLabel(SQLModel, table=True):
    __tablename__ = "judge_calibration_label"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_result_id: int = Field(foreign_key="run_result.id", index=True)
    human_score: int
    labeled_by: str
    notes: Optional[str] = Field(default=None)
    labeled_at: datetime = Field(default_factory=_utcnow)
