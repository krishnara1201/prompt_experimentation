"""create eval_example, run, run_result tables

Revision ID: 0001_create_initial_tables
Revises:
Create Date: 2026-08-25

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "0001_create_initial_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_example",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("gold_label", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("source", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("arm_names", sa.JSON(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=True),
        sa.Column("repeats", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("total_calls", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "run_result",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("example_id", sa.Integer(), nullable=False),
        sa.Column("arm_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("repeat_index", sa.Integer(), nullable=False),
        sa.Column("output_text", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_estimate_usd", sa.Float(), nullable=True),
        sa.Column("judge_score", sa.Float(), nullable=True),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("celery_task_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["run.id"]),
        sa.ForeignKeyConstraint(["example_id"], ["eval_example.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("run_result")
    op.drop_table("run")
    op.drop_table("eval_example")
