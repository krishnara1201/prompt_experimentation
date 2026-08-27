"""add judge columns to run_result, create judge_calibration_label

Revision ID: 0002_add_judge_layer
Revises: 0001_create_initial_tables
Create Date: 2026-08-27

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "0002_add_judge_layer"
down_revision = "0001_create_initial_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("run_result", sa.Column("judge_rationale", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column(
        "run_result",
        sa.Column("judge_status", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="pending"),
    )
    op.add_column("run_result", sa.Column("judge_error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column("run_result", sa.Column("judge_celery_task_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    op.create_table(
        "judge_calibration_label",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_result_id", sa.Integer(), nullable=False),
        sa.Column("human_score", sa.Integer(), nullable=False),
        sa.Column("labeled_by", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("notes", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("labeled_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_result_id"], ["run_result.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_judge_calibration_label_run_result_id", "judge_calibration_label", ["run_result_id"])


def downgrade() -> None:
    op.drop_index("ix_judge_calibration_label_run_result_id", table_name="judge_calibration_label")
    op.drop_table("judge_calibration_label")
    op.drop_column("run_result", "judge_celery_task_id")
    op.drop_column("run_result", "judge_error_message")
    op.drop_column("run_result", "judge_status")
    op.drop_column("run_result", "judge_rationale")
