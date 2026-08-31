"""add task column to run

Revision ID: 0003_add_run_task
Revises: 0002_add_judge_layer
Create Date: 2026-08-30

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

revision = "0003_add_run_task"
down_revision = "0002_add_judge_layer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "run",
        sa.Column(
            "task",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="financial_sentiment",
        ),
    )


def downgrade() -> None:
    op.drop_column("run", "task")
