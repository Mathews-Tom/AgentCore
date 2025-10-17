"""add_training_tables

Revision ID: e9ba324940b6
Revises: d1509698cef3
Create Date: 2025-10-17 01:31:57.352209

"""
from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "e9ba324940b6"
down_revision: str | Sequence[str] | None = "d1509698cef3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: Add training infrastructure tables."""
    # Create training_jobs table
    op.create_table(
        "training_jobs",
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("agent_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("config", postgresql.JSONB, nullable=False),
        sa.Column("training_data", postgresql.JSONB, nullable=False),
        sa.Column("current_iteration", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_iterations", sa.Integer, nullable=False),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "cost_usd",
            sa.DECIMAL(10, 2),
            nullable=False,
            server_default="0.00",
        ),
        sa.Column("budget_usd", sa.DECIMAL(10, 2), nullable=False),
        sa.Column("best_checkpoint_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("error_message", sa.String(1000), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="check_training_job_status",
        ),
    )

    # Create indexes for training_jobs
    op.create_index("idx_training_jobs_agent", "training_jobs", ["agent_id"])
    op.create_index("idx_training_jobs_status", "training_jobs", ["status"])
    op.create_index(
        "idx_training_jobs_created",
        "training_jobs",
        [sa.text("created_at DESC")],
        postgresql_ops={"created_at": "DESC"},
    )

    # Create trajectories table
    op.create_table(
        "trajectories",
        sa.Column(
            "trajectory_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("agent_id", sa.String(255), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("steps", postgresql.JSONB, nullable=False),
        sa.Column("reward", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("normalized_reward", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("advantage", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("execution_time_ms", sa.Integer, nullable=True),
        sa.Column("success", sa.Boolean, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["training_jobs.job_id"],
            name="fk_trajectories_job_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes for trajectories
    op.create_index("idx_trajectories_job", "trajectories", ["job_id"])
    op.create_index("idx_trajectories_agent", "trajectories", ["agent_id"])
    op.create_index(
        "idx_trajectories_created",
        "trajectories",
        [sa.text("created_at DESC")],
        postgresql_ops={"created_at": "DESC"},
    )
    op.create_index("idx_trajectories_success", "trajectories", ["success"])
    # GIN index for JSONB column (supports containment and existence queries)
    op.create_index(
        "idx_trajectories_steps_gin",
        "trajectories",
        ["steps"],
        postgresql_using="gin",
    )

    # Create policy_checkpoints table
    op.create_table(
        "policy_checkpoints",
        sa.Column(
            "checkpoint_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("agent_id", sa.String(255), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("iteration", sa.Integer, nullable=False),
        sa.Column("policy_data", postgresql.JSONB, nullable=True),
        sa.Column("policy_s3_path", sa.String(500), nullable=True),
        sa.Column(
            "validation_score", sa.Float, nullable=False, server_default="0.0"
        ),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["training_jobs.job_id"],
            name="fk_policy_checkpoints_job_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes for policy_checkpoints
    op.create_index(
        "idx_policy_checkpoints_agent", "policy_checkpoints", ["agent_id"]
    )
    op.create_index("idx_policy_checkpoints_job", "policy_checkpoints", ["job_id"])
    op.create_index(
        "idx_policy_checkpoints_created",
        "policy_checkpoints",
        [sa.text("created_at DESC")],
        postgresql_ops={"created_at": "DESC"},
    )
    op.create_index(
        "idx_policy_checkpoints_validation",
        "policy_checkpoints",
        ["validation_score"],
    )


def downgrade() -> None:
    """Downgrade schema: Drop training infrastructure tables."""
    # Drop tables in reverse order (children first due to foreign keys)
    op.drop_table("policy_checkpoints")
    op.drop_table("trajectories")
    op.drop_table("training_jobs")
