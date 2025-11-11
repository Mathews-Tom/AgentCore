"""add_modular_executions_tables

Add database schema for Modular Agent Core execution tracking:
- modular_executions: Top-level execution records
- execution_plans: Structured execution plans with JSON storage
- plan_steps: Individual steps with dependencies
- module_transitions: Module flow tracking

Revision ID: e247cdc4c183
Revises: c03db99da40b
Create Date: 2025-11-11 00:27:18.351432

"""
# type: ignore  # SQLAlchemy JSONB is untyped
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e247cdc4c183'
down_revision: Union[str, Sequence[str], None] = 'c03db99da40b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create enum types for modular execution statuses
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'planstatus') THEN
                CREATE TYPE planstatus AS ENUM ('pending', 'in_progress', 'completed', 'failed', 'cancelled');
            END IF;
        END$$;
    """)

    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'stepstatus') THEN
                CREATE TYPE stepstatus AS ENUM ('pending', 'in_progress', 'completed', 'failed', 'skipped');
            END IF;
        END$$;
    """)

    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'moduletype') THEN
                CREATE TYPE moduletype AS ENUM ('planner', 'executor', 'verifier', 'generator');
            END IF;
        END$$;
    """)

    # Create modular_executions table
    op.create_table(
        'modular_executions',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('plan_id', sa.String(length=255), nullable=True),
        sa.Column('iterations', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('final_result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('status', postgresql.ENUM('pending', 'in_progress', 'completed', 'failed', 'cancelled', name='planstatus', create_type=False), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_modular_executions_status', 'modular_executions', ['status'], unique=False)
    op.create_index('idx_modular_executions_created_at', 'modular_executions', ['created_at'], unique=False)
    op.create_index('idx_modular_executions_plan_id', 'modular_executions', ['plan_id'], unique=False)

    # Create execution_plans table
    op.create_table(
        'execution_plans',
        sa.Column('plan_id', sa.String(length=255), nullable=False),
        sa.Column('execution_id', sa.String(length=255), nullable=False),
        sa.Column('plan_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'in_progress', 'completed', 'failed', 'cancelled', name='planstatus', create_type=False), nullable=False),
        sa.Column('max_iterations', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('current_iteration', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('success_criteria', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('final_result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('total_estimated_cost', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('actual_cost', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('plan_id'),
        sa.ForeignKeyConstraint(['execution_id'], ['modular_executions.id'], ondelete='CASCADE')
    )
    op.create_index('idx_execution_plans_execution_id', 'execution_plans', ['execution_id'], unique=False)
    op.create_index('idx_execution_plans_status', 'execution_plans', ['status'], unique=False)

    # Create plan_steps table
    op.create_table(
        'plan_steps',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('step_id', sa.String(length=255), nullable=False),
        sa.Column('plan_id', sa.String(length=255), nullable=False),
        sa.Column('action', sa.String(length=255), nullable=False),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'in_progress', 'completed', 'failed', 'skipped', name='stepstatus', create_type=False), nullable=False),
        sa.Column('dependencies', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('tool_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('estimated_cost', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('actual_cost', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['execution_plans.plan_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('plan_id', 'step_id', name='uq_plan_step')
    )
    op.create_index('idx_plan_steps_plan_id', 'plan_steps', ['plan_id'], unique=False)
    op.create_index('idx_plan_steps_status', 'plan_steps', ['status'], unique=False)
    op.create_index('idx_plan_steps_step_id', 'plan_steps', ['step_id'], unique=False)

    # Create module_transitions table
    op.create_table(
        'module_transitions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('transition_id', sa.String(length=255), nullable=False, unique=True),
        sa.Column('plan_id', sa.String(length=255), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('from_module', postgresql.ENUM('planner', 'executor', 'verifier', 'generator', name='moduletype', create_type=False), nullable=False),
        sa.Column('to_module', postgresql.ENUM('planner', 'executor', 'verifier', 'generator', name='moduletype', create_type=False), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('trigger', sa.String(length=255), nullable=True),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('duration_in_from_module', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['execution_plans.plan_id'], ondelete='CASCADE')
    )
    op.create_index('idx_module_transitions_plan_id', 'module_transitions', ['plan_id'], unique=False)
    op.create_index('idx_module_transitions_timestamp', 'module_transitions', ['timestamp'], unique=False)
    op.create_index('idx_module_transitions_from_to', 'module_transitions', ['from_module', 'to_module'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('module_transitions')
    op.drop_table('plan_steps')
    op.drop_table('execution_plans')
    op.drop_table('modular_executions')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS moduletype;")
    op.execute("DROP TYPE IF EXISTS stepstatus;")
    op.execute("DROP TYPE IF EXISTS planstatus;")
