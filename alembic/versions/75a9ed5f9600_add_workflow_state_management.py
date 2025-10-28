"""add_workflow_state_management

Revision ID: 75a9ed5f9600
Revises: 7a170db0b688
Create Date: 2025-10-20 23:35:08.869683

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '75a9ed5f9600'
down_revision: Union[str, Sequence[str], None] = '7a170db0b688'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create WorkflowStatus enum (idempotent)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'workflowstatus') THEN
                CREATE TYPE workflowstatus AS ENUM ('pending', 'planning', 'executing', 'paused', 'completed', 'failed', 'compensating', 'compensated', 'compensation_failed', 'cancelled');
            END IF;
        END$$;
    """)

    # Create workflow_executions table
    op.create_table(
        'workflow_executions',
        sa.Column('execution_id', sa.String(255), primary_key=True),
        sa.Column('workflow_id', sa.String(255), nullable=False, index=True),
        sa.Column('workflow_name', sa.String(255), nullable=False),
        sa.Column('workflow_version', sa.String(50), nullable=False, server_default='1.0'),
        sa.Column('orchestration_pattern', sa.String(50), nullable=False, index=True),
        sa.Column('status', postgresql.ENUM('pending', 'planning', 'executing', 'paused', 'completed', 'failed', 'compensating', 'compensated', 'compensation_failed', 'cancelled', name='workflowstatus', create_type=False), nullable=False, server_default='pending', index=True),
        sa.Column('workflow_definition', JSONB, nullable=False),
        sa.Column('execution_state', JSONB, nullable=False, server_default='{}'),
        sa.Column('allocated_agents', JSONB, nullable=False, server_default='{}'),
        sa.Column('task_states', JSONB, nullable=False, server_default='{}'),
        sa.Column('completed_tasks', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('failed_tasks', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('checkpoint_data', JSONB, nullable=True),
        sa.Column('checkpoint_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('last_checkpoint_at', sa.DateTime, nullable=True),
        sa.Column('coordination_overhead_ms', sa.Integer, nullable=True),
        sa.Column('total_tasks', sa.Integer, nullable=False, server_default='0'),
        sa.Column('completed_task_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('failed_task_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('duration_seconds', sa.Integer, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_stack_trace', sa.Text, nullable=True),
        sa.Column('compensation_errors', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('input_data', JSONB, nullable=True),
        sa.Column('output_data', JSONB, nullable=True),
        sa.Column('tags', JSONB, nullable=False, server_default='[]'),
        sa.Column('workflow_metadata', JSONB, nullable=False, server_default='{}'),
    )

    # Create indexes for workflow_executions (idempotent)
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_status_created ON workflow_executions (status, created_at)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_name_status ON workflow_executions (workflow_name, status)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_pattern ON workflow_executions (orchestration_pattern)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_execution_state ON workflow_executions USING gin (execution_state)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_task_states ON workflow_executions USING gin (task_states)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_metadata ON workflow_executions USING gin (workflow_metadata)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_tags ON workflow_executions USING gin (tags)')
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_workflow_completed_performance
        ON workflow_executions (status, duration_seconds)
        WHERE status IN ('completed', 'failed', 'compensated')
    """)

    # Create workflow_state_history table
    op.create_table(
        'workflow_state_history',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('execution_id', sa.String(255), sa.ForeignKey('workflow_executions.execution_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('version', sa.Integer, nullable=False),
        sa.Column('state_type', sa.String(50), nullable=False, index=True),
        sa.Column('state_snapshot', JSONB, nullable=False),
        sa.Column('changed_fields', sa.JSON, nullable=True),
        sa.Column('change_reason', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('state_metadata', JSONB, nullable=False, server_default='{}'),
    )

    # Create indexes for workflow_state_history (idempotent)
    op.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_state_execution_version ON workflow_state_history (execution_id, version)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_state_type_created ON workflow_state_history (state_type, created_at)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_state_snapshot ON workflow_state_history USING gin (state_snapshot)')

    # Create workflow_state_versions table
    op.create_table(
        'workflow_state_versions',
        sa.Column('version_id', sa.String(255), primary_key=True),
        sa.Column('schema_version', sa.Integer, nullable=False, unique=True, index=True),
        sa.Column('workflow_type', sa.String(50), nullable=False, index=True),
        sa.Column('state_schema', JSONB, nullable=False),
        sa.Column('migration_script', sa.Text, nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('is_active', sa.Integer, nullable=False, server_default='1', index=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('deprecated_at', sa.DateTime, nullable=True),
        sa.Column('applied_to_executions', sa.JSON, nullable=False, server_default='[]'),
    )

    # Create indexes for workflow_state_versions (idempotent)
    op.execute('CREATE INDEX IF NOT EXISTS idx_version_type_active ON workflow_state_versions (workflow_type, is_active)')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order
    op.drop_table('workflow_state_versions')
    op.drop_table('workflow_state_history')
    op.drop_table('workflow_executions')

    # Drop enum type (idempotent)
    op.execute('DROP TYPE IF EXISTS workflowstatus')
