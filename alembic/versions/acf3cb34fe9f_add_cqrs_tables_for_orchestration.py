"""add cqrs tables for orchestration

Revision ID: acf3cb34fe9f
Revises: 068b96d43e02
Create Date: 2025-10-20 04:39:07.707060

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'acf3cb34fe9f'
down_revision: Union[str, Sequence[str], None] = '068b96d43e02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create orchestration_events table
    op.create_table(
        'orchestration_events',
        sa.Column('event_id', sa.String(length=36), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('aggregate_id', sa.String(length=36), nullable=False),
        sa.Column('aggregate_type', sa.String(length=50), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('event_data', sa.Text(), nullable=False),
        sa.Column('event_metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('event_id'),
    )
    # Create indexes with IF NOT EXISTS for idempotence
    op.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_aggregate_version ON orchestration_events (aggregate_id, version)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_event_timestamp ON orchestration_events (timestamp)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_event_type_timestamp ON orchestration_events (event_type, timestamp)')
    op.create_index(op.f('ix_orchestration_events_aggregate_id'), 'orchestration_events', ['aggregate_id'])
    op.create_index(op.f('ix_orchestration_events_aggregate_type'), 'orchestration_events', ['aggregate_type'])
    op.create_index(op.f('ix_orchestration_events_event_type'), 'orchestration_events', ['event_type'])

    # Create orchestration_snapshots table
    op.create_table(
        'orchestration_snapshots',
        sa.Column('aggregate_id', sa.String(length=36), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('aggregate_type', sa.String(length=50), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('snapshot_data', sa.Text(), nullable=False),
        sa.Column('snapshot_metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('aggregate_id', 'version'),
    )
    op.execute('CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp ON orchestration_snapshots (timestamp)')
    op.create_index(op.f('ix_orchestration_snapshots_aggregate_type'), 'orchestration_snapshots', ['aggregate_type'])

    # Create workflow_read_model table
    op.create_table(
        'workflow_read_model',
        sa.Column('workflow_id', sa.String(length=36), nullable=False),
        sa.Column('workflow_name', sa.String(length=255), nullable=False),
        sa.Column('orchestration_pattern', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('agent_requirements', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('task_definitions', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('workflow_config', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('total_executions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('successful_executions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_executions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('average_execution_time_ms', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('workflow_id'),
    )
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_status_created ON workflow_read_model (status, created_at)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_workflow_pattern_status ON workflow_read_model (orchestration_pattern, status)')
    op.create_index(op.f('ix_workflow_read_model_created_by'), 'workflow_read_model', ['created_by'])
    op.create_index(op.f('ix_workflow_read_model_orchestration_pattern'), 'workflow_read_model', ['orchestration_pattern'])
    op.create_index(op.f('ix_workflow_read_model_status'), 'workflow_read_model', ['status'])
    op.create_index(op.f('ix_workflow_read_model_workflow_name'), 'workflow_read_model', ['workflow_name'])

    # Create execution_read_model table
    op.create_table(
        'execution_read_model',
        sa.Column('execution_id', sa.String(length=36), nullable=False),
        sa.Column('workflow_id', sa.String(length=36), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('input_data', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('output_data', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('execution_options', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('started_by', sa.String(length=255), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('paused_at', sa.DateTime(), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('tasks_total', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_failed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_pending', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_type', sa.String(length=255), nullable=True),
        sa.Column('failed_task_id', sa.String(length=36), nullable=True),
        sa.PrimaryKeyConstraint('execution_id'),
    )
    op.execute('CREATE INDEX IF NOT EXISTS idx_execution_workflow_status ON execution_read_model (workflow_id, status)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_execution_started_at ON execution_read_model (started_at)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_execution_workflow_started ON execution_read_model (workflow_id, started_at)')
    op.create_index(op.f('ix_execution_read_model_status'), 'execution_read_model', ['status'])
    op.create_index(op.f('ix_execution_read_model_workflow_id'), 'execution_read_model', ['workflow_id'])

    # Create agent_assignment_read_model table
    op.create_table(
        'agent_assignment_read_model',
        sa.Column('workflow_id', sa.String(length=36), nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('agent_role', sa.String(length=100), nullable=False),
        sa.Column('capabilities', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.Column('unassigned_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('tasks_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_failed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('average_task_time_ms', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('workflow_id', 'agent_id', 'agent_role'),
    )
    op.execute('CREATE INDEX IF NOT EXISTS idx_assignment_agent_active ON agent_assignment_read_model (agent_id, is_active)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_assignment_workflow_active ON agent_assignment_read_model (workflow_id, is_active)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_assignment_role ON agent_assignment_read_model (agent_role)')
    op.create_index(op.f('ix_agent_assignment_read_model_is_active'), 'agent_assignment_read_model', ['is_active'])

    # Create task_read_model table
    op.create_table(
        'task_read_model',
        sa.Column('task_id', sa.String(length=36), nullable=False),
        sa.Column('workflow_id', sa.String(length=36), nullable=False),
        sa.Column('execution_id', sa.String(length=36), nullable=True),
        sa.Column('task_type', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=True),
        sa.Column('input_data', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('output_data', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('dependencies', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('failed_at', sa.DateTime(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_type', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('task_id'),
    )
    # Create indexes with IF NOT EXISTS for idempotence
    op.execute('CREATE INDEX IF NOT EXISTS idx_task_workflow_status ON task_read_model (workflow_id, status)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_task_execution_status ON task_read_model (execution_id, status)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_task_agent_status ON task_read_model (agent_id, status)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_task_started_at ON task_read_model (started_at)')
    op.create_index(op.f('ix_task_read_model_agent_id'), 'task_read_model', ['agent_id'])
    op.create_index(op.f('ix_task_read_model_execution_id'), 'task_read_model', ['execution_id'])
    op.create_index(op.f('ix_task_read_model_status'), 'task_read_model', ['status'])
    op.create_index(op.f('ix_task_read_model_task_type'), 'task_read_model', ['task_type'])
    op.create_index(op.f('ix_task_read_model_workflow_id'), 'task_read_model', ['workflow_id'])

    # Create workflow_metrics_read_model table
    op.create_table(
        'workflow_metrics_read_model',
        sa.Column('workflow_id', sa.String(length=36), nullable=False),
        sa.Column('execution_id', sa.String(length=36), nullable=False),
        sa.Column('total_execution_time_ms', sa.Integer(), nullable=False),
        sa.Column('coordination_overhead_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('average_task_time_ms', sa.Float(), nullable=True),
        sa.Column('agents_allocated', sa.Integer(), nullable=False),
        sa.Column('max_concurrent_tasks', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_tasks', sa.Integer(), nullable=False),
        sa.Column('tasks_completed_successfully', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_failed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tasks_retried', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column('retry_rate', sa.Float(), nullable=True),
        sa.Column('throughput_tasks_per_second', sa.Float(), nullable=True),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('workflow_id', 'execution_id'),
    )
    op.execute('CREATE INDEX IF NOT EXISTS idx_metrics_workflow ON workflow_metrics_read_model (workflow_id, computed_at)')
    op.create_index(op.f('ix_workflow_metrics_read_model_computed_at'), 'workflow_metrics_read_model', ['computed_at'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order
    op.drop_table('workflow_metrics_read_model')
    op.drop_table('task_read_model')
    op.drop_table('agent_assignment_read_model')
    op.drop_table('execution_read_model')
    op.drop_table('workflow_read_model')
    op.drop_table('orchestration_snapshots')
    op.drop_table('orchestration_events')
