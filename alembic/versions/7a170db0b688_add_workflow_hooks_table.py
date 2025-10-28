"""add_workflow_hooks_table

Revision ID: 7a170db0b688
Revises: d1509698cef3
Create Date: 2025-10-20 23:25:01.014544

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a170db0b688'
down_revision: Union[str, Sequence[str], None] = 'acf3cb34fe9f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create workflow_hooks table
    op.create_table(
        'workflow_hooks',
        sa.Column('hook_id', sa.UUID(), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('trigger', sa.String(length=50), nullable=False),
        sa.Column('command', sa.Text(), nullable=False),
        sa.Column('args', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('always_run', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('timeout_ms', sa.Integer(), nullable=False, server_default='30000'),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('execution_mode', sa.String(length=50), nullable=False, server_default='async'),
        sa.Column('event_filters', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('metadata', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('retry_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('retry_delay_ms', sa.Integer(), nullable=False, server_default='1000'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('hook_id')
    )

    # Create indexes for efficient querying
    op.create_index('idx_hooks_trigger', 'workflow_hooks', ['trigger'], unique=False, postgresql_where=sa.text('enabled = true'))
    op.create_index('idx_hooks_priority', 'workflow_hooks', ['priority'], unique=False)
    op.create_index('idx_hooks_enabled', 'workflow_hooks', ['enabled'], unique=False)

    # Create hook_executions table for tracking
    op.create_table(
        'hook_executions',
        sa.Column('execution_id', sa.UUID(), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('hook_id', sa.UUID(), nullable=False),
        sa.Column('trigger', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('input_data', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_traceback', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_retry', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('workflow_id', sa.String(length=255), nullable=True),
        sa.Column('task_id', sa.String(length=255), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('execution_id'),
        sa.ForeignKeyConstraint(['hook_id'], ['workflow_hooks.hook_id'], ondelete='CASCADE')
    )

    # Create indexes for hook_executions
    op.create_index('idx_hook_executions_hook_id', 'hook_executions', ['hook_id'], unique=False)
    op.create_index('idx_hook_executions_status', 'hook_executions', ['status'], unique=False)
    op.create_index('idx_hook_executions_started_at', 'hook_executions', ['started_at'], unique=False)
    op.create_index('idx_hook_executions_workflow_id', 'hook_executions', ['workflow_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index('idx_hook_executions_workflow_id', table_name='hook_executions')
    op.drop_index('idx_hook_executions_started_at', table_name='hook_executions')
    op.drop_index('idx_hook_executions_status', table_name='hook_executions')
    op.drop_index('idx_hook_executions_hook_id', table_name='hook_executions')

    op.drop_index('idx_hooks_enabled', table_name='workflow_hooks')
    op.drop_index('idx_hooks_priority', table_name='workflow_hooks')
    op.drop_index('idx_hooks_trigger', table_name='workflow_hooks')

    # Drop tables
    op.drop_table('hook_executions')
    op.drop_table('workflow_hooks')
