"""enhance_tool_executions_schema

Revision ID: 0a8dea1094f6
Revises: b64aba14d716
Create Date: 2025-11-12 19:51:26.466642

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0a8dea1094f6'
down_revision: Union[str, Sequence[str], None] = 'b64aba14d716'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to enhance tool_executions table.

    Adds:
    - user_id column extracted from execution_context for direct queries
    - trace_id column for distributed tracing
    - success boolean column (derived from status) for error analysis
    - Additional indexes for fast lookups per TOOL-005 requirements:
      * B-tree index on user_id
      * B-tree index on trace_id
      * Composite index on (tool_id, user_id) for user-specific queries
      * Partial index on (success = false) for error analysis
    """
    # Add new columns
    op.add_column('tool_executions', sa.Column('user_id', sa.String(length=255), nullable=True))
    op.add_column('tool_executions', sa.Column('trace_id', sa.String(length=255), nullable=True))
    op.add_column('tool_executions', sa.Column('success', sa.Boolean(), nullable=False, server_default='false'))

    # Populate success column from status
    op.execute("""
        UPDATE tool_executions
        SET success = CASE WHEN status = 'success' THEN true ELSE false END
    """)

    # Create B-tree indexes per TOOL-005 requirements
    op.create_index('idx_tool_executions_user_id', 'tool_executions', ['user_id'], unique=False)
    op.create_index('idx_tool_executions_trace_id', 'tool_executions', ['trace_id'], unique=False)

    # Composite index for user-specific queries
    op.create_index('idx_tool_executions_tool_user', 'tool_executions', ['tool_id', 'user_id'], unique=False)

    # Partial index for error analysis (success = false)
    op.execute("""
        CREATE INDEX idx_tool_executions_failures
        ON tool_executions(tool_id, created_at DESC)
        WHERE success = false
    """)


def downgrade() -> None:
    """Downgrade schema by removing enhancements."""
    # Drop indexes
    op.drop_index('idx_tool_executions_failures', table_name='tool_executions')
    op.drop_index('idx_tool_executions_tool_user', table_name='tool_executions')
    op.drop_index('idx_tool_executions_trace_id', table_name='tool_executions')
    op.drop_index('idx_tool_executions_user_id', table_name='tool_executions')

    # Drop columns
    op.drop_column('tool_executions', 'success')
    op.drop_column('tool_executions', 'trace_id')
    op.drop_column('tool_executions', 'user_id')
