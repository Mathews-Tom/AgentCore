"""add_session_snapshots_table

Revision ID: 16ae5b2f867b
Revises: 001
Create Date: 2025-10-01 18:53:49.590927

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '16ae5b2f867b'
down_revision: Union[str, Sequence[str], None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create enum types with checkfirst=True to avoid duplicates
    session_state = postgresql.ENUM(
        'active', 'paused', 'suspended', 'completed', 'failed', 'expired',
        name='sessionstate',
        create_type=False
    )
    session_state.create(op.get_bind(), checkfirst=True)

    session_priority = postgresql.ENUM(
        'low', 'normal', 'high', 'critical',
        name='sessionpriority',
        create_type=False
    )
    session_priority.create(op.get_bind(), checkfirst=True)

    # Create session_snapshots table
    op.create_table(
        'session_snapshots',
        sa.Column('session_id', sa.String(255), primary_key=True, index=True),

        # Session metadata
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('state', sa.Enum('active', 'paused', 'suspended', 'completed', 'failed', 'expired', name='sessionstate'), nullable=False, server_default='active', index=True),
        sa.Column('priority', sa.Enum('low', 'normal', 'high', 'critical', name='sessionpriority'), nullable=False, server_default='normal'),

        # Participants
        sa.Column('owner_agent', sa.String(255), nullable=False, index=True),
        sa.Column('participant_agents', sa.JSON, nullable=False, server_default='[]'),

        # Context and state
        sa.Column('context', sa.JSON, nullable=False, server_default='{}'),

        # Associated resources
        sa.Column('task_ids', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('artifact_ids', sa.JSON, nullable=False, server_default='[]'),

        # Lifecycle tracking
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), index=True),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime, nullable=True, index=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),

        # Session configuration
        sa.Column('timeout_seconds', sa.Integer, nullable=False, server_default='3600'),
        sa.Column('max_idle_seconds', sa.Integer, nullable=False, server_default='300'),

        # Metadata and tags
        sa.Column('tags', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('session_metadata', sa.JSON, nullable=False, server_default='{}'),

        # Checkpointing
        sa.Column('checkpoint_interval_seconds', sa.Integer, nullable=False, server_default='60'),
        sa.Column('last_checkpoint_at', sa.DateTime, nullable=True),
        sa.Column('checkpoint_count', sa.Integer, nullable=False, server_default='0'),
    )

    # Create indexes
    op.create_index('idx_session_state', 'session_snapshots', ['state'])
    op.create_index('idx_session_owner', 'session_snapshots', ['owner_agent'])
    op.create_index('idx_session_created_at', 'session_snapshots', ['created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_session_expires_at', 'session_snapshots', ['expires_at'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_session_expires_at', table_name='session_snapshots')
    op.drop_index('idx_session_created_at', table_name='session_snapshots')
    op.drop_index('idx_session_owner', table_name='session_snapshots')
    op.drop_index('idx_session_state', table_name='session_snapshots')

    # Drop table
    op.drop_table('session_snapshots')

    # Drop enums
    sa.Enum(name='sessionstate').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='sessionpriority').drop(op.get_bind(), checkfirst=True)
