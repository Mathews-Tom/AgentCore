"""Initial A2A protocol schema

Revision ID: 001
Revises:
Create Date: 2025-09-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### Create enum types first (idempotent) ###
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agentstatus') THEN
                CREATE TYPE agentstatus AS ENUM ('active', 'inactive', 'maintenance', 'error');
            END IF;
        END$$;
    """)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'taskstatus') THEN
                CREATE TYPE taskstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
            END IF;
        END$$;
    """)

    # ### Create agents table ###
    op.create_table(
        'agents',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('status', postgresql.ENUM('active', 'inactive', 'maintenance', 'error', name='agentstatus', create_type=False), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('capabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('agent_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('endpoint', sa.String(length=512), nullable=True),
        sa.Column('current_load', sa.Integer(), nullable=False),
        sa.Column('max_load', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_seen', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_agent_capabilities', 'agents', ['capabilities'], unique=False, postgresql_using='gin')
    op.create_index('idx_agent_status_load', 'agents', ['status', 'current_load'], unique=False)
    op.create_index(op.f('ix_agents_id'), 'agents', ['id'], unique=False)
    op.create_index(op.f('ix_agents_status'), 'agents', ['status'], unique=False)

    # ### Create tasks table ###
    op.create_table(
        'tasks',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', postgresql.ENUM('pending', 'running', 'completed', 'failed', 'cancelled', name='taskstatus', create_type=False), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('required_capabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('assigned_agent_id', sa.String(length=255), nullable=True),
        sa.Column('depends_on', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('task_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['assigned_agent_id'], ['agents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_task_agent_status', 'tasks', ['assigned_agent_id', 'status'], unique=False)
    op.create_index('idx_task_status_priority', 'tasks', ['status', 'priority'], unique=False)
    op.create_index(op.f('ix_tasks_assigned_agent_id'), 'tasks', ['assigned_agent_id'], unique=False)
    op.create_index(op.f('ix_tasks_created_at'), 'tasks', ['created_at'], unique=False)
    op.create_index(op.f('ix_tasks_id'), 'tasks', ['id'], unique=False)
    op.create_index(op.f('ix_tasks_priority'), 'tasks', ['priority'], unique=False)
    op.create_index(op.f('ix_tasks_status'), 'tasks', ['status'], unique=False)

    # ### Create agent_health_metrics table ###
    op.create_table(
        'agent_health_metrics',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('is_healthy', sa.Boolean(), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('consecutive_failures', sa.Integer(), nullable=False),
        sa.Column('cpu_percent', sa.Float(), nullable=True),
        sa.Column('memory_mb', sa.Float(), nullable=True),
        sa.Column('checked_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_health_agent_time', 'agent_health_metrics', ['agent_id', 'checked_at'], unique=False)
    op.create_index('idx_health_status', 'agent_health_metrics', ['is_healthy', 'checked_at'], unique=False)
    op.create_index(op.f('ix_agent_health_metrics_agent_id'), 'agent_health_metrics', ['agent_id'], unique=False)
    op.create_index(op.f('ix_agent_health_metrics_checked_at'), 'agent_health_metrics', ['checked_at'], unique=False)

    # ### Create message_queue table ###
    op.create_table(
        'message_queue',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('target_agent_id', sa.String(length=255), nullable=False),
        sa.Column('required_capabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('message_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('ttl_seconds', sa.Integer(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_queue_expires', 'message_queue', ['expires_at'], unique=False)
    op.create_index('idx_queue_target_priority', 'message_queue', ['target_agent_id', 'priority'], unique=False)
    op.create_index(op.f('ix_message_queue_created_at'), 'message_queue', ['created_at'], unique=False)
    op.create_index(op.f('ix_message_queue_expires_at'), 'message_queue', ['expires_at'], unique=False)
    op.create_index(op.f('ix_message_queue_id'), 'message_queue', ['id'], unique=False)
    op.create_index(op.f('ix_message_queue_target_agent_id'), 'message_queue', ['target_agent_id'], unique=False)

    # ### Create event_subscriptions table ###
    op.create_table(
        'event_subscriptions',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('subscriber_id', sa.String(length=255), nullable=False),
        sa.Column('event_types', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('filters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_subscription_expires', 'event_subscriptions', ['expires_at'], unique=False)
    op.create_index('idx_subscription_subscriber_active', 'event_subscriptions', ['subscriber_id', 'is_active'], unique=False)
    op.create_index(op.f('ix_event_subscriptions_expires_at'), 'event_subscriptions', ['expires_at'], unique=False)
    op.create_index(op.f('ix_event_subscriptions_id'), 'event_subscriptions', ['id'], unique=False)
    op.create_index(op.f('ix_event_subscriptions_is_active'), 'event_subscriptions', ['is_active'], unique=False)
    op.create_index(op.f('ix_event_subscriptions_subscriber_id'), 'event_subscriptions', ['subscriber_id'], unique=False)

    # ### Create security_tokens table ###
    op.create_table(
        'security_tokens',
        sa.Column('jti', sa.String(length=255), nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('token_type', sa.String(length=50), nullable=False),
        sa.Column('issued_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revocation_reason', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('jti')
    )
    op.create_index('idx_token_agent_active', 'security_tokens', ['agent_id', 'is_revoked'], unique=False)
    op.create_index('idx_token_expires', 'security_tokens', ['expires_at'], unique=False)
    op.create_index(op.f('ix_security_tokens_agent_id'), 'security_tokens', ['agent_id'], unique=False)
    op.create_index(op.f('ix_security_tokens_expires_at'), 'security_tokens', ['expires_at'], unique=False)
    op.create_index(op.f('ix_security_tokens_is_revoked'), 'security_tokens', ['is_revoked'], unique=False)
    op.create_index(op.f('ix_security_tokens_issued_at'), 'security_tokens', ['issued_at'], unique=False)
    op.create_index(op.f('ix_security_tokens_jti'), 'security_tokens', ['jti'], unique=False)

    # ### Create rate_limits table ###
    op.create_table(
        'rate_limits',
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('request_count', sa.Integer(), nullable=False),
        sa.Column('window_start', sa.DateTime(), nullable=False),
        sa.Column('max_requests', sa.Integer(), nullable=False),
        sa.Column('window_seconds', sa.Integer(), nullable=False),
        sa.Column('total_violations', sa.Integer(), nullable=False),
        sa.Column('last_violation', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('agent_id')
    )
    op.create_index(op.f('ix_rate_limits_agent_id'), 'rate_limits', ['agent_id'], unique=False)
    op.create_index(op.f('ix_rate_limits_window_start'), 'rate_limits', ['window_start'], unique=False)

    # ### Create agent_public_keys table ###
    op.create_table(
        'agent_public_keys',
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('public_key_pem', sa.Text(), nullable=False),
        sa.Column('algorithm', sa.String(length=50), nullable=False),
        sa.Column('registered_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('replaced_by', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('agent_id')
    )
    op.create_index('idx_public_key_active', 'agent_public_keys', ['agent_id', 'is_active'], unique=False)
    op.create_index(op.f('ix_agent_public_keys_agent_id'), 'agent_public_keys', ['agent_id'], unique=False)
    op.create_index(op.f('ix_agent_public_keys_is_active'), 'agent_public_keys', ['is_active'], unique=False)


def downgrade() -> None:
    # ### Drop all tables ###
    op.drop_index(op.f('ix_agent_public_keys_is_active'), table_name='agent_public_keys')
    op.drop_index(op.f('ix_agent_public_keys_agent_id'), table_name='agent_public_keys')
    op.drop_index('idx_public_key_active', table_name='agent_public_keys')
    op.drop_table('agent_public_keys')

    op.drop_index(op.f('ix_rate_limits_window_start'), table_name='rate_limits')
    op.drop_index(op.f('ix_rate_limits_agent_id'), table_name='rate_limits')
    op.drop_table('rate_limits')

    op.drop_index(op.f('ix_security_tokens_jti'), table_name='security_tokens')
    op.drop_index(op.f('ix_security_tokens_issued_at'), table_name='security_tokens')
    op.drop_index(op.f('ix_security_tokens_is_revoked'), table_name='security_tokens')
    op.drop_index(op.f('ix_security_tokens_expires_at'), table_name='security_tokens')
    op.drop_index(op.f('ix_security_tokens_agent_id'), table_name='security_tokens')
    op.drop_index('idx_token_expires', table_name='security_tokens')
    op.drop_index('idx_token_agent_active', table_name='security_tokens')
    op.drop_table('security_tokens')

    op.drop_index(op.f('ix_event_subscriptions_subscriber_id'), table_name='event_subscriptions')
    op.drop_index(op.f('ix_event_subscriptions_is_active'), table_name='event_subscriptions')
    op.drop_index(op.f('ix_event_subscriptions_id'), table_name='event_subscriptions')
    op.drop_index(op.f('ix_event_subscriptions_expires_at'), table_name='event_subscriptions')
    op.drop_index('idx_subscription_subscriber_active', table_name='event_subscriptions')
    op.drop_index('idx_subscription_expires', table_name='event_subscriptions')
    op.drop_table('event_subscriptions')

    op.drop_index(op.f('ix_message_queue_target_agent_id'), table_name='message_queue')
    op.drop_index(op.f('ix_message_queue_id'), table_name='message_queue')
    op.drop_index(op.f('ix_message_queue_expires_at'), table_name='message_queue')
    op.drop_index(op.f('ix_message_queue_created_at'), table_name='message_queue')
    op.drop_index('idx_queue_target_priority', table_name='message_queue')
    op.drop_index('idx_queue_expires', table_name='message_queue')
    op.drop_table('message_queue')

    op.drop_index(op.f('ix_agent_health_metrics_checked_at'), table_name='agent_health_metrics')
    op.drop_index(op.f('ix_agent_health_metrics_agent_id'), table_name='agent_health_metrics')
    op.drop_index('idx_health_status', table_name='agent_health_metrics')
    op.drop_index('idx_health_agent_time', table_name='agent_health_metrics')
    op.drop_table('agent_health_metrics')

    op.drop_index(op.f('ix_tasks_status'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_priority'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_id'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_created_at'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_assigned_agent_id'), table_name='tasks')
    op.drop_index('idx_task_status_priority', table_name='tasks')
    op.drop_index('idx_task_agent_status', table_name='tasks')
    op.drop_table('tasks')

    op.drop_index(op.f('ix_agents_status'), table_name='agents')
    op.drop_index(op.f('ix_agents_id'), table_name='agents')
    op.drop_index('idx_agent_status_load', table_name='agents')
    op.drop_index('idx_agent_capabilities', table_name='agents')
    op.drop_table('agents')

    # ### Drop enum types ###
    op.execute("DROP TYPE IF EXISTS taskstatus")
    op.execute("DROP TYPE IF EXISTS agentstatus")