"""Add memory system tables

Revision ID: f9c3a8b2e1d4
Revises: 0a8dea1094f6
Create Date: 2025-11-14

Implements hybrid database migration for memory system (MEM-004):
- memories: Main memory storage with pgvector embeddings
- stage_memories: COMPASS stage compression summaries
- task_contexts: Progressive task context tracking
- error_records: Error tracking with pattern detection
- compression_metrics: Cost tracking for compression operations

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f9c3a8b2e1d4'
down_revision: Union[str, None] = '0a8dea1094f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create memory system tables with pgvector support."""

    # Enable pgvector extension if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create enum types for memory system
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memorylayer') THEN
                CREATE TYPE memorylayer AS ENUM ('working', 'episodic', 'semantic', 'procedural');
            END IF;
        END$$;
    """)

    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'stagetype') THEN
                CREATE TYPE stagetype AS ENUM ('planning', 'execution', 'reflection', 'verification');
            END IF;
        END$$;
    """)

    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'errortype') THEN
                CREATE TYPE errortype AS ENUM ('hallucination', 'missing_info', 'incorrect_action', 'context_degradation');
            END IF;
        END$$;
    """)

    # Create memories table (all layers except working which uses Redis)
    op.create_table(
        'memories',
        sa.Column('memory_id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('memory_layer', postgresql.ENUM('working', 'episodic', 'semantic', 'procedural', name='memorylayer', create_type=False), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),  # Will be VECTOR type for pgvector

        # Scope fields
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True),

        # Metadata
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('entities', postgresql.ARRAY(sa.Text()), nullable=False, server_default='{}'),
        sa.Column('facts', postgresql.ARRAY(sa.Text()), nullable=False, server_default='{}'),
        sa.Column('keywords', postgresql.ARRAY(sa.Text()), nullable=False, server_default='{}'),

        # Relationships
        sa.Column('related_memory_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default='{}'),
        sa.Column('parent_memory_id', postgresql.UUID(as_uuid=True), nullable=True),

        # Tracking
        sa.Column('relevance_score', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('access_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),

        # COMPASS enhancements
        sa.Column('stage_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_critical', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('criticality_reason', sa.Text(), nullable=True),

        # Procedural memory fields
        sa.Column('actions', postgresql.ARRAY(sa.Text()), nullable=False, server_default='{}'),
        sa.Column('outcome', sa.Text(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True),

        # Audit
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),

        sa.PrimaryKeyConstraint('memory_id')
    )

    # Create indexes for memories table
    op.create_index('idx_memories_layer', 'memories', ['memory_layer'])
    op.create_index('idx_memories_agent', 'memories', ['agent_id'], postgresql_where=sa.text('agent_id IS NOT NULL'))
    op.create_index('idx_memories_session', 'memories', ['session_id'], postgresql_where=sa.text('session_id IS NOT NULL'))
    op.create_index('idx_memories_task', 'memories', ['task_id'], postgresql_where=sa.text('task_id IS NOT NULL'))
    op.create_index('idx_memories_stage', 'memories', ['stage_id'], postgresql_where=sa.text('stage_id IS NOT NULL'))
    op.create_index('idx_memories_critical', 'memories', ['is_critical'], postgresql_where=sa.text('is_critical = true'))
    op.create_index('idx_memories_timestamp', 'memories', ['timestamp'])

    # Composite indexes for efficient queries
    op.create_index('idx_memories_agent_session', 'memories', ['agent_id', 'session_id'])
    op.create_index('idx_memories_task_stage', 'memories', ['task_id', 'stage_id'])
    op.create_index('idx_memories_layer_agent', 'memories', ['memory_layer', 'agent_id'])

    # Note: Vector index will be created after data is loaded for better performance
    # Manual creation: CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

    # Create stage_memories table (COMPASS hierarchical organization)
    op.create_table(
        'stage_memories',
        sa.Column('stage_id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stage_type', postgresql.ENUM('planning', 'execution', 'reflection', 'verification', name='stagetype', create_type=False), nullable=False),
        sa.Column('stage_summary', sa.Text(), nullable=False),
        sa.Column('stage_insights', postgresql.ARRAY(sa.Text()), nullable=False, server_default='{}'),
        sa.Column('raw_memory_refs', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('compression_ratio', sa.Float(), nullable=True),
        sa.Column('compression_model', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),

        sa.PrimaryKeyConstraint('stage_id')
    )

    # Create indexes for stage_memories table
    op.create_index('idx_stage_memories_task', 'stage_memories', ['task_id'])
    op.create_index('idx_stage_memories_type', 'stage_memories', ['stage_type'])
    op.create_index('idx_stage_memories_agent', 'stage_memories', ['agent_id'])
    op.create_index('idx_stage_memories_task_type', 'stage_memories', ['task_id', 'stage_type'])

    # Create task_contexts table (COMPASS progressive task summarization)
    op.create_table(
        'task_contexts',
        sa.Column('task_id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_goal', sa.Text(), nullable=False),
        sa.Column('current_stage_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('task_progress_summary', sa.Text(), nullable=True),
        sa.Column('critical_constraints', postgresql.ARRAY(sa.Text()), nullable=False, server_default='{}'),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),

        sa.PrimaryKeyConstraint('task_id')
    )

    # Create indexes for task_contexts table
    op.create_index('idx_task_contexts_agent', 'task_contexts', ['agent_id'])
    op.create_index('idx_task_contexts_current_stage', 'task_contexts', ['current_stage_id'], postgresql_where=sa.text('current_stage_id IS NOT NULL'))

    # Create error_records table (COMPASS error tracking)
    op.create_table(
        'error_records',
        sa.Column('error_id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stage_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('error_type', postgresql.ENUM('hallucination', 'missing_info', 'incorrect_action', 'context_degradation', name='errortype', create_type=False), nullable=False),
        sa.Column('error_description', sa.Text(), nullable=False),
        sa.Column('context_when_occurred', sa.Text(), nullable=True),
        sa.Column('recovery_action', sa.Text(), nullable=True),
        sa.Column('error_severity', sa.Float(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),

        sa.PrimaryKeyConstraint('error_id'),
        sa.CheckConstraint('error_severity >= 0.0 AND error_severity <= 1.0', name='error_severity_range')
    )

    # Create indexes for error_records table
    op.create_index('idx_error_records_task', 'error_records', ['task_id'])
    op.create_index('idx_error_records_stage', 'error_records', ['stage_id'], postgresql_where=sa.text('stage_id IS NOT NULL'))
    op.create_index('idx_error_records_type', 'error_records', ['error_type'])
    op.create_index('idx_error_records_agent', 'error_records', ['agent_id'])
    op.create_index('idx_error_records_task_type', 'error_records', ['task_id', 'error_type'])
    op.create_index('idx_error_records_recorded_at', 'error_records', ['recorded_at'])

    # Create compression_metrics table (COMPASS cost tracking)
    op.create_table(
        'compression_metrics',
        sa.Column('metric_id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('stage_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('compression_type', sa.String(50), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('compression_ratio', sa.Float(), nullable=False),
        sa.Column('critical_fact_retention_rate', sa.Float(), nullable=True),
        sa.Column('coherence_score', sa.Float(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(10, 4), nullable=False),
        sa.Column('model_used', sa.String(100), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),

        sa.PrimaryKeyConstraint('metric_id')
    )

    # Create indexes for compression_metrics table
    op.create_index('idx_compression_metrics_task', 'compression_metrics', ['task_id'], postgresql_where=sa.text('task_id IS NOT NULL'))
    op.create_index('idx_compression_metrics_stage', 'compression_metrics', ['stage_id'], postgresql_where=sa.text('stage_id IS NOT NULL'))
    op.create_index('idx_compression_metrics_type', 'compression_metrics', ['compression_type'])
    op.create_index('idx_compression_metrics_recorded_at', 'compression_metrics', ['recorded_at'])
    op.create_index('idx_compression_metrics_model', 'compression_metrics', ['model_used'])


def downgrade() -> None:
    """Drop memory system tables and enums."""

    # Drop tables in reverse order
    op.drop_table('compression_metrics')
    op.drop_table('error_records')
    op.drop_table('task_contexts')
    op.drop_table('stage_memories')
    op.drop_table('memories')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS errortype")
    op.execute("DROP TYPE IF EXISTS stagetype")
    op.execute("DROP TYPE IF EXISTS memorylayer")

    # Note: Not dropping pgvector extension as other tables may use it
