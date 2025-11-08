"""add_performance_metrics_table_compass_ace1

Revision ID: c4e8f7d2a3b1
Revises: 2b034c2a4021
Create Date: 2025-11-08 14:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c4e8f7d2a3b1'
down_revision: Union[str, Sequence[str], None] = '2b034c2a4021'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create performance_metrics table (COMPASS ACE-1)
    op.create_table(
        'performance_metrics',
        sa.Column('metric_id', postgresql.UUID(as_uuid=True),
                  server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('stage', sa.String(50), nullable=False),

        # Stage-specific metrics
        sa.Column('stage_success_rate', sa.Float(), nullable=False),
        sa.Column('stage_error_rate', sa.Float(), nullable=False),
        sa.Column('stage_duration_ms', sa.Integer(), nullable=False),
        sa.Column('stage_action_count', sa.Integer(), nullable=False),

        # Cross-stage metrics
        sa.Column('overall_progress_velocity', sa.Float(), nullable=False),
        sa.Column('error_accumulation_rate', sa.Float(), nullable=False),
        sa.Column('context_staleness_score', sa.Float(), nullable=False),
        sa.Column('intervention_effectiveness', sa.Float(), nullable=True),

        # Baseline comparison
        sa.Column('baseline_delta', postgresql.JSONB(),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),

        sa.Column('recorded_at', sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text('NOW()')),

        sa.PrimaryKeyConstraint('metric_id'),

        # Check constraints
        sa.CheckConstraint(
            'stage_success_rate >= 0.0 AND stage_success_rate <= 1.0',
            name='check_success_rate_range'
        ),
        sa.CheckConstraint(
            'stage_error_rate >= 0.0 AND stage_error_rate <= 1.0',
            name='check_error_rate_range'
        ),
        sa.CheckConstraint(
            'context_staleness_score >= 0.0 AND context_staleness_score <= 1.0',
            name='check_staleness_range'
        ),
        sa.CheckConstraint(
            'intervention_effectiveness IS NULL OR '
            '(intervention_effectiveness >= 0.0 AND intervention_effectiveness <= 1.0)',
            name='check_intervention_effectiveness_range'
        ),
        sa.CheckConstraint(
            "stage IN ('planning', 'execution', 'reflection', 'verification')",
            name='check_stage_values'
        ),
    )

    # Create indexes
    op.create_index('idx_metrics_task', 'performance_metrics', ['task_id'])
    op.create_index('idx_metrics_agent', 'performance_metrics', ['agent_id'])
    op.create_index('idx_metrics_stage', 'performance_metrics', ['stage'])
    op.create_index('idx_metrics_recorded', 'performance_metrics', ['recorded_at'],
                    postgresql_ops={'recorded_at': 'DESC'})
    op.create_index('idx_metrics_agent_stage', 'performance_metrics',
                    ['agent_id', 'stage', 'recorded_at'],
                    postgresql_ops={'recorded_at': 'DESC'})


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_metrics_agent_stage', table_name='performance_metrics')
    op.drop_index('idx_metrics_recorded', table_name='performance_metrics')
    op.drop_index('idx_metrics_stage', table_name='performance_metrics')
    op.drop_index('idx_metrics_agent', table_name='performance_metrics')
    op.drop_index('idx_metrics_task', table_name='performance_metrics')

    # Drop table
    op.drop_table('performance_metrics')
