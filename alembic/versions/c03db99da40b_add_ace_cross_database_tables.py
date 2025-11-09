"""add_ace_cross_database_tables

Revision ID: c03db99da40b
Revises: 83591e23ca42
Create Date: 2025-11-09 03:27:11.879859

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c03db99da40b'
down_revision: Union[str, Sequence[str], None] = '83591e23ca42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - create ACE tables with cross-database compatibility."""

    # Create context_playbooks table
    op.create_table(
        'context_playbooks',
        sa.Column('playbook_id', sa.String(36), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('context', sa.JSON(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('playbook_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE')
    )
    op.create_index('idx_playbooks_agent', 'context_playbooks', ['agent_id'])
    op.create_index('idx_playbooks_updated', 'context_playbooks', ['updated_at'])

    # Create context_deltas table
    op.create_table(
        'context_deltas',
        sa.Column('delta_id', sa.String(36), nullable=False),
        sa.Column('playbook_id', sa.String(36), nullable=False),
        sa.Column('changes', sa.JSON(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('applied', sa.Boolean(), nullable=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('delta_id'),
        sa.ForeignKeyConstraint(['playbook_id'], ['context_playbooks.playbook_id'], ondelete='CASCADE'),
        sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='check_confidence_range')
    )
    op.create_index('idx_deltas_playbook', 'context_deltas', ['playbook_id'])
    op.create_index('idx_deltas_confidence', 'context_deltas', ['confidence'])
    op.create_index('idx_deltas_applied', 'context_deltas', ['applied', 'applied_at'])

    # Create execution_traces table
    op.create_table(
        'execution_traces',
        sa.Column('trace_id', sa.String(36), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('task_id', sa.String(255), nullable=True),
        sa.Column('execution_time', sa.Float(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('output_quality', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('trace_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.CheckConstraint(
            'output_quality IS NULL OR (output_quality >= 0.0 AND output_quality <= 1.0)',
            name='check_output_quality_range'
        )
    )
    op.create_index('idx_traces_agent', 'execution_traces', ['agent_id', 'captured_at'])
    op.create_index('idx_traces_success', 'execution_traces', ['success', 'captured_at'])

    # Create evolution_status table
    op.create_table(
        'evolution_status',
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('last_evolution', sa.DateTime(timezone=True), nullable=True),
        sa.Column('pending_traces', sa.Integer(), nullable=False),
        sa.Column('deltas_generated', sa.Integer(), nullable=False),
        sa.Column('deltas_applied', sa.Integer(), nullable=False),
        sa.Column('total_cost', sa.Float(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.PrimaryKeyConstraint('agent_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.CheckConstraint("status IN ('idle', 'processing', 'failed')", name='check_status_values')
    )
    op.create_index('idx_evolution_status', 'evolution_status', ['status', 'last_evolution'])

    # Create performance_metrics table (COMPASS ACE-1)
    op.create_table(
        'performance_metrics',
        sa.Column('metric_id', sa.String(36), nullable=False),
        sa.Column('task_id', sa.String(36), nullable=False),
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
        sa.Column('baseline_delta', sa.JSON(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
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
    # Create indexes for performance_metrics
    op.create_index('idx_metrics_task', 'performance_metrics', ['task_id'])
    op.create_index('idx_metrics_agent', 'performance_metrics', ['agent_id'])
    op.create_index('idx_metrics_stage', 'performance_metrics', ['stage'])
    op.create_index('idx_metrics_recorded', 'performance_metrics', ['recorded_at'])
    op.create_index('idx_metrics_agent_stage', 'performance_metrics', ['agent_id', 'stage', 'recorded_at'])

    # Create intervention_records table (COMPASS ACE-2)
    op.create_table(
        'intervention_records',
        sa.Column('intervention_id', sa.String(36), nullable=False),
        sa.Column('task_id', sa.String(36), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        # Trigger information
        sa.Column('trigger_type', sa.String(50), nullable=False),
        sa.Column('trigger_signals', sa.JSON(), nullable=False),
        sa.Column('trigger_metric_id', sa.String(36), nullable=True),
        # Decision information
        sa.Column('intervention_type', sa.String(50), nullable=False),
        sa.Column('intervention_rationale', sa.Text(), nullable=False),
        sa.Column('decision_confidence', sa.Float(), nullable=False),
        # Execution information
        sa.Column('executed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('execution_duration_ms', sa.Integer(), nullable=False),
        sa.Column('execution_status', sa.String(20), nullable=False),
        sa.Column('execution_error', sa.Text(), nullable=True),
        # Outcome tracking
        sa.Column('pre_metric_id', sa.String(36), nullable=True),
        sa.Column('post_metric_id', sa.String(36), nullable=True),
        sa.Column('effectiveness_delta', sa.Float(), nullable=True),
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('intervention_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['trigger_metric_id'], ['performance_metrics.metric_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['pre_metric_id'], ['performance_metrics.metric_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['post_metric_id'], ['performance_metrics.metric_id'], ondelete='SET NULL'),
        # Check constraints
        sa.CheckConstraint(
            'decision_confidence >= 0.0 AND decision_confidence <= 1.0',
            name='check_decision_confidence_range'
        ),
        sa.CheckConstraint(
            'effectiveness_delta IS NULL OR (effectiveness_delta >= -1.0 AND effectiveness_delta <= 1.0)',
            name='check_effectiveness_delta_range'
        ),
        sa.CheckConstraint(
            "trigger_type IN ('performance_degradation', 'error_accumulation', 'context_staleness', 'capability_mismatch')",
            name='check_trigger_type_values'
        ),
        sa.CheckConstraint(
            "intervention_type IN ('context_refresh', 'replan', 'reflect', 'capability_switch')",
            name='check_intervention_type_values'
        ),
        sa.CheckConstraint(
            "execution_status IN ('success', 'failure', 'partial', 'pending')",
            name='check_execution_status_values'
        ),
    )
    # Create indexes for intervention_records
    op.create_index('idx_intervention_task', 'intervention_records', ['task_id'])
    op.create_index('idx_intervention_agent', 'intervention_records', ['agent_id'])
    op.create_index('idx_intervention_trigger_type', 'intervention_records', ['trigger_type'])
    op.create_index('idx_intervention_type', 'intervention_records', ['intervention_type'])
    op.create_index('idx_intervention_executed', 'intervention_records', ['executed_at'])
    op.create_index('idx_intervention_agent_task', 'intervention_records', ['agent_id', 'task_id', 'executed_at'])


def downgrade() -> None:
    """Downgrade schema - drop ACE tables in reverse order."""

    # Drop intervention_records table and indexes
    op.drop_index('idx_intervention_agent_task', table_name='intervention_records')
    op.drop_index('idx_intervention_executed', table_name='intervention_records')
    op.drop_index('idx_intervention_type', table_name='intervention_records')
    op.drop_index('idx_intervention_trigger_type', table_name='intervention_records')
    op.drop_index('idx_intervention_agent', table_name='intervention_records')
    op.drop_index('idx_intervention_task', table_name='intervention_records')
    op.drop_table('intervention_records')

    # Drop performance_metrics table and indexes
    op.drop_index('idx_metrics_agent_stage', table_name='performance_metrics')
    op.drop_index('idx_metrics_recorded', table_name='performance_metrics')
    op.drop_index('idx_metrics_stage', table_name='performance_metrics')
    op.drop_index('idx_metrics_agent', table_name='performance_metrics')
    op.drop_index('idx_metrics_task', table_name='performance_metrics')
    op.drop_table('performance_metrics')

    # Drop evolution_status table
    op.drop_index('idx_evolution_status', table_name='evolution_status')
    op.drop_table('evolution_status')

    # Drop execution_traces table
    op.drop_index('idx_traces_success', table_name='execution_traces')
    op.drop_index('idx_traces_agent', table_name='execution_traces')
    op.drop_table('execution_traces')

    # Drop context_deltas table
    op.drop_index('idx_deltas_applied', table_name='context_deltas')
    op.drop_index('idx_deltas_confidence', table_name='context_deltas')
    op.drop_index('idx_deltas_playbook', table_name='context_deltas')
    op.drop_table('context_deltas')

    # Drop context_playbooks table
    op.drop_index('idx_playbooks_updated', table_name='context_playbooks')
    op.drop_index('idx_playbooks_agent', table_name='context_playbooks')
    op.drop_table('context_playbooks')
