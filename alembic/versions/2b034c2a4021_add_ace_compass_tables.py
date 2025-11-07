"""add_ace_compass_tables

Revision ID: 2b034c2a4021
Revises: 83591e23ca42
Create Date: 2025-11-08 02:34:24.758711

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2b034c2a4021'
down_revision: Union[str, Sequence[str], None] = '83591e23ca42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create context_playbooks table
    op.create_table(
        'context_playbooks',
        sa.Column('playbook_id', sa.dialects.postgresql.UUID(as_uuid=True),
                  server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('context', sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text('NOW()')),
        sa.Column('metadata', sa.dialects.postgresql.JSONB(),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint('playbook_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'],
                               ondelete='CASCADE')
    )
    op.create_index('idx_playbooks_agent', 'context_playbooks', ['agent_id'])
    op.create_index('idx_playbooks_updated', 'context_playbooks', ['updated_at'],
                    postgresql_ops={'updated_at': 'DESC'})

    # Create context_deltas table
    op.create_table(
        'context_deltas',
        sa.Column('delta_id', sa.dialects.postgresql.UUID(as_uuid=True),
                  server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('playbook_id', sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('changes', sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text('NOW()')),
        sa.Column('applied', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('delta_id'),
        sa.ForeignKeyConstraint(['playbook_id'], ['context_playbooks.playbook_id'],
                               ondelete='CASCADE'),
        sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0',
                          name='check_confidence_range')
    )
    op.create_index('idx_deltas_playbook', 'context_deltas', ['playbook_id'])
    op.create_index('idx_deltas_confidence', 'context_deltas', ['confidence'],
                    postgresql_ops={'confidence': 'DESC'})
    op.create_index('idx_deltas_applied', 'context_deltas', ['applied', 'applied_at'],
                    postgresql_ops={'applied_at': 'DESC'})

    # Create execution_traces table
    op.create_table(
        'execution_traces',
        sa.Column('trace_id', sa.dialects.postgresql.UUID(as_uuid=True),
                  server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('task_id', sa.String(255), nullable=True),
        sa.Column('execution_time', sa.Float(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('output_quality', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', sa.dialects.postgresql.JSONB(),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('captured_at', sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('trace_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'],
                               ondelete='CASCADE'),
        sa.CheckConstraint('output_quality IS NULL OR (output_quality >= 0.0 AND output_quality <= 1.0)',
                          name='check_output_quality_range')
    )
    op.create_index('idx_traces_agent', 'execution_traces', ['agent_id', 'captured_at'],
                    postgresql_ops={'captured_at': 'DESC'})
    op.create_index('idx_traces_success', 'execution_traces', ['success', 'captured_at'],
                    postgresql_ops={'captured_at': 'DESC'})

    # Create evolution_status table
    op.create_table(
        'evolution_status',
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('last_evolution', sa.DateTime(timezone=True), nullable=True),
        sa.Column('pending_traces', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('deltas_generated', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('deltas_applied', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_cost', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('status', sa.String(50), nullable=False, server_default="'idle'"),
        sa.PrimaryKeyConstraint('agent_id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'],
                               ondelete='CASCADE'),
        sa.CheckConstraint("status IN ('idle', 'processing', 'failed')",
                          name='check_status_values')
    )
    op.create_index('idx_evolution_status', 'evolution_status', ['status', 'last_evolution'],
                    postgresql_ops={'last_evolution': 'DESC'})


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_index('idx_evolution_status', table_name='evolution_status')
    op.drop_table('evolution_status')

    op.drop_index('idx_traces_success', table_name='execution_traces')
    op.drop_index('idx_traces_agent', table_name='execution_traces')
    op.drop_table('execution_traces')

    op.drop_index('idx_deltas_applied', table_name='context_deltas')
    op.drop_index('idx_deltas_confidence', table_name='context_deltas')
    op.drop_index('idx_deltas_playbook', table_name='context_deltas')
    op.drop_table('context_deltas')

    op.drop_index('idx_playbooks_updated', table_name='context_playbooks')
    op.drop_index('idx_playbooks_agent', table_name='context_playbooks')
    op.drop_table('context_playbooks')
