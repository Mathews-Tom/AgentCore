"""add_timescaledb_compression_for_ace_metrics

Configure TimescaleDB compression for ACE performance metrics table.
Enables automatic compression with 90-day retention policy.

Revision ID: 54d726783080
Revises: c03db99da40b
Create Date: 2025-11-09 21:37:39.714683

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '54d726783080'
down_revision: Union[str, Sequence[str], None] = 'c03db99da40b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Configure TimescaleDB compression for performance_metrics.

    Benefits:
    - 90-95% storage reduction
    - Faster queries on compressed data
    - Automatic 90-day retention
    """
    # Enable compression on performance_metrics hypertable
    # Note: This assumes performance_metrics is already a hypertable
    # from the ACE database migration (c03db99da40b)

    op.execute("""
        -- Enable compression on performance_metrics
        ALTER TABLE performance_metrics SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'agent_id,task_id',
            timescaledb.compress_orderby = 'recorded_at DESC'
        );
    """)

    # Add compression policy: compress chunks older than 7 days
    op.execute("""
        SELECT add_compression_policy(
            'performance_metrics',
            INTERVAL '7 days'
        );
    """)

    # Add retention policy: drop chunks older than 90 days
    op.execute("""
        SELECT add_retention_policy(
            'performance_metrics',
            INTERVAL '90 days'
        );
    """)


def downgrade() -> None:
    """Remove TimescaleDB compression configuration."""

    # Remove retention policy
    op.execute("""
        SELECT remove_retention_policy('performance_metrics');
    """)

    # Remove compression policy
    op.execute("""
        SELECT remove_compression_policy('performance_metrics');
    """)

    # Disable compression
    op.execute("""
        ALTER TABLE performance_metrics SET (
            timescaledb.compress = false
        );
    """)
