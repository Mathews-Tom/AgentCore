"""add_trace_id_to_modular_executions

Revision ID: e83e054db270
Revises: f9c3a8b2e1d4
Create Date: 2025-11-23 23:30:09.099263

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e83e054db270'
down_revision: Union[str, Sequence[str], None] = 'f9c3a8b2e1d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Add trace_id column to modular_executions table."""
    op.add_column(
        'modular_executions',
        sa.Column('trace_id', sa.String(255), nullable=True, index=True)
    )


def downgrade() -> None:
    """Downgrade schema: Remove trace_id column from modular_executions table."""
    op.drop_column('modular_executions', 'trace_id')
