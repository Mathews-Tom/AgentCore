"""add_updated_at_to_workflow_executions

Revision ID: a5ce830ff941
Revises: e83e054db270
Create Date: 2025-11-30 22:59:09.679414

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a5ce830ff941'
down_revision: Union[str, Sequence[str], None] = 'e83e054db270'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add updated_at column to workflow_executions table."""
    # Add updated_at column with default value
    op.add_column(
        'workflow_executions',
        sa.Column(
            'updated_at',
            sa.DateTime,
            nullable=False,
            server_default=sa.func.now()
        )
    )

    # Set existing rows' updated_at to created_at
    op.execute(
        "UPDATE workflow_executions SET updated_at = created_at WHERE updated_at IS NULL"
    )


def downgrade() -> None:
    """Remove updated_at column from workflow_executions table."""
    op.drop_column('workflow_executions', 'updated_at')
