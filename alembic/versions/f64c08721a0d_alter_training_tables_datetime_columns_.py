"""alter training tables datetime columns to use timezone

Revision ID: f64c08721a0d
Revises: e9ba324940b6
Create Date: 2025-10-18 00:23:13.684365

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f64c08721a0d'
down_revision: Union[str, Sequence[str], None] = 'e9ba324940b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Training jobs table
    op.execute("""
        ALTER TABLE training_jobs
        ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING created_at AT TIME ZONE 'UTC',
        ALTER COLUMN started_at TYPE TIMESTAMP WITH TIME ZONE USING started_at AT TIME ZONE 'UTC',
        ALTER COLUMN completed_at TYPE TIMESTAMP WITH TIME ZONE USING completed_at AT TIME ZONE 'UTC'
    """)

    # Trajectories table
    op.execute("""
        ALTER TABLE trajectories
        ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING created_at AT TIME ZONE 'UTC'
    """)

    # Policy checkpoints table
    op.execute("""
        ALTER TABLE policy_checkpoints
        ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING created_at AT TIME ZONE 'UTC'
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Training jobs table
    op.execute("""
        ALTER TABLE training_jobs
        ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE,
        ALTER COLUMN started_at TYPE TIMESTAMP WITHOUT TIME ZONE,
        ALTER COLUMN completed_at TYPE TIMESTAMP WITHOUT TIME ZONE
    """)

    # Trajectories table
    op.execute("""
        ALTER TABLE trajectories
        ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE
    """)

    # Policy checkpoints table
    op.execute("""
        ALTER TABLE policy_checkpoints
        ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE
    """)

