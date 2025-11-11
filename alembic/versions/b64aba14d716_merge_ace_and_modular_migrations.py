"""merge ace and modular migrations

Revision ID: b64aba14d716
Revises: 54d726783080, e247cdc4c183
Create Date: 2025-11-11 10:58:05.924548

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b64aba14d716'
down_revision: Union[str, Sequence[str], None] = ('54d726783080', 'e247cdc4c183')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
