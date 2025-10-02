"""add_semantic_and_context_fields

Adds support for:
- A2A-016: Semantic capability matching with pgvector
- A2A-018: Context engineering fields for agents

Revision ID: 068b96d43e02
Revises: 16ae5b2f867b
Create Date: 2025-10-02 15:36:00.581385

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '068b96d43e02'
down_revision: Union[str, Sequence[str], None] = '16ae5b2f867b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Add pgvector column for semantic capability matching (A2A-016)
    # 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
    op.add_column('agents', sa.Column('capability_embedding', postgresql.ARRAY(sa.Float), nullable=True))

    # Add context engineering fields (A2A-018)
    op.add_column('agents', sa.Column('system_context', sa.Text(), nullable=True))
    op.add_column('agents', sa.Column('interaction_examples', sa.JSON(), nullable=True))

    # Create HNSW index for vector similarity search
    # m=16 (number of connections), ef_construction=64 (search quality)
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_agent_capability_embedding
        ON agents USING hnsw (capability_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    ''')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop index
    op.execute('DROP INDEX IF EXISTS idx_agent_capability_embedding')

    # Drop columns
    op.drop_column('agents', 'interaction_examples')
    op.drop_column('agents', 'system_context')
    op.drop_column('agents', 'capability_embedding')

    # Note: Not dropping pgvector extension in case other tables use it
