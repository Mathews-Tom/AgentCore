import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool, event
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set flag to prevent enum auto-creation during migrations
os.environ['ALEMBIC_RUNNING'] = '1'

# Import Base and models for autogenerate support
from agentcore.a2a_protocol.database.connection import Base, get_database_url
from agentcore.a2a_protocol.database import models  # noqa: F401 - Ensure models are loaded

# Suppress automatic enum creation from table events during Alembic runs
# Migrations will explicitly create enums using postgresql.ENUM().create()
from sqlalchemy import event as sa_event

# Only suppress during Alembic runs
if os.getenv('ALEMBIC_RUNNING'):
    from sqlalchemy.sql.sqltypes import Enum as SQLAEnum
    from sqlalchemy.dialects.postgresql import ENUM as PgENUM

    # Disable enum auto-creation for all enum columns before metadata is processed
    @sa_event.listens_for(Base.metadata, "before_create", propagate=True)
    def receive_before_create(target, connection, **kw):
        """Suppress automatic enum type creation - migrations handle this explicitly."""
        # Iterate through all tables and columns
        for table_name, table in Base.metadata.tables.items():
            for column in table.columns:
                # Check if column type is an Enum
                if isinstance(column.type, (SQLAEnum, PgENUM)):
                    # Disable automatic creation
                    column.type.create_type = False

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override sqlalchemy.url from our config
config.set_main_option("sqlalchemy.url", get_database_url())

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
