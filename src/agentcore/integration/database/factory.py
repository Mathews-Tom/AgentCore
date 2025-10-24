"""Database connector factory for creating database instances.

Provides a centralized factory for creating database connectors
based on database type with proper configuration.
"""

from __future__ import annotations

from typing import Type

from agentcore.integration.database.base import DatabaseConfig, DatabaseConnector
from agentcore.integration.database.postgresql import PostgreSQLConnector


class DatabaseFactory:
    """Factory for creating database connectors.

    Provides a registry of database connector implementations
    and creates instances based on database type.
    """

    _connectors: dict[str, Type[DatabaseConnector]] = {
        "postgresql": PostgreSQLConnector,
        "postgres": PostgreSQLConnector,  # Alias
    }

    @classmethod
    def create(cls, config: DatabaseConfig) -> DatabaseConnector:
        """Create database connector instance.

        Args:
            config: Database connection configuration

        Returns:
            Database connector instance for specified type

        Raises:
            ValueError: If database type is not supported
        """
        db_type = config.type.lower()

        if db_type not in cls._connectors:
            supported = ", ".join(cls._connectors.keys())
            raise ValueError(
                f"Unsupported database type: {config.type}. "
                f"Supported types: {supported}"
            )

        connector_class = cls._connectors[db_type]
        return connector_class(config)

    @classmethod
    def register_connector(
        cls,
        db_type: str,
        connector_class: Type[DatabaseConnector],
    ) -> None:
        """Register a new database connector type.

        Allows extending the factory with custom connector implementations.

        Args:
            db_type: Database type identifier (e.g., 'mysql', 'oracle')
            connector_class: Connector class implementing DatabaseConnector
        """
        cls._connectors[db_type.lower()] = connector_class

    @classmethod
    def supported_types(cls) -> list[str]:
        """Get list of supported database types.

        Returns:
            List of registered database type identifiers
        """
        return list(cls._connectors.keys())
