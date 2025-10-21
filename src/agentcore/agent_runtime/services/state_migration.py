"""State migration service for schema version changes."""

from collections.abc import Callable
from typing import Any

import structlog

from ..models.state_persistence import AgentStateSnapshot, StateMigration, StateVersion
from .state_serializer import CURRENT_STATE_VERSION

logger = structlog.get_logger()


class MigrationError(Exception):
    """Raised when migration fails."""


class StateMigrationService:
    """Service for handling state schema migrations."""

    def __init__(self) -> None:
        """Initialize migration service."""
        self._migrations: dict[tuple[str, str], StateMigration] = {}
        self._migration_functions: dict[
            str, Callable[[dict[str, Any]], dict[str, Any]]
        ] = {}

        # Register built-in migrations
        self._register_builtin_migrations()

        logger.info("state_migration_service_initialized")

    def register_migration(
        self,
        from_version: StateVersion,
        to_version: StateVersion,
        migration_func: Callable[[dict[str, Any]], dict[str, Any]],
        description: str,
        breaking_change: bool = False,
    ) -> None:
        """
        Register a state migration.

        Args:
            from_version: Source version
            to_version: Target version
            migration_func: Function to perform migration
            description: Migration description
            breaking_change: Whether migration is breaking
        """
        migration_key = (str(from_version), str(to_version))
        function_name = f"migrate_{from_version}__to__{to_version}"

        migration = StateMigration(
            from_version=from_version,
            to_version=to_version,
            migration_function=function_name,
            description=description,
            breaking_change=breaking_change,
        )

        self._migrations[migration_key] = migration
        self._migration_functions[function_name] = migration_func

        logger.info(
            "migration_registered",
            from_version=str(from_version),
            to_version=str(to_version),
            breaking=breaking_change,
        )

    def can_migrate(
        self,
        from_version: StateVersion,
        to_version: StateVersion,
    ) -> bool:
        """
        Check if migration is possible between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            True if migration path exists
        """
        if from_version == to_version:
            return True

        # Check for direct migration
        migration_key = (str(from_version), str(to_version))
        if migration_key in self._migrations:
            return True

        # Check for multi-step migration path
        migration_path = self._find_migration_path(from_version, to_version)
        return migration_path is not None

    def migrate_state(
        self,
        snapshot: AgentStateSnapshot,
        target_version: StateVersion | None = None,
    ) -> tuple[AgentStateSnapshot, list[StateMigration]]:
        """
        Migrate state snapshot to target version.

        Args:
            snapshot: State snapshot to migrate
            target_version: Target version (or current if None)

        Returns:
            Tuple of (migrated snapshot, list of applied migrations)

        Raises:
            MigrationError: If migration fails
        """
        target_version = target_version or CURRENT_STATE_VERSION
        current_version = snapshot.version

        # No migration needed
        if current_version == target_version:
            return snapshot, []

        # Find migration path
        migration_path = self._find_migration_path(current_version, target_version)
        if migration_path is None:
            raise MigrationError(
                f"No migration path from {current_version} to {target_version}"
            )

        # Apply migrations in sequence
        applied_migrations = []
        current_data = snapshot.model_dump()

        for migration in migration_path:
            try:
                logger.info(
                    "applying_migration",
                    from_version=str(migration.from_version),
                    to_version=str(migration.to_version),
                )

                # Get migration function
                migration_func = self._migration_functions[migration.migration_function]

                # Apply migration
                current_data = migration_func(current_data)
                current_data["version"] = migration.to_version.model_dump()

                # Record applied migration
                migration.applied_at = None  # Will be set by persistence layer
                applied_migrations.append(migration)

            except Exception as e:
                logger.error(
                    "migration_failed",
                    from_version=str(migration.from_version),
                    to_version=str(migration.to_version),
                    error=str(e),
                )
                raise MigrationError(
                    f"Migration failed from {migration.from_version} "
                    f"to {migration.to_version}: {e}"
                ) from e

        # Create migrated snapshot
        migrated_snapshot = AgentStateSnapshot.model_validate(current_data)

        logger.info(
            "state_migrated",
            agent_id=snapshot.agent_id,
            from_version=str(current_version),
            to_version=str(target_version),
            migrations_applied=len(applied_migrations),
        )

        return migrated_snapshot, applied_migrations

    def _find_migration_path(
        self,
        from_version: StateVersion,
        to_version: StateVersion,
    ) -> list[StateMigration] | None:
        """
        Find migration path between versions using BFS.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of migrations to apply, or None if no path exists
        """
        if from_version == to_version:
            return []

        # Check for direct path
        direct_key = (str(from_version), str(to_version))
        if direct_key in self._migrations:
            return [self._migrations[direct_key]]

        # BFS to find shortest path
        queue: list[tuple[StateVersion, list[StateMigration]]] = [(from_version, [])]
        visited = {str(from_version)}

        while queue:
            current_version, path = queue.pop(0)

            # Check all migrations from current version
            for migration_key, migration in self._migrations.items():
                src_version_str, dst_version_str = migration_key

                if src_version_str != str(current_version):
                    continue

                dst_version = StateVersion.parse(dst_version_str)

                # Check if we reached target
                if dst_version == to_version:
                    return path + [migration]

                # Add to queue if not visited
                if dst_version_str not in visited:
                    visited.add(dst_version_str)
                    queue.append((dst_version, path + [migration]))

        return None

    def _register_builtin_migrations(self) -> None:
        """Register built-in state migrations."""
        # Example: Migration from 1.0.0 to 1.1.0
        # This is a placeholder for future migrations

        def migrate_1_0_0_to_1_1_0(data: dict[str, Any]) -> dict[str, Any]:
            """Migrate from version 1.0.0 to 1.1.0."""
            # Example: Add new field with default value
            if "metadata" not in data:
                data["metadata"] = {}
            return data

        # Uncomment when needed:
        # self.register_migration(
        #     from_version=StateVersion(major=1, minor=0, patch=0),
        #     to_version=StateVersion(major=1, minor=1, patch=0),
        #     migration_func=migrate_1_0_0_to_1_1_0,
        #     description="Add metadata field with backward compatibility",
        #     breaking_change=False,
        # )


# Global migration service instance
_migration_service: StateMigrationService | None = None


def get_migration_service() -> StateMigrationService:
    """Get global migration service instance."""
    global _migration_service
    if _migration_service is None:
        _migration_service = StateMigrationService()
    return _migration_service
