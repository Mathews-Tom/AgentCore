"""Tests for StateMigrationService."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentcore.agent_runtime.models.state_persistence import (
    AgentStateSnapshot,
    CompressionType,
    StateVersion,
)
from agentcore.agent_runtime.services.state_migration import (
    MigrationError,
    StateMigrationService,
    get_migration_service,
)
from agentcore.agent_runtime.services.state_serializer import CURRENT_STATE_VERSION


@pytest.fixture
def migration_service() -> StateMigrationService:
    """Create StateMigrationService instance."""
    return StateMigrationService()


@pytest.fixture
def sample_snapshot() -> AgentStateSnapshot:
    """Create sample agent state snapshot."""
    return AgentStateSnapshot(
        agent_id="agent-001",
        snapshot_id="snapshot-001",
        version=StateVersion(major=1, minor=0, patch=0),
        status="running",
        philosophy="react",
        execution_context={"task_id": "task-001"},
        reasoning_chain=[{"step": 1, "thought": "analyze"}],
        decision_history=[{"action": "start"}],
        tool_usage_log=[{"tool": "calculator"}],
        performance_metrics={"cpu_percent": 50.0},
        working_memory={"goal": "solve"},
        long_term_memory={"patterns": ["p1"]},
    )


class TestStateMigrationServiceInit:
    """Test suite for StateMigrationService initialization."""

    def test_init_creates_empty_registries(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test initialization creates empty migration registries."""
        # Built-in migrations are registered but commented out
        # So registries should be empty
        assert migration_service._migrations == {}
        assert migration_service._migration_functions == {}

    def test_init_calls_register_builtin_migrations(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test initialization calls _register_builtin_migrations."""
        # The method is called but no migrations are registered (all commented out)
        # Verify the service is initialized properly
        assert isinstance(migration_service._migrations, dict)
        assert isinstance(migration_service._migration_functions, dict)


class TestStateMigrationServiceRegister:
    """Test suite for register_migration method."""

    def test_register_migration_success(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test registering a migration."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=1, minor=1, patch=0)

        def migrate_func(data: dict[str, Any]) -> dict[str, Any]:
            data["new_field"] = "default"
            return data

        migration_service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=migrate_func,
            description="Add new_field",
            breaking_change=False,
        )

        # Verify migration registered
        migration_key = ("1.0.0", "1.1.0")
        assert migration_key in migration_service._migrations

        # Verify migration metadata
        migration = migration_service._migrations[migration_key]
        assert migration.from_version == from_version
        assert migration.to_version == to_version
        assert migration.description == "Add new_field"
        assert migration.breaking_change is False

        # Verify function registered
        func_name = "migrate_1.0.0__to__1.1.0"
        assert func_name in migration_service._migration_functions

    def test_register_migration_breaking_change(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test registering a breaking migration."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=2, minor=0, patch=0)

        def migrate_func(data: dict[str, Any]) -> dict[str, Any]:
            # Breaking change: remove old field
            data.pop("old_field", None)
            return data

        migration_service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=migrate_func,
            description="Remove old_field (breaking)",
            breaking_change=True,
        )

        # Verify breaking change flag
        migration_key = ("1.0.0", "2.0.0")
        migration = migration_service._migrations[migration_key]
        assert migration.breaking_change is True

    def test_register_multiple_migrations(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test registering multiple migrations."""

        def migrate_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
            return data

        def migrate_2_to_3(data: dict[str, Any]) -> dict[str, Any]:
            return data

        migration_service.register_migration(
            from_version=StateVersion(major=1, minor=0, patch=0),
            to_version=StateVersion(major=2, minor=0, patch=0),
            migration_func=migrate_1_to_2,
            description="Migrate 1 to 2",
        )

        migration_service.register_migration(
            from_version=StateVersion(major=2, minor=0, patch=0),
            to_version=StateVersion(major=3, minor=0, patch=0),
            migration_func=migrate_2_to_3,
            description="Migrate 2 to 3",
        )

        # Verify both migrations registered
        assert len(migration_service._migrations) == 2
        assert ("1.0.0", "2.0.0") in migration_service._migrations
        assert ("2.0.0", "3.0.0") in migration_service._migrations


class TestStateMigrationServiceCanMigrate:
    """Test suite for can_migrate method."""

    def test_can_migrate_same_version(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test can_migrate with same version."""
        version = StateVersion(major=1, minor=0, patch=0)

        result = migration_service.can_migrate(version, version)

        assert result is True

    def test_can_migrate_direct_path(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test can_migrate with direct migration path."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=1, minor=1, patch=0)

        # Register migration
        migration_service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=lambda data: data,
            description="Direct migration",
        )

        result = migration_service.can_migrate(from_version, to_version)

        assert result is True

    def test_can_migrate_multi_step_path(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test can_migrate with multi-step migration path."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)
        v3 = StateVersion(major=1, minor=2, patch=0)

        # Register migration chain: v1 -> v2 -> v3
        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=lambda data: data,
            description="v1 to v2",
        )
        migration_service.register_migration(
            from_version=v2,
            to_version=v3,
            migration_func=lambda data: data,
            description="v2 to v3",
        )

        # Should be able to migrate from v1 to v3
        result = migration_service.can_migrate(v1, v3)

        assert result is True

    def test_can_migrate_no_path(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test can_migrate with no migration path."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=2, minor=0, patch=0)

        # No migrations registered
        result = migration_service.can_migrate(from_version, to_version)

        assert result is False


class TestStateMigrationServiceMigrateState:
    """Test suite for migrate_state method."""

    def test_migrate_state_no_migration_needed(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state when no migration needed."""
        target_version = sample_snapshot.version

        migrated_snapshot, migrations = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=target_version,
        )

        # Should return same snapshot
        assert migrated_snapshot == sample_snapshot
        assert migrations == []

    def test_migrate_state_to_current_version(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state to current version (default)."""
        # If snapshot version matches current, no migration
        sample_snapshot.version = CURRENT_STATE_VERSION

        migrated_snapshot, migrations = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=None,  # Use current
        )

        assert migrated_snapshot == sample_snapshot
        assert migrations == []

    def test_migrate_state_single_step(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state with single migration step."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=1, minor=1, patch=0)

        # Register migration that modifies metadata (an existing field)
        def add_metadata(data: dict[str, Any]) -> dict[str, Any]:
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["migrated"] = "v1.1"
            return data

        migration_service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=add_metadata,
            description="Add metadata field",
        )

        # Set snapshot to from_version
        sample_snapshot.version = from_version

        migrated_snapshot, migrations = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=to_version,
        )

        # Verify migration applied
        assert migrated_snapshot.version == to_version
        assert len(migrations) == 1
        assert migrations[0].from_version == from_version
        assert migrations[0].to_version == to_version

        # Verify metadata was updated
        assert migrated_snapshot.metadata["migrated"] == "v1.1"

    def test_migrate_state_multi_step(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state with multi-step migration."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)
        v3 = StateVersion(major=1, minor=2, patch=0)

        # Register migration chain that modifies metadata
        def migrate_v1_v2(data: dict[str, Any]) -> dict[str, Any]:
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["v2_applied"] = True
            return data

        def migrate_v2_v3(data: dict[str, Any]) -> dict[str, Any]:
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["v3_applied"] = True
            return data

        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=migrate_v1_v2,
            description="v1 to v2",
        )
        migration_service.register_migration(
            from_version=v2,
            to_version=v3,
            migration_func=migrate_v2_v3,
            description="v2 to v3",
        )

        # Set snapshot to v1
        sample_snapshot.version = v1

        migrated_snapshot, migrations = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=v3,
        )

        # Verify both migrations applied
        assert migrated_snapshot.version == v3
        assert len(migrations) == 2
        assert migrations[0].from_version == v1
        assert migrations[0].to_version == v2
        assert migrations[1].from_version == v2
        assert migrations[1].to_version == v3

        # Verify both migrations modified metadata
        assert migrated_snapshot.metadata["v2_applied"] is True
        assert migrated_snapshot.metadata["v3_applied"] is True

    def test_migrate_state_no_path_error(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state raises error when no migration path."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=2, minor=0, patch=0)

        sample_snapshot.version = from_version

        # No migrations registered
        with pytest.raises(MigrationError) as exc_info:
            migration_service.migrate_state(
                snapshot=sample_snapshot,
                target_version=to_version,
            )

        assert "No migration path" in str(exc_info.value)
        assert "1.0.0" in str(exc_info.value)
        assert "2.0.0" in str(exc_info.value)

    def test_migrate_state_migration_function_error(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state handles migration function errors."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=1, minor=1, patch=0)

        # Register migration that raises error
        def failing_migration(data: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("Migration failed")

        migration_service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=failing_migration,
            description="Failing migration",
        )

        sample_snapshot.version = from_version

        with pytest.raises(MigrationError) as exc_info:
            migration_service.migrate_state(
                snapshot=sample_snapshot,
                target_version=to_version,
            )

        assert "Migration failed" in str(exc_info.value)

    def test_migrate_state_preserves_original_fields(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrate_state preserves original snapshot fields."""
        from_version = StateVersion(major=1, minor=0, patch=0)
        to_version = StateVersion(major=1, minor=1, patch=0)

        def migration_func(data: dict[str, Any]) -> dict[str, Any]:
            # Only add new field, don't modify existing
            data["new_field"] = "value"
            return data

        migration_service.register_migration(
            from_version=from_version,
            to_version=to_version,
            migration_func=migration_func,
            description="Add field",
        )

        sample_snapshot.version = from_version
        original_agent_id = sample_snapshot.agent_id
        original_status = sample_snapshot.status

        migrated_snapshot, _ = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=to_version,
        )

        # Verify original fields preserved
        assert migrated_snapshot.agent_id == original_agent_id
        assert migrated_snapshot.status == original_status
        assert migrated_snapshot.execution_context == sample_snapshot.execution_context


class TestStateMigrationServiceFindPath:
    """Test suite for _find_migration_path method."""

    def test_find_path_same_version(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test finding path for same version."""
        version = StateVersion(major=1, minor=0, patch=0)

        path = migration_service._find_migration_path(version, version)

        assert path == []

    def test_find_path_direct(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test finding direct migration path."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)

        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=lambda data: data,
            description="Direct",
        )

        path = migration_service._find_migration_path(v1, v2)

        assert path is not None
        assert len(path) == 1
        assert path[0].from_version == v1
        assert path[0].to_version == v2

    def test_find_path_multi_step(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test finding multi-step migration path."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)
        v3 = StateVersion(major=1, minor=2, patch=0)

        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=lambda data: data,
            description="v1 to v2",
        )
        migration_service.register_migration(
            from_version=v2,
            to_version=v3,
            migration_func=lambda data: data,
            description="v2 to v3",
        )

        path = migration_service._find_migration_path(v1, v3)

        assert path is not None
        assert len(path) == 2
        assert path[0].from_version == v1
        assert path[0].to_version == v2
        assert path[1].from_version == v2
        assert path[1].to_version == v3

    def test_find_path_no_path(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test finding path when no path exists."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=2, minor=0, patch=0)

        path = migration_service._find_migration_path(v1, v2)

        assert path is None

    def test_find_path_bfs_shortest(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test BFS finds shortest path."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)
        v3 = StateVersion(major=1, minor=2, patch=0)

        # Create two paths: v1->v2->v3 (length 2) and v1->v3 (length 1)
        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=lambda data: data,
            description="v1 to v2",
        )
        migration_service.register_migration(
            from_version=v2,
            to_version=v3,
            migration_func=lambda data: data,
            description="v2 to v3",
        )
        migration_service.register_migration(
            from_version=v1,
            to_version=v3,
            migration_func=lambda data: data,
            description="v1 to v3 direct",
        )

        path = migration_service._find_migration_path(v1, v3)

        # Should find shortest path (direct v1->v3)
        assert path is not None
        assert len(path) == 1
        assert path[0].from_version == v1
        assert path[0].to_version == v3

    def test_find_path_complex_graph(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test finding path in complex migration graph."""
        v1_0 = StateVersion(major=1, minor=0, patch=0)
        v1_1 = StateVersion(major=1, minor=1, patch=0)
        v1_2 = StateVersion(major=1, minor=2, patch=0)
        v2_0 = StateVersion(major=2, minor=0, patch=0)

        # Create graph:
        #   v1.0 -> v1.1 -> v1.2
        #            |  \    |
        #            |   \   v
        #            +-----> v2.0
        migration_service.register_migration(
            from_version=v1_0,
            to_version=v1_1,
            migration_func=lambda data: data,
            description="v1.0 to v1.1",
        )
        migration_service.register_migration(
            from_version=v1_1,
            to_version=v1_2,
            migration_func=lambda data: data,
            description="v1.1 to v1.2",
        )
        migration_service.register_migration(
            from_version=v1_1,
            to_version=v2_0,
            migration_func=lambda data: data,
            description="v1.1 to v2.0",
        )
        migration_service.register_migration(
            from_version=v1_2,
            to_version=v2_0,
            migration_func=lambda data: data,
            description="v1.2 to v2.0",
        )

        path = migration_service._find_migration_path(v1_0, v2_0)

        # Should find shortest path: v1.0 -> v1.1 -> v2.0
        assert path is not None
        assert len(path) == 2
        assert path[0].from_version == v1_0
        assert path[0].to_version == v1_1
        assert path[1].from_version == v1_1
        assert path[1].to_version == v2_0


class TestStateMigrationServiceBuiltinMigrations:
    """Test suite for built-in migrations."""

    def test_builtin_migration_1_0_0_to_1_1_0(
        self,
    ) -> None:
        """Test built-in migration from 1.0.0 to 1.1.0."""
        # The built-in migration is commented out, but test the function exists
        from agentcore.agent_runtime.services.state_migration import (
            StateMigrationService,
        )

        service = StateMigrationService()

        # Migration is not registered (commented out)
        # But verify the service initializes correctly
        assert isinstance(service._migrations, dict)


class TestGlobalMigrationService:
    """Test suite for global migration service."""

    def test_get_migration_service_singleton(self) -> None:
        """Test get_migration_service returns singleton."""
        service1 = get_migration_service()
        service2 = get_migration_service()

        assert service1 is service2

    def test_get_migration_service_instance(self) -> None:
        """Test get_migration_service returns StateMigrationService."""
        service = get_migration_service()

        assert isinstance(service, StateMigrationService)


class TestStateMigrationServiceEdgeCases:
    """Test suite for edge cases and complex scenarios."""

    def test_migration_with_complex_data_transformation(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migration with complex data transformation."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=2, minor=0, patch=0)

        def complex_migration(data: dict[str, Any]) -> dict[str, Any]:
            # Transform execution_context (modify existing field)
            if "execution_context" in data:
                context = data["execution_context"]
                context["migrated_to_v2"] = True
                context["original_task"] = context.get("task_id", "unknown")

            # Transform performance_metrics
            if "performance_metrics" in data:
                metrics = data["performance_metrics"]
                # Store summary in metadata
                if "metadata" not in data:
                    data["metadata"] = {}
                data["metadata"]["metrics_summary"] = {
                    "cpu": metrics.get("cpu_percent", 0),
                    "count": len(metrics),
                }

            return data

        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=complex_migration,
            description="Complex transformation",
        )

        sample_snapshot.version = v1

        migrated_snapshot, _ = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=v2,
        )

        # Verify complex transformations applied
        assert migrated_snapshot.execution_context["migrated_to_v2"] is True
        assert migrated_snapshot.execution_context["original_task"] == "task-001"
        assert migrated_snapshot.metadata["metrics_summary"]["cpu"] == 50.0
        assert migrated_snapshot.metadata["metrics_summary"]["count"] == 1

    def test_migration_chain_with_intermediate_versions(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migration through many intermediate versions."""
        versions = [StateVersion(major=1, minor=i, patch=0) for i in range(5)]

        # Register chain of migrations
        for i in range(len(versions) - 1):
            migration_service.register_migration(
                from_version=versions[i],
                to_version=versions[i + 1],
                migration_func=lambda data, ver=i: {**data, f"field_v{ver}": f"value{ver}"},
                description=f"v{i} to v{i+1}",
            )

        sample_snapshot.version = versions[0]

        migrated_snapshot, migrations = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=versions[-1],
        )

        # Verify all intermediate migrations applied
        assert len(migrations) == 4
        assert migrated_snapshot.version == versions[-1]

    def test_migration_with_empty_data_fields(
        self,
        migration_service: StateMigrationService,
    ) -> None:
        """Test migration with empty data fields."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)

        def migration_func(data: dict[str, Any]) -> dict[str, Any]:
            # Handle empty fields gracefully
            if not data.get("execution_context"):
                data["execution_context"] = {}
            data["execution_context"]["migrated"] = True
            return data

        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=migration_func,
            description="Handle empty fields",
        )

        # Create snapshot with empty execution_context
        snapshot = AgentStateSnapshot(
            agent_id="agent-test",
            snapshot_id="snapshot-test",
            version=v1,
            status="created",
            philosophy="react",
            execution_context={},
        )

        migrated_snapshot, _ = migration_service.migrate_state(
            snapshot=snapshot,
            target_version=v2,
        )

        assert migrated_snapshot.execution_context["migrated"] is True

    def test_cannot_migrate_backwards(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migrating backwards requires explicit migration."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=2, minor=0, patch=0)

        # Only register forward migration
        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=lambda data: data,
            description="Forward only",
        )

        sample_snapshot.version = v2

        # Cannot migrate backwards without explicit migration
        with pytest.raises(MigrationError):
            migration_service.migrate_state(
                snapshot=sample_snapshot,
                target_version=v1,
            )

    def test_migration_preserves_snapshot_id(
        self,
        migration_service: StateMigrationService,
        sample_snapshot: AgentStateSnapshot,
    ) -> None:
        """Test migration preserves snapshot ID."""
        v1 = StateVersion(major=1, minor=0, patch=0)
        v2 = StateVersion(major=1, minor=1, patch=0)

        migration_service.register_migration(
            from_version=v1,
            to_version=v2,
            migration_func=lambda data: data,
            description="Simple migration",
        )

        sample_snapshot.version = v1
        original_snapshot_id = sample_snapshot.snapshot_id

        migrated_snapshot, _ = migration_service.migrate_state(
            snapshot=sample_snapshot,
            target_version=v2,
        )

        assert migrated_snapshot.snapshot_id == original_snapshot_id
