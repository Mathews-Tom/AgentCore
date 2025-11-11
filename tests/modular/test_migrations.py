"""
Tests for Modular Agent Core Database Migrations

Validates migration structure, DDL correctness, and rollback functionality.
"""

from __future__ import annotations

import pytest
import re
from pathlib import Path


# Get project root directory (two levels up from this test file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MIGRATION_FILE = PROJECT_ROOT / "alembic" / "versions" / "e247cdc4c183_add_modular_executions_tables.py"


class TestModularExecutionsMigration:
    """Test the modular_executions migration."""

    @pytest.fixture
    def migration_file(self) -> str:
        """Load the migration file content."""
        return MIGRATION_FILE.read_text()

    def test_migration_file_exists(self) -> None:
        """Test that migration file exists."""
        assert MIGRATION_FILE.exists(), f"Migration file not found at {MIGRATION_FILE}"

    def test_migration_has_correct_revision(self, migration_file: str) -> None:
        """Test migration has correct revision ID."""
        assert "revision: str = 'e247cdc4c183'" in migration_file
        assert "down_revision: Union[str, Sequence[str], None] = 'c03db99da40b'" in migration_file

    def test_migration_creates_enum_types(self, migration_file: str) -> None:
        """Test migration creates required enum types."""
        # Check for planstatus enum
        assert "CREATE TYPE planstatus" in migration_file
        assert "pending" in migration_file
        assert "in_progress" in migration_file
        assert "completed" in migration_file
        assert "failed" in migration_file
        assert "cancelled" in migration_file

        # Check for stepstatus enum
        assert "CREATE TYPE stepstatus" in migration_file
        assert "skipped" in migration_file

        # Check for moduletype enum
        assert "CREATE TYPE moduletype" in migration_file
        assert "planner" in migration_file
        assert "executor" in migration_file
        assert "verifier" in migration_file
        assert "generator" in migration_file

    def test_migration_creates_modular_executions_table(self, migration_file: str) -> None:
        """Test migration creates modular_executions table."""
        assert "create_table(" in migration_file
        assert "'modular_executions'" in migration_file

        # Check required columns
        required_columns = [
            "id",
            "query",
            "plan_id",
            "iterations",
            "final_result",
            "status",
            "error",
            "created_at",
            "completed_at",
            "metadata",
        ]

        for column in required_columns:
            assert f"'{column}'" in migration_file, f"Column {column} not found"

        # Check indexes
        assert "idx_modular_executions_status" in migration_file
        assert "idx_modular_executions_created_at" in migration_file
        assert "idx_modular_executions_plan_id" in migration_file

    def test_migration_creates_execution_plans_table(self, migration_file: str) -> None:
        """Test migration creates execution_plans table."""
        assert "'execution_plans'" in migration_file

        # Check required columns
        required_columns = [
            "plan_id",
            "execution_id",
            "plan_data",
            "status",
            "max_iterations",
            "current_iteration",
            "success_criteria",
            "final_result",
            "error",
            "created_at",
            "started_at",
            "completed_at",
            "duration_seconds",
            "total_estimated_cost",
            "actual_cost",
            "metadata",
        ]

        for column in required_columns:
            assert f"'{column}'" in migration_file, f"Column {column} not found"

        # Check foreign key
        assert "ForeignKeyConstraint" in migration_file
        assert "execution_id" in migration_file
        assert "modular_executions.id" in migration_file
        assert "ondelete='CASCADE'" in migration_file

        # Check indexes
        assert "idx_execution_plans_execution_id" in migration_file
        assert "idx_execution_plans_status" in migration_file

    def test_migration_creates_plan_steps_table(self, migration_file: str) -> None:
        """Test migration creates plan_steps table."""
        assert "'plan_steps'" in migration_file

        # Check required columns
        required_columns = [
            "step_id",
            "plan_id",
            "action",
            "parameters",
            "status",
            "dependencies",
            "tool_requirements",
            "started_at",
            "completed_at",
            "duration_seconds",
            "retry_count",
            "max_retries",
            "result",
            "error",
            "estimated_cost",
            "actual_cost",
            "metadata",
        ]

        for column in required_columns:
            assert f"'{column}'" in migration_file, f"Column {column} not found"

        # Check foreign key
        assert "execution_plans.plan_id" in migration_file

        # Check unique constraint
        assert "UniqueConstraint" in migration_file
        assert "uq_plan_step" in migration_file

        # Check indexes
        assert "idx_plan_steps_plan_id" in migration_file
        assert "idx_plan_steps_status" in migration_file
        assert "idx_plan_steps_step_id" in migration_file

    def test_migration_creates_module_transitions_table(self, migration_file: str) -> None:
        """Test migration creates module_transitions table."""
        assert "'module_transitions'" in migration_file

        # Check required columns
        required_columns = [
            "transition_id",
            "plan_id",
            "iteration",
            "from_module",
            "to_module",
            "timestamp",
            "reason",
            "trigger",
            "data",
            "duration_in_from_module",
            "metadata",
        ]

        for column in required_columns:
            assert f"'{column}'" in migration_file, f"Column {column} not found"

        # Check foreign key
        assert "execution_plans.plan_id" in migration_file

        # Check unique constraint on transition_id
        assert "unique=True" in migration_file

        # Check indexes
        assert "idx_module_transitions_plan_id" in migration_file
        assert "idx_module_transitions_timestamp" in migration_file
        assert "idx_module_transitions_from_to" in migration_file

    def test_migration_has_downgrade_function(self, migration_file: str) -> None:
        """Test migration has proper downgrade function."""
        assert "def downgrade() -> None:" in migration_file

        # Check that downgrade drops tables in reverse order
        assert "drop_table('module_transitions')" in migration_file
        assert "drop_table('plan_steps')" in migration_file
        assert "drop_table('execution_plans')" in migration_file
        assert "drop_table('modular_executions')" in migration_file

        # Check that downgrade drops enum types
        assert "DROP TYPE IF EXISTS moduletype" in migration_file
        assert "DROP TYPE IF EXISTS stepstatus" in migration_file
        assert "DROP TYPE IF EXISTS planstatus" in migration_file

    def test_migration_uses_jsonb_for_json_columns(self, migration_file: str) -> None:
        """Test migration uses JSONB for JSON columns (PostgreSQL-specific)."""
        # Check that JSONB is used (more efficient than JSON in PostgreSQL)
        jsonb_columns = [
            "final_result",
            "metadata",
            "plan_data",
            "success_criteria",
            "parameters",
            "dependencies",
            "tool_requirements",
            "result",
            "data",
        ]

        for column in jsonb_columns:
            # Should find postgresql.JSONB usage
            pattern = f"'{column}'.*postgresql\\.JSONB"
            assert re.search(pattern, migration_file, re.DOTALL), \
                f"Column {column} should use postgresql.JSONB"

    def test_migration_has_proper_cascade_deletes(self, migration_file: str) -> None:
        """Test migration has CASCADE deletes for referential integrity."""
        # Count CASCADE deletes - should have 4:
        # 1. execution_plans -> modular_executions
        # 2. plan_steps -> execution_plans
        # 3. module_transitions -> execution_plans
        cascade_count = migration_file.count("ondelete='CASCADE'")
        assert cascade_count == 3, f"Expected 3 CASCADE deletes, found {cascade_count}"

    def test_migration_has_default_values(self, migration_file: str) -> None:
        """Test migration sets appropriate default values."""
        # Check server defaults
        assert "server_default='0'" in migration_file  # iterations, retry_count, current_iteration
        assert "server_default='10'" in migration_file  # max_iterations
        assert "server_default='3'" in migration_file  # max_retries
        assert "server_default='0.0'" in migration_file  # cost fields

    def test_migration_indexes_frequently_queried_columns(self, migration_file: str) -> None:
        """Test migration creates indexes on frequently queried columns."""
        # Should have indexes on:
        # - status columns (for filtering)
        # - created_at/timestamp (for sorting)
        # - plan_id (for joins)
        # - foreign keys

        expected_indexes = [
            "idx_modular_executions_status",
            "idx_modular_executions_created_at",
            "idx_modular_executions_plan_id",
            "idx_execution_plans_execution_id",
            "idx_execution_plans_status",
            "idx_plan_steps_plan_id",
            "idx_plan_steps_status",
            "idx_plan_steps_step_id",
            "idx_module_transitions_plan_id",
            "idx_module_transitions_timestamp",
            "idx_module_transitions_from_to",
        ]

        for index_name in expected_indexes:
            assert index_name in migration_file, f"Index {index_name} not found"

    def test_migration_table_order_respects_foreign_keys(self, migration_file: str) -> None:
        """Test tables are created in correct order (parent before child)."""
        # Extract table creation order
        tables = re.findall(r"create_table\(\s*'(\w+)'", migration_file)

        # Check order: modular_executions -> execution_plans -> plan_steps/module_transitions
        assert tables.index('modular_executions') < tables.index('execution_plans')
        assert tables.index('execution_plans') < tables.index('plan_steps')
        assert tables.index('execution_plans') < tables.index('module_transitions')

    def test_migration_downgrade_order_reverse_of_upgrade(self, migration_file: str) -> None:
        """Test downgrade drops tables in reverse order of creation."""
        # Extract drop order
        drops = re.findall(r"drop_table\('(\w+)'\)", migration_file)

        # Should be reverse of creation order
        assert drops == [
            'module_transitions',
            'plan_steps',
            'execution_plans',
            'modular_executions'
        ]
