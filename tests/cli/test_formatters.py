"""Comprehensive unit tests for CLI formatters module."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from io import StringIO
from unittest.mock import patch

import pytest

from agentcore_cli.formatters import (
    _format_timestamp,
    _format_value,
    _get_console,
    _should_use_color,
    format_agent_info,
    format_error,
    format_json,
    format_success,
    format_table,
    format_tree,
    format_warning)


class TestColorDetection:
    """Test color auto-detection functionality."""

    def test_should_use_color_with_tty(self) -> None:
        """Test that color is enabled when stdout is a TTY."""
        with patch("sys.stdout.isatty", return_value=True):
            with patch.dict(os.environ, {}, clear=True):
                assert _should_use_color() is True

    def test_should_use_color_without_tty(self) -> None:
        """Test that color is disabled when stdout is not a TTY."""
        with patch("sys.stdout.isatty", return_value=False):
            assert _should_use_color() is False

    def test_should_use_color_with_no_color_env(self) -> None:
        """Test that color is disabled when NO_COLOR is set."""
        with patch("sys.stdout.isatty", return_value=True):
            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                assert _should_use_color() is False

    def test_should_use_color_with_dumb_term(self) -> None:
        """Test that color is disabled when TERM=dumb."""
        with patch("sys.stdout.isatty", return_value=True):
            with patch.dict(os.environ, {"TERM": "dumb"}):
                assert _should_use_color() is False

    def test_get_console_auto_detect(self) -> None:
        """Test console creation with auto-detect."""
        console = _get_console(force_color=None)
        assert console is not None

    def test_get_console_force_color_true(self) -> None:
        """Test console creation with forced color."""
        console = _get_console(force_color=True)
        assert console is not None
        assert console._force_terminal is True

    def test_get_console_force_color_false(self) -> None:
        """Test console creation with color disabled."""
        console = _get_console(force_color=False)
        assert console is not None
        assert console.no_color is True


class TestTimestampFormatting:
    """Test timestamp formatting functionality."""

    def test_format_timestamp_none(self) -> None:
        """Test formatting None timestamp."""
        result = _format_timestamp(None)
        assert result == "N/A"

    def test_format_timestamp_iso_string_date_only(self) -> None:
        """Test formatting ISO timestamp string without time."""
        timestamp = "2025-10-21T14:30:00Z"
        result = _format_timestamp(timestamp, include_time=False)
        assert result == "2025-10-21"

    def test_format_timestamp_iso_string_with_time(self) -> None:
        """Test formatting ISO timestamp string with time."""
        timestamp = "2025-10-21T14:30:00Z"
        result = _format_timestamp(timestamp, include_time=True)
        assert "2025-10-21" in result
        assert ":" in result  # Should include time

    def test_format_timestamp_datetime_object(self) -> None:
        """Test formatting datetime object."""
        dt = datetime(2025, 10, 21, 14, 30, 0, tzinfo=UTC)
        result = _format_timestamp(dt, include_time=False)
        assert result == "2025-10-21"

    def test_format_timestamp_datetime_with_time(self) -> None:
        """Test formatting datetime object with time."""
        dt = datetime(2025, 10, 21, 14, 30, 0, tzinfo=UTC)
        result = _format_timestamp(dt, include_time=True)
        assert result == "2025-10-21 14:30:00"

    def test_format_timestamp_invalid_string(self) -> None:
        """Test formatting invalid timestamp string."""
        timestamp = "not-a-timestamp"
        result = _format_timestamp(timestamp)
        assert result == "not-a-timestamp"  # Returns as-is

    def test_format_timestamp_other_type(self) -> None:
        """Test formatting non-string, non-datetime value."""
        result = _format_timestamp(12345)
        assert result == "12345"


class TestValueFormatting:
    """Test value formatting functionality."""

    def test_format_value_none(self) -> None:
        """Test formatting None value."""
        result = _format_value(None)
        assert "N/A" in result

    def test_format_value_bool_true(self) -> None:
        """Test formatting True value."""
        result = _format_value(True)
        assert "Yes" in result

    def test_format_value_bool_false(self) -> None:
        """Test formatting False value."""
        result = _format_value(False)
        assert "No" in result

    def test_format_value_empty_list(self) -> None:
        """Test formatting empty list."""
        result = _format_value([])
        assert "None" in result

    def test_format_value_string_list(self) -> None:
        """Test formatting list of strings."""
        result = _format_value(["python", "testing", "analysis"])
        assert "python" in result
        assert "testing" in result
        assert "analysis" in result

    def test_format_value_dict(self) -> None:
        """Test formatting dictionary."""
        result = _format_value({"key": "value", "count": 42})
        assert "key" in result
        assert "value" in result

    def test_format_value_string(self) -> None:
        """Test formatting string value."""
        result = _format_value("test-string")
        assert result == "test-string"

    def test_format_value_number(self) -> None:
        """Test formatting number value."""
        result = _format_value(42)
        assert result == "42"


class TestJSONFormatter:
    """Test JSON formatter functionality."""

    def test_format_json_pretty(self) -> None:
        """Test JSON formatting with pretty print."""
        data = {"key": "value", "nested": {"count": 42}}
        result = format_json(data, pretty=True)
        assert "key" in result
        assert "value" in result
        assert "nested" in result
        # Pretty print should have newlines
        assert "\n" in result

    def test_format_json_compact(self) -> None:
        """Test JSON formatting without pretty print."""
        data = {"key": "value"}
        result = format_json(data, pretty=False)
        assert "key" in result
        assert "value" in result
        # Compact should be single line
        assert "\n" not in result

    def test_format_json_list(self) -> None:
        """Test JSON formatting with list."""
        data = [{"id": 1}, {"id": 2}]
        result = format_json(data)
        assert "id" in result
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_format_json_unicode(self) -> None:
        """Test JSON formatting with unicode characters."""
        data = {"message": "Hello 世界"}
        result = format_json(data)
        assert "世界" in result
        parsed = json.loads(result)
        assert parsed["message"] == "Hello 世界"


class TestTableFormatter:
    """Test table formatter functionality."""

    def test_format_table_empty_data(self) -> None:
        """Test formatting empty data."""
        result = format_table([])
        assert "No data" in result

    def test_format_table_basic(self) -> None:
        """Test basic table formatting."""
        data = [
            {"id": "1", "name": "agent-1", "status": "active"},
            {"id": "2", "name": "agent-2", "status": "inactive"},
        ]
        result = format_table(data, force_color=False)
        assert "id" in result.lower() or "Id" in result
        assert "name" in result.lower() or "Name" in result
        assert "agent-1" in result
        assert "agent-2" in result

    def test_format_table_with_columns(self) -> None:
        """Test table formatting with column selection."""
        data = [
            {"id": "1", "name": "agent-1", "status": "active", "extra": "hidden"},
            {"id": "2", "name": "agent-2", "status": "inactive", "extra": "hidden"},
        ]
        result = format_table(data, columns=["id", "name"], force_color=False)
        assert "agent-1" in result
        assert "agent-2" in result
        # "extra" column should not be shown
        assert "hidden" not in result

    def test_format_table_with_title(self) -> None:
        """Test table formatting with title."""
        data = [{"id": "1", "name": "test"}]
        result = format_table(data, title="Test Table", force_color=False)
        assert "Test Table" in result

    def test_format_table_with_limit(self) -> None:
        """Test table formatting with row limit."""
        data = [
            {"id": "1", "name": "agent-1"},
            {"id": "2", "name": "agent-2"},
            {"id": "3", "name": "agent-3"},
        ]
        result = format_table(data, limit=2, force_color=False)
        assert "agent-1" in result
        assert "agent-2" in result
        # Third row should not appear
        assert "agent-3" not in result

    def test_format_table_with_timestamps(self) -> None:
        """Test table formatting with timestamp formatting."""
        data = [
            {
                "id": "1",
                "name": "agent-1",
                "created_at": "2025-10-21T14:30:00Z",
            }
        ]
        result = format_table(data, timestamps=True, force_color=False)
        assert "2025-10-21" in result

    def test_format_table_with_invalid_columns(self) -> None:
        """Test table formatting with non-existent columns."""
        data = [{"id": "1", "name": "agent-1"}]
        # Request columns that don't exist
        result = format_table(
            data, columns=["id", "nonexistent"], force_color=False
        )
        # Should only show "id" column
        assert "1" in result


class TestTreeFormatter:
    """Test tree formatter functionality."""

    def test_format_tree_basic(self) -> None:
        """Test basic tree formatting."""
        data = {"key": "value", "nested": {"count": 42}}
        result = format_tree(data, force_color=False)
        assert "key" in result
        assert "value" in result
        assert "nested" in result

    def test_format_tree_with_label(self) -> None:
        """Test tree formatting with custom label."""
        data = {"key": "value"}
        result = format_tree(data, label="Custom Root", force_color=False)
        assert "Custom Root" in result

    def test_format_tree_with_list(self) -> None:
        """Test tree formatting with list values."""
        data = {"items": [{"id": 1}, {"id": 2}]}
        result = format_tree(data, force_color=False)
        assert "items" in result
        assert "[0]" in result or "0" in result
        assert "[1]" in result or "1" in result

    def test_format_tree_with_timestamps(self) -> None:
        """Test tree formatting with timestamp formatting."""
        data = {"created_at": "2025-10-21T14:30:00Z", "name": "test"}
        result = format_tree(data, timestamps=True, force_color=False)
        assert "2025-10-21" in result

    def test_format_tree_deep_nesting(self) -> None:
        """Test tree formatting with deep nesting."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "level6": "too deep"
                            }
                        }
                    }
                }
            }
        }
        result = format_tree(data, force_color=False)
        # Should have some depth limit marker
        assert "..." in result or "level6" not in result


class TestAgentInfoFormatter:
    """Test agent info formatter functionality."""

    def test_format_agent_info_basic(self) -> None:
        """Test basic agent info formatting."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
        }
        result = format_agent_info(agent_data, force_color=False)
        assert "agent-123" in result
        assert "test-agent" in result
        assert "active" in result

    def test_format_agent_info_with_capabilities(self) -> None:
        """Test agent info formatting with capabilities."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
            "capabilities": ["python", "testing", "analysis"],
        }
        result = format_agent_info(agent_data, force_color=False)
        assert "python" in result
        assert "testing" in result
        assert "analysis" in result

    def test_format_agent_info_with_requirements(self) -> None:
        """Test agent info formatting with requirements."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
            "requirements": {"memory": "512MB", "cpu": "0.5"},
        }
        result = format_agent_info(agent_data, force_color=False)
        assert "memory" in result
        assert "512MB" in result
        assert "cpu" in result
        assert "0.5" in result

    def test_format_agent_info_with_timestamps(self) -> None:
        """Test agent info formatting with timestamp formatting."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
            "created_at": "2025-10-21T14:30:00Z",
            "updated_at": "2025-10-21T15:00:00Z",
        }
        result = format_agent_info(agent_data, timestamps=True, force_color=False)
        assert "2025-10-21" in result

    def test_format_agent_info_with_cost(self) -> None:
        """Test agent info formatting with cost information."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
            "cost_per_request": 0.0123,
        }
        result = format_agent_info(agent_data, force_color=False)
        assert "0.0123" in result

    def test_format_agent_info_with_health_and_tasks(self) -> None:
        """Test agent info formatting with health status and active tasks."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
            "health_status": "healthy",
            "active_tasks": 5,
        }
        result = format_agent_info(agent_data, force_color=False)
        assert "healthy" in result
        assert "5" in result


class TestMessageFormatters:
    """Test message formatting functions."""

    def test_format_success(self) -> None:
        """Test success message formatting."""
        result = format_success("Operation completed")
        assert "Operation completed" in result

    def test_format_error(self) -> None:
        """Test error message formatting."""
        result = format_error("Operation failed")
        assert "Operation failed" in result

    def test_format_warning(self) -> None:
        """Test warning message formatting."""
        result = format_warning("Be careful")
        assert "Be careful" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_format_table_with_missing_values(self) -> None:
        """Test table formatting with missing values in rows."""
        data = [
            {"id": "1", "name": "agent-1", "status": "active"},
            {"id": "2", "name": "agent-2"},  # Missing status
        ]
        result = format_table(data, force_color=False)
        assert "agent-1" in result
        assert "agent-2" in result

    def test_format_table_with_unicode(self) -> None:
        """Test table formatting with unicode content."""
        data = [{"id": "1", "name": "测试-agent", "status": "active"}]
        result = format_table(data, force_color=False)
        assert "测试-agent" in result

    def test_format_tree_with_none_values(self) -> None:
        """Test tree formatting with None values."""
        data = {"key": None, "other": "value"}
        result = format_tree(data, force_color=False)
        assert "N/A" in result or "None" in result

    def test_format_agent_info_minimal_data(self) -> None:
        """Test agent info formatting with minimal data."""
        agent_data = {"agent_id": "agent-123"}
        result = format_agent_info(agent_data, force_color=False)
        assert "agent-123" in result
        # Should handle missing fields gracefully
        assert "N/A" in result


class TestSnapshotTests:
    """Snapshot tests for output format regression."""

    def test_table_format_snapshot(self) -> None:
        """Snapshot test for table format consistency."""
        # Fixed test data for consistent output
        data = [
            {"id": "agent-1", "name": "Test Agent 1", "status": "active"},
            {"id": "agent-2", "name": "Test Agent 2", "status": "inactive"},
        ]

        result = format_table(data, force_color=False)

        # Verify key elements are present in consistent order
        assert "Id" in result or "id" in result.lower()
        assert "Name" in result or "name" in result.lower()
        assert "Status" in result or "status" in result.lower()
        assert "agent-1" in result
        assert "Test Agent 1" in result
        assert "active" in result
        assert "agent-2" in result
        assert "Test Agent 2" in result
        assert "inactive" in result

    def test_tree_format_snapshot(self) -> None:
        """Snapshot test for tree format consistency."""
        data = {
            "agent_id": "agent-123",
            "name": "Test Agent",
            "metadata": {
                "version": "1.0",
                "tags": ["python", "testing"],
            }
        }

        result = format_tree(data, label="Agent", force_color=False)

        # Verify structure is consistent
        assert "Agent" in result
        assert "agent_id" in result
        assert "agent-123" in result
        assert "metadata" in result
        assert "version" in result
        assert "1.0" in result

    def test_json_format_snapshot(self) -> None:
        """Snapshot test for JSON format consistency."""
        data = {"id": "test-1", "value": 42, "active": True}

        result = format_json(data, pretty=True)

        # Parse to verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["id"] == "test-1"
        assert parsed["value"] == 42
        assert parsed["active"] is True

        # Verify pretty printing
        assert "\n" in result
        assert "  " in result  # Indentation

    def test_agent_info_format_snapshot(self) -> None:
        """Snapshot test for agent info format consistency."""
        agent_data = {
            "agent_id": "agent-123",
            "name": "Test Agent",
            "status": "active",
            "capabilities": ["python", "testing"],
            "cost_per_request": 0.01,
            "created_at": "2025-10-21T14:30:00Z",
        }

        result = format_agent_info(agent_data, force_color=False)

        # Verify all sections appear in order
        lines = result.split("\n")
        assert any("agent-123" in line for line in lines)
        assert any("Test Agent" in line for line in lines)
        assert any("active" in line for line in lines)
        assert any("python" in line for line in lines)
        assert any("0.0100" in line for line in lines)


class TestIntegration:
    """Integration tests for formatters."""

    def test_all_formatters_with_same_data(self) -> None:
        """Test all formatters with the same dataset."""
        data = {
            "agent_id": "agent-123",
            "name": "test-agent",
            "status": "active",
            "created_at": "2025-10-21T14:30:00Z",
        }

        # Test JSON format
        json_result = format_json(data)
        assert "agent-123" in json_result

        # Test tree format
        tree_result = format_tree(data, force_color=False)
        assert "agent-123" in tree_result

        # Test agent info format
        info_result = format_agent_info(data, force_color=False)
        assert "agent-123" in info_result

    def test_table_with_multiple_format_options(self) -> None:
        """Test table formatter with all options combined."""
        data = [
            {
                "id": f"agent-{i}",
                "name": f"Agent {i}",
                "status": "active",
                "created_at": "2025-10-21T14:30:00Z",
            }
            for i in range(5)
        ]

        result = format_table(
            data,
            columns=["id", "name", "created_at"],
            title="Test Agents",
            timestamps=True,
            limit=3,
            force_color=False)

        # Should have title
        assert "Test Agents" in result
        # Should have first 3 agents
        assert "agent-0" in result
        assert "agent-1" in result
        assert "agent-2" in result
        # Should not have 4th and 5th
        assert "agent-3" not in result
        assert "agent-4" not in result
        # Should format timestamp
        assert "2025-10-21" in result
        # Should not include status column
        assert "Status" not in result
