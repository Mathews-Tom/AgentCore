"""Integration tests for output format validation across all commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("agentcore_cli.commands.agent.Config") as mock_config_class:
        config = MagicMock()
        config.api.url = "http://localhost:8001"
        config.api.timeout = 30
        config.api.retries = 3
        config.api.verify_ssl = True
        config.auth.type = "none"
        config.auth.token = None
        mock_config_class.load.return_value = config
        yield config


@pytest.fixture
def mock_client():
    """Mock AgentCore client."""
    with patch("agentcore_cli.commands.agent.AgentCoreClient") as mock_client_class:
        client = MagicMock()
        mock_client_class.return_value = client
        yield client


class TestAgentCommandsOutputFormat:
    """Test output format validation for agent commands."""

    def test_agent_list_table_format(self, mock_config, mock_client):
        """Test agent list command produces valid table format."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "test-agent",
                    "status": "active",
                    "capabilities": ["python", "testing"],
                    "cost_per_request": 0.01,
                    "registered_at": "2025-10-21T10:00:00Z",
                },
            ]
        }

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        # Verify table structure
        assert "agent-1" in result.stdout
        assert "test-agent" in result.stdout
        assert "active" in result.stdout

    def test_agent_list_json_format(self, mock_config, mock_client):
        """Test agent list command produces valid JSON format."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "test-agent",
                    "status": "active",
                },
            ]
        }

        result = runner.invoke(app, ["agent", "list", "--json"])

        assert result.exit_code == 0

        # Verify JSON is valid and parseable
        output = json.loads(result.stdout)
        assert isinstance(output, list)
        assert len(output) == 1
        assert output[0]["agent_id"] == "agent-1"

    def test_agent_info_tree_format(self, mock_config, mock_client):
        """Test agent info command produces valid tree/detailed format."""
        mock_client.call.return_value = {
            "agent_id": "agent-1",
            "name": "test-agent",
            "status": "active",
            "capabilities": ["python", "testing"],
            "requirements": {"memory": "512MB"},
            "cost_per_request": 0.01,
            "created_at": "2025-10-21T10:00:00Z",
        }

        result = runner.invoke(app, ["agent", "info", "agent-1"])

        assert result.exit_code == 0
        # Verify tree/detailed structure
        assert "Agent ID: agent-1" in result.stdout
        assert "Name: test-agent" in result.stdout
        assert "Capabilities:" in result.stdout
        assert "Requirements:" in result.stdout

    def test_agent_info_json_format(self, mock_config, mock_client):
        """Test agent info command produces valid JSON format."""
        mock_client.call.return_value = {
            "agent_id": "agent-1",
            "name": "test-agent",
            "status": "active",
        }

        result = runner.invoke(app, ["agent", "info", "agent-1", "--json"])

        assert result.exit_code == 0

        # Verify JSON is valid
        output = json.loads(result.stdout)
        assert output["agent_id"] == "agent-1"
        assert output["name"] == "test-agent"


class TestTaskCommandsOutputFormat:
    """Test output format validation for task commands."""

    def test_task_list_table_format(self, mock_config):
        """Test task list command produces valid table format."""
        with patch("agentcore_cli.commands.task.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "tasks": [
                    {
                        "task_id": "task-1",
                        "type": "analysis",
                        "status": "running",
                        "priority": "high",
                        "created_at": "2025-10-21T10:00:00Z",
                    },
                ]
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["task", "list"])

            assert result.exit_code == 0
            # Verify table contains task data
            assert "task-1" in result.stdout
            assert "analysis" in result.stdout

    def test_task_list_json_format(self, mock_config):
        """Test task list command produces valid JSON format."""
        with patch("agentcore_cli.commands.task.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "tasks": [
                    {
                        "task_id": "task-1",
                        "type": "analysis",
                        "status": "running",
                    },
                ]
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["task", "list", "--json"])

            assert result.exit_code == 0

            # Verify JSON is valid
            output = json.loads(result.stdout)
            assert isinstance(output, list)
            assert output[0]["task_id"] == "task-1"

    def test_task_status_detailed_format(self, mock_config):
        """Test task status command produces valid detailed format."""
        with patch("agentcore_cli.commands.task.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "task_id": "task-1",
                "type": "analysis",
                "status": "running",
                "progress": 50,
                "agent_id": "agent-1",
                "created_at": "2025-10-21T10:00:00Z",
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["task", "status", "task-1"])

            assert result.exit_code == 0
            # Verify detailed structure
            assert "Task ID: task-1" in result.stdout
            assert "Type: analysis" in result.stdout

    def test_task_result_json_format(self, mock_config):
        """Test task result command produces valid JSON format."""
        with patch("agentcore_cli.commands.task.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "task_id": "task-1",
                "status": "completed",
                "output": {"success": True},
                "artifacts": [],
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["task", "result", "task-1", "--json"])

            assert result.exit_code == 0

            # Verify JSON is valid
            output = json.loads(result.stdout)
            assert "task_id" in output
            assert "output" in output


class TestSessionCommandsOutputFormat:
    """Test output format validation for session commands."""

    def test_session_list_table_format(self, mock_config):
        """Test session list command produces valid table format."""
        with patch("agentcore_cli.commands.session.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "name": "test-session",
                        "status": "active",
                        "tasks_count": 5,
                        "created_at": "2025-10-21T10:00:00Z",
                    },
                ]
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["session", "list"])

            assert result.exit_code == 0
            # Verify table structure
            assert "session-1" in result.stdout
            assert "test-session" in result.stdout

    def test_session_list_json_format(self, mock_config):
        """Test session list command produces valid JSON format."""
        with patch("agentcore_cli.commands.session.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "name": "test-session",
                        "status": "active",
                    },
                ]
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["session", "list", "--json"])

            assert result.exit_code == 0

            # Verify JSON is valid
            output = json.loads(result.stdout)
            assert isinstance(output, list)
            assert output[0]["session_id"] == "session-1"

    def test_session_info_detailed_format(self, mock_config):
        """Test session info command produces valid detailed format."""
        with patch("agentcore_cli.commands.session.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "session_id": "session-1",
                "name": "test-session",
                "status": "active",
                "description": "Test description",
                "tags": ["test", "backend"],
                "tasks_count": 5,
                "agents_count": 2,
                "created_at": "2025-10-21T10:00:00Z",
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["session", "info", "session-1"])

            assert result.exit_code == 0
            # Verify detailed structure
            assert "Session ID: session-1" in result.stdout
            assert "Name: test-session" in result.stdout
            assert "Tasks: 5" in result.stdout


class TestWorkflowCommandsOutputFormat:
    """Test output format validation for workflow commands."""

    def test_workflow_list_table_format(self, mock_config):
        """Test workflow list command produces valid table format."""
        with patch("agentcore_cli.commands.workflow.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "workflows": [
                    {
                        "workflow_id": "workflow-1",
                        "name": "test-workflow",
                        "status": "running",
                        "tasks_total": 3,
                        "tasks_completed": 1,
                    },
                ]
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["workflow", "list"])

            assert result.exit_code == 0
            # Verify table structure
            assert "workflow-1" in result.stdout
            assert "test-workflow" in result.stdout

    def test_workflow_list_json_format(self, mock_config):
        """Test workflow list command produces valid JSON format."""
        with patch("agentcore_cli.commands.workflow.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "workflows": [
                    {
                        "workflow_id": "workflow-1",
                        "name": "test-workflow",
                        "status": "running",
                    },
                ]
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["workflow", "list", "--json"])

            assert result.exit_code == 0

            # Verify JSON is valid
            output = json.loads(result.stdout)
            assert isinstance(output, list)
            assert output[0]["workflow_id"] == "workflow-1"

    def test_workflow_visualize_tree_format(self, mock_config):
        """Test workflow visualize command produces valid tree format."""
        with patch("agentcore_cli.commands.workflow.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {
                "workflow_id": "workflow-1",
                "name": "test-workflow",
                "status": "running",
                "tasks": [
                    {"name": "task-1", "type": "test", "status": "completed", "depends_on": []},
                    {"name": "task-2", "type": "test", "status": "running", "depends_on": ["task-1"]},
                ],
            }
            mock_client_class.return_value = client

            result = runner.invoke(app, ["workflow", "visualize", "workflow-1"])

            assert result.exit_code == 0
            # Verify tree structure contains tasks
            assert "task-1" in result.stdout
            assert "task-2" in result.stdout


class TestConfigCommandsOutputFormat:
    """Test output format validation for config commands."""

    def test_config_show_detailed_format(self, tmp_path):
        """Test config show command produces valid detailed format."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Verify detailed config structure
        assert "Configuration" in result.stdout or "api" in result.stdout

    def test_config_show_json_format(self, tmp_path):
        """Test config show command produces valid JSON format."""
        import os
        os.chdir(tmp_path)

        # Initialize config first
        runner.invoke(app, ["config", "init"])

        result = runner.invoke(app, ["config", "show", "--json"])

        assert result.exit_code == 0

        # Verify JSON is valid
        output = json.loads(result.stdout)
        assert "api" in output


class TestCrossCommandOutputConsistency:
    """Test output format consistency across different commands."""

    def test_json_output_always_parseable(self, mock_config, mock_client):
        """Test that all --json outputs are valid JSON."""
        # Test agent list
        mock_client.call.return_value = {"agents": []}
        result = runner.invoke(app, ["agent", "list", "--json"])
        assert result.exit_code == 0
        json.loads(result.stdout)  # Should not raise

        # Test agent search
        mock_client.call.return_value = {"agents": []}
        result = runner.invoke(app, ["agent", "search", "--capability", "test", "--json"])
        assert result.exit_code == 0
        json.loads(result.stdout)  # Should not raise

    def test_table_output_has_headers(self, mock_config, mock_client):
        """Test that all table outputs have proper headers."""
        # Test agent list
        mock_client.call.return_value = {
            "agents": [
                {"agent_id": "agent-1", "name": "test", "status": "active"},
            ]
        }
        result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        # Headers should be present (in some form)
        assert "agent" in result.stdout.lower() or "id" in result.stdout.lower()

    def test_empty_results_handling(self, mock_config, mock_client):
        """Test that empty results are handled consistently across commands."""
        # Agent list with no results
        mock_client.call.return_value = {"agents": []}
        result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        assert "No agents found" in result.stdout

        # Agent search with no results
        mock_client.call.return_value = {"agents": []}
        result = runner.invoke(app, ["agent", "search", "--capability", "nonexistent"])
        assert result.exit_code == 0
        assert "No agents found" in result.stdout


class TestOutputFormatWithSpecialCharacters:
    """Test output format handling of special characters and unicode."""

    def test_unicode_in_table_output(self, mock_config, mock_client):
        """Test that table output handles unicode correctly."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "测试-agent-日本語",
                    "status": "active",
                },
            ]
        }

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        assert "测试-agent-日本語" in result.stdout

    def test_unicode_in_json_output(self, mock_config, mock_client):
        """Test that JSON output handles unicode correctly."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "测试-agent-日本語",
                    "status": "active",
                },
            ]
        }

        result = runner.invoke(app, ["agent", "list", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output[0]["name"] == "测试-agent-日本語"

    def test_special_characters_in_output(self, mock_config, mock_client):
        """Test that output handles special characters correctly."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "agent-with-\"quotes\"-and-'apostrophes'",
                    "status": "active",
                },
            ]
        }

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        # Should not crash on special characters


class TestOutputFormatWithLargeData:
    """Test output format handling of large datasets."""

    def test_table_with_many_rows(self, mock_config, mock_client):
        """Test table output with many rows."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": f"agent-{i}",
                    "name": f"agent-{i}",
                    "status": "active",
                }
                for i in range(50)
            ]
        }

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        # Should handle large dataset without crashing
        assert "agent-0" in result.stdout
        assert "agent-49" in result.stdout

    def test_json_with_many_items(self, mock_config, mock_client):
        """Test JSON output with many items."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": f"agent-{i}",
                    "name": f"agent-{i}",
                    "status": "active",
                }
                for i in range(100)
            ]
        }

        result = runner.invoke(app, ["agent", "list", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert len(output) == 100


class TestOutputFormatPipeline:
    """Test output format in pipeline/scripting scenarios."""

    def test_json_output_for_scripting(self, mock_config, mock_client):
        """Test that JSON output is suitable for scripting."""
        mock_client.call.return_value = {
            "agents": [
                {"agent_id": "agent-1", "name": "test", "status": "active"},
            ]
        }

        result = runner.invoke(app, ["agent", "list", "--json"])

        assert result.exit_code == 0

        # JSON should be on single line or easily parseable
        output = json.loads(result.stdout)
        assert isinstance(output, list)

        # Should be able to extract specific fields
        agent_ids = [agent["agent_id"] for agent in output]
        assert "agent-1" in agent_ids

    def test_quiet_mode_minimal_output(self, mock_config, mock_client):
        """Test that quiet mode produces minimal output where applicable."""
        # This would test if commands support --quiet flag
        # For now, just verify normal operation doesn't have excessive output
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        # Output should be concise
        lines = result.stdout.split("\n")
        assert len(lines) < 20  # Reasonable for empty result
