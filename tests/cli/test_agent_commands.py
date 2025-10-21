"""Unit tests for agent commands."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from agentcore_cli.exceptions import (
    AuthenticationError,
    ConnectionError as CliConnectionError,
    JsonRpcError,
)
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


class TestAgentRegister:
    """Tests for agent register command."""

    def test_register_success(self, mock_config, mock_client):
        """Test successful agent registration."""
        mock_client.call.return_value = {
            "agent_id": "agent-12345",
            "name": "test-agent",
            "capabilities": ["python", "testing"],
        }

        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python,testing",
        ])

        assert result.exit_code == 0
        assert "Agent registered: agent-12345" in result.stdout
        mock_client.call.assert_called_once_with("agent.register", {
            "name": "test-agent",
            "capabilities": ["python", "testing"],
            "cost_per_request": 0.01,
            "requirements": {},
        })

    def test_register_with_requirements(self, mock_config, mock_client):
        """Test agent registration with requirements."""
        mock_client.call.return_value = {"agent_id": "agent-12345"}

        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python",
            "--requirements", '{"memory": "512MB"}',
            "--cost-per-request", "0.05",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("agent.register", {
            "name": "test-agent",
            "capabilities": ["python"],
            "cost_per_request": 0.05,
            "requirements": {"memory": "512MB"},
        })

    def test_register_json_output(self, mock_config, mock_client):
        """Test agent registration with JSON output."""
        mock_client.call.return_value = {
            "agent_id": "agent-12345",
            "name": "test-agent",
        }

        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python",
            "--json",
        ])

        assert result.exit_code == 0
        assert '"agent_id": "agent-12345"' in result.stdout

    def test_register_empty_capabilities(self, mock_config, mock_client):
        """Test registration with empty capabilities."""
        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "",
        ])

        assert result.exit_code == 2
        assert "At least one capability is required" in result.stdout

    def test_register_invalid_requirements_json(self, mock_config, mock_client):
        """Test registration with invalid requirements JSON."""
        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python",
            "--requirements", "invalid-json",
        ])

        assert result.exit_code == 2
        assert "Invalid JSON in requirements" in result.stdout

    def test_register_requirements_not_dict(self, mock_config, mock_client):
        """Test registration with requirements not being a dict."""
        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python",
            "--requirements", '["not", "a", "dict"]',
        ])

        assert result.exit_code == 2
        assert "Requirements must be a JSON object" in result.stdout

    def test_register_connection_error(self, mock_config, mock_client):
        """Test registration with connection error."""
        mock_client.call.side_effect = CliConnectionError("Cannot connect")

        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python",
        ])

        assert result.exit_code == 3
        assert "Cannot connect" in result.stdout

    def test_register_authentication_error(self, mock_config, mock_client):
        """Test registration with authentication error."""
        mock_client.call.side_effect = AuthenticationError("Auth failed")

        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", "python",
        ])

        assert result.exit_code == 4
        assert "Auth failed" in result.stdout


class TestAgentList:
    """Tests for agent list command."""

    def test_list_success(self, mock_config, mock_client):
        """Test successful agent listing."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "agent-one",
                    "status": "active",
                    "capabilities": ["python"],
                },
                {
                    "agent_id": "agent-2",
                    "name": "agent-two",
                    "status": "inactive",
                    "capabilities": ["testing"],
                },
            ]
        }

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "agent-2" in result.stdout
        assert "Total: 2 agent(s)" in result.stdout
        mock_client.call.assert_called_once_with("agent.list", {"limit": 100})

    def test_list_with_status_filter(self, mock_config, mock_client):
        """Test agent listing with status filter."""
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, [
            "agent", "list",
            "--status", "active",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("agent.list", {
            "limit": 100,
            "status": "active",
        })

    def test_list_with_limit(self, mock_config, mock_client):
        """Test agent listing with custom limit."""
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, [
            "agent", "list",
            "--limit", "10",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("agent.list", {"limit": 10})

    def test_list_empty(self, mock_config, mock_client):
        """Test listing when no agents exist."""
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        assert "No agents found" in result.stdout

    def test_list_json_output(self, mock_config, mock_client):
        """Test agent listing with JSON output."""
        mock_client.call.return_value = {
            "agents": [{"agent_id": "agent-1", "name": "test"}]
        }

        result = runner.invoke(app, ["agent", "list", "--json"])

        assert result.exit_code == 0
        assert '"agent_id": "agent-1"' in result.stdout


class TestAgentInfo:
    """Tests for agent info command."""

    def test_info_success(self, mock_config, mock_client):
        """Test successful agent info retrieval."""
        mock_client.call.return_value = {
            "agent_id": "agent-12345",
            "name": "test-agent",
            "status": "active",
            "capabilities": ["python", "testing"],
            "cost_per_request": 0.01,
        }

        result = runner.invoke(app, ["agent", "info", "agent-12345"])

        assert result.exit_code == 0
        assert "Agent ID: agent-12345" in result.stdout
        assert "Name: test-agent" in result.stdout
        mock_client.call.assert_called_once_with("agent.info", {
            "agent_id": "agent-12345"
        })

    def test_info_json_output(self, mock_config, mock_client):
        """Test agent info with JSON output."""
        mock_client.call.return_value = {
            "agent_id": "agent-12345",
            "name": "test-agent",
        }

        result = runner.invoke(app, [
            "agent", "info", "agent-12345", "--json"
        ])

        assert result.exit_code == 0
        assert '"agent_id": "agent-12345"' in result.stdout

    def test_info_not_found(self, mock_config, mock_client):
        """Test info for non-existent agent."""
        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Agent not found"
        })

        result = runner.invoke(app, ["agent", "info", "agent-99999"])

        assert result.exit_code == 1
        assert "Agent not found" in result.stdout


class TestAgentRemove:
    """Tests for agent remove command."""

    def test_remove_with_force(self, mock_config, mock_client):
        """Test agent removal with force flag."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "agent", "remove", "agent-12345", "--force"
        ])

        assert result.exit_code == 0
        assert "Agent removed: agent-12345" in result.stdout
        mock_client.call.assert_called_once_with("agent.remove", {
            "agent_id": "agent-12345"
        })

    def test_remove_with_confirmation(self, mock_config, mock_client):
        """Test agent removal with confirmation prompt."""
        # Mock agent info call
        mock_client.call.side_effect = [
            {"agent_id": "agent-12345", "name": "test-agent"},
            {"success": True}
        ]

        result = runner.invoke(app, [
            "agent", "remove", "agent-12345"
        ], input="y\n")

        assert result.exit_code == 0
        assert "Agent removed: agent-12345" in result.stdout
        assert mock_client.call.call_count == 2

    def test_remove_cancelled(self, mock_config, mock_client):
        """Test agent removal cancelled by user."""
        mock_client.call.return_value = {
            "agent_id": "agent-12345",
            "name": "test-agent"
        }

        result = runner.invoke(app, [
            "agent", "remove", "agent-12345"
        ], input="n\n")

        assert result.exit_code == 0
        assert "Operation cancelled" in result.stdout

    def test_remove_json_output(self, mock_config, mock_client):
        """Test agent removal with JSON output."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "agent", "remove", "agent-12345", "--force", "--json"
        ])

        assert result.exit_code == 0
        assert '"success": true' in result.stdout


class TestAgentSearch:
    """Tests for agent search command."""

    def test_search_single_capability(self, mock_config, mock_client):
        """Test search with single capability."""
        mock_client.call.return_value = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "name": "python-agent",
                    "status": "active",
                    "capabilities": ["python"],
                }
            ]
        }

        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
        ])

        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "Total: 1 agent(s)" in result.stdout
        mock_client.call.assert_called_once_with("agent.search", {
            "capabilities": ["python"],
            "limit": 100,
        })

    def test_search_multiple_capabilities(self, mock_config, mock_client):
        """Test search with multiple capabilities."""
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
            "--capability", "testing",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("agent.search", {
            "capabilities": ["python", "testing"],
            "limit": 100,
        })

    def test_search_no_capability(self, mock_config, mock_client):
        """Test search without capability."""
        result = runner.invoke(app, ["agent", "search"])

        # Exit code 2 indicates validation error
        assert result.exit_code == 2

    def test_search_no_results(self, mock_config, mock_client):
        """Test search with no results."""
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "nonexistent",
        ])

        assert result.exit_code == 0
        assert "No agents found" in result.stdout

    def test_search_json_output(self, mock_config, mock_client):
        """Test search with JSON output."""
        mock_client.call.return_value = {
            "agents": [{"agent_id": "agent-1"}]
        }

        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
            "--json",
        ])

        assert result.exit_code == 0
        assert '"agent_id": "agent-1"' in result.stdout

    def test_search_with_limit(self, mock_config, mock_client):
        """Test search with custom limit."""
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
            "--limit", "10",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("agent.search", {
            "capabilities": ["python"],
            "limit": 10,
        })
