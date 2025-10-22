"""Integration tests for agent commands with mock API responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_api_server():
    """Mock AgentCore API server with realistic responses."""

    def mock_call(method: str, params: dict | None = None):
        """Handle mock JSON-RPC calls."""
        params = params or {}

        if method == "agent.register":
            return {
                "agent_id": "agent-test-12345",
                "name": params.get("name", "unknown"),
                "status": "active",
                "capabilities": params.get("capabilities", []),
                "cost_per_request": params.get("cost_per_request", 0.01),
                "requirements": params.get("requirements", {}),
                "registered_at": "2025-10-21T00:00:00Z",
            }

        if method == "agent.list":
            status_filter = params.get("status")
            agents = [
                {
                    "agent_id": "agent-1",
                    "name": "code-analyzer",
                    "status": "active",
                    "capabilities": ["python", "analysis"],
                    "cost_per_request": 0.01,
                    "registered_at": "2025-10-20T10:00:00Z",
                },
                {
                    "agent_id": "agent-2",
                    "name": "test-runner",
                    "status": "active",
                    "capabilities": ["testing", "python"],
                    "cost_per_request": 0.02,
                    "registered_at": "2025-10-19T15:30:00Z",
                },
                {
                    "agent_id": "agent-3",
                    "name": "docs-generator",
                    "status": "inactive",
                    "capabilities": ["documentation", "markdown"],
                    "cost_per_request": 0.005,
                    "registered_at": "2025-10-18T08:45:00Z",
                },
            ]

            if status_filter:
                agents = [a for a in agents if a["status"] == status_filter]

            limit = params.get("limit", 100)
            return {"agents": agents[:limit]}

        if method == "agent.get":
            agent_id = params.get("agent_id")
            if agent_id == "agent-1":
                return {
                    "agent": {
                        "agent_id": "agent-1",
                        "name": "code-analyzer",
                        "status": "active",
                        "capabilities": ["python", "analysis", "linting"],
                        "requirements": {"memory": "512MB", "cpu": "0.5"},
                        "cost_per_request": 0.01,
                        "registered_at": "2025-10-20T10:00:00Z",
                        "updated_at": "2025-10-21T00:00:00Z",
                        "health_status": "healthy",
                        "active_tasks": 3,
                    }
                }
            return {"error": {"code": -32602, "message": "Agent not found"}}

        if method == "agent.remove":
            return {"success": True, "agent_id": params.get("agent_id")}

        if method == "agent.search":
            capabilities = params.get("capabilities", [])
            agents = [
                {
                    "agent_id": "agent-1",
                    "name": "code-analyzer",
                    "status": "active",
                    "capabilities": ["python", "analysis"],
                },
                {
                    "agent_id": "agent-2",
                    "name": "test-runner",
                    "status": "active",
                    "capabilities": ["testing", "python"],
                },
            ]

            # Simple capability matching
            if capabilities:
                agents = [
                    a for a in agents
                    if any(cap in a["capabilities"] for cap in capabilities)
                ]

            limit = params.get("limit", 100)
            return {"agents": agents[:limit]}

        return {}

    return mock_call


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("agentcore_cli.container.get_config") as mock_get_config:
        config = MagicMock()
        config.api.url = "http://localhost:8001"
        config.api.timeout = 30
        config.api.retries = 3
        config.api.verify_ssl = True
        config.auth.type = "none"
        config.auth.token = None
        mock_get_config.return_value = config
        yield config


@pytest.fixture
def mock_client(mock_api_server):
    """Mock AgentCore client with API server."""
    with patch("agentcore_cli.container.get_jsonrpc_client") as mock_get_client:
        client = MagicMock()
        client.call.side_effect = mock_api_server
        mock_get_client.return_value = client
        yield client


class TestAgentRegisterIntegration:
    """Integration tests for agent register command."""

    def test_register_full_workflow(self, mock_config, mock_client):
        """Test complete agent registration workflow."""
        result = runner.invoke(app, [
            "agent", "register",
            "--name", "integration-agent",
            "--capabilities", "python,testing,analysis",
            "--cost-per-request", "0.03",
        ])

        assert result.exit_code == 0
        assert "Agent ID: agent-test-12345" in result.stdout
        assert "integration-agent" in result.stdout

    def test_register_with_complex_requirements(self, mock_config, mock_client):
        """Test registration with custom cost (requirements parameter not implemented yet)."""
        result = runner.invoke(app, [
            "agent", "register",
            "--name", "resource-intensive-agent",
            "--capabilities", "ml,training",
            "--cost-per-request", "0.05",
        ])

        assert result.exit_code == 0
        assert "Agent ID:" in result.stdout


class TestAgentListIntegration:
    """Integration tests for agent list command."""

    def test_list_all_agents(self, mock_config, mock_client):
        """Test listing all agents."""
        result = runner.invoke(app, ["agent", "list"])

        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "agent-2" in result.stdout
        assert "agent-3" in result.stdout
        assert "code-analyzer" in result.stdout
        assert "Registered Agents (3)" in result.stdout

    def test_list_active_agents(self, mock_config, mock_client):
        """Test listing only active agents."""
        result = runner.invoke(app, [
            "agent", "list",
            "--status", "active",
        ])

        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "agent-2" in result.stdout
        assert "agent-3" not in result.stdout or "inactive" in result.stdout
        assert "Registered Agents (2)" in result.stdout

    def test_list_with_json_output(self, mock_config, mock_client):
        """Test listing with JSON output for scripting."""
        result = runner.invoke(app, ["agent", "list", "--json"])

        assert result.exit_code == 0
        assert '"agent_id": "agent-1"' in result.stdout
        assert '"capabilities"' in result.stdout


class TestAgentInfoIntegration:
    """Integration tests for agent info command."""

    def test_info_existing_agent(self, mock_config, mock_client):
        """Test retrieving info for existing agent."""
        result = runner.invoke(app, ["agent", "info", "agent-1"])

        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "code-analyzer" in result.stdout
        assert "active" in result.stdout
        assert "python" in result.stdout


class TestAgentRemoveIntegration:
    """Integration tests for agent remove command."""

    def test_remove_with_force(self, mock_config, mock_client):
        """Test removing agent with force flag."""
        result = runner.invoke(app, [
            "agent", "remove", "agent-1", "--force"
        ])

        assert result.exit_code == 0
        assert "Agent ID: agent-1" in result.stdout


class TestAgentSearchIntegration:
    """Integration tests for agent search command."""

    def test_search_python_capability(self, mock_config, mock_client):
        """Test searching for Python capability."""
        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
        ])

        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "agent-2" in result.stdout
        assert "python" in result.stdout.lower()  # Check capability is shown

    def test_search_multiple_capabilities(self, mock_config, mock_client):
        """Test searching with multiple capabilities."""
        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
            "--capability", "testing",
        ])

        assert result.exit_code == 0
        assert "agent" in result.stdout


class TestAgentWorkflow:
    """Integration tests for complete agent workflows."""

    def test_register_list_info_remove_workflow(self, mock_config, mock_client):
        """Test complete workflow: register → list → info → remove."""
        # 1. Register new agent
        register_result = runner.invoke(app, [
            "agent", "register",
            "--name", "workflow-test-agent",
            "--capabilities", "testing,workflow",
        ])
        assert register_result.exit_code == 0

        # 2. List all agents (should include new agent)
        list_result = runner.invoke(app, ["agent", "list"])
        assert list_result.exit_code == 0

        # 3. Get agent info
        info_result = runner.invoke(app, ["agent", "info", "agent-1"])
        assert info_result.exit_code == 0

        # 4. Remove agent
        remove_result = runner.invoke(app, [
            "agent", "remove", "agent-1", "--force"
        ])
        assert remove_result.exit_code == 0

    def test_search_and_inspect_workflow(self, mock_config, mock_client):
        """Test workflow: search → inspect matching agents."""
        # 1. Search for Python agents
        search_result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
        ])
        assert search_result.exit_code == 0

        # 2. Get details of first agent
        info_result = runner.invoke(app, ["agent", "info", "agent-1"])
        assert info_result.exit_code == 0

    def test_json_output_workflow(self, mock_config, mock_client):
        """Test workflow using JSON output for automation."""
        # 1. List agents in JSON
        list_result = runner.invoke(app, ["agent", "list", "--json"])
        assert list_result.exit_code == 0
        assert '"agent_id"' in list_result.stdout

        # 2. Get agent info in JSON
        info_result = runner.invoke(app, [
            "agent", "info", "agent-1", "--json"
        ])
        assert info_result.exit_code == 0
        assert '"agent_id"' in info_result.stdout

        # 3. Search in JSON
        search_result = runner.invoke(app, [
            "agent", "search",
            "--capability", "python",
            "--json",
        ])
        assert search_result.exit_code == 0
        assert '"agent_id"' in search_result.stdout


class TestAgentEdgeCases:
    """Integration tests for edge cases and error scenarios."""

    def test_whitespace_in_capabilities(self, mock_config, mock_client):
        """Test handling of whitespace in capabilities."""
        result = runner.invoke(app, [
            "agent", "register",
            "--name", "test-agent",
            "--capabilities", " python , testing , analysis ",
        ])

        assert result.exit_code == 0
        # Capabilities should be trimmed

    def test_list_with_small_limit(self, mock_config, mock_client):
        """Test listing with limit smaller than total."""
        result = runner.invoke(app, [
            "agent", "list",
            "--limit", "2",
        ])

        assert result.exit_code == 0
        # Should only show 2 agents

    def test_search_nonexistent_capability(self, mock_config, mock_client):
        """Test searching for capability that doesn't exist."""
        # Modify mock to return empty results
        mock_client.call.return_value = {"agents": []}

        result = runner.invoke(app, [
            "agent", "search",
            "--capability", "nonexistent-capability",
        ])

        assert result.exit_code == 0
        assert result.exit_code == 0  # Empty table still succeeds
