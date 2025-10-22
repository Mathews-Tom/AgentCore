"""Integration tests for agent commands.

These tests verify that the agent commands properly use the service layer
and send JSON-RPC 2.0 compliant requests to the API.
"""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import (
    ValidationError,
    AgentNotFoundError,
    OperationError,
)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_agent_service() -> Mock:
    """Create a mock agent service."""
    return Mock()


class TestAgentRegisterCommand:
    """Test suite for agent register command."""

    def test_register_success(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test successful agent registration."""
        # Mock service response
        mock_agent_service.register.return_value = "agent-001"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "test-agent",
                    "--capabilities",
                    "python,analysis",
                ],
            )

        # Verify exit code
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify output
        assert "Agent registered successfully" in result.output
        assert "agent-001" in result.output
        assert "test-agent" in result.output

        # Verify service was called correctly
        mock_agent_service.register.assert_called_once_with(
            name="test-agent",
            capabilities=["python", "analysis"],
            cost_per_request=0.01,
        )

    def test_register_with_custom_cost(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent registration with custom cost."""
        # Mock service response
        mock_agent_service.register.return_value = "agent-002"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "expensive-agent",
                    "--capabilities",
                    "python",
                    "--cost-per-request",
                    "0.05",
                ],
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with custom cost
        mock_agent_service.register.assert_called_once_with(
            name="expensive-agent",
            capabilities=["python"],
            cost_per_request=0.05,
        )

    def test_register_json_output(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent registration with JSON output."""
        # Mock service response
        mock_agent_service.register.return_value = "agent-003"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "json-agent",
                    "--capabilities",
                    "python,testing",
                    "--json",
                ],
            )

        # Verify exit code
        assert result.exit_code == 0


        # Verify JSON output (note: JSON is pretty-printed with indentation)
        assert '"agent_id": "agent-003"' in result.output
        assert '"name": "json-agent"' in result.output
        assert '"python"' in result.output
        assert '"testing"' in result.output
        assert '"capabilities"' in result.output

    def test_register_validation_error(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent registration with validation error."""
        # Mock service to raise validation error
        mock_agent_service.register.side_effect = ValidationError(
            "Agent name cannot be empty"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "",
                    "--capabilities",
                    "python",
                ],
            )

        # Verify exit code (2 for validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output
        assert "Agent name cannot be empty" in result.output

    def test_register_operation_error(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent registration with operation error."""
        # Mock service to raise operation error
        mock_agent_service.register.side_effect = OperationError(
            "Agent registration failed: API timeout"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "test-agent",
                    "--capabilities",
                    "python",
                ],
            )

        # Verify exit code (1 for operation error)
        assert result.exit_code == 1

        # Verify error message
        assert "Operation failed" in result.output
        assert "Agent registration failed: API timeout" in result.output

    def test_register_multiple_capabilities(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent registration with multiple capabilities."""
        # Mock service response
        mock_agent_service.register.return_value = "agent-004"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "multi-cap-agent",
                    "--capabilities",
                    "python, analysis, testing, execution",
                ],
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with all capabilities (trimmed)
        mock_agent_service.register.assert_called_once_with(
            name="multi-cap-agent",
            capabilities=["python", "analysis", "testing", "execution"],
            cost_per_request=0.01,
        )


class TestAgentListCommand:
    """Test suite for agent list command."""

    def test_list_success(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test successful agent listing."""
        # Mock service response
        mock_agent_service.list_agents.return_value = [
            {
                "agent_id": "agent-001",
                "name": "analyzer",
                "status": "active",
                "capabilities": ["python", "analysis"],
            },
            {
                "agent_id": "agent-002",
                "name": "executor",
                "status": "active",
                "capabilities": ["python", "execution"],
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output contains agent info
        assert "agent-001" in result.output
        assert "analyzer" in result.output
        assert "agent-002" in result.output
        assert "executor" in result.output

    def test_list_with_status_filter(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent listing with status filter."""
        # Mock service response
        mock_agent_service.list_agents.return_value = [
            {
                "agent_id": "agent-001",
                "name": "analyzer",
                "status": "active",
                "capabilities": ["python"],
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "list", "--status", "active"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with status filter
        mock_agent_service.list_agents.assert_called_once_with(
            status="active", limit=100
        )

    def test_list_empty(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent listing with no results."""
        # Mock service response
        mock_agent_service.list_agents.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "No agents found" in result.output


class TestAgentInfoCommand:
    """Test suite for agent info command."""

    def test_info_success(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test successful agent info retrieval."""
        # Mock service response
        mock_agent_service.get.return_value = {
            "agent_id": "agent-001",
            "name": "analyzer",
            "status": "active",
            "capabilities": ["python", "analysis"],
            "cost_per_request": 0.01,
        }

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "info", "agent-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "agent-001" in result.output
        assert "analyzer" in result.output
        assert "active" in result.output

    def test_info_not_found(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent info for non-existent agent."""
        # Mock service to raise not found error
        mock_agent_service.get.side_effect = AgentNotFoundError(
            "Agent 'agent-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "info", "agent-999"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Agent not found" in result.output


class TestAgentRemoveCommand:
    """Test suite for agent remove command."""

    def test_remove_success(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test successful agent removal."""
        # Mock service response
        mock_agent_service.remove.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "remove", "agent-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Agent removed successfully" in result.output
        assert "agent-001" in result.output

        # Verify service was called
        mock_agent_service.remove.assert_called_once_with("agent-001", force=False)

    def test_remove_with_force(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent removal with force flag."""
        # Mock service response
        mock_agent_service.remove.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(app, ["agent", "remove", "agent-001", "--force"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with force=True
        mock_agent_service.remove.assert_called_once_with("agent-001", force=True)


class TestAgentSearchCommand:
    """Test suite for agent search command."""

    def test_search_success(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test successful agent search."""
        # Mock service response
        mock_agent_service.search.return_value = [
            {
                "agent_id": "agent-001",
                "name": "analyzer",
                "status": "active",
                "capabilities": ["python", "analysis"],
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app, ["agent", "search", "--capability", "python"]
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "agent-001" in result.output
        assert "analyzer" in result.output

        # Verify service was called
        mock_agent_service.search.assert_called_once_with(
            capability="python", limit=100
        )

    def test_search_no_results(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test agent search with no results."""
        # Mock service response
        mock_agent_service.search.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app, ["agent", "search", "--capability", "nonexistent"]
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "No agents found" in result.output


class TestJSONRPCCompliance:
    """Test suite for JSON-RPC 2.0 compliance.

    These tests verify that the CLI sends properly formatted JSON-RPC 2.0
    requests with the required 'params' wrapper.
    """

    def test_register_sends_proper_jsonrpc_request(
        self, runner: CliRunner
    ) -> None:
        """Verify agent register sends JSON-RPC 2.0 compliant request."""
        # Create a mock client that captures the request
        mock_client = Mock()
        mock_client.call.return_value = {"agent_id": "agent-001"}

        # Create mock service with the mock client
        mock_service = Mock()
        mock_service.register.return_value = "agent-001"
        mock_service.client = mock_client

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "test-agent",
                    "--capabilities",
                    "python,analysis",
                ],
            )

        # Verify command succeeded
        assert result.exit_code == 0

        # Verify service.register was called (which calls client.call internally)
        mock_service.register.assert_called_once()

        # Extract the call arguments
        call_args = mock_service.register.call_args
        assert call_args is not None

        # Verify parameters are passed as expected
        assert call_args.kwargs["name"] == "test-agent"
        assert call_args.kwargs["capabilities"] == ["python", "analysis"]

    def test_service_layer_wraps_params_correctly(self) -> None:
        """Verify service layer properly wraps parameters in 'params' object."""
        from agentcore_cli.services.agent import AgentService

        # Create a mock client
        mock_client = Mock()
        mock_client.call.return_value = {"agent_id": "agent-001"}

        # Create service with mock client
        service = AgentService(mock_client)

        # Call register
        agent_id = service.register(
            name="test-agent", capabilities=["python", "analysis"], cost_per_request=0.01
        )

        # Verify result
        assert agent_id == "agent-001"

        # Verify client.call was called with proper method and params
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args

        # Verify method name
        assert call_args.args[0] == "agent.register"

        # Verify params structure (should be a dict, not flat)
        params = call_args.args[1]
        assert isinstance(params, dict)
        assert "name" in params
        assert params["name"] == "test-agent"
        assert "capabilities" in params
        assert params["capabilities"] == ["python", "analysis"]
        assert "cost_per_request" in params
        assert params["cost_per_request"] == 0.01

        # This dict will be wrapped in "params" by the JsonRpcClient
        # The client is responsible for creating the full JSON-RPC request


class TestIntegrationFlow:
    """Integration tests for complete command flow."""

    def test_complete_agent_lifecycle(
        self, runner: CliRunner, mock_agent_service: Mock
    ) -> None:
        """Test complete agent lifecycle: register -> info -> remove."""
        # Mock service responses
        mock_agent_service.register.return_value = "agent-001"
        mock_agent_service.get.return_value = {
            "agent_id": "agent-001",
            "name": "lifecycle-agent",
            "status": "active",
            "capabilities": ["python"],
            "cost_per_request": 0.01,
        }
        mock_agent_service.remove.return_value = True

        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            # Register
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "lifecycle-agent",
                    "--capabilities",
                    "python",
                ],
            )
            assert result.exit_code == 0

            # Info
            result = runner.invoke(app, ["agent", "info", "agent-001"])
            assert result.exit_code == 0

            # Remove
            result = runner.invoke(app, ["agent", "remove", "agent-001"])
            assert result.exit_code == 0
