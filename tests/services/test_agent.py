"""Unit tests for AgentService.

Tests cover:
- Business validation
- Parameter transformation
- JSON-RPC method calls
- Error handling
- Result validation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, MagicMock

import pytest

from agentcore_cli.services.agent import AgentService
from agentcore_cli.services.exceptions import (
    ValidationError,
    AgentNotFoundError,
    OperationError,
)


class TestAgentServiceRegister:
    """Test AgentService.register() method."""

    def test_register_success(self) -> None:
        """Test successful agent registration."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agent_id": "agent-001"}
        service = AgentService(mock_client)

        # Act
        agent_id = service.register("test-agent", ["python", "analysis"])

        # Assert
        assert agent_id == "agent-001"
        # Verify A2A-compliant agent_card structure is sent
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "agent.register"
        params = call_args[0][1]
        assert "agent_card" in params
        assert params["agent_card"]["agent_name"] == "test-agent"
        assert params["agent_card"]["agent_version"] == "1.0.0"
        assert params["agent_card"]["status"] == "active"
        assert len(params["agent_card"]["capabilities"]) == 2
        assert params["agent_card"]["capabilities"][0]["name"] == "python"
        assert params["agent_card"]["capabilities"][0]["cost_per_request"] == 0.01
        assert params["agent_card"]["capabilities"][1]["name"] == "analysis"
        assert params["override_existing"] == False

    def test_register_with_requirements(self) -> None:
        """Test registration with optional requirements."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agent_id": "agent-002"}
        service = AgentService(mock_client)

        # Act
        agent_id = service.register(
            "test-agent",
            ["python"],
            cost_per_request=0.05,
            requirements={"memory": "4GB"},
        )

        # Assert
        assert agent_id == "agent-002"
        # Verify A2A-compliant agent_card structure with requirements
        call_args = mock_client.call.call_args
        params = call_args[0][1]
        assert params["agent_card"]["agent_name"] == "test-agent"
        assert params["agent_card"]["capabilities"][0]["cost_per_request"] == 0.05
        assert params["agent_card"]["requirements"] == {"memory": "4GB"}

    def test_register_strips_whitespace(self) -> None:
        """Test that agent name is stripped of whitespace."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agent_id": "agent-003"}
        service = AgentService(mock_client)

        # Act
        service.register("  test-agent  ", ["python"])

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["agent_card"]["agent_name"] == "test-agent"

    def test_register_empty_name_raises_validation_error(self) -> None:
        """Test that empty name raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Agent name cannot be empty"):
            service.register("", ["python"])

    def test_register_whitespace_name_raises_validation_error(self) -> None:
        """Test that whitespace-only name raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Agent name cannot be empty"):
            service.register("   ", ["python"])

    def test_register_no_capabilities_raises_validation_error(self) -> None:
        """Test that empty capabilities raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="At least one capability required"):
            service.register("test-agent", [])

    def test_register_negative_cost_raises_validation_error(self) -> None:
        """Test that negative cost raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Cost per request cannot be negative"):
            service.register("test-agent", ["python"], cost_per_request=-0.01)

    def test_register_api_error_raises_operation_error(self) -> None:
        """Test that API errors are wrapped in OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API connection failed")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Agent registration failed"):
            service.register("test-agent", ["python"])

    def test_register_missing_agent_id_raises_operation_error(self) -> None:
        """Test that missing agent_id raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {}
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return agent_id"):
            service.register("test-agent", ["python"])


class TestAgentServiceListAgents:
    """Test AgentService.list() method."""

    def test_list_success(self) -> None:
        """Test successful agent listing."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "agents": [
                {"agent_id": "agent-001", "name": "test-1"},
                {"agent_id": "agent-002", "name": "test-2"},
            ]
        }
        service = AgentService(mock_client)

        # Act
        agents = service.list_agents()

        # Assert
        assert len(agents) == 2
        assert agents[0]["agent_id"] == "agent-001"
        mock_client.call.assert_called_once_with(
            "agent.list",
            {"limit": 100, "offset": 0},
        )

    def test_list_with_status_filter(self) -> None:
        """Test listing with status filter."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agents": []}
        service = AgentService(mock_client)

        # Act
        service.list_agents(status="active", limit=10, offset=5)

        # Assert
        mock_client.call.assert_called_once_with(
            "agent.list",
            {"limit": 10, "offset": 5, "status": "active"},
        )

    def test_list_invalid_limit_raises_validation_error(self) -> None:
        """Test that invalid limit raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Limit must be positive"):
            service.list_agents(limit=0)

    def test_list_negative_offset_raises_validation_error(self) -> None:
        """Test that negative offset raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Offset cannot be negative"):
            service.list_agents(offset=-1)

    def test_list_invalid_status_raises_validation_error(self) -> None:
        """Test that invalid status raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid status"):
            service.list_agents(status="invalid")

    def test_list_api_error_raises_operation_error(self) -> None:
        """Test that API errors are wrapped in OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Agent listing failed"):
            service.list_agents()

    def test_list_invalid_response_raises_operation_error(self) -> None:
        """Test that invalid response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agents": "not-a-list"}
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API returned invalid agents list"):
            service.list_agents()


class TestAgentServiceGet:
    """Test AgentService.get() method."""

    def test_get_success(self) -> None:
        """Test successful agent retrieval."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "agent": {"agent_id": "agent-001", "name": "test-agent"}
        }
        service = AgentService(mock_client)

        # Act
        agent = service.get("agent-001")

        # Assert
        assert agent["agent_id"] == "agent-001"
        mock_client.call.assert_called_once_with(
            "agent.get",
            {"agent_id": "agent-001"},
        )

    def test_get_strips_whitespace(self) -> None:
        """Test that agent_id is stripped of whitespace."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agent": {"agent_id": "agent-001"}}
        service = AgentService(mock_client)

        # Act
        service.get("  agent-001  ")

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["agent_id"] == "agent-001"

    def test_get_empty_id_raises_validation_error(self) -> None:
        """Test that empty agent_id raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Agent ID cannot be empty"):
            service.get("")

    def test_get_not_found_raises_agent_not_found_error(self) -> None:
        """Test that 'not found' error raises AgentNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Agent not found")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(AgentNotFoundError, match="Agent 'agent-001' not found"):
            service.get("agent-001")

    def test_get_api_error_raises_operation_error(self) -> None:
        """Test that other API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Agent retrieval failed"):
            service.get("agent-001")

    def test_get_missing_agent_raises_operation_error(self) -> None:
        """Test that missing agent in response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {}
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return agent information"):
            service.get("agent-001")


class TestAgentServiceRemove:
    """Test AgentService.remove() method."""

    def test_remove_success(self) -> None:
        """Test successful agent removal."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = AgentService(mock_client)

        # Act
        success = service.remove("agent-001")

        # Assert
        assert success is True
        mock_client.call.assert_called_once_with(
            "agent.unregister",
            {"agent_id": "agent-001", "force": False},
        )

    def test_remove_with_force(self) -> None:
        """Test removal with force flag."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = AgentService(mock_client)

        # Act
        service.remove("agent-001", force=True)

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["force"] is True

    def test_remove_empty_id_raises_validation_error(self) -> None:
        """Test that empty agent_id raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Agent ID cannot be empty"):
            service.remove("")

    def test_remove_not_found_raises_agent_not_found_error(self) -> None:
        """Test that 'not found' error raises AgentNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Agent not found")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(AgentNotFoundError, match="Agent 'agent-001' not found"):
            service.remove("agent-001")

    def test_remove_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Agent removal failed"):
            service.remove("agent-001")


class TestAgentServiceSearch:
    """Test AgentService.search() method."""

    def test_search_success(self) -> None:
        """Test successful agent search."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "agents": [{"agent_id": "agent-001", "capabilities": ["python"]}]
        }
        service = AgentService(mock_client)

        # Act
        agents = service.search("python")

        # Assert
        assert len(agents) == 1
        mock_client.call.assert_called_once_with(
            "agent.discover",
            {"capabilities": ["python"], "limit": 100},
        )

    def test_search_with_limit(self) -> None:
        """Test search with custom limit."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agents": []}
        service = AgentService(mock_client)

        # Act
        service.search("python", limit=10)

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["limit"] == 10

    def test_search_strips_whitespace(self) -> None:
        """Test that capability is stripped of whitespace."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agents": []}
        service = AgentService(mock_client)

        # Act
        service.search("  python  ")

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["capabilities"] == ["python"]

    def test_search_empty_capability_raises_validation_error(self) -> None:
        """Test that empty capability raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Capability cannot be empty"):
            service.search("")

    def test_search_invalid_limit_raises_validation_error(self) -> None:
        """Test that invalid limit raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Limit must be positive"):
            service.search("python", limit=0)

    def test_search_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Agent search failed"):
            service.search("python")

    def test_search_invalid_response_raises_operation_error(self) -> None:
        """Test that invalid response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"agents": "not-a-list"}
        service = AgentService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API returned invalid agents list"):
            service.search("python")
