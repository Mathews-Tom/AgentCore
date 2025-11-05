"""Unit tests for Coordination JSON-RPC Service.

Tests for coordination signal management and agent selection JSON-RPC method handlers.

Methods tested:
- coordination.signal: Register sensitivity signal
- coordination.state: Get agent coordination state
- coordination.metrics: Get coordination metrics
- coordination.predict_overload: Predict agent overload
"""

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.coordination_service import coordination_service


class TestCoordinationSignal:
    """Test coordination.signal JSON-RPC method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_register_signal_success(self) -> None:
        """Test successful signal registration."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_signal,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.signal",
            params={
                "agent_id": "agent-001",
                "signal_type": "LOAD",
                "value": 0.75,
                "ttl_seconds": 120,
                "confidence": 0.95,
            },
            id="1",
        )

        result = await handle_coordination_signal(request)

        assert result["status"] == "registered"
        assert result["agent_id"] == "agent-001"
        assert "signal_id" in result
        assert "routing_score" in result
        assert "message" in result

    @pytest.mark.asyncio
    async def test_register_signal_missing_agent_id(self) -> None:
        """Test signal registration with missing agent_id."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_signal,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.signal",
            params={"signal_type": "LOAD", "value": 0.5},
            id="1",
        )

        with pytest.raises(ValueError, match="Missing required parameter: agent_id"):
            await handle_coordination_signal(request)

    @pytest.mark.asyncio
    async def test_register_signal_missing_signal_type(self) -> None:
        """Test signal registration with missing signal_type."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_signal,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.signal",
            params={"agent_id": "agent-001", "value": 0.5},
            id="1",
        )

        with pytest.raises(ValueError, match="Missing required parameter: signal_type"):
            await handle_coordination_signal(request)

    @pytest.mark.asyncio
    async def test_register_signal_invalid_signal_type(self) -> None:
        """Test signal registration with invalid signal_type."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_signal,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.signal",
            params={"agent_id": "agent-001", "signal_type": "INVALID", "value": 0.5},
            id="1",
        )

        with pytest.raises(ValueError, match="Invalid signal_type: INVALID"):
            await handle_coordination_signal(request)

    @pytest.mark.asyncio
    async def test_register_signal_invalid_value(self) -> None:
        """Test signal registration with out-of-range value."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_signal,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.signal",
            params={"agent_id": "agent-001", "signal_type": "LOAD", "value": 1.5},
            id="1",
        )

        with pytest.raises(ValueError, match="Signal validation failed"):
            await handle_coordination_signal(request)

    @pytest.mark.asyncio
    async def test_register_signal_with_defaults(self) -> None:
        """Test signal registration uses default values for optional params."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_signal,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.signal",
            params={"agent_id": "agent-002", "signal_type": "CAPACITY", "value": 0.8},
            id="1",
        )

        result = await handle_coordination_signal(request)

        assert result["status"] == "registered"
        # Verify signal was registered with defaults (ttl=60, confidence=1.0)
        state = coordination_service.get_coordination_state("agent-002")
        assert state is not None
        assert SignalType.CAPACITY in state.signals


class TestCoordinationState:
    """Test coordination.state JSON-RPC method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_get_state_existing_agent(self) -> None:
        """Test getting state for agent with signals."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_state,
        )

        # Register signal first
        signal = SensitivitySignal(
            agent_id="agent-state", signal_type=SignalType.LOAD, value=0.6, ttl_seconds=60
        )
        coordination_service.register_signal(signal)

        request = JsonRpcRequest(
            jsonrpc="2.0", method="coordination.state", params={"agent_id": "agent-state"}, id="1"
        )

        result = await handle_coordination_state(request)

        assert result["agent_id"] == "agent-state"
        assert "signals" in result
        assert "load_score" in result
        assert "routing_score" in result
        assert "last_updated" in result

    @pytest.mark.asyncio
    async def test_get_state_not_found(self) -> None:
        """Test getting state for non-existent agent."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_state,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.state",
            params={"agent_id": "unknown-agent"},
            id="1",
        )

        result = await handle_coordination_state(request)

        assert result["agent_id"] == "unknown-agent"
        assert result["status"] == "not_found"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_get_state_missing_agent_id(self) -> None:
        """Test getting state with missing agent_id."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_state,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0", method="coordination.state", params={}, id="1"
        )

        with pytest.raises(ValueError, match="Parameters required: agent_id"):
            await handle_coordination_state(request)


class TestCoordinationMetrics:
    """Test coordination.metrics JSON-RPC method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_get_metrics_empty(self) -> None:
        """Test getting metrics with no signals."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_metrics,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0", method="coordination.metrics", params=None, id="1"
        )

        result = await handle_coordination_metrics(request)

        assert result["total_signals"] == 0
        assert result["agents_tracked"] == 0
        assert "signals_by_type" in result
        assert "coordination_score_avg" in result

    @pytest.mark.asyncio
    async def test_get_metrics_with_signals(self) -> None:
        """Test getting metrics after registering signals."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_metrics,
        )

        # Register some signals
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-m1", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-m2", signal_type=SignalType.CAPACITY, value=0.8, ttl_seconds=60
            )
        )

        request = JsonRpcRequest(
            jsonrpc="2.0", method="coordination.metrics", params={}, id="1"
        )

        result = await handle_coordination_metrics(request)

        assert result["total_signals"] == 2
        assert result["agents_tracked"] == 2
        assert "LOAD" in result["signals_by_type"]
        assert "CAPACITY" in result["signals_by_type"]


class TestCoordinationPredictOverload:
    """Test coordination.predict_overload JSON-RPC method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_predict_overload_no_data(self) -> None:
        """Test overload prediction with no historical data."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_predict_overload,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.predict_overload",
            params={"agent_id": "agent-pred"},
            id="1",
        )

        result = await handle_coordination_predict_overload(request)

        assert result["agent_id"] == "agent-pred"
        assert result["will_overload"] is False
        assert result["probability"] == 0.0
        assert "message" in result

    @pytest.mark.asyncio
    async def test_predict_overload_with_custom_params(self) -> None:
        """Test overload prediction with custom forecast and threshold."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_predict_overload,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.predict_overload",
            params={"agent_id": "agent-pred", "forecast_seconds": 120, "threshold": 0.9},
            id="1",
        )

        result = await handle_coordination_predict_overload(request)

        assert result["forecast_seconds"] == 120
        assert result["threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_predict_overload_missing_agent_id(self) -> None:
        """Test prediction with missing agent_id."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_predict_overload,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0", method="coordination.predict_overload", params={}, id="1"
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_coordination_predict_overload(request)

    @pytest.mark.asyncio
    async def test_predict_overload_invalid_forecast_seconds(self) -> None:
        """Test prediction with invalid forecast_seconds."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_predict_overload,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.predict_overload",
            params={"agent_id": "agent-pred", "forecast_seconds": -10},
            id="1",
        )

        with pytest.raises(ValueError, match="forecast_seconds must be a positive number"):
            await handle_coordination_predict_overload(request)

    @pytest.mark.asyncio
    async def test_predict_overload_invalid_threshold(self) -> None:
        """Test prediction with invalid threshold."""
        from agentcore.a2a_protocol.services.coordination_jsonrpc import (
            handle_coordination_predict_overload,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="coordination.predict_overload",
            params={"agent_id": "agent-pred", "threshold": 1.5},
            id="1",
        )

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            await handle_coordination_predict_overload(request)
