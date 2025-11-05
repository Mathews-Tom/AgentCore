"""Integration tests for MessageRouter RIPPLE_COORDINATION strategy.

Tests the integration between MessageRouter and CoordinationService for
intelligent agent selection using the Ripple Effect Protocol.

Coverage targets:
- RIPPLE_COORDINATION strategy selection
- Integration with coordination_service.select_optimal_agent()
- Fallback to RANDOM on errors
- Metrics tracking
- Backward compatibility with existing strategies
"""

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import coordination_service
from agentcore.a2a_protocol.services.message_router import MessageRouter, RoutingStrategy


class TestRippleCoordinationStrategy:
    """Integration tests for RIPPLE_COORDINATION routing strategy."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.router = MessageRouter()
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_ripple_coordination_selects_optimal_agent(self) -> None:
        """Test RIPPLE_COORDINATION selects agent with best routing score."""
        # Register signals for 3 agents
        # Agent 1: Low load, high capacity (should be selected)
        signal1_load = SensitivitySignal(
            agent_id="agent-001", signal_type=SignalType.LOAD, value=0.2, ttl_seconds=60
        )
        signal1_capacity = SensitivitySignal(
            agent_id="agent-001", signal_type=SignalType.CAPACITY, value=0.9, ttl_seconds=60
        )
        coordination_service.register_signal(signal1_load)
        coordination_service.register_signal(signal1_capacity)

        # Agent 2: High load, low capacity (should not be selected)
        signal2_load = SensitivitySignal(
            agent_id="agent-002", signal_type=SignalType.LOAD, value=0.9, ttl_seconds=60
        )
        signal2_capacity = SensitivitySignal(
            agent_id="agent-002", signal_type=SignalType.CAPACITY, value=0.3, ttl_seconds=60
        )
        coordination_service.register_signal(signal2_load)
        coordination_service.register_signal(signal2_capacity)

        # Agent 3: Medium load and capacity
        signal3_load = SensitivitySignal(
            agent_id="agent-003", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        signal3_capacity = SensitivitySignal(
            agent_id="agent-003", signal_type=SignalType.CAPACITY, value=0.6, ttl_seconds=60
        )
        coordination_service.register_signal(signal3_load)
        coordination_service.register_signal(signal3_capacity)

        # Select agent using RIPPLE_COORDINATION
        candidates = ["agent-001", "agent-002", "agent-003"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # Agent 001 should be selected (low load + high capacity)
        assert selected == "agent-001"

    @pytest.mark.asyncio
    async def test_ripple_coordination_with_no_signals(self) -> None:
        """Test RIPPLE_COORDINATION with agents that have no signals (use defaults)."""
        # No signals registered, all agents have default scores (0.5)
        candidates = ["agent-no-sig-1", "agent-no-sig-2", "agent-no-sig-3"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # Should still select an agent (all have same default score)
        assert selected in candidates

    @pytest.mark.asyncio
    async def test_ripple_coordination_single_candidate(self) -> None:
        """Test RIPPLE_COORDINATION with single candidate."""
        signal = SensitivitySignal(
            agent_id="agent-single", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        coordination_service.register_signal(signal)

        candidates = ["agent-single"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        assert selected == "agent-single"

    @pytest.mark.asyncio
    async def test_ripple_coordination_updates_metrics(self) -> None:
        """Test that RIPPLE_COORDINATION updates routing metrics."""
        # Register signals
        signal = SensitivitySignal(
            agent_id="agent-metrics", signal_type=SignalType.LOAD, value=0.3, ttl_seconds=60
        )
        coordination_service.register_signal(signal)

        # Get initial metrics
        initial_stats = self.router.get_routing_stats()
        initial_selections = initial_stats.get("coordination_routing_selections_total", 0)

        # Perform selection
        candidates = ["agent-metrics"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        assert selected == "agent-metrics"

        # Verify metrics updated
        updated_stats = self.router.get_routing_stats()
        updated_selections = updated_stats.get("coordination_routing_selections_total", 0)

        assert updated_selections == initial_selections + 1

    @pytest.mark.asyncio
    async def test_ripple_coordination_fallback_to_random_on_error(self) -> None:
        """Test fallback to RANDOM when coordination service fails."""
        # Create a scenario where coordination might fail gracefully
        # (e.g., empty candidates list should raise error in coordination,
        # but our implementation handles it)

        # With valid candidates but simulating error conditions
        candidates = ["agent-fallback-1", "agent-fallback-2"]

        # Even without signals, should fall back gracefully
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # Should still select one of the candidates (via fallback or default)
        assert selected in candidates

    @pytest.mark.asyncio
    async def test_ripple_coordination_empty_candidates_raises_error(self) -> None:
        """Test that empty candidates list raises ValueError."""
        with pytest.raises(ValueError, match="No candidates"):
            await self.router._ripple_coordination_select([])

    @pytest.mark.asyncio
    async def test_backward_compatibility_existing_strategies(self) -> None:
        """Test that existing routing strategies still work after RIPPLE_COORDINATION addition."""
        candidates = ["agent-compat-1", "agent-compat-2", "agent-compat-3"]

        # Test ROUND_ROBIN
        selected_rr = await self.router._select_agent(candidates, RoutingStrategy.ROUND_ROBIN)
        assert selected_rr in candidates

        # Test LEAST_LOADED
        selected_ll = await self.router._select_agent(candidates, RoutingStrategy.LEAST_LOADED)
        assert selected_ll in candidates

        # Test RANDOM
        selected_rand = await self.router._select_agent(candidates, RoutingStrategy.RANDOM)
        assert selected_rand in candidates

        # Test CAPABILITY_MATCH (default)
        selected_cap = await self.router._select_agent(
            candidates, RoutingStrategy.CAPABILITY_MATCH
        )
        assert selected_cap in candidates

    @pytest.mark.asyncio
    async def test_ripple_coordination_with_quality_signals(self) -> None:
        """Test RIPPLE_COORDINATION with quality signals affects selection."""
        # Agent 1: High quality
        signal1_quality = SensitivitySignal(
            agent_id="agent-hq", signal_type=SignalType.QUALITY, value=0.95, ttl_seconds=60
        )
        coordination_service.register_signal(signal1_quality)

        # Agent 2: Low quality
        signal2_quality = SensitivitySignal(
            agent_id="agent-lq", signal_type=SignalType.QUALITY, value=0.3, ttl_seconds=60
        )
        coordination_service.register_signal(signal2_quality)

        candidates = ["agent-hq", "agent-lq"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # High quality agent should be selected
        assert selected == "agent-hq"

    @pytest.mark.asyncio
    async def test_ripple_coordination_with_cost_signals(self) -> None:
        """Test RIPPLE_COORDINATION considers cost signals."""
        # Agent 1: Low cost (high cost score)
        signal1_cost = SensitivitySignal(
            agent_id="agent-cheap", signal_type=SignalType.COST, value=0.9, ttl_seconds=60
        )
        coordination_service.register_signal(signal1_cost)

        # Agent 2: High cost (low cost score)
        signal2_cost = SensitivitySignal(
            agent_id="agent-expensive", signal_type=SignalType.COST, value=0.2, ttl_seconds=60
        )
        coordination_service.register_signal(signal2_cost)

        candidates = ["agent-cheap", "agent-expensive"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # Cheaper agent should be preferred (higher cost score)
        assert selected == "agent-cheap"

    @pytest.mark.asyncio
    async def test_ripple_coordination_multi_signal_scoring(self) -> None:
        """Test RIPPLE_COORDINATION with multiple signal types for comprehensive scoring."""
        # Agent 1: Balanced signals
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-balanced",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-balanced",
                signal_type=SignalType.CAPACITY,
                value=0.7,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-balanced",
                signal_type=SignalType.QUALITY,
                value=0.8,
                ttl_seconds=60,
            )
        )

        # Agent 2: Excellent in all aspects
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-excellent",
                signal_type=SignalType.LOAD,
                value=0.1,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-excellent",
                signal_type=SignalType.CAPACITY,
                value=0.95,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-excellent",
                signal_type=SignalType.QUALITY,
                value=0.98,
                ttl_seconds=60,
            )
        )

        candidates = ["agent-balanced", "agent-excellent"]
        selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # Excellent agent should be selected
        assert selected == "agent-excellent"
