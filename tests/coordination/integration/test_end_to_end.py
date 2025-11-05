"""End-to-end coordination integration tests.

Comprehensive integration tests demonstrating complete coordination workflows
including MessageRouter integration, routing strategy comparison, and
overload prediction accuracy.

Coverage:
- RIPPLE_COORDINATION vs RANDOM routing comparison
- Complete signal lifecycle (registration → routing → cleanup)
- Overload prediction accuracy validation
- MessageRouter integration
- Real-world coordination scenarios
"""

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import coordination_service
from agentcore.a2a_protocol.services.message_router import MessageRouter, RoutingStrategy


class TestRoutingStrategyComparison:
    """Compare RIPPLE_COORDINATION vs RANDOM routing strategies."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.router = MessageRouter()
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_ripple_vs_random_routing(self) -> None:
        """Test that RIPPLE_COORDINATION outperforms RANDOM in selecting optimal agents."""
        # Create agents with varied loads
        # Agent 1: Optimal (low load, high capacity)
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-optimal",
                signal_type=SignalType.LOAD,
                value=0.1,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-optimal",
                signal_type=SignalType.CAPACITY,
                value=0.95,
                ttl_seconds=60,
            )
        )

        # Agent 2: Poor (high load, low capacity)
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-poor",
                signal_type=SignalType.LOAD,
                value=0.9,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-poor",
                signal_type=SignalType.CAPACITY,
                value=0.2,
                ttl_seconds=60,
            )
        )

        # Agent 3: Medium
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-medium",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=60,
            )
        )
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-medium",
                signal_type=SignalType.CAPACITY,
                value=0.5,
                ttl_seconds=60,
            )
        )

        candidates = ["agent-optimal", "agent-poor", "agent-medium"]

        # Test RIPPLE_COORDINATION - should always select optimal
        ripple_selections = []
        for _ in range(10):
            selected = await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)
            ripple_selections.append(selected)

        # RIPPLE should consistently select the optimal agent
        assert all(s == "agent-optimal" for s in ripple_selections)

        # Test RANDOM - should have varied selections
        random_selections = []
        for _ in range(20):
            selected = await self.router._select_agent(candidates, RoutingStrategy.RANDOM)
            random_selections.append(selected)

        # RANDOM should select different agents
        assert len(set(random_selections)) > 1

    @pytest.mark.asyncio
    async def test_routing_strategy_metrics(self) -> None:
        """Test that routing strategies update appropriate metrics."""
        # Register signals
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-metrics",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=60,
            )
        )

        candidates = ["agent-metrics"]

        # Get initial metrics
        initial_stats = self.router.get_routing_stats()
        initial_coord = initial_stats.get("coordination_routing_selections_total", 0)

        # Perform RIPPLE_COORDINATION selection
        await self.router._select_agent(candidates, RoutingStrategy.RIPPLE_COORDINATION)

        # Verify metrics updated
        updated_stats = self.router.get_routing_stats()
        updated_coord = updated_stats.get("coordination_routing_selections_total", 0)

        assert updated_coord == initial_coord + 1


class TestCompleteWorkflow:
    """Test complete coordination workflow from signal to routing to cleanup."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.router = MessageRouter()
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    def test_signal_to_routing_to_cleanup_lifecycle(self) -> None:
        """Test complete lifecycle: register → route → cleanup."""
        agent_id = "agent-lifecycle"

        # Step 1: Register signal
        signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.3,
            ttl_seconds=300,
        )
        coordination_service.register_signal(signal)

        # Verify signal registered
        state = coordination_service.get_coordination_state(agent_id)
        assert state is not None
        assert SignalType.LOAD in state.signals

        # Step 2: Route using coordination
        candidates = [agent_id]
        selected = coordination_service.select_optimal_agent(candidates)
        assert selected == agent_id

        # Verify selection metrics updated
        assert coordination_service.metrics.total_selections > 0

        # Step 3: Cleanup (no signals should be expired yet)
        stats = coordination_service.remove_expired_signals()
        assert stats["signals_removed"] == 0
        assert stats["agents_removed"] == 0

        # Verify agent still exists
        state_after = coordination_service.get_coordination_state(agent_id)
        assert state_after is not None

    def test_multi_agent_workflow(self) -> None:
        """Test workflow with multiple agents in realistic scenario."""
        # Scenario: 5 agents handling requests, signals updated regularly

        # Initial registration
        agents = [f"agent-workflow-{i}" for i in range(5)]
        for i, agent_id in enumerate(agents):
            # Vary initial loads
            load = 0.2 * i  # 0.0, 0.2, 0.4, 0.6, 0.8
            coordination_service.register_signal(
                SensitivitySignal(
                    agent_id=agent_id,
                    signal_type=SignalType.LOAD,
                    value=load,
                    ttl_seconds=60,
                )
            )

        # Simulate 10 routing decisions
        for _ in range(10):
            selected = coordination_service.select_optimal_agent(agents)
            assert selected in agents

            # Selected agent should have low load
            state = coordination_service.get_coordination_state(selected)
            assert state is not None
            # Agent 0 has lowest load (0.0), should be selected most often
            assert selected in ["agent-workflow-0", "agent-workflow-1"]

        # Verify metrics
        assert coordination_service.metrics.total_selections == 10
        assert coordination_service.metrics.agents_tracked == 5


class TestOverloadPredictionAccuracy:
    """Test overload prediction accuracy in realistic scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    def test_overload_prediction_with_trending_load(self) -> None:
        """Test overload prediction with realistic trending load."""
        agent_id = "agent-trending"

        # Simulate increasing load trend over time
        loads = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7]
        for i, load in enumerate(loads):
            signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.LOAD,
                value=load,
                ttl_seconds=60,
            )
            coordination_service.register_signal(signal)

        # Predict overload with threshold 0.8
        will_overload, probability = coordination_service.predict_overload(
            agent_id, forecast_seconds=60, threshold=0.8
        )

        # With increasing trend, should predict overload
        assert will_overload is True
        assert probability > 0.7

    def test_no_overload_with_stable_load(self) -> None:
        """Test that stable load doesn't predict overload."""
        agent_id = "agent-stable"

        # Register stable load signals
        for _ in range(5):
            signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.LOAD,
                value=0.5,  # Constant
                ttl_seconds=60,
            )
            coordination_service.register_signal(signal)

        # Predict overload
        will_overload, probability = coordination_service.predict_overload(
            agent_id, forecast_seconds=60, threshold=0.8
        )

        # Stable load below threshold should not predict overload
        assert will_overload is False
        assert probability == 0.5  # Current load

    def test_overload_prediction_with_decreasing_load(self) -> None:
        """Test overload prediction with decreasing load."""
        agent_id = "agent-decreasing"

        # Simulate decreasing load
        loads = [0.9, 0.75, 0.6, 0.5, 0.4]
        for load in loads:
            signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.LOAD,
                value=load,
                ttl_seconds=60,
            )
            coordination_service.register_signal(signal)

        # Predict overload
        will_overload, probability = coordination_service.predict_overload(
            agent_id, forecast_seconds=60, threshold=0.8
        )

        # Decreasing trend should not predict overload
        assert will_overload is False
        assert probability < 0.5  # Predicted load should be low


class TestRealisticScenarios:
    """Test realistic coordination scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.router = MessageRouter()
        coordination_service.clear_state()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        coordination_service.clear_state()

    @pytest.mark.asyncio
    async def test_dynamic_load_balancing(self) -> None:
        """Test dynamic load balancing as agent loads change."""
        # Start with 3 agents, all low load
        agents = ["agent-lb-1", "agent-lb-2", "agent-lb-3"]

        for agent_id in agents:
            coordination_service.register_signal(
                SensitivitySignal(
                    agent_id=agent_id,
                    signal_type=SignalType.LOAD,
                    value=0.2,
                    ttl_seconds=60,
                )
            )

        # First selection - any agent is fine
        first_selected = await self.router._select_agent(
            agents, RoutingStrategy.RIPPLE_COORDINATION
        )
        assert first_selected in agents

        # Simulate first agent becoming loaded
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id=first_selected,
                signal_type=SignalType.LOAD,
                value=0.9,  # High load
                ttl_seconds=60,
            )
        )

        # Next selection should avoid the loaded agent
        second_selected = await self.router._select_agent(
            agents, RoutingStrategy.RIPPLE_COORDINATION
        )

        # Should select a different agent with lower load
        assert second_selected != first_selected
        state = coordination_service.get_coordination_state(second_selected)
        assert state is not None
        assert state.signals[SignalType.LOAD].value < 0.5

    def test_quality_based_routing(self) -> None:
        """Test that quality signals influence routing decisions."""
        # Agent 1: High quality
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-hq",
                signal_type=SignalType.QUALITY,
                value=0.98,
                ttl_seconds=60,
            )
        )

        # Agent 2: Low quality
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-lq",
                signal_type=SignalType.QUALITY,
                value=0.3,
                ttl_seconds=60,
            )
        )

        # Selection should prefer high quality
        candidates = ["agent-hq", "agent-lq"]
        selected = coordination_service.select_optimal_agent(candidates)

        assert selected == "agent-hq"

    def test_cost_optimization(self) -> None:
        """Test that cost signals influence routing for cost optimization."""
        # Agent 1: Low cost (high cost score)
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-cheap",
                signal_type=SignalType.COST,
                value=0.9,  # High cost score = low actual cost
                ttl_seconds=60,
            )
        )

        # Agent 2: High cost (low cost score)
        coordination_service.register_signal(
            SensitivitySignal(
                agent_id="agent-expensive",
                signal_type=SignalType.COST,
                value=0.2,  # Low cost score = high actual cost
                ttl_seconds=60,
            )
        )

        # Selection should prefer low-cost agent
        candidates = ["agent-cheap", "agent-expensive"]
        selected = coordination_service.select_optimal_agent(candidates)

        assert selected == "agent-cheap"
