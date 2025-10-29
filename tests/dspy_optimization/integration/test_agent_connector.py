"""Tests for Agent Runtime Connector."""

from __future__ import annotations

import pytest

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCapability,
    AgentCard,
    AgentEndpoint,
    AuthenticationType,
    EndpointType,
)
from agentcore.a2a_protocol.services.agent_manager import AgentManager
from agentcore.dspy_optimization.integration.agent_connector import AgentRuntimeConnector
from agentcore.dspy_optimization.models import (
    OptimizationTargetType,
    PerformanceMetrics,
)


@pytest.fixture
def agent_manager() -> AgentManager:
    """Create agent manager for testing."""
    return AgentManager()


@pytest.fixture
def connector(agent_manager: AgentManager) -> AgentRuntimeConnector:
    """Create agent runtime connector for testing."""
    return AgentRuntimeConnector(agent_manager)


@pytest.fixture
def sample_agent_card() -> AgentCard:
    """Create sample agent card with performance metrics."""
    return AgentCard(
        agent_id="test-agent-001",
        agent_name="Test Agent",
        agent_version="1.0.0",
        endpoints=[
            AgentEndpoint(
                url="https://api.example.com/agent",
                type=EndpointType.HTTPS,
            )
        ],
        capabilities=[
            AgentCapability(
                name="text_generation",
                cost_per_request=0.01,
                avg_latency_ms=150.0,
                quality_score=0.85,
            ),
            AgentCapability(
                name="summarization",
                cost_per_request=0.005,
                avg_latency_ms=100.0,
                quality_score=0.9,
            ),
        ],
        authentication=AgentAuthentication(
            type=AuthenticationType.JWT,
            config={"algorithm": "RS256", "public_key_url": "https://example.com/key"},
        ),
    )


@pytest.mark.asyncio
async def test_get_agent_card(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test getting agent card from runtime."""
    # Register agent
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    # Get agent card
    agent_card = await connector.get_agent_card("test-agent-001")

    assert agent_card is not None
    assert agent_card.agent_id == "test-agent-001"
    assert agent_card.agent_name == "Test Agent"


@pytest.mark.asyncio
async def test_get_agent_card_not_found(connector: AgentRuntimeConnector) -> None:
    """Test getting non-existent agent card."""
    agent_card = await connector.get_agent_card("non-existent")
    assert agent_card is None


@pytest.mark.asyncio
async def test_get_agent_capabilities(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test getting agent capabilities."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    capabilities = await connector.get_agent_capabilities("test-agent-001")

    assert len(capabilities) == 2
    assert capabilities[0].name == "text_generation"
    assert capabilities[1].name == "summarization"


@pytest.mark.asyncio
async def test_get_agent_performance_metrics(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test extracting performance metrics from agent capabilities."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    metrics = await connector.get_agent_performance_metrics("test-agent-001")

    assert metrics is not None
    assert metrics.success_rate == 0.9  # Default value
    assert metrics.avg_cost_per_task == 0.0075  # Average of 0.01 and 0.005
    assert metrics.avg_latency_ms == 125  # Average of 150 and 100
    assert metrics.quality_score == 0.875  # Average of 0.85 and 0.9


@pytest.mark.asyncio
async def test_get_agent_performance_metrics_cached(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test performance metrics caching."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    # First call - should cache
    metrics1 = await connector.get_agent_performance_metrics("test-agent-001")

    # Second call - should use cache
    metrics2 = await connector.get_agent_performance_metrics("test-agent-001")

    assert metrics1 == metrics2


@pytest.mark.asyncio
async def test_update_agent_performance_metrics(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test updating agent performance metrics."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    new_metrics = PerformanceMetrics(
        success_rate=0.95,
        avg_cost_per_task=0.008,
        avg_latency_ms=120,
        quality_score=0.9,
    )

    success = await connector.update_agent_performance_metrics("test-agent-001", new_metrics)

    assert success is True

    # Verify update
    updated_agent = await agent_manager.get_agent("test-agent-001")
    assert updated_agent is not None
    assert updated_agent.capabilities[0].cost_per_request == 0.008
    assert updated_agent.capabilities[0].avg_latency_ms == 120.0
    assert updated_agent.capabilities[0].quality_score == 0.9


@pytest.mark.asyncio
async def test_update_agent_performance_metrics_not_found(
    connector: AgentRuntimeConnector,
) -> None:
    """Test updating metrics for non-existent agent."""
    metrics = PerformanceMetrics(
        success_rate=0.95,
        avg_cost_per_task=0.008,
        avg_latency_ms=120,
        quality_score=0.9,
    )

    success = await connector.update_agent_performance_metrics("non-existent", metrics)
    assert success is False


@pytest.mark.asyncio
async def test_create_optimization_target(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test creating optimization target."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    target = await connector.create_optimization_target("test-agent-001")

    assert target is not None
    assert target.type == OptimizationTargetType.AGENT
    assert target.id == "test-agent-001"


@pytest.mark.asyncio
async def test_create_optimization_target_not_found(
    connector: AgentRuntimeConnector,
) -> None:
    """Test creating optimization target for non-existent agent."""
    target = await connector.create_optimization_target("non-existent")
    assert target is None


@pytest.mark.asyncio
async def test_is_agent_eligible_for_optimization(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test checking agent eligibility for optimization."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    is_eligible = await connector.is_agent_eligible_for_optimization("test-agent-001")
    assert is_eligible is True


@pytest.mark.asyncio
async def test_is_agent_not_eligible_without_metrics(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
) -> None:
    """Test agent without metrics is not eligible."""
    agent_card = AgentCard(
        agent_id="test-agent-002",
        agent_name="Test Agent 2",
        endpoints=[
            AgentEndpoint(
                url="https://api.example.com/agent2",
                type=EndpointType.HTTPS,
            )
        ],
        capabilities=[
            AgentCapability(name="basic_task")  # No performance metrics
        ],
        authentication=AgentAuthentication(type=AuthenticationType.NONE),
    )

    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=agent_card)
    await agent_manager.register_agent(request)

    is_eligible = await connector.is_agent_eligible_for_optimization("test-agent-002")
    assert is_eligible is False


@pytest.mark.asyncio
async def test_get_all_optimizable_agents(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test getting all optimizable agents."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    # Register eligible agent
    request1 = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request1)

    # Register non-eligible agent
    agent_card2 = AgentCard(
        agent_id="test-agent-003",
        agent_name="Test Agent 3",
        endpoints=[
            AgentEndpoint(
                url="https://api.example.com/agent3",
                type=EndpointType.HTTPS,
            )
        ],
        capabilities=[AgentCapability(name="basic_task")],
        authentication=AgentAuthentication(type=AuthenticationType.NONE),
    )
    request2 = AgentRegistrationRequest(agent_card=agent_card2)
    await agent_manager.register_agent(request2)

    optimizable = await connector.get_all_optimizable_agents()

    assert len(optimizable) == 1
    assert "test-agent-001" in optimizable


@pytest.mark.asyncio
async def test_clear_cache(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test clearing performance metrics cache."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    # Cache metrics
    await connector.get_agent_performance_metrics("test-agent-001")

    # Clear cache
    connector.clear_cache("test-agent-001")

    # Verify cache is empty
    assert "test-agent-001" not in connector._performance_cache


@pytest.mark.asyncio
async def test_clear_all_cache(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test clearing all performance metrics cache."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    # Cache metrics
    await connector.get_agent_performance_metrics("test-agent-001")

    # Clear all cache
    connector.clear_cache()

    # Verify cache is empty
    assert len(connector._performance_cache) == 0


@pytest.mark.asyncio
async def test_validate_optimization_request(
    connector: AgentRuntimeConnector,
    agent_manager: AgentManager,
    sample_agent_card: AgentCard,
) -> None:
    """Test validating optimization request."""
    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest
    from agentcore.dspy_optimization.models import (
        OptimizationObjective,
        OptimizationRequest,
        OptimizationTarget,
        MetricType,
    )

    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await agent_manager.register_agent(request)

    opt_request = OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent-001",
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.95,
            )
        ],
    )

    is_valid, error = await connector.validate_optimization_request(opt_request)

    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_validate_optimization_request_agent_not_found(
    connector: AgentRuntimeConnector,
) -> None:
    """Test validating optimization request for non-existent agent."""
    from agentcore.dspy_optimization.models import (
        OptimizationObjective,
        OptimizationRequest,
        OptimizationTarget,
        MetricType,
    )

    opt_request = OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="non-existent",
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.95,
            )
        ],
    )

    is_valid, error = await connector.validate_optimization_request(opt_request)

    assert is_valid is False
    assert "not found" in error
