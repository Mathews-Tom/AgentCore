"""Tests for Agent Performance Feedback Loop."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

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
from agentcore.dspy_optimization.integration.feedback_loop import (
    AgentPerformanceFeedbackLoop,
    AgentPerformanceTracker,
    FeedbackLoopConfig,
)
from agentcore.dspy_optimization.integration.monitoring_hooks import OptimizationMonitor
from agentcore.dspy_optimization.models import PerformanceMetrics
from agentcore.dspy_optimization.pipeline import OptimizationPipeline


@pytest.fixture
def tracker() -> AgentPerformanceTracker:
    """Create performance tracker for testing."""
    return AgentPerformanceTracker()


@pytest.fixture
def sample_metrics() -> PerformanceMetrics:
    """Create sample performance metrics."""
    return PerformanceMetrics(
        success_rate=0.9,
        avg_cost_per_task=0.01,
        avg_latency_ms=150,
        quality_score=0.85,
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
def pipeline() -> OptimizationPipeline:
    """Create optimization pipeline for testing."""
    return OptimizationPipeline(enable_tracking=False)


@pytest.fixture
def monitor() -> OptimizationMonitor:
    """Create optimization monitor for testing."""
    return OptimizationMonitor()


@pytest.fixture
def feedback_loop(
    connector: AgentRuntimeConnector,
    pipeline: OptimizationPipeline,
    monitor: OptimizationMonitor,
) -> AgentPerformanceFeedbackLoop:
    """Create feedback loop for testing."""
    config = FeedbackLoopConfig(
        check_interval_seconds=1,
        performance_degradation_threshold=0.1,
        enable_auto_optimization=False,  # Disable for testing
    )
    return AgentPerformanceFeedbackLoop(connector, pipeline, config, monitor)


def test_tracker_record_performance(tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics) -> None:
    """Test recording agent performance metrics."""
    tracker.record_performance("agent-123", sample_metrics)

    recent = tracker.get_recent_performance("agent-123")
    assert len(recent) == 1
    assert recent[0] == sample_metrics


def test_tracker_set_baseline(tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics) -> None:
    """Test setting baseline metrics."""
    tracker.set_baseline("agent-123", sample_metrics)

    baseline = tracker.get_baseline("agent-123")
    assert baseline == sample_metrics


def test_tracker_get_baseline_not_found(tracker: AgentPerformanceTracker) -> None:
    """Test getting baseline for non-existent agent."""
    baseline = tracker.get_baseline("non-existent")
    assert baseline is None


def test_tracker_get_recent_performance(tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics) -> None:
    """Test getting recent performance within time window."""
    tracker.record_performance("agent-123", sample_metrics)

    recent = tracker.get_recent_performance("agent-123", window_seconds=3600)
    assert len(recent) == 1


def test_tracker_get_recent_performance_outside_window(
    tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics
) -> None:
    """Test that old performance metrics are filtered out."""
    # Record performance
    tracker.record_performance("agent-123", sample_metrics)

    # Manually adjust timestamp to be outside window
    old_timestamp = datetime.utcnow() - timedelta(hours=2)
    tracker._performance_history["agent-123"][0] = (old_timestamp, sample_metrics)

    # Should not return old metrics
    recent = tracker.get_recent_performance("agent-123", window_seconds=3600)
    assert len(recent) == 0


def test_tracker_calculate_average_performance(
    tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics
) -> None:
    """Test calculating average performance."""
    metrics1 = PerformanceMetrics(
        success_rate=0.9,
        avg_cost_per_task=0.01,
        avg_latency_ms=150,
        quality_score=0.85,
    )
    metrics2 = PerformanceMetrics(
        success_rate=0.95,
        avg_cost_per_task=0.008,
        avg_latency_ms=120,
        quality_score=0.9,
    )

    tracker.record_performance("agent-123", metrics1)
    tracker.record_performance("agent-123", metrics2)

    avg = tracker.calculate_average_performance("agent-123")

    assert avg is not None
    assert avg.success_rate == 0.925  # (0.9 + 0.95) / 2
    assert abs(avg.avg_cost_per_task - 0.009) < 0.0001  # (0.01 + 0.008) / 2
    assert avg.avg_latency_ms == 135  # (150 + 120) / 2
    assert avg.quality_score == 0.875  # (0.85 + 0.9) / 2


def test_tracker_calculate_average_performance_no_data(tracker: AgentPerformanceTracker) -> None:
    """Test calculating average with no data."""
    avg = tracker.calculate_average_performance("agent-123")
    assert avg is None


def test_tracker_detect_performance_degradation(tracker: AgentPerformanceTracker) -> None:
    """Test detecting performance degradation."""
    baseline = PerformanceMetrics(
        success_rate=0.95,
        avg_cost_per_task=0.01,
        avg_latency_ms=150,
        quality_score=0.9,
    )

    degraded = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.015,
        avg_latency_ms=200,
        quality_score=0.75,
    )

    tracker.set_baseline("agent-123", baseline)
    tracker.record_performance("agent-123", degraded)

    has_degraded, percentage = tracker.detect_performance_degradation("agent-123", threshold=0.1)

    assert has_degraded is True
    assert percentage > 0.1


def test_tracker_detect_no_degradation(tracker: AgentPerformanceTracker) -> None:
    """Test detecting no degradation when performance is good."""
    baseline = PerformanceMetrics(
        success_rate=0.95,
        avg_cost_per_task=0.01,
        avg_latency_ms=150,
        quality_score=0.9,
    )

    good_metrics = PerformanceMetrics(
        success_rate=0.96,
        avg_cost_per_task=0.009,
        avg_latency_ms=140,
        quality_score=0.91,
    )

    tracker.set_baseline("agent-123", baseline)
    tracker.record_performance("agent-123", good_metrics)

    has_degraded, percentage = tracker.detect_performance_degradation("agent-123", threshold=0.1)

    assert has_degraded is False


def test_tracker_detect_degradation_no_baseline(tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics) -> None:
    """Test degradation detection without baseline."""
    tracker.record_performance("agent-123", sample_metrics)

    has_degraded, percentage = tracker.detect_performance_degradation("agent-123")

    assert has_degraded is False
    assert percentage == 0.0


def test_feedback_loop_add_agent(feedback_loop: AgentPerformanceFeedbackLoop, sample_metrics: PerformanceMetrics) -> None:
    """Test adding agent to feedback loop."""
    feedback_loop.add_agent("agent-123", baseline_metrics=sample_metrics)

    assert "agent-123" in feedback_loop._monitored_agents
    assert feedback_loop.tracker.get_baseline("agent-123") == sample_metrics


def test_feedback_loop_remove_agent(feedback_loop: AgentPerformanceFeedbackLoop) -> None:
    """Test removing agent from feedback loop."""
    feedback_loop.add_agent("agent-123")
    feedback_loop.remove_agent("agent-123")

    assert "agent-123" not in feedback_loop._monitored_agents


@pytest.mark.asyncio
async def test_feedback_loop_start_stop(feedback_loop: AgentPerformanceFeedbackLoop) -> None:
    """Test starting and stopping feedback loop."""
    await feedback_loop.start()
    assert feedback_loop.is_running() is True

    await feedback_loop.stop()
    assert feedback_loop.is_running() is False


@pytest.mark.asyncio
async def test_feedback_loop_get_agent_status(
    feedback_loop: AgentPerformanceFeedbackLoop, sample_metrics: PerformanceMetrics
) -> None:
    """Test getting agent status from feedback loop."""
    feedback_loop.add_agent("agent-123", baseline_metrics=sample_metrics)

    status = await feedback_loop.get_agent_status("agent-123")

    assert status["agent_id"] == "agent-123"
    assert status["is_monitored"] is True
    assert status["has_baseline"] is True
    assert status["baseline_metrics"] is not None


@pytest.mark.asyncio
async def test_feedback_loop_get_monitored_agents(feedback_loop: AgentPerformanceFeedbackLoop) -> None:
    """Test getting list of monitored agents."""
    feedback_loop.add_agent("agent-123")
    feedback_loop.add_agent("agent-456")

    agents = feedback_loop.get_monitored_agents()

    assert len(agents) == 2
    assert "agent-123" in agents
    assert "agent-456" in agents


def test_feedback_loop_can_optimize_no_cooldown(feedback_loop: AgentPerformanceFeedbackLoop) -> None:
    """Test that agent can be optimized when not in cooldown."""
    can_optimize = feedback_loop._can_optimize("agent-123")
    assert can_optimize is True


def test_feedback_loop_can_optimize_in_cooldown(feedback_loop: AgentPerformanceFeedbackLoop) -> None:
    """Test that agent cannot be optimized during cooldown."""
    feedback_loop._last_optimization["agent-123"] = datetime.utcnow()

    can_optimize = feedback_loop._can_optimize("agent-123")
    assert can_optimize is False


def test_feedback_loop_can_optimize_after_cooldown(feedback_loop: AgentPerformanceFeedbackLoop) -> None:
    """Test that agent can be optimized after cooldown expires."""
    feedback_loop._last_optimization["agent-123"] = datetime.utcnow() - timedelta(hours=2)

    can_optimize = feedback_loop._can_optimize("agent-123")
    assert can_optimize is True


def test_feedback_loop_config_defaults() -> None:
    """Test feedback loop configuration defaults."""
    config = FeedbackLoopConfig()

    assert config.check_interval_seconds == 300
    assert config.performance_degradation_threshold == 0.1
    assert config.min_data_points == 10
    assert config.cooldown_period_seconds == 3600
    assert config.enable_auto_optimization is True


def test_feedback_loop_config_custom() -> None:
    """Test feedback loop configuration with custom values."""
    config = FeedbackLoopConfig(
        check_interval_seconds=600,
        performance_degradation_threshold=0.15,
        min_data_points=20,
        cooldown_period_seconds=7200,
        enable_auto_optimization=False,
    )

    assert config.check_interval_seconds == 600
    assert config.performance_degradation_threshold == 0.15
    assert config.min_data_points == 20
    assert config.cooldown_period_seconds == 7200
    assert config.enable_auto_optimization is False


@pytest.mark.asyncio
async def test_feedback_loop_check_agent_performance(
    feedback_loop: AgentPerformanceFeedbackLoop,
    agent_manager: AgentManager,
    sample_metrics: PerformanceMetrics,
) -> None:
    """Test checking individual agent performance."""
    # Create and register agent
    agent_card = AgentCard(
        agent_id="agent-123",
        agent_name="Test Agent",
        endpoints=[
            AgentEndpoint(
                url="https://api.example.com/agent",
                type=EndpointType.HTTPS,
            )
        ],
        capabilities=[
            AgentCapability(
                name="test_capability",
                cost_per_request=0.01,
                avg_latency_ms=150.0,
                quality_score=0.85,
            )
        ],
        authentication=AgentAuthentication(type=AuthenticationType.NONE),
    )

    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=agent_card)
    await agent_manager.register_agent(request)

    feedback_loop.add_agent("agent-123")

    # Check performance
    await feedback_loop._check_agent_performance("agent-123")

    # Verify baseline was set
    baseline = feedback_loop.tracker.get_baseline("agent-123")
    assert baseline is not None


@pytest.mark.asyncio
async def test_feedback_loop_validate_request_integration(
    feedback_loop: AgentPerformanceFeedbackLoop,
    agent_manager: AgentManager,
) -> None:
    """Test validation of optimization request in feedback loop context."""
    # Create and register agent
    agent_card = AgentCard(
        agent_id="agent-123",
        agent_name="Test Agent",
        endpoints=[
            AgentEndpoint(
                url="https://api.example.com/agent",
                type=EndpointType.HTTPS,
            )
        ],
        capabilities=[
            AgentCapability(
                name="test_capability",
                cost_per_request=0.01,
                avg_latency_ms=150.0,
                quality_score=0.85,
            )
        ],
        authentication=AgentAuthentication(type=AuthenticationType.NONE),
    )

    from agentcore.a2a_protocol.models.agent import AgentRegistrationRequest

    request = AgentRegistrationRequest(agent_card=agent_card)
    await agent_manager.register_agent(request)

    # Validate via connector
    from agentcore.dspy_optimization.integration.target_spec import (
        AgentOptimizationTarget,
    )

    opt_request = AgentOptimizationTarget.create_optimization_request("agent-123")
    is_valid, error = await feedback_loop.connector.validate_optimization_request(opt_request)

    assert is_valid is True
    assert error is None


def test_tracker_history_limit(tracker: AgentPerformanceTracker, sample_metrics: PerformanceMetrics) -> None:
    """Test that performance history is limited."""
    # Record 1100 metrics
    for _ in range(1100):
        tracker.record_performance("agent-123", sample_metrics)

    history = tracker._performance_history["agent-123"]
    assert len(history) == 1000  # Should be capped at 1000
