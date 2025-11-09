"""
Integration tests for ACE strategic context queries (ACE-022).

Tests the 4 specialized query methods with various scenarios including
success cases, graceful degradation, and performance validation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agentcore.ace.integration.mem_interface import ACEMemoryInterface
from agentcore.ace.models.ace_models import (
    MemoryQueryResult,
    PerformanceMetrics,
    QueryType,
    StrategicContext,
    TriggerSignal,
    TriggerType,
)
from agentcore.ace.monitors.error_accumulator import ErrorRecord, ErrorSeverity


# Test Fixtures


@pytest.fixture
def mem_interface() -> ACEMemoryInterface:
    """Create ACEMemoryInterface instance for testing."""
    return ACEMemoryInterface(seed=42)


@pytest.fixture
def sample_trigger() -> TriggerSignal:
    """Create sample TriggerSignal for testing."""
    return TriggerSignal(
        trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
        signals=["velocity_drop_50pct", "error_rate_2x"],
        rationale="Task velocity dropped 50% below baseline with 2x error rate increase",
        confidence=0.92,
        metric_values={
            "velocity_ratio": 0.5,
            "error_rate_ratio": 2.0,
            "baseline_velocity": 0.6,
            "current_velocity": 0.3,
        },
    )


@pytest.fixture
def sample_errors() -> list[ErrorRecord]:
    """Create sample ErrorRecord list for testing."""
    task_id = uuid4()
    return [
        ErrorRecord(
            error_id=0,
            agent_id="agent-001",
            task_id=task_id,
            stage="execution",
            error_type="file_parsing",
            severity=ErrorSeverity.HIGH,
            error_message="Failed to parse CSV file",
            timestamp=datetime.now(UTC),
        ),
        ErrorRecord(
            error_id=1,
            agent_id="agent-001",
            task_id=task_id,
            stage="execution",
            error_type="memory_retrieval",
            severity=ErrorSeverity.MEDIUM,
            error_message="Memory retrieval timeout",
            timestamp=datetime.now(UTC),
        ),
        ErrorRecord(
            error_id=2,
            agent_id="agent-001",
            task_id=task_id,
            stage="execution",
            error_type="file_parsing",
            severity=ErrorSeverity.CRITICAL,
            error_message="File parsing failed again",
            timestamp=datetime.now(UTC),
        ),
    ]


@pytest.fixture
def sample_performance_metrics() -> PerformanceMetrics:
    """Create sample PerformanceMetrics for testing."""
    return PerformanceMetrics(
        task_id=uuid4(),
        agent_id="agent-001",
        stage="execution",
        stage_success_rate=0.75,
        stage_error_rate=0.25,
        stage_duration_ms=2500,
        stage_action_count=12,
        overall_progress_velocity=4.2,
        error_accumulation_rate=0.35,
        context_staleness_score=0.45,
        intervention_effectiveness=None,
    )


# Test: query_for_strategic_decision


@pytest.mark.asyncio
async def test_query_for_strategic_decision_performance_degradation(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test strategic decision query with PERFORMANCE_DEGRADATION trigger."""
    trigger = TriggerSignal(
        trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
        signals=["velocity_drop_50pct"],
        rationale="Velocity dropped significantly",
        confidence=0.85,
        metric_values={"velocity_ratio": 0.5},
    )
    agent_id = "agent-001"
    task_id = uuid4()

    result = await mem_interface.query_for_strategic_decision(trigger, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.strategic_context is not None
    assert result.relevance_score >= 0.0
    assert result.relevance_score <= 1.0
    assert result.query_latency_ms >= 0
    assert result.query_latency_ms < 150  # <150ms requirement
    assert "fallback" not in result.metadata or not result.metadata["fallback"]


@pytest.mark.asyncio
async def test_query_for_strategic_decision_error_accumulation(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test strategic decision query with ERROR_ACCUMULATION trigger."""
    trigger = TriggerSignal(
        trigger_type=TriggerType.ERROR_ACCUMULATION,
        signals=["error_rate_3x", "compounding_errors"],
        rationale="Error rate increased 3x with compounding errors detected",
        confidence=0.95,
        metric_values={"error_rate_ratio": 3.0, "compounding_count": 5},
    )
    agent_id = "agent-002"
    task_id = uuid4()

    result = await mem_interface.query_for_strategic_decision(trigger, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_strategic_decision_context_staleness(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test strategic decision query with CONTEXT_STALENESS trigger."""
    trigger = TriggerSignal(
        trigger_type=TriggerType.CONTEXT_STALENESS,
        signals=["staleness_high"],
        rationale="Context staleness score exceeded threshold",
        confidence=0.78,
        metric_values={"staleness_score": 0.85},
    )
    agent_id = "agent-003"
    task_id = uuid4()

    result = await mem_interface.query_for_strategic_decision(trigger, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_strategic_decision_capability_mismatch(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test strategic decision query with CAPABILITY_MISMATCH trigger."""
    trigger = TriggerSignal(
        trigger_type=TriggerType.CAPABILITY_MISMATCH,
        signals=["missing_capabilities"],
        rationale="Agent lacks required capabilities for task",
        confidence=0.88,
        metric_values={"coverage_ratio": 0.4},
    )
    agent_id = "agent-004"
    task_id = uuid4()

    result = await mem_interface.query_for_strategic_decision(trigger, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_strategic_decision_graceful_degradation(
    mem_interface: ACEMemoryInterface,
    sample_trigger: TriggerSignal,
) -> None:
    """Test graceful degradation when get_strategic_context fails."""
    agent_id = "agent-001"
    task_id = uuid4()

    # Mock get_strategic_context to raise exception
    with patch.object(
        mem_interface,
        "get_strategic_context",
        side_effect=RuntimeError("MEM service unavailable"),
    ):
        result = await mem_interface.query_for_strategic_decision(
            sample_trigger, agent_id, task_id
        )

    # Verify fallback behavior
    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score == 0.3  # Low fallback score
    assert result.query_latency_ms == 0
    assert result.metadata["fallback"] is True
    assert "error" in result.metadata
    assert result.strategic_context.context_health_score == 0.3
    assert len(result.strategic_context.relevant_stage_summaries) > 0
    assert "Fallback" in result.strategic_context.relevant_stage_summaries[0]


# Test: query_for_error_analysis


@pytest.mark.asyncio
async def test_query_for_error_analysis_multiple_errors(
    mem_interface: ACEMemoryInterface,
    sample_errors: list[ErrorRecord],
) -> None:
    """Test error analysis query with multiple errors."""
    agent_id = "agent-001"
    task_id = uuid4()

    result = await mem_interface.query_for_error_analysis(sample_errors, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.strategic_context is not None
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150
    assert "fallback" not in result.metadata or not result.metadata["fallback"]


@pytest.mark.asyncio
async def test_query_for_error_analysis_single_error(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test error analysis query with single error."""
    task_id = uuid4()
    errors = [
        ErrorRecord(
            error_id=0,
            agent_id="agent-001",
            task_id=task_id,
            stage="planning",
            error_type="validation",
            severity=ErrorSeverity.LOW,
            error_message="Input validation failed",
            timestamp=datetime.now(UTC),
        ),
    ]
    agent_id = "agent-001"

    result = await mem_interface.query_for_error_analysis(errors, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_error_analysis_empty_errors(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test error analysis query with empty error list."""
    agent_id = "agent-001"
    task_id = uuid4()
    errors: list[ErrorRecord] = []

    result = await mem_interface.query_for_error_analysis(errors, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_error_analysis_compounding_errors(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test error analysis query with compounding errors (same type)."""
    task_id = uuid4()
    # Create 5 errors of same type
    errors = [
        ErrorRecord(
            error_id=i,
            agent_id="agent-001",
            task_id=task_id,
            stage="execution",
            error_type="api_timeout",
            severity=ErrorSeverity.HIGH,
            error_message=f"API timeout error {i}",
            timestamp=datetime.now(UTC),
        )
        for i in range(5)
    ]
    agent_id = "agent-001"

    result = await mem_interface.query_for_error_analysis(errors, agent_id, task_id)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_error_analysis_graceful_degradation(
    mem_interface: ACEMemoryInterface,
    sample_errors: list[ErrorRecord],
) -> None:
    """Test graceful degradation when get_strategic_context fails."""
    agent_id = "agent-001"
    task_id = uuid4()

    # Mock get_strategic_context to raise exception
    with patch.object(
        mem_interface,
        "get_strategic_context",
        side_effect=RuntimeError("MEM service unavailable"),
    ):
        result = await mem_interface.query_for_error_analysis(
            sample_errors, agent_id, task_id
        )

    # Verify fallback behavior
    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score == 0.3
    assert result.query_latency_ms == 0
    assert result.metadata["fallback"] is True
    assert "error" in result.metadata
    assert result.metadata["error_count"] == len(sample_errors)
    assert result.strategic_context.context_health_score == 0.3
    assert "Fallback" in result.strategic_context.relevant_stage_summaries[0]
    assert str(len(sample_errors)) in result.strategic_context.relevant_stage_summaries[0]


# Test: query_for_capability_evaluation


@pytest.mark.asyncio
async def test_query_for_capability_evaluation_full_match(
    mem_interface: ACEMemoryInterface,
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test capability evaluation with full capability match."""
    task_requirements = ["file_read", "api_call", "data_transform"]
    agent_capabilities = ["file_read", "api_call", "data_transform", "search"]
    agent_id = "agent-001"
    task_id = uuid4()

    result = await mem_interface.query_for_capability_evaluation(
        task_requirements, agent_capabilities, sample_performance_metrics, agent_id, task_id
    )

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150
    assert "fallback" not in result.metadata or not result.metadata["fallback"]


@pytest.mark.asyncio
async def test_query_for_capability_evaluation_partial_match(
    mem_interface: ACEMemoryInterface,
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test capability evaluation with partial capability match."""
    task_requirements = ["file_read", "api_call", "data_transform", "parallel_exec"]
    agent_capabilities = ["file_read", "api_call"]  # Missing 2 capabilities
    agent_id = "agent-002"
    task_id = uuid4()

    result = await mem_interface.query_for_capability_evaluation(
        task_requirements, agent_capabilities, sample_performance_metrics, agent_id, task_id
    )

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_capability_evaluation_no_match(
    mem_interface: ACEMemoryInterface,
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test capability evaluation with no capability match."""
    task_requirements = ["advanced_ml", "gpu_compute"]
    agent_capabilities = ["file_read", "api_call"]  # No overlap
    agent_id = "agent-003"
    task_id = uuid4()

    result = await mem_interface.query_for_capability_evaluation(
        task_requirements, agent_capabilities, sample_performance_metrics, agent_id, task_id
    )

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_capability_evaluation_empty_requirements(
    mem_interface: ACEMemoryInterface,
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test capability evaluation with empty requirements."""
    task_requirements: list[str] = []
    agent_capabilities = ["file_read", "api_call"]
    agent_id = "agent-004"
    task_id = uuid4()

    result = await mem_interface.query_for_capability_evaluation(
        task_requirements, agent_capabilities, sample_performance_metrics, agent_id, task_id
    )

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_capability_evaluation_graceful_degradation(
    mem_interface: ACEMemoryInterface,
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test graceful degradation when get_strategic_context fails."""
    task_requirements = ["file_read", "api_call"]
    agent_capabilities = ["file_read"]
    agent_id = "agent-001"
    task_id = uuid4()

    # Mock get_strategic_context to raise exception
    with patch.object(
        mem_interface,
        "get_strategic_context",
        side_effect=RuntimeError("MEM service unavailable"),
    ):
        result = await mem_interface.query_for_capability_evaluation(
            task_requirements,
            agent_capabilities,
            sample_performance_metrics,
            agent_id,
            task_id,
        )

    # Verify fallback behavior
    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score == 0.3
    assert result.query_latency_ms == 0
    assert result.metadata["fallback"] is True
    assert "error" in result.metadata
    assert "coverage_ratio" in result.metadata
    assert result.strategic_context.context_health_score == 0.3
    assert "Fallback" in result.strategic_context.relevant_stage_summaries[0]
    assert "50%" in result.strategic_context.relevant_stage_summaries[0]  # 1/2 coverage


# Test: query_for_context_refresh


@pytest.mark.asyncio
async def test_query_for_context_refresh_planning_stage(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test context refresh query for planning stage."""
    agent_id = "agent-001"
    task_id = uuid4()
    current_stage = "planning"

    result = await mem_interface.query_for_context_refresh(agent_id, task_id, current_stage)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150
    assert "fallback" not in result.metadata or not result.metadata["fallback"]


@pytest.mark.asyncio
async def test_query_for_context_refresh_execution_stage(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test context refresh query for execution stage."""
    agent_id = "agent-002"
    task_id = uuid4()
    current_stage = "execution"

    result = await mem_interface.query_for_context_refresh(agent_id, task_id, current_stage)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_context_refresh_reflection_stage(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test context refresh query for reflection stage."""
    agent_id = "agent-003"
    task_id = uuid4()
    current_stage = "reflection"

    result = await mem_interface.query_for_context_refresh(agent_id, task_id, current_stage)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_context_refresh_verification_stage(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test context refresh query for verification stage."""
    agent_id = "agent-004"
    task_id = uuid4()
    current_stage = "verification"

    result = await mem_interface.query_for_context_refresh(agent_id, task_id, current_stage)

    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score >= 0.0
    assert result.query_latency_ms < 150


@pytest.mark.asyncio
async def test_query_for_context_refresh_invalid_stage(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test context refresh query with invalid stage uses graceful degradation."""
    agent_id = "agent-001"
    task_id = uuid4()
    current_stage = "invalid_stage"

    # Invalid stage triggers graceful degradation (not a hard failure)
    result = await mem_interface.query_for_context_refresh(agent_id, task_id, current_stage)

    # Should return fallback result
    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score == 0.3
    assert result.metadata["fallback"] is True
    assert "Invalid stage" in result.metadata["error"]


@pytest.mark.asyncio
async def test_query_for_context_refresh_graceful_degradation(
    mem_interface: ACEMemoryInterface,
) -> None:
    """Test graceful degradation when get_strategic_context fails."""
    agent_id = "agent-001"
    task_id = uuid4()
    current_stage = "execution"

    # Mock get_strategic_context to raise exception
    with patch.object(
        mem_interface,
        "get_strategic_context",
        side_effect=RuntimeError("MEM service unavailable"),
    ):
        result = await mem_interface.query_for_context_refresh(
            agent_id, task_id, current_stage
        )

    # Verify fallback behavior
    assert isinstance(result, MemoryQueryResult)
    assert result.relevance_score == 0.3
    assert result.query_latency_ms == 0
    assert result.metadata["fallback"] is True
    assert "error" in result.metadata
    assert result.metadata["current_stage"] == current_stage
    assert result.strategic_context.context_health_score == 0.3
    assert "Fallback" in result.strategic_context.relevant_stage_summaries[0]
    assert current_stage in result.strategic_context.relevant_stage_summaries[0]


# Test: Performance validation


@pytest.mark.asyncio
async def test_all_queries_meet_latency_requirement(
    mem_interface: ACEMemoryInterface,
    sample_trigger: TriggerSignal,
    sample_errors: list[ErrorRecord],
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test that all query methods meet <150ms latency requirement."""
    agent_id = "agent-001"
    task_id = uuid4()

    # Test strategic decision query
    result1 = await mem_interface.query_for_strategic_decision(
        sample_trigger, agent_id, task_id
    )
    assert result1.query_latency_ms < 150

    # Test error analysis query
    result2 = await mem_interface.query_for_error_analysis(sample_errors, agent_id, task_id)
    assert result2.query_latency_ms < 150

    # Test capability evaluation query
    result3 = await mem_interface.query_for_capability_evaluation(
        ["file_read"], ["file_read"], sample_performance_metrics, agent_id, task_id
    )
    assert result3.query_latency_ms < 150

    # Test context refresh query
    result4 = await mem_interface.query_for_context_refresh(agent_id, task_id, "execution")
    assert result4.query_latency_ms < 150


@pytest.mark.asyncio
async def test_relevance_scores_are_valid(
    mem_interface: ACEMemoryInterface,
    sample_trigger: TriggerSignal,
    sample_errors: list[ErrorRecord],
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test that all query methods return valid relevance scores (0-1)."""
    agent_id = "agent-001"
    task_id = uuid4()

    results = [
        await mem_interface.query_for_strategic_decision(sample_trigger, agent_id, task_id),
        await mem_interface.query_for_error_analysis(sample_errors, agent_id, task_id),
        await mem_interface.query_for_capability_evaluation(
            ["file_read"], ["file_read"], sample_performance_metrics, agent_id, task_id
        ),
        await mem_interface.query_for_context_refresh(agent_id, task_id, "execution"),
    ]

    for result in results:
        assert 0.0 <= result.relevance_score <= 1.0
        assert 0.0 <= result.strategic_context.context_health_score <= 1.0


@pytest.mark.asyncio
async def test_fallback_results_have_consistent_structure(
    mem_interface: ACEMemoryInterface,
    sample_trigger: TriggerSignal,
    sample_errors: list[ErrorRecord],
    sample_performance_metrics: PerformanceMetrics,
) -> None:
    """Test that fallback results have consistent structure across all query types."""
    agent_id = "agent-001"
    task_id = uuid4()

    # Mock all queries to fail
    with patch.object(
        mem_interface,
        "get_strategic_context",
        side_effect=RuntimeError("MEM unavailable"),
    ):
        fallback_results = [
            await mem_interface.query_for_strategic_decision(
                sample_trigger, agent_id, task_id
            ),
            await mem_interface.query_for_error_analysis(sample_errors, agent_id, task_id),
            await mem_interface.query_for_capability_evaluation(
                ["file_read"], ["file_read"], sample_performance_metrics, agent_id, task_id
            ),
            await mem_interface.query_for_context_refresh(agent_id, task_id, "execution"),
        ]

    # All fallback results should have consistent structure
    for result in fallback_results:
        assert result.relevance_score == 0.3
        assert result.query_latency_ms == 0
        assert result.metadata["fallback"] is True
        assert "error" in result.metadata
        assert result.strategic_context.context_health_score == 0.3
        assert len(result.strategic_context.relevant_stage_summaries) > 0
        assert "Fallback" in result.strategic_context.relevant_stage_summaries[0]
