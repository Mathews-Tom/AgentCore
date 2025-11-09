"""
Unit tests for ErrorAccumulator (ACE-011).

Tests error tracking, severity distribution, compounding error detection,
and pattern detection functionality.
"""

import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.ace.monitors.error_accumulator import (
    ErrorAccumulator,
    ErrorPattern,
    ErrorRecord,
    ErrorSeverity,
)


# Test fixtures


@pytest.fixture
def error_accumulator():
    """Create ErrorAccumulator instance."""
    return ErrorAccumulator()


@pytest.fixture
def sample_task_id():
    """Generate sample task ID."""
    return uuid4()


# Test error tracking


def test_track_error_success(error_accumulator, sample_task_id):
    """Test successful error tracking."""
    error_record = error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Test error message",
    )

    assert error_record is not None
    assert error_record.agent_id == "test-agent"
    assert error_record.task_id == sample_task_id
    assert error_record.stage == "execution"
    assert error_record.error_type == "ValueError"
    assert error_record.severity == ErrorSeverity.MEDIUM
    assert error_record.error_message == "Test error message"
    assert error_record.error_id == 0  # First error


def test_track_error_with_metadata(error_accumulator, sample_task_id):
    """Test error tracking with metadata."""
    metadata = {"line_number": 42, "function": "test_func"}

    error_record = error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="RuntimeError",
        severity=ErrorSeverity.HIGH,
        error_message="Runtime failure",
        metadata=metadata,
    )

    assert error_record.metadata == metadata


def test_track_error_invalid_stage(error_accumulator, sample_task_id):
    """Test error tracking with invalid stage."""
    with pytest.raises(ValueError, match="Invalid stage"):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="invalid_stage",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message="Test error",
        )


def test_track_error_sequential_ids(error_accumulator, sample_task_id):
    """Test sequential error ID assignment."""
    error1 = error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Error 1",
    )

    error2 = error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Error 2",
    )

    assert error1.error_id == 0
    assert error2.error_id == 1


def test_track_error_performance(error_accumulator, sample_task_id):
    """Test error tracking meets <50ms performance target."""
    start_time = time.perf_counter()

    for i in range(100):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="execution",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message=f"Error {i}",
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Average should be well under 50ms per error
    avg_ms = elapsed_ms / 100
    assert avg_ms < 10, f"Average tracking time {avg_ms:.2f}ms exceeds 10ms target"


# Test error count


def test_get_error_count_empty(error_accumulator, sample_task_id):
    """Test error count for task with no errors."""
    count = error_accumulator.get_error_count(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert count == 0


def test_get_error_count_total(error_accumulator, sample_task_id):
    """Test total error count."""
    # Add 5 errors across different stages
    for i in range(5):
        stage = ["planning", "execution", "reflection", "verification", "execution"][i]
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage=stage,
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message=f"Error {i}",
        )

    count = error_accumulator.get_error_count(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert count == 5


def test_get_error_count_by_stage(error_accumulator, sample_task_id):
    """Test error count filtered by stage."""
    # Add errors to different stages
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Error 1",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Error 2",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="RuntimeError",
        severity=ErrorSeverity.HIGH,
        error_message="Error 3",
    )

    execution_count = error_accumulator.get_error_count(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
    )

    planning_count = error_accumulator.get_error_count(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
    )

    assert execution_count == 2
    assert planning_count == 1


def test_get_error_count_invalid_stage(error_accumulator, sample_task_id):
    """Test error count with invalid stage."""
    with pytest.raises(ValueError, match="Invalid stage"):
        error_accumulator.get_error_count(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="invalid_stage",
        )


# Test severity distribution


def test_get_severity_distribution_empty(error_accumulator, sample_task_id):
    """Test severity distribution for task with no errors."""
    distribution = error_accumulator.get_severity_distribution(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert distribution[ErrorSeverity.LOW] == 0
    assert distribution[ErrorSeverity.MEDIUM] == 0
    assert distribution[ErrorSeverity.HIGH] == 0
    assert distribution[ErrorSeverity.CRITICAL] == 0


def test_get_severity_distribution(error_accumulator, sample_task_id):
    """Test severity distribution calculation."""
    # Add errors with different severities
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Low error 1",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Low error 2",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Medium error",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="reflection",
        error_type="RuntimeError",
        severity=ErrorSeverity.CRITICAL,
        error_message="Critical error",
    )

    distribution = error_accumulator.get_severity_distribution(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert distribution[ErrorSeverity.LOW] == 2
    assert distribution[ErrorSeverity.MEDIUM] == 1
    assert distribution[ErrorSeverity.HIGH] == 0
    assert distribution[ErrorSeverity.CRITICAL] == 1


def test_get_severity_distribution_by_stage(error_accumulator, sample_task_id):
    """Test severity distribution filtered by stage."""
    # Add errors to different stages
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Execution low",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="TypeError",
        severity=ErrorSeverity.CRITICAL,
        error_message="Execution critical",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="RuntimeError",
        severity=ErrorSeverity.CRITICAL,
        error_message="Planning critical",
    )

    execution_distribution = error_accumulator.get_severity_distribution(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
    )

    assert execution_distribution[ErrorSeverity.LOW] == 1
    assert execution_distribution[ErrorSeverity.CRITICAL] == 1


def test_get_severity_distribution_invalid_stage(error_accumulator, sample_task_id):
    """Test severity distribution with invalid stage."""
    with pytest.raises(ValueError, match="Invalid stage"):
        error_accumulator.get_severity_distribution(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="invalid_stage",
        )


# Test error rate per stage


def test_get_error_rate_per_stage_empty(error_accumulator, sample_task_id):
    """Test error rate per stage with no errors."""
    rates = error_accumulator.get_error_rate_per_stage(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert rates["planning"] == 0
    assert rates["execution"] == 0
    assert rates["reflection"] == 0
    assert rates["verification"] == 0


def test_get_error_rate_per_stage(error_accumulator, sample_task_id):
    """Test error rate per stage calculation."""
    # Add errors to different stages
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Planning 1",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Execution 1",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="RuntimeError",
        severity=ErrorSeverity.HIGH,
        error_message="Execution 2",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.CRITICAL,
        error_message="Execution 3",
    )

    rates = error_accumulator.get_error_rate_per_stage(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert rates["planning"] == 1
    assert rates["execution"] == 3
    assert rates["reflection"] == 0
    assert rates["verification"] == 0


# Test compounding error detection


def test_detect_compounding_errors_none(error_accumulator, sample_task_id):
    """Test compounding error detection with no patterns."""
    # Add single error
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Single error",
    )

    patterns = error_accumulator.detect_compounding_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert len(patterns) == 0


def test_detect_sequential_errors(error_accumulator, sample_task_id):
    """Test detection of sequential same-type errors."""
    # Add 3 ValueError errors
    for i in range(3):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="execution",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message=f"ValueError {i}",
        )

    patterns = error_accumulator.detect_compounding_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    # Should detect sequential and compounding patterns
    assert len(patterns) > 0

    # Check for sequential pattern
    sequential_patterns = [p for p in patterns if p.pattern_type == "sequential"]
    assert len(sequential_patterns) > 0

    sequential = sequential_patterns[0]
    assert sequential.metadata["error_type"] == "ValueError"
    assert sequential.metadata["occurrence_count"] >= 2


def test_detect_cascading_errors(error_accumulator, sample_task_id):
    """Test detection of cascading errors across stages."""
    # Add errors in consecutive stages
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="ValueError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Planning error",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="RuntimeError",
        severity=ErrorSeverity.HIGH,
        error_message="Execution error",
    )

    patterns = error_accumulator.detect_compounding_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    # Check for cascading pattern
    cascading_patterns = [p for p in patterns if p.pattern_type == "cascading"]
    assert len(cascading_patterns) > 0

    cascading = cascading_patterns[0]
    assert cascading.metadata["from_stage"] == "planning"
    assert cascading.metadata["to_stage"] == "execution"


def test_detect_stage_compounding(error_accumulator, sample_task_id):
    """Test detection of compounding errors in single stage."""
    # Add 4 errors to same stage (exceeds threshold of 3)
    for i in range(4):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="execution",
            error_type=["ValueError", "TypeError", "RuntimeError", "KeyError"][i],
            severity=ErrorSeverity.MEDIUM,
            error_message=f"Execution error {i}",
        )

    patterns = error_accumulator.detect_compounding_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    # Check for compounding pattern
    compounding_patterns = [p for p in patterns if p.pattern_type == "compounding"]
    assert len(compounding_patterns) > 0

    compounding = compounding_patterns[0]
    assert compounding.metadata["stage"] == "execution"
    assert compounding.metadata["error_count"] >= 3


def test_detect_compounding_errors_performance(error_accumulator, sample_task_id):
    """Test compounding error detection meets <50ms performance target."""
    # Add 20 errors to trigger pattern detection
    for i in range(20):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="execution",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message=f"Error {i}",
        )

    # Measure pattern detection time
    start_time = time.perf_counter()

    patterns = error_accumulator.detect_compounding_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Should complete well under 50ms
    assert elapsed_ms < 50, f"Pattern detection {elapsed_ms:.2f}ms exceeds 50ms target"
    assert len(patterns) > 0  # Should detect patterns


# Test error trends


def test_get_error_trends_empty(error_accumulator, sample_task_id):
    """Test error trends with no errors."""
    trends = error_accumulator.get_error_trends(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert trends["total_errors"] == 0
    assert trends["critical_error_count"] == 0
    assert len(trends["detected_patterns"]) == 0


def test_get_error_trends(error_accumulator, sample_task_id):
    """Test error trends calculation."""
    # Add various errors
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Planning error",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="TypeError",
        severity=ErrorSeverity.CRITICAL,
        error_message="Execution error",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="RuntimeError",
        severity=ErrorSeverity.CRITICAL,
        error_message="Execution error 2",
    )

    trends = error_accumulator.get_error_trends(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert trends["total_errors"] == 3
    assert trends["critical_error_count"] == 2
    assert trends["errors_per_stage"]["planning"] == 1
    assert trends["errors_per_stage"]["execution"] == 2
    assert trends["error_types"]["ValueError"] == 1
    assert trends["error_types"]["TypeError"] == 1
    assert trends["error_types"]["RuntimeError"] == 1


def test_get_error_trends_performance(error_accumulator, sample_task_id):
    """Test error trends computation meets <50ms performance target."""
    # Add 50 errors
    for i in range(50):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage=["planning", "execution", "reflection", "verification"][i % 4],
            error_type=["ValueError", "TypeError", "RuntimeError"][i % 3],
            severity=[ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH][i % 3],
            error_message=f"Error {i}",
        )

    # Measure trends computation time
    start_time = time.perf_counter()

    trends = error_accumulator.get_error_trends(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Should complete well under 50ms
    assert elapsed_ms < 50, f"Trends computation {elapsed_ms:.2f}ms exceeds 50ms target"
    assert trends["total_errors"] == 50


# Test MEM integration placeholder


def test_query_mem_error_patterns_stub(error_accumulator, sample_task_id):
    """Test MEM error pattern query placeholder."""
    result = error_accumulator.query_mem_error_patterns(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert result["status"] == "not_implemented"
    assert "Phase 4" in result["message"]
    assert result["agent_id"] == "test-agent"


# Test reset functionality


def test_reset_errors(error_accumulator, sample_task_id):
    """Test error accumulation reset."""
    # Add errors
    for i in range(5):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="execution",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message=f"Error {i}",
        )

    # Verify errors exist
    assert error_accumulator.get_error_count("test-agent", sample_task_id) == 5

    # Reset
    error_accumulator.reset_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    # Verify errors cleared
    assert error_accumulator.get_error_count("test-agent", sample_task_id) == 0
    patterns = error_accumulator.detect_compounding_errors("test-agent", sample_task_id)
    assert len(patterns) == 0


# Test get_all_errors


def test_get_all_errors_empty(error_accumulator, sample_task_id):
    """Test get_all_errors with no errors."""
    errors = error_accumulator.get_all_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert len(errors) == 0


def test_get_all_errors(error_accumulator, sample_task_id):
    """Test get_all_errors retrieval."""
    # Add errors
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="planning",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Error 1",
    )
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Error 2",
    )

    errors = error_accumulator.get_all_errors(
        agent_id="test-agent",
        task_id=sample_task_id,
    )

    assert len(errors) == 2
    assert errors[0].error_type == "ValueError"
    assert errors[1].error_type == "TypeError"


def test_get_all_errors_returns_copy(error_accumulator, sample_task_id):
    """Test get_all_errors returns copy, not reference."""
    # Add error
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Error 1",
    )

    errors1 = error_accumulator.get_all_errors("test-agent", sample_task_id)
    errors2 = error_accumulator.get_all_errors("test-agent", sample_task_id)

    # Should be different list instances
    assert errors1 is not errors2
    # But same content
    assert len(errors1) == len(errors2)


# Test error isolation between tasks


def test_error_isolation_between_tasks(error_accumulator):
    """Test errors are isolated between different tasks."""
    task1 = uuid4()
    task2 = uuid4()

    # Add errors to task1
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=task1,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Task 1 error",
    )

    # Add errors to task2
    error_accumulator.track_error(
        agent_id="test-agent",
        task_id=task2,
        stage="planning",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Task 2 error",
    )

    # Verify isolation
    task1_count = error_accumulator.get_error_count("test-agent", task1)
    task2_count = error_accumulator.get_error_count("test-agent", task2)

    assert task1_count == 1
    assert task2_count == 1

    task1_errors = error_accumulator.get_all_errors("test-agent", task1)
    task2_errors = error_accumulator.get_all_errors("test-agent", task2)

    assert task1_errors[0].error_type == "ValueError"
    assert task2_errors[0].error_type == "TypeError"


# Test edge cases


def test_error_record_repr(error_accumulator, sample_task_id):
    """Test ErrorRecord __repr__ method."""
    error_record = error_accumulator.track_error(
        agent_id="test-agent",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Test error",
    )

    repr_str = repr(error_record)
    assert "ErrorRecord" in repr_str
    assert "ValueError" in repr_str
    assert "medium" in repr_str
    assert "execution" in repr_str


def test_error_pattern_repr(error_accumulator, sample_task_id):
    """Test ErrorPattern __repr__ method."""
    # Create pattern by adding sequential errors
    for i in range(3):
        error_accumulator.track_error(
            agent_id="test-agent",
            task_id=sample_task_id,
            stage="execution",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,
            error_message=f"Error {i}",
        )

    patterns = error_accumulator.detect_compounding_errors("test-agent", sample_task_id)
    assert len(patterns) > 0

    repr_str = repr(patterns[0])
    assert "ErrorPattern" in repr_str


def test_multiple_agents_same_task(error_accumulator, sample_task_id):
    """Test multiple agents can have errors for same task ID."""
    # Agent 1 errors
    error_accumulator.track_error(
        agent_id="agent-1",
        task_id=sample_task_id,
        stage="execution",
        error_type="ValueError",
        severity=ErrorSeverity.LOW,
        error_message="Agent 1 error",
    )

    # Agent 2 errors
    error_accumulator.track_error(
        agent_id="agent-2",
        task_id=sample_task_id,
        stage="planning",
        error_type="TypeError",
        severity=ErrorSeverity.MEDIUM,
        error_message="Agent 2 error",
    )

    # Verify isolation
    agent1_count = error_accumulator.get_error_count("agent-1", sample_task_id)
    agent2_count = error_accumulator.get_error_count("agent-2", sample_task_id)

    assert agent1_count == 1
    assert agent2_count == 1
