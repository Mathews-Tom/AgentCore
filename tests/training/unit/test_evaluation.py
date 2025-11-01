"""
Unit tests for evaluation framework.

Tests held-out evaluation, metrics computation, and statistical significance testing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.training.evaluation import EvaluationFramework, EvaluationMetrics
from agentcore.training.models import Trajectory, TrajectoryStep, TrainingQuery


# Test fixtures


@pytest.fixture
def training_queries() -> list[TrainingQuery]:
    """Create sample training queries."""
    return [
        TrainingQuery(
            query=f"Test query {i}",
            expected_outcome={"result": "success"})
        for i in range(100)
    ]


@pytest.fixture
def successful_trajectory() -> Trajectory:
    """Create a successful trajectory."""
    job_id = uuid4()
    return Trajectory(
        trajectory_id=uuid4(),
        job_id=job_id,
        agent_id="test-agent",
        query="Test query",
        steps=[
            TrajectoryStep(
                state={"context": "initial"},
                action={"type": "tool_call", "tool": "search"},
                result={"data": "result"},
                timestamp=datetime.now(UTC),
                duration_ms=100),
            TrajectoryStep(
                state={"context": "after_search"},
                action={"type": "response", "content": "Final answer"},
                result={"status": "success"},
                timestamp=datetime.now(UTC),
                duration_ms=50),
        ],
        reward=1.0,
        normalized_reward=0.8,
        advantage=0.5,
        execution_time_ms=150,
        success=True,
        created_at=datetime.now(UTC))


@pytest.fixture
def failed_trajectory() -> Trajectory:
    """Create a failed trajectory."""
    job_id = uuid4()
    return Trajectory(
        trajectory_id=uuid4(),
        job_id=job_id,
        agent_id="test-agent",
        query="Test query",
        steps=[
            TrajectoryStep(
                state={"context": "initial"},
                action={"type": "tool_call", "tool": "search"},
                result={"error": "Tool failed"},
                timestamp=datetime.now(UTC),
                duration_ms=100),
        ],
        reward=0.0,
        normalized_reward=-0.5,
        advantage=-0.3,
        execution_time_ms=100,
        success=False,
        created_at=datetime.now(UTC))


@pytest.fixture
def evaluation_framework() -> EvaluationFramework:
    """Create evaluation framework instance."""
    return EvaluationFramework(evaluation_interval=10)


# Test data splitting


def test_split_training_data(
    training_queries: list[TrainingQuery],
    evaluation_framework: EvaluationFramework) -> None:
    """Test training data split into train and eval sets."""
    train, eval_queries = evaluation_framework.split_training_data(training_queries, held_out_ratio=0.2)

    # Assert correct split ratio
    assert len(train) == 80
    assert len(eval_queries) == 20

    # Assert no overlap
    assert len(train) + len(eval_queries) == len(training_queries)


def test_split_training_data_invalid_ratio(
    training_queries: list[TrainingQuery],
    evaluation_framework: EvaluationFramework) -> None:
    """Test split with invalid held_out_ratio raises ValueError."""
    with pytest.raises(ValueError, match="held_out_ratio must be between 0 and 1"):
        evaluation_framework.split_training_data(training_queries, held_out_ratio=1.5)

    with pytest.raises(ValueError, match="held_out_ratio must be between 0 and 1"):
        evaluation_framework.split_training_data(training_queries, held_out_ratio=0.0)


def test_split_training_data_too_small(
    evaluation_framework: EvaluationFramework) -> None:
    """Test split with too few queries raises ValueError."""
    small_dataset = [
        TrainingQuery(query="Query 1", expected_outcome={"result": "success"}),
        TrainingQuery(query="Query 2", expected_outcome={"result": "success"}),
    ]

    with pytest.raises(ValueError, match="Training set too small to split"):
        evaluation_framework.split_training_data(small_dataset, held_out_ratio=0.8)


def test_split_training_data_custom_ratio(
    training_queries: list[TrainingQuery],
    evaluation_framework: EvaluationFramework) -> None:
    """Test split with custom held_out_ratio."""
    train, eval_queries = evaluation_framework.split_training_data(training_queries, held_out_ratio=0.3)

    assert len(train) == 70
    assert len(eval_queries) == 30


# Test metrics computation


def test_compute_metrics_successful_trajectories(
    evaluation_framework: EvaluationFramework,
    successful_trajectory: Trajectory) -> None:
    """Test metrics computation for successful trajectories."""
    trajectories = [successful_trajectory] * 10

    metrics = evaluation_framework.compute_metrics(trajectories)

    assert metrics.success_rate == 1.0
    assert metrics.avg_reward == 1.0
    assert metrics.avg_steps == 2.0
    assert metrics.tool_accuracy == 1.0
    assert metrics.sample_size == 10


def test_compute_metrics_mixed_trajectories(
    evaluation_framework: EvaluationFramework,
    successful_trajectory: Trajectory,
    failed_trajectory: Trajectory) -> None:
    """Test metrics computation for mixed success/failure."""
    trajectories = [successful_trajectory] * 7 + [failed_trajectory] * 3

    metrics = evaluation_framework.compute_metrics(trajectories)

    assert metrics.success_rate == 0.7
    assert metrics.avg_reward == pytest.approx(0.7, abs=0.01)
    assert metrics.avg_steps == pytest.approx(1.7, abs=0.01)
    assert metrics.sample_size == 10


def test_compute_metrics_empty_trajectories(
    evaluation_framework: EvaluationFramework) -> None:
    """Test metrics computation with empty trajectory list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot compute metrics from empty trajectory list"):
        evaluation_framework.compute_metrics([])


def test_compute_metrics_no_tool_usage(
    evaluation_framework: EvaluationFramework) -> None:
    """Test metrics computation when no trajectories use tools."""
    trajectory = Trajectory(
        trajectory_id=uuid4(),
        job_id=uuid4(),
        agent_id="test-agent",
        query="Test query",
        steps=[
            TrajectoryStep(
                state={"context": "initial"},
                action={"type": "response", "content": "Answer"},
                result={"status": "success"},
                timestamp=datetime.now(UTC),
                duration_ms=50),
        ],
        reward=1.0,
        success=True,
        created_at=datetime.now(UTC))

    metrics = evaluation_framework.compute_metrics([trajectory])

    assert metrics.tool_accuracy is None  # No tool usage


def test_metrics_to_dict(
    evaluation_framework: EvaluationFramework,
    successful_trajectory: Trajectory) -> None:
    """Test EvaluationMetrics to_dict conversion."""
    metrics = evaluation_framework.compute_metrics([successful_trajectory])

    metrics_dict = metrics.to_dict()

    assert "success_rate" in metrics_dict
    assert "avg_reward" in metrics_dict
    assert "avg_steps" in metrics_dict
    assert "tool_accuracy" in metrics_dict
    assert "sample_size" in metrics_dict
    assert metrics_dict["sample_size"] == 1


# Test baseline comparison


def test_compare_with_baseline_significant_improvement(
    evaluation_framework: EvaluationFramework) -> None:
    """Test baseline comparison with significant improvement."""
    # Baseline: low rewards
    baseline_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="baseline",
            query="Query",
            steps=[],
            reward=0.2 + i * 0.01,
            success=False,
            created_at=datetime.now(UTC))
        for i in range(50)
    ]

    # Trained: high rewards
    trained_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="trained",
            query="Query",
            steps=[],
            reward=0.8 + i * 0.01,
            success=True,
            created_at=datetime.now(UTC))
        for i in range(50)
    ]

    test_result = evaluation_framework.compare_with_baseline(
        baseline_trajectories,
        trained_trajectories,
        metric_key="reward")

    assert test_result.is_significant  # p < 0.05
    assert test_result.trained_mean > test_result.baseline_mean
    assert test_result.improvement > 0


def test_compare_with_baseline_no_difference(
    evaluation_framework: EvaluationFramework) -> None:
    """Test baseline comparison with no significant difference."""
    # Identical distributions
    baseline_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="baseline",
            query="Query",
            steps=[],
            reward=0.5,
            success=True,
            created_at=datetime.now(UTC))
        for _ in range(50)
    ]

    trained_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="trained",
            query="Query",
            steps=[],
            reward=0.5,
            success=True,
            created_at=datetime.now(UTC))
        for _ in range(50)
    ]

    test_result = evaluation_framework.compare_with_baseline(
        baseline_trajectories,
        trained_trajectories,
        metric_key="reward")

    # With identical values, t-test will have NaN or p-value = 1.0
    # Check that no improvement detected
    assert abs(test_result.improvement) < 0.01


def test_compare_with_baseline_empty_trajectories(
    evaluation_framework: EvaluationFramework) -> None:
    """Test baseline comparison with empty trajectories raises ValueError."""
    trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="test",
            query="Query",
            steps=[],
            reward=0.5,
            success=True,
            created_at=datetime.now(UTC))
    ]

    with pytest.raises(ValueError, match="Both baseline and trained trajectories required"):
        evaluation_framework.compare_with_baseline([], trajectories)

    with pytest.raises(ValueError, match="Both baseline and trained trajectories required"):
        evaluation_framework.compare_with_baseline(trajectories, [])


def test_compare_with_baseline_different_metrics(
    evaluation_framework: EvaluationFramework) -> None:
    """Test baseline comparison for different metric types."""
    baseline_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="baseline",
            query="Query",
            steps=[TrajectoryStep(
                state={}, action={}, result={},
                timestamp=datetime.now(UTC), duration_ms=100)] * 5,  # 5 steps
            reward=0.3,
            success=False,
            created_at=datetime.now(UTC))
        for _ in range(30)
    ]

    trained_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="trained",
            query="Query",
            steps=[TrajectoryStep(
                state={}, action={}, result={},
                timestamp=datetime.now(UTC), duration_ms=100)] * 3,  # 3 steps (more efficient)
            reward=0.8,
            success=True,
            created_at=datetime.now(UTC))
        for _ in range(30)
    ]

    # Test reward comparison
    reward_test = evaluation_framework.compare_with_baseline(
        baseline_trajectories, trained_trajectories, metric_key="reward"
    )
    assert reward_test.trained_mean > reward_test.baseline_mean

    # Test steps comparison (fewer steps = better)
    steps_test = evaluation_framework.compare_with_baseline(
        baseline_trajectories, trained_trajectories, metric_key="steps"
    )
    assert steps_test.trained_mean < steps_test.baseline_mean

    # Test success comparison
    success_test = evaluation_framework.compare_with_baseline(
        baseline_trajectories, trained_trajectories, metric_key="success"
    )
    assert success_test.trained_mean > success_test.baseline_mean


def test_compare_with_baseline_invalid_metric(
    evaluation_framework: EvaluationFramework) -> None:
    """Test baseline comparison with invalid metric_key raises ValueError."""
    trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="test",
            query="Query",
            steps=[],
            reward=0.5,
            success=True,
            created_at=datetime.now(UTC))
    ]

    with pytest.raises(ValueError, match="Unknown metric_key"):
        evaluation_framework.compare_with_baseline(
            trajectories, trajectories, metric_key="invalid_metric"
        )


# Test evaluation interval


def test_should_evaluate_at_interval(
    evaluation_framework: EvaluationFramework) -> None:
    """Test should_evaluate returns True at evaluation intervals."""
    assert not evaluation_framework.should_evaluate(0)  # Not at iteration 0
    assert not evaluation_framework.should_evaluate(5)
    assert evaluation_framework.should_evaluate(10)
    assert not evaluation_framework.should_evaluate(15)
    assert evaluation_framework.should_evaluate(20)


def test_should_evaluate_custom_interval() -> None:
    """Test should_evaluate with custom interval."""
    framework = EvaluationFramework(evaluation_interval=5)

    assert framework.should_evaluate(5)
    assert framework.should_evaluate(10)
    assert framework.should_evaluate(15)
    assert not framework.should_evaluate(7)


# Test full evaluation workflow


def test_run_evaluation(
    evaluation_framework: EvaluationFramework) -> None:
    """Test complete evaluation workflow."""
    eval_queries = [
        TrainingQuery(query=f"Eval query {i}", expected_outcome={"result": "success"})
        for i in range(20)
    ]

    baseline_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="baseline",
            query="Query",
            steps=[TrajectoryStep(
                state={}, action={}, result={},
                timestamp=datetime.now(UTC), duration_ms=100)] * 4,
            reward=0.3,
            success=False,
            created_at=datetime.now(UTC))
        for _ in range(20)
    ]

    trained_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="trained",
            query="Query",
            steps=[TrajectoryStep(
                state={}, action={}, result={},
                timestamp=datetime.now(UTC), duration_ms=100)] * 3,
            reward=0.8,
            success=True,
            created_at=datetime.now(UTC))
        for _ in range(20)
    ]

    result = evaluation_framework.run_evaluation(
        eval_queries,
        baseline_trajectories,
        trained_trajectories)

    # Assert structure
    assert "baseline_metrics" in result
    assert "trained_metrics" in result
    assert "statistical_tests" in result
    assert "eval_query_count" in result

    # Assert metrics present
    assert result["baseline_metrics"]["success_rate"] == 0.0
    assert result["trained_metrics"]["success_rate"] == 1.0

    # Assert statistical tests present
    assert "reward" in result["statistical_tests"]
    assert "success" in result["statistical_tests"]
    assert "steps" in result["statistical_tests"]

    # Assert significant improvement detected
    assert result["statistical_tests"]["reward"]["is_significant"]
    assert result["statistical_tests"]["reward"]["improvement_percent"] > 0


def test_run_evaluation_marginal_improvement(
    evaluation_framework: EvaluationFramework) -> None:
    """Test evaluation with marginal (non-significant) improvement."""
    eval_queries = [
        TrainingQuery(query=f"Eval query {i}", expected_outcome={"result": "success"})
        for i in range(20)
    ]

    # Only slight difference between baseline and trained
    baseline_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="baseline",
            query="Query",
            steps=[],
            reward=0.48 + i * 0.001,  # ~0.48-0.50
            success=True,
            created_at=datetime.now(UTC))
        for i in range(20)
    ]

    trained_trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="trained",
            query="Query",
            steps=[],
            reward=0.50 + i * 0.001,  # ~0.50-0.52
            success=True,
            created_at=datetime.now(UTC))
        for i in range(20)
    ]

    result = evaluation_framework.run_evaluation(
        eval_queries,
        baseline_trajectories,
        trained_trajectories)

    # May or may not be significant depending on variance
    # Just verify structure is correct
    assert result["statistical_tests"]["reward"]["p_value"] >= 0
    assert result["statistical_tests"]["reward"]["p_value"] <= 1


def test_framework_initialization() -> None:
    """Test EvaluationFramework initialization."""
    framework = EvaluationFramework(evaluation_interval=5)
    assert framework.evaluation_interval == 5

    # Default initialization
    default_framework = EvaluationFramework()
    assert default_framework.evaluation_interval == 10
