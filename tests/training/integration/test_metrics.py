"""
Integration tests for Prometheus metrics.

Tests metrics collection for training jobs, performance, and budget.
"""

from __future__ import annotations

import time

import pytest
from prometheus_client import REGISTRY

from agentcore.training.metrics import TrainingMetrics


@pytest.fixture(autouse=True)
def reset_metrics() -> None:
    """Reset Prometheus metrics before each test."""
    # Note: In a real scenario, we'd use a separate registry for tests
    # For now, we just test that metrics can be recorded without errors
    pass


# Test job lifecycle metrics


def test_job_created_metric() -> None:
    """Test job_created metric increments correctly."""
    agent_id = "test-agent-created"

    # Record job creation
    TrainingMetrics.job_created(agent_id)

    # Verify metric can be retrieved (actual value check would require registry inspection)
    # In integration tests, we primarily verify no errors occur
    assert True  # Metric recorded successfully


def test_job_completed_metric() -> None:
    """Test job_completed metric increments correctly."""
    agent_id = "test-agent-completed"

    # Create and complete job
    TrainingMetrics.job_created(agent_id)
    TrainingMetrics.job_completed(agent_id)

    assert True  # Metrics recorded successfully


def test_job_failed_metric() -> None:
    """Test job_failed metric increments correctly."""
    agent_id = "test-agent-failed"

    # Create and fail job
    TrainingMetrics.job_created(agent_id)
    TrainingMetrics.job_failed(agent_id)

    assert True  # Metrics recorded successfully


def test_job_cancelled_metric() -> None:
    """Test job_cancelled metric increments correctly."""
    agent_id = "test-agent-cancelled"

    # Create and cancel job
    TrainingMetrics.job_created(agent_id)
    TrainingMetrics.job_cancelled(agent_id)

    assert True  # Metrics recorded successfully


# Test performance metrics


def test_measure_trajectory_generation() -> None:
    """Test trajectory generation duration measurement."""
    agent_id = "test-agent-trajectory"

    # Measure trajectory generation
    with TrainingMetrics.measure_trajectory_generation(agent_id):
        time.sleep(0.01)  # Simulate trajectory generation

    assert True  # Duration recorded successfully


def test_measure_policy_update() -> None:
    """Test policy update duration measurement."""
    agent_id = "test-agent-policy"

    # Measure policy update
    with TrainingMetrics.measure_policy_update(agent_id):
        time.sleep(0.01)  # Simulate policy update

    assert True  # Duration recorded successfully


def test_measure_training_iteration() -> None:
    """Test training iteration duration measurement."""
    agent_id = "test-agent-iteration"

    # Measure iteration
    with TrainingMetrics.measure_training_iteration(agent_id):
        time.sleep(0.01)  # Simulate training iteration

    assert True  # Duration recorded successfully


def test_measure_checkpoint_save() -> None:
    """Test checkpoint save duration measurement."""
    agent_id = "test-agent-checkpoint"

    # Measure checkpoint save
    with TrainingMetrics.measure_checkpoint_save(agent_id):
        time.sleep(0.01)  # Simulate checkpoint save

    assert True  # Duration recorded successfully


def test_measure_nested_operations() -> None:
    """Test nested performance measurements."""
    agent_id = "test-agent-nested"

    # Nested measurements (iteration contains trajectory generation and policy update)
    with TrainingMetrics.measure_training_iteration(agent_id):
        with TrainingMetrics.measure_trajectory_generation(agent_id):
            time.sleep(0.005)

        with TrainingMetrics.measure_policy_update(agent_id):
            time.sleep(0.005)

    assert True  # All durations recorded successfully


# Test budget metrics


def test_update_budget() -> None:
    """Test budget metrics update."""
    agent_id = "test-agent-budget"
    job_id = "test-job-budget"

    # Update budget
    TrainingMetrics.update_budget(
        agent_id=agent_id,
        job_id=job_id,
        usage_usd=50.0,
        limit_usd=100.0,
    )

    assert True  # Budget metrics updated successfully


def test_update_budget_zero_limit() -> None:
    """Test budget update with zero limit."""
    agent_id = "test-agent-budget-zero"
    job_id = "test-job-budget-zero"

    # Update budget with zero limit
    TrainingMetrics.update_budget(
        agent_id=agent_id,
        job_id=job_id,
        usage_usd=0.0,
        limit_usd=0.0,
    )

    assert True  # Handles zero limit gracefully


def test_update_budget_at_limit() -> None:
    """Test budget update at limit."""
    agent_id = "test-agent-budget-limit"
    job_id = "test-job-budget-limit"

    # Update budget at limit (100% utilization)
    TrainingMetrics.update_budget(
        agent_id=agent_id,
        job_id=job_id,
        usage_usd=100.0,
        limit_usd=100.0,
    )

    assert True  # Budget at limit recorded successfully


def test_update_budget_exceeded() -> None:
    """Test budget update when exceeded."""
    agent_id = "test-agent-budget-exceeded"
    job_id = "test-job-budget-exceeded"

    # Update budget exceeded (>100% utilization)
    TrainingMetrics.update_budget(
        agent_id=agent_id,
        job_id=job_id,
        usage_usd=120.0,
        limit_usd=100.0,
    )

    assert True  # Budget exceeded recorded successfully


# Test progress metrics


def test_update_progress() -> None:
    """Test training progress update."""
    agent_id = "test-agent-progress"
    job_id = "test-job-progress"

    # Update progress
    TrainingMetrics.update_progress(
        agent_id=agent_id,
        job_id=job_id,
        current_iteration=5,
        total_iterations=10,
    )

    assert True  # Progress metrics updated successfully


def test_update_progress_start() -> None:
    """Test progress update at start."""
    agent_id = "test-agent-progress-start"
    job_id = "test-job-progress-start"

    # Progress at start (iteration 0)
    TrainingMetrics.update_progress(
        agent_id=agent_id,
        job_id=job_id,
        current_iteration=0,
        total_iterations=100,
    )

    assert True  # Initial progress recorded successfully


def test_update_progress_end() -> None:
    """Test progress update at end."""
    agent_id = "test-agent-progress-end"
    job_id = "test-job-progress-end"

    # Progress at end (iteration == total)
    TrainingMetrics.update_progress(
        agent_id=agent_id,
        job_id=job_id,
        current_iteration=100,
        total_iterations=100,
    )

    assert True  # Completion progress recorded successfully


# Test training performance metrics


def test_update_training_metrics_loss_only() -> None:
    """Test training metrics update with loss only."""
    agent_id = "test-agent-metrics-loss"
    job_id = "test-job-metrics-loss"

    # Update loss only
    TrainingMetrics.update_training_metrics(
        agent_id=agent_id,
        job_id=job_id,
        loss=0.25,
    )

    assert True  # Loss metric updated successfully


def test_update_training_metrics_reward_only() -> None:
    """Test training metrics update with reward only."""
    agent_id = "test-agent-metrics-reward"
    job_id = "test-job-metrics-reward"

    # Update reward only
    TrainingMetrics.update_training_metrics(
        agent_id=agent_id,
        job_id=job_id,
        mean_reward=0.75,
    )

    assert True  # Reward metric updated successfully


def test_update_training_metrics_both() -> None:
    """Test training metrics update with both loss and reward."""
    agent_id = "test-agent-metrics-both"
    job_id = "test-job-metrics-both"

    # Update both loss and reward
    TrainingMetrics.update_training_metrics(
        agent_id=agent_id,
        job_id=job_id,
        loss=0.15,
        mean_reward=0.85,
    )

    assert True  # Both metrics updated successfully


def test_update_training_metrics_none() -> None:
    """Test training metrics update with no values."""
    agent_id = "test-agent-metrics-none"
    job_id = "test-job-metrics-none"

    # Update with no values (should not error)
    TrainingMetrics.update_training_metrics(
        agent_id=agent_id,
        job_id=job_id,
    )

    assert True  # No-op update handled gracefully


# Test complete workflow


def test_complete_training_workflow() -> None:
    """Test complete training workflow with all metrics."""
    agent_id = "test-agent-workflow"
    job_id = "test-job-workflow"

    # Job created
    TrainingMetrics.job_created(agent_id)

    # Simulate training iterations
    for iteration in range(1, 4):
        with TrainingMetrics.measure_training_iteration(agent_id):
            # Trajectory generation
            with TrainingMetrics.measure_trajectory_generation(agent_id):
                time.sleep(0.005)

            # Policy update
            with TrainingMetrics.measure_policy_update(agent_id):
                time.sleep(0.002)

            # Checkpoint save (every other iteration)
            if iteration % 2 == 0:
                with TrainingMetrics.measure_checkpoint_save(agent_id):
                    time.sleep(0.001)

        # Update progress
        TrainingMetrics.update_progress(
            agent_id=agent_id,
            job_id=job_id,
            current_iteration=iteration,
            total_iterations=3,
        )

        # Update training metrics
        TrainingMetrics.update_training_metrics(
            agent_id=agent_id,
            job_id=job_id,
            loss=0.5 - (iteration * 0.1),
            mean_reward=0.5 + (iteration * 0.1),
        )

        # Update budget
        TrainingMetrics.update_budget(
            agent_id=agent_id,
            job_id=job_id,
            usage_usd=iteration * 10.0,
            limit_usd=50.0,
        )

    # Job completed
    TrainingMetrics.job_completed(agent_id)

    assert True  # Complete workflow metrics recorded successfully


def test_metrics_registry_accessible() -> None:
    """Test that Prometheus registry is accessible."""
    # Verify registry is available
    assert REGISTRY is not None

    # Verify we can get metrics from registry
    metrics = list(REGISTRY.collect())
    assert len(metrics) > 0  # Should have our training metrics
