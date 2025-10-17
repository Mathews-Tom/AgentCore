"""
Unit tests for training job scheduler.

Tests scheduler logic without requiring Redis connection.
"""

from __future__ import annotations

import pytest

from agentcore.training.scheduler import JobPriority


def test_job_priority_enum():
    """Test JobPriority enum values."""
    assert JobPriority.P0.value == 0
    assert JobPriority.P1.value == 1
    assert JobPriority.P2.value == 2


def test_job_priority_ordering():
    """Test that priority values are correctly ordered (P0 < P1 < P2)."""
    assert JobPriority.P0.value < JobPriority.P1.value
    assert JobPriority.P1.value < JobPriority.P2.value


def test_job_priority_names():
    """Test JobPriority enum names."""
    assert JobPriority.P0.name == "P0"
    assert JobPriority.P1.name == "P1"
    assert JobPriority.P2.name == "P2"


def test_scheduler_queue_names():
    """Test queue name generation."""
    from agentcore.training.scheduler import TrainingJobScheduler

    assert TrainingJobScheduler.QUEUE_NAMES[JobPriority.P0] == "training:queue:p0"
    assert TrainingJobScheduler.QUEUE_NAMES[JobPriority.P1] == "training:queue:p1"
    assert TrainingJobScheduler.QUEUE_NAMES[JobPriority.P2] == "training:queue:p2"


def test_worker_health_prefix():
    """Test worker health key prefix."""
    from agentcore.training.scheduler import TrainingJobScheduler

    assert TrainingJobScheduler.WORKER_HEALTH_PREFIX == "training:worker:health:"


def test_scheduler_initialization():
    """Test scheduler initialization."""
    from agentcore.training.job_manager import TrainingJobManager
    from agentcore.training.scheduler import TrainingJobScheduler

    job_manager = TrainingJobManager()
    scheduler = TrainingJobScheduler(job_manager)

    assert scheduler.job_manager is job_manager
    assert scheduler.redis_url == "redis://localhost:6379/0"  # Default from config
    assert scheduler.get_worker_count() == 0
    assert not scheduler._running


def test_scheduler_initialization_custom_redis_url():
    """Test scheduler initialization with custom Redis URL."""
    from agentcore.training.job_manager import TrainingJobManager
    from agentcore.training.scheduler import TrainingJobScheduler

    job_manager = TrainingJobManager()
    custom_url = "redis://custom-host:6380/1"
    scheduler = TrainingJobScheduler(job_manager, redis_url=custom_url)

    assert scheduler.redis_url == custom_url
