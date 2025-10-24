"""Tests for cost tracking functionality."""

from datetime import datetime, timedelta

import pytest

from agentcore.integration.portkey.cost_models import (
    BudgetAlertSeverity,
    BudgetConfig,
    BudgetThreshold,
    CostMetrics,
    CostPeriod,
)
from agentcore.integration.portkey.cost_tracker import CostTracker
from agentcore.integration.portkey.exceptions import PortkeyBudgetExceededError


@pytest.fixture
def cost_tracker() -> CostTracker:
    """Create a cost tracker for testing."""
    return CostTracker(history_retention_days=30, alert_debounce_seconds=60)


@pytest.fixture
def sample_cost_metrics() -> CostMetrics:
    """Create sample cost metrics."""
    return CostMetrics(
        total_cost=0.05,
        input_cost=0.02,
        output_cost=0.03,
        input_tokens=1000,
        output_tokens=1500,
        provider_id="openai",
        model="gpt-4",
        timestamp=datetime.now(),
        latency_ms=1200,
        request_id="req-123",
        tenant_id="tenant-1",
        workflow_id="workflow-1",
        agent_id="agent-1",
        tags={"env": "production"},
    )


def test_record_cost(cost_tracker: CostTracker, sample_cost_metrics: CostMetrics) -> None:
    """Test recording cost metrics."""
    cost_tracker.record_cost(sample_cost_metrics)

    history = cost_tracker.get_cost_history(limit=1)
    assert len(history) == 1
    assert history[0].total_cost == 0.05
    assert history[0].provider_id == "openai"
    assert history[0].tenant_id == "tenant-1"


def test_calculate_request_cost(cost_tracker: CostTracker) -> None:
    """Test cost calculation for a request."""
    metrics = cost_tracker.calculate_request_cost(
        provider_id="anthropic",
        input_tokens=2000,
        output_tokens=1000,
        input_token_price=0.01,  # Per 1K tokens
        output_token_price=0.03,  # Per 1K tokens
        model="claude-3-opus",
        latency_ms=800,
        tenant_id="tenant-2",
    )

    # Expected: (2000/1000 * 0.01) + (1000/1000 * 0.03) = 0.02 + 0.03 = 0.05
    assert metrics.total_cost == 0.05
    assert metrics.input_cost == 0.02
    assert metrics.output_cost == 0.03
    assert metrics.provider_id == "anthropic"
    assert metrics.model == "claude-3-opus"


def test_get_summary_empty(cost_tracker: CostTracker) -> None:
    """Test getting summary with no data."""
    summary = cost_tracker.get_summary()

    assert summary.total_cost == 0.0
    assert summary.total_requests == 0
    assert summary.total_input_tokens == 0
    assert summary.total_output_tokens == 0


def test_get_summary_with_data(
    cost_tracker: CostTracker,
    sample_cost_metrics: CostMetrics,
) -> None:
    """Test getting cost summary with data."""
    # Record multiple requests
    for i in range(5):
        metrics = CostMetrics(
            total_cost=0.05 * (i + 1),
            input_cost=0.02 * (i + 1),
            output_cost=0.03 * (i + 1),
            input_tokens=1000 * (i + 1),
            output_tokens=1500 * (i + 1),
            provider_id="openai" if i % 2 == 0 else "anthropic",
            model="gpt-4",
            timestamp=datetime.now(),
            tenant_id="tenant-1",
        )
        cost_tracker.record_cost(metrics)

    summary = cost_tracker.get_summary()

    assert summary.total_requests == 5
    assert summary.total_cost == 0.75  # 0.05 + 0.10 + 0.15 + 0.20 + 0.25
    assert summary.total_input_tokens == 15000
    assert summary.total_output_tokens == 22500
    assert len(summary.provider_breakdown) == 2


def test_set_and_get_budget(cost_tracker: CostTracker) -> None:
    """Test budget configuration."""
    now = datetime.now()
    budget = BudgetConfig(
        tenant_id="tenant-1",
        limit_amount=100.0,
        period=CostPeriod.MONTHLY,
        period_start=now,
        period_end=now + timedelta(days=30),
        hard_limit=True,
    )

    cost_tracker.set_budget(budget)

    retrieved = cost_tracker.get_budget("tenant-1")
    assert retrieved is not None
    assert retrieved.limit_amount == 100.0
    assert retrieved.hard_limit is True


def test_budget_enforcement_soft_limit(cost_tracker: CostTracker) -> None:
    """Test budget enforcement without hard limit."""
    now = datetime.now()
    budget = BudgetConfig(
        tenant_id="tenant-1",
        limit_amount=0.10,
        period=CostPeriod.MONTHLY,
        period_start=now,
        period_end=now + timedelta(days=30),
        hard_limit=False,
    )
    cost_tracker.set_budget(budget)

    # Record cost exceeding budget (should not raise error)
    metrics = CostMetrics(
        total_cost=0.15,
        input_cost=0.06,
        output_cost=0.09,
        input_tokens=3000,
        output_tokens=3000,
        provider_id="openai",
        model="gpt-4",
        timestamp=now,
        tenant_id="tenant-1",
    )

    # Should not raise exception (soft limit)
    cost_tracker.record_cost(metrics)

    # Budget should be updated
    updated_budget = cost_tracker.get_budget("tenant-1")
    assert updated_budget is not None
    assert updated_budget.current_spend == 0.15


def test_budget_enforcement_hard_limit(cost_tracker: CostTracker) -> None:
    """Test budget enforcement with hard limit."""
    now = datetime.now()
    budget = BudgetConfig(
        tenant_id="tenant-1",
        limit_amount=0.10,
        period=CostPeriod.MONTHLY,
        period_start=now,
        period_end=now + timedelta(days=30),
        hard_limit=True,
    )
    cost_tracker.set_budget(budget)

    # First request within budget
    metrics1 = CostMetrics(
        total_cost=0.06,
        input_cost=0.03,
        output_cost=0.03,
        input_tokens=1500,
        output_tokens=1500,
        provider_id="openai",
        model="gpt-4",
        timestamp=now,
        tenant_id="tenant-1",
    )
    cost_tracker.record_cost(metrics1)

    # Second request exceeding budget (should raise error)
    metrics2 = CostMetrics(
        total_cost=0.06,
        input_cost=0.03,
        output_cost=0.03,
        input_tokens=1500,
        output_tokens=1500,
        provider_id="openai",
        model="gpt-4",
        timestamp=now,
        tenant_id="tenant-1",
    )

    with pytest.raises(PortkeyBudgetExceededError) as exc_info:
        cost_tracker.record_cost(metrics2)

    assert "Budget exceeded" in str(exc_info.value)


def test_budget_threshold_alerts(cost_tracker: CostTracker) -> None:
    """Test budget threshold alerts."""
    now = datetime.now()
    threshold = BudgetThreshold(
        threshold_percent=80.0,
        severity=BudgetAlertSeverity.WARNING,
        notify_emails=["admin@example.com"],
    )

    budget = BudgetConfig(
        tenant_id="tenant-1",
        limit_amount=1.00,
        period=CostPeriod.MONTHLY,
        period_start=now,
        period_end=now + timedelta(days=30),
        thresholds=[threshold],
    )
    cost_tracker.set_budget(budget)

    # Record cost reaching 85% of budget
    metrics = CostMetrics(
        total_cost=0.85,
        input_cost=0.40,
        output_cost=0.45,
        input_tokens=20000,
        output_tokens=15000,
        provider_id="openai",
        model="gpt-4",
        timestamp=now,
        tenant_id="tenant-1",
    )
    cost_tracker.record_cost(metrics)

    # Check for alerts
    alerts = cost_tracker.get_alerts(tenant_id="tenant-1", acknowledged=False)
    assert len(alerts) > 0
    assert alerts[0].threshold.threshold_percent == 80.0
    assert alerts[0].threshold.severity == BudgetAlertSeverity.WARNING


def test_check_budget_available(cost_tracker: CostTracker) -> None:
    """Test checking budget availability."""
    now = datetime.now()
    budget = BudgetConfig(
        tenant_id="tenant-1",
        limit_amount=1.00,
        period=CostPeriod.MONTHLY,
        period_start=now,
        period_end=now + timedelta(days=30),
        current_spend=0.70,
        hard_limit=True,
    )
    cost_tracker.set_budget(budget)

    # Check if 0.20 is available (should be ok)
    available, reason = cost_tracker.check_budget_available("tenant-1", 0.20)
    assert available is True
    assert reason is None

    # Check if 0.40 is available (would exceed budget)
    available, reason = cost_tracker.check_budget_available("tenant-1", 0.40)
    assert available is False
    assert reason is not None
    assert "would be exceeded" in reason


def test_acknowledge_alert(cost_tracker: CostTracker) -> None:
    """Test acknowledging budget alerts."""
    now = datetime.now()
    threshold = BudgetThreshold(
        threshold_percent=90.0,
        severity=BudgetAlertSeverity.CRITICAL,
    )

    budget = BudgetConfig(
        tenant_id="tenant-1",
        limit_amount=1.00,
        period=CostPeriod.MONTHLY,
        period_start=now,
        period_end=now + timedelta(days=30),
        thresholds=[threshold],
    )
    cost_tracker.set_budget(budget)

    # Trigger alert
    metrics = CostMetrics(
        total_cost=0.95,
        input_cost=0.45,
        output_cost=0.50,
        input_tokens=22500,
        output_tokens=16667,
        provider_id="openai",
        model="gpt-4",
        timestamp=now,
        tenant_id="tenant-1",
    )
    cost_tracker.record_cost(metrics)

    # Get unacknowledged alerts
    alerts = cost_tracker.get_alerts(tenant_id="tenant-1", acknowledged=False)
    assert len(alerts) > 0

    # Acknowledge alert
    alert_id = alerts[0].alert_id
    result = cost_tracker.acknowledge_alert(alert_id)
    assert result is True

    # Check acknowledged status
    acknowledged_alerts = cost_tracker.get_alerts(tenant_id="tenant-1", acknowledged=True)
    assert len(acknowledged_alerts) > 0


def test_get_cost_history_with_filters(
    cost_tracker: CostTracker,
    sample_cost_metrics: CostMetrics,
) -> None:
    """Test getting cost history with filters."""
    now = datetime.now()

    # Record multiple requests with different attributes
    for i in range(10):
        metrics = CostMetrics(
            total_cost=0.01 * (i + 1),
            input_cost=0.005 * (i + 1),
            output_cost=0.005 * (i + 1),
            input_tokens=500 * (i + 1),
            output_tokens=500 * (i + 1),
            provider_id="openai" if i < 5 else "anthropic",
            model="gpt-4",
            timestamp=now - timedelta(hours=i),
            tenant_id="tenant-1" if i < 7 else "tenant-2",
        )
        cost_tracker.record_cost(metrics)

    # Filter by provider
    openai_history = cost_tracker.get_cost_history(provider_id="openai")
    assert len(openai_history) == 5

    # Filter by tenant
    tenant1_history = cost_tracker.get_cost_history(tenant_id="tenant-1")
    assert len(tenant1_history) == 7

    # Filter by time range
    recent_history = cost_tracker.get_cost_history(
        start_time=now - timedelta(hours=5),
        end_time=now,
    )
    assert len(recent_history) <= 6

    # Filter with limit
    limited_history = cost_tracker.get_cost_history(limit=3)
    assert len(limited_history) == 3


def test_get_stats(cost_tracker: CostTracker) -> None:
    """Test getting cost tracker statistics."""
    now = datetime.now()

    # Record some requests
    for i in range(5):
        metrics = CostMetrics(
            total_cost=0.10,
            input_cost=0.05,
            output_cost=0.05,
            input_tokens=2500,
            output_tokens=1667,
            provider_id="openai",
            model="gpt-4",
            timestamp=now - timedelta(hours=i),
            tenant_id="tenant-1",
        )
        cost_tracker.record_cost(metrics)

    stats = cost_tracker.get_stats()

    assert stats["total_records"] == 5
    assert stats["last_24h_requests"] == 5
    assert stats["last_24h_cost"] == 0.50
    assert stats["last_7d_requests"] == 5
    assert stats["last_7d_cost"] == 0.50
