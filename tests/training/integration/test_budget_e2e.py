"""
End-to-end integration tests for budget enforcement (FLOW-011).

Tests the complete budget enforcement workflow including:
- Budget tracking during training
- Budget limit enforcement
- Job cancellation on budget exceed
- Cost estimation and warnings
- Budget alerts at thresholds

Updated to match actual BudgetEnforcer implementation (synchronous).
"""

from __future__ import annotations

from uuid import uuid4
from decimal import Decimal

import pytest

from agentcore.training.utils.budget import (
    BudgetEnforcer,
    BudgetStatus)
from agentcore.training.models import (
    TrainingJob,
    GRPOConfig)


@pytest.fixture
def budget_enforcer():
    """Create budget enforcer instance."""
    return BudgetEnforcer(
        max_budget_usd=Decimal("10.00"),
        warning_threshold_75=0.75,
        warning_threshold_90=0.90)


@pytest.fixture
def training_job_with_budget() -> TrainingJob:
    """Create training job with budget configuration."""
    # Create minimal training data (100 items required)
    from agentcore.training.models import TrainingQuery

    training_data = [
        TrainingQuery(
            query=f"Test query {i}",
            expected_outcome={"result": "success"})
        for i in range(100)
    ]

    return TrainingJob(
        job_id=uuid4(),
        agent_id="test_agent",
        config=GRPOConfig(
            n_iterations=100,
            batch_size=16,
            n_trajectories_per_query=8,
            max_budget_usd=Decimal("10.00")),
        training_data=training_data,
        total_iterations=100,
        budget_usd=Decimal("10.00"),
        status="queued")


class TestBudgetEnforcement:
    """Integration tests for budget enforcement."""

    def test_budget_tracking_basic(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test basic budget tracking functionality."""
        # Track some costs
        budget_enforcer.add_cost(Decimal("1.50"))
        budget_enforcer.add_cost(Decimal("2.25"))

        # Verify total
        assert budget_enforcer.current_cost_usd == Decimal("3.75")

        # Get remaining budget
        remaining = budget_enforcer.get_remaining_budget()
        assert remaining == Decimal("6.25")

    def test_budget_status_ok(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test budget status returns OK when under 75% threshold."""
        # Add cost under 75%
        budget_enforcer.add_cost(Decimal("5.00"))  # 50% of $10

        status = budget_enforcer.get_status()
        assert status["status"] == BudgetStatus.OK.value
        assert status["is_exceeded"] is False

    def test_budget_status_warning_75(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test budget status returns WARNING_75 at 75% threshold."""
        # Add cost at 75%
        budget_enforcer.add_cost(Decimal("7.50"))  # 75% of $10

        status = budget_enforcer.get_status()
        assert status["status"] == BudgetStatus.WARNING_75.value
        assert status["is_exceeded"] is False

    def test_budget_status_warning_90(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test budget status returns WARNING_90 at 90% threshold."""
        # Add cost at 90%
        budget_enforcer.add_cost(Decimal("9.00"))  # 90% of $10

        status = budget_enforcer.get_status()
        assert status["status"] == BudgetStatus.WARNING_90.value
        assert status["is_exceeded"] is False

    def test_budget_status_exceeded(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test budget status returns EXCEEDED when over limit."""
        # Exceed budget
        budget_enforcer.add_cost(Decimal("10.50"))  # 105% of $10

        status = budget_enforcer.get_status()
        assert status["status"] == BudgetStatus.EXCEEDED.value
        assert status["is_exceeded"] is True

    def test_check_budget_available_ok(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test checking if budget is available for operation."""
        # Current: $0, Available: $10
        available, status = budget_enforcer.check_budget_available(Decimal("5.00"))

        assert available is True
        assert status == BudgetStatus.OK

    def test_check_budget_available_at_warning_75(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test budget check at 75% warning threshold."""
        # Use up 70%
        budget_enforcer.add_cost(Decimal("7.00"))

        # Check if we can add more
        available, status = budget_enforcer.check_budget_available(Decimal("0.50"))

        assert available is True
        assert status == BudgetStatus.WARNING_75

    def test_check_budget_available_exceeded(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test budget check when operation would exceed limit."""
        # Use up $9
        budget_enforcer.add_cost(Decimal("9.00"))

        # Check if we can add $2 (would exceed $10 limit)
        available, status = budget_enforcer.check_budget_available(Decimal("2.00"))

        assert available is False
        assert status == BudgetStatus.EXCEEDED

    def test_get_utilization_percentage(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test getting budget utilization percentage."""
        # Add 30% of budget
        budget_enforcer.add_cost(Decimal("3.00"))

        percentage = budget_enforcer.get_utilization_percentage()
        assert percentage == 30.0

    def test_reset_budget(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test resetting budget to zero."""
        # Add some costs
        budget_enforcer.add_cost(Decimal("5.00"))
        assert budget_enforcer.current_cost_usd == Decimal("5.00")

        # Reset
        budget_enforcer.reset()

        # Verify reset
        assert budget_enforcer.current_cost_usd == Decimal("0.00")
        status = budget_enforcer.get_status()
        assert status["status"] == BudgetStatus.OK.value
        assert status["is_exceeded"] is False

    def test_multiple_small_costs_accumulate(
        self,
        budget_enforcer: BudgetEnforcer) -> None:
        """Test that multiple small costs accumulate correctly."""
        # Add many small costs
        for _ in range(10):
            budget_enforcer.add_cost(Decimal("0.50"))

        # Total should be $5.00
        assert budget_enforcer.current_cost_usd == Decimal("5.00")
        assert budget_enforcer.get_remaining_budget() == Decimal("5.00")

    def test_budget_enforcer_with_training_job(
        self,
        training_job_with_budget: TrainingJob) -> None:
        """Test budget enforcer integration with training job config."""
        # Create enforcer from job config
        enforcer = BudgetEnforcer(
            max_budget_usd=training_job_with_budget.config.max_budget_usd)

        # Simulate training costs
        enforcer.add_cost(Decimal("2.50"))  # First iteration
        enforcer.add_cost(Decimal("3.00"))  # Second iteration

        # Check status
        status = enforcer.get_status()
        assert status["status"] == BudgetStatus.OK.value
        assert status["is_exceeded"] is False

        # Verify we haven't exceeded job budget
        assert enforcer.current_cost_usd <= training_job_with_budget.config.max_budget_usd
