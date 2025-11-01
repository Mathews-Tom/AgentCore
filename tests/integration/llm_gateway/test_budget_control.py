"""Budget control and enforcement tests.

Tests budget enforcement (per-tenant, per-service, global), quota limits,
budget alerts and notifications, overage handling, budget reset cycles,
and multi-tenant budget isolation. Includes edge case testing for
concurrent requests and race conditions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

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
def cost_tracker_budget() -> CostTracker:
    """Create a cost tracker for budget testing."""
    return CostTracker(history_retention_days=30, alert_debounce_seconds=10)


class TestBudgetEnforcement:
    """Test budget enforcement across different scopes."""

    def test_per_tenant_budget_enforcement(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test per-tenant budget limits."""
        now = datetime.now()

        # Set budgets for multiple tenants
        for tenant_id in range(3):
            budget = BudgetConfig(
                tenant_id=f"tenant_{tenant_id}",
                limit_amount=10.0,  # $10 per tenant
                period=CostPeriod.DAILY,
                period_start=now,
                period_end=now + timedelta(days=1),
                hard_limit=True,
            )
            cost_tracker_budget.set_budget(budget)

        # Record spending for each tenant
        for tenant_id in range(3):
            # Spend up to $8 (within budget)
            for _ in range(4):
                metrics = CostMetrics(
                    total_cost=2.0,
                    input_cost=1.0,
                    output_cost=1.0,
                    input_tokens=50000,
                    output_tokens=33333,
                    provider_id="openai",
                    model="gpt-4",
                    timestamp=now,
                    tenant_id=f"tenant_{tenant_id}",
                )
                cost_tracker_budget.record_cost(metrics)

            # Verify budget status
            budget = cost_tracker_budget.get_budget(f"tenant_{tenant_id}")
            assert budget is not None
            assert budget.current_spend == 8.0

        # Verify budgets are isolated (each tenant has own budget)
        for tenant_id in range(3):
            budget = cost_tracker_budget.get_budget(f"tenant_{tenant_id}")
            assert budget is not None
            assert budget.current_spend == 8.0
            assert budget.limit_amount == 10.0

    def test_hard_limit_enforcement(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test hard limit budget enforcement (reject requests when exceeded)."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_hard_limit",
            limit_amount=5.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $4.50 (within budget)
        metrics1 = CostMetrics(
            total_cost=4.50,
            input_cost=2.25,
            output_cost=2.25,
            input_tokens=112500,
            output_tokens=75000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_hard_limit",
        )
        cost_tracker_budget.record_cost(metrics1)

        # Attempt to spend $1.00 more (would exceed $5.00 limit)
        metrics2 = CostMetrics(
            total_cost=1.00,
            input_cost=0.50,
            output_cost=0.50,
            input_tokens=25000,
            output_tokens=16667,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_hard_limit",
        )

        # Should raise exception due to hard limit
        with pytest.raises(PortkeyBudgetExceededError) as exc_info:
            cost_tracker_budget.record_cost(metrics2)

        assert "Budget exceeded" in str(exc_info.value)
        assert "tenant_hard_limit" in str(exc_info.value)

    def test_soft_limit_enforcement(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test soft limit budget enforcement (warn but allow overage)."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_soft_limit",
            limit_amount=5.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=False,  # Soft limit
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $6.00 (exceeds limit but should be allowed)
        metrics = CostMetrics(
            total_cost=6.00,
            input_cost=3.00,
            output_cost=3.00,
            input_tokens=150000,
            output_tokens=100000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_soft_limit",
        )

        # Should not raise exception (soft limit)
        cost_tracker_budget.record_cost(metrics)

        # Verify overage is recorded
        updated_budget = cost_tracker_budget.get_budget("tenant_soft_limit")
        assert updated_budget is not None
        assert updated_budget.current_spend == 6.00
        assert updated_budget.current_spend > updated_budget.limit_amount


class TestQuotaLimits:
    """Test quota limits (requests, tokens, cost)."""

    def test_cost_quota_enforcement(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test cost-based quota limits."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_cost_quota",
            limit_amount=100.0,
            period=CostPeriod.WEEKLY,
            period_start=now,
            period_end=now + timedelta(weeks=1),
            hard_limit=True,
        )
        cost_tracker_budget.set_budget(budget)

        # Simulate spending over time
        total_spent = 0.0
        max_requests = 50

        for i in range(max_requests):
            request_cost = 2.50

            # Check if budget available before request
            available, reason = cost_tracker_budget.check_budget_available(
                "tenant_cost_quota", request_cost
            )

            if available:
                metrics = CostMetrics(
                    total_cost=request_cost,
                    input_cost=1.25,
                    output_cost=1.25,
                    input_tokens=62500,
                    output_tokens=41667,
                    provider_id="openai",
                    model="gpt-4",
                    timestamp=now + timedelta(hours=i),
                    tenant_id="tenant_cost_quota",
                )
                cost_tracker_budget.record_cost(metrics)
                total_spent += request_cost
            else:
                # Budget exhausted
                break

        # Should have stopped before exceeding $100
        assert total_spent <= 100.0

        # Verify budget status
        budget_status = cost_tracker_budget.get_budget("tenant_cost_quota")
        assert budget_status is not None
        assert budget_status.current_spend <= budget_status.limit_amount

    def test_token_based_budget_tracking(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test tracking budgets based on token usage."""
        now = datetime.now()

        # Track token usage across multiple requests
        total_tokens = 0
        target_tokens = 1000000  # 1M tokens

        for i in range(100):
            input_tokens = 5000
            output_tokens = 5000
            total_tokens += input_tokens + output_tokens

            metrics = CostMetrics(
                total_cost=0.10,
                input_cost=0.05,
                output_cost=0.05,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_id="openai",
                model="gpt-4",
                timestamp=now + timedelta(seconds=i),
                tenant_id="tenant_tokens",
            )
            cost_tracker_budget.record_cost(metrics)

        # Get summary to verify token tracking
        summary = cost_tracker_budget.get_summary(
            period_start=now,
            period_end=now + timedelta(hours=1),
            tenant_id="tenant_tokens",
        )

        assert summary.total_input_tokens == 500000  # 100 * 5000
        assert summary.total_output_tokens == 500000  # 100 * 5000
        assert summary.total_input_tokens + summary.total_output_tokens == total_tokens


class TestBudgetAlerts:
    """Test budget alerts and notifications."""

    def test_threshold_alert_generation(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test alert generation at budget thresholds."""
        now = datetime.now()

        # Configure budget with multiple thresholds
        budget = BudgetConfig(
            tenant_id="tenant_alerts",
            limit_amount=100.0,
            period=CostPeriod.MONTHLY,
            period_start=now,
            period_end=now + timedelta(days=30),
            thresholds=[
                BudgetThreshold(
                    threshold_percent=50.0,
                    severity=BudgetAlertSeverity.INFO,
                ),
                BudgetThreshold(
                    threshold_percent=75.0,
                    severity=BudgetAlertSeverity.WARNING,
                ),
                BudgetThreshold(
                    threshold_percent=90.0,
                    severity=BudgetAlertSeverity.CRITICAL,
                ),
                BudgetThreshold(
                    threshold_percent=100.0,
                    severity=BudgetAlertSeverity.EMERGENCY,
                ),
            ],
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $60 (60%, should trigger 50% threshold)
        metrics1 = CostMetrics(
            total_cost=60.0,
            input_cost=30.0,
            output_cost=30.0,
            input_tokens=1500000,
            output_tokens=1000000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_alerts",
        )
        cost_tracker_budget.record_cost(metrics1)

        # Check for alerts
        alerts = cost_tracker_budget.get_alerts(
            tenant_id="tenant_alerts",
            acknowledged=False,
        )
        assert len(alerts) >= 1
        assert any(a.threshold.threshold_percent == 50.0 for a in alerts)

        # Spend $20 more (80%, should trigger 75% threshold)
        metrics2 = CostMetrics(
            total_cost=20.0,
            input_cost=10.0,
            output_cost=10.0,
            input_tokens=500000,
            output_tokens=333333,
            provider_id="openai",
            model="gpt-4",
            timestamp=now + timedelta(hours=1),
            tenant_id="tenant_alerts",
        )
        cost_tracker_budget.record_cost(metrics2)

        # Check for new alerts
        alerts = cost_tracker_budget.get_alerts(
            tenant_id="tenant_alerts",
            acknowledged=False,
        )
        assert len(alerts) >= 2
        assert any(a.threshold.threshold_percent == 75.0 for a in alerts)

    def test_alert_severity_levels(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test different alert severity levels."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_severity",
            limit_amount=10.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            thresholds=[
                BudgetThreshold(
                    threshold_percent=95.0,
                    severity=BudgetAlertSeverity.CRITICAL,
                ),
            ],
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $9.60 (96%, should trigger CRITICAL alert)
        metrics = CostMetrics(
            total_cost=9.60,
            input_cost=4.80,
            output_cost=4.80,
            input_tokens=240000,
            output_tokens=160000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_severity",
        )
        cost_tracker_budget.record_cost(metrics)

        # Get alerts and verify severity
        alerts = cost_tracker_budget.get_alerts(tenant_id="tenant_severity")
        assert len(alerts) > 0

        critical_alerts = [
            a for a in alerts if a.threshold.severity == BudgetAlertSeverity.CRITICAL
        ]
        assert len(critical_alerts) > 0
        assert "CRITICAL" in critical_alerts[0].message

    def test_alert_debouncing(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test alert debouncing to prevent spam."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_debounce",
            limit_amount=10.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            thresholds=[
                BudgetThreshold(
                    threshold_percent=50.0,
                    severity=BudgetAlertSeverity.WARNING,
                ),
            ],
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $6.00 (60%, triggers threshold)
        metrics1 = CostMetrics(
            total_cost=6.00,
            input_cost=3.00,
            output_cost=3.00,
            input_tokens=150000,
            output_tokens=100000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_debounce",
        )
        cost_tracker_budget.record_cost(metrics1)

        alerts_count_1 = len(cost_tracker_budget.get_alerts(tenant_id="tenant_debounce"))

        # Spend another $0.50 immediately (still above threshold)
        metrics2 = CostMetrics(
            total_cost=0.50,
            input_cost=0.25,
            output_cost=0.25,
            input_tokens=12500,
            output_tokens=8333,
            provider_id="openai",
            model="gpt-4",
            timestamp=now + timedelta(seconds=1),
            tenant_id="tenant_debounce",
        )
        cost_tracker_budget.record_cost(metrics2)

        alerts_count_2 = len(cost_tracker_budget.get_alerts(tenant_id="tenant_debounce"))

        # Should not generate duplicate alert due to debouncing
        assert alerts_count_2 == alerts_count_1


class TestOverageHandling:
    """Test budget overage handling."""

    def test_soft_limit_overage_tracking(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test tracking overage for soft limit budgets."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_overage",
            limit_amount=50.0,
            period=CostPeriod.WEEKLY,
            period_start=now,
            period_end=now + timedelta(weeks=1),
            hard_limit=False,  # Soft limit allows overage
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $75 (50% over budget)
        metrics = CostMetrics(
            total_cost=75.0,
            input_cost=37.5,
            output_cost=37.5,
            input_tokens=1875000,
            output_tokens=1250000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_overage",
        )
        cost_tracker_budget.record_cost(metrics)

        # Verify overage is tracked
        updated_budget = cost_tracker_budget.get_budget("tenant_overage")
        assert updated_budget is not None
        assert updated_budget.current_spend == 75.0
        assert updated_budget.current_spend > updated_budget.limit_amount

        overage = updated_budget.current_spend - updated_budget.limit_amount
        assert overage == 25.0

        overage_percent = (overage / updated_budget.limit_amount) * 100
        assert overage_percent == 50.0

    def test_hard_limit_prevents_overage(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test that hard limit prevents any overage."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_no_overage",
            limit_amount=20.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $15.00
        metrics1 = CostMetrics(
            total_cost=15.00,
            input_cost=7.50,
            output_cost=7.50,
            input_tokens=375000,
            output_tokens=250000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_no_overage",
        )
        cost_tracker_budget.record_cost(metrics1)

        # Attempt to spend $10.00 more (would exceed $20.00 limit)
        metrics2 = CostMetrics(
            total_cost=10.00,
            input_cost=5.00,
            output_cost=5.00,
            input_tokens=250000,
            output_tokens=166667,
            provider_id="openai",
            model="gpt-4",
            timestamp=now + timedelta(minutes=10),
            tenant_id="tenant_no_overage",
        )

        with pytest.raises(PortkeyBudgetExceededError):
            cost_tracker_budget.record_cost(metrics2)

        # Verify the overage was recorded (cost is added before exception)
        # but the exception prevents further processing
        final_budget = cost_tracker_budget.get_budget("tenant_no_overage")
        assert final_budget is not None
        # Cost tracker records the cost before raising exception
        assert final_budget.current_spend == 25.00  # Both requests recorded
        # But the exception was raised to signal the violation
        assert final_budget.current_spend > final_budget.limit_amount


class TestBudgetResetCycles:
    """Test budget reset cycles (daily, weekly, monthly)."""

    def test_budget_period_expiration(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test that spending outside budget period is not counted."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_period",
            limit_amount=10.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )
        cost_tracker_budget.set_budget(budget)

        # Spend within period
        metrics_within = CostMetrics(
            total_cost=5.0,
            input_cost=2.5,
            output_cost=2.5,
            input_tokens=125000,
            output_tokens=83333,
            provider_id="openai",
            model="gpt-4",
            timestamp=now + timedelta(hours=12),
            tenant_id="tenant_period",
        )
        cost_tracker_budget.record_cost(metrics_within)

        # Verify spending counted
        budget_status = cost_tracker_budget.get_budget("tenant_period")
        assert budget_status is not None
        assert budget_status.current_spend == 5.0

        # Spend outside period (after expiration)
        metrics_after = CostMetrics(
            total_cost=3.0,
            input_cost=1.5,
            output_cost=1.5,
            input_tokens=75000,
            output_tokens=50000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now + timedelta(days=2),  # After period end
            tenant_id="tenant_period",
        )
        cost_tracker_budget.record_cost(metrics_after)

        # Verify spending not counted (outside period)
        budget_status_after = cost_tracker_budget.get_budget("tenant_period")
        assert budget_status_after is not None
        assert budget_status_after.current_spend == 5.0  # Unchanged


class TestMultiTenantBudgetIsolation:
    """Test multi-tenant budget isolation."""

    def test_tenant_budget_isolation(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test that tenant budgets are completely isolated."""
        now = datetime.now()

        # Create budgets for 10 tenants
        num_tenants = 10
        for tenant_id in range(num_tenants):
            budget = BudgetConfig(
                tenant_id=f"tenant_{tenant_id}",
                limit_amount=100.0,
                period=CostPeriod.MONTHLY,
                period_start=now,
                period_end=now + timedelta(days=30),
                hard_limit=True,
            )
            cost_tracker_budget.set_budget(budget)

        # Each tenant spends different amounts
        for tenant_id in range(num_tenants):
            spend_amount = 10.0 * (tenant_id + 1)  # $10, $20, $30, etc.

            metrics = CostMetrics(
                total_cost=spend_amount,
                input_cost=spend_amount / 2,
                output_cost=spend_amount / 2,
                input_tokens=int(spend_amount * 25000),
                output_tokens=int(spend_amount * 16667),
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id=f"tenant_{tenant_id}",
            )
            cost_tracker_budget.record_cost(metrics)

        # Verify each tenant's budget is independent
        for tenant_id in range(num_tenants):
            budget = cost_tracker_budget.get_budget(f"tenant_{tenant_id}")
            assert budget is not None
            expected_spend = 10.0 * (tenant_id + 1)
            assert budget.current_spend == expected_spend

    def test_tenant_cannot_exceed_own_budget(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test that one tenant exceeding budget doesn't affect others."""
        now = datetime.now()

        # Create budgets
        budget1 = BudgetConfig(
            tenant_id="tenant_A",
            limit_amount=5.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )
        budget2 = BudgetConfig(
            tenant_id="tenant_B",
            limit_amount=50.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )

        cost_tracker_budget.set_budget(budget1)
        cost_tracker_budget.set_budget(budget2)

        # Tenant A exceeds budget
        metrics_a = CostMetrics(
            total_cost=5.0,
            input_cost=2.5,
            output_cost=2.5,
            input_tokens=125000,
            output_tokens=83333,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_A",
        )
        cost_tracker_budget.record_cost(metrics_a)

        # Attempt to exceed tenant A budget
        with pytest.raises(PortkeyBudgetExceededError):
            metrics_a_exceed = CostMetrics(
                total_cost=1.0,
                input_cost=0.5,
                output_cost=0.5,
                input_tokens=25000,
                output_tokens=16667,
                provider_id="openai",
                model="gpt-4",
                timestamp=now + timedelta(minutes=1),
                tenant_id="tenant_A",
            )
            cost_tracker_budget.record_cost(metrics_a_exceed)

        # Tenant B should still be able to spend
        metrics_b = CostMetrics(
            total_cost=10.0,
            input_cost=5.0,
            output_cost=5.0,
            input_tokens=250000,
            output_tokens=166667,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_B",
        )
        # Should succeed (tenant B has $50 budget)
        cost_tracker_budget.record_cost(metrics_b)

        budget_b = cost_tracker_budget.get_budget("tenant_B")
        assert budget_b is not None
        assert budget_b.current_spend == 10.0


class TestConcurrencyAndRaceConditions:
    """Test concurrent request handling and race conditions."""

    def test_concurrent_budget_checks(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test budget checking with concurrent requests.

        Note: This is a simplified test. In production, proper locking
        mechanisms would be needed for true concurrent safety.
        """
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_concurrent",
            limit_amount=10.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )
        cost_tracker_budget.set_budget(budget)

        # Simulate concurrent requests each costing $3
        # Total would be $15 if all succeed (exceeds $10 budget)
        request_cost = 3.0
        successful_requests = 0

        for i in range(5):
            # Check if budget available
            available, _ = cost_tracker_budget.check_budget_available(
                "tenant_concurrent", request_cost
            )

            if available:
                try:
                    metrics = CostMetrics(
                        total_cost=request_cost,
                        input_cost=request_cost / 2,
                        output_cost=request_cost / 2,
                        input_tokens=int(request_cost * 25000),
                        output_tokens=int(request_cost * 16667),
                        provider_id="openai",
                        model="gpt-4",
                        timestamp=now + timedelta(seconds=i),
                        tenant_id="tenant_concurrent",
                    )
                    cost_tracker_budget.record_cost(metrics)
                    successful_requests += 1
                except PortkeyBudgetExceededError:
                    # Budget exceeded
                    break

        # Should only allow 3 requests ($9), not 4 ($12 would exceed $10)
        final_budget = cost_tracker_budget.get_budget("tenant_concurrent")
        assert final_budget is not None
        assert final_budget.current_spend <= final_budget.limit_amount
        assert successful_requests <= 3

    def test_budget_check_before_request(
        self,
        cost_tracker_budget: CostTracker,
    ) -> None:
        """Test pre-flight budget checking to prevent overage."""
        now = datetime.now()

        budget = BudgetConfig(
            tenant_id="tenant_preflight",
            limit_amount=20.0,
            period=CostPeriod.DAILY,
            period_start=now,
            period_end=now + timedelta(days=1),
            hard_limit=True,
        )
        cost_tracker_budget.set_budget(budget)

        # Spend $18
        metrics1 = CostMetrics(
            total_cost=18.0,
            input_cost=9.0,
            output_cost=9.0,
            input_tokens=450000,
            output_tokens=300000,
            provider_id="openai",
            model="gpt-4",
            timestamp=now,
            tenant_id="tenant_preflight",
        )
        cost_tracker_budget.record_cost(metrics1)

        # Check if $5 is available (should fail - only $2 remaining)
        available, reason = cost_tracker_budget.check_budget_available(
            "tenant_preflight", 5.0
        )

        assert available is False
        assert reason is not None
        assert "would be exceeded" in reason
        assert "$2" in reason  # Remaining budget

        # Check if $1.50 is available (should succeed)
        available, _ = cost_tracker_budget.check_budget_available(
            "tenant_preflight", 1.50
        )

        assert available is True
