"""Cost tracking accuracy tests.

Tests cost calculation accuracy, token counting accuracy, cache savings
calculation, cost allocation across tenants, chargeback report generation,
and comparison with real provider pricing. Validates edge cases including
free tier, discounts, and promotional pricing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.llm_gateway.cost_models import CostMetrics
from agentcore.llm_gateway.cost_tracker import CostTracker


@pytest.fixture
def cost_tracker_accuracy() -> CostTracker:
    """Create a cost tracker for accuracy testing."""
    return CostTracker()


class TestCostCalculationAccuracy:
    """Test accuracy of cost calculations."""

    def test_cost_calculation_precision(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost calculation with high precision."""
        # Test case: GPT-4 pricing (realistic)
        # Input: $0.01 per 1K tokens
        # Output: $0.03 per 1K tokens

        input_tokens = 1000
        output_tokens = 500
        input_price = 0.01  # Per 1K
        output_price = 0.03  # Per 1K

        metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_token_price=input_price,
            output_token_price=output_price,
            model="gpt-4",
        )

        # Expected: (1000/1000 * 0.01) + (500/1000 * 0.03) = 0.01 + 0.015 = 0.025
        expected_cost = 0.025
        assert metrics.total_cost == pytest.approx(expected_cost, abs=0.0001)
        assert metrics.input_cost == pytest.approx(0.01, abs=0.0001)
        assert metrics.output_cost == pytest.approx(0.015, abs=0.0001)

    def test_cost_calculation_large_volumes(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost calculation with large token volumes."""
        # Large request: 100K input tokens, 50K output tokens
        input_tokens = 100000
        output_tokens = 50000
        input_price = 0.01
        output_price = 0.03

        metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_token_price=input_price,
            output_token_price=output_price,
            model="gpt-4",
        )

        # Expected: (100000/1000 * 0.01) + (50000/1000 * 0.03) = 1.0 + 1.5 = 2.5
        expected_cost = 2.5
        assert metrics.total_cost == pytest.approx(expected_cost, abs=0.001)

    def test_cost_calculation_small_volumes(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost calculation with small token volumes."""
        # Small request: 10 input tokens, 20 output tokens
        input_tokens = 10
        output_tokens = 20
        input_price = 0.0005  # Cheap provider
        output_price = 0.0015

        metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="cheap_provider",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_token_price=input_price,
            output_token_price=output_price,
            model="gpt-3.5",
        )

        # Expected: (10/1000 * 0.0005) + (20/1000 * 0.0015) = 0.000005 + 0.00003 = 0.000035
        expected_cost = 0.000035
        assert metrics.total_cost == pytest.approx(expected_cost, abs=0.000001)

    def test_cost_calculation_real_provider_pricing(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost calculation with real provider pricing."""
        # Test multiple real-world provider pricing scenarios

        test_cases = [
            # (provider, model, input_price, output_price, input_tokens, output_tokens, expected_cost)
            ("openai", "gpt-4-turbo", 0.01, 0.03, 1000, 500, 0.025),
            ("openai", "gpt-3.5-turbo", 0.0005, 0.0015, 1000, 500, 0.00125),
            ("anthropic", "claude-opus-4-1-20250805", 0.015, 0.075, 1000, 500, 0.0525),
            ("anthropic", "claude-3-sonnet", 0.003, 0.015, 1000, 500, 0.0105),
            ("anthropic", "claude-3-haiku", 0.00025, 0.00125, 1000, 500, 0.000875),
        ]

        for (
            provider,
            model,
            input_price,
            output_price,
            input_tokens,
            output_tokens,
            expected,
        ) in test_cases:
            metrics = cost_tracker_accuracy.calculate_request_cost(
                provider_id=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_token_price=input_price,
                output_token_price=output_price,
                model=model,
            )

            assert metrics.total_cost == pytest.approx(expected, rel=0.01), (
                f"Cost mismatch for {provider}/{model}: "
                f"expected ${expected}, got ${metrics.total_cost}"
            )


class TestTokenCountingAccuracy:
    """Test token counting and tracking accuracy."""

    def test_token_aggregation_accuracy(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test accurate token aggregation across multiple requests."""
        now = datetime.now(UTC)

        # Record multiple requests with varying token counts
        test_requests = [
            (1000, 500),
            (2000, 1500),
            (500, 750),
            (1500, 1000),
            (3000, 2000),
        ]

        total_input = 0
        total_output = 0

        for input_tokens, output_tokens in test_requests:
            total_input += input_tokens
            total_output += output_tokens

            metrics = CostMetrics(
                total_cost=0.01,
                input_cost=0.005,
                output_cost=0.005,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id="tenant_1",
            )
            cost_tracker_accuracy.record_cost(metrics)

        # Get summary and verify token counts
        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        assert summary.total_input_tokens == total_input  # 8000
        assert summary.total_output_tokens == total_output  # 5750
        assert summary.total_requests == len(test_requests)

    def test_token_cost_per_1k_calculation(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost per 1K tokens calculation."""
        now = datetime.now(UTC)

        # Record requests with known costs
        for _ in range(10):
            metrics = CostMetrics(
                total_cost=0.04,  # $0.04 per request
                input_cost=0.01,
                output_cost=0.03,
                input_tokens=1000,
                output_tokens=1000,
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id="tenant_1",
            )
            cost_tracker_accuracy.record_cost(metrics)

        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        # Total: 10 requests * $0.04 = $0.40
        # Total tokens: 10 * 2000 = 20,000
        # Cost per 1K: ($0.40 / 20,000) * 1000 = $0.02
        expected_cost_per_1k = 0.02
        assert summary.average_cost_per_1k_tokens == pytest.approx(
            expected_cost_per_1k, abs=0.001
        )


class TestCostAllocationAcrossTenants:
    """Test cost allocation and tracking across multiple tenants."""

    def test_per_tenant_cost_tracking(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test accurate per-tenant cost tracking."""
        now = datetime.now(UTC)

        # Record costs for multiple tenants
        tenant_costs = {
            "tenant_A": 10.50,
            "tenant_B": 25.75,
            "tenant_C": 5.25,
            "tenant_D": 42.00,
        }

        for tenant_id, cost in tenant_costs.items():
            metrics = CostMetrics(
                total_cost=cost,
                input_cost=cost / 2,
                output_cost=cost / 2,
                input_tokens=int(cost * 25000),
                output_tokens=int(cost * 16667),
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id=tenant_id,
            )
            cost_tracker_accuracy.record_cost(metrics)

        # Get overall summary
        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        # Verify tenant breakdown
        assert len(summary.tenant_breakdown) == len(tenant_costs)
        for tenant_id, expected_cost in tenant_costs.items():
            assert summary.tenant_breakdown[tenant_id] == pytest.approx(
                expected_cost, abs=0.01
            )

        # Verify total
        total_cost = sum(tenant_costs.values())
        assert summary.total_cost == pytest.approx(total_cost, abs=0.01)

    def test_tenant_cost_isolation(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test that tenant costs are properly isolated."""
        now = datetime.now(UTC)

        # Record costs for tenant A
        for _ in range(5):
            metrics = CostMetrics(
                total_cost=2.0,
                input_cost=1.0,
                output_cost=1.0,
                input_tokens=50000,
                output_tokens=33333,
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id="tenant_A",
            )
            cost_tracker_accuracy.record_cost(metrics)

        # Record costs for tenant B
        for _ in range(3):
            metrics = CostMetrics(
                total_cost=5.0,
                input_cost=2.5,
                output_cost=2.5,
                input_tokens=125000,
                output_tokens=83333,
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id="tenant_B",
            )
            cost_tracker_accuracy.record_cost(metrics)

        # Get summary for tenant A only
        summary_a = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
            tenant_id="tenant_A",
        )

        assert summary_a.total_cost == pytest.approx(10.0, abs=0.01)  # 5 * $2
        assert summary_a.total_requests == 5

        # Get summary for tenant B only
        summary_b = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
            tenant_id="tenant_B",
        )

        assert summary_b.total_cost == pytest.approx(15.0, abs=0.01)  # 3 * $5
        assert summary_b.total_requests == 3


class TestProviderCostBreakdown:
    """Test cost breakdown by provider."""

    def test_multi_provider_cost_breakdown(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost breakdown across multiple providers."""
        now = datetime.now(UTC)

        provider_costs = {
            "openai": 50.00,
            "anthropic": 30.00,
            "google": 20.00,
        }

        for provider_id, cost in provider_costs.items():
            metrics = CostMetrics(
                total_cost=cost,
                input_cost=cost / 2,
                output_cost=cost / 2,
                input_tokens=int(cost * 25000),
                output_tokens=int(cost * 16667),
                provider_id=provider_id,
                model="model",
                timestamp=now,
                tenant_id="tenant_1",
            )
            cost_tracker_accuracy.record_cost(metrics)

        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        # Verify provider breakdown
        assert len(summary.provider_breakdown) == len(provider_costs)
        for provider_id, expected_cost in provider_costs.items():
            assert summary.provider_breakdown[provider_id] == pytest.approx(
                expected_cost, abs=0.01
            )

        # Verify total
        assert summary.total_cost == pytest.approx(
            sum(provider_costs.values()), abs=0.01
        )


class TestChargebackReportGeneration:
    """Test chargeback report generation for enterprise billing."""

    def test_chargeback_by_tenant(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test chargeback reporting by tenant."""
        now = datetime.now(UTC)

        # Simulate a month of usage
        tenants = ["sales", "engineering", "marketing", "support"]
        monthly_costs = {
            "sales": 500.00,
            "engineering": 1200.00,
            "marketing": 300.00,
            "support": 450.00,
        }

        for tenant, cost in monthly_costs.items():
            # Distribute cost across multiple requests
            num_requests = 10
            cost_per_request = cost / num_requests

            for i in range(num_requests):
                metrics = CostMetrics(
                    total_cost=cost_per_request,
                    input_cost=cost_per_request / 2,
                    output_cost=cost_per_request / 2,
                    input_tokens=int(cost_per_request * 25000),
                    output_tokens=int(cost_per_request * 16667),
                    provider_id="openai",
                    model="gpt-4",
                    timestamp=now + timedelta(days=i),
                    tenant_id=tenant,
                    tags={"department": tenant, "billable": "true"},
                )
                cost_tracker_accuracy.record_cost(metrics)

        # Generate chargeback report
        summary = cost_tracker_accuracy.get_summary(
            period_start=now,
            period_end=now + timedelta(days=30),
        )

        # Verify chargeback amounts
        for tenant, expected_cost in monthly_costs.items():
            actual_cost = summary.tenant_breakdown.get(tenant, 0.0)
            assert actual_cost == pytest.approx(expected_cost, rel=0.01), (
                f"Chargeback mismatch for {tenant}: "
                f"expected ${expected_cost}, got ${actual_cost}"
            )

        # Verify total monthly bill
        total_bill = sum(monthly_costs.values())
        assert summary.total_cost == pytest.approx(total_bill, rel=0.01)

    def test_chargeback_by_workflow(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test cost allocation by workflow for project billing."""
        now = datetime.now(UTC)

        workflows = {
            "data_pipeline": 150.00,
            "customer_support": 200.00,
            "content_generation": 100.00,
        }

        for workflow_id, cost in workflows.items():
            metrics = CostMetrics(
                total_cost=cost,
                input_cost=cost / 2,
                output_cost=cost / 2,
                input_tokens=int(cost * 25000),
                output_tokens=int(cost * 16667),
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id="acme_corp",
                workflow_id=workflow_id,
            )
            cost_tracker_accuracy.record_cost(metrics)

        # Get all costs for tenant
        history = cost_tracker_accuracy.get_cost_history(
            tenant_id="acme_corp",
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=1),
        )

        # Aggregate by workflow
        workflow_totals: dict[str, float] = {}
        for metrics in history:
            if metrics.workflow_id:
                workflow_totals[metrics.workflow_id] = (
                    workflow_totals.get(metrics.workflow_id, 0.0) + metrics.total_cost
                )

        # Verify workflow costs
        for workflow_id, expected_cost in workflows.items():
            actual_cost = workflow_totals.get(workflow_id, 0.0)
            assert actual_cost == pytest.approx(expected_cost, abs=0.01)


class TestCostReportingAccuracy:
    """Test accuracy of cost reporting and analytics."""

    def test_average_cost_per_request(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test average cost per request calculation."""
        now = datetime.now(UTC)

        # Record requests with varying costs
        request_costs = [0.10, 0.20, 0.15, 0.25, 0.30]

        for cost in request_costs:
            metrics = CostMetrics(
                total_cost=cost,
                input_cost=cost / 2,
                output_cost=cost / 2,
                input_tokens=int(cost * 50000),
                output_tokens=int(cost * 33333),
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                tenant_id="tenant_1",
            )
            cost_tracker_accuracy.record_cost(metrics)

        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        # Expected average: (0.10 + 0.20 + 0.15 + 0.25 + 0.30) / 5 = 0.20
        expected_avg = sum(request_costs) / len(request_costs)
        assert summary.average_cost_per_request == pytest.approx(expected_avg, abs=0.01)

    def test_latency_tracking_accuracy(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test latency tracking and average calculation."""
        now = datetime.now(UTC)

        # Record requests with varying latencies
        latencies = [800, 1200, 900, 1500, 1000]

        for latency in latencies:
            metrics = CostMetrics(
                total_cost=0.05,
                input_cost=0.025,
                output_cost=0.025,
                input_tokens=1250,
                output_tokens=833,
                provider_id="openai",
                model="gpt-4",
                timestamp=now,
                latency_ms=latency,
                tenant_id="tenant_1",
            )
            cost_tracker_accuracy.record_cost(metrics)

        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        # Expected average: (800 + 1200 + 900 + 1500 + 1000) / 5 = 1080
        expected_avg = sum(latencies) // len(latencies)
        assert summary.average_latency_ms == expected_avg


class TestEdgeCasesAndSpecialPricing:
    """Test edge cases including free tier, discounts, and promotional pricing."""

    def test_free_tier_usage(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test tracking of free tier usage (zero cost)."""
        now = datetime.now(UTC)

        # Free tier request
        metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="free_provider",
            input_tokens=1000,
            output_tokens=500,
            input_token_price=0.0,  # Free
            output_token_price=0.0,  # Free
            model="free-model",
        )

        assert metrics.total_cost == 0.0
        assert metrics.input_cost == 0.0
        assert metrics.output_cost == 0.0

        # Record and verify
        cost_tracker_accuracy.record_cost(metrics)

        summary = cost_tracker_accuracy.get_summary(
            period_start=now - timedelta(minutes=1),
            period_end=now + timedelta(minutes=1),
        )

        assert summary.total_cost == 0.0
        assert summary.total_requests == 1

    def test_zero_token_request(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test handling of zero-token requests."""
        metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="openai",
            input_tokens=0,
            output_tokens=0,
            input_token_price=0.01,
            output_token_price=0.03,
            model="gpt-4",
        )

        assert metrics.total_cost == 0.0
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0

    def test_very_small_costs(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test handling of very small costs (micro-transactions)."""
        # 1 token with cheap pricing
        metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="ultra_cheap",
            input_tokens=1,
            output_tokens=1,
            input_token_price=0.0001,
            output_token_price=0.0002,
            model="cheap-model",
        )

        # Expected: (1/1000 * 0.0001) + (1/1000 * 0.0002) = 0.0000001 + 0.0000002 = 0.0000003
        expected_cost = 0.0000003
        assert metrics.total_cost == pytest.approx(expected_cost, abs=0.00000001)

    def test_promotional_pricing_discount(
        self,
        cost_tracker_accuracy: CostTracker,
    ) -> None:
        """Test tracking costs with promotional discounts."""
        now = datetime.now(UTC)

        # Regular pricing
        regular_price_input = 0.01
        regular_price_output = 0.03

        # Promotional pricing (50% off)
        promo_price_input = regular_price_input * 0.5
        promo_price_output = regular_price_output * 0.5

        # Record regular request
        regular_metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="openai",
            input_tokens=1000,
            output_tokens=500,
            input_token_price=regular_price_input,
            output_token_price=regular_price_output,
            model="gpt-4",
        )
        regular_metrics.timestamp = now
        regular_metrics.tags = {"pricing": "regular"}
        cost_tracker_accuracy.record_cost(regular_metrics)

        # Record promotional request
        promo_metrics = cost_tracker_accuracy.calculate_request_cost(
            provider_id="openai",
            input_tokens=1000,
            output_tokens=500,
            input_token_price=promo_price_input,
            output_token_price=promo_price_output,
            model="gpt-4",
        )
        promo_metrics.timestamp = now
        promo_metrics.tags = {"pricing": "promotional"}
        cost_tracker_accuracy.record_cost(promo_metrics)

        # Verify promotional savings
        assert promo_metrics.total_cost == pytest.approx(
            regular_metrics.total_cost * 0.5, rel=0.01
        )

        # Get history and verify tags
        history = cost_tracker_accuracy.get_cost_history(
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=1),
        )

        regular_records = [h for h in history if h.tags.get("pricing") == "regular"]
        promo_records = [h for h in history if h.tags.get("pricing") == "promotional"]

        assert len(regular_records) == 1
        assert len(promo_records) == 1
