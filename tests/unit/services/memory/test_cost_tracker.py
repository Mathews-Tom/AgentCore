"""
Unit Tests for CostTracker

Tests cost tracking, budget monitoring, and alerting for compression operations.
Covers all public methods with 90%+ coverage requirement.

Component ID: MEM-014
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.services.memory.cost_tracker import (
    BudgetAlert,
    CostTracker,
)


@pytest.fixture
def mock_session():
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def cost_tracker():
    """Create CostTracker instance."""
    return CostTracker(trace_id="test-trace-123")


class TestCostTrackerInit:
    """Test CostTracker initialization."""

    def test_init_with_trace_id(self):
        """Test initialization with trace ID."""
        tracker = CostTracker(trace_id="test-trace")
        assert tracker._trace_id == "test-trace"

    def test_init_without_trace_id(self):
        """Test initialization without trace ID."""
        tracker = CostTracker()
        assert tracker._trace_id is None


class TestRecordCompressionCost:
    """Test record_compression_cost method."""

    @pytest.mark.asyncio
    async def test_record_cost_basic(self, cost_tracker, mock_session):
        """Test basic cost recording."""
        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            # Mock check_budget_threshold to avoid budget check
            with patch.object(cost_tracker, "check_budget_threshold", AsyncMock()):
                metric_id = await cost_tracker.record_compression_cost(
                    compression_type="stage",
                    input_tokens=1000,
                    output_tokens=100,
                    compression_ratio=10.0,
                    cost_usd=0.0021,
                    model_used="gpt-4.1-mini",
                    stage_id=str(uuid4()),
                    task_id=str(uuid4()),
                    agent_id=str(uuid4()),
                    critical_fact_retention_rate=0.97,
                    coherence_score=0.95,
                )

                # Verify metric_id is UUID string
                assert UUID(metric_id)

                # Verify session.execute called
                assert mock_session.execute.called
                assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_record_cost_optional_fields(self, cost_tracker, mock_session):
        """Test cost recording with optional fields only."""
        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            with patch.object(cost_tracker, "check_budget_threshold", AsyncMock()):
                metric_id = await cost_tracker.record_compression_cost(
                    compression_type="task",
                    input_tokens=500,
                    output_tokens=100,
                    compression_ratio=5.0,
                    cost_usd=0.0012,
                    model_used="gpt-4.1-mini",
                )

                assert UUID(metric_id)
                assert mock_session.execute.called

    @pytest.mark.asyncio
    async def test_record_cost_triggers_budget_check(self, cost_tracker, mock_session):
        """Test that recording cost triggers budget check."""
        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            with patch.object(
                cost_tracker, "check_budget_threshold", AsyncMock()
            ) as mock_check:
                await cost_tracker.record_compression_cost(
                    compression_type="stage",
                    input_tokens=1000,
                    output_tokens=100,
                    compression_ratio=10.0,
                    cost_usd=0.0021,
                    model_used="gpt-4.1-mini",
                )

                # Verify budget check was called
                mock_check.assert_called_once()


class TestGetMonthlyUsage:
    """Test get_monthly_usage method."""

    @pytest.mark.asyncio
    async def test_get_monthly_usage_current_month(self, cost_tracker, mock_session):
        """Test getting current month usage."""
        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (
            Decimal("15.50"),  # total_cost
            42,  # total_operations
            50000,  # total_input_tokens
            5000,  # total_output_tokens
        )
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            usage = await cost_tracker.get_monthly_usage()

            assert usage["total_cost"] == 15.50
            assert usage["total_operations"] == 42
            assert usage["total_input_tokens"] == 50000
            assert usage["total_output_tokens"] == 5000
            assert usage["avg_cost_per_operation"] == pytest.approx(15.50 / 42)
            assert "period_start" in usage
            assert "period_end" in usage

    @pytest.mark.asyncio
    async def test_get_monthly_usage_specific_month(self, cost_tracker, mock_session):
        """Test getting specific month usage."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (
            Decimal("25.00"),
            100,
            100000,
            10000,
        )
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            usage = await cost_tracker.get_monthly_usage(year=2024, month=6)

            assert usage["total_cost"] == 25.00
            assert "2024-06" in usage["period_start"]

    @pytest.mark.asyncio
    async def test_get_monthly_usage_no_data(self, cost_tracker, mock_session):
        """Test getting usage when no data exists."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (
            Decimal("0.00"),
            0,
            0,
            0,
        )
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            usage = await cost_tracker.get_monthly_usage()

            assert usage["total_cost"] == 0.0
            assert usage["total_operations"] == 0
            assert usage["avg_cost_per_operation"] == 0.0


class TestCheckBudgetThreshold:
    """Test check_budget_threshold method."""

    @pytest.mark.asyncio
    async def test_no_alert_under_threshold(self, cost_tracker):
        """Test no alert when under threshold."""
        with patch.object(
            cost_tracker, "get_monthly_usage", AsyncMock(return_value={"total_cost": 50.0})
        ):
            with patch(
                "agentcore.a2a_protocol.services.memory.cost_tracker.settings.MONTHLY_TOKEN_BUDGET_USD",
                100.0,
            ):
                alert = await cost_tracker.check_budget_threshold()
                assert alert is None

    @pytest.mark.asyncio
    async def test_alert_at_threshold(self, cost_tracker):
        """Test alert triggered at 75% threshold."""
        with patch.object(
            cost_tracker,
            "get_monthly_usage",
            AsyncMock(
                return_value={
                    "total_cost": 75.0,
                    "period_start": datetime.now(UTC).isoformat(),
                    "period_end": datetime.now(UTC).isoformat(),
                }
            ),
        ):
            with patch(
                "agentcore.a2a_protocol.services.memory.cost_tracker.settings.MONTHLY_TOKEN_BUDGET_USD",
                100.0,
            ):
                alert = await cost_tracker.check_budget_threshold()

                assert alert is not None
                assert isinstance(alert, BudgetAlert)
                assert alert.current_spend == 75.0
                assert alert.budget_limit == 100.0
                assert alert.consumption_percentage == 75.0
                assert alert.threshold_percentage == 75.0
                assert "75.0%" in alert.message

    @pytest.mark.asyncio
    async def test_alert_over_threshold(self, cost_tracker):
        """Test alert triggered when over threshold."""
        with patch.object(
            cost_tracker,
            "get_monthly_usage",
            AsyncMock(
                return_value={
                    "total_cost": 95.0,
                    "period_start": datetime.now(UTC).isoformat(),
                    "period_end": datetime.now(UTC).isoformat(),
                }
            ),
        ):
            with patch(
                "agentcore.a2a_protocol.services.memory.cost_tracker.settings.MONTHLY_TOKEN_BUDGET_USD",
                100.0,
            ):
                alert = await cost_tracker.check_budget_threshold()

                assert alert is not None
                assert alert.consumption_percentage == 95.0


class TestGetCostBreakdown:
    """Test get_cost_breakdown method."""

    @pytest.mark.asyncio
    async def test_breakdown_by_operation_type(self, cost_tracker, mock_session):
        """Test cost breakdown by operation type."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("stage", Decimal("10.50"), 30, 40000),
            ("task", Decimal("5.25"), 15, 20000),
        ]
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            breakdown = await cost_tracker.get_cost_breakdown(by="operation_type")

            assert len(breakdown["breakdown"]) == 2
            assert breakdown["breakdown"][0]["key"] == "stage"
            assert breakdown["breakdown"][0]["cost"] == 10.50
            assert breakdown["breakdown"][0]["operations"] == 30
            assert breakdown["total_cost"] == 15.75
            assert breakdown["total_operations"] == 45

    @pytest.mark.asyncio
    async def test_breakdown_by_agent(self, cost_tracker, mock_session):
        """Test cost breakdown by agent."""
        agent_id = str(uuid4())
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (agent_id, Decimal("12.00"), 25, 35000),
        ]
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            breakdown = await cost_tracker.get_cost_breakdown(by="agent")

            assert len(breakdown["breakdown"]) == 1
            assert breakdown["breakdown"][0]["key"] == agent_id

    @pytest.mark.asyncio
    async def test_breakdown_invalid_dimension(self, cost_tracker):
        """Test breakdown with invalid dimension."""
        with pytest.raises(ValueError, match="Invalid breakdown dimension"):
            await cost_tracker.get_cost_breakdown(by="invalid_dimension")

    @pytest.mark.asyncio
    async def test_breakdown_with_date_range(self, cost_tracker, mock_session):
        """Test breakdown with custom date range."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            start = datetime.now(UTC) - timedelta(days=7)
            end = datetime.now(UTC)

            breakdown = await cost_tracker.get_cost_breakdown(
                by="operation_type",
                start_date=start,
                end_date=end,
            )

            assert breakdown["period_start"] == start.isoformat()
            assert breakdown["period_end"] == end.isoformat()


class TestGetCostsByDateRange:
    """Test get_costs_by_date_range method."""

    @pytest.mark.asyncio
    async def test_costs_by_day(self, cost_tracker, mock_session):
        """Test costs grouped by day."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (datetime(2024, 6, 1, tzinfo=UTC), Decimal("5.00"), 10),
            (datetime(2024, 6, 2, tzinfo=UTC), Decimal("7.50"), 15),
        ]
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            start = datetime(2024, 6, 1, tzinfo=UTC)
            end = datetime(2024, 6, 2, tzinfo=UTC)

            result = await cost_tracker.get_costs_by_date_range(
                start_date=start,
                end_date=end,
                group_by="day",
            )

            assert len(result["costs"]) == 2
            assert result["costs"][0]["cost"] == 5.00
            assert result["costs"][0]["operations"] == 10
            assert result["total_cost"] == 12.50
            assert result["total_operations"] == 25
            assert result["group_by"] == "day"

    @pytest.mark.asyncio
    async def test_costs_by_hour(self, cost_tracker, mock_session):
        """Test costs grouped by hour."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            start = datetime.now(UTC) - timedelta(hours=24)
            end = datetime.now(UTC)

            result = await cost_tracker.get_costs_by_date_range(
                start_date=start,
                end_date=end,
                group_by="hour",
            )

            assert result["group_by"] == "hour"

    @pytest.mark.asyncio
    async def test_costs_invalid_group_by(self, cost_tracker):
        """Test costs with invalid group_by."""
        with pytest.raises(ValueError, match="Invalid group_by"):
            await cost_tracker.get_costs_by_date_range(
                start_date=datetime.now(UTC),
                end_date=datetime.now(UTC),
                group_by="invalid",
            )


class TestGetCostsByAgent:
    """Test get_costs_by_agent method."""

    @pytest.mark.asyncio
    async def test_costs_by_agent(self, cost_tracker, mock_session):
        """Test getting costs by agent."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (str(uuid4()), Decimal("10.00"), 20, 30000),
        ]
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            result = await cost_tracker.get_costs_by_agent()

            assert "breakdown" in result
            assert len(result["breakdown"]) == 1


class TestGetCostsByOperationType:
    """Test get_costs_by_operation_type method."""

    @pytest.mark.asyncio
    async def test_costs_by_operation_type(self, cost_tracker, mock_session):
        """Test getting costs by operation type."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("stage", Decimal("15.00"), 25, 40000),
            ("task", Decimal("8.00"), 12, 18000),
        ]
        mock_session.execute.return_value = mock_result

        with patch(
            "agentcore.a2a_protocol.services.memory.cost_tracker.get_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            result = await cost_tracker.get_costs_by_operation_type()

            assert "breakdown" in result
            assert len(result["breakdown"]) == 2
            assert result["breakdown"][0]["key"] == "stage"
            assert result["breakdown"][1]["key"] == "task"


class TestBudgetAlert:
    """Test BudgetAlert model."""

    def test_budget_alert_creation(self):
        """Test creating budget alert."""
        now = datetime.now(UTC)
        alert = BudgetAlert(
            alert_id="alert-123",
            current_spend=75.0,
            budget_limit=100.0,
            consumption_percentage=75.0,
            threshold_percentage=75.0,
            period_start=now,
            period_end=now + timedelta(days=30),
            message="Budget alert message",
        )

        assert alert.alert_id == "alert-123"
        assert alert.current_spend == 75.0
        assert alert.budget_limit == 100.0
        assert alert.consumption_percentage == 75.0
        assert alert.threshold_percentage == 75.0
        assert "Budget alert message" in alert.message


class TestCompressionMetricsModel:
    """Test CompressionMetricsModel."""

    def test_compression_metrics_model_creation(self):
        """Test creating compression metrics model."""
        from agentcore.a2a_protocol.services.memory.cost_tracker import (
            CompressionMetricsModel,
        )

        metric_id = uuid4()
        stage_id = uuid4()
        task_id = uuid4()
        agent_id = uuid4()
        now = datetime.now(UTC)

        model = CompressionMetricsModel(
            metric_id=metric_id,
            stage_id=stage_id,
            task_id=task_id,
            compression_type="stage",
            input_tokens=1000,
            output_tokens=100,
            compression_ratio=10.0,
            critical_fact_retention_rate=0.97,
            coherence_score=0.95,
            cost_usd=Decimal("0.0021"),
            model_used="gpt-4.1-mini",
            recorded_at=now,
            agent_id=agent_id,
        )

        assert model.metric_id == metric_id
        assert model.stage_id == stage_id
        assert model.task_id == task_id
        assert model.compression_type == "stage"
        assert model.input_tokens == 1000
        assert model.output_tokens == 100
        assert model.compression_ratio == 10.0
        assert model.critical_fact_retention_rate == 0.97
        assert model.coherence_score == 0.95
        assert model.cost_usd == Decimal("0.0021")
        assert model.model_used == "gpt-4.1-mini"
        assert model.recorded_at == now
        assert model.agent_id == agent_id
