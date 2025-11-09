"""
Unit tests for ACEMemoryInterface (COMPASS ACE-3 - ACE-021).

Tests all 4 query types, mock data generation, health score computation,
relevance scoring, query latency, and error handling.

Coverage target: 90%+
Performance target: <150ms query latency (p95)
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from agentcore.ace.integration.mem_interface import ACEMemoryInterface
from agentcore.ace.models.ace_models import QueryType


class TestACEMemoryInterfaceInit:
    """Test ACEMemoryInterface initialization."""

    def test_init_success_default(self):
        """Test successful initialization with defaults."""
        interface = ACEMemoryInterface()
        assert interface.mem_client is None
        assert interface._random is not None

    def test_init_success_with_client(self):
        """Test successful initialization with MEM client."""
        mock_client = object()
        interface = ACEMemoryInterface(mem_client=mock_client)
        assert interface.mem_client is mock_client

    def test_init_success_with_seed(self):
        """Test successful initialization with deterministic seed."""
        interface1 = ACEMemoryInterface(seed=42)
        interface2 = ACEMemoryInterface(seed=42)
        assert interface1._random.random() == interface2._random.random()

    def test_init_different_seeds_produce_different_randoms(self):
        """Test different seeds produce different random sequences."""
        interface1 = ACEMemoryInterface(seed=42)
        interface2 = ACEMemoryInterface(seed=99)
        assert interface1._random.random() != interface2._random.random()


class TestGetStrategicContext:
    """Test get_strategic_context method for all query types."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface with deterministic seed."""
        return ACEMemoryInterface(seed=42)

    @pytest.fixture
    def agent_id(self):
        """Test agent ID."""
        return "agent-001"

    @pytest.fixture
    def task_id(self):
        """Test task ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_strategic_decision_query(self, interface, agent_id, task_id):
        """Test STRATEGIC_DECISION query type."""
        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None
        assert 0.0 <= result.relevance_score <= 1.0
        assert result.query_latency_ms > 0
        assert result.metadata["query_type"] == "strategic_decision"

        # Validate strategic context structure
        ctx = result.strategic_context
        assert len(ctx.relevant_stage_summaries) >= 3
        assert len(ctx.critical_facts) >= 5
        assert len(ctx.error_patterns) >= 1
        assert len(ctx.successful_patterns) >= 2
        assert 0.7 <= ctx.context_health_score <= 0.9  # Healthy for decisions

    @pytest.mark.asyncio
    async def test_error_analysis_query(self, interface, agent_id, task_id):
        """Test ERROR_ANALYSIS query type."""
        result = await interface.get_strategic_context(
            query_type=QueryType.ERROR_ANALYSIS,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None
        assert 0.0 <= result.relevance_score <= 1.0
        assert result.query_latency_ms > 0
        assert result.metadata["query_type"] == "error_analysis"

        # Validate strategic context structure
        ctx = result.strategic_context
        assert len(ctx.relevant_stage_summaries) >= 2
        assert len(ctx.critical_facts) >= 3
        assert len(ctx.error_patterns) >= 4  # More errors for analysis
        assert len(ctx.successful_patterns) >= 1
        assert 0.4 <= ctx.context_health_score <= 0.7  # Degraded due to errors

    @pytest.mark.asyncio
    async def test_capability_evaluation_query(self, interface, agent_id, task_id):
        """Test CAPABILITY_EVALUATION query type."""
        result = await interface.get_strategic_context(
            query_type=QueryType.CAPABILITY_EVALUATION,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None
        assert 0.0 <= result.relevance_score <= 1.0
        assert result.query_latency_ms > 0
        assert result.metadata["query_type"] == "capability_evaluation"

        # Validate strategic context structure
        ctx = result.strategic_context
        assert len(ctx.relevant_stage_summaries) >= 2
        assert len(ctx.critical_facts) >= 5
        assert len(ctx.error_patterns) >= 1
        assert len(ctx.successful_patterns) >= 3
        assert 0.6 <= ctx.context_health_score <= 0.85  # Moderate health

    @pytest.mark.asyncio
    async def test_context_refresh_query(self, interface, agent_id, task_id):
        """Test CONTEXT_REFRESH query type."""
        result = await interface.get_strategic_context(
            query_type=QueryType.CONTEXT_REFRESH,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None
        assert 0.0 <= result.relevance_score <= 1.0
        assert result.query_latency_ms > 0
        assert result.metadata["query_type"] == "context_refresh"

        # Validate strategic context structure
        ctx = result.strategic_context
        assert len(ctx.relevant_stage_summaries) >= 4  # Comprehensive
        assert len(ctx.critical_facts) >= 8  # Many facts after refresh
        assert len(ctx.error_patterns) <= 2  # Minimal errors
        assert len(ctx.successful_patterns) >= 4
        assert 0.8 <= ctx.context_health_score <= 1.0  # High after refresh (can be near perfect)

    @pytest.mark.asyncio
    async def test_query_with_context(self, interface, agent_id, task_id):
        """Test query with additional context."""
        context = {"intervention_type": "replan", "confidence": 0.85}

        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
            context=context,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None

    @pytest.mark.asyncio
    async def test_query_empty_agent_id_raises_error(self, interface, task_id):
        """Test query with empty agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            await interface.get_strategic_context(
                query_type=QueryType.STRATEGIC_DECISION,
                agent_id="",
                task_id=task_id,
            )

    @pytest.mark.asyncio
    async def test_query_whitespace_agent_id_raises_error(self, interface, task_id):
        """Test query with whitespace agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            await interface.get_strategic_context(
                query_type=QueryType.STRATEGIC_DECISION,
                agent_id="   ",
                task_id=task_id,
            )

    @pytest.mark.asyncio
    async def test_query_none_context_uses_empty_dict(self, interface, agent_id, task_id):
        """Test query with None context uses empty dict."""
        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
            context=None,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None


class TestQueryLatency:
    """Test query latency performance."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface with deterministic seed."""
        return ACEMemoryInterface(seed=42)

    @pytest.mark.asyncio
    async def test_query_latency_under_150ms_p95(self, interface):
        """Test query latency is <150ms (p95)."""
        latencies = []
        agent_id = "agent-001"
        task_id = uuid4()

        # Run 100 queries
        for _ in range(100):
            start = time.perf_counter()
            result = await interface.get_strategic_context(
                query_type=QueryType.STRATEGIC_DECISION,
                agent_id=agent_id,
                task_id=task_id,
            )
            end = time.perf_counter()
            latency_ms = int((end - start) * 1000)
            latencies.append(latency_ms)

        # Check p95 latency
        latencies.sort()
        p95_latency = latencies[94]  # 95th percentile (0-indexed)
        assert p95_latency < 150, f"P95 latency {p95_latency}ms exceeds 150ms target"

    @pytest.mark.asyncio
    async def test_query_latency_reported_correctly(self, interface):
        """Test reported latency is positive and realistic."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Reported latency should be within realistic range (50-150ms due to simulation)
        assert 50 <= result.query_latency_ms <= 150


class TestHealthScoreComputation:
    """Test context health score computation."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface instance."""
        return ACEMemoryInterface()

    def test_health_score_high_freshness_high_facts_low_errors(self, interface):
        """Test health score with optimal conditions."""
        health_score = interface._compute_health_score(
            stage_summaries=["summary1", "summary2"],
            critical_facts=["fact1", "fact2", "fact3", "fact4", "fact5", "fact6", "fact7"],
            error_patterns=[],
            successful_patterns=["pattern1", "pattern2", "pattern3"],
            freshness=0.95,
        )

        assert 0.85 <= health_score <= 1.0

    def test_health_score_low_freshness_low_facts_high_errors(self, interface):
        """Test health score with poor conditions."""
        health_score = interface._compute_health_score(
            stage_summaries=["summary1"],
            critical_facts=["fact1", "fact2"],
            error_patterns=["error1", "error2", "error3"],
            successful_patterns=["pattern1"],
            freshness=0.30,
        )

        assert 0.0 <= health_score <= 0.5

    def test_health_score_no_patterns_no_errors(self, interface):
        """Test health score with no patterns or errors."""
        health_score = interface._compute_health_score(
            stage_summaries=["summary1"],
            critical_facts=["fact1", "fact2", "fact3", "fact4"],
            error_patterns=[],
            successful_patterns=[],
            freshness=0.80,
        )

        # With no patterns, error_density is 0, so health should be decent
        assert 0.5 <= health_score <= 0.85

    def test_health_score_medium_conditions(self, interface):
        """Test health score with medium conditions."""
        health_score = interface._compute_health_score(
            stage_summaries=["summary1", "summary2"],
            critical_facts=["fact1", "fact2", "fact3", "fact4", "fact5"],
            error_patterns=["error1"],
            successful_patterns=["pattern1", "pattern2"],
            freshness=0.70,
        )

        assert 0.5 <= health_score <= 0.8

    def test_health_score_clamped_to_zero(self, interface):
        """Test health score is clamped to 0 for extreme negative conditions."""
        health_score = interface._compute_health_score(
            stage_summaries=[],
            critical_facts=[],
            error_patterns=["error1", "error2", "error3", "error4", "error5"],
            successful_patterns=[],
            freshness=0.0,
        )

        assert health_score >= 0.0

    def test_health_score_clamped_to_one(self, interface):
        """Test health score is clamped to 1 for extreme positive conditions."""
        health_score = interface._compute_health_score(
            stage_summaries=["s1", "s2", "s3"],
            critical_facts=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            error_patterns=[],
            successful_patterns=["p1", "p2", "p3", "p4", "p5"],
            freshness=1.0,
        )

        assert health_score <= 1.0


class TestRelevanceScoring:
    """Test relevance score computation."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface instance."""
        return ACEMemoryInterface()

    @pytest.mark.asyncio
    async def test_error_analysis_higher_relevance_with_errors(self, interface):
        """Test ERROR_ANALYSIS has higher relevance with many error patterns."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.ERROR_ANALYSIS,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Error analysis should have high relevance when errors present
        assert result.relevance_score >= 0.5

    @pytest.mark.asyncio
    async def test_strategic_decision_relevance_with_facts(self, interface):
        """Test STRATEGIC_DECISION has higher relevance with many facts."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Strategic decision should have high relevance with abundant facts
        assert result.relevance_score >= 0.7

    @pytest.mark.asyncio
    async def test_capability_evaluation_relevance_with_patterns(self, interface):
        """Test CAPABILITY_EVALUATION has higher relevance with patterns."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.CAPABILITY_EVALUATION,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Capability evaluation should have high relevance with patterns
        assert result.relevance_score >= 0.6

    @pytest.mark.asyncio
    async def test_context_refresh_relevance_comprehensive(self, interface):
        """Test CONTEXT_REFRESH has higher relevance when comprehensive."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.CONTEXT_REFRESH,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Context refresh should have high relevance when comprehensive
        assert result.relevance_score >= 0.8


class TestTokenEstimation:
    """Test token count estimation."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface instance."""
        return ACEMemoryInterface()

    @pytest.mark.asyncio
    async def test_token_count_in_metadata(self, interface):
        """Test token count is included in metadata."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert "token_count" in result.metadata
        assert result.metadata["token_count"] > 0

    @pytest.mark.asyncio
    async def test_token_count_increases_with_content(self, interface):
        """Test token count increases with more content."""
        agent_id = "agent-001"
        task_id = uuid4()

        # Context refresh has more content than error analysis
        refresh_result = await interface.get_strategic_context(
            query_type=QueryType.CONTEXT_REFRESH,
            agent_id=agent_id,
            task_id=task_id,
        )

        error_result = await interface.get_strategic_context(
            query_type=QueryType.ERROR_ANALYSIS,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Refresh should have more tokens due to comprehensive content
        assert refresh_result.metadata["token_count"] > error_result.metadata["token_count"]

    def test_estimate_token_count_empty_context(self, interface):
        """Test token estimation with empty context."""
        from agentcore.ace.models.ace_models import StrategicContext

        ctx = StrategicContext(
            relevant_stage_summaries=[],
            critical_facts=[],
            error_patterns=[],
            successful_patterns=[],
            context_health_score=0.5,
        )

        token_count = interface._estimate_token_count(ctx)
        assert token_count == 0

    def test_estimate_token_count_non_empty_context(self, interface):
        """Test token estimation with non-empty context."""
        from agentcore.ace.models.ace_models import StrategicContext

        ctx = StrategicContext(
            relevant_stage_summaries=["This is a summary"],
            critical_facts=["This is a fact"],
            error_patterns=["This is an error"],
            successful_patterns=["This is a success"],
            context_health_score=0.75,
        )

        token_count = interface._estimate_token_count(ctx)
        assert token_count > 0


class TestConcurrentQueries:
    """Test concurrent query execution."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface instance."""
        return ACEMemoryInterface()

    @pytest.mark.asyncio
    async def test_concurrent_queries_succeed(self, interface):
        """Test multiple concurrent queries complete successfully."""
        import asyncio

        agent_id = "agent-001"
        task_ids = [uuid4() for _ in range(10)]

        # Execute 10 concurrent queries
        tasks = [
            interface.get_strategic_context(
                query_type=QueryType.STRATEGIC_DECISION,
                agent_id=agent_id,
                task_id=task_id,
            )
            for task_id in task_ids
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert result.query_id is not None
            assert result.strategic_context is not None
            assert 0.0 <= result.relevance_score <= 1.0


class TestDeterministicMockData:
    """Test deterministic mock data generation with seed."""

    @pytest.mark.asyncio
    async def test_same_seed_same_results(self):
        """Test same seed produces identical results."""
        agent_id = "agent-001"
        task_id = uuid4()

        interface1 = ACEMemoryInterface(seed=42)
        interface2 = ACEMemoryInterface(seed=42)

        result1 = await interface1.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        result2 = await interface2.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Latency should be identical with same seed
        assert result1.query_latency_ms == result2.query_latency_ms

    @pytest.mark.asyncio
    async def test_different_seed_different_results(self):
        """Test different seeds produce different results."""
        agent_id = "agent-001"
        task_id = uuid4()

        interface1 = ACEMemoryInterface(seed=42)
        interface2 = ACEMemoryInterface(seed=99)

        result1 = await interface1.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        result2 = await interface2.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        # Latency should differ with different seeds
        assert result1.query_latency_ms != result2.query_latency_ms


class TestMetadataFields:
    """Test metadata field population."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface instance."""
        return ACEMemoryInterface()

    @pytest.mark.asyncio
    async def test_metadata_contains_source(self, interface):
        """Test metadata contains source field."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert "source" in result.metadata
        assert result.metadata["source"] == "mock_mem_generator"

    @pytest.mark.asyncio
    async def test_metadata_contains_query_type(self, interface):
        """Test metadata contains query_type field."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.ERROR_ANALYSIS,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert "query_type" in result.metadata
        assert result.metadata["query_type"] == "error_analysis"

    @pytest.mark.asyncio
    async def test_metadata_contains_token_count(self, interface):
        """Test metadata contains token_count field."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=QueryType.STRATEGIC_DECISION,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert "token_count" in result.metadata
        assert isinstance(result.metadata["token_count"], int)
        assert result.metadata["token_count"] > 0


class TestAllQueryTypes:
    """Test all query types return valid results."""

    @pytest.fixture
    def interface(self):
        """ACEMemoryInterface instance."""
        return ACEMemoryInterface()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_type",
        [
            QueryType.STRATEGIC_DECISION,
            QueryType.ERROR_ANALYSIS,
            QueryType.CAPABILITY_EVALUATION,
            QueryType.CONTEXT_REFRESH,
        ],
    )
    async def test_query_type_returns_valid_result(self, interface, query_type):
        """Test each query type returns valid MemoryQueryResult."""
        agent_id = "agent-001"
        task_id = uuid4()

        result = await interface.get_strategic_context(
            query_type=query_type,
            agent_id=agent_id,
            task_id=task_id,
        )

        assert result.query_id is not None
        assert result.strategic_context is not None
        assert 0.0 <= result.relevance_score <= 1.0
        assert result.query_latency_ms > 0
        assert 0.0 <= result.strategic_context.context_health_score <= 1.0
        assert len(result.strategic_context.relevant_stage_summaries) > 0
        assert len(result.strategic_context.critical_facts) > 0
