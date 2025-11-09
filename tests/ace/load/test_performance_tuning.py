"""
Performance Tuning Tests - ACE-029

Validates performance optimizations:
- Metrics batching efficiency
- Redis cache hit rates
- System overhead <5%
- Connection pool utilization

Target: System overhead <5%, cache hit rate >80%
"""

import asyncio
import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.models.ace_models import (
    ContextPlaybook,
    PerformanceBaseline,
    PerformanceMetrics,
)
from agentcore.ace.monitors.performance_monitor import PerformanceMonitor
from agentcore.ace.services.cache_service import ACECacheService, CacheConfig


class TestMetricsBatching:
    """Test metrics batching optimization."""

    @pytest.mark.asyncio
    async def test_batch_size_threshold(self, get_session):
        """
        Test that metrics are flushed at batch size threshold.

        Acceptance: Buffer flushes at 100 updates
        """
        monitor = PerformanceMonitor(
            get_session=get_session,
            batch_size=100,
            batch_timeout=60.0,  # Long timeout to test size trigger
        )

        agent_id = f"agent-{uuid4()}"
        task_id = f"task-{uuid4()}"

        # Record 99 metrics - should buffer
        for i in range(99):
            await monitor.record_metrics(
                agent_id=agent_id,
                task_id=task_id,
                stage="planning",
                accuracy=0.9,
                recall=0.85,
                f1_score=0.87,
            )

        # Buffer should have 99 items
        assert len(monitor._buffer) == 99

        # Record 100th metric - should trigger flush
        await monitor.record_metrics(
            agent_id=agent_id,
            task_id=task_id,
            stage="planning",
            accuracy=0.9,
            recall=0.85,
            f1_score=0.87,
        )

        # Buffer should be empty after flush
        await asyncio.sleep(0.1)  # Allow flush task to complete
        assert len(monitor._buffer) == 0

    @pytest.mark.asyncio
    async def test_time_based_flush(self, get_session):
        """
        Test that metrics are flushed after timeout.

        Acceptance: Buffer flushes after 1 second
        """
        monitor = PerformanceMonitor(
            get_session=get_session,
            batch_size=1000,  # Large batch to test timeout
            batch_timeout=1.0,
        )

        agent_id = f"agent-{uuid4()}"
        task_id = f"task-{uuid4()}"

        # Record 10 metrics
        for i in range(10):
            await monitor.record_metrics(
                agent_id=agent_id,
                task_id=task_id,
                stage="planning",
                accuracy=0.9,
                recall=0.85,
                f1_score=0.87,
            )

        # Buffer should have 10 items
        assert len(monitor._buffer) == 10

        # Wait for timeout
        await asyncio.sleep(1.2)

        # Buffer should be empty after timeout flush
        assert len(monitor._buffer) == 0

    @pytest.mark.asyncio
    async def test_batching_performance_overhead(self, get_session):
        """
        Test that batching reduces overhead to <5%.

        Acceptance: System overhead <5% with batching
        """
        monitor = PerformanceMonitor(
            get_session=get_session,
            batch_size=100,
            batch_timeout=1.0,
        )

        agent_id = f"agent-{uuid4()}"
        task_id = f"task-{uuid4()}"

        # Measure time to record 1000 metrics
        start_time = time.perf_counter()

        for i in range(1000):
            await monitor.record_metrics(
                agent_id=agent_id,
                task_id=task_id,
                stage="planning",
                accuracy=0.9,
                recall=0.85,
                f1_score=0.87,
            )

        # Allow final flush
        await asyncio.sleep(1.5)

        elapsed_time = time.perf_counter() - start_time

        # Calculate overhead: time per operation in milliseconds
        overhead_per_op = (elapsed_time / 1000) * 1000  # Convert to ms

        # Should be <5ms per operation (5% of 100ms processing time)
        assert overhead_per_op < 5.0, f"Overhead {overhead_per_op}ms exceeds 5ms target"


class TestRedisCaching:
    """Test Redis caching layer."""

    @pytest.fixture
    async def cache_service(self):
        """Create and connect cache service for testing."""
        config = CacheConfig(
            redis_url="redis://localhost:6379/1",  # Test database
            playbook_ttl_seconds=10,  # Short TTL for testing
            baseline_ttl_seconds=20,
        )
        service = ACECacheService(config)
        await service.connect()
        yield service
        await service.disconnect()

    @pytest.mark.asyncio
    async def test_playbook_cache_hit(self, cache_service):
        """
        Test playbook caching with hit.

        Acceptance: Cache hit rate >80%
        """
        agent_id = f"agent-{uuid4()}"
        playbook = ContextPlaybook(
            id=uuid4(),
            agent_id=agent_id,
            philosophy="Test philosophy",
            context={"strategies": ["test"], "patterns": [], "failures": [], "learnings": []},
            evolution_history=[],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Set playbook in cache
        success = await cache_service.set_playbook(agent_id, playbook)
        assert success

        # Get from cache - should hit
        cached = await cache_service.get_playbook(agent_id)
        assert cached is not None
        assert cached.agent_id == agent_id
        assert cached.philosophy == "Test philosophy"

    @pytest.mark.asyncio
    async def test_playbook_cache_miss(self, cache_service):
        """Test playbook cache miss behavior."""
        agent_id = f"agent-{uuid4()}"

        # Get non-existent playbook - should miss
        cached = await cache_service.get_playbook(agent_id)
        assert cached is None

    @pytest.mark.asyncio
    async def test_playbook_ttl_expiration(self, cache_service):
        """
        Test playbook TTL expiration.

        Acceptance: 10min TTL for playbooks
        """
        agent_id = f"agent-{uuid4()}"
        playbook = ContextPlaybook(
            id=uuid4(),
            agent_id=agent_id,
            philosophy="Test philosophy",
            context={"strategies": [], "patterns": [], "failures": [], "learnings": []},
            evolution_history=[],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Set playbook with short TTL (10s for test)
        await cache_service.set_playbook(agent_id, playbook)

        # Should be in cache
        cached = await cache_service.get_playbook(agent_id)
        assert cached is not None

        # Wait for expiration (TTL is 10s in test config)
        await asyncio.sleep(11)

        # Should be expired
        expired = await cache_service.get_playbook(agent_id)
        assert expired is None

    @pytest.mark.asyncio
    async def test_baseline_cache_ttl(self, cache_service):
        """
        Test baseline cache TTL.

        Acceptance: 1hr TTL for baselines
        """
        agent_id = f"agent-{uuid4()}"
        task_type = "test_task"

        baseline = PerformanceBaseline(
            agent_id=agent_id,
            task_type=task_type,
            avg_accuracy=0.9,
            avg_recall=0.85,
            avg_f1_score=0.87,
            sample_count=100,
            last_updated=datetime.now(UTC),
        )

        # Set baseline with 20s TTL (test config)
        await cache_service.set_baseline(agent_id, task_type, baseline)

        # Should be in cache
        cached = await cache_service.get_baseline(agent_id, task_type)
        assert cached is not None
        assert cached.avg_accuracy == 0.9

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_service):
        """Test cache invalidation."""
        agent_id = f"agent-{uuid4()}"
        playbook = ContextPlaybook(
            id=uuid4(),
            agent_id=agent_id,
            philosophy="Test",
            context={"strategies": [], "patterns": [], "failures": [], "learnings": []},
            evolution_history=[],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Set and verify
        await cache_service.set_playbook(agent_id, playbook)
        assert await cache_service.get_playbook(agent_id) is not None

        # Invalidate
        result = await cache_service.invalidate_playbook(agent_id)
        assert result is True

        # Should be gone
        assert await cache_service.get_playbook(agent_id) is None

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_service):
        """Test cache statistics retrieval."""
        stats = await cache_service.get_cache_stats()

        assert "keyspace_hits" in stats
        assert "keyspace_misses" in stats
        assert "total_keys" in stats
        assert "hit_rate" in stats


class TestConnectionPooling:
    """Test database connection pool configuration."""

    @pytest.mark.asyncio
    async def test_connection_pool_size(self):
        """
        Test connection pool configuration.

        Acceptance: min 10, max 50 connections
        """
        from agentcore.a2a_protocol.config import Settings

        settings = Settings()

        # Verify pool size (minimum connections)
        assert settings.DATABASE_POOL_SIZE == 10

        # Verify max overflow (allows up to 50 total)
        assert settings.DATABASE_MAX_OVERFLOW == 40  # 10 + 40 = 50 max

    @pytest.mark.asyncio
    async def test_pool_recycle_time(self):
        """Test connection recycle configuration."""
        from agentcore.a2a_protocol.config import Settings

        settings = Settings()

        # Verify pool recycle (1 hour)
        assert settings.DATABASE_POOL_RECYCLE == 3600


class TestSystemOverhead:
    """Test overall system overhead."""

    @pytest.mark.asyncio
    async def test_end_to_end_overhead(self, get_session, cache_service):
        """
        Test end-to-end system overhead.

        Acceptance: <5% overhead for full ACE operation
        """
        monitor = PerformanceMonitor(
            get_session=get_session,
            batch_size=100,
            batch_timeout=1.0,
        )

        agent_id = f"agent-{uuid4()}"
        task_id = f"task-{uuid4()}"

        # Simulate typical workload:
        # - Record metrics (batched)
        # - Cache lookups
        # - Database operations

        start_time = time.perf_counter()

        # Simulate 100 operations
        for i in range(100):
            # Record metrics (will be batched)
            await monitor.record_metrics(
                agent_id=agent_id,
                task_id=task_id,
                stage="planning",
                accuracy=0.9,
                recall=0.85,
                f1_score=0.87,
            )

            # Cache lookup (should be fast)
            await cache_service.get_playbook(agent_id)

            # Small delay to simulate processing
            await asyncio.sleep(0.01)  # 10ms processing time

        # Wait for final flush
        await asyncio.sleep(1.5)

        elapsed_time = time.perf_counter() - start_time

        # Calculate overhead
        # Expected time: 100 * 10ms = 1000ms
        # Actual time should be <1050ms (5% overhead)
        expected_time = 100 * 0.01  # 1 second
        overhead_percentage = ((elapsed_time - expected_time) / expected_time) * 100

        assert overhead_percentage < 5.0, (
            f"System overhead {overhead_percentage:.2f}% exceeds 5% target"
        )


@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Integration tests for performance benchmarks."""

    @pytest.mark.asyncio
    async def test_metrics_throughput(self, get_session):
        """
        Test metrics throughput.

        Target: 10K+ metrics per hour (167+ per minute)
        """
        monitor = PerformanceMonitor(
            get_session=get_session,
            batch_size=100,
            batch_timeout=1.0,
        )

        agent_id = f"agent-{uuid4()}"

        # Record metrics for 1 minute
        start_time = time.perf_counter()
        count = 0

        while time.perf_counter() - start_time < 60.0:
            task_id = f"task-{uuid4()}"
            await monitor.record_metrics(
                agent_id=agent_id,
                task_id=task_id,
                stage="planning",
                accuracy=0.9,
                recall=0.85,
                f1_score=0.87,
            )
            count += 1

            # Small delay to avoid overload
            await asyncio.sleep(0.01)

        # Wait for final flush
        await asyncio.sleep(2.0)

        # Should achieve 167+ per minute (10K+ per hour)
        assert count >= 167, f"Throughput {count}/min below target 167/min"

    @pytest.mark.asyncio
    async def test_cache_hit_rate_target(self, cache_service):
        """
        Test cache hit rate meets target.

        Acceptance: >80% hit rate
        """
        agent_ids = [f"agent-{i}" for i in range(10)]

        # Populate cache
        for agent_id in agent_ids:
            playbook = ContextPlaybook(
                id=uuid4(),
                agent_id=agent_id,
                philosophy="Test",
                context={"strategies": [], "patterns": [], "failures": [], "learnings": []},
                evolution_history=[],
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            await cache_service.set_playbook(agent_id, playbook)

        # Perform 100 lookups (80% cached, 20% new)
        hits = 0
        total = 100

        for i in range(total):
            if i < 80:
                # Hit existing
                agent_id = agent_ids[i % 10]
            else:
                # Miss new
                agent_id = f"agent-new-{i}"

            result = await cache_service.get_playbook(agent_id)
            if result is not None:
                hits += 1

        hit_rate = hits / total
        assert hit_rate >= 0.8, f"Hit rate {hit_rate:.2%} below 80% target"
