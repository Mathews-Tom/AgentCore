"""Performance benchmarks for Integration Layer components.

Validates:
- Provider latency optimization
- Resource utilization optimization
- Scalability validation
- Memory efficiency
- CPU usage optimization
"""

import asyncio
import gc
import os
import time
from typing import Any
from uuid import uuid4

import psutil
import pytest
from memory_profiler import memory_usage

from agentcore.integration.webhook import (
    DeliveryService,
    EventPayload,
    EventPublisher,
    WebhookConfig,
    WebhookEvent,
    WebhookManager,
    WebhookRegistration,
)


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None
        self.baseline_cpu = None

    def measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def measure_cpu(self) -> float:
        """Measure CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)

    def start_measurement(self):
        """Start performance measurement."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.measure_memory()
        self.baseline_cpu = self.measure_cpu()

    def end_measurement(self) -> dict[str, Any]:
        """End measurement and return metrics."""
        final_memory = self.measure_memory()
        final_cpu = self.measure_cpu()

        return {
            "memory_baseline_mb": self.baseline_memory,
            "memory_final_mb": final_memory,
            "memory_delta_mb": final_memory - self.baseline_memory,
            "cpu_baseline_pct": self.baseline_cpu,
            "cpu_final_pct": final_cpu,
            "cpu_delta_pct": final_cpu - self.baseline_cpu,
        }


@pytest.fixture
def webhook_config():
    """Performance-optimized webhook configuration."""
    return WebhookConfig(
        default_max_retries=1,
        default_retry_delay_seconds=1,
        max_concurrent_deliveries=1000,  # High concurrency
        event_queue_size=50000,  # Large queue
        event_batch_size=500,  # Large batches
    )


@pytest.fixture
async def webhook_manager(webhook_config):
    """Create webhook manager."""
    return WebhookManager(config=webhook_config)


@pytest.fixture
async def event_publisher(webhook_config):
    """Create event publisher."""
    publisher = EventPublisher(config=webhook_config)
    await publisher.start()
    yield publisher
    await publisher.stop()


@pytest.fixture
async def delivery_service(webhook_config):
    """Create delivery service."""
    service = DeliveryService(config=webhook_config)
    yield service
    await service.close()


class TestWebhookManagerPerformance:
    """Performance benchmarks for WebhookManager."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_high_volume_registration(self, webhook_manager, benchmark):
        """Benchmark: Register 10,000 webhooks."""

        async def register_webhooks():
            webhooks = []
            for i in range(10000):
                webhook = await webhook_manager.register(
                    name=f"Webhook {i}",
                    url=f"https://api.example.com/webhook-{i}",
                    events=[WebhookEvent.TASK_COMPLETED],
                )
                webhooks.append(webhook)
            return webhooks

        perf = PerformanceBenchmark()
        perf.start_measurement()

        start_time = time.time()
        webhooks = await register_webhooks()
        duration = time.time() - start_time

        metrics = perf.end_measurement()

        print(f"\n=== Webhook Registration Performance ===")
        print(f"Webhooks Registered: {len(webhooks):,}")
        print(f"Total Time: {duration:.2f}s")
        print(f"Rate: {len(webhooks) / duration:.0f} webhooks/sec")
        print(f"Memory Delta: {metrics['memory_delta_mb']:.2f} MB")
        print(f"CPU Usage: {metrics['cpu_final_pct']:.1f}%")

        # Assertions
        assert len(webhooks) == 10000
        assert duration < 10  # Should complete in < 10 seconds
        assert metrics["memory_delta_mb"] < 200  # < 200 MB memory growth

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_list_operations(self, webhook_manager):
        """Benchmark: Concurrent webhook listing."""
        # Pre-populate webhooks
        for i in range(1000):
            await webhook_manager.register(
                name=f"Webhook {i}",
                url=f"https://api.example.com/webhook-{i}",
                events=[WebhookEvent.TASK_COMPLETED],
            )

        async def concurrent_lists():
            tasks = [webhook_manager.list() for _ in range(1000)]
            return await asyncio.gather(*tasks)

        perf = PerformanceBenchmark()
        perf.start_measurement()

        start_time = time.time()
        results = await concurrent_lists()
        duration = time.time() - start_time

        metrics = perf.end_measurement()

        print(f"\n=== Concurrent List Performance ===")
        print(f"Concurrent Requests: {len(results):,}")
        print(f"Total Time: {duration:.2f}s")
        print(f"Rate: {len(results) / duration:.0f} req/sec")
        print(f"CPU Usage: {metrics['cpu_final_pct']:.1f}%")

        # Assertions
        assert len(results) == 1000
        assert duration < 5  # Should complete in < 5 seconds


class TestEventPublisherPerformance:
    """Performance benchmarks for EventPublisher."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_high_throughput_publishing(self, event_publisher):
        """Benchmark: Publish 50,000 events."""

        async def publish_events():
            tasks = []
            for i in range(50000):
                task = event_publisher.publish(
                    event_type=WebhookEvent.TASK_COMPLETED,
                    data={"task_id": f"task-{i}", "status": "completed"},
                    source="benchmark",
                )
                tasks.append(task)

            return await asyncio.gather(*tasks)

        perf = PerformanceBenchmark()
        perf.start_measurement()

        start_time = time.time()
        events = await publish_events()
        duration = time.time() - start_time

        metrics = perf.end_measurement()

        print(f"\n=== Event Publishing Performance ===")
        print(f"Events Published: {len(events):,}")
        print(f"Total Time: {duration:.2f}s")
        print(f"Rate: {len(events) / duration:.0f} events/sec")
        print(f"Memory Delta: {metrics['memory_delta_mb']:.2f} MB")
        print(f"Queue Size: {await event_publisher.get_queue_size()}")

        # Assertions
        assert len(events) == 50000
        assert duration < 30  # Should complete in < 30 seconds
        assert len(events) / duration > 1000  # > 1,000 events/sec

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_subscription_scalability(self, event_publisher):
        """Benchmark: Handle 10,000 subscriptions."""
        subscriptions = []

        start_time = time.time()
        for i in range(10000):
            sub = await event_publisher.subscribe(
                webhook_id=uuid4(),
                event_types=[WebhookEvent.TASK_COMPLETED],
            )
            subscriptions.append(sub)

        duration = time.time() - start_time

        print(f"\n=== Subscription Scalability ===")
        print(f"Subscriptions Created: {len(subscriptions):,}")
        print(f"Total Time: {duration:.2f}s")
        print(f"Rate: {len(subscriptions) / duration:.0f} subs/sec")

        # Assertions
        assert len(subscriptions) == 10000
        assert duration < 10  # Should complete in < 10 seconds


class TestDeliveryServicePerformance:
    """Performance benchmarks for DeliveryService."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_delivery_scheduling(self, delivery_service):
        """Benchmark: Schedule 5,000 deliveries concurrently."""
        webhook = WebhookRegistration(
            name="Benchmark Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="a" * 32,
        )

        events = [
            EventPayload(
                event_type=WebhookEvent.TASK_COMPLETED,
                data={"task_id": f"task-{i}"},
                source="benchmark",
            )
            for i in range(5000)
        ]

        perf = PerformanceBenchmark()
        perf.start_measurement()

        start_time = time.time()
        deliveries = await asyncio.gather(
            *[delivery_service.schedule(webhook, event) for event in events]
        )
        duration = time.time() - start_time

        metrics = perf.end_measurement()

        print(f"\n=== Delivery Scheduling Performance ===")
        print(f"Deliveries Scheduled: {len(deliveries):,}")
        print(f"Total Time: {duration:.2f}s")
        print(f"Rate: {len(deliveries) / duration:.0f} deliveries/sec")
        print(f"Memory Delta: {metrics['memory_delta_mb']:.2f} MB")

        # Assertions
        assert len(deliveries) == 5000
        assert duration < 10  # Should complete in < 10 seconds


class TestResourceUtilization:
    """Resource utilization optimization tests."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_efficiency_under_load(self, webhook_manager, event_publisher):
        """Benchmark: Memory efficiency with 100K operations."""

        def run_operations():
            async def operations():
                # Create webhooks
                webhooks = []
                for i in range(1000):
                    webhook = await webhook_manager.register(
                        name=f"Webhook {i}",
                        url=f"https://api.example.com/webhook-{i}",
                        events=[WebhookEvent.TASK_COMPLETED],
                    )
                    webhooks.append(webhook)

                # Publish events
                events = []
                for i in range(10000):
                    event = await event_publisher.publish(
                        event_type=WebhookEvent.TASK_COMPLETED,
                        data={"task_id": f"task-{i}"},
                        source="benchmark",
                    )
                    events.append(event)

                return len(webhooks), len(events)

            return asyncio.run(operations())

        # Measure memory usage
        mem_usage = memory_usage(run_operations, interval=0.1, max_usage=True)

        print(f"\n=== Memory Efficiency Test ===")
        print(f"Peak Memory Usage: {mem_usage:.2f} MB")

        # Assertions
        assert mem_usage < 500  # Peak memory < 500 MB

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cpu_efficiency_under_load(self, webhook_manager):
        """Benchmark: CPU efficiency with sustained load."""
        perf = PerformanceBenchmark()
        perf.start_measurement()

        # Sustained operations for 10 seconds
        start_time = time.time()
        operation_count = 0

        while time.time() - start_time < 10:
            await webhook_manager.register(
                name=f"Webhook {operation_count}",
                url=f"https://api.example.com/webhook-{operation_count}",
                events=[WebhookEvent.TASK_COMPLETED],
            )
            operation_count += 1

            # Brief yield to allow CPU measurement
            if operation_count % 100 == 0:
                await asyncio.sleep(0.01)

        metrics = perf.end_measurement()

        print(f"\n=== CPU Efficiency Test ===")
        print(f"Operations: {operation_count:,}")
        print(f"Duration: 10s")
        print(f"Rate: {operation_count / 10:.0f} ops/sec")
        print(f"Avg CPU Usage: {metrics['cpu_final_pct']:.1f}%")

        # Assertions
        assert metrics["cpu_final_pct"] < 80  # CPU usage < 80%


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-m", "benchmark", "-s"])
