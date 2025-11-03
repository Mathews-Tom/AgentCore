# Performance Optimization System

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Caching Strategies](#caching-strategies)
5. [Container Pooling](#container-pooling)
6. [Resource Prediction](#resource-prediction)
7. [Memory Optimization](#memory-optimization)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [Testing](#testing)

## Overview

The Performance Optimization System provides comprehensive performance enhancements for agent runtime operations including:

- **High-performance caching** with TTL and LRU eviction
- **Container pooling** for fast agent startup (warm starts)
- **Resource prediction** based on historical usage patterns
- **Memory optimization** via garbage collection tuning
- **Background optimization** with automated cleanup

### Key Features

- **92% Test Coverage**: Comprehensive test suite with 32 test scenarios
- **Configurable Components**: Enable/disable caching, pooling, and GC optimization independently
- **Performance Metrics**: Detailed tracking of cache hits, warm/cold starts, memory usage
- **Weak References**: Memory-efficient agent tracking prevents leaks
- **Async-First**: All operations designed for asyncio

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  PerformanceOptimizer (Main Service)                        │
│  ├─ Configuration: caching, pooling, gc optimization        │
│  ├─ Global instance: get_performance_optimizer()            │
│  └─ Background optimization loop (5 min interval)           │
└─────────────────────────────────────────────────────────────┘
           │
           ├──────────────┬──────────────┬──────────────┐
           │              │              │              │
           ▼              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ PerformanceCache │ │  ContainerPool   │ │ResourcePredictor │ │ Memory Optimizer │
│                  │ │                  │ │                  │ │                  │
│ - TTL cache      │ │ - Warm starts    │ │ - History-based  │ │ - GC collection  │
│ - LRU eviction   │ │ - Pre-warming    │ │ - Philosophy     │ │ - Memory tracking│
│ - Hit/miss stats │ │ - Pool stats     │ │   profiling      │ │ - Auto cleanup   │
│ - Key patterns:  │ │ - Lock-based     │ │ - Confidence     │ │ - psutil metrics │
│   * tool:{id}    │ │   concurrency    │ │   scoring        │ │                  │
│   * pattern:{key}│ │ - Per-philosophy │ │ - 20% buffer     │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Architecture

### Design Principles

1. **Separation of Concerns**: Each component handles specific optimization aspect
2. **Composable**: Components can be enabled/disabled independently
3. **Observable**: Comprehensive metrics for monitoring and tuning
4. **Safe**: Weak references prevent memory leaks, locks prevent races

### Component Interaction

```python
# Main service initialization
optimizer = PerformanceOptimizer(
    enable_caching=True,      # PerformanceCache
    enable_pooling=True,      # ContainerPool
    enable_gc_optimization=True,  # Memory optimization
    cache_size=1000,          # Cache max entries
    pool_size=10,             # Containers per philosophy
)

# Background optimization
await optimizer.start()       # Starts background loop
# ... operations ...
await optimizer.stop()        # Stops background loop
```

## Components

### CacheEntry

**Purpose**: Individual cache entry with TTL and access tracking.

**Key Features**:
- Time-to-live (TTL) expiration
- Access count tracking
- Last accessed timestamp
- Automatic expiration checking

**Implementation**:
```python
class CacheEntry:
    """Cache entry with TTL and access tracking."""

    def __init__(self, value: Any, ttl_seconds: int = 300) -> None:
        self.value = value
        self.created_at = datetime.now(UTC)
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now(UTC) > self.expires_at

    def access(self) -> Any:
        """Access cached value and update metrics."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)
        return self.value
```

**Usage**:
```python
# Create cache entry with 5 minute TTL
entry = CacheEntry("my_value", ttl_seconds=300)

# Access the value
if not entry.is_expired():
    value = entry.access()  # Updates access_count and last_accessed
```

### PerformanceCache

**Purpose**: High-performance cache with TTL and LRU eviction policies.

**Key Features**:
- Maximum size enforcement
- TTL-based expiration
- LRU eviction when full
- Hit/miss statistics
- Configurable default TTL

**Eviction Strategies**:
1. **Expired Entries First**: Remove all expired entries before evicting LRU
2. **LRU Eviction**: If still full, evict least recently used entry
3. **Automatic Cleanup**: Background task cleans expired entries every 5 minutes

**Implementation**:
```python
class PerformanceCache:
    """High-performance cache with TTL and eviction policies."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        entry = self._cache.get(key)
        if entry is None or entry.is_expired():
            self._misses += 1
            return None

        self._hits += 1
        return entry.access()

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with eviction if needed."""
        if len(self._cache) >= self._max_size:
            self._evict_expired()
        if len(self._cache) >= self._max_size:
            self._evict_lru()

        ttl = ttl if ttl is not None else self._default_ttl
        self._cache[key] = CacheEntry(value, ttl)
```

**Cache Statistics**:
```python
stats = cache.get_stats()
# Returns:
# {
#     "size": 150,              # Current entries
#     "max_size": 1000,         # Maximum capacity
#     "hits": 1000,             # Successful retrievals
#     "misses": 50,             # Failed retrievals
#     "hit_rate_percent": 95.24 # Hit rate percentage
# }
```

### ContainerPool

**Purpose**: Pre-warmed container pool for fast agent startup.

**Key Concepts**:
- **Warm Start**: Container from pool (milliseconds startup)
- **Cold Start**: New container creation (seconds startup)
- **Pool Size**: Target number of containers per philosophy
- **Lock-Based Concurrency**: Thread-safe acquire/release

**Implementation**:
```python
class ContainerPool:
    """Pre-warmed container pool for fast agent startup."""

    def __init__(self, pool_size: int = 10, warm_start_enabled: bool = True) -> None:
        self._pool_size = pool_size
        self._warm_start_enabled = warm_start_enabled
        self._pools: dict[AgentPhilosophy, list[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        philosophy: AgentPhilosophy,
        create_fn: Callable[[], str],
    ) -> tuple[str, bool]:
        """Acquire container (warm or cold start)."""
        async with self._lock:
            # Try pool first
            if self._pools[philosophy]:
                container_id = self._pools[philosophy].pop(0)
                return container_id, True  # Warm start

            # Cold start
            container_id = await create_fn()
            return container_id, False

    async def release(
        self,
        philosophy: AgentPhilosophy,
        container_id: str,
    ) -> None:
        """Release container back to pool."""
        async with self._lock:
            if len(self._pools[philosophy]) < self._pool_size:
                self._pools[philosophy].append(container_id)
```

**Pre-Warming**:
```python
# Pre-warm pool with containers
await pool.pre_warm(
    AgentPhilosophy.REACT,
    create_container_fn,
    count=5  # Create 5 containers
)
```

### ResourcePredictor

**Purpose**: Predict resource requirements based on historical usage patterns.

**Key Features**:
- Per-philosophy learning
- Historical sample tracking (default 100 samples)
- Conservative defaults for unknown philosophies
- 20% safety buffer on predictions
- Confidence scoring based on sample count

**Implementation**:
```python
class ResourcePredictor:
    """Predicts resource requirements based on historical usage."""

    def __init__(self, history_size: int = 100) -> None:
        self._history: dict[str, list[dict[str, float]]] = defaultdict(list)
        self._history_size = history_size

    def record_usage(
        self,
        philosophy: str,
        metrics: dict[str, float],
    ) -> None:
        """Record resource usage for learning."""
        history = self._history[philosophy]
        history.append(metrics)

        # Keep only recent history
        if len(history) > self._history_size:
            history.pop(0)

    def predict_requirements(self, philosophy: str) -> dict[str, float]:
        """Predict resource requirements with 20% buffer."""
        history = self._history.get(philosophy, [])

        if not history:
            # Conservative defaults
            return {
                "cpu_percent": 50.0,
                "memory_mb": 256.0,
                "execution_time_seconds": 60.0,
            }

        # Calculate averages with 20% buffer
        avg_cpu = sum(h.get("cpu_percent", 0) for h in history) / len(history)
        avg_memory = sum(h.get("memory_usage_mb", 0) for h in history) / len(history)
        avg_time = sum(h.get("execution_time", 0) for h in history) / len(history)

        return {
            "cpu_percent": round(avg_cpu * 1.2, 2),
            "memory_mb": round(avg_memory * 1.2, 2),
            "execution_time_seconds": round(avg_time * 1.2, 2),
        }
```

**Confidence Scoring**:
```python
confidence = predictor.get_prediction_confidence("react")
# Returns: 0.0 (no data) to 1.0 (full history)
```

### PerformanceOptimizer

**Purpose**: Main performance optimization service integrating all components.

**Initialization**:
```python
optimizer = PerformanceOptimizer(
    enable_caching=True,           # Enable cache optimization
    enable_pooling=True,           # Enable container pooling
    enable_gc_optimization=True,   # Enable memory optimization
    cache_size=1000,               # Max cache entries
    pool_size=10,                  # Containers per philosophy
)
```

**Key Methods**:

#### Container Creation Optimization
```python
container_id, metrics = await optimizer.optimize_container_creation(
    config=agent_config,
    create_fn=create_container_async,
)

# metrics = {
#     "warm_start": True,          # From pool?
#     "startup_time_ms": 15.3      # Milliseconds
# }
```

#### Tool Metadata Caching
```python
# Cache tool metadata (10 minute TTL)
optimizer.cache_tool_metadata("calculator", {
    "name": "Calculator",
    "type": "math",
    "capabilities": ["add", "subtract"],
})

# Retrieve cached metadata
metadata = optimizer.get_cached_tool_metadata("calculator")
```

#### Execution Pattern Caching
```python
# Cache execution patterns (5 minute TTL)
optimizer.cache_execution_pattern("react:basic", {
    "steps": ["think", "act", "observe"],
    "avg_time_ms": 150,
})

# Retrieve cached pattern
pattern = optimizer.get_cached_pattern("react:basic")
```

#### Resource Tracking & Prediction
```python
# Track agent resource usage
optimizer.track_agent_resources(agent_state, {
    "philosophy": "react",
    "cpu_percent": 45.0,
    "memory_usage_mb": 180.0,
    "execution_time": 25.0,
})

# Predict requirements for new agent
requirements = optimizer.predict_resource_requirements(agent_config)
# Returns: {"cpu_percent": 54.0, "memory_mb": 216.0, "execution_time_seconds": 30.0}
```

#### Memory Optimization
```python
result = await optimizer.optimize_memory()
# Returns:
# {
#     "enabled": True,
#     "objects_collected": 142,
#     "memory_before_mb": 512.5,
#     "memory_after_mb": 489.2,
#     "memory_released_mb": 23.3
# }
```

#### Performance Metrics
```python
metrics = optimizer.get_performance_metrics()
# Returns:
# {
#     "optimization_metrics": {
#         "cache_hits": 1000,
#         "cache_misses": 50,
#         "warm_starts": 150,
#         "cold_starts": 10,
#         "gc_collections": 5,
#         "memory_released_mb": 125.5
#     },
#     "cache_stats": {
#         "size": 150,
#         "max_size": 1000,
#         "hits": 1000,
#         "misses": 50,
#         "hit_rate_percent": 95.24
#     },
#     "pool_stats": {
#         "pool_size": 10,
#         "warm_start_enabled": True,
#         "pools": {
#             "react": {"available": 8, "utilization_percent": 80.0},
#             "chain_of_thought": {"available": 5, "utilization_percent": 50.0}
#         }
#     },
#     "warm_start_rate_percent": 93.75
# }
```

## Caching Strategies

### Cache Key Patterns

The optimizer uses structured cache keys:

- **Tool Metadata**: `tool:{tool_id}`
  - TTL: 600 seconds (10 minutes)
  - Purpose: Avoid repeated tool metadata lookups

- **Execution Patterns**: `pattern:{pattern_key}`
  - TTL: 300 seconds (5 minutes)
  - Purpose: Cache common execution workflows

### Cache Tuning

**Cache Size**: Balance between memory usage and hit rate
```python
# Small deployments (< 100 agents)
optimizer = PerformanceOptimizer(cache_size=500)

# Medium deployments (100-1000 agents)
optimizer = PerformanceOptimizer(cache_size=1000)

# Large deployments (> 1000 agents)
optimizer = PerformanceOptimizer(cache_size=5000)
```

**TTL Configuration**: Adjust based on data volatility
```python
# Frequently changing tool metadata
optimizer.cache_tool_metadata("dynamic_tool", metadata, ttl=60)

# Stable tool metadata
optimizer.cache_tool_metadata("stable_tool", metadata, ttl=3600)
```

### Cache Hit Rate Optimization

**Target**: 90%+ hit rate for production workloads

**Strategies**:
1. **Increase cache size** if hit rate < 80%
2. **Adjust TTL** based on data change frequency
3. **Pre-populate** cache for common tools/patterns
4. **Monitor metrics** and tune based on workload

## Container Pooling

### Warm Start vs Cold Start

**Performance Impact**:
- **Warm Start**: 10-50 milliseconds (from pool)
- **Cold Start**: 1-3 seconds (create new container)
- **Speedup**: 20-300x faster

### Pool Size Tuning

**Formula**: `pool_size = peak_concurrent_agents / philosophies + buffer`

**Examples**:
```python
# Low concurrency (< 10 agents)
pool = ContainerPool(pool_size=5)

# Medium concurrency (10-50 agents)
pool = ContainerPool(pool_size=10)

# High concurrency (> 50 agents)
pool = ContainerPool(pool_size=20)
```

### Pre-Warming Strategies

**Application Startup**:
```python
async def warm_pools_on_startup():
    """Pre-warm container pools during application initialization."""
    optimizer = get_performance_optimizer()

    # Pre-warm common philosophies
    for philosophy in [AgentPhilosophy.REACT, AgentPhilosophy.CHAIN_OF_THOUGHT]:
        await optimizer._pool.pre_warm(
            philosophy,
            create_container_for_philosophy,
            count=5
        )
```

**Dynamic Pre-Warming**:
```python
async def monitor_and_prewarm():
    """Monitor pool utilization and pre-warm as needed."""
    while True:
        stats = optimizer._pool.get_pool_stats()

        for philosophy, pool_stats in stats["pools"].items():
            if pool_stats["utilization_percent"] < 20:
                # Pool running low, pre-warm more containers
                await optimizer._pool.pre_warm(
                    AgentPhilosophy[philosophy.upper()],
                    create_container_fn,
                    count=3
                )

        await asyncio.sleep(60)
```

### Pool Monitoring

**Key Metrics**:
- **Warm Start Rate**: Target > 90%
- **Pool Utilization**: Target 50-80%
- **Available Containers**: Always > 0 for active philosophies

```python
stats = optimizer._pool.get_pool_stats()

if stats["pools"]["react"]["available"] == 0:
    logger.warning("REACT pool empty, expect cold starts")

warm_rate = stats.get("warm_start_rate_percent", 0)
if warm_rate < 90:
    logger.warning(f"Low warm start rate: {warm_rate}%")
```

## Resource Prediction

### Learning Process

1. **Record Usage**: Track actual resource consumption after agent execution
2. **Build History**: Maintain per-philosophy historical samples
3. **Calculate Averages**: Compute mean resource requirements
4. **Apply Buffer**: Add 20% safety margin
5. **Return Prediction**: Provide resource allocation recommendation

### Usage Pattern

**Training Phase**:
```python
# After agent execution
optimizer.track_agent_resources(agent_state, {
    "philosophy": "react",
    "cpu_percent": 45.0,
    "memory_usage_mb": 180.0,
    "execution_time": 25.0,
})
```

**Prediction Phase**:
```python
# Before agent creation
requirements = optimizer.predict_resource_requirements(config)

# Use predictions for resource allocation
container_config = {
    "cpu_limit": requirements["cpu_percent"],
    "memory_limit_mb": requirements["memory_mb"],
    "timeout_seconds": requirements["execution_time_seconds"],
}
```

### Confidence-Based Decisions

```python
confidence = optimizer._predictor.get_prediction_confidence("react")

if confidence < 0.3:
    # Low confidence, use conservative defaults
    logger.warning("Low prediction confidence, using defaults")
    resources = get_conservative_defaults()
elif confidence < 0.7:
    # Medium confidence, use predictions with higher buffer
    resources = apply_buffer(requirements, buffer=1.5)
else:
    # High confidence, use predictions with standard buffer
    resources = apply_buffer(requirements, buffer=1.2)
```

### Default Requirements

For unknown philosophies or insufficient data:

```python
{
    "cpu_percent": 50.0,             # 50% CPU
    "memory_mb": 256.0,              # 256 MB RAM
    "execution_time_seconds": 60.0   # 1 minute timeout
}
```

## Memory Optimization

### Garbage Collection Tuning

The optimizer performs periodic memory optimization:

1. **Measure Memory**: Capture RSS before GC
2. **Force Collection**: `gc.collect()`
3. **Measure Again**: Capture RSS after GC
4. **Calculate Savings**: `memory_before - memory_after`
5. **Log Metrics**: Track objects collected and memory released

**Implementation**:
```python
async def optimize_memory() -> dict[str, Any]:
    """Perform memory optimization."""
    import psutil

    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)

    # Force garbage collection
    collected = gc.collect()

    memory_after = process.memory_info().rss / (1024 * 1024)
    memory_released = memory_before - memory_after

    return {
        "enabled": True,
        "objects_collected": collected,
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_released_mb": round(memory_released, 2),
    }
```

### Weak References

The optimizer uses weak references for agent tracking to prevent memory leaks:

```python
# Weak reference dictionary
self._tracked_agents: weakref.WeakValueDictionary[str, AgentExecutionState] = (
    weakref.WeakValueDictionary()
)

# Agents are automatically removed when no longer referenced
self._tracked_agents[agent.agent_id] = agent  # Weak reference
```

### Background Optimization Loop

**Interval**: 5 minutes (300 seconds)

**Tasks**:
1. Memory optimization via GC
2. Expired cache entry cleanup

```python
async def _optimization_loop(self) -> None:
    """Background optimization loop."""
    try:
        while True:
            await asyncio.sleep(300)  # 5 minutes

            # Perform memory optimization
            await self.optimize_memory()

            # Clean up cache
            if self._cache:
                self._cache._evict_expired()
    except asyncio.CancelledError:
        raise
```

## Usage Examples

### Complete Initialization

```python
from agentcore.agent_runtime.services.performance_optimizer import (
    get_performance_optimizer,
    PerformanceOptimizer,
)

# Get global instance (recommended)
optimizer = get_performance_optimizer()

# Or create custom instance
optimizer = PerformanceOptimizer(
    enable_caching=True,
    enable_pooling=True,
    enable_gc_optimization=True,
    cache_size=1000,
    pool_size=10,
)

# Start background optimization
await optimizer.start()
```

### Agent Lifecycle Integration

```python
async def create_agent_with_optimization(config: AgentConfig):
    """Create agent with performance optimizations."""
    optimizer = get_performance_optimizer()

    # 1. Predict resource requirements
    requirements = optimizer.predict_resource_requirements(config)

    # 2. Create container with pooling
    container_id, metrics = await optimizer.optimize_container_creation(
        config=config,
        create_fn=lambda: create_container(config, requirements),
    )

    # Log startup metrics
    if metrics["warm_start"]:
        logger.info(f"Warm start: {metrics['startup_time_ms']}ms")
    else:
        logger.info(f"Cold start: {metrics['startup_time_ms']}ms")

    # 3. Create agent state
    agent_state = AgentExecutionState(
        agent_id=config.agent_id,
        container_id=container_id,
        status="initializing",
    )

    return agent_state
```

### Tool Execution with Caching

```python
async def execute_tool_with_cache(tool_id: str, args: dict):
    """Execute tool with metadata caching."""
    optimizer = get_performance_optimizer()

    # Check cache first
    metadata = optimizer.get_cached_tool_metadata(tool_id)

    if metadata is None:
        # Cache miss - load metadata
        metadata = await load_tool_metadata(tool_id)
        optimizer.cache_tool_metadata(tool_id, metadata)

    # Execute tool with metadata
    result = await execute_tool(metadata, args)

    return result
```

### Pattern-Based Optimization

```python
async def execute_react_pattern(agent_id: str, task: str):
    """Execute ReAct pattern with caching."""
    optimizer = get_performance_optimizer()

    # Check for cached pattern
    pattern = optimizer.get_cached_pattern("react:standard")

    if pattern is None:
        # Define and cache pattern
        pattern = {
            "steps": ["think", "act", "observe"],
            "max_iterations": 5,
            "timeout_per_step": 30,
        }
        optimizer.cache_execution_pattern("react:standard", pattern)

    # Execute with cached pattern
    result = await execute_pattern(agent_id, task, pattern)

    # Track resource usage for future predictions
    optimizer.track_agent_resources(agent_state, {
        "philosophy": "react",
        "cpu_percent": result.cpu_usage,
        "memory_usage_mb": result.memory_usage,
        "execution_time": result.duration,
    })

    return result
```

### Resource Tracking Loop

```python
async def track_agent_resources_continuously(agent_id: str):
    """Continuously track agent resource usage."""
    optimizer = get_performance_optimizer()

    while agent_running:
        # Collect current metrics
        metrics = await get_agent_metrics(agent_id)

        # Track for prediction learning
        optimizer.track_agent_resources(agent_state, {
            "philosophy": agent_state.philosophy,
            "cpu_percent": metrics.cpu_percent,
            "memory_usage_mb": metrics.memory_mb,
            "execution_time": metrics.uptime_seconds,
        })

        await asyncio.sleep(30)  # Track every 30 seconds
```

### Performance Monitoring

```python
async def monitor_performance_metrics():
    """Monitor and log performance metrics."""
    optimizer = get_performance_optimizer()

    while True:
        metrics = optimizer.get_performance_metrics()

        # Log cache performance
        cache_hit_rate = metrics["cache_stats"]["hit_rate_percent"]
        logger.info(f"Cache hit rate: {cache_hit_rate}%")

        # Log warm start rate
        warm_rate = metrics.get("warm_start_rate_percent", 0)
        logger.info(f"Warm start rate: {warm_rate}%")

        # Log memory savings
        memory_saved = metrics["optimization_metrics"]["memory_released_mb"]
        logger.info(f"Total memory saved: {memory_saved} MB")

        # Alert if performance degrades
        if cache_hit_rate < 80:
            logger.warning("Low cache hit rate, consider increasing cache size")

        if warm_rate < 80:
            logger.warning("Low warm start rate, consider increasing pool size")

        await asyncio.sleep(300)  # Check every 5 minutes
```

## Best Practices

### 1. Use Global Instance

**Recommended**:
```python
from agentcore.agent_runtime.services.performance_optimizer import get_performance_optimizer

optimizer = get_performance_optimizer()
```

**Reason**: Ensures single optimizer instance across application, shared cache and pool.

### 2. Start Background Optimization Early

```python
async def app_startup():
    """Application startup handler."""
    optimizer = get_performance_optimizer()
    await optimizer.start()  # Start background tasks
```

**Reason**: Enables automatic memory optimization and cache cleanup.

### 3. Pre-Warm Pools for Common Philosophies

```python
async def prewarm_common_philosophies():
    """Pre-warm pools for frequently used philosophies."""
    optimizer = get_performance_optimizer()

    for philosophy in [AgentPhilosophy.REACT, AgentPhilosophy.CHAIN_OF_THOUGHT]:
        await optimizer._pool.pre_warm(
            philosophy,
            create_container_fn,
            count=5
        )
```

**Reason**: Eliminates cold starts for initial agent creations.

### 4. Monitor and Tune Based on Metrics

```python
metrics = optimizer.get_performance_metrics()

# Tune cache size if hit rate is low
if metrics["cache_stats"]["hit_rate_percent"] < 80:
    optimizer._cache._max_size *= 2

# Tune pool size if warm start rate is low
if metrics.get("warm_start_rate_percent", 100) < 80:
    optimizer._pool._pool_size += 5
```

**Reason**: Adaptive tuning based on actual workload patterns.

### 5. Use Appropriate TTL for Cache Entries

```python
# Frequently changing data: short TTL
optimizer.cache_tool_metadata("dynamic_api", metadata, ttl=60)

# Stable data: long TTL
optimizer.cache_tool_metadata("static_config", metadata, ttl=3600)
```

**Reason**: Balance between freshness and cache efficiency.

### 6. Track Resources for All Philosophies

```python
# Always track after agent execution
optimizer.track_agent_resources(agent_state, {
    "philosophy": config.philosophy.value,
    "cpu_percent": actual_cpu,
    "memory_usage_mb": actual_memory,
    "execution_time": actual_time,
})
```

**Reason**: Builds accurate prediction models over time.

### 7. Handle Low Confidence Predictions

```python
confidence = optimizer._predictor.get_prediction_confidence(philosophy)

if confidence < 0.5:
    # Use conservative defaults for safety
    requirements = {
        "cpu_percent": 75.0,  # Higher buffer
        "memory_mb": 512.0,
        "execution_time_seconds": 120.0,
    }
else:
    requirements = optimizer.predict_resource_requirements(config)
```

**Reason**: Avoid under-provisioning when prediction confidence is low.

### 8. Gracefully Handle Disabled Components

```python
# Check if caching is enabled before using
if optimizer._cache:
    cached = optimizer.get_cached_tool_metadata(tool_id)
else:
    cached = None

# Check if pooling is enabled
if optimizer._pool:
    container_id, metrics = await optimizer.optimize_container_creation(...)
else:
    container_id = await create_container_direct(...)
```

**Reason**: Allows flexible configuration without code changes.

## Testing

### Test Coverage

**Overall**: 32 test scenarios, 92% coverage

**Breakdown**:
- **CacheEntry**: 3 tests (creation, expiration, access tracking)
- **PerformanceCache**: 8 tests (set/get, miss, expiration, invalidation, clear, stats, eviction, LRU)
- **ContainerPool**: 6 tests (cold/warm start, release, max size, pre-warm, stats)
- **ResourcePredictor**: 4 tests (defaults, learning, confidence, history limit)
- **PerformanceOptimizer**: 11 tests (init, start/stop, container creation, caching, tracking, prediction, memory optimization, metrics, background loop)

### Running Tests

```bash
# Run all performance optimizer tests
uv run pytest tests/agent_runtime/test_performance_optimizer.py -v

# Run with coverage
uv run pytest tests/agent_runtime/test_performance_optimizer.py \
    --cov=src/agentcore/agent_runtime/services/performance_optimizer \
    --cov-report=term-missing

# Run specific test class
uv run pytest tests/agent_runtime/test_performance_optimizer.py::TestPerformanceOptimizer -v

# Run single test
uv run pytest tests/agent_runtime/test_performance_optimizer.py::TestPerformanceCache::test_cache_lru_eviction -v
```

### Test Scenarios

#### CacheEntry Tests

1. **test_cache_entry_creation**: Verify entry created with correct attributes
2. **test_cache_entry_expiration**: Verify TTL expiration works
3. **test_cache_entry_access_tracking**: Verify access count and timestamp updates

#### PerformanceCache Tests

1. **test_cache_set_and_get**: Basic cache operations
2. **test_cache_miss**: Returns None for missing keys
3. **test_cache_expiration**: Expired entries return None
4. **test_cache_invalidation**: Manual invalidation works
5. **test_cache_clear**: Clear removes all entries
6. **test_cache_stats**: Statistics tracking accurate
7. **test_cache_eviction_on_max_size**: Max size enforced
8. **test_cache_lru_eviction**: LRU eviction works correctly

#### ContainerPool Tests

1. **test_pool_cold_start**: Empty pool creates new container
2. **test_pool_warm_start**: Pool returns existing container
3. **test_pool_release**: Container returned to pool
4. **test_pool_max_size_enforcement**: Pool size limit enforced
5. **test_pool_pre_warm**: Pre-warming creates containers
6. **test_pool_stats**: Statistics reporting accurate

#### ResourcePredictor Tests

1. **test_predictor_default_requirements**: Unknown philosophy gets defaults
2. **test_predictor_learning_from_usage**: Learns from historical data
3. **test_predictor_confidence**: Confidence increases with data
4. **test_predictor_history_limit**: History size limited

#### PerformanceOptimizer Tests

1. **test_optimizer_initialization**: Initializes with correct settings
2. **test_optimizer_start_stop**: Background tasks start/stop
3. **test_optimizer_container_creation_without_pool**: Works without pooling
4. **test_optimizer_tool_metadata_caching**: Tool metadata cached
5. **test_optimizer_execution_pattern_caching**: Patterns cached
6. **test_optimizer_resource_tracking**: Tracks agent resources
7. **test_optimizer_resource_prediction**: Predicts requirements
8. **test_optimizer_memory_optimization**: GC optimization works
9. **test_optimizer_memory_optimization_disabled**: Respects disabled flag
10. **test_optimizer_performance_metrics**: Metrics reported correctly
11. **test_optimizer_background_loop**: Background loop runs without errors

### Performance Benchmarks

Expected performance characteristics:

**Cache Operations**:
- Get (hit): < 1 μs
- Get (miss): < 1 μs
- Set: < 5 μs
- Eviction: < 100 μs per entry

**Container Pool**:
- Warm start: 10-50 ms
- Cold start: 1-3 seconds
- Acquire/release: < 10 ms

**Resource Prediction**:
- Prediction: < 1 ms
- Recording: < 100 μs

**Memory Optimization**:
- GC collection: 100-500 ms
- Memory released: 10-100 MB typical

## Additional Resources

- [Python gc module](https://docs.python.org/3/library/gc.html) - Garbage collection interface
- [Python weakref module](https://docs.python.org/3/library/weakref.html) - Weak references
- [psutil documentation](https://psutil.readthedocs.io/) - System and process utilities
- [LRU Cache Algorithm](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU) - Eviction strategy
- [Container Performance](https://www.docker.com/blog/intro-guide-to-dockerfile-best-practices/) - Docker optimization

## Support

For performance optimization questions:
- Review this documentation
- Check test coverage in `tests/agent_runtime/test_performance_optimizer.py`
- Monitor performance metrics via `get_performance_metrics()`
- Tune based on actual workload patterns
