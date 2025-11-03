# Error Handling & Recovery System

This guide covers the comprehensive error handling and recovery system implemented in AgentCore Runtime for resilient agent execution.

## Table of Contents

1. [Overview](#overview)
2. [Error Categories](#error-categories)
3. [Error Severity Levels](#error-severity-levels)
4. [Recovery Strategies](#recovery-strategies)
5. [Circuit Breaker Pattern](#circuit-breaker-pattern)
6. [Degradation Levels](#degradation-levels)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Testing](#testing)

## Overview

AgentCore Runtime implements a multi-layered error handling system for fault-tolerant agent execution:

- **Error Classification**: 10 error categories with 4 severity levels
- **Recovery Strategies**: 9 automated recovery strategies
- **Circuit Breaker**: Prevents cascading failures with state management
- **Graceful Degradation**: 4 degradation levels for partial functionality
- **Error History**: Tracks error patterns for adaptive recovery
- **Retry Logic**: Exponential and constant backoff with jitter

```
┌─────────────────────────────────────────────────────┐
│  Application Layer                                  │
│  ├─ ErrorRecoveryService (execute_with_recovery)   │
│  ├─ Error categorization & severity                │
│  └─ Strategy selection & execution                 │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Recovery Layer                                     │
│  ├─ Retry strategies (exponential/constant)        │
│  ├─ Circuit breaker integration                    │
│  ├─ Graceful degradation                           │
│  └─ Error history tracking                         │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  Circuit Breaker Layer                              │
│  ├─ State management (CLOSED/OPEN/HALF_OPEN)       │
│  ├─ Failure threshold monitoring                   │
│  ├─ Automatic recovery testing                     │
│  └─ Statistics collection                          │
└─────────────────────────────────────────────────────┘
```

## Error Categories

AgentCore classifies errors into 10 categories for targeted recovery:

### INFRASTRUCTURE
System-level failures (Docker, networking, file system)

```python
from agentcore.agent_runtime.models.error_types import ErrorCategory

# Infrastructure errors trigger restart strategies
error_category = ErrorCategory.INFRASTRUCTURE
```

### RESOURCE_EXHAUSTION
Resource limit violations (memory, CPU, storage, file descriptors)

```python
# Resource errors trigger degradation
error_category = ErrorCategory.RESOURCE_EXHAUSTION
```

### NETWORK
Network communication failures (connection, timeout, DNS)

```python
# Network errors trigger retry with exponential backoff
error_category = ErrorCategory.NETWORK
```

### SECURITY
Security violations (authentication, authorization, validation)

```python
# Security errors require manual intervention
error_category = ErrorCategory.SECURITY
```

### CONFIGURATION
Configuration errors (invalid settings, missing values)

### EXECUTION
Code execution errors (runtime exceptions, assertions)

### STATE
State management errors (corruption, inconsistency)

### EXTERNAL_SERVICE
Third-party service failures (APIs, databases)

### TIMEOUT
Operation timeout errors

### UNKNOWN
Unclassified errors (default category)

## Error Severity Levels

Four severity levels determine max retry attempts and recovery urgency:

### CRITICAL (Max 1 retry)
System-critical errors requiring immediate attention
- Security violations
- Infrastructure failures
- State corruption

```python
from agentcore.agent_runtime.models.error_types import ErrorSeverity

# Critical errors get minimal retries
severity = ErrorSeverity.CRITICAL  # max_retries = 1
```

### HIGH (Max 3 retries)
Important errors affecting core functionality
- Resource exhaustion
- External service failures

```python
severity = ErrorSeverity.HIGH  # max_retries = 3
```

### MEDIUM (Max 5 retries)
Moderate errors with localized impact
- Network timeouts
- Temporary connection failures

```python
severity = ErrorSeverity.MEDIUM  # max_retries = 5
```

### LOW (Max 10 retries)
Minor errors with minimal impact
- Transient failures
- Optional feature errors

```python
severity = ErrorSeverity.LOW  # max_retries = 10
```

## Recovery Strategies

AgentCore provides 9 recovery strategies, automatically selected based on error category:

### 1. RETRY_EXPONENTIAL
Exponential backoff retry (1s, 2s, 4s, 8s, ...)

**Use Case**: Network errors, API rate limits, transient failures

```python
from agentcore.agent_runtime.models.error_types import RecoveryStrategy, RetryConfig

retry_config = RetryConfig(
    max_attempts=5,
    initial_delay_seconds=1.0,
    max_delay_seconds=60.0,
    exponential_base=2.0,
    jitter=True  # Add randomness to prevent thundering herd
)

# Network errors default to exponential retry
error_category = ErrorCategory.NETWORK
# Strategies: [RETRY_EXPONENTIAL, CIRCUIT_BREAK]
```

### 2. RETRY_CONSTANT
Constant delay retry (1s, 1s, 1s, ...)

**Use Case**: Timeout errors, quick recovery scenarios

```python
retry_config = RetryConfig(
    max_attempts=3,
    initial_delay_seconds=1.0,
    jitter=False
)

# Timeout errors default to constant retry
error_category = ErrorCategory.TIMEOUT
# Strategies: [RETRY_CONSTANT, MANUAL]
```

### 3. RESTART_CHECKPOINT
Restart from last checkpoint

**Use Case**: Execution errors with state preservation

```python
# Execution errors can restart from checkpoint
error_category = ErrorCategory.EXECUTION
# Strategies: [RETRY_EXPONENTIAL, RESTART_CHECKPOINT, DEGRADE]
```

### 4. RESTART_CLEAN
Clean restart without state

**Use Case**: State corruption, configuration errors

```python
# Configuration errors trigger clean restart
error_category = ErrorCategory.CONFIGURATION
# Strategies: [RESTART_CLEAN, MANUAL]
```

### 5. FAILOVER
Switch to backup resource/service

**Use Case**: External service failures, infrastructure issues

```python
# Infrastructure errors support failover
error_category = ErrorCategory.INFRASTRUCTURE
# Strategies: [FAILOVER, RESTART_CHECKPOINT, RESTART_CLEAN]
```

### 6. DEGRADE
Reduce functionality to essential operations

**Use Case**: Resource exhaustion, degraded system state

```python
# Resource exhaustion triggers degradation
error_category = ErrorCategory.RESOURCE_EXHAUSTION
# Strategies: [DEGRADE, RESTART_CHECKPOINT]
```

### 7. CIRCUIT_BREAK
Circuit breaker pattern for cascading failure prevention

**Use Case**: Network errors, external service failures

```python
from agentcore.agent_runtime.services.error_recovery import get_error_recovery_service

recovery_service = get_error_recovery_service()

# Enable circuit breaker
result, recovery_result = await recovery_service.execute_with_recovery(
    function,
    agent_id="my-agent",
    error_category=ErrorCategory.NETWORK,
    use_circuit_breaker=True  # Enable circuit breaker
)
```

### 8. MANUAL
Require manual intervention

**Use Case**: Security errors, unknown errors

```python
# Security errors require manual intervention
error_category = ErrorCategory.SECURITY
# Strategies: [MANUAL]
```

### 9. NONE
No automatic recovery

## Circuit Breaker Pattern

Circuit breaker prevents cascading failures with three states:

### States

**CLOSED** (Normal operation)
- All requests allowed
- Monitors failure rate
- Transitions to OPEN after failure_threshold failures

**OPEN** (Failure mode)
- All requests blocked immediately
- Waits timeout_seconds before testing recovery
- Raises `CircuitBreakerError` for blocked requests

**HALF_OPEN** (Testing recovery)
- Limited requests allowed for testing
- Success transitions to CLOSED
- Failure returns to OPEN

```
    ┌─────────┐
    │ CLOSED  │ ◄─────────────┐
    └────┬────┘                │
         │                     │
    failure_threshold      success_threshold
    reached                successes
         │                     │
         ▼                     │
    ┌─────────┐           ┌───┴────────┐
    │  OPEN   │ ─────────►│ HALF_OPEN  │
    └─────────┘  timeout  └────────────┘
       seconds
```

### Configuration

```python
from agentcore.agent_runtime.models.error_types import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,           # Open after 5 failures
    success_threshold=2,            # Close after 2 successes in half-open
    timeout_seconds=60.0,          # Wait 60s before half-open
    half_open_max_attempts=3       # Max 3 attempts in half-open
)
```

### Usage

```python
from agentcore.agent_runtime.services.circuit_breaker import get_circuit_breaker_registry

registry = get_circuit_breaker_registry()

# Get or create breaker
breaker = await registry.get_breaker("external-api", config)

# Execute with protection
try:
    result = await breaker.call(external_api_call, arg1, arg2)
except CircuitBreakerError:
    # Circuit is open, skip request
    logger.warning("Circuit breaker open for external-api")
    return cached_response
```

### Registry Management

```python
# List all breakers
breakers = registry.list_breakers()

# Get statistics
stats = registry.get_all_stats()
# {
#   "external-api": {
#     "state": "open",
#     "failure_count": 5,
#     "last_failure_time": "2025-11-03T12:00:00Z"
#   }
# }

# Reset all breakers
await registry.reset_all()

# Remove specific breaker
await registry.remove_breaker("external-api")
```

## Degradation Levels

AgentCore supports 4 degradation levels for partial functionality:

### FULL (No degradation)
All features operational

### REDUCED (Minor degradation)
Non-critical features disabled
- Triggered after 1-4 recent errors

```python
from agentcore.agent_runtime.models.error_types import DegradationLevel

# Function receives degradation flag
async def process_request(degraded: bool = False, degradation_level: DegradationLevel = None):
    if degraded and degradation_level == DegradationLevel.REDUCED:
        # Disable caching, reduce quality
        return process_with_reduced_features()
    return process_normally()
```

### MINIMAL (Significant degradation)
Only essential features enabled
- Triggered after 5-9 recent errors

```python
async def process_request(degraded: bool = False, degradation_level: DegradationLevel = None):
    if degradation_level == DegradationLevel.MINIMAL:
        # Essential operations only
        return process_minimal()
```

### EMERGENCY (Critical degradation)
Bare minimum functionality
- Triggered after 10+ recent errors

```python
async def process_request(degraded: bool = False, degradation_level: DegradationLevel = None):
    if degradation_level == DegradationLevel.EMERGENCY:
        # Emergency mode
        return emergency_response()
```

## Usage Examples

### Basic Error Recovery

```python
from agentcore.agent_runtime.services.error_recovery import get_error_recovery_service
from agentcore.agent_runtime.models.error_types import ErrorCategory

recovery_service = get_error_recovery_service()

async def risky_operation():
    # Operation that may fail
    return await external_api.call()

# Execute with automatic recovery
result, recovery_result = await recovery_service.execute_with_recovery(
    risky_operation,
    agent_id="my-agent",
    error_category=ErrorCategory.NETWORK
)

if recovery_result:
    print(f"Recovered using {recovery_result.strategy_used}")
    print(f"Attempts: {recovery_result.attempts}")
    print(f"Duration: {recovery_result.duration_seconds}s")
```

### Custom Retry Configuration

```python
from agentcore.agent_runtime.models.error_types import RetryConfig

# Custom exponential backoff
retry_config = RetryConfig(
    max_attempts=10,
    initial_delay_seconds=0.5,
    max_delay_seconds=30.0,
    exponential_base=1.5,
    jitter=True
)

result, _ = await recovery_service.execute_with_recovery(
    risky_operation,
    agent_id="my-agent",
    error_category=ErrorCategory.NETWORK,
    retry_config=retry_config
)
```

### Circuit Breaker Integration

```python
# Enable circuit breaker for external service
result, recovery_result = await recovery_service.execute_with_recovery(
    external_service_call,
    agent_id="my-agent",
    error_category=ErrorCategory.EXTERNAL_SERVICE,
    use_circuit_breaker=True  # Enable circuit breaker
)
```

### Error History Analysis

```python
# Get error history for agent
history = await recovery_service.get_error_history("my-agent", limit=10)

for error in history:
    print(f"Category: {error.category.value}")
    print(f"Severity: {error.severity.value}")
    print(f"Message: {error.message}")
    print(f"Time: {error.timestamp}")
```

### Degradation Management

```python
# Check degradation state
degradation = await recovery_service.get_degradation_state("my-agent")
if degradation:
    print(f"Agent degraded to level: {degradation.value}")

# Reset degradation
await recovery_service.reset_degradation("my-agent")
```

### Statistics Collection

```python
# Get global statistics
stats = await recovery_service.get_statistics()

print(f"Total errors: {stats['total_errors']}")
print(f"Agents with errors: {stats['agents_with_errors']}")
print(f"Degraded agents: {stats['degraded_agents']}")

# Errors by category
for category, count in stats['errors_by_category'].items():
    print(f"{category}: {count}")

# Errors by severity
for severity, count in stats['errors_by_severity'].items():
    print(f"{severity}: {count}")

# Circuit breaker stats
for breaker_name, breaker_stats in stats['circuit_breakers'].items():
    print(f"{breaker_name}: {breaker_stats['state']}")
```

## Best Practices

### 1. Use Appropriate Error Categories

```python
# Correct: Specific category for targeted recovery
result, _ = await recovery_service.execute_with_recovery(
    database_query,
    agent_id="my-agent",
    error_category=ErrorCategory.EXTERNAL_SERVICE  # Database is external service
)

# Incorrect: Generic category
error_category=ErrorCategory.UNKNOWN  # Misses targeted recovery strategies
```

### 2. Enable Circuit Breaker for External Services

```python
# Always enable circuit breaker for external APIs
result, _ = await recovery_service.execute_with_recovery(
    external_api_call,
    agent_id="my-agent",
    error_category=ErrorCategory.EXTERNAL_SERVICE,
    use_circuit_breaker=True  # Prevent cascading failures
)
```

### 3. Implement Degradation Support

```python
# Design functions to support degradation
async def process_data(
    data: dict,
    degraded: bool = False,
    degradation_level: DegradationLevel | None = None
) -> dict:
    """Process data with degradation support."""

    if degradation_level == DegradationLevel.EMERGENCY:
        # Emergency mode: skip validation, caching
        return minimal_processing(data)

    if degradation_level == DegradationLevel.MINIMAL:
        # Minimal mode: skip optional features
        return basic_processing(data)

    if degradation_level == DegradationLevel.REDUCED:
        # Reduced mode: disable expensive operations
        return reduced_processing(data)

    # Full processing
    return full_processing(data)
```

### 4. Configure Retry Based on Operation Type

```python
# Fast operations: Short delays, more retries
fast_retry = RetryConfig(
    max_attempts=10,
    initial_delay_seconds=0.1,
    max_delay_seconds=5.0,
    exponential_base=2.0,
    jitter=True
)

# Slow operations: Longer delays, fewer retries
slow_retry = RetryConfig(
    max_attempts=3,
    initial_delay_seconds=2.0,
    max_delay_seconds=30.0,
    exponential_base=2.0,
    jitter=True
)
```

### 5. Monitor Error Patterns

```python
# Regularly check error history
async def monitor_agent_health(agent_id: str):
    """Monitor agent error patterns."""
    history = await recovery_service.get_error_history(agent_id, limit=50)

    # Check for repeated errors
    error_counts: dict[ErrorCategory, int] = {}
    for error in history:
        error_counts[error.category] = error_counts.get(error.category, 0) + 1

    # Alert if too many errors in specific category
    for category, count in error_counts.items():
        if count > 10:
            alert_ops_team(f"Agent {agent_id} has {count} {category.value} errors")
```

### 6. Handle Circuit Breaker Errors Gracefully

```python
from agentcore.agent_runtime.services.circuit_breaker import CircuitBreakerError

try:
    result = await breaker.call(external_api_call)
except CircuitBreakerError:
    # Circuit open, use fallback
    logger.warning("Circuit breaker open, using cached response")
    return get_cached_response()
```

### 7. Reset Degradation After Recovery

```python
# After successful operations, reset degradation
try:
    result = await process_request()
    # Success! Reset degradation
    await recovery_service.reset_degradation(agent_id)
except Exception as e:
    # Degradation will increase automatically
    logger.error(f"Operation failed: {e}")
```

### 8. Use Jitter for Retry Delays

```python
# Enable jitter to prevent thundering herd
retry_config = RetryConfig(
    max_attempts=5,
    initial_delay_seconds=1.0,
    max_delay_seconds=10.0,
    exponential_base=2.0,
    jitter=True  # Adds randomness: delay * random(0.5, 1.5)
)
```

## Testing

AgentCore includes comprehensive tests for error handling:

### Error Recovery Tests (17 scenarios)

```bash
# Run error recovery tests
uv run pytest tests/agent_runtime/test_error_recovery.py -v
```

**Test Coverage**:
- Successful execution without errors
- Exponential retry with backoff
- Constant delay retry
- Retry exhaustion handling
- Circuit breaker integration
- Degraded execution
- Error history recording
- Degradation level progression
- Statistics collection
- Error severity determination
- Recovery strategy selection

### Circuit Breaker Tests (16 scenarios)

```bash
# Run circuit breaker tests
uv run pytest tests/agent_runtime/test_circuit_breaker.py -v
```

**Test Coverage**:
- Initial state verification
- Failure recording and circuit opening
- Open circuit blocking
- Half-open transition after timeout
- Circuit closing after success
- Half-open failure reopening
- Manual reset
- Statistics collection
- Registry management

### Integration Tests

```bash
# Run all error handling integration tests
uv run pytest tests/agent_runtime/integration/ -k error -v
```

## Default Recovery Strategies

AgentCore maps error categories to default recovery strategies:

| Error Category        | Recovery Strategies                               |
|-----------------------|--------------------------------------------------|
| INFRASTRUCTURE        | FAILOVER → RESTART_CHECKPOINT → RESTART_CLEAN    |
| RESOURCE_EXHAUSTION   | DEGRADE → RESTART_CHECKPOINT                     |
| NETWORK              | RETRY_EXPONENTIAL → CIRCUIT_BREAK                |
| SECURITY             | MANUAL                                           |
| CONFIGURATION        | RESTART_CLEAN → MANUAL                           |
| EXECUTION            | RETRY_EXPONENTIAL → RESTART_CHECKPOINT → DEGRADE |
| STATE                | RESTART_CHECKPOINT → RESTART_CLEAN               |
| EXTERNAL_SERVICE     | RETRY_EXPONENTIAL → CIRCUIT_BREAK → FAILOVER    |
| TIMEOUT              | RETRY_CONSTANT → MANUAL                          |
| UNKNOWN              | RETRY_CONSTANT → MANUAL                          |

## Additional Resources

- [Error Types Model](../../src/agentcore/agent_runtime/models/error_types.py)
- [Error Recovery Service](../../src/agentcore/agent_runtime/services/error_recovery.py)
- [Circuit Breaker Implementation](../../src/agentcore/agent_runtime/services/circuit_breaker.py)
- [Error Recovery Tests](../../tests/agent_runtime/test_error_recovery.py)
- [Circuit Breaker Tests](../../tests/agent_runtime/test_circuit_breaker.py)

## Support

For error handling questions or issues:
- Review existing tests for usage patterns
- Check error history and statistics for debugging
- Monitor circuit breaker states for external service health
- Adjust retry configurations based on operation characteristics
