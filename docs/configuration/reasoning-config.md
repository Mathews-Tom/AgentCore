# Bounded Context Reasoning Configuration Guide

This document describes all configuration options for the bounded context reasoning feature.

## Environment Variables

### LLM Client Configuration

```bash
# OpenAI Configuration (default provider)
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, defaults to OpenAI
OPENAI_MODEL=gpt-4  # Default model for reasoning

# Alternative: Custom LLM Provider
LLM_BASE_URL=https://your-llm-provider.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=custom-model-name
```

### Reasoning Engine Configuration

```bash
# Performance Tuning
REASONING_DEFAULT_CHUNK_SIZE=8192  # Tokens per iteration (1024-32768)
REASONING_DEFAULT_CARRYOVER_SIZE=4096  # Tokens carried between iterations (512-16384)
REASONING_DEFAULT_MAX_ITERATIONS=5  # Maximum iterations (1-50)
REASONING_DEFAULT_TEMPERATURE=0.7  # Sampling temperature (0.0-2.0)

# Security Limits
REASONING_MAX_QUERY_LENGTH=100000  # Maximum query length in characters
REASONING_MAX_ITERATIONS=50  # Hard limit on iterations
REASONING_REQUEST_TIMEOUT=60  # Request timeout in seconds

# Rate Limiting
REASONING_RATE_LIMIT_PER_USER=100  # Requests per minute per user
REASONING_RATE_LIMIT_GLOBAL=1000  # Global requests per minute
```

### Metrics and Monitoring

```bash
# Prometheus Metrics
ENABLE_METRICS=true  # Enable/disable Prometheus metrics
METRICS_PORT=8001  # Port for /metrics endpoint

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json or text
```

### A/B Testing

```bash
# Feature Flags
ENABLE_BOUNDED_REASONING=false  # Enable bounded reasoning feature (default: false)
BOUNDED_REASONING_ROLLOUT_PCT=0  # Percentage of traffic (0-100)
```

## Configuration File (Optional)

Create `config/reasoning.yaml` for structured configuration:

```yaml
reasoning:
  llm:
    provider: openai  # openai, anthropic, custom
    api_key: ${OPENAI_API_KEY}  # Reference env var
    base_url: https://api.openai.com/v1
    model: gpt-4
    timeout: 30
    max_retries: 3

  engine:
    default_chunk_size: 8192
    default_carryover_size: 4096
    default_max_iterations: 5
    default_temperature: 0.7

  security:
    max_query_length: 100000
    max_iterations: 50
    request_timeout: 60
    enable_input_sanitization: true

  rate_limiting:
    per_user_rpm: 100  # Requests per minute
    global_rpm: 1000
    burst_size: 10

  monitoring:
    enable_metrics: true
    enable_tracing: true
    log_level: INFO
```

## Configuration Validation

The system validates all configuration on startup:

```python
from agentcore.reasoning.config import validate_reasoning_config

# Validate configuration
errors = validate_reasoning_config()
if errors:
    print("Configuration errors:", errors)
    exit(1)
```

## Configuration Precedence

Configuration is loaded in the following order (later sources override earlier):

1. Default values (hardcoded)
2. Configuration file (`config/reasoning.yaml`)
3. Environment variables
4. Runtime overrides (via API)

## Performance Tuning Guidelines

### Chunk Size Selection

| Query Size | Recommended Chunk Size | Carryover Size | Max Iterations |
|-----------|------------------------|----------------|----------------|
| Small (<10K tokens) | 4096 | 2048 | 3-5 |
| Medium (10-25K tokens) | 8192 | 4096 | 5-8 |
| Large (25-50K tokens) | 16384 | 8192 | 8-12 |
| Very Large (>50K tokens) | 32768 | 16384 | 10-15 |

### Memory vs Speed Tradeoffs

**Larger chunk sizes:**
- Pro: Fewer iterations, better context retention
- Con: Higher memory usage per iteration, slower per-iteration processing

**Smaller chunk sizes:**
- Pro: Lower memory footprint, faster iterations
- Con: More iterations needed, potential context loss

### Carryover Size Optimization

Carryover size should be 25-50% of chunk size for optimal balance:

```bash
# Conservative (better context retention)
REASONING_DEFAULT_CHUNK_SIZE=8192
REASONING_DEFAULT_CARRYOVER_SIZE=4096  # 50%

# Aggressive (more compute savings)
REASONING_DEFAULT_CHUNK_SIZE=8192
REASONING_DEFAULT_CARRYOVER_SIZE=2048  # 25%
```

## Security Configuration

### Input Sanitization

Enabled by default. Prevents prompt injection attacks:

```bash
REASONING_ENABLE_INPUT_SANITIZATION=true
REASONING_SANITIZATION_STRICT_MODE=false  # Strict mode rejects more patterns
```

### Authentication Requirements

All reasoning requests require JWT authentication:

```bash
JWT_SECRET_KEY=your-secret-key  # MUST change in production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

Required permission: `reasoning:execute`

### TLS/SSL Configuration

For LLM provider connections:

```bash
LLM_VERIFY_SSL=true  # Verify SSL certificates
LLM_CA_BUNDLE=/path/to/ca-bundle.crt  # Optional custom CA bundle
```

## Rate Limiting Configuration

### Per-User Limits

Prevent individual users from overloading the system:

```bash
REASONING_RATE_LIMIT_PER_USER=100  # Requests per minute
REASONING_RATE_LIMIT_BURST=10  # Burst allowance
```

### Global Limits

System-wide protection:

```bash
REASONING_RATE_LIMIT_GLOBAL=1000  # Total requests per minute
REASONING_RATE_LIMIT_GLOBAL_BURST=50
```

### Rate Limit Headers

Responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1634567890
```

## Monitoring Configuration

### Prometheus Metrics

All metrics are prefixed with `reasoning_bounded_context_`:

```bash
# Enable metrics endpoint
ENABLE_METRICS=true

# Metrics are exposed at:
# http://localhost:8001/metrics
```

### Grafana Dashboards

Import dashboards from `grafana/dashboards/reasoning-dashboard.json`

### Alerting

Configure alerts in `prometheus/alerts/reasoning-alerts.yml`

## Troubleshooting

### Common Configuration Issues

**1. LLM API Key Not Found**

```
Error: LLM API key not configured
Solution: Set OPENAI_API_KEY or LLM_API_KEY environment variable
```

**2. Rate Limit Too Restrictive**

```
Error: Rate limit exceeded
Solution: Increase REASONING_RATE_LIMIT_PER_USER or REASONING_RATE_LIMIT_GLOBAL
```

**3. Memory Exhaustion**

```
Error: Out of memory during reasoning
Solution: Reduce REASONING_DEFAULT_CHUNK_SIZE or enable memory limits
```

**4. Slow Response Times**

```
Issue: Requests taking >5 seconds
Solution: Increase REASONING_DEFAULT_CHUNK_SIZE to reduce iterations
```

### Configuration Validation Commands

```bash
# Check current configuration
uv run python -c "from agentcore.reasoning.config import get_config; print(get_config())"

# Validate configuration
uv run python -c "from agentcore.reasoning.config import validate_config; validate_config()"

# Test LLM connection
uv run python -c "from agentcore.reasoning.services.llm_client import LLMClient; LLMClient.test_connection()"
```

## Production Deployment Checklist

- [ ] Change JWT_SECRET_KEY to secure random value
- [ ] Set OPENAI_API_KEY or custom LLM credentials
- [ ] Enable TLS/SSL (LLM_VERIFY_SSL=true)
- [ ] Configure rate limiting for expected load
- [ ] Enable metrics and monitoring
- [ ] Set appropriate LOG_LEVEL (INFO or WARNING)
- [ ] Configure alerting rules
- [ ] Test configuration with validation commands
- [ ] Document any custom configuration changes

## Examples

### Development Configuration

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export REASONING_DEFAULT_MAX_ITERATIONS=3
export REASONING_RATE_LIMIT_PER_USER=1000  # Relaxed for testing
export ENABLE_BOUNDED_REASONING=true  # Enable feature
```

### Production Configuration

```bash
export DEBUG=false
export LOG_LEVEL=INFO
export REASONING_DEFAULT_MAX_ITERATIONS=10
export REASONING_RATE_LIMIT_PER_USER=100
export REASONING_RATE_LIMIT_GLOBAL=1000
export ENABLE_METRICS=true
export LLM_VERIFY_SSL=true
export JWT_SECRET_KEY="$(openssl rand -hex 32)"
```

### High-Performance Configuration

```bash
export REASONING_DEFAULT_CHUNK_SIZE=16384
export REASONING_DEFAULT_CARRYOVER_SIZE=8192
export REASONING_DEFAULT_MAX_ITERATIONS=15
export REASONING_RATE_LIMIT_GLOBAL=5000  # Higher throughput
```

## References

- [Bounded Context Reasoning Specification](../specs/bounded-context-reasoning/spec.md)
- [Implementation Plan](../specs/bounded-context-reasoning/plan.md)
- [API Documentation](../api/reasoning-api.md)
- [Grafana Dashboards](../../grafana/README.md)
- [Prometheus Alerts](../../prometheus/alerts/reasoning-alerts.yml)
