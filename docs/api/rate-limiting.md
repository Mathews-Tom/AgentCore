# Rate Limiting Guide

AgentCore API Gateway implements intelligent rate limiting to ensure fair usage and protect against abuse.

## Overview

Rate limiting controls the number of requests a client can make within a specific time window. The gateway uses Redis-based distributed rate limiting with multiple algorithms.

## Rate Limit Policies

### Default Limits

| Limit Type | Requests | Window | Scope |
|------------|----------|--------|-------|
| **Per IP** | 1,000 | 60 seconds | All endpoints |
| **Per User** | 5,000 | 60 seconds | Authenticated requests |
| **Per Endpoint** | 100 | 60 seconds | Specific endpoints |

### Exempt Endpoints

The following endpoints are exempt from rate limiting:

- `/health` - Health check
- `/metrics` - Prometheus metrics
- `/.well-known/` - Discovery endpoints

## Rate Limit Headers

Every API response includes rate limit information in headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1729323600
X-RateLimit-Policy: client_ip
```

### Header Descriptions

- **X-RateLimit-Limit**: Maximum requests allowed in current window
- **X-RateLimit-Remaining**: Requests remaining in current window
- **X-RateLimit-Reset**: Unix timestamp when rate limit resets
- **X-RateLimit-Policy**: Applied policy (client_ip, user, endpoint)

## Rate Limit Exceeded

When rate limits are exceeded, the API returns HTTP 429:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 30
```

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": 60,
      "retry_after": 30,
      "policy": "client_ip"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

### Handling Rate Limits

#### Respect Retry-After Header

```python
import time
import requests

response = requests.get("https://api.agentcore.ai/auth/me")

if response.status_code == 429:
    retry_after = int(response.headers.get("Retry-After", 60))
    print(f"Rate limited. Waiting {retry_after} seconds...")
    time.sleep(retry_after)
    # Retry request
    response = requests.get("https://api.agentcore.ai/auth/me")
```

#### Implement Exponential Backoff

```python
import time
import random

def api_call_with_backoff(url, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url)

        if response.status_code != 429:
            return response

        # Exponential backoff with jitter
        wait_time = (2 ** attempt) + random.uniform(0, 1)
        print(f"Rate limited. Waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)

    raise Exception("Max retries exceeded")
```

## Rate Limiting Algorithms

The gateway supports multiple rate limiting algorithms:

### 1. Sliding Window (Default)

Most accurate algorithm that provides smooth rate limiting without sudden bursts.

**How it works:**
- Tracks exact request timestamps
- Counts requests in a rolling time window
- Prevents burst traffic at window boundaries

**Best for:** General API usage with consistent traffic patterns

### 2. Fixed Window

Simple algorithm that resets counters at fixed intervals.

**How it works:**
- Counts requests in fixed time buckets
- Resets counter at bucket boundaries
- Allows burst traffic at window edges

**Best for:** Simple rate limiting with predictable windows

### 3. Token Bucket

Allows controlled bursts while maintaining average rate.

**How it works:**
- Tokens added to bucket at constant rate
- Requests consume tokens from bucket
- Bucket capacity limits maximum burst

**Best for:** APIs that benefit from burst traffic handling

### 4. Leaky Bucket

Smooths out traffic by processing requests at constant rate.

**How it works:**
- Requests added to queue (bucket)
- Processed at constant rate
- Excess requests overflow and are rejected

**Best for:** Protecting backend services from traffic spikes

## DDoS Protection

In addition to rate limiting, the gateway implements DDoS protection:

### Global Rate Limits

- **Global RPS:** 10,000 requests/second
- **Global RPM:** 500,000 requests/minute

### IP-Based Protection

- **IP RPS:** 100 requests/second per IP
- **IP RPM:** 1,000 requests/minute per IP

### Burst Detection

- **Burst Threshold:** 5x normal rate
- **Burst Window:** 10 seconds
- **Action:** Automatic IP blocking

### Automatic IP Blocking

When an IP exceeds DDoS thresholds:

1. IP is automatically blocked for 1 hour
2. Block duration doubles for repeated violations
3. Administrator notification is sent
4. IP can be manually unblocked if needed

## Best Practices

### 1. Monitor Rate Limit Headers

Always check rate limit headers to avoid hitting limits:

```python
response = requests.get(url)
remaining = int(response.headers.get("X-RateLimit-Remaining", 0))

if remaining < 10:
    print("WARNING: Approaching rate limit")
    # Implement backoff strategy
```

### 2. Use Authenticated Requests

Authenticated users have higher rate limits (5,000 vs 1,000):

```bash
# Lower limit (1,000/min)
curl "https://api.agentcore.ai/public/endpoint"

# Higher limit (5,000/min)
curl "https://api.agentcore.ai/auth/me" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. Batch Requests

Combine multiple operations into single requests when possible:

```python
# Bad: Multiple requests
for agent_id in agent_ids:
    get_agent(agent_id)

# Good: Single batch request
get_agents_batch(agent_ids)
```

### 4. Cache Responses

Cache API responses to reduce request frequency:

```python
import requests_cache

# Cache responses for 5 minutes
requests_cache.install_cache(expire_after=300)

# Subsequent identical requests use cache
response = requests.get(url)
```

### 5. Implement Client-Side Rate Limiting

Enforce rate limits on client side to prevent server rejections:

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=1000, period=60)  # Match server limits
def api_call():
    return requests.get(url)
```

## Configuration

### Environment Variables

Configure rate limiting behavior through environment variables:

```bash
# Enable/disable rate limiting
GATEWAY_RATE_LIMIT_ENABLED=true

# Redis connection
GATEWAY_RATE_LIMIT_REDIS_URL=redis://localhost:6379/2

# Algorithm selection
GATEWAY_RATE_LIMIT_ALGORITHM=sliding_window

# Per-IP limits
GATEWAY_RATE_LIMIT_CLIENT_IP_LIMIT=1000
GATEWAY_RATE_LIMIT_CLIENT_IP_WINDOW=60

# Per-user limits
GATEWAY_RATE_LIMIT_USER_LIMIT=5000
GATEWAY_RATE_LIMIT_USER_WINDOW=60

# Per-endpoint limits
GATEWAY_RATE_LIMIT_ENDPOINT_LIMIT=100
GATEWAY_RATE_LIMIT_ENDPOINT_WINDOW=60

# DDoS protection
GATEWAY_DDOS_PROTECTION_ENABLED=true
GATEWAY_DDOS_GLOBAL_REQUESTS_PER_SECOND=10000
GATEWAY_DDOS_IP_REQUESTS_PER_SECOND=100
GATEWAY_DDOS_AUTO_BLOCKING_ENABLED=true
```

## Monitoring

### Metrics

Rate limiting metrics are available at `/metrics`:

```
# Request rate by client
rate_limit_requests_total{client="192.168.1.100",policy="client_ip"} 875

# Rate limit rejections
rate_limit_rejections_total{policy="client_ip"} 12

# Current usage
rate_limit_usage{client="user123",policy="user"} 0.75
```

### Logging

Rate limit events are logged with structured logging:

```json
{
  "event": "rate_limit_exceeded",
  "client_ip": "192.168.1.100",
  "user_id": "user123",
  "policy": "client_ip",
  "limit": 1000,
  "window": 60,
  "timestamp": "2025-10-18T10:30:00Z"
}
```

## Troubleshooting

### Issue: Constant 429 Errors

**Symptoms:** Receiving 429 errors on every request

**Solutions:**
1. Check if IP is blocked: Contact support to unblock
2. Verify rate limit headers to understand current usage
3. Implement proper backoff and retry logic
4. Use authenticated requests for higher limits

### Issue: Unexpected Rate Limit Resets

**Symptoms:** Rate limits reset unexpectedly

**Causes:**
- Clock drift on Redis server
- Redis cache eviction due to memory pressure
- Algorithm change (fixed window has hard resets)

**Solutions:**
1. Use sliding window algorithm for smooth limits
2. Ensure Redis has sufficient memory
3. Monitor Redis performance metrics

### Issue: Burst Traffic Rejection

**Symptoms:** Burst traffic is rejected despite low average rate

**Solutions:**
1. Switch to token bucket algorithm for burst handling
2. Increase burst threshold multiplier
3. Implement client-side request queuing

## Enterprise Options

For enterprise customers requiring higher limits:

### Custom Rate Limits

Contact support to configure custom limits:
- Per-organization limits
- Per-API key limits
- Geographic distribution
- Priority queuing

### Dedicated Infrastructure

Enterprise plans include:
- Dedicated rate limiting tier
- Custom Redis cluster
- Priority support
- SLA guarantees

## Next Steps

- [Authentication Guide](authentication.md)
- [Real-time Communication](realtime.md)
- [Error Handling](errors.md)
- [Code Examples](../examples/)
