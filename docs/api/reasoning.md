# Reasoning API Reference

## Overview

The Reasoning API provides a unified interface for executing various reasoning strategies through AgentCore. It supports multiple reasoning approaches (Chain of Thought, Bounded Context, ReAct) with automatic strategy selection based on request, agent, and system configuration.

**Base Endpoint:** `/api/v1/jsonrpc`

**Protocol:** JSON-RPC 2.0

**Authentication:** JWT token required (see [Authentication](./authentication.md))

## Quick Start

```python
import httpx

# Execute reasoning with automatic strategy selection
response = httpx.post("http://localhost:8001/api/v1/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "reasoning.execute",
    "params": {
        "auth_token": "eyJhbGciOiJIUzI1NiIs...",
        "query": "What are the implications of quantum computing on cryptography?",
        "strategy": "bounded_context"  # optional
    },
    "id": 1
})

result = response.json()["result"]
print(result["answer"])
```

## JSON-RPC Method

### `reasoning.execute`

Execute reasoning using a specified or automatically-selected strategy.

**Method Name:** `reasoning.execute`

**Request Schema:**

```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "auth_token": "string (required)",
    "query": "string (required, 1-100,000 chars)",
    "strategy": "string (optional)",
    "strategy_config": "object (optional)",
    "agent_capabilities": "array (optional)"
  },
  "id": "integer|string (required)"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `auth_token` | string | Yes | JWT authentication token with `reasoning:execute` permission |
| `query` | string | Yes | The problem or question to solve (1-100,000 characters) |
| `strategy` | string | No | Reasoning strategy to use: `bounded_context`, `chain_of_thought`, `react`. If omitted, system uses default or agent-preferred strategy |
| `strategy_config` | object | No | Strategy-specific configuration parameters (see [Strategy-Specific Parameters](#strategy-specific-parameters)) |
| `agent_capabilities` | array | No | Agent capabilities for capability-based strategy selection |

**Response Schema:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "answer": "string",
    "strategy_used": "string",
    "metrics": {
      "total_tokens": "integer",
      "execution_time_ms": "integer",
      "strategy_specific": "object"
    },
    "trace": "array (optional)"
  },
  "id": "integer|string"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Final answer from reasoning process |
| `strategy_used` | string | Strategy that was executed |
| `metrics.total_tokens` | integer | Total tokens consumed |
| `metrics.execution_time_ms` | integer | Total execution time in milliseconds |
| `metrics.strategy_specific` | object | Strategy-specific metrics (varies by strategy) |
| `trace` | array | Optional execution trace with intermediate steps |

## Strategy-Specific Parameters

### Chain of Thought

Simple single-pass reasoning with explicit step-by-step thinking.

**When to use:**
- Simple to medium complexity problems
- Problems that fit in one context window
- When low latency is important
- When no external data is needed

**Configuration:**

```json
{
  "strategy": "chain_of_thought",
  "strategy_config": {
    "max_tokens": 4096,
    "temperature": 0.7,
    "show_reasoning": true
  }
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_tokens` | integer | 4096 | 100-32768 | Maximum tokens for output |
| `temperature` | float | 0.7 | 0.0-2.0 | Sampling temperature for generation |
| `show_reasoning` | boolean | true | - | Include reasoning trace in response |

**Strategy-Specific Metrics:**

```json
{
  "temperature": 0.7,
  "max_tokens": 4096,
  "finish_reason": "stop",
  "model": "gpt-4"
}
```

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "query": "Calculate the compound interest on $10,000 at 5% for 3 years",
    "strategy": "chain_of_thought",
    "strategy_config": {
      "temperature": 0.5,
      "show_reasoning": true
    }
  },
  "id": 1
}
```

**Example Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "answer": "$11,576.25",
    "strategy_used": "chain_of_thought",
    "metrics": {
      "total_tokens": 450,
      "execution_time_ms": 1200,
      "strategy_specific": {
        "temperature": 0.5,
        "max_tokens": 4096,
        "finish_reason": "stop",
        "model": "gpt-4"
      }
    },
    "trace": [
      {
        "type": "reasoning",
        "content": "Step 1: Identify the formula...\nStep 2: Calculate..."
      },
      {
        "type": "answer",
        "content": "$11,576.25"
      }
    ]
  },
  "id": 1
}
```

### Bounded Context

Multi-iteration reasoning with fixed context windows and carryover compression.

**When to use:**
- Large/complex problems requiring multiple steps
- Long-form content analysis
- When token efficiency is important
- When consistent memory usage is needed

**Configuration:**

```json
{
  "strategy": "bounded_context",
  "strategy_config": {
    "chunk_size": 8192,
    "carryover_size": 4096,
    "max_iterations": 5
  }
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `chunk_size` | integer | 8192 | 1024-32768 | Maximum tokens per iteration |
| `carryover_size` | integer | 4096 | 512-16384 | Tokens to carry forward between iterations |
| `max_iterations` | integer | 5 | 1-50 | Maximum reasoning iterations |

**Strategy-Specific Metrics:**

```json
{
  "total_iterations": 3,
  "compute_savings_pct": 45.2,
  "iterations": [
    {
      "iteration": 0,
      "tokens": 8192,
      "has_answer": false,
      "carryover_generated": true,
      "execution_time_ms": 2500
    },
    {
      "iteration": 1,
      "tokens": 7500,
      "has_answer": false,
      "carryover_generated": true,
      "execution_time_ms": 2300
    },
    {
      "iteration": 2,
      "tokens": 4800,
      "has_answer": true,
      "carryover_generated": false,
      "execution_time_ms": 1800
    }
  ]
}
```

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "query": "Analyze this 50-page research paper and summarize the key findings, methodology, and implications...",
    "strategy": "bounded_context",
    "strategy_config": {
      "chunk_size": 8192,
      "max_iterations": 10
    }
  },
  "id": 1
}
```

**Example Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "answer": "The research paper presents three key findings...",
    "strategy_used": "bounded_context",
    "metrics": {
      "total_tokens": 24500,
      "execution_time_ms": 15000,
      "strategy_specific": {
        "total_iterations": 4,
        "compute_savings_pct": 52.3,
        "iterations": [...]
      }
    },
    "trace": [...]
  },
  "id": 1
}
```

### ReAct (Reasoning + Acting)

Iterative thought-action-observation cycles with optional tool execution.

**When to use:**
- Problems requiring external data/tools
- Tasks needing verification
- Interactive exploration scenarios
- When action execution is part of solution

**Configuration:**

```json
{
  "strategy": "react",
  "strategy_config": {
    "max_iterations": 10,
    "max_tokens_per_step": 2048,
    "temperature": 0.7,
    "allow_tool_use": false
  }
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_iterations` | integer | 10 | 1-50 | Maximum thought-action-observation cycles |
| `max_tokens_per_step` | integer | 2048 | 100-8192 | Maximum tokens per reasoning step |
| `temperature` | float | 0.7 | 0.0-2.0 | Sampling temperature |
| `allow_tool_use` | boolean | false | - | Enable external tool/API calls (disabled by default for safety) |

**Strategy-Specific Metrics:**

```json
{
  "total_iterations": 4,
  "answer_found_at_iteration": 3,
  "temperature": 0.7,
  "max_iterations": 10
}
```

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "query": "Find the current weather in New York and recommend appropriate clothing",
    "strategy": "react",
    "strategy_config": {
      "max_iterations": 5,
      "allow_tool_use": true
    }
  },
  "id": 1
}
```

**Example Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "answer": "Based on current weather (65°F, partly cloudy), recommend light jacket...",
    "strategy_used": "react",
    "metrics": {
      "total_tokens": 3200,
      "execution_time_ms": 8500,
      "strategy_specific": {
        "total_iterations": 3,
        "answer_found_at_iteration": 2,
        "temperature": 0.7,
        "max_iterations": 5
      }
    },
    "trace": [
      {
        "iteration": 0,
        "thought": "I need to search for current weather in New York",
        "action": "Search for New York weather",
        "observation": "Simulated search result...",
        "answer_found": false,
        "tokens": 1000
      },
      {
        "iteration": 1,
        "thought": "Now I can recommend clothing based on 65°F",
        "action": "Answer",
        "observation": "Based on current weather...",
        "answer_found": true,
        "tokens": 1200
      }
    ]
  },
  "id": 1
}
```

## Strategy Selection

### Automatic Strategy Selection

If no strategy is specified, the system selects a strategy based on:

1. **Agent-preferred strategy** (from AgentCard capabilities)
2. **System default strategy** (from configuration)

### Strategy Selection Precedence

The system uses the following precedence order:

1. **Request-level**: Strategy specified in `params.strategy`
2. **Agent-level**: Agent's preferred strategy from capabilities
3. **System-level**: Default strategy from `config.toml`

**Example with agent capabilities:**

```json
{
  "jsonrpc": "2.0",
  "method": "reasoning.execute",
  "params": {
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "query": "Complex analysis task...",
    "agent_capabilities": [
      "reasoning.strategy.bounded_context",
      "reasoning.strategy.chain_of_thought"
    ]
  },
  "id": 1
}
```

The system will use the first available strategy from agent capabilities.

## Error Handling

### Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32602 | Invalid params | Request validation failed (invalid parameters) |
| -32603 | Internal error | Authentication, authorization, or strategy execution failure |
| -32001 | Strategy not found | Requested strategy is not registered or available |
| -32002 | Strategy not supported | Agent doesn't support requested strategy |

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Strategy not found: 'nonexistent_strategy'"
  },
  "id": 1
}
```

### Common Errors

**Authentication Error:**

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32603,
    "message": "Authentication required: Missing JWT token. Provide 'auth_token' in params"
  },
  "id": 1
}
```

**Authorization Error:**

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32603,
    "message": "Authorization failed: Missing required permission 'reasoning:execute'"
  },
  "id": 1
}
```

**Validation Error:**

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Request validation failed: query must be between 1 and 100000 characters"
  },
  "id": 1
}
```

**Strategy Not Found:**

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Strategy not found: 'custom_strategy'"
  },
  "id": 1
}
```

## Authentication

All reasoning requests require authentication with a JWT token that has the `reasoning:execute` permission.

**Obtaining a Token:**

```python
import httpx

# Request token (example - actual implementation may vary)
response = httpx.post("http://localhost:8001/api/v1/auth/token", json={
    "agent_id": "my-agent",
    "secret": "my-secret"
})

token = response.json()["access_token"]
```

**Token Requirements:**

- Must include `reasoning:execute` permission
- Must be valid (not expired)
- Must belong to an active agent or user

See [Authentication Documentation](./authentication.md) for details.

## Rate Limiting

Reasoning requests may be subject to rate limiting based on:

- Token usage per minute
- Concurrent requests per agent
- Total requests per hour

See [Rate Limiting Documentation](./rate-limiting.md) for details.

## Best Practices

### 1. Choose the Right Strategy

**Use Chain of Thought when:**
- Problem is simple to medium complexity
- Fits in one context window (< 4K tokens)
- Low latency is critical
- No external data needed

**Use Bounded Context when:**
- Problem is large/complex (> 10K tokens)
- Token efficiency is important
- Memory usage must be predictable
- Multi-step reasoning needed

**Use ReAct when:**
- External tools/APIs are needed
- Problem requires verification
- Real-time data is required
- Interactive exploration is helpful

### 2. Optimize Token Usage

**For Bounded Context:**
- Set `chunk_size` based on problem complexity (4K-8K typical)
- Set `carryover_size` to 50% of `chunk_size`
- Increase `max_iterations` for very large problems

**For Chain of Thought:**
- Set `max_tokens` based on expected answer length
- Use lower `temperature` (0.3-0.5) for factual answers
- Use higher `temperature` (0.7-0.9) for creative tasks

**For ReAct:**
- Set `max_iterations` based on expected action count
- Set `max_tokens_per_step` to 1K-2K for efficiency
- Only enable `allow_tool_use` when needed

### 3. Handle Errors Gracefully

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def execute_reasoning(query: str, strategy: str):
    try:
        response = await httpx.AsyncClient().post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.execute",
                "params": {
                    "auth_token": token,
                    "query": query,
                    "strategy": strategy
                },
                "id": 1
            }
        )

        result = response.json()

        if "error" in result:
            raise ValueError(f"Reasoning failed: {result['error']['message']}")

        return result["result"]

    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        raise
```

### 4. Monitor Performance

Track these metrics:

- **Token usage**: Monitor costs and optimize prompts
- **Execution time**: Identify slow queries
- **Success rate**: Track error patterns
- **Strategy distribution**: Understand usage patterns

### 5. Security

- **Never log tokens**: Tokens contain sensitive information
- **Validate inputs**: Sanitize user queries before sending
- **Use HTTPS**: Encrypt all communication
- **Rotate tokens**: Regularly refresh authentication tokens

## Code Examples

### Python

```python
import httpx
import asyncio

async def reasoning_example():
    async with httpx.AsyncClient() as client:
        # Chain of Thought example
        response = await client.post(
            "http://localhost:8001/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.execute",
                "params": {
                    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
                    "query": "What is 15% of 240?",
                    "strategy": "chain_of_thought"
                },
                "id": 1
            }
        )

        result = response.json()["result"]
        print(f"Answer: {result['answer']}")
        print(f"Tokens used: {result['metrics']['total_tokens']}")

asyncio.run(reasoning_example())
```

### JavaScript

```javascript
const axios = require('axios');

async function reasoningExample() {
  const response = await axios.post('http://localhost:8001/api/v1/jsonrpc', {
    jsonrpc: '2.0',
    method: 'reasoning.execute',
    params: {
      auth_token: 'eyJhbGciOiJIUzI1NiIs...',
      query: 'Explain quantum entanglement',
      strategy: 'chain_of_thought'
    },
    id: 1
  });

  const result = response.data.result;
  console.log(`Answer: ${result.answer}`);
  console.log(`Tokens: ${result.metrics.total_tokens}`);
}

reasoningExample();
```

### cURL

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.execute",
    "params": {
      "auth_token": "eyJhbGciOiJIUzI1NiIs...",
      "query": "Calculate compound interest",
      "strategy": "chain_of_thought",
      "strategy_config": {
        "temperature": 0.5
      }
    },
    "id": 1
  }'
```

## Performance Characteristics

### Latency

| Strategy | Small Query | Medium Query | Large Query |
|----------|-------------|--------------|-------------|
| Chain of Thought | 1-2s | 2-5s | N/A |
| Bounded Context | 3-5s | 8-15s | 15-30s |
| ReAct | 5-10s | 15-30s | 30-60s |

### Token Usage

| Strategy | Small (<2K) | Medium (2-10K) | Large (10-50K) |
|----------|-------------|----------------|----------------|
| Chain of Thought | 1,000 | 5,000 | N/A |
| Bounded Context | 2,000 | 8,000 | 25,000 |
| ReAct | 3,000 | 10,000 | 30,000 |

### Cost Estimates (GPT-4)

| Strategy | Small Query | Medium Query | Large Query |
|----------|-------------|--------------|-------------|
| Chain of Thought | $0.01 | $0.05 | N/A |
| Bounded Context | $0.02 | $0.08 | $0.25 |
| ReAct | $0.03 | $0.10 | $0.30 |

*Assumes GPT-4 pricing: $0.01/1K prompt tokens, $0.03/1K completion tokens*

## See Also

- [Strategy Comparison Guide](../reasoning/strategy-comparison.md)
- [Getting Started with Reasoning](../reasoning/getting-started.md)
- [Architecture Overview](../reasoning/architecture.md)
- [Authentication Documentation](./authentication.md)
- [Error Handling](./errors.md)
