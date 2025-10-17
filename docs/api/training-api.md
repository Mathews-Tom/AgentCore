# Training API Reference

**Version:** 1.0
**Protocol:** JSON-RPC 2.0
**Base Path:** `/api/v1/jsonrpc`
**Last Updated:** 2025-10-17

---

## Overview

The AgentCore Training API provides JSON-RPC 2.0 endpoints for managing GRPO (Group Refined Policy Optimization) training jobs. All endpoints require JWT authentication and follow the JSON-RPC 2.0 specification.

**Key Features:**
- Start and manage training jobs
- Monitor training progress
- Evaluate trained agents
- Export training data
- Budget enforcement
- Checkpoint management

---

## Authentication

All training API calls require JWT authentication:

```http
POST /api/v1/jsonrpc
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Required Permissions:**
- `training:start` - Start training jobs
- `training:view` - View job status
- `training:cancel` - Cancel running jobs
- `training:evaluate` - Run evaluations
- `data:export` - Export trajectories

---

## JSON-RPC 2.0 Format

### Request

```json
{
  "jsonrpc": "2.0",
  "method": "training.start_grpo",
  "params": {
    "agent_id": "my-agent-v1",
    "config": { ... }
  },
  "id": "req-123"
}
```

### Successful Response

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued"
  },
  "id": "req-123"
}
```

### Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32600,
    "message": "Invalid training configuration",
    "data": {
      "field": "n_iterations",
      "error": "Must be between 1 and 10000"
    }
  },
  "id": "req-123"
}
```

---

## API Methods

### 1. training.start_grpo

Start a new GRPO training job.

**Method:** `training.start_grpo`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Agent identifier to train |
| `config` | GRPOConfig | Yes | Training configuration |
| `training_data` | TrainingQuery[] | Yes | Training queries (min 100) |
| `evaluation_data` | TrainingQuery[] | No | Held-out evaluation queries |

**GRPOConfig Schema:**

```typescript
{
  n_iterations: number          // 1-10000, default: 1000
  batch_size: number           // 1-128, default: 16
  n_trajectories_per_query: number  // 1-16, default: 8
  learning_rate: number        // 0.0-1.0, default: 0.0001
  max_budget_usd: number       // Min: 0, default: 100.00
  checkpoint_interval: number  // Min: 1, default: 10
  max_steps_per_trajectory: number  // 1-100, default: 20
  gamma: number               // 0.0-1.0, default: 0.99
}
```

**TrainingQuery Schema:**

```typescript
{
  query: string                    // User query/task prompt
  expected_outcome: object         // Expected result for reward
}
```

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.start_grpo",
  "params": {
    "agent_id": "code-generator-v2",
    "config": {
      "n_iterations": 500,
      "batch_size": 16,
      "n_trajectories_per_query": 8,
      "learning_rate": 0.0001,
      "max_budget_usd": 50.00,
      "checkpoint_interval": 10
    },
    "training_data": [
      {
        "query": "Write a function to sort a list",
        "expected_outcome": {
          "test_passed": true,
          "execution_time_ms": 100
        }
      },
      // ... 99+ more queries
    ],
    "evaluation_data": [
      {
        "query": "Write a binary search function",
        "expected_outcome": {
          "test_passed": true
        }
      }
      // ... more evaluation queries
    ]
  },
  "id": "train-001"
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "agent_id": "code-generator-v2",
    "total_iterations": 500,
    "created_at": "2025-10-17T10:30:00Z"
  },
  "id": "train-001"
}
```

**Error Codes:**

- `-32600` Invalid Request - Missing or invalid parameters
- `-32603` Internal Error - Database or system error
- `-40001` Budget Exceeded - Insufficient budget
- `-40002` Validation Failed - Training data validation failed

---

### 2. training.get_status

Get current status and metrics for a training job.

**Method:** `training.get_status`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | UUID | Yes | Training job identifier |

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.get_status",
  "params": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "id": "status-001"
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_id": "code-generator-v2",
    "status": "running",
    "current_iteration": 150,
    "total_iterations": 500,
    "progress_percent": 30.0,
    "metrics": {
      "train_loss": 0.234,
      "validation_accuracy": 0.82,
      "avg_reward": 0.65,
      "trajectories_generated": 1200
    },
    "cost_usd": "12.50",
    "budget_usd": "50.00",
    "budget_remaining_percent": 75.0,
    "started_at": "2025-10-17T10:30:05Z",
    "estimated_completion": "2025-10-17T12:15:00Z",
    "best_checkpoint_id": "abc-123-def"
  },
  "id": "status-001"
}
```

**Job Status Values:**

- `queued` - Job waiting in queue
- `running` - Training in progress
- `completed` - Training finished successfully
- `failed` - Training encountered error
- `cancelled` - Job cancelled by user

**Error Codes:**

- `-32602` Invalid Params - Invalid job_id format
- `-32001` Not Found - Job not found

---

### 3. training.cancel

Cancel a running or queued training job.

**Method:** `training.cancel`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | UUID | Yes | Training job identifier |
| `reason` | string | No | Cancellation reason |

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.cancel",
  "params": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "reason": "Budget exceeded manually"
  },
  "id": "cancel-001"
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "cancelled",
    "cancelled_at": "2025-10-17T11:00:00Z",
    "iterations_completed": 150,
    "checkpoint_saved": true,
    "checkpoint_id": "abc-123-def"
  },
  "id": "cancel-001"
}
```

**Error Codes:**

- `-32001` Not Found - Job not found
- `-40003` Cannot Cancel - Job already completed/cancelled

---

### 4. training.evaluate

Run evaluation on trained agent with held-out queries.

**Method:** `training.evaluate`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | UUID | Yes | Training job identifier |
| `checkpoint_id` | UUID | No | Specific checkpoint (default: best) |
| `evaluation_queries` | TrainingQuery[] | No | Custom queries (uses job eval data if not provided) |

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.evaluate",
  "params": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "checkpoint_id": "abc-123-def"
  },
  "id": "eval-001"
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "checkpoint_id": "abc-123-def",
    "metrics": {
      "success_rate": 0.85,
      "avg_reward": 0.72,
      "avg_steps": 8.5,
      "tool_accuracy": 0.90
    },
    "baseline_comparison": {
      "success_rate_improvement": 0.15,
      "avg_reward_improvement": 0.22,
      "p_value": 0.001,
      "statistically_significant": true
    },
    "queries_evaluated": 50,
    "evaluation_duration_ms": 45000
  },
  "id": "eval-001"
}
```

**Error Codes:**

- `-32001` Not Found - Job or checkpoint not found
- `-40004` Evaluation Failed - Evaluation execution error

---

### 5. training.export_trajectories

Export trajectory data for analysis or debugging.

**Method:** `training.export_trajectories`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | UUID | Yes | Training job identifier |
| `success_only` | boolean | No | Only export successful trajectories |
| `min_reward` | number | No | Minimum reward threshold |
| `limit` | number | No | Max trajectories (default: 1000, max: 10000) |
| `offset` | number | No | Pagination offset (default: 0) |

**Example Request:**

```json
{
  "jsonrpc": "2.0",
  "method": "training.export_trajectories",
  "params": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "success_only": true,
    "min_reward": 0.5,
    "limit": 100,
    "offset": 0
  },
  "id": "export-001"
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "trajectories": [
      {
        "trajectory_id": "traj-001",
        "query": "Write a sorting function",
        "steps": [
          {
            "state": { "context": "..." },
            "action": { "tool": "code_generator", "params": "..." },
            "result": { "code": "def sort(arr): ..." },
            "timestamp": "2025-10-17T10:30:10Z",
            "duration_ms": 500
          }
        ],
        "reward": 0.85,
        "normalized_reward": 1.2,
        "advantage": 0.8,
        "success": true,
        "execution_time_ms": 1500
      }
      // ... more trajectories
    ],
    "total_count": 1250,
    "returned_count": 100,
    "has_more": true
  },
  "id": "export-001"
}
```

**Error Codes:**

- `-32001` Not Found - Job not found
- `-40005` Export Limit Exceeded - Requested more than max limit

---

## Batch Requests

Multiple API calls can be batched in a single HTTP request:

```json
[
  {
    "jsonrpc": "2.0",
    "method": "training.get_status",
    "params": { "job_id": "job-1" },
    "id": "1"
  },
  {
    "jsonrpc": "2.0",
    "method": "training.get_status",
    "params": { "job_id": "job-2" },
    "id": "2"
  }
]
```

**Batch Response:**

```json
[
  {
    "jsonrpc": "2.0",
    "result": { "job_id": "job-1", "status": "running", ... },
    "id": "1"
  },
  {
    "jsonrpc": "2.0",
    "result": { "job_id": "job-2", "status": "completed", ... },
    "id": "2"
  }
]
```

---

## Error Codes Reference

| Code | Name | Description |
|------|------|-------------|
| `-32700` | Parse Error | Invalid JSON |
| `-32600` | Invalid Request | Missing required fields |
| `-32601` | Method Not Found | Unknown method |
| `-32602` | Invalid Params | Invalid parameter types |
| `-32603` | Internal Error | Server error |
| `-32001` | Not Found | Resource not found |
| `-40001` | Budget Exceeded | Training budget limit reached |
| `-40002` | Validation Failed | Data validation failed |
| `-40003` | Cannot Cancel | Job in non-cancellable state |
| `-40004` | Evaluation Failed | Evaluation execution error |
| `-40005` | Export Limit Exceeded | Too many trajectories requested |

---

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `training.start_grpo` | 10 requests | 1 minute |
| `training.get_status` | 100 requests | 1 minute |
| `training.cancel` | 20 requests | 1 minute |
| `training.evaluate` | 5 requests | 1 minute |
| `training.export_trajectories` | 10 requests | 1 minute |

**Rate Limit Headers:**

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1634567890
```

---

## Webhooks

Training jobs can trigger webhooks for event notifications:

**Supported Events:**
- `training.job.started`
- `training.job.progress` (every 10 iterations)
- `training.job.completed`
- `training.job.failed`
- `training.job.cancelled`
- `training.budget.warning` (75%, 90% thresholds)

**Webhook Payload:**

```json
{
  "event": "training.job.completed",
  "timestamp": "2025-10-17T12:00:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_id": "code-generator-v2",
    "status": "completed",
    "final_metrics": {
      "success_rate": 0.85,
      "cost_usd": "45.50"
    }
  }
}
```

---

## SDKs

### Python

```python
from agentcore.sdk import TrainingClient

client = TrainingClient(
    base_url="https://api.agentcore.ai",
    api_key="your-api-key"
)

# Start training job
job = client.start_training(
    agent_id="my-agent",
    config={
        "n_iterations": 500,
        "batch_size": 16
    },
    training_data=my_queries
)

# Monitor status
status = client.get_status(job["job_id"])
print(f"Progress: {status['progress_percent']}%")
```

### cURL

```bash
curl -X POST https://api.agentcore.ai/api/v1/jsonrpc \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "training.get_status",
    "params": {
      "job_id": "550e8400-e29b-41d4-a716-446655440000"
    },
    "id": "1"
  }'
```

---

## Best Practices

1. **Polling Intervals**: Poll `training.get_status` every 30-60 seconds
2. **Budget Monitoring**: Set budget alerts at 75% and 90%
3. **Checkpointing**: Use `checkpoint_interval=10` for resumability
4. **Evaluation Data**: Hold out 20% of queries for validation
5. **Batch Size**: Start with `batch_size=16` for most use cases
6. **Learning Rate**: Use default `0.0001` unless experiencing issues

---

## Further Reading

- [Developer Guide](../guides/training-agents.md)
- [Operational Runbook](../ops/training-runbook.md)
- [Custom Rewards Guide](../guides/custom_rewards.md)
- [Architecture Overview](../architecture/training-system.md)
