# Training Examples

This directory contains practical examples for using the AgentCore training infrastructure.

## Prerequisites

```bash
# Install dependencies
uv add httpx structlog

# Set JWT token
export JWT_TOKEN="your-jwt-token-here"
```

## Examples

### 1. simple_training_job.py

Start a basic GRPO training job.

```bash
uv run python examples/training/simple_training_job.py
```

**Demonstrates:**
- Creating training data
- Configuring GRPO parameters
- Starting a training job via JSON-RPC API

### 2. monitor_training.py

Monitor a training job's progress until completion.

```bash
uv run python examples/training/monitor_training.py <job_id>
```

**Demonstrates:**
- Polling job status
- Displaying progress metrics
- Running evaluation after completion

### 3. custom_reward_example.py

Create and use custom reward functions.

```bash
uv run python examples/training/custom_reward_example.py
```

**Demonstrates:**
- Defining custom reward functions
- Registering functions in RewardRegistry
- Configuring agent-type strategies
- Computing rewards with RewardEngine

### 4. batch_training.py

Manage multiple training jobs concurrently.

```bash
uv run python examples/training/batch_training.py
```

**Demonstrates:**
- Batch JSON-RPC requests
- Starting multiple jobs in parallel
- Monitoring batch progress
- Exporting trajectories from multiple jobs
- Cancelling jobs

## API Documentation

For complete API reference, see:
- [Training API Reference](../../docs/api/training-api.md)
- [Developer Guide](../../docs/guides/training-agents.md)
- [Custom Rewards Guide](../../docs/guides/custom_rewards.md)

## Configuration

Update the following constants in each example:
- `API_URL`: Training API endpoint (default: `http://localhost:8001/api/v1/jsonrpc`)
- `JWT_TOKEN`: Your authentication token

## Further Reading

- [Operational Runbook](../../docs/ops/training-runbook.md)
- [Training System Architecture](../../docs/architecture/training-system.md)
