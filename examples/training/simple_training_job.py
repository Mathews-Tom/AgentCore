"""
Simple Training Job Example

Demonstrates how to start a basic GRPO training job using the Training API.
"""

import asyncio
from typing import Any

import httpx

# Configuration
API_URL = "http://localhost:8001/api/v1/jsonrpc"
JWT_TOKEN = "your-jwt-token-here"  # Replace with actual token


async def start_training_job() -> dict[str, Any]:
    """Start a simple training job."""

    # Prepare training queries (minimum 100 required)
    training_data = [
        {
            "query": f"Write a function to {task}",
            "expected_outcome": {"test_passed": True, "execution_time_ms": 100},
        }
        for task in [
            "sort a list",
            "reverse a string",
            "find max element",
            "calculate factorial",
            "check if palindrome",
            # ... add 95 more queries to reach minimum of 100
        ]
        * 20  # Repeat to get 100+ queries
    ]

    # Training configuration
    config = {
        "n_iterations": 100,
        "batch_size": 16,
        "n_trajectories_per_query": 8,
        "learning_rate": 0.0001,
        "max_budget_usd": 10.00,
        "checkpoint_interval": 10,
        "max_steps_per_trajectory": 20,
        "gamma": 0.99,
    }

    # JSON-RPC request
    request = {
        "jsonrpc": "2.0",
        "method": "training.start_grpo",
        "params": {
            "agent_id": "code-generator-v1",
            "config": config,
            "training_data": training_data,
        },
        "id": "train-001",
    }

    # Send request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=request,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

    # Check for errors
    if "error" in result:
        raise Exception(f"Training job failed: {result['error']}")

    return result["result"]


async def main() -> None:
    """Main execution."""
    print("Starting training job...")

    try:
        result = await start_training_job()

        print(f"\n✓ Training job created successfully!")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Agent ID: {result['agent_id']}")
        print(f"  Total Iterations: {result['total_iterations']}")
        print(f"  Created At: {result['created_at']}")

    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
