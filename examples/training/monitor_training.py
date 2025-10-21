"""
Training Job Monitoring Example

Demonstrates how to monitor a training job's progress and retrieve final results.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

# Configuration
API_URL = "http://localhost:8001/api/v1/jsonrpc"
JWT_TOKEN = "your-jwt-token-here"  # Replace with actual token
POLL_INTERVAL = 10  # seconds


async def get_job_status(job_id: str) -> dict[str, Any]:
    """Get current status of a training job."""
    request = {
        "jsonrpc": "2.0",
        "method": "training.get_status",
        "params": {"job_id": job_id},
        "id": "status-001",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=request,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()

    if "error" in result:
        raise Exception(f"Failed to get status: {result['error']}")

    return result["result"]


async def monitor_job(job_id: str) -> dict[str, Any]:
    """Monitor a training job until completion."""
    print(f"Monitoring job {job_id}...")
    print(f"Polling every {POLL_INTERVAL} seconds\n")

    while True:
        status = await get_job_status(job_id)

        # Display current status
        print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] Status Update:")
        print(f"  Status: {status['status']}")
        print(
            f"  Progress: {status['current_iteration']}/{status['total_iterations']} "
            f"({status['progress_percent']:.1f}%)"
        )

        if "metrics" in status:
            metrics = status["metrics"]
            print(f"  Metrics:")
            print(f"    - Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
            print(
                f"    - Validation Accuracy: {metrics.get('validation_accuracy', 'N/A'):.2f}"
            )
            print(f"    - Avg Reward: {metrics.get('avg_reward', 'N/A'):.2f}")
            print(f"    - Trajectories: {metrics.get('trajectories_generated', 'N/A')}")

        print(
            f"  Cost: ${status['cost_usd']} / ${status['budget_usd']} "
            f"({status['budget_remaining_percent']:.1f}% remaining)"
        )

        if "estimated_completion" in status:
            print(f"  ETA: {status['estimated_completion']}")

        # Check if job is complete
        if status["status"] in ["completed", "failed", "cancelled"]:
            print(f"\n{'=' * 60}")
            print(f"Job {status['status'].upper()}!")

            if status["status"] == "completed":
                print(f"  Best Checkpoint: {status.get('best_checkpoint_id', 'N/A')}")
                print(f"  Final Cost: ${status['cost_usd']}")
                print(f"\n✓ Training completed successfully!")
            elif status["status"] == "failed":
                print(f"  Error: {status.get('error_message', 'Unknown error')}")
                print(f"\n✗ Training failed!")
            else:
                print(f"\n⚠ Training was cancelled!")

            return status

        print()  # Blank line for readability
        await asyncio.sleep(POLL_INTERVAL)


async def evaluate_job(job_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Run evaluation on trained agent."""
    request = {
        "jsonrpc": "2.0",
        "method": "training.evaluate",
        "params": {
            "job_id": job_id,
            "checkpoint_id": checkpoint_id,  # None uses best checkpoint
        },
        "id": "eval-001",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=request,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=300.0,  # Evaluation can take longer
        )
        response.raise_for_status()
        result = response.json()

    if "error" in result:
        raise Exception(f"Evaluation failed: {result['error']}")

    return result["result"]


async def main(job_id: str) -> None:
    """Main execution."""
    try:
        # Monitor job until completion
        final_status = await monitor_job(job_id)

        # If completed successfully, run evaluation
        if final_status["status"] == "completed":
            print(f"\nRunning evaluation on best checkpoint...")
            eval_results = await evaluate_job(
                job_id, final_status.get("best_checkpoint_id")
            )

            print(f"\n{'=' * 60}")
            print("EVALUATION RESULTS:")
            print(f"  Success Rate: {eval_results['metrics']['success_rate']:.2%}")
            print(f"  Avg Reward: {eval_results['metrics']['avg_reward']:.4f}")
            print(f"  Avg Steps: {eval_results['metrics']['avg_steps']:.1f}")
            print(f"  Tool Accuracy: {eval_results['metrics']['tool_accuracy']:.2%}")

            if "baseline_comparison" in eval_results:
                comp = eval_results["baseline_comparison"]
                print(f"\n  Baseline Comparison:")
                print(
                    f"    - Success Rate Improvement: {comp['success_rate_improvement']:+.2%}"
                )
                print(
                    f"    - Avg Reward Improvement: {comp['avg_reward_improvement']:+.4f}"
                )
                print(
                    f"    - Statistical Significance: {'Yes' if comp['statistically_significant'] else 'No'} "
                    f"(p={comp['p_value']:.4f})"
                )

            print(f"\n  Queries Evaluated: {eval_results['queries_evaluated']}")
            print(f"  Duration: {eval_results['evaluation_duration_ms'] / 1000:.1f}s")

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    asyncio.run(main(job_id))
