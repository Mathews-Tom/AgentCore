"""
Batch Training Operations Example

Demonstrates how to manage multiple training jobs concurrently using batch JSON-RPC requests.
"""

import asyncio
import httpx
from typing import Any


# Configuration
API_URL = "http://localhost:8001/api/v1/jsonrpc"
JWT_TOKEN = "your-jwt-token-here"  # Replace with actual token


async def start_multiple_jobs(agents: list[str]) -> list[dict[str, Any]]:
    """
    Start training jobs for multiple agents using batch JSON-RPC request.

    Args:
        agents: List of agent IDs to train

    Returns:
        List of job creation results
    """
    # Common training data
    training_data = [
        {
            "query": f"Task {i}",
            "expected_outcome": {"success": True}
        }
        for i in range(100)  # Minimum 100 queries
    ]

    # Common config
    config = {
        "n_iterations": 50,
        "batch_size": 16,
        "n_trajectories_per_query": 4,
        "learning_rate": 0.0001,
        "max_budget_usd": 5.00,
    }

    # Create batch request (multiple jobs)
    batch_request = [
        {
            "jsonrpc": "2.0",
            "method": "training.start_grpo",
            "params": {
                "agent_id": agent_id,
                "config": config,
                "training_data": training_data,
            },
            "id": f"train-{i}",
        }
        for i, agent_id in enumerate(agents)
    ]

    # Send batch request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=batch_request,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=30.0,
        )
        response.raise_for_status()
        results = response.json()

    return results


async def check_multiple_statuses(job_ids: list[str]) -> list[dict[str, Any]]:
    """
    Check status of multiple jobs using batch JSON-RPC request.

    Args:
        job_ids: List of job IDs to check

    Returns:
        List of job statuses
    """
    # Create batch status request
    batch_request = [
        {
            "jsonrpc": "2.0",
            "method": "training.get_status",
            "params": {"job_id": job_id},
            "id": f"status-{i}",
        }
        for i, job_id in enumerate(job_ids)
    ]

    # Send batch request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=batch_request,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json()

    return results


async def cancel_jobs(job_ids: list[str], reason: str = "Batch cancellation") -> list[dict[str, Any]]:
    """
    Cancel multiple jobs using batch JSON-RPC request.

    Args:
        job_ids: List of job IDs to cancel
        reason: Cancellation reason

    Returns:
        List of cancellation results
    """
    # Create batch cancel request
    batch_request = [
        {
            "jsonrpc": "2.0",
            "method": "training.cancel",
            "params": {
                "job_id": job_id,
                "reason": reason,
            },
            "id": f"cancel-{i}",
        }
        for i, job_id in enumerate(job_ids)
    ]

    # Send batch request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=batch_request,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json()

    return results


async def export_trajectories_parallel(job_ids: list[str]) -> dict[str, Any]:
    """
    Export trajectories from multiple jobs in parallel.

    Args:
        job_ids: List of job IDs to export from

    Returns:
        Dictionary mapping job_id to trajectories
    """
    async def export_single_job(job_id: str) -> tuple[str, list[dict[str, Any]]]:
        """Export trajectories for a single job."""
        request = {
            "jsonrpc": "2.0",
            "method": "training.export_trajectories",
            "params": {
                "job_id": job_id,
                "success_only": True,
                "limit": 100,
            },
            "id": f"export-{job_id}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL,
                json=request,
                headers={"Authorization": f"Bearer {JWT_TOKEN}"},
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

        if "error" in result:
            return job_id, []

        return job_id, result["result"]["trajectories"]

    # Export all jobs in parallel
    tasks = [export_single_job(job_id) for job_id in job_ids]
    results = await asyncio.gather(*tasks)

    return dict(results)


async def monitor_batch_progress(job_ids: list[str], interval: int = 30) -> None:
    """
    Monitor multiple training jobs until all complete.

    Args:
        job_ids: List of job IDs to monitor
        interval: Poll interval in seconds
    """
    print(f"Monitoring {len(job_ids)} jobs...")
    print(f"Polling every {interval} seconds\n")

    completed = set()

    while len(completed) < len(job_ids):
        # Get status for all jobs
        statuses = await check_multiple_statuses(job_ids)

        print(f"{'='*80}")
        print(f"Batch Status Update ({len(completed)}/{len(job_ids)} completed)")
        print(f"{'='*80}")

        for status_response in statuses:
            if "error" in status_response:
                job_id = status_response["id"].replace("status-", "")
                print(f"✗ {job_id}: ERROR - {status_response['error']['message']}")
                completed.add(job_id)
                continue

            status = status_response["result"]
            job_id = status["job_id"]

            # Display status
            status_icon = {
                "queued": "⏳",
                "running": "🔄",
                "completed": "✓",
                "failed": "✗",
                "cancelled": "⚠",
            }.get(status["status"], "?")

            progress = f"{status.get('progress_percent', 0):.1f}%"
            cost = f"${status.get('cost_usd', 0)} / ${status.get('budget_usd', 0)}"

            print(f"{status_icon} {job_id[:8]}... | {status['agent_id']:<20} | "
                  f"{status['status']:<10} | {progress:>6} | {cost}")

            # Mark as complete if terminal state
            if status["status"] in ["completed", "failed", "cancelled"]:
                completed.add(job_id)

        print()

        # Wait before next poll (unless all done)
        if len(completed) < len(job_ids):
            await asyncio.sleep(interval)

    print(f"{'='*80}")
    print(f"All jobs completed!")
    print(f"{'='*80}")


async def main() -> None:
    """Main execution."""
    print("Batch Training Operations Example\n")

    # Define agents to train
    agents = [
        "code-generator-v1",
        "code-generator-v2",
        "test-generator-v1",
        "doc-generator-v1",
    ]

    try:
        # 1. Start multiple jobs
        print(f"Starting training jobs for {len(agents)} agents...")
        start_results = await start_multiple_jobs(agents)

        job_ids = []
        for result in start_results:
            if "error" in result:
                print(f"✗ Failed to start job: {result['error']['message']}")
                continue

            job_info = result["result"]
            job_ids.append(job_info["job_id"])
            print(f"✓ Job started: {job_info['job_id']} ({job_info['agent_id']})")

        if not job_ids:
            print("No jobs started successfully.")
            return

        print(f"\n{len(job_ids)} jobs started successfully!\n")

        # 2. Monitor progress
        await monitor_batch_progress(job_ids, interval=30)

        # 3. Get final statuses
        print("Fetching final statuses...")
        final_statuses = await check_multiple_statuses(job_ids)

        completed_jobs = []
        failed_jobs = []

        for status_response in final_statuses:
            if "error" not in status_response:
                status = status_response["result"]
                if status["status"] == "completed":
                    completed_jobs.append(status["job_id"])
                elif status["status"] == "failed":
                    failed_jobs.append(status["job_id"])

        print(f"\nResults:")
        print(f"  ✓ Completed: {len(completed_jobs)}")
        print(f"  ✗ Failed: {len(failed_jobs)}")

        # 4. Export trajectories from successful jobs
        if completed_jobs:
            print(f"\nExporting trajectories from {len(completed_jobs)} completed jobs...")
            trajectories = await export_trajectories_parallel(completed_jobs)

            total_trajectories = sum(len(trajs) for trajs in trajectories.values())
            print(f"✓ Exported {total_trajectories} trajectories")

            for job_id, trajs in trajectories.items():
                print(f"  - {job_id[:8]}...: {len(trajs)} trajectories")

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        print("Cancelling running jobs...")

        try:
            cancel_results = await cancel_jobs(job_ids, "User interrupted")
            print(f"✓ Cancelled {len(cancel_results)} jobs")
        except Exception as e:
            print(f"✗ Error cancelling jobs: {e}")

    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
