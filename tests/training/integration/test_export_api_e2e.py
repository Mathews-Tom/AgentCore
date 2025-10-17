"""
End-to-end integration tests for data export API (FLOW-014).

Tests the complete trajectory export workflow including:
- Trajectory export with filters
- Pagination support
- Success/failure filtering
- Reward threshold filtering
- Export size limits
- Authorization checks
"""

from __future__ import annotations

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from agentcore.training.models import (
    Trajectory,
    TrajectoryStep,
)


@pytest.fixture
def sample_trajectories_for_export() -> list[Trajectory]:
    """Create sample trajectories for export testing."""
    trajectories = []

    for i in range(150):  # Create 150 trajectories
        steps = [
            TrajectoryStep(
                state={"step": j},
                action={"type": "action", "value": j},
                result={"success": True},
                timestamp=datetime.now(timezone.utc),
                duration_ms=100 + j,
            )
            for j in range(3)
        ]

        trajectory = Trajectory(
            job_id=uuid4(),
            agent_id="test_agent",
            query=f"Query {i}",
            steps=steps,
            success=i % 3 != 0,  # ~67% success rate
            reward=0.1 + (i * 0.005),  # Rewards from 0.1 to 0.85
            execution_time_ms=300 + i * 10,
        )

        trajectories.append(trajectory)

    return trajectories


class TestExportAPI:
    """Integration tests for trajectory export API."""

    @pytest.mark.asyncio
    async def test_basic_trajectory_export(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test basic trajectory export without filters."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        # Store trajectories (simulated)
        await export_service.store_trajectories(
            job_id, sample_trajectories_for_export[:50]
        )

        # Export all trajectories
        exported = await export_service.export_trajectories(
            job_id=job_id,
            limit=100,
            offset=0,
        )

        # Verify export
        assert len(exported["trajectories"]) == 50
        assert exported["total_count"] == 50
        assert exported["has_more"] is False

    @pytest.mark.asyncio
    async def test_export_with_success_filter(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test export with success_only filter."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        # Store trajectories
        await export_service.store_trajectories(job_id, sample_trajectories_for_export[:30])

        # Export only successful trajectories
        exported = await export_service.export_trajectories(
            job_id=job_id,
            success_only=True,
            limit=100,
        )

        # Verify all exported are successful
        assert all(t["success"] for t in exported["trajectories"])

        # Count successful (every 3rd fails, so ~20 out of 30)
        successful_count = sum(1 for t in sample_trajectories_for_export[:30] if t.success)
        assert len(exported["trajectories"]) == successful_count

    @pytest.mark.asyncio
    async def test_export_with_reward_threshold(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test export with minimum reward threshold."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        await export_service.store_trajectories(job_id, sample_trajectories_for_export[:100])

        # Export with minimum reward of 0.5
        exported = await export_service.export_trajectories(
            job_id=job_id,
            min_reward=0.5,
            limit=200,
        )

        # Verify all meet threshold
        assert all(t["reward"] >= 0.5 for t in exported["trajectories"])

        # Count trajectories with reward >= 0.5
        high_reward_count = sum(1 for t in sample_trajectories_for_export[:100] if t.reward and t.reward >= 0.5)
        assert len(exported["trajectories"]) == high_reward_count

    @pytest.mark.asyncio
    async def test_export_pagination(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test pagination support in export."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        # Store 150 trajectories
        await export_service.store_trajectories(job_id, sample_trajectories_for_export)

        # First page (limit 50)
        page1 = await export_service.export_trajectories(
            job_id=job_id,
            limit=50,
            offset=0,
        )

        assert len(page1["trajectories"]) == 50
        assert page1["total_count"] == 150
        assert page1["has_more"] is True

        # Second page
        page2 = await export_service.export_trajectories(
            job_id=job_id,
            limit=50,
            offset=50,
        )

        assert len(page2["trajectories"]) == 50
        assert page2["has_more"] is True

        # Third page (last 50)
        page3 = await export_service.export_trajectories(
            job_id=job_id,
            limit=50,
            offset=100,
        )

        assert len(page3["trajectories"]) == 50
        assert page3["has_more"] is False

        # Verify no overlap
        page1_ids = {t["trajectory_id"] for t in page1["trajectories"]}
        page2_ids = {t["trajectory_id"] for t in page2["trajectories"]}
        page3_ids = {t["trajectory_id"] for t in page3["trajectories"]}

        assert page1_ids.isdisjoint(page2_ids)
        assert page2_ids.isdisjoint(page3_ids)
        assert page1_ids.isdisjoint(page3_ids)

    @pytest.mark.asyncio
    async def test_export_size_limit_enforcement(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test that export size limits are enforced (max 10,000)."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        await export_service.store_trajectories(job_id, sample_trajectories_for_export)

        # Attempt to export more than limit
        with pytest.raises(ValueError, match="limit exceeds maximum"):
            await export_service.export_trajectories(
                job_id=job_id,
                limit=15000,  # Exceeds 10,000 limit
            )

        # Valid limit should work
        exported = await export_service.export_trajectories(
            job_id=job_id,
            limit=10000,
        )

        assert len(exported["trajectories"]) <= 10000

    @pytest.mark.asyncio
    async def test_export_combined_filters(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test export with multiple filters combined."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        await export_service.store_trajectories(job_id, sample_trajectories_for_export[:100])

        # Export with success_only AND min_reward
        exported = await export_service.export_trajectories(
            job_id=job_id,
            success_only=True,
            min_reward=0.4,
            limit=100,
        )

        # Verify all meet both criteria
        for trajectory in exported["trajectories"]:
            assert trajectory["success"] is True
            assert trajectory["reward"] >= 0.4

    @pytest.mark.asyncio
    async def test_export_json_format(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test that export JSON format is correct."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        await export_service.store_trajectories(job_id, sample_trajectories_for_export[:5])

        exported = await export_service.export_trajectories(job_id=job_id, limit=5)

        # Verify structure
        assert "job_id" in exported
        assert "trajectories" in exported
        assert "total_count" in exported
        assert "returned_count" in exported
        assert "has_more" in exported

        # Verify trajectory structure
        trajectory = exported["trajectories"][0]
        assert "trajectory_id" in trajectory
        assert "query" in trajectory
        assert "steps" in trajectory
        assert "reward" in trajectory
        assert "success" in trajectory
        assert "execution_time_ms" in trajectory

        # Verify step structure
        step = trajectory["steps"][0]
        assert "state" in step
        assert "action" in step
        assert "result" in step
        assert "duration_ms" in step

    @pytest.mark.asyncio
    async def test_export_authorization(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test that export requires data:export permission."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        await export_service.store_trajectories(job_id, sample_trajectories_for_export[:10])

        # Attempt export without permission (mock user)
        class MockUser:
            permissions = []  # No permissions

        with pytest.raises(PermissionError, match="data:export"):
            await export_service.export_trajectories(
                job_id=job_id,
                limit=10,
                user=MockUser(),
            )

        # With permission should work
        class AuthorizedUser:
            permissions = ["data:export"]

        exported = await export_service.export_trajectories(
            job_id=job_id,
            limit=10,
            user=AuthorizedUser(),
        )

        assert len(exported["trajectories"]) > 0

    @pytest.mark.asyncio
    async def test_export_empty_job(self) -> None:
        """Test export for job with no trajectories."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        # Export from empty job
        exported = await export_service.export_trajectories(job_id=job_id, limit=100)

        # Verify empty result
        assert exported["total_count"] == 0
        assert exported["returned_count"] == 0
        assert len(exported["trajectories"]) == 0
        assert exported["has_more"] is False

    @pytest.mark.asyncio
    async def test_export_nonexistent_job(self) -> None:
        """Test export for non-existent job."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        nonexistent_job_id = uuid4()

        # Should raise error for non-existent job
        with pytest.raises(KeyError, match="Job not found"):
            await export_service.export_trajectories(
                job_id=nonexistent_job_id,
                limit=10,
            )

    @pytest.mark.asyncio
    async def test_export_performance_with_large_dataset(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test export performance with large trajectory dataset."""
        from agentcore.training.export import ExportService
        import time

        export_service = ExportService()
        job_id = uuid4()

        # Store all 150 trajectories
        await export_service.store_trajectories(job_id, sample_trajectories_for_export)

        # Time the export
        start_time = time.time()

        exported = await export_service.export_trajectories(
            job_id=job_id,
            limit=100,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Verify export completed in reasonable time (<2 seconds)
        assert duration < 2.0
        assert len(exported["trajectories"]) == 100

    @pytest.mark.asyncio
    async def test_export_offset_beyond_total(
        self,
        sample_trajectories_for_export: list[Trajectory],
    ) -> None:
        """Test export with offset beyond total count."""
        from agentcore.training.export import ExportService

        export_service = ExportService()
        job_id = uuid4()

        await export_service.store_trajectories(job_id, sample_trajectories_for_export[:50])

        # Export with offset beyond total
        exported = await export_service.export_trajectories(
            job_id=job_id,
            limit=10,
            offset=100,  # Beyond 50 trajectories
        )

        # Should return empty result
        assert len(exported["trajectories"]) == 0
        assert exported["returned_count"] == 0
        assert exported["has_more"] is False
