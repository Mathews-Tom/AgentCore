"""
Unit tests for Hierarchical Pattern Implementation.

Tests multi-level hierarchies, delegation, escalation, authority management,
and communication flow optimization.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from agentcore.orchestration.patterns.hierarchical import (
    AuthorityLevel,
    DelegationPolicy,
    EscalationReason,
    HierarchicalConfig,
    HierarchicalCoordinator)


class TestHierarchicalCoordinator:
    """Test suite for HierarchicalCoordinator."""

    @pytest.fixture
    def config(self) -> HierarchicalConfig:
        """Create test configuration."""
        return HierarchicalConfig(
            max_hierarchy_depth=5,
            delegation_policy=DelegationPolicy.STRICT_HIERARCHY,
            enable_escalation=True,
            escalation_threshold_failures=2)

    @pytest.fixture
    def coordinator(self, config: HierarchicalConfig) -> HierarchicalCoordinator:
        """Create hierarchical coordinator instance."""
        return HierarchicalCoordinator(
            coordinator_id="test-hierarchy",
            config=config)

    @pytest.mark.asyncio
    async def test_add_agent(self, coordinator: HierarchicalCoordinator) -> None:
        """Test adding agent to hierarchy."""
        # Add root agent
        await coordinator.add_agent(
            agent_id="ceo",
            authority_level=AuthorityLevel.EXECUTIVE,
            capabilities=["strategy", "decision_making"])

        # Verify hierarchy
        tree = await coordinator.get_hierarchy_tree()
        assert tree["total_agents"] == 1
        assert len(tree["roots"]) == 1
        assert tree["roots"][0]["agent_id"] == "ceo"

    @pytest.mark.asyncio
    async def test_add_child_agent(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test adding child agents."""
        # Build hierarchy: CEO -> Manager -> Worker
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo"
        )
        await coordinator.add_agent(
            "worker", AuthorityLevel.WORKER, parent_id="manager"
        )

        # Verify hierarchy
        tree = await coordinator.get_hierarchy_tree()
        assert tree["total_agents"] == 3
        assert tree["max_depth"] == 3

        # Check structure
        root = tree["roots"][0]
        assert root["agent_id"] == "ceo"
        assert len(root["children"]) == 1
        assert root["children"][0]["agent_id"] == "manager"
        assert len(root["children"][0]["children"]) == 1
        assert root["children"][0]["children"][0]["agent_id"] == "worker"

    @pytest.mark.asyncio
    async def test_remove_agent(self, coordinator: HierarchicalCoordinator) -> None:
        """Test removing agent from hierarchy."""
        # Build hierarchy
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo"
        )
        await coordinator.add_agent(
            "worker", AuthorityLevel.WORKER, parent_id="manager"
        )

        # Remove manager (worker should move to ceo)
        await coordinator.remove_agent("manager")

        # Verify
        tree = await coordinator.get_hierarchy_tree()
        assert tree["total_agents"] == 2
        assert tree["roots"][0]["children"][0]["agent_id"] == "worker"

    @pytest.mark.asyncio
    async def test_delegate_task(self, coordinator: HierarchicalCoordinator) -> None:
        """Test task delegation."""
        # Build hierarchy
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo", capabilities=["task_a"]
        )

        # Delegate task
        task_id = uuid4()
        target = await coordinator.delegate_task(
            task_id=task_id,
            from_agent_id="ceo",
            task_data={"type": "test"},
            required_capabilities=["task_a"])

        # Verify delegation
        assert target == "manager"
        status = await coordinator.get_hierarchy_status()
        assert status["active_delegations"] == 1

    @pytest.mark.asyncio
    async def test_delegate_task_no_suitable_agent(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test delegation failure when no suitable agent exists."""
        # Add agent without required capability
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo", capabilities=["task_a"]
        )

        # Try to delegate with different capability
        task_id = uuid4()
        target = await coordinator.delegate_task(
            task_id=task_id,
            from_agent_id="ceo",
            task_data={"type": "test"},
            required_capabilities=["task_b"],  # Manager doesn't have this
        )

        # Should fail
        assert target is None

    @pytest.mark.asyncio
    async def test_escalate_task(self, coordinator: HierarchicalCoordinator) -> None:
        """Test task escalation."""
        # Build hierarchy
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo"
        )

        # Escalate task from manager to ceo
        task_id = uuid4()
        target = await coordinator.escalate_task(
            task_id=task_id,
            from_agent_id="manager",
            reason=EscalationReason.INSUFFICIENT_AUTHORITY,
            context={"details": "needs executive approval"})

        # Verify escalation
        assert target == "ceo"
        status = await coordinator.get_hierarchy_status()
        assert status["escalations"] == 1

    @pytest.mark.asyncio
    async def test_escalate_from_root(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test escalation from root agent (should fail)."""
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)

        # Try to escalate from root
        task_id = uuid4()
        target = await coordinator.escalate_task(
            task_id=task_id,
            from_agent_id="ceo",
            reason=EscalationReason.EXPLICIT_REQUEST)

        # Should fail (no parent)
        assert target is None

    @pytest.mark.asyncio
    async def test_automatic_escalation_on_failure(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test automatic escalation after failure threshold."""
        # Build hierarchy
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "worker", AuthorityLevel.WORKER, parent_id="ceo"
        )

        task_id = uuid4()

        # First failure - should not escalate
        escalated = await coordinator.handle_task_failure(
            task_id, "worker", "Error 1"
        )
        assert not escalated

        # Second failure - should escalate (threshold=2)
        escalated = await coordinator.handle_task_failure(
            task_id, "worker", "Error 2"
        )
        assert escalated

    @pytest.mark.asyncio
    async def test_check_authority(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test authority checking."""
        # Add agents with different levels
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent("manager", AuthorityLevel.SENIOR)
        await coordinator.add_agent("worker", AuthorityLevel.WORKER)

        # CEO can do everything
        assert await coordinator.check_authority("ceo", AuthorityLevel.EXECUTIVE)
        assert await coordinator.check_authority("ceo", AuthorityLevel.SENIOR)
        assert await coordinator.check_authority("ceo", AuthorityLevel.WORKER)

        # Manager can do SENIOR and below
        assert not await coordinator.check_authority("manager", AuthorityLevel.EXECUTIVE)
        assert await coordinator.check_authority("manager", AuthorityLevel.SENIOR)
        assert await coordinator.check_authority("manager", AuthorityLevel.WORKER)

        # Worker can only do WORKER level
        assert not await coordinator.check_authority("worker", AuthorityLevel.SENIOR)
        assert await coordinator.check_authority("worker", AuthorityLevel.WORKER)

    @pytest.mark.asyncio
    async def test_permission_management(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test permission granting and revoking."""
        await coordinator.add_agent("agent1", AuthorityLevel.SENIOR)

        # Initially no permission
        assert not await coordinator.check_authority(
            "agent1", AuthorityLevel.SENIOR, permission="admin"
        )

        # Grant permission
        await coordinator.grant_permission("agent1", "admin")
        assert await coordinator.check_authority(
            "agent1", AuthorityLevel.SENIOR, permission="admin"
        )

        # Revoke permission
        await coordinator.revoke_permission("agent1", "admin")
        assert not await coordinator.check_authority(
            "agent1", AuthorityLevel.SENIOR, permission="admin"
        )

    @pytest.mark.asyncio
    async def test_get_communication_path(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test communication path finding."""
        # Build hierarchy: CEO -> (Manager1, Manager2) -> (Worker1, Worker2)
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager1", AuthorityLevel.SENIOR, parent_id="ceo"
        )
        await coordinator.add_agent(
            "manager2", AuthorityLevel.SENIOR, parent_id="ceo"
        )
        await coordinator.add_agent(
            "worker1", AuthorityLevel.WORKER, parent_id="manager1"
        )
        await coordinator.add_agent(
            "worker2", AuthorityLevel.WORKER, parent_id="manager2"
        )

        # Path between workers should go through CEO
        path = await coordinator.get_communication_path("worker1", "worker2")
        assert "ceo" in path
        assert path.index("worker1") < path.index("ceo") < path.index("worker2")

    @pytest.mark.asyncio
    async def test_delegation_policy_strict(self) -> None:
        """Test strict hierarchy delegation policy."""
        config = HierarchicalConfig(
            delegation_policy=DelegationPolicy.STRICT_HIERARCHY
        )
        coordinator = HierarchicalCoordinator("test", config)

        # Build hierarchy: CEO -> Manager -> Worker
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo", capabilities=["task"]
        )
        await coordinator.add_agent(
            "worker",
            AuthorityLevel.WORKER,
            parent_id="manager",
            capabilities=["task"])

        # Delegate from CEO should go to Manager (direct child), not Worker
        task_id = uuid4()
        target = await coordinator.delegate_task(
            task_id, "ceo", {}, required_capabilities=["task"]
        )
        assert target == "manager"

    @pytest.mark.asyncio
    async def test_delegation_policy_best_fit(self) -> None:
        """Test best fit delegation policy."""
        config = HierarchicalConfig(delegation_policy=DelegationPolicy.BEST_FIT)
        coordinator = HierarchicalCoordinator("test", config)

        # Build hierarchy where worker has better capability match
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager", AuthorityLevel.SENIOR, parent_id="ceo", capabilities=["basic"]
        )
        await coordinator.add_agent(
            "worker",
            AuthorityLevel.WORKER,
            parent_id="manager",
            capabilities=["task"])

        # With best fit, should find worker with matching capability
        task_id = uuid4()
        target = await coordinator.delegate_task(
            task_id, "ceo", {}, required_capabilities=["task"]
        )
        assert target == "worker"

    @pytest.mark.asyncio
    async def test_hierarchy_status(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test hierarchy status reporting."""
        # Build multi-level hierarchy
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "manager1", AuthorityLevel.SENIOR, parent_id="ceo"
        )
        await coordinator.add_agent(
            "manager2", AuthorityLevel.SENIOR, parent_id="ceo"
        )
        await coordinator.add_agent(
            "worker", AuthorityLevel.WORKER, parent_id="manager1"
        )

        # Get status
        status = await coordinator.get_hierarchy_status()

        assert status["total_agents"] == 4
        assert status["root_agents"] == 1
        assert status["max_depth"] == 3
        assert status["agents_by_level"]["EXECUTIVE"] == 1
        assert status["agents_by_level"]["SENIOR"] == 2
        assert status["agents_by_level"]["WORKER"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test thread safety with concurrent operations."""
        # Add agents concurrently
        tasks = []
        for i in range(10):
            tasks.append(
                coordinator.add_agent(f"agent-{i}", AuthorityLevel.WORKER)
            )
        await asyncio.gather(*tasks)

        tree = await coordinator.get_hierarchy_tree()
        assert tree["total_agents"] == 10

    @pytest.mark.asyncio
    async def test_max_concurrent_tasks_limit(
        self, coordinator: HierarchicalCoordinator
    ) -> None:
        """Test that delegation respects max concurrent tasks limit."""
        # Add agents
        await coordinator.add_agent("ceo", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent(
            "worker",
            AuthorityLevel.WORKER,
            parent_id="ceo",
            capabilities=["task"])

        # Set max concurrent to 2
        async with coordinator._lock:
            coordinator._hierarchy["worker"].max_concurrent_tasks = 2

        # Delegate 3 tasks - third should fail
        task1 = uuid4()
        task2 = uuid4()
        task3 = uuid4()

        target1 = await coordinator.delegate_task(task1, "ceo", {})
        target2 = await coordinator.delegate_task(task2, "ceo", {})
        target3 = await coordinator.delegate_task(task3, "ceo", {})

        assert target1 == "worker"
        assert target2 == "worker"
        assert target3 is None  # Exceeded capacity


class TestHierarchyEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_delegate_nonexistent_agent(self) -> None:
        """Test delegation from non-existent agent."""
        coordinator = HierarchicalCoordinator("test")

        task_id = uuid4()
        target = await coordinator.delegate_task(
            task_id, "nonexistent", {}
        )
        assert target is None

    @pytest.mark.asyncio
    async def test_escalate_nonexistent_agent(self) -> None:
        """Test escalation from non-existent agent."""
        coordinator = HierarchicalCoordinator("test")

        task_id = uuid4()
        target = await coordinator.escalate_task(
            task_id,
            "nonexistent",
            EscalationReason.EXPLICIT_REQUEST)
        assert target is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_agent(self) -> None:
        """Test removing non-existent agent (should not error)."""
        coordinator = HierarchicalCoordinator("test")
        await coordinator.remove_agent("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_deep_hierarchy(self) -> None:
        """Test deep hierarchy creation."""
        coordinator = HierarchicalCoordinator("test")

        # Create 5-level hierarchy
        await coordinator.add_agent("L0", AuthorityLevel.EXECUTIVE)
        await coordinator.add_agent("L1", AuthorityLevel.SENIOR, parent_id="L0")
        await coordinator.add_agent(
            "L2", AuthorityLevel.INTERMEDIATE, parent_id="L1"
        )
        await coordinator.add_agent("L3", AuthorityLevel.JUNIOR, parent_id="L2")
        await coordinator.add_agent("L4", AuthorityLevel.WORKER, parent_id="L3")

        tree = await coordinator.get_hierarchy_tree()
        assert tree["max_depth"] == 5
