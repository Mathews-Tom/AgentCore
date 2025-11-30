"""
Integration Tests for PEVG Module A2A Registration

Tests the registration, discovery, and health checking of all four PEVG modules
(Planner, Executor, Verifier, Generator) as A2A agents.

Coverage:
- Module registration with AgentManager
- AgentCard creation and validation
- Discovery via /.well-known/agents/{module-id}
- Health check endpoints
- Capability advertisement
- Unregistration on shutdown
"""

import pytest

from agentcore.a2a_protocol.models.agent import AgentStatus
from agentcore.a2a_protocol.services.agent_manager import agent_manager
from agentcore.modular.registration import (
    create_executor_agent_card,
    create_generator_agent_card,
    create_planner_agent_card,
    create_verifier_agent_card,
    get_module_health_status,
    register_module_agents,
    unregister_module_agents,
)


@pytest.mark.asyncio
@pytest.mark.integration
class TestModuleRegistration:
    """Test PEVG module registration as A2A agents."""

    async def test_create_planner_agent_card(self) -> None:
        """Test Planner AgentCard creation."""
        agent_card = create_planner_agent_card()

        # Verify core fields
        assert agent_card.agent_id == "agentcore.modular.planner"
        assert agent_card.agent_name == "Modular Agent - Planner"
        assert agent_card.agent_version == "1.0.0"
        assert agent_card.status == AgentStatus.ACTIVE

        # Verify capabilities
        capability_names = [cap.name for cap in agent_card.capabilities]
        assert "planning" in capability_names
        assert "task-decomposition" in capability_names
        assert "workflow-design" in capability_names
        assert "plan-refinement" in capability_names
        assert len(agent_card.capabilities) == 4

        # Verify each capability has cost and latency
        for cap in agent_card.capabilities:
            assert cap.cost_per_request is not None
            assert cap.cost_per_request >= 0.0
            assert cap.avg_latency_ms is not None
            assert cap.avg_latency_ms > 0.0
            assert cap.quality_score is not None
            assert 0.0 <= cap.quality_score <= 1.0

        # Verify endpoints
        assert len(agent_card.endpoints) == 1
        assert agent_card.endpoints[0].protocols == ["jsonrpc-2.0"]

        # Verify metadata
        assert agent_card.metadata is not None
        assert "planning" in agent_card.metadata.tags
        assert "pevg" in agent_card.metadata.tags
        assert agent_card.metadata.category == "modular-agent"

    async def test_create_executor_agent_card(self) -> None:
        """Test Executor AgentCard creation."""
        agent_card = create_executor_agent_card()

        # Verify core fields
        assert agent_card.agent_id == "agentcore.modular.executor"
        assert agent_card.agent_name == "Modular Agent - Executor"
        assert agent_card.status == AgentStatus.ACTIVE

        # Verify capabilities
        capability_names = [cap.name for cap in agent_card.capabilities]
        assert "tool-execution" in capability_names
        assert "step-execution" in capability_names
        assert "retry-logic" in capability_names
        assert "circuit-breaker" in capability_names
        assert len(agent_card.capabilities) == 4

        # Verify circuit-breaker capability has proper config
        circuit_breaker_cap = next(
            cap for cap in agent_card.capabilities if cap.name == "circuit-breaker"
        )
        assert circuit_breaker_cap.parameters["enabled"] is True
        assert "failure_threshold" in circuit_breaker_cap.parameters

    async def test_create_verifier_agent_card(self) -> None:
        """Test Verifier AgentCard creation."""
        agent_card = create_verifier_agent_card()

        # Verify core fields
        assert agent_card.agent_id == "agentcore.modular.verifier"
        assert agent_card.agent_name == "Modular Agent - Verifier"
        assert agent_card.status == AgentStatus.ACTIVE

        # Verify capabilities
        capability_names = [cap.name for cap in agent_card.capabilities]
        assert "result-validation" in capability_names
        assert "quality-assurance" in capability_names
        assert "feedback-generation" in capability_names
        assert "consistency-checking" in capability_names
        assert len(agent_card.capabilities) == 4

        # Verify validation capability parameters
        validation_cap = next(
            cap for cap in agent_card.capabilities if cap.name == "result-validation"
        )
        assert "validation_types" in validation_cap.parameters
        assert "confidence_threshold" in validation_cap.parameters
        assert validation_cap.parameters["confidence_threshold"] == 0.7

    async def test_create_generator_agent_card(self) -> None:
        """Test Generator AgentCard creation."""
        agent_card = create_generator_agent_card()

        # Verify core fields
        assert agent_card.agent_id == "agentcore.modular.generator"
        assert agent_card.agent_name == "Modular Agent - Generator"
        assert agent_card.status == AgentStatus.ACTIVE

        # Verify capabilities
        capability_names = [cap.name for cap in agent_card.capabilities]
        assert "response-synthesis" in capability_names
        assert "output-formatting" in capability_names
        assert "explanation-generation" in capability_names
        assert "source-tracking" in capability_names
        assert len(agent_card.capabilities) == 4

        # Verify output formats supported
        formatting_cap = next(
            cap for cap in agent_card.capabilities if cap.name == "output-formatting"
        )
        assert "formats" in formatting_cap.parameters
        supported_formats = formatting_cap.parameters["formats"]
        assert "text" in supported_formats
        assert "json" in supported_formats
        assert "markdown" in supported_formats
        assert "html" in supported_formats

    async def test_register_module_agents(self) -> None:
        """Test registration of all PEVG modules."""
        # Register modules
        discovery_urls = await register_module_agents()

        # Verify all modules registered
        assert len(discovery_urls) == 4
        assert "planner" in discovery_urls
        assert "executor" in discovery_urls
        assert "verifier" in discovery_urls
        assert "generator" in discovery_urls

        # Verify discovery URLs
        for module_name, url in discovery_urls.items():
            assert url.startswith("/.well-known/agents/")
            assert f"agentcore.modular.{module_name}" in url

        # Verify modules are retrievable from AgentManager
        planner = await agent_manager.get_agent("agentcore.modular.planner")
        assert planner is not None
        assert planner.agent_name == "Modular Agent - Planner"

        executor = await agent_manager.get_agent("agentcore.modular.executor")
        assert executor is not None
        assert executor.agent_name == "Modular Agent - Executor"

        verifier = await agent_manager.get_agent("agentcore.modular.verifier")
        assert verifier is not None
        assert verifier.agent_name == "Modular Agent - Verifier"

        generator = await agent_manager.get_agent("agentcore.modular.generator")
        assert generator is not None
        assert generator.agent_name == "Modular Agent - Generator"

        # Clean up
        await unregister_module_agents()

    async def test_register_modules_allows_override(self) -> None:
        """Test that re-registration is allowed with override flag."""
        # Register modules first time
        await register_module_agents()

        # Re-register (should succeed with override=True)
        discovery_urls = await register_module_agents()

        assert len(discovery_urls) == 4

        # Clean up
        await unregister_module_agents()

    async def test_unregister_module_agents(self) -> None:
        """Test unregistration of all PEVG modules."""
        # Register modules first
        await register_module_agents()

        # Verify modules are registered
        planner = await agent_manager.get_agent("agentcore.modular.planner")
        assert planner is not None

        # Unregister modules
        await unregister_module_agents()

        # Verify modules are unregistered
        planner = await agent_manager.get_agent("agentcore.modular.planner")
        assert planner is None

        executor = await agent_manager.get_agent("agentcore.modular.executor")
        assert executor is None

        verifier = await agent_manager.get_agent("agentcore.modular.verifier")
        assert verifier is None

        generator = await agent_manager.get_agent("agentcore.modular.generator")
        assert generator is None

    async def test_module_discovery_by_id(self) -> None:
        """Test module discovery by agent ID."""
        # Register modules
        await register_module_agents()

        # Discover each module by ID
        planner_summary = await agent_manager.get_agent_summary(
            "agentcore.modular.planner"
        )
        assert planner_summary is not None
        assert planner_summary["agent_id"] == "agentcore.modular.planner"
        assert "planning" in planner_summary["capabilities"]

        executor_summary = await agent_manager.get_agent_summary(
            "agentcore.modular.executor"
        )
        assert executor_summary is not None
        assert executor_summary["agent_id"] == "agentcore.modular.executor"
        assert "tool-execution" in executor_summary["capabilities"]

        # Clean up
        await unregister_module_agents()

    async def test_module_discovery_by_capability(self) -> None:
        """Test discovering modules by capability."""
        # Register modules
        await register_module_agents()

        # Discover modules with planning capability
        from agentcore.a2a_protocol.models.agent import AgentDiscoveryQuery

        query = AgentDiscoveryQuery(capabilities=["planning"])
        discovery_response = await agent_manager.discover_agents(query)

        assert discovery_response.total_count >= 1
        planner_found = False
        for agent_summary in discovery_response.agents:
            if agent_summary["agent_id"] == "agentcore.modular.planner":
                planner_found = True
                assert "planning" in agent_summary["capabilities"]
                break
        assert planner_found

        # Discover modules with tool-execution capability
        query = AgentDiscoveryQuery(capabilities=["tool-execution"])
        discovery_response = await agent_manager.discover_agents(query)

        assert discovery_response.total_count >= 1
        executor_found = False
        for agent_summary in discovery_response.agents:
            if agent_summary["agent_id"] == "agentcore.modular.executor":
                executor_found = True
                assert "tool-execution" in agent_summary["capabilities"]
                break
        assert executor_found

        # Clean up
        await unregister_module_agents()

    async def test_get_module_health_status(self) -> None:
        """Test getting health status for all modules."""
        # Register modules
        await register_module_agents()

        # Get health status
        health = await get_module_health_status()

        # Verify overall status
        assert health["overall_status"] == "healthy"
        assert health["total_count"] == 4
        assert health["registered_count"] == 4

        # Verify each module has health info
        assert "planner" in health["modules"]
        assert "executor" in health["modules"]
        assert "verifier" in health["modules"]
        assert "generator" in health["modules"]

        # Verify planner health
        planner_health = health["modules"]["planner"]
        assert planner_health["status"] == "active"
        assert planner_health["agent_id"] == "agentcore.modular.planner"
        assert planner_health["is_active"] is True
        assert planner_health["capabilities_count"] == 4

        # Clean up
        await unregister_module_agents()

    async def test_module_health_status_when_not_registered(self) -> None:
        """Test health status when modules are not registered."""
        # Ensure modules are not registered
        await unregister_module_agents()

        # Get health status
        health = await get_module_health_status()

        # Verify overall status is degraded
        assert health["overall_status"] == "degraded"
        assert health["registered_count"] == 0

        # Verify each module shows not registered
        for module_name in ["planner", "executor", "verifier", "generator"]:
            module_health = health["modules"][module_name]
            assert module_health["status"] == "not_registered"

    async def test_module_capability_costs_and_latency(self) -> None:
        """Test that all module capabilities include cost and latency info."""
        # Register modules
        await register_module_agents()

        # Get all registered agents
        agents = await agent_manager.list_all_agents()

        # Filter for PEVG modules
        pevg_agents = [
            a for a in agents if a["agent_id"].startswith("agentcore.modular.")
        ]

        assert len(pevg_agents) == 4

        # Verify each agent from discovery
        for agent_summary in pevg_agents:
            agent_id = agent_summary["agent_id"]
            agent = await agent_manager.get_agent(agent_id)
            assert agent is not None

            # Verify each capability has cost and latency
            for capability in agent.capabilities:
                assert (
                    capability.cost_per_request is not None
                ), f"{agent_id} capability {capability.name} missing cost_per_request"
                assert (
                    capability.cost_per_request >= 0.0
                ), f"{agent_id} capability {capability.name} has negative cost"

                assert (
                    capability.avg_latency_ms is not None
                ), f"{agent_id} capability {capability.name} missing avg_latency_ms"
                assert (
                    capability.avg_latency_ms > 0.0
                ), f"{agent_id} capability {capability.name} has non-positive latency"

                assert (
                    capability.quality_score is not None
                ), f"{agent_id} capability {capability.name} missing quality_score"
                assert (
                    0.0 <= capability.quality_score <= 1.0
                ), f"{agent_id} capability {capability.name} has invalid quality_score"

        # Clean up
        await unregister_module_agents()

    async def test_module_metadata_and_tags(self) -> None:
        """Test that all modules have proper metadata and tags."""
        # Register modules
        await register_module_agents()

        module_ids = [
            "agentcore.modular.planner",
            "agentcore.modular.executor",
            "agentcore.modular.verifier",
            "agentcore.modular.generator",
        ]

        for agent_id in module_ids:
            agent = await agent_manager.get_agent(agent_id)
            assert agent is not None

            # Verify metadata exists
            assert agent.metadata is not None
            assert agent.metadata.category == "modular-agent"

            # Verify tags include 'pevg'
            assert "pevg" in agent.metadata.tags

            # Verify documentation URL
            assert agent.metadata.documentation_url is not None

        # Clean up
        await unregister_module_agents()

    async def test_module_discovery_all_list(self) -> None:
        """Test listing all PEVG modules via discovery."""
        # Register modules
        await register_module_agents()

        # List all agents
        all_agents = await agent_manager.list_all_agents()

        # Filter for PEVG modules
        pevg_agents = [
            a for a in all_agents if a["agent_id"].startswith("agentcore.modular.")
        ]

        # Verify all 4 modules are present
        assert len(pevg_agents) == 4

        agent_ids = {a["agent_id"] for a in pevg_agents}
        assert "agentcore.modular.planner" in agent_ids
        assert "agentcore.modular.executor" in agent_ids
        assert "agentcore.modular.verifier" in agent_ids
        assert "agentcore.modular.generator" in agent_ids

        # Clean up
        await unregister_module_agents()
