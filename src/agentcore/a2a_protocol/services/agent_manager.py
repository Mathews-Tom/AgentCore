"""
Agent Management Service

Handles agent registration, discovery, and lifecycle management.
Provides in-memory storage with future support for persistence backends.
"""

import re
from datetime import UTC, datetime, timedelta

import structlog

from agentcore.a2a_protocol.models.agent import (
    AgentCard,
    AgentDiscoveryQuery,
    AgentDiscoveryResponse,
    AgentRegistrationRequest,
    AgentRegistrationResponse,
    AgentStatus,
)
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest

logger = structlog.get_logger()


class AgentManager:
    """
    Agent lifecycle and discovery manager.

    Handles agent registration, discovery, health monitoring, and CRUD operations.
    Currently uses in-memory storage but designed to support database backends.
    """

    def __init__(self) -> None:
        """Initialize the agent manager."""
        # In-memory storage (TODO: Replace with Redis/PostgreSQL)
        self._agents: dict[str, AgentCard] = {}
        self._agent_index: dict[str, set[str]] = {
            "capabilities": {},
            "tags": {},
            "categories": {},
            "status": {},
        }

        logger.info("Agent manager initialized")

    async def register_agent(
        self, request: AgentRegistrationRequest
    ) -> AgentRegistrationResponse:
        """
        Register a new agent or update existing registration.

        Args:
            request: Agent registration request

        Returns:
            Registration response with agent ID and status

        Raises:
            ValueError: If registration validation fails
        """
        agent_card = request.agent_card
        agent_id = agent_card.agent_id

        logger.info(
            "Registering agent",
            agent_id=agent_id,
            agent_name=agent_card.agent_name,
            override=request.override_existing,
        )

        # Check if agent already exists
        existing_agent = self._agents.get(agent_id)
        if existing_agent and not request.override_existing:
            raise ValueError(
                f"Agent {agent_id} already registered. Use override_existing=true to update."
            )

        # Validate agent card
        await self._validate_agent_card(agent_card)

        # Update timestamps
        if existing_agent:
            agent_card.created_at = existing_agent.created_at
        agent_card.updated_at = datetime.now(UTC)
        agent_card.last_seen = datetime.now(UTC)

        # Store agent
        self._agents[agent_id] = agent_card

        # Update indexes
        self._update_agent_indexes(agent_card)

        # Create discovery URL
        discovery_url = f"/.well-known/agents/{agent_id}"

        logger.info(
            "Agent registered successfully",
            agent_id=agent_id,
            agent_name=agent_card.agent_name,
            capabilities=len(agent_card.capabilities),
            endpoints=len(agent_card.endpoints),
        )

        return AgentRegistrationResponse(
            agent_id=agent_id,
            status="registered",
            discovery_url=discovery_url,
            message="Agent registered successfully",
        )

    async def get_agent(self, agent_id: str) -> AgentCard | None:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent card or None if not found
        """
        agent = self._agents.get(agent_id)
        if agent:
            logger.debug("Agent retrieved", agent_id=agent_id)
        return agent

    async def get_agent_summary(self, agent_id: str) -> dict | None:
        """
        Get agent discovery summary by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent summary or None if not found
        """
        agent = await self.get_agent(agent_id)
        if agent:
            return agent.to_discovery_summary()
        return None

    async def discover_agents(
        self, query: AgentDiscoveryQuery
    ) -> AgentDiscoveryResponse:
        """
        Discover agents based on query criteria.

        Args:
            query: Discovery query parameters

        Returns:
            Discovery response with matching agents
        """
        logger.info(
            "Discovering agents",
            capabilities=query.capabilities,
            status=query.status,
            tags=query.tags,
            limit=query.limit,
            offset=query.offset,
        )

        # Get all agents
        all_agents = list(self._agents.values())

        # Apply filters
        filtered_agents = self._filter_agents(all_agents, query)

        # Calculate pagination
        total_count = len(filtered_agents)
        start_idx = query.offset
        end_idx = start_idx + query.limit
        paginated_agents = filtered_agents[start_idx:end_idx]

        # Convert to discovery summaries
        agent_summaries = [agent.to_discovery_summary() for agent in paginated_agents]

        has_more = end_idx < total_count

        logger.info(
            "Agent discovery completed",
            total_found=total_count,
            returned=len(agent_summaries),
            has_more=has_more,
        )

        return AgentDiscoveryResponse(
            agents=agent_summaries,
            total_count=total_count,
            has_more=has_more,
            query=query,
        )

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent was removed, False if not found
        """
        agent = self._agents.pop(agent_id, None)
        if agent:
            self._remove_from_indexes(agent)
            logger.info(
                "Agent unregistered", agent_id=agent_id, agent_name=agent.agent_name
            )
            return True

        logger.warning("Attempted to unregister non-existent agent", agent_id=agent_id)
        return False

    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """
        Update agent status.

        Args:
            agent_id: Agent identifier
            status: New status

        Returns:
            True if updated successfully, False if agent not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        old_status = agent.status
        agent.status = status
        agent.updated_at = datetime.now(UTC)

        # Update indexes if status changed
        if old_status != status:
            self._update_agent_indexes(agent)

        logger.info(
            "Agent status updated",
            agent_id=agent_id,
            old_status=old_status.value,
            new_status=status.value,
        )
        return True

    async def ping_agent(self, agent_id: str) -> bool:
        """
        Update agent last seen timestamp (heartbeat).

        Args:
            agent_id: Agent identifier

        Returns:
            True if updated successfully, False if agent not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.update_last_seen()
        logger.debug("Agent pinged", agent_id=agent_id)
        return True

    async def list_all_agents(self) -> list[dict]:
        """
        List all registered agents (summaries).

        Returns:
            List of agent discovery summaries
        """
        return [agent.to_discovery_summary() for agent in self._agents.values()]

    async def get_agent_count(self) -> int:
        """Get total number of registered agents."""
        return len(self._agents)

    async def get_capabilities_index(self) -> dict[str, int]:
        """Get capabilities index with agent counts."""
        capabilities_count = {}
        for agent in self._agents.values():
            for capability in agent.capabilities:
                capabilities_count[capability.name] = (
                    capabilities_count.get(capability.name, 0) + 1
                )
        return capabilities_count

    async def cleanup_inactive_agents(self, max_inactive_hours: int = 24) -> int:
        """
        Remove agents that haven't been seen for the specified duration.

        Args:
            max_inactive_hours: Maximum hours of inactivity before removal

        Returns:
            Number of agents removed
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_inactive_hours)
        agents_to_remove = []

        for agent_id, agent in self._agents.items():
            if agent.last_seen and agent.last_seen < cutoff_time:
                agents_to_remove.append(agent_id)

        removed_count = 0
        for agent_id in agents_to_remove:
            if await self.unregister_agent(agent_id):
                removed_count += 1

        if removed_count > 0:
            logger.info(
                "Inactive agents cleaned up",
                removed_count=removed_count,
                max_inactive_hours=max_inactive_hours,
            )

        return removed_count

    async def _validate_agent_card(self, agent_card: AgentCard) -> None:
        """
        Validate agent card for registration.

        Args:
            agent_card: Agent card to validate

        Raises:
            ValueError: If validation fails
        """
        # Basic validation is handled by Pydantic models
        # Additional business logic validation can be added here

        # Validate capability names are unique
        capability_names = [cap.name for cap in agent_card.capabilities]
        if len(capability_names) != len(set(capability_names)):
            raise ValueError("Agent capabilities must have unique names")

        # BCR-017: Validate reasoning capabilities
        if agent_card.reasoning_config:
            self._validate_reasoning_config(agent_card.reasoning_config)

        # Validate endpoint URLs are accessible (future implementation)
        # This could include actual HTTP checks to endpoints

        logger.debug("Agent card validation passed", agent_id=agent_card.agent_id)

    def _validate_reasoning_config(self, reasoning_config: dict) -> None:
        """
        Validate reasoning configuration parameters.

        Args:
            reasoning_config: Reasoning configuration dictionary

        Raises:
            ValueError: If reasoning config is invalid
        """
        if not isinstance(reasoning_config, dict):
            raise ValueError("reasoning_config must be a dictionary")

        # Validate max_iterations if present
        if "max_iterations" in reasoning_config:
            max_iterations = reasoning_config["max_iterations"]
            if not isinstance(max_iterations, int):
                raise ValueError("reasoning_config.max_iterations must be an integer")
            if max_iterations < 1 or max_iterations > 50:
                raise ValueError(
                    "reasoning_config.max_iterations must be between 1 and 50"
                )

        # Validate chunk_size if present
        if "chunk_size" in reasoning_config:
            chunk_size = reasoning_config["chunk_size"]
            if not isinstance(chunk_size, int):
                raise ValueError("reasoning_config.chunk_size must be an integer")
            if chunk_size < 1024 or chunk_size > 32768:
                raise ValueError(
                    "reasoning_config.chunk_size must be between 1024 and 32768"
                )

        # Validate carryover_size if present
        if "carryover_size" in reasoning_config:
            carryover_size = reasoning_config["carryover_size"]
            if not isinstance(carryover_size, int):
                raise ValueError("reasoning_config.carryover_size must be an integer")
            if carryover_size < 512 or carryover_size > 16384:
                raise ValueError(
                    "reasoning_config.carryover_size must be between 512 and 16384"
                )

        # Validate temperature if present
        if "temperature" in reasoning_config:
            temperature = reasoning_config["temperature"]
            if not isinstance(temperature, (int, float)):
                raise ValueError("reasoning_config.temperature must be a number")
            if temperature < 0.0 or temperature > 2.0:
                raise ValueError(
                    "reasoning_config.temperature must be between 0.0 and 2.0"
                )

        # Validate carryover_size < chunk_size if both present
        if "chunk_size" in reasoning_config and "carryover_size" in reasoning_config:
            if reasoning_config["carryover_size"] >= reasoning_config["chunk_size"]:
                raise ValueError(
                    "reasoning_config.carryover_size must be less than chunk_size"
                )

        logger.debug(
            "Reasoning config validation passed",
            max_iterations=reasoning_config.get("max_iterations"),
            chunk_size=reasoning_config.get("chunk_size"),
        )

    def _filter_agents(
        self, agents: list[AgentCard], query: AgentDiscoveryQuery
    ) -> list[AgentCard]:
        """
        Filter agents based on discovery query.

        Args:
            agents: List of all agents
            query: Discovery query criteria

        Returns:
            Filtered list of agents
        """
        filtered = agents

        # Filter by status
        if query.status:
            filtered = [a for a in filtered if a.status == query.status]

        # Filter by capabilities
        if query.capabilities:
            filtered = [
                a
                for a in filtered
                if all(a.has_capability(cap) for cap in query.capabilities)
            ]

        # Filter by tags
        if query.tags and query.tags:
            filtered = [
                a
                for a in filtered
                if a.metadata and all(tag in a.metadata.tags for tag in query.tags)
            ]

        # Filter by category
        if query.category:
            filtered = [
                a
                for a in filtered
                if a.metadata and a.metadata.category == query.category
            ]

        # Filter by name pattern
        if query.name_pattern:
            try:
                pattern = re.compile(query.name_pattern, re.IGNORECASE)
                filtered = [a for a in filtered if pattern.search(a.agent_name)]
            except re.error:
                logger.warning("Invalid regex pattern", pattern=query.name_pattern)

        # BCR-018: Filter by bounded reasoning support
        if query.has_bounded_reasoning is not None:
            filtered = [
                a
                for a in filtered
                if a.supports_bounded_reasoning == query.has_bounded_reasoning
            ]

        return filtered

    def _update_agent_indexes(self, agent: AgentCard) -> None:
        """Update search indexes for the agent."""
        agent_id = agent.agent_id

        # Update capabilities index
        for capability in agent.capabilities:
            if capability.name not in self._agent_index["capabilities"]:
                self._agent_index["capabilities"][capability.name] = set()
            self._agent_index["capabilities"][capability.name].add(agent_id)

        # Update tags index
        if agent.metadata and agent.metadata.tags:
            for tag in agent.metadata.tags:
                if tag not in self._agent_index["tags"]:
                    self._agent_index["tags"][tag] = set()
                self._agent_index["tags"][tag].add(agent_id)

        # Update category index
        if agent.metadata and agent.metadata.category:
            category = agent.metadata.category
            if category not in self._agent_index["categories"]:
                self._agent_index["categories"][category] = set()
            self._agent_index["categories"][category].add(agent_id)

        # Update status index
        status = agent.status.value
        if status not in self._agent_index["status"]:
            self._agent_index["status"][status] = set()
        self._agent_index["status"][status].add(agent_id)

    def _remove_from_indexes(self, agent: AgentCard) -> None:
        """Remove agent from all search indexes."""
        agent_id = agent.agent_id

        # Remove from all index sets
        for index_type in self._agent_index.values():
            for agent_set in index_type.values():
                agent_set.discard(agent_id)


# Global agent manager instance
agent_manager = AgentManager()
