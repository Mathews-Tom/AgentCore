"""Multi-agent coordination and communication service."""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    VOTE = "vote"
    CONSENSUS = "consensus"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentMessage(BaseModel):
    """Message for inter-agent communication."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = Field(description="Sender agent ID")
    recipient_id: str | None = Field(
        default=None,
        description="Recipient agent ID (None for broadcast)",
    )
    message_type: MessageType = Field(description="Type of message")
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="Message priority",
    )
    content: dict[str, Any] = Field(description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)
    reply_to: str | None = Field(default=None, description="ID of message being replied to")
    expires_at: datetime | None = Field(default=None, description="Message expiration")


class VoteOption(BaseModel):
    """Voting option for consensus mechanisms."""

    option_id: str = Field(description="Option identifier")
    description: str = Field(description="Option description")
    votes: list[str] = Field(default_factory=list, description="List of agent IDs that voted")
    weight: float = Field(default=1.0, description="Option weight/importance")


class ConsensusRequest(BaseModel):
    """Request for consensus among agents."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    initiator_id: str = Field(description="Agent initiating consensus")
    topic: str = Field(description="Topic for consensus")
    options: list[VoteOption] = Field(description="Available voting options")
    participating_agents: list[str] = Field(description="Agents participating in vote")
    required_votes: int = Field(description="Minimum votes required")
    timeout_seconds: int = Field(default=60, description="Voting timeout")
    created_at: datetime = Field(default_factory=datetime.now)


class ConsensusResult(BaseModel):
    """Result of consensus voting."""

    request_id: str = Field(description="Corresponding consensus request ID")
    winning_option: VoteOption | None = Field(description="Winning option if consensus reached")
    consensus_reached: bool = Field(description="Whether consensus was reached")
    total_votes: int = Field(description="Total votes cast")
    vote_distribution: dict[str, int] = Field(description="Votes per option")
    timestamp: datetime = Field(default_factory=datetime.now)


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    PRIORITY_BASED = "priority_based"
    ROUND_ROBIN = "round_robin"
    FIRST_COME_FIRST_SERVED = "fcfs"


class SharedState(BaseModel):
    """Shared state among coordinated agents."""

    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: str = Field(description="Agent owning the state")
    data: dict[str, Any] = Field(description="State data")
    version: int = Field(default=1, description="State version")
    last_modified: datetime = Field(default_factory=datetime.now)
    locked_by: str | None = Field(default=None, description="Agent holding lock")
    access_control: dict[str, list[str]] = Field(
        default_factory=lambda: {"read": [], "write": []},
        description="Access control lists",
    )


class MultiAgentCoordinator:
    """Coordinator for multi-agent communication and consensus."""

    def __init__(self) -> None:
        """Initialize multi-agent coordinator."""
        self._message_queues: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._active_consensus: dict[str, ConsensusRequest] = {}
        self._shared_states: dict[str, SharedState] = {}
        self._agent_registry: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def register_agent(self, agent_id: str, metadata: dict[str, Any]) -> None:
        """
        Register an agent for coordination.

        Args:
            agent_id: Agent identifier
            metadata: Agent metadata (capabilities, priority, etc.)
        """
        async with self._lock:
            self._agent_registry[agent_id] = metadata
            self._message_queues[agent_id] = asyncio.Queue()

        logger.info("agent_registered", agent_id=agent_id)

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from coordination.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            self._agent_registry.pop(agent_id, None)
            self._message_queues.pop(agent_id, None)

        logger.info("agent_unregistered", agent_id=agent_id)

    async def send_message(self, message: AgentMessage) -> None:
        """
        Send message to agent(s).

        Args:
            message: Message to send
        """
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self._message_queues:
                await self._message_queues[message.recipient_id].put(message)
                logger.debug(
                    "message_sent",
                    message_id=message.message_id,
                    sender=message.sender_id,
                    recipient=message.recipient_id,
                )
        else:
            # Broadcast message
            for agent_id, queue in self._message_queues.items():
                if agent_id != message.sender_id:  # Don't send to self
                    await queue.put(message)

            logger.debug(
                "message_broadcast",
                message_id=message.message_id,
                sender=message.sender_id,
                recipients=len(self._message_queues) - 1,
            )

    async def receive_message(
        self,
        agent_id: str,
        timeout: float | None = None,
    ) -> AgentMessage | None:
        """
        Receive message for agent.

        Args:
            agent_id: Agent identifier
            timeout: Receive timeout in seconds

        Returns:
            Received message or None if timeout
        """
        if agent_id not in self._message_queues:
            raise ValueError(f"Agent {agent_id} not registered")

        try:
            if timeout:
                message = await asyncio.wait_for(
                    self._message_queues[agent_id].get(),
                    timeout=timeout,
                )
            else:
                message = await self._message_queues[agent_id].get()

            logger.debug("message_received", agent_id=agent_id, message_id=message.message_id)
            return message

        except asyncio.TimeoutError:
            return None

    async def initiate_consensus(self, request: ConsensusRequest) -> str:
        """
        Initiate consensus voting.

        Args:
            request: Consensus request

        Returns:
            Request ID
        """
        async with self._lock:
            self._active_consensus[request.request_id] = request

        # Broadcast voting request to participants
        vote_message = AgentMessage(
            sender_id=request.initiator_id,
            message_type=MessageType.VOTE,
            priority=MessagePriority.HIGH,
            content={
                "request_id": request.request_id,
                "topic": request.topic,
                "options": [opt.model_dump() for opt in request.options],
                "timeout_seconds": request.timeout_seconds,
            },
        )

        for agent_id in request.participating_agents:
            if agent_id != request.initiator_id:
                vote_message.recipient_id = agent_id
                await self.send_message(vote_message)

        logger.info(
            "consensus_initiated",
            request_id=request.request_id,
            topic=request.topic,
            participants=len(request.participating_agents),
        )

        return request.request_id

    async def cast_vote(
        self,
        request_id: str,
        agent_id: str,
        option_id: str,
    ) -> None:
        """
        Cast vote in consensus.

        Args:
            request_id: Consensus request ID
            agent_id: Voting agent ID
            option_id: Selected option ID
        """
        async with self._lock:
            if request_id not in self._active_consensus:
                raise ValueError(f"Consensus request {request_id} not found")

            request = self._active_consensus[request_id]

            # Find and update option
            for option in request.options:
                if option.option_id == option_id:
                    if agent_id not in option.votes:
                        option.votes.append(agent_id)
                        logger.info(
                            "vote_cast",
                            request_id=request_id,
                            agent_id=agent_id,
                            option_id=option_id,
                        )
                    break

    async def check_consensus(self, request_id: str) -> ConsensusResult:
        """
        Check consensus status.

        Args:
            request_id: Consensus request ID

        Returns:
            Consensus result
        """
        async with self._lock:
            if request_id not in self._active_consensus:
                raise ValueError(f"Consensus request {request_id} not found")

            request = self._active_consensus[request_id]

            # Count votes
            total_votes = sum(len(opt.votes) for opt in request.options)
            vote_distribution = {opt.option_id: len(opt.votes) for opt in request.options}

            # Find winning option
            winning_option = max(request.options, key=lambda opt: len(opt.votes))
            consensus_reached = len(winning_option.votes) >= request.required_votes

            result = ConsensusResult(
                request_id=request_id,
                winning_option=winning_option if consensus_reached else None,
                consensus_reached=consensus_reached,
                total_votes=total_votes,
                vote_distribution=vote_distribution,
            )

            if consensus_reached:
                # Clean up completed consensus
                del self._active_consensus[request_id]

                # Broadcast result
                result_message = AgentMessage(
                    sender_id=request.initiator_id,
                    message_type=MessageType.CONSENSUS,
                    priority=MessagePriority.HIGH,
                    content={"result": result.model_dump()},
                )
                await self.send_message(result_message)

                logger.info(
                    "consensus_reached",
                    request_id=request_id,
                    winning_option=winning_option.option_id,
                    votes=len(winning_option.votes),
                )

            return result

    async def resolve_conflict(
        self,
        conflict_data: dict[str, Any],
        strategy: ConflictResolutionStrategy,
        involved_agents: list[str],
    ) -> dict[str, Any]:
        """
        Resolve conflict between agents.

        Args:
            conflict_data: Conflict information
            strategy: Resolution strategy
            involved_agents: Agents involved in conflict

        Returns:
            Resolution result
        """
        logger.info(
            "conflict_resolution_start",
            strategy=strategy.value,
            agents=len(involved_agents),
        )

        if strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            # Create consensus request
            options = [
                VoteOption(
                    option_id=f"option_{i}",
                    description=str(option),
                )
                for i, option in enumerate(conflict_data.get("options", []))
            ]

            request = ConsensusRequest(
                initiator_id="coordinator",
                topic=conflict_data.get("topic", "Conflict resolution"),
                options=options,
                participating_agents=involved_agents,
                required_votes=(len(involved_agents) // 2) + 1,
            )

            request_id = await self.initiate_consensus(request)

            # Wait for votes (with timeout)
            await asyncio.sleep(request.timeout_seconds)

            result = await self.check_consensus(request_id)

            return {
                "strategy": strategy.value,
                "resolution": result.winning_option.option_id if result.winning_option else None,
                "consensus_reached": result.consensus_reached,
            }

        elif strategy == ConflictResolutionStrategy.PRIORITY_BASED:
            # Select based on agent priority
            priorities = {
                agent_id: self._agent_registry.get(agent_id, {}).get("priority", 0)
                for agent_id in involved_agents
            }
            highest_priority_agent = max(priorities, key=priorities.get)

            return {
                "strategy": strategy.value,
                "selected_agent": highest_priority_agent,
                "priority": priorities[highest_priority_agent],
            }

        else:
            raise ValueError(f"Unsupported conflict resolution strategy: {strategy}")

    async def create_shared_state(self, state: SharedState) -> str:
        """
        Create shared state.

        Args:
            state: Shared state data

        Returns:
            State ID
        """
        async with self._lock:
            self._shared_states[state.state_id] = state

        logger.info("shared_state_created", state_id=state.state_id, owner=state.owner_id)
        return state.state_id

    async def read_shared_state(self, state_id: str, agent_id: str) -> SharedState:
        """
        Read shared state.

        Args:
            state_id: State identifier
            agent_id: Requesting agent ID

        Returns:
            Shared state
        """
        async with self._lock:
            if state_id not in self._shared_states:
                raise ValueError(f"Shared state {state_id} not found")

            state = self._shared_states[state_id]

            # Check read access
            if agent_id not in state.access_control["read"] and state.owner_id != agent_id:
                raise PermissionError(f"Agent {agent_id} does not have read access")

            return state

    async def update_shared_state(
        self,
        state_id: str,
        agent_id: str,
        updates: dict[str, Any],
    ) -> SharedState:
        """
        Update shared state.

        Args:
            state_id: State identifier
            agent_id: Updating agent ID
            updates: State updates

        Returns:
            Updated shared state
        """
        async with self._lock:
            if state_id not in self._shared_states:
                raise ValueError(f"Shared state {state_id} not found")

            state = self._shared_states[state_id]

            # Check write access
            if agent_id not in state.access_control["write"] and state.owner_id != agent_id:
                raise PermissionError(f"Agent {agent_id} does not have write access")

            # Check lock
            if state.locked_by and state.locked_by != agent_id:
                raise RuntimeError(f"State locked by {state.locked_by}")

            # Apply updates
            state.data.update(updates)
            state.version += 1
            state.last_modified = datetime.now()

            logger.info(
                "shared_state_updated",
                state_id=state_id,
                agent_id=agent_id,
                version=state.version,
            )

            return state

    async def lock_shared_state(self, state_id: str, agent_id: str) -> bool:
        """
        Lock shared state for exclusive access.

        Args:
            state_id: State identifier
            agent_id: Locking agent ID

        Returns:
            True if lock acquired
        """
        async with self._lock:
            if state_id not in self._shared_states:
                raise ValueError(f"Shared state {state_id} not found")

            state = self._shared_states[state_id]

            if state.locked_by and state.locked_by != agent_id:
                return False

            state.locked_by = agent_id
            logger.info("shared_state_locked", state_id=state_id, agent_id=agent_id)
            return True

    async def unlock_shared_state(self, state_id: str, agent_id: str) -> None:
        """
        Unlock shared state.

        Args:
            state_id: State identifier
            agent_id: Unlocking agent ID
        """
        async with self._lock:
            if state_id not in self._shared_states:
                raise ValueError(f"Shared state {state_id} not found")

            state = self._shared_states[state_id]

            if state.locked_by != agent_id:
                raise PermissionError(f"Agent {agent_id} does not hold lock")

            state.locked_by = None
            logger.info("shared_state_unlocked", state_id=state_id, agent_id=agent_id)

    def get_active_agents(self) -> list[str]:
        """
        Get list of active agent IDs.

        Returns:
            List of agent IDs
        """
        return list(self._agent_registry.keys())

    def get_agent_metadata(self, agent_id: str) -> dict[str, Any]:
        """
        Get agent metadata.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent metadata
        """
        return self._agent_registry.get(agent_id, {})


# Global coordinator instance
_coordinator: MultiAgentCoordinator | None = None


def get_coordinator() -> MultiAgentCoordinator:
    """
    Get global coordinator instance.

    Returns:
        Global coordinator
    """
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiAgentCoordinator()
    return _coordinator
