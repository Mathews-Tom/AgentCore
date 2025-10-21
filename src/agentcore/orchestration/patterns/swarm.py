"""
Swarm Pattern Implementation

Distributed coordination algorithms for emergent behavior with consensus and voting mechanisms.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.streams.models import OrchestrationEvent
from agentcore.orchestration.streams.producer import StreamProducer


class ConsensusStrategy(str, Enum):
    """Strategy for reaching consensus among swarm agents."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous"
    QUORUM_BASED = "quorum_based"
    BEST_PROPOSAL = "best_proposal"


class AgentRole(str, Enum):
    """Role of an agent within the swarm."""

    MEMBER = "member"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"


class ProposalStatus(str, Enum):
    """Status of a proposal in the swarm."""

    PENDING = "pending"
    VOTING = "voting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXECUTED = "executed"


class AgentState(BaseModel):
    """State of an agent in the swarm."""

    agent_id: str = Field(description="Unique agent identifier")
    role: AgentRole = Field(default=AgentRole.MEMBER)
    capabilities: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, description="Voting weight (0.0-1.0)")
    active: bool = Field(default=True)
    status: str = Field(default="active")  # Compatibility field
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    contribution_score: float = Field(
        default=0.0, description="Historical contribution score"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class Vote(BaseModel):
    """Individual vote on a proposal."""

    vote_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(description="Agent casting the vote")
    proposal_id: UUID = Field(description="Proposal being voted on")
    approved: bool = Field(description="Vote approval (True) or rejection (False)")
    weight: float = Field(default=1.0, description="Vote weight")
    reasoning: str | None = Field(default=None, description="Optional vote reasoning")
    cast_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentProposal(BaseModel):
    """Proposal submitted by a swarm agent."""

    proposal_id: UUID = Field(default_factory=uuid4)
    proposer_id: str = Field(description="Agent proposing the action")
    proposal_type: str = Field(description="Type of proposal")
    proposal_data: dict[str, Any] = Field(
        description="Proposal data and parameters"
    )
    status: ProposalStatus = Field(default=ProposalStatus.PENDING)

    required_votes: int | None = Field(
        default=None, description="Required number of votes (None = auto-calculate)"
    )
    votes: list[Vote] = Field(default_factory=list)

    quality_score: float = Field(
        default=0.0, description="Quality/fitness score of proposal"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    voting_started_at: datetime | None = None
    resolved_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class SwarmTask(BaseModel):
    """Task being coordinated by the swarm."""

    task_id: UUID = Field(default_factory=uuid4)
    task_type: str = Field(description="Type of task")
    task_data: dict[str, Any] = Field(default_factory=dict, description="Task input data")
    input_data: dict[str, Any] | None = Field(default=None, description="Alias for task_data")

    assigned_agents: list[str] = Field(default_factory=list)
    status: str = Field(default="pending")

    proposals: list[UUID] = Field(
        default_factory=list, description="Proposals related to this task"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    result_data: dict[str, Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        """Handle input_data alias after initialization."""
        if self.input_data is not None and not self.task_data:
            self.task_data = self.input_data


class SwarmConfig(BaseModel):
    """Configuration for swarm pattern."""

    consensus_strategy: ConsensusStrategy = Field(
        default=ConsensusStrategy.MAJORITY_VOTE
    )
    quorum_percentage: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Required quorum (0.0-1.0)"
    )
    voting_timeout_seconds: int = Field(default=60, description="Voting timeout")
    enable_weighted_voting: bool = Field(
        default=False, description="Enable weighted voting by contribution score"
    )
    max_swarm_size: int = Field(
        default=100, ge=1, description="Maximum number of agents"
    )
    heartbeat_interval_seconds: int = Field(
        default=10, description="Agent heartbeat interval"
    )
    agent_timeout_seconds: int = Field(
        default=30, description="Agent timeout threshold"
    )
    enable_emergent_behavior: bool = Field(
        default=True, description="Enable emergent behavior patterns"
    )


class SwarmCoordinator:
    """
    Swarm pattern coordinator.

    Implements distributed coordination with:
    - Emergent behavior management
    - Consensus and voting mechanisms
    - Distributed decision-making
    - Performance optimization for large swarms
    """

    def __init__(
        self,
        swarm_id: str,
        config: SwarmConfig | None = None,
        event_producer: StreamProducer | None = None,
    ) -> None:
        """
        Initialize swarm coordinator.

        Args:
            swarm_id: Unique swarm identifier
            config: Swarm configuration
            event_producer: Event stream producer
        """
        self.swarm_id = swarm_id
        self.config = config or SwarmConfig()
        self.event_producer = event_producer

        # Swarm members
        self._agents: dict[str, AgentState] = {}

        # Proposal tracking
        self._proposals: dict[UUID, AgentProposal] = {}
        self._active_proposals: set[UUID] = set()

        # Task tracking
        self._tasks: dict[UUID, SwarmTask] = {}

        # Voting cache for performance
        self._vote_cache: dict[UUID, dict[str, Vote]] = defaultdict(dict)

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @property
    def agents(self) -> dict[str, AgentState]:
        """Get all agents in the swarm."""
        return self._agents

    def register_agent(self, agent: AgentState) -> None:
        """
        Synchronous method to register an agent (compatibility method).

        Args:
            agent: Agent state to register
        """
        if len(self._agents) >= self.config.max_swarm_size:
            raise ValueError(
                f"Swarm size limit reached ({self.config.max_swarm_size})"
            )
        self._agents[agent.agent_id] = agent

    async def join_swarm(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        role: AgentRole = AgentRole.MEMBER,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Agent joins the swarm.

        Args:
            agent_id: Agent identifier
            capabilities: Agent capabilities
            role: Agent role in swarm
            weight: Voting weight
            metadata: Additional agent metadata
        """
        async with self._lock:
            if len(self._agents) >= self.config.max_swarm_size:
                raise ValueError(
                    f"Swarm size limit reached ({self.config.max_swarm_size})"
                )

            self._agents[agent_id] = AgentState(
                agent_id=agent_id,
                role=role,
                capabilities=capabilities or [],
                weight=weight,
                active=True,
                metadata=metadata or {},
            )

        # Publish join event
        if self.event_producer:
            await self._publish_swarm_event(
                "agent_joined",
                {"agent_id": agent_id, "role": role, "swarm_size": len(self._agents)},
            )

    async def leave_swarm(self, agent_id: str) -> None:
        """
        Agent leaves the swarm.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].active = False
                del self._agents[agent_id]

        # Publish leave event
        if self.event_producer:
            await self._publish_swarm_event(
                "agent_left",
                {"agent_id": agent_id, "swarm_size": len(self._agents)},
            )

    async def submit_proposal(
        self,
        proposer_id: str,
        proposal_type: str,
        proposal_data: dict[str, Any],
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """
        Submit a proposal for swarm voting.

        Args:
            proposer_id: Agent proposing
            proposal_type: Type of proposal
            proposal_data: Proposal details
            task_id: Associated task ID
            metadata: Additional metadata

        Returns:
            Proposal ID
        """
        async with self._lock:
            if proposer_id not in self._agents:
                raise ValueError(f"Agent not in swarm: {proposer_id}")

            # Create proposal
            proposal = AgentProposal(
                proposer_id=proposer_id,
                proposal_type=proposal_type,
                proposal_data=proposal_data,
                status=ProposalStatus.PENDING,
                metadata=metadata or {},
            )

            # Calculate required votes based on consensus strategy
            proposal.required_votes = await self._calculate_required_votes()

            # Store proposal
            self._proposals[proposal.proposal_id] = proposal
            self._active_proposals.add(proposal.proposal_id)

            # Link to task if provided
            if task_id and task_id in self._tasks:
                self._tasks[task_id].proposals.append(proposal.proposal_id)

        # Publish proposal event
        if self.event_producer:
            await self._publish_swarm_event(
                "proposal_submitted",
                {
                    "proposal_id": str(proposal.proposal_id),
                    "proposer_id": proposer_id,
                    "proposal_type": proposal_type,
                    "required_votes": proposal.required_votes,
                },
            )

        # Start voting
        await self._start_voting(proposal.proposal_id)

        return proposal.proposal_id

    async def cast_vote(
        self,
        agent_id: str,
        proposal_id: UUID,
        approved: bool,
        reasoning: str | None = None,
    ) -> None:
        """
        Cast a vote on a proposal.

        Args:
            agent_id: Agent casting vote
            proposal_id: Proposal to vote on
            approved: Vote approval
            reasoning: Optional reasoning
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent not in swarm: {agent_id}")

            if proposal_id not in self._proposals:
                raise ValueError(f"Proposal not found: {proposal_id}")

            proposal = self._proposals[proposal_id]

            if proposal.status != ProposalStatus.VOTING:
                raise ValueError(
                    f"Proposal not in voting state: {proposal.status}"
                )

            # Check if already voted
            if agent_id in self._vote_cache[proposal_id]:
                raise ValueError(f"Agent already voted: {agent_id}")

            # Get agent weight
            agent = self._agents[agent_id]
            vote_weight = (
                agent.weight * agent.contribution_score
                if self.config.enable_weighted_voting
                else agent.weight
            )

            # Create vote
            vote = Vote(
                agent_id=agent_id,
                proposal_id=proposal_id,
                approved=approved,
                weight=vote_weight,
                reasoning=reasoning,
            )

            # Store vote
            proposal.votes.append(vote)
            self._vote_cache[proposal_id][agent_id] = vote

        # Publish vote event
        if self.event_producer:
            await self._publish_swarm_event(
                "vote_cast",
                {
                    "proposal_id": str(proposal_id),
                    "agent_id": agent_id,
                    "approved": approved,
                    "votes_received": len(proposal.votes),
                },
            )

        # Check if consensus reached (non-blocking)
        await self._check_consensus(proposal_id)

    async def create_task(
        self,
        task_type: str,
        task_data: dict[str, Any],
    ) -> UUID:
        """
        Create a task for swarm coordination.

        Args:
            task_type: Type of task
            task_data: Task input data

        Returns:
            Task ID
        """
        async with self._lock:
            task = SwarmTask(
                task_type=task_type,
                task_data=task_data,
            )

            self._tasks[task.task_id] = task

        # Publish task created event
        if self.event_producer:
            await self._publish_swarm_event(
                "task_created",
                {"task_id": str(task.task_id), "task_type": task_type},
            )

        return task.task_id

    async def execute_proposal(self, proposal_id: UUID) -> dict[str, Any]:
        """
        Execute an accepted proposal.

        Args:
            proposal_id: Proposal to execute

        Returns:
            Execution result
        """
        async with self._lock:
            if proposal_id not in self._proposals:
                raise ValueError(f"Proposal not found: {proposal_id}")

            proposal = self._proposals[proposal_id]

            if proposal.status != ProposalStatus.ACCEPTED:
                raise ValueError(
                    f"Proposal not accepted: {proposal.status}"
                )

        # Execute proposal logic (implementation specific)
        result = {
            "proposal_id": str(proposal_id),
            "executed_at": datetime.now(UTC).isoformat(),
            "result": "success",
        }

        # Update proposal status
        async with self._lock:
            proposal.status = ProposalStatus.EXECUTED
            proposal.resolved_at = datetime.now(UTC)

        # Publish execution event
        if self.event_producer:
            await self._publish_swarm_event(
                "proposal_executed",
                result,
            )

        return result

    async def heartbeat(self, agent_id: str) -> None:
        """
        Record agent heartbeat.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = datetime.now(UTC)

    async def monitor_agents(self) -> None:
        """
        Monitor agent health and handle timeouts.

        Should be called periodically.
        """
        now = datetime.now(UTC)
        timeout_threshold = now.timestamp() - self.config.agent_timeout_seconds

        async with self._lock:
            for agent_id, agent in list(self._agents.items()):
                if agent.last_heartbeat.timestamp() < timeout_threshold:
                    # Agent timeout - mark inactive
                    agent.active = False

                    # Publish timeout event
                    if self.event_producer:
                        await self._publish_swarm_event(
                            "agent_timeout",
                            {
                                "agent_id": agent_id,
                                "last_heartbeat": agent.last_heartbeat.isoformat(),
                            },
                        )

    async def _start_voting(self, proposal_id: UUID) -> None:
        """
        Start voting process for a proposal.

        Args:
            proposal_id: Proposal ID
        """
        async with self._lock:
            if proposal_id not in self._proposals:
                return

            proposal = self._proposals[proposal_id]
            proposal.status = ProposalStatus.VOTING
            proposal.voting_started_at = datetime.now(UTC)

        # Schedule voting timeout
        asyncio.create_task(self._handle_voting_timeout(proposal_id))

    async def _handle_voting_timeout(self, proposal_id: UUID) -> None:
        """
        Handle voting timeout for a proposal.

        Args:
            proposal_id: Proposal ID
        """
        await asyncio.sleep(self.config.voting_timeout_seconds)

        async with self._lock:
            if proposal_id not in self._proposals:
                return

            proposal = self._proposals[proposal_id]

            if proposal.status == ProposalStatus.VOTING:
                # Timeout reached - check if minimum votes met
                await self._check_consensus(proposal_id, force_resolution=True)

    async def _calculate_required_votes(self) -> int:
        """
        Calculate required votes based on consensus strategy.

        Returns:
            Number of required votes
        """
        active_agents = len([a for a in self._agents.values() if a.active])

        if self.config.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE:
            return (active_agents // 2) + 1
        elif self.config.consensus_strategy == ConsensusStrategy.UNANIMOUS:
            return active_agents
        elif self.config.consensus_strategy == ConsensusStrategy.QUORUM_BASED:
            return max(1, int(active_agents * self.config.quorum_percentage))
        else:
            # Default to majority
            return (active_agents // 2) + 1

    async def _check_consensus(
        self, proposal_id: UUID, force_resolution: bool = False
    ) -> None:
        """
        Check if consensus has been reached on a proposal.

        Args:
            proposal_id: Proposal to check
            force_resolution: Force resolution even if voting not complete
        """
        async with self._lock:
            if proposal_id not in self._proposals:
                return

            proposal = self._proposals[proposal_id]

            if proposal.status != ProposalStatus.VOTING:
                return

            # Count votes
            total_votes = len(proposal.votes)
            approved_votes = sum(1 for v in proposal.votes if v.approved)
            rejected_votes = total_votes - approved_votes

            # For weighted voting
            if self.config.enable_weighted_voting:
                total_weight = sum(v.weight for v in proposal.votes)
                approved_weight = sum(v.weight for v in proposal.votes if v.approved)
                approval_ratio = (
                    approved_weight / total_weight if total_weight > 0 else 0.0
                )
            else:
                approval_ratio = (
                    approved_votes / total_votes if total_votes > 0 else 0.0
                )

            # Check consensus
            consensus_reached = False
            accepted = False

            if self.config.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE:
                # Only resolve when all votes received or forced
                if total_votes >= (proposal.required_votes or 0) or force_resolution:
                    if approved_votes > rejected_votes:
                        consensus_reached = True
                        accepted = True
                    elif force_resolution:
                        consensus_reached = True
                        accepted = approved_votes >= rejected_votes

            elif self.config.consensus_strategy == ConsensusStrategy.UNANIMOUS:
                if total_votes >= (proposal.required_votes or 0):
                    consensus_reached = True
                    accepted = approved_votes == total_votes

            elif self.config.consensus_strategy == ConsensusStrategy.WEIGHTED_VOTE:
                if total_votes >= (proposal.required_votes or 0):
                    consensus_reached = True
                    accepted = approval_ratio >= 0.5

            elif self.config.consensus_strategy == ConsensusStrategy.QUORUM_BASED:
                if total_votes >= (proposal.required_votes or 0):
                    consensus_reached = True
                    accepted = approval_ratio >= self.config.quorum_percentage

            # Resolve proposal if consensus reached
            if consensus_reached:
                proposal.status = (
                    ProposalStatus.ACCEPTED if accepted else ProposalStatus.REJECTED
                )
                proposal.resolved_at = datetime.now(UTC)
                self._active_proposals.discard(proposal_id)

                # Publish consensus event
                if self.event_producer:
                    await self._publish_swarm_event(
                        "consensus_reached",
                        {
                            "proposal_id": str(proposal_id),
                            "status": proposal.status,
                            "total_votes": total_votes,
                            "approved_votes": approved_votes,
                            "approval_ratio": approval_ratio,
                        },
                    )

    async def _publish_swarm_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """
        Publish swarm event to stream.

        Args:
            event_type: Event type
            data: Event data
        """
        if not self.event_producer:
            return

        event_data = {
            "event_type": event_type,
            "swarm_id": self.swarm_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }

        await self.event_producer.publish(event_data)

    async def get_swarm_status(self) -> dict[str, Any]:
        """
        Get current swarm status.

        Returns:
            Status dictionary
        """
        async with self._lock:
            active_agents = [a for a in self._agents.values() if a.active]
            return {
                "swarm_id": self.swarm_id,
                "agents": {
                    "total": len(self._agents),
                    "active": len(active_agents),
                    "inactive": len(self._agents) - len(active_agents),
                    "by_role": {
                        role: sum(
                            1 for a in active_agents if a.role == role
                        )
                        for role in AgentRole
                    },
                },
                "proposals": {
                    "total": len(self._proposals),
                    "active": len(self._active_proposals),
                    "pending": sum(
                        1
                        for p in self._proposals.values()
                        if p.status == ProposalStatus.PENDING
                    ),
                    "voting": sum(
                        1
                        for p in self._proposals.values()
                        if p.status == ProposalStatus.VOTING
                    ),
                    "accepted": sum(
                        1
                        for p in self._proposals.values()
                        if p.status == ProposalStatus.ACCEPTED
                    ),
                    "rejected": sum(
                        1
                        for p in self._proposals.values()
                        if p.status == ProposalStatus.REJECTED
                    ),
                },
                "tasks": {
                    "total": len(self._tasks),
                    "pending": sum(
                        1 for t in self._tasks.values() if t.status == "pending"
                    ),
                    "active": sum(
                        1 for t in self._tasks.values() if t.status == "active"
                    ),
                    "completed": sum(
                        1 for t in self._tasks.values() if t.status == "completed"
                    ),
                },
                "config": self.config.model_dump(),
            }

    async def get_agent_states(self) -> list[AgentState]:
        """
        Get states of all swarm agents.

        Returns:
            List of agent states
        """
        async with self._lock:
            return list(self._agents.values())

    async def get_proposal(self, proposal_id: UUID) -> AgentProposal:
        """
        Get proposal by ID.

        Args:
            proposal_id: Proposal identifier

        Returns:
            Proposal

        Raises:
            ValueError: If proposal not found
        """
        async with self._lock:
            if proposal_id not in self._proposals:
                raise ValueError(f"Proposal not found: {proposal_id}")
            return self._proposals[proposal_id]

    async def get_active_proposals(self) -> list[AgentProposal]:
        """
        Get all active proposals.

        Returns:
            List of active proposals
        """
        async with self._lock:
            return [
                self._proposals[pid]
                for pid in self._active_proposals
                if pid in self._proposals
            ]

    def assign_task(self, task: SwarmTask) -> list[str]:
        """
        Assign task to agents (compatibility method).

        Args:
            task: Task to assign

        Returns:
            List of assigned agent IDs
        """
        # Simple assignment - pick active agents with matching capabilities
        active_agents = [
            agent_id
            for agent_id, agent in self._agents.items()
            if agent.active
        ]

        # Assign based on required_agents if specified
        required = getattr(task, "required_agents", len(active_agents))
        assigned = active_agents[:required]
        task.assigned_agents = assigned
        # task_id is already a UUID, no need to convert
        self._tasks[task.task_id] = task
        return assigned

    def submit_vote(
        self,
        agent_id: str,
        proposal_id: str,
        vote: bool,
    ) -> None:
        """
        Submit vote synchronously (compatibility method).

        Args:
            agent_id: Agent ID
            proposal_id: Proposal ID
            vote: Vote value
        """
        proposal_uuid = UUID(proposal_id)
        if proposal_uuid not in self._proposals:
            # Create a pseudo-proposal for the vote
            self._proposals[proposal_uuid] = AgentProposal(
                proposal_id=proposal_uuid,
                proposer_id="system",
                proposal_type="task_vote",
                proposal_data={},
                status=ProposalStatus.VOTING,
            )

        proposal = self._proposals[proposal_uuid]
        vote_obj = Vote(
            agent_id=agent_id,
            proposal_id=proposal_uuid,
            approved=vote,
            weight=1.0,
        )
        proposal.votes.append(vote_obj)
        self._vote_cache[proposal_uuid][agent_id] = vote_obj

    def check_consensus(self, proposal_id: str) -> bool:
        """
        Check if consensus reached (compatibility method).

        Args:
            proposal_id: Proposal ID

        Returns:
            True if consensus reached
        """
        proposal_uuid = UUID(proposal_id)
        if proposal_uuid not in self._proposals:
            return False

        proposal = self._proposals[proposal_uuid]
        total_votes = len(proposal.votes)
        approved_votes = sum(1 for v in proposal.votes if v.approved)

        # Simple majority
        return approved_votes > (total_votes / 2)
