"""
Session Management Models

A2A Protocol v0.2 compliant session management for long-running workflows,
state persistence, and context preservation across agent interactions.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class SessionState(str, Enum):
    """Session lifecycle states."""
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class SessionPriority(str, Enum):
    """Session priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SessionContext(BaseModel):
    """Session context containing runtime state and metadata."""
    variables: Dict[str, Any] = Field(default_factory=dict, description="Session-scoped variables")
    agent_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-agent state")
    shared_memory: Dict[str, Any] = Field(default_factory=dict, description="Shared memory across agents")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Execution event history")


class SessionSnapshot(BaseModel):
    """
    Session snapshot for persistence and recovery.

    Captures complete session state for long-running workflows.
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique session identifier")

    # Session metadata
    name: str = Field(..., description="Human-readable session name")
    description: Optional[str] = Field(None, description="Session description")
    state: SessionState = Field(default=SessionState.ACTIVE, description="Current session state")
    priority: SessionPriority = Field(default=SessionPriority.NORMAL, description="Session priority")

    # Participants
    owner_agent: str = Field(..., description="Session owner agent ID")
    participant_agents: List[str] = Field(default_factory=list, description="Participating agent IDs")

    # Context and state
    context: SessionContext = Field(default_factory=SessionContext, description="Session context")

    # Associated resources
    task_ids: List[str] = Field(default_factory=list, description="Associated task IDs")
    artifact_ids: List[str] = Field(default_factory=list, description="Associated artifact IDs")

    # Lifecycle tracking
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Session expiration timestamp")
    completed_at: Optional[datetime] = Field(None, description="Session completion timestamp")

    # Session configuration
    timeout_seconds: int = Field(default=3600, ge=60, le=86400, description="Session timeout (60s-24h)")
    max_idle_seconds: int = Field(default=300, ge=30, le=3600, description="Max idle time (30s-1h)")

    # Metadata and tags
    tags: List[str] = Field(default_factory=list, description="Session tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

    # Checkpointing
    checkpoint_interval_seconds: int = Field(default=60, ge=10, description="Checkpoint interval")
    last_checkpoint_at: Optional[datetime] = Field(None, description="Last checkpoint timestamp")
    checkpoint_count: int = Field(default=0, ge=0, description="Number of checkpoints created")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate session name."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Session name cannot be empty")
        if len(v) > 200:
            raise ValueError("Session name cannot exceed 200 characters")
        return v.strip()

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    @property
    def is_terminal(self) -> bool:
        """Check if session is in terminal state."""
        return self.state in [SessionState.COMPLETED, SessionState.FAILED, SessionState.EXPIRED]

    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def time_since_update(self) -> float:
        """Get time since last update in seconds."""
        return (datetime.utcnow() - self.updated_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    @property
    def is_idle(self) -> bool:
        """Check if session has exceeded max idle time."""
        return self.time_since_update > self.max_idle_seconds

    def can_transition_to(self, new_state: SessionState) -> bool:
        """Check if session can transition to a new state."""
        valid_transitions = {
            SessionState.ACTIVE: [SessionState.PAUSED, SessionState.SUSPENDED, SessionState.COMPLETED, SessionState.FAILED],
            SessionState.PAUSED: [SessionState.ACTIVE, SessionState.SUSPENDED, SessionState.FAILED],
            SessionState.SUSPENDED: [SessionState.ACTIVE, SessionState.FAILED, SessionState.EXPIRED],
            SessionState.COMPLETED: [],  # Terminal state
            SessionState.FAILED: [],  # Terminal state
            SessionState.EXPIRED: [],  # Terminal state
        }
        return new_state in valid_transitions.get(self.state, [])

    def pause(self) -> None:
        """Pause active session."""
        if not self.can_transition_to(SessionState.PAUSED):
            raise ValueError(f"Cannot pause session in {self.state} state")
        self.state = SessionState.PAUSED
        self.updated_at = datetime.utcnow()

    def resume(self) -> None:
        """Resume paused or suspended session."""
        if not self.can_transition_to(SessionState.ACTIVE):
            raise ValueError(f"Cannot resume session in {self.state} state")
        self.state = SessionState.ACTIVE
        self.updated_at = datetime.utcnow()

    def suspend(self) -> None:
        """Suspend session for later resumption."""
        if not self.can_transition_to(SessionState.SUSPENDED):
            raise ValueError(f"Cannot suspend session in {self.state} state")
        self.state = SessionState.SUSPENDED
        self.updated_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark session as completed."""
        if not self.can_transition_to(SessionState.COMPLETED):
            raise ValueError(f"Cannot complete session in {self.state} state")
        self.state = SessionState.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at

    def fail(self, reason: Optional[str] = None) -> None:
        """Mark session as failed."""
        if not self.can_transition_to(SessionState.FAILED):
            raise ValueError(f"Cannot fail session in {self.state} state")
        self.state = SessionState.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        if reason:
            self.metadata["failure_reason"] = reason

    def expire(self) -> None:
        """Mark session as expired."""
        if not self.can_transition_to(SessionState.EXPIRED):
            raise ValueError(f"Cannot expire session in {self.state} state")
        self.state = SessionState.EXPIRED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at

    def update_context(self, updates: Dict[str, Any]) -> None:
        """Update session context variables."""
        self.context.variables.update(updates)
        self.updated_at = datetime.utcnow()

    def set_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Set state for a specific agent."""
        self.context.agent_states[agent_id] = state
        self.updated_at = datetime.utcnow()

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get state for a specific agent."""
        return self.context.agent_states.get(agent_id)

    def add_task(self, task_id: str) -> None:
        """Add task to session."""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.updated_at = datetime.utcnow()

    def add_artifact(self, artifact_id: str) -> None:
        """Add artifact to session."""
        if artifact_id not in self.artifact_ids:
            self.artifact_ids.append(artifact_id)
            self.updated_at = datetime.utcnow()

    def add_participant(self, agent_id: str) -> None:
        """Add participant agent to session."""
        if agent_id not in self.participant_agents:
            self.participant_agents.append(agent_id)
            self.updated_at = datetime.utcnow()

    def record_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Record execution event in history."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": event_data
        }
        self.context.execution_history.append(event)
        self.updated_at = datetime.utcnow()

    def create_checkpoint(self) -> None:
        """Create checkpoint marker."""
        self.last_checkpoint_at = datetime.utcnow()
        self.checkpoint_count += 1
        self.updated_at = self.last_checkpoint_at

    def to_summary(self) -> Dict[str, Any]:
        """Create session summary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "state": self.state.value,
            "priority": self.priority.value,
            "owner_agent": self.owner_agent,
            "participant_count": len(self.participant_agents),
            "task_count": len(self.task_ids),
            "artifact_count": len(self.artifact_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "duration": self.duration,
            "is_idle": self.is_idle,
            "is_expired": self.is_expired,
            "checkpoint_count": self.checkpoint_count,
            "tags": self.tags,
        }


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    name: str = Field(..., description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    owner_agent: str = Field(..., description="Owner agent ID")
    priority: SessionPriority = Field(default=SessionPriority.NORMAL, description="Session priority")
    timeout_seconds: int = Field(default=3600, ge=60, le=86400, description="Session timeout")
    max_idle_seconds: int = Field(default=300, ge=30, le=3600, description="Max idle time")
    tags: List[str] = Field(default_factory=list, description="Session tags")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Initial context variables")


class SessionCreateResponse(BaseModel):
    """Response for session creation."""
    session_id: str = Field(..., description="Session ID")
    state: str = Field(..., description="Session state")
    message: str = Field(..., description="Creation status message")


class SessionQuery(BaseModel):
    """Session query parameters."""
    state: Optional[SessionState] = Field(None, description="Filter by state")
    owner_agent: Optional[str] = Field(None, description="Filter by owner agent")
    participant_agent: Optional[str] = Field(None, description="Filter by participant agent")
    priority: Optional[SessionPriority] = Field(None, description="Filter by priority")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    created_after: Optional[datetime] = Field(None, description="Filter by creation time")
    created_before: Optional[datetime] = Field(None, description="Filter by creation time")
    include_expired: bool = Field(default=False, description="Include expired sessions")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset")


class SessionQueryResponse(BaseModel):
    """Response for session queries."""
    sessions: List[Dict[str, Any]] = Field(..., description="Session summaries")
    total_count: int = Field(..., description="Total matching sessions")
    has_more: bool = Field(..., description="More results available")
    query: SessionQuery = Field(..., description="Original query")
