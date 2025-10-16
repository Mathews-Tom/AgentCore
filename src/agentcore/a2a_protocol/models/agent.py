"""
Agent Card Models

A2A Protocol v0.2 compliant agent registration and discovery models.
Implements AgentCard specification with capabilities, endpoints, and authentication.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator


class AgentStatus(str, Enum):
    """Agent operational status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class EndpointType(str, Enum):
    """Supported endpoint types."""

    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    WEBSOCKET_SECURE = "wss"


class AuthenticationType(str, Enum):
    """Supported authentication methods."""

    NONE = "none"
    BEARER_TOKEN = "bearer_token"
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    MTLS = "mtls"


class AgentEndpoint(BaseModel):
    """Agent communication endpoint."""

    url: HttpUrl = Field(..., description="Endpoint URL")
    type: EndpointType = Field(..., description="Endpoint type")
    protocols: list[str] = Field(
        default=["jsonrpc-2.0"], description="Supported protocols"
    )
    description: str | None = Field(None, description="Endpoint description")
    health_check_path: str | None = Field(
        default="/health", description="Health check path"
    )

    @field_validator("protocols")
    @classmethod
    def validate_protocols(cls, v: list[str]) -> list[str]:
        """Validate supported protocols."""
        valid_protocols = ["jsonrpc-2.0", "rest", "graphql", "grpc"]
        for protocol in v:
            if protocol not in valid_protocols:
                raise ValueError(f"Unsupported protocol: {protocol}")
        return v


class AgentAuthentication(BaseModel):
    """Agent authentication configuration."""

    type: AuthenticationType = Field(..., description="Authentication type")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Authentication configuration"
    )
    required: bool = Field(
        default=True, description="Whether authentication is required"
    )

    @field_validator("config")
    @classmethod
    def validate_auth_config(
        cls, v: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        """Validate authentication configuration based on type."""
        # In Pydantic v2, info.data contains the validated data so far
        auth_type = info.data.get("type") if info.data else None

        if auth_type == AuthenticationType.BEARER_TOKEN:
            if "token_header" not in v:
                v["token_header"] = "Authorization"
        elif auth_type == AuthenticationType.JWT:
            required_fields = ["algorithm", "public_key_url"]
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"JWT authentication requires '{field}' in config")
        elif auth_type == AuthenticationType.API_KEY:
            if "header_name" not in v:
                v["header_name"] = "X-API-Key"

        return v


class AgentCapability(BaseModel):
    """Agent capability definition."""

    name: str = Field(..., description="Capability name")
    version: str = Field(default="1.0.0", description="Capability version")
    description: str | None = Field(None, description="Capability description")
    input_schema: dict[str, Any] | None = Field(
        None, description="Input schema (JSON Schema)"
    )
    output_schema: dict[str, Any] | None = Field(
        None, description="Output schema (JSON Schema)"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Capability parameters"
    )

    # A2A-017: Cost-biased agent selection fields
    cost_per_request: float | None = Field(
        None, ge=0.0, description="Cost per request in USD"
    )
    avg_latency_ms: float | None = Field(
        None, ge=0.0, description="Average latency in milliseconds"
    )
    quality_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Quality score (0.0-1.0)"
    )

    @field_validator("name")
    @classmethod
    def validate_capability_name(cls, v: str) -> str:
        """Validate capability name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Capability name must be a non-empty string")
        if not v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                "Capability name must contain only alphanumeric characters, hyphens, underscores, and dots"
            )
        return v


class AgentRequirements(BaseModel):
    """Agent operational requirements."""

    min_memory_mb: int | None = Field(
        None, description="Minimum memory requirement in MB"
    )
    min_cpu_cores: float | None = Field(None, description="Minimum CPU cores")
    required_capabilities: list[str] = Field(
        default_factory=list, description="Required capabilities from other agents"
    )
    supported_languages: list[str] = Field(
        default_factory=list, description="Supported programming languages"
    )
    runtime_environment: str | None = Field(
        None, description="Required runtime environment"
    )


class AgentMetadata(BaseModel):
    """Additional agent metadata."""

    tags: list[str] = Field(
        default_factory=list, description="Agent tags for categorization"
    )
    category: str | None = Field(None, description="Agent category")
    license: str | None = Field(None, description="License information")
    documentation_url: HttpUrl | None = Field(None, description="Documentation URL")
    source_code_url: HttpUrl | None = Field(None, description="Source code URL")
    support_contact: str | None = Field(None, description="Support contact information")


class AgentCard(BaseModel):
    """
    A2A Protocol v0.2 compliant Agent Card.

    Represents an agent's identity, capabilities, and operational configuration
    for discovery and communication within the A2A ecosystem.
    """

    # Core identification
    schema_version: str = Field(default="0.2", description="A2A protocol version")
    agent_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique agent identifier"
    )
    agent_name: str = Field(..., description="Human-readable agent name")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Operational information
    status: AgentStatus = Field(
        default=AgentStatus.ACTIVE, description="Current agent status"
    )
    description: str | None = Field(None, description="Agent description")

    # Communication endpoints
    endpoints: list[AgentEndpoint] = Field(
        ..., min_length=1, description="Agent communication endpoints"
    )

    # Capabilities and requirements
    capabilities: list[AgentCapability] = Field(
        default_factory=list, description="Agent capabilities"
    )
    requirements: AgentRequirements | None = Field(
        None, description="Agent requirements"
    )

    # Security and authentication
    authentication: AgentAuthentication = Field(
        ..., description="Authentication configuration"
    )

    # Metadata
    metadata: AgentMetadata | None = Field(None, description="Additional metadata")

    # A2A-018: Context engineering fields
    system_context: str | None = Field(
        None, description="System context for agent behavior and instructions"
    )
    interaction_examples: list[dict[str, str]] | None = Field(
        None,
        description="Example interactions for few-shot learning (input/output pairs)",
    )

    # BCR-016: Bounded reasoning capabilities
    supports_bounded_reasoning: bool = Field(
        default=False,
        description="Whether agent supports bounded context reasoning",
    )
    reasoning_config: dict[str, Any] | None = Field(
        None,
        description="Optional reasoning configuration (max_iterations, chunk_size, carryover_size, temperature)",
    )

    # Timestamps (managed by the system)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Registration timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    last_seen: datetime | None = Field(None, description="Last activity timestamp")

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Agent name cannot be empty")
        if len(v) > 100:
            raise ValueError("Agent name cannot exceed 100 characters")
        return v.strip()

    @field_validator("endpoints")
    @classmethod
    def validate_endpoints(cls, v: list[AgentEndpoint]) -> list[AgentEndpoint]:
        """Ensure at least one endpoint is provided."""
        if not v:
            raise ValueError("At least one endpoint must be provided")
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Ensure updated_at matches created_at for new agents
        if self.updated_at == self.created_at:
            self.updated_at = self.created_at

    def update_last_seen(self) -> None:
        """Update the last seen timestamp."""
        self.last_seen = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def is_active(self) -> bool:
        """Check if agent is currently active."""
        return self.status == AgentStatus.ACTIVE

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        return any(cap.name == capability_name for cap in self.capabilities)

    def get_capability(self, capability_name: str) -> AgentCapability | None:
        """Get a specific capability by name."""
        for capability in self.capabilities:
            if capability.name == capability_name:
                return capability
        return None

    def get_primary_endpoint(self) -> AgentEndpoint | None:
        """Get the primary communication endpoint."""
        if not self.endpoints:
            return None

        # Prefer HTTPS, then HTTP, then WebSocket
        for endpoint_type in [
            EndpointType.HTTPS,
            EndpointType.HTTP,
            EndpointType.WEBSOCKET_SECURE,
            EndpointType.WEBSOCKET,
        ]:
            for endpoint in self.endpoints:
                if endpoint.type == endpoint_type:
                    return endpoint

        # Return first endpoint if no preferred type found
        return self.endpoints[0]

    def to_discovery_summary(self) -> dict[str, Any]:
        """Create a summary for agent discovery."""
        primary_endpoint = self.get_primary_endpoint()
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "status": self.status.value,
            "description": self.description,
            "capabilities": [cap.name for cap in self.capabilities],
            "primary_endpoint": str(primary_endpoint.url) if primary_endpoint else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "created_at": self.created_at.isoformat(),
            "supports_bounded_reasoning": self.supports_bounded_reasoning,
        }


class AgentRegistrationRequest(BaseModel):
    """Agent registration request payload."""

    agent_card: AgentCard = Field(..., description="Agent card to register")
    override_existing: bool = Field(
        default=False, description="Whether to override existing registration"
    )


class AgentRegistrationResponse(BaseModel):
    """Agent registration response."""

    agent_id: str = Field(..., description="Registered agent ID")
    status: str = Field(..., description="Registration status")
    discovery_url: str = Field(..., description="Agent discovery URL")
    message: str | None = Field(None, description="Additional information")


class AgentDiscoveryQuery(BaseModel):
    """Agent discovery query parameters."""

    capabilities: list[str] | None = Field(None, description="Required capabilities")
    status: AgentStatus | None = Field(None, description="Agent status filter")
    tags: list[str] | None = Field(None, description="Agent tags filter")
    category: str | None = Field(None, description="Agent category filter")
    name_pattern: str | None = Field(None, description="Agent name pattern (regex)")
    limit: int = Field(
        default=50, ge=1, le=1000, description="Maximum number of results"
    )
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")


class AgentDiscoveryResponse(BaseModel):
    """Agent discovery response."""

    agents: list[dict[str, Any]] = Field(..., description="Discovered agents")
    total_count: int = Field(..., description="Total number of matching agents")
    has_more: bool = Field(..., description="Whether more results are available")
    query: AgentDiscoveryQuery = Field(..., description="Original query parameters")
