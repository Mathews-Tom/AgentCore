"""
Security Models

Data models for authentication, authorization, and security tokens.
"""

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class TokenType(str, Enum):
    """JWT token type."""

    ACCESS = "access"
    REFRESH = "refresh"


class Permission(str, Enum):
    """Agent permissions."""

    AGENT_REGISTER = "agent:register"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    ROUTE_MESSAGE = "route:message"
    ADMIN = "admin"


class Role(str, Enum):
    """Agent roles."""

    AGENT = "agent"
    SERVICE = "service"
    ADMIN = "admin"


# Role-based permission mapping
ROLE_PERMISSIONS: dict[Role, list[Permission]] = {
    Role.AGENT: [
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.TASK_CREATE,
        Permission.TASK_READ,
        Permission.TASK_UPDATE,
        Permission.ROUTE_MESSAGE,
    ],
    Role.SERVICE: [
        Permission.AGENT_REGISTER,
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.AGENT_DELETE,
        Permission.TASK_CREATE,
        Permission.TASK_READ,
        Permission.TASK_UPDATE,
        Permission.TASK_DELETE,
        Permission.ROUTE_MESSAGE,
    ],
    Role.ADMIN: [Permission.ADMIN],  # Admin has all permissions
}


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str = Field(..., description="Subject (agent_id or user_id)")
    jti: str = Field(default_factory=lambda: str(uuid4()), description="JWT ID")
    token_type: TokenType = Field(..., description="Token type")
    role: Role = Field(..., description="User role")
    permissions: list[Permission] = Field(
        default_factory=list, description="Permissions"
    )
    exp: datetime = Field(..., description="Expiration timestamp")
    iat: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Issued at timestamp"
    )
    agent_id: str | None = Field(None, description="Agent identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("exp", "iat", mode="before")
    @classmethod
    def convert_timestamp(cls, v: datetime | int | float) -> datetime:
        """Convert Unix timestamp to datetime if needed."""
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, UTC)
        return v

    @classmethod
    def create(
        cls,
        subject: str,
        role: Role,
        token_type: TokenType = TokenType.ACCESS,
        expiration_hours: int = 24,
        agent_id: str | None = None,
        additional_permissions: list[Permission] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TokenPayload":
        """
        Create a new token payload.

        Args:
            subject: Token subject (agent_id or user_id)
            role: User role
            token_type: Token type
            expiration_hours: Token expiration in hours
            agent_id: Optional agent identifier
            additional_permissions: Additional permissions beyond role defaults
            metadata: Additional metadata

        Returns:
            Token payload
        """
        # Get role-based permissions
        permissions = list(ROLE_PERMISSIONS.get(role, []))

        # Add additional permissions
        if additional_permissions:
            permissions.extend(additional_permissions)

        # Remove duplicates
        permissions = list(set(permissions))

        return cls(
            sub=subject,
            token_type=token_type,
            role=role,
            permissions=permissions,
            exp=datetime.now(UTC) + timedelta(hours=expiration_hours),
            agent_id=agent_id,
            metadata=metadata or {},
        )

    def has_permission(self, permission: Permission) -> bool:
        """Check if token has specific permission."""
        # Admin has all permissions
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(UTC) > self.exp


class SignedRequest(BaseModel):
    """Request with RSA signature."""

    agent_id: str = Field(..., description="Agent identifier")
    timestamp: datetime = Field(..., description="Request timestamp")
    nonce: str = Field(
        default_factory=lambda: str(uuid4()), description="Request nonce"
    )
    payload: dict[str, Any] = Field(..., description="Request payload")
    signature: str = Field(..., description="RSA signature (base64 encoded)")

    def is_expired(self, max_age_seconds: int = 300) -> bool:
        """Check if request is expired (default 5 minutes)."""
        return (datetime.now(UTC) - self.timestamp).total_seconds() > max_age_seconds


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    agent_id: str = Field(..., description="Agent identifier")
    requests_count: int = Field(default=0, description="Current request count")
    window_start: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Rate limit window start"
    )
    window_duration_seconds: int = Field(
        default=60, description="Rate limit window duration"
    )
    max_requests: int = Field(default=1000, description="Max requests per window")

    def reset_if_expired(self) -> None:
        """Reset counter if window has expired."""
        if self.is_window_expired():
            self.requests_count = 0
            self.window_start = datetime.now(UTC)

    def is_window_expired(self) -> bool:
        """Check if rate limit window has expired."""
        elapsed = (datetime.now(UTC) - self.window_start).total_seconds()
        return elapsed >= self.window_duration_seconds

    def is_rate_limited(self) -> bool:
        """Check if rate limit is exceeded."""
        self.reset_if_expired()
        return self.requests_count >= self.max_requests

    def increment(self) -> bool:
        """
        Increment request counter.

        Returns:
            True if request is allowed, False if rate limited
        """
        self.reset_if_expired()

        if self.is_rate_limited():
            return False

        self.requests_count += 1
        return True

    def get_reset_time(self) -> datetime:
        """Get time when rate limit window resets."""
        return self.window_start + timedelta(seconds=self.window_duration_seconds)

    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        self.reset_if_expired()
        return max(0, self.max_requests - self.requests_count)


class AuthenticationRequest(BaseModel):
    """Authentication request."""

    agent_id: str = Field(..., description="Agent identifier")
    credentials: dict[str, Any] = Field(
        ..., description="Credentials (e.g., API key, certificate)"
    )
    requested_permissions: list[Permission] | None = Field(
        None, description="Requested permissions"
    )


class AuthenticationResponse(BaseModel):
    """Authentication response."""

    success: bool = Field(..., description="Authentication success")
    access_token: str | None = Field(None, description="JWT access token")
    refresh_token: str | None = Field(None, description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int | None = Field(None, description="Token expiration in seconds")
    error_message: str | None = Field(None, description="Error message on failure")
