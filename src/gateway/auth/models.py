"""
Authentication Models

Pydantic models for JWT tokens, user authentication, and session management.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for RBAC authorization."""

    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


class User(BaseModel):
    """User model for authentication and authorization."""

    id: UUID = Field(default_factory=uuid4, description="User unique identifier")
    username: str = Field(..., description="Username for authentication")
    email: str | None = Field(None, description="User email address")
    roles: list[UserRole] = Field(default=[UserRole.USER], description="User roles")
    is_active: bool = Field(default=True, description="User account status")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="User creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="User last update timestamp")

    model_config = {"from_attributes": True}


class TokenRequest(BaseModel):
    """Token generation request."""

    grant_type: str = Field(
        ...,
        description="OAuth grant type (password, client_credentials, refresh_token)",
        examples=["password", "client_credentials"],
    )
    username: str | None = Field(
        None,
        description="Username for password grant",
        examples=["user", "admin"],
    )
    password: str | None = Field(
        None,
        description="Password for password grant",
        examples=["user123"],
    )
    client_id: str | None = Field(
        None,
        description="Client ID for client credentials",
        examples=["service"],
    )
    client_secret: str | None = Field(
        None,
        description="Client secret for client credentials",
        examples=["service123"],
    )
    refresh_token: str | None = Field(
        None,
        description="Refresh token for token refresh",
        examples=["eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )
    scope: str | None = Field(
        None,
        description="Requested token scope",
        examples=["user:read user:write agent:read agent:execute"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "grant_type": "password",
                    "username": "user",
                    "password": "user123",
                    "scope": "user:read user:write",
                },
                {
                    "grant_type": "client_credentials",
                    "client_id": "service",
                    "client_secret": "service123",
                    "scope": "service:read service:write",
                },
            ]
        }
    }


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str = Field(
        ...,
        description="JWT access token",
        examples=["eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDIiLCJ1c2VybmFtZSI6InVzZXIiLCJyb2xlcyI6WyJ1c2VyIl0sInNlc3Npb25faWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAiLCJpYXQiOjE3MjkzMjAwMDAsImV4cCI6MTcyOTMyMzYwMCwic2NvcGUiOiJ1c2VyOnJlYWQgdXNlcjp3cml0ZSIsImp0aSI6Ijc3MGU4NDAwLWUyOWItNDFkNC1hNzE2LTQ0NjY1NTQ0MDAwMCJ9.signature"],
    )
    token_type: str = Field(
        default="Bearer",
        description="Token type",
        examples=["Bearer"],
    )
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds",
        examples=[3600],
    )
    refresh_token: str | None = Field(
        None,
        description="Refresh token for token renewal",
        examples=["eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDIiLCJzZXNzaW9uX2lkIjoiNTUwZTg0MDAtZTI5Yi00MWQ0LWE3MTYtNDQ2NjU1NDQwMDAwIiwiaWF0IjoxNzI5MzIwMDAwLCJleHAiOjE3Mjk5MjQ4MDAsImp0aSI6Ijg4MGU4NDAwLWUyOWItNDFkNC1hNzE2LTQ0NjY1NTQ0MDAwMCIsInRva2VuX3R5cGUiOiJyZWZyZXNoIn0.signature"],
    )
    scope: str | None = Field(
        None,
        description="Granted token scope",
        examples=["user:read user:write"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDIiLCJ1c2VybmFtZSI6InVzZXIiLCJyb2xlcyI6WyJ1c2VyIl0sInNlc3Npb25faWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAiLCJpYXQiOjE3MjkzMjAwMDAsImV4cCI6MTcyOTMyMzYwMCwic2NvcGUiOiJ1c2VyOnJlYWQgdXNlcjp3cml0ZSIsImp0aSI6Ijc3MGU4NDAwLWUyOWItNDFkNC1hNzE2LTQ0NjY1NTQ0MDAwMCJ9.signature",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDIiLCJzZXNzaW9uX2lkIjoiNTUwZTg0MDAtZTI5Yi00MWQ0LWE3MTYtNDQ2NjU1NDQwMDAwIiwiaWF0IjoxNzI5MzIwMDAwLCJleHAiOjE3Mjk5MjQ4MDAsImp0aSI6Ijg4MGU4NDAwLWUyOWItNDFkNC1hNzE2LTQ0NjY1NTQ0MDAwMCIsInRva2VuX3R5cGUiOiJyZWZyZXNoIn0.signature",
                    "scope": "user:read user:write",
                }
            ]
        }
    }


class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str = Field(..., description="Subject (user ID)")
    username: str = Field(..., description="Username")
    roles: list[UserRole] = Field(..., description="User roles")
    session_id: str = Field(..., description="Session identifier")
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")
    scope: str | None = Field(None, description="Token scope")
    jti: str = Field(..., description="JWT ID for tracking")


class RefreshTokenPayload(BaseModel):
    """Refresh token payload structure."""

    sub: str = Field(..., description="Subject (user ID)")
    session_id: str = Field(..., description="Session identifier")
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")
    jti: str = Field(..., description="JWT ID for tracking")
    token_type: str = Field(default="refresh", description="Token type")


class Session(BaseModel):
    """User session information."""

    session_id: str = Field(..., description="Session unique identifier")
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    roles: list[UserRole] = Field(..., description="User roles")
    created_at: datetime = Field(..., description="Session creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    ip_address: str | None = Field(None, description="Client IP address")
    user_agent: str | None = Field(None, description="Client user agent")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")


class AuthError(BaseModel):
    """Authentication error response."""

    error: str = Field(..., description="Error code")
    error_description: str = Field(..., description="Human-readable error description")
    error_uri: str | None = Field(None, description="URI for error details")
