"""Dependency injection container for AgentCore CLI.

This module provides a centralized dependency injection container that manages
object creation and wiring across all layers of the CLI architecture.

Design principles:
- Singleton instances for infrastructure (config, transport, client)
- On-demand creation for services (no caching)
- Easy to mock for testing
- Clear dependency graph
- Lazy initialization

Factory functions:
- get_config(): Load and cache configuration
- get_transport(): Create and cache HTTP transport
- get_jsonrpc_client(): Create and cache JSON-RPC client
- get_agent_service(): Create agent service instance (no caching)
- get_task_service(): Create task service instance (no caching)
- get_session_service(): Create session service instance (no caching)
- get_workflow_service(): Create workflow service instance (no caching)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any
import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentcore_cli.transport.http import HttpTransport
from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.agent import AgentService
from agentcore_cli.services.task import TaskService
from agentcore_cli.services.session import SessionService
from agentcore_cli.services.workflow import WorkflowService


class ApiConfig(BaseModel):
    """API server configuration.

    Attributes:
        url: Base URL for API server (e.g., "http://localhost:8001")
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        verify_ssl: Whether to verify SSL certificates
    """

    url: str = Field(default="http://localhost:8001")
    timeout: int = Field(default=30, ge=1, le=300)
    retries: int = Field(default=3, ge=0, le=10)
    verify_ssl: bool = Field(default=True)


class AuthConfig(BaseModel):
    """Authentication configuration.

    Attributes:
        type: Authentication type ("none", "jwt", "api_key")
        token: Authentication token (JWT or API key)
    """

    type: str = Field(default="none", pattern="^(none|jwt|api_key)$")
    token: str | None = Field(default=None)


class Config(BaseSettings):
    """AgentCore CLI configuration.

    Configuration precedence (highest to lowest):
    1. Environment variables (AGENTCORE_*)
    2. Configuration file (.agentcore.toml or ~/.agentcore/config.toml)
    3. Defaults

    Environment variables:
    - AGENTCORE_API_URL: API server URL
    - AGENTCORE_API_TIMEOUT: Request timeout in seconds
    - AGENTCORE_API_RETRIES: Number of retry attempts
    - AGENTCORE_API_VERIFY_SSL: Whether to verify SSL (true/false)
    - AGENTCORE_AUTH_TYPE: Authentication type (none/jwt/api_key)
    - AGENTCORE_AUTH_TOKEN: Authentication token

    Attributes:
        api: API server configuration
        auth: Authentication configuration

    Example:
        >>> config = Config()
        >>> print(config.api.url)
        'http://localhost:8001'
    """

    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    model_config = SettingsConfigDict(
        env_prefix="AGENTCORE_",
        env_nested_delimiter="_",
        case_sensitive=False,
    )


# Global container state for testing/mocking
_overrides: dict[str, Any] = {}


def set_override(key: str, value: Any) -> None:
    """Override a container dependency for testing.

    This allows tests to inject mock implementations without modifying
    the container code.

    Args:
        key: Dependency key (e.g., "transport", "client", "config")
        value: Mock or test implementation

    Example:
        >>> mock_transport = Mock(spec=HttpTransport)
        >>> set_override("transport", mock_transport)
        >>> transport = get_transport()  # Returns mock
        >>> clear_overrides()
    """
    _overrides[key] = value


def clear_overrides() -> None:
    """Clear all dependency overrides.

    Should be called in test teardown to reset container state.

    Example:
        >>> clear_overrides()
        >>> get_config.cache_clear()  # Clear LRU cache
    """
    _overrides.clear()


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache configuration.

    Configuration is loaded once and cached for the lifetime of the process.
    Uses pydantic_settings to load from environment variables.

    Returns:
        Configuration instance

    Example:
        >>> config = get_config()
        >>> print(config.api.url)
        'http://localhost:8001'

    Note:
        For testing, use set_override("config", mock_config) to inject
        a test configuration.
    """
    if "config" in _overrides:
        override = _overrides["config"]
        if not isinstance(override, Config):
            raise TypeError("Override for 'config' must be a Config instance")
        return override

    return Config()


@lru_cache(maxsize=1)
def get_transport() -> HttpTransport:
    """Create and cache HTTP transport.

    Transport is created once and cached for connection pooling efficiency.
    Configuration is loaded from get_config().

    Returns:
        HTTP transport instance

    Example:
        >>> transport = get_transport()
        >>> response = transport.post("/api/v1/jsonrpc", {...})

    Note:
        For testing, use set_override("transport", mock_transport) to inject
        a mock transport.
    """
    if "transport" in _overrides:
        override = _overrides["transport"]
        if not isinstance(override, HttpTransport):
            raise TypeError("Override for 'transport' must be an HttpTransport instance")
        return override

    config = get_config()
    return HttpTransport(
        base_url=config.api.url,
        timeout=config.api.timeout,
        retries=config.api.retries,
        verify_ssl=config.api.verify_ssl,
    )


@lru_cache(maxsize=1)
def get_jsonrpc_client() -> JsonRpcClient:
    """Create and cache JSON-RPC client.

    Client is created once and cached. Depends on get_transport() and get_config().
    Authentication token is included if auth.type is "jwt".

    Returns:
        JSON-RPC client instance

    Example:
        >>> client = get_jsonrpc_client()
        >>> result = client.call("agent.register", {"name": "test"})

    Note:
        For testing, use set_override("client", mock_client) to inject
        a mock client.
    """
    if "client" in _overrides:
        override = _overrides["client"]
        if not isinstance(override, JsonRpcClient):
            raise TypeError("Override for 'client' must be a JsonRpcClient instance")
        return override

    config = get_config()
    transport = get_transport()

    # Include auth token if configured
    auth_token = None
    if config.auth.type == "jwt" and config.auth.token:
        auth_token = config.auth.token

    return JsonRpcClient(
        transport=transport,
        auth_token=auth_token,
    )


def get_agent_service() -> AgentService:
    """Create agent service instance.

    Service is created on-demand (NOT cached) to avoid state issues.
    Depends on get_jsonrpc_client().

    Returns:
        Agent service instance

    Example:
        >>> service = get_agent_service()
        >>> agent_id = service.register("analyzer", ["python"])

    Note:
        For testing, use set_override("agent_service", mock_service) to inject
        a mock service.
    """
    if "agent_service" in _overrides:
        override = _overrides["agent_service"]
        if not isinstance(override, AgentService):
            raise TypeError("Override for 'agent_service' must be an AgentService instance")
        return override

    client = get_jsonrpc_client()
    return AgentService(client)


def get_task_service() -> TaskService:
    """Create task service instance.

    Service is created on-demand (NOT cached) to avoid state issues.
    Depends on get_jsonrpc_client().

    Returns:
        Task service instance

    Example:
        >>> service = get_task_service()
        >>> task_id = service.create("Process data", {...})

    Note:
        For testing, use set_override("task_service", mock_service) to inject
        a mock service.
    """
    if "task_service" in _overrides:
        override = _overrides["task_service"]
        if not isinstance(override, TaskService):
            raise TypeError("Override for 'task_service' must be a TaskService instance")
        return override

    client = get_jsonrpc_client()
    return TaskService(client)


def get_session_service() -> SessionService:
    """Create session service instance.

    Service is created on-demand (NOT cached) to avoid state issues.
    Depends on get_jsonrpc_client().

    Returns:
        Session service instance

    Example:
        >>> service = get_session_service()
        >>> session_id = service.create("my-session", {...})

    Note:
        For testing, use set_override("session_service", mock_service) to inject
        a mock service.
    """
    if "session_service" in _overrides:
        override = _overrides["session_service"]
        if not isinstance(override, SessionService):
            raise TypeError("Override for 'session_service' must be a SessionService instance")
        return override

    client = get_jsonrpc_client()
    return SessionService(client)


def get_workflow_service() -> WorkflowService:
    """Create workflow service instance.

    Service is created on-demand (NOT cached) to avoid state issues.
    Depends on get_jsonrpc_client().

    Returns:
        Workflow service instance

    Example:
        >>> service = get_workflow_service()
        >>> workflow_id = service.run("workflow.yaml", {...})

    Note:
        For testing, use set_override("workflow_service", mock_service) to inject
        a mock service.
    """
    if "workflow_service" in _overrides:
        override = _overrides["workflow_service"]
        if not isinstance(override, WorkflowService):
            raise TypeError("Override for 'workflow_service' must be a WorkflowService instance")
        return override

    client = get_jsonrpc_client()
    return WorkflowService(client)


def reset_container() -> None:
    """Reset container state for testing.

    Clears all caches and overrides. Should be called in test teardown.

    Example:
        >>> reset_container()
    """
    clear_overrides()
    get_config.cache_clear()
    get_transport.cache_clear()
    get_jsonrpc_client.cache_clear()
