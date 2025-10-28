"""Tests for dependency injection container."""

from __future__ import annotations

from unittest.mock import Mock, patch
import os
import pytest

from agentcore_cli.container import (
    ApiConfig,
    AuthConfig,
    Config,
    get_config,
    get_transport,
    get_jsonrpc_client,
    get_agent_service,
    get_task_service,
    get_session_service,
    get_workflow_service,
    set_override,
    clear_overrides,
    reset_container)
from agentcore_cli.transport.http import HttpTransport
from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.agent import AgentService
from agentcore_cli.services.task import TaskService
from agentcore_cli.services.session import SessionService
from agentcore_cli.services.workflow import WorkflowService


class TestApiConfig:
    """Tests for ApiConfig model."""

    def test_default_values(self) -> None:
        """Test that ApiConfig has correct default values."""
        config = ApiConfig()
        assert config.url == "http://localhost:8001"
        assert config.timeout == 30
        assert config.retries == 3
        assert config.verify_ssl is True

    def test_custom_values(self) -> None:
        """Test that ApiConfig accepts custom values."""
        config = ApiConfig(
            url="https://api.example.com",
            timeout=60,
            retries=5,
            verify_ssl=False)
        assert config.url == "https://api.example.com"
        assert config.timeout == 60
        assert config.retries == 5
        assert config.verify_ssl is False

    def test_timeout_validation(self) -> None:
        """Test that timeout is validated."""
        # Valid values
        ApiConfig(timeout=1)
        ApiConfig(timeout=300)

        # Invalid values
        with pytest.raises(Exception):  # Pydantic validation error
            ApiConfig(timeout=0)
        with pytest.raises(Exception):
            ApiConfig(timeout=301)
        with pytest.raises(Exception):
            ApiConfig(timeout=-1)

    def test_retries_validation(self) -> None:
        """Test that retries are validated."""
        # Valid values
        ApiConfig(retries=0)
        ApiConfig(retries=10)

        # Invalid values
        with pytest.raises(Exception):  # Pydantic validation error
            ApiConfig(retries=-1)
        with pytest.raises(Exception):
            ApiConfig(retries=11)


class TestAuthConfig:
    """Tests for AuthConfig model."""

    def test_default_values(self) -> None:
        """Test that AuthConfig has correct default values."""
        config = AuthConfig()
        assert config.type == "none"
        assert config.token is None

    def test_jwt_auth(self) -> None:
        """Test JWT authentication configuration."""
        config = AuthConfig(type="jwt", token="jwt-token-123")
        assert config.type == "jwt"
        assert config.token == "jwt-token-123"

    def test_api_key_auth(self) -> None:
        """Test API key authentication configuration."""
        config = AuthConfig(type="api_key", token="api-key-123")
        assert config.type == "api_key"
        assert config.token == "api-key-123"

    def test_invalid_auth_type(self) -> None:
        """Test that invalid auth types are rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            AuthConfig(type="invalid")


class TestConfig:
    """Tests for Config model."""

    def test_default_values(self) -> None:
        """Test that Config has correct default values."""
        config = Config()
        assert config.api.url == "http://localhost:8001"
        assert config.api.timeout == 30
        assert config.api.retries == 3
        assert config.api.verify_ssl is True
        assert config.auth.type == "none"
        assert config.auth.token is None

    def test_environment_variables(self) -> None:
        """Test that Config loads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AGENTCORE_API_URL": "https://custom.example.com",
                "AGENTCORE_API_TIMEOUT": "60",
                "AGENTCORE_API_RETRIES": "5",
                "AGENTCORE_AUTH_TYPE": "jwt",
                "AGENTCORE_AUTH_TOKEN": "test-token",
            },
            clear=False):
            config = Config()
            assert config.api.url == "https://custom.example.com"
            assert config.api.timeout == 60
            assert config.api.retries == 5
            assert config.auth.type == "jwt"
            assert config.auth.token == "test-token"


class TestGetConfig:
    """Tests for get_config factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_config_instance(self) -> None:
        """Test that get_config returns Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_caching_works(self) -> None:
        """Test that get_config caches the config instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2  # Same instance

    def test_override_works(self) -> None:
        """Test that set_override works for config."""
        mock_config = Mock(spec=Config)
        set_override("config", mock_config)

        config = get_config()
        assert config is mock_config

    def test_clear_overrides_works(self) -> None:
        """Test that clear_overrides removes config override."""
        mock_config = Mock(spec=Config)
        set_override("config", mock_config)
        clear_overrides()

        # Cache should still have old value, clear it
        get_config.cache_clear()

        config = get_config()
        assert config is not mock_config
        assert isinstance(config, Config)


class TestGetTransport:
    """Tests for get_transport factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_transport_instance(self) -> None:
        """Test that get_transport returns HttpTransport instance."""
        transport = get_transport()
        assert isinstance(transport, HttpTransport)

    def test_caching_works(self) -> None:
        """Test that get_transport caches the transport instance."""
        transport1 = get_transport()
        transport2 = get_transport()
        assert transport1 is transport2  # Same instance

    def test_uses_config_values(self) -> None:
        """Test that transport is created with config values."""
        with patch.dict(
            os.environ,
            {
                "AGENTCORE_API_URL": "https://test.example.com",
                "AGENTCORE_API_TIMEOUT": "45",
                "AGENTCORE_API_RETRIES": "2",
            },
            clear=False):
            reset_container()  # Clear cache
            transport = get_transport()
            assert transport.base_url == "https://test.example.com"
            assert transport.timeout == 45

    def test_override_works(self) -> None:
        """Test that set_override works for transport."""
        mock_transport = Mock(spec=HttpTransport)
        set_override("transport", mock_transport)

        transport = get_transport()
        assert transport is mock_transport


class TestGetJsonRpcClient:
    """Tests for get_jsonrpc_client factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_client_instance(self) -> None:
        """Test that get_jsonrpc_client returns JsonRpcClient instance."""
        client = get_jsonrpc_client()
        assert isinstance(client, JsonRpcClient)

    def test_caching_works(self) -> None:
        """Test that get_jsonrpc_client caches the client instance."""
        client1 = get_jsonrpc_client()
        client2 = get_jsonrpc_client()
        assert client1 is client2  # Same instance

    def test_no_auth_token_by_default(self) -> None:
        """Test that client has no auth token by default."""
        client = get_jsonrpc_client()
        assert client.auth_token is None

    def test_jwt_auth_token_included(self) -> None:
        """Test that JWT auth token is included when configured."""
        with patch.dict(
            os.environ,
            {
                "AGENTCORE_AUTH_TYPE": "jwt",
                "AGENTCORE_AUTH_TOKEN": "test-jwt-token",
            },
            clear=False):
            reset_container()  # Clear cache
            client = get_jsonrpc_client()
            assert client.auth_token == "test-jwt-token"

    def test_api_key_auth_not_included(self) -> None:
        """Test that API key auth is not included (only JWT supported)."""
        with patch.dict(
            os.environ,
            {
                "AGENTCORE_AUTH_TYPE": "api_key",
                "AGENTCORE_AUTH_TOKEN": "test-api-key",
            },
            clear=False):
            reset_container()  # Clear cache
            client = get_jsonrpc_client()
            # API key auth is not JWT, so token should be None
            assert client.auth_token is None

    def test_override_works(self) -> None:
        """Test that set_override works for client."""
        mock_client = Mock(spec=JsonRpcClient)
        set_override("client", mock_client)

        client = get_jsonrpc_client()
        assert client is mock_client


class TestGetAgentService:
    """Tests for get_agent_service factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_service_instance(self) -> None:
        """Test that get_agent_service returns AgentService instance."""
        service = get_agent_service()
        assert isinstance(service, AgentService)

    def test_no_caching(self) -> None:
        """Test that get_agent_service does NOT cache instances."""
        service1 = get_agent_service()
        service2 = get_agent_service()
        # Should be different instances
        assert service1 is not service2

    def test_uses_jsonrpc_client(self) -> None:
        """Test that service is created with jsonrpc client."""
        service = get_agent_service()
        assert isinstance(service.client, JsonRpcClient)

    def test_override_works(self) -> None:
        """Test that set_override works for agent service."""
        mock_service = Mock(spec=AgentService)
        set_override("agent_service", mock_service)

        service = get_agent_service()
        assert service is mock_service


class TestGetTaskService:
    """Tests for get_task_service factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_service_instance(self) -> None:
        """Test that get_task_service returns TaskService instance."""
        service = get_task_service()
        assert isinstance(service, TaskService)

    def test_no_caching(self) -> None:
        """Test that get_task_service does NOT cache instances."""
        service1 = get_task_service()
        service2 = get_task_service()
        # Should be different instances
        assert service1 is not service2

    def test_uses_jsonrpc_client(self) -> None:
        """Test that service is created with jsonrpc client."""
        service = get_task_service()
        assert isinstance(service.client, JsonRpcClient)

    def test_override_works(self) -> None:
        """Test that set_override works for task service."""
        mock_service = Mock(spec=TaskService)
        set_override("task_service", mock_service)

        service = get_task_service()
        assert service is mock_service


class TestGetSessionService:
    """Tests for get_session_service factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_service_instance(self) -> None:
        """Test that get_session_service returns SessionService instance."""
        service = get_session_service()
        assert isinstance(service, SessionService)

    def test_no_caching(self) -> None:
        """Test that get_session_service does NOT cache instances."""
        service1 = get_session_service()
        service2 = get_session_service()
        # Should be different instances
        assert service1 is not service2

    def test_uses_jsonrpc_client(self) -> None:
        """Test that service is created with jsonrpc client."""
        service = get_session_service()
        assert isinstance(service.client, JsonRpcClient)

    def test_override_works(self) -> None:
        """Test that set_override works for session service."""
        mock_service = Mock(spec=SessionService)
        set_override("session_service", mock_service)

        service = get_session_service()
        assert service is mock_service


class TestGetWorkflowService:
    """Tests for get_workflow_service factory function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_returns_service_instance(self) -> None:
        """Test that get_workflow_service returns WorkflowService instance."""
        service = get_workflow_service()
        assert isinstance(service, WorkflowService)

    def test_no_caching(self) -> None:
        """Test that get_workflow_service does NOT cache instances."""
        service1 = get_workflow_service()
        service2 = get_workflow_service()
        # Should be different instances
        assert service1 is not service2

    def test_uses_jsonrpc_client(self) -> None:
        """Test that service is created with jsonrpc client."""
        service = get_workflow_service()
        assert isinstance(service.client, JsonRpcClient)

    def test_override_works(self) -> None:
        """Test that set_override works for workflow service."""
        mock_service = Mock(spec=WorkflowService)
        set_override("workflow_service", mock_service)

        service = get_workflow_service()
        assert service is mock_service


class TestResetContainer:
    """Tests for reset_container function."""

    def test_clears_all_caches(self) -> None:
        """Test that reset_container clears all caches."""
        # Create instances to populate caches
        get_config()
        get_transport()
        get_jsonrpc_client()

        # Reset container
        reset_container()

        # New instances should be created
        config1 = get_config()
        transport1 = get_transport()
        client1 = get_jsonrpc_client()

        # Reset again
        reset_container()

        # Should get new instances
        config2 = get_config()
        transport2 = get_transport()
        client2 = get_jsonrpc_client()

        # Instances should be different after reset
        assert config1 is not config2
        assert transport1 is not transport2
        assert client1 is not client2

    def test_clears_overrides(self) -> None:
        """Test that reset_container clears overrides."""
        mock_config = Mock(spec=Config)
        set_override("config", mock_config)

        reset_container()

        config = get_config()
        assert config is not mock_config
        assert isinstance(config, Config)


class TestDependencyWiring:
    """Tests for dependency wiring across layers."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_transport_depends_on_config(self) -> None:
        """Test that transport uses config values."""
        with patch.dict(
            os.environ,
            {
                "AGENTCORE_API_URL": "https://wiring-test.com",
            },
            clear=False):
            reset_container()
            transport = get_transport()
            assert transport.base_url == "https://wiring-test.com"

    def test_client_depends_on_transport(self) -> None:
        """Test that client uses transport instance."""
        client = get_jsonrpc_client()
        transport = get_transport()
        # Client should have the same transport instance
        assert client.transport is transport

    def test_client_depends_on_config_for_auth(self) -> None:
        """Test that client uses config for auth token."""
        with patch.dict(
            os.environ,
            {
                "AGENTCORE_AUTH_TYPE": "jwt",
                "AGENTCORE_AUTH_TOKEN": "wiring-test-token",
            },
            clear=False):
            reset_container()
            client = get_jsonrpc_client()
            assert client.auth_token == "wiring-test-token"

    def test_services_depend_on_client(self) -> None:
        """Test that all services use the same client instance."""
        client = get_jsonrpc_client()
        agent_service = get_agent_service()
        task_service = get_task_service()
        session_service = get_session_service()
        workflow_service = get_workflow_service()

        # All services should use the same client instance
        assert agent_service.client is client
        assert task_service.client is client
        assert session_service.client is client
        assert workflow_service.client is client

    def test_override_propagates_to_services(self) -> None:
        """Test that overriding client affects all services."""
        mock_client = Mock(spec=JsonRpcClient)
        set_override("client", mock_client)

        agent_service = get_agent_service()
        task_service = get_task_service()

        # Services should use the mock client
        assert agent_service.client is mock_client
        assert task_service.client is mock_client


class TestOverrideTypeValidation:
    """Tests for type validation of overrides."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container()

    def test_config_override_requires_config_instance(self) -> None:
        """Test that config override must be Config instance."""
        set_override("config", "not-a-config")
        with pytest.raises(TypeError, match="Override for 'config' must be a Config instance"):
            get_config()

    def test_transport_override_requires_transport_instance(self) -> None:
        """Test that transport override must be HttpTransport instance."""
        set_override("transport", "not-a-transport")
        with pytest.raises(TypeError, match="Override for 'transport' must be an HttpTransport instance"):
            get_transport()

    def test_client_override_requires_client_instance(self) -> None:
        """Test that client override must be JsonRpcClient instance."""
        set_override("client", "not-a-client")
        with pytest.raises(TypeError, match="Override for 'client' must be a JsonRpcClient instance"):
            get_jsonrpc_client()

    def test_agent_service_override_requires_service_instance(self) -> None:
        """Test that agent service override must be AgentService instance."""
        set_override("agent_service", "not-a-service")
        with pytest.raises(TypeError, match="Override for 'agent_service' must be an AgentService instance"):
            get_agent_service()

    def test_task_service_override_requires_service_instance(self) -> None:
        """Test that task service override must be TaskService instance."""
        set_override("task_service", "not-a-service")
        with pytest.raises(TypeError, match="Override for 'task_service' must be a TaskService instance"):
            get_task_service()

    def test_session_service_override_requires_service_instance(self) -> None:
        """Test that session service override must be SessionService instance."""
        set_override("session_service", "not-a-service")
        with pytest.raises(TypeError, match="Override for 'session_service' must be a SessionService instance"):
            get_session_service()

    def test_workflow_service_override_requires_service_instance(self) -> None:
        """Test that workflow service override must be WorkflowService instance."""
        set_override("workflow_service", "not-a-service")
        with pytest.raises(TypeError, match="Override for 'workflow_service' must be a WorkflowService instance"):
            get_workflow_service()
