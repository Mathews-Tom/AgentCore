"""Tests for tool startup and registration configuration."""

import os
from unittest.mock import patch

import pytest

from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.agent_runtime.tools.startup import (
    ToolStartupConfig,
    initialize_tool_system,
    register_builtin_tools,
)


class TestToolStartupConfig:
    """Test tool startup configuration from environment variables."""

    def test_config_defaults(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = ToolStartupConfig()

            assert config.google_api_key is None
            assert config.google_search_engine_id is None
            assert config.enable_code_execution is False
            assert config.code_execution_timeout == 30
            assert config.enable_http_tools is True
            assert config.enable_file_operations is True
            assert config.enable_all_tools is True

    def test_config_from_environment(self):
        """Test configuration reads from environment variables."""
        env_vars = {
            "GOOGLE_API_KEY": "test_key",
            "GOOGLE_SEARCH_ENGINE_ID": "test_engine_id",
            "ENABLE_CODE_EXECUTION": "true",
            "CODE_EXECUTION_TIMEOUT": "60",
            "ENABLE_HTTP_TOOLS": "false",
            "ENABLE_FILE_OPERATIONS": "false",
            "ENABLE_ALL_TOOLS": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ToolStartupConfig()

            assert config.google_api_key == "test_key"
            assert config.google_search_engine_id == "test_engine_id"
            assert config.enable_code_execution is True
            assert config.code_execution_timeout == 60
            assert config.enable_http_tools is False
            assert config.enable_file_operations is False
            assert config.enable_all_tools is False

    def test_config_file_allowed_directories(self):
        """Test file allowed directories parsing."""
        env_vars = {"FILE_ALLOWED_DIRECTORIES": "/tmp,/var/tmp,/data"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = ToolStartupConfig()

            assert config.file_allowed_directories == ["/tmp", "/var/tmp", "/data"]

    def test_should_register_google_search_with_credentials(self):
        """Test Google Search registration when credentials are present."""
        env_vars = {
            "GOOGLE_API_KEY": "test_key",
            "GOOGLE_SEARCH_ENGINE_ID": "test_engine_id",
            "ENABLE_ALL_TOOLS": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ToolStartupConfig()
            assert config.should_register_google_search() is True

    def test_should_not_register_google_search_without_credentials(self):
        """Test Google Search not registered when credentials missing."""
        with patch.dict(os.environ, {"ENABLE_ALL_TOOLS": "true"}, clear=True):
            config = ToolStartupConfig()
            assert config.should_register_google_search() is False

    def test_should_register_code_execution_when_enabled(self):
        """Test code execution registration when enabled."""
        env_vars = {"ENABLE_CODE_EXECUTION": "true", "ENABLE_ALL_TOOLS": "true"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = ToolStartupConfig()
            assert config.should_register_code_execution() is True

    def test_should_not_register_code_execution_when_disabled(self):
        """Test code execution not registered when disabled."""
        env_vars = {"ENABLE_CODE_EXECUTION": "false", "ENABLE_ALL_TOOLS": "true"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = ToolStartupConfig()
            assert config.should_register_code_execution() is False

    def test_should_not_register_tools_when_all_disabled(self):
        """Test no tools registered when ENABLE_ALL_TOOLS=false."""
        env_vars = {"ENABLE_ALL_TOOLS": "false"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = ToolStartupConfig()

            assert config.should_register_utility_tools() is False
            assert config.should_register_google_search() is False
            assert config.should_register_wikipedia_search() is False
            assert config.should_register_code_execution() is False
            assert config.should_register_http_tools() is False
            assert config.should_register_file_operations() is False


class TestRegisterBuiltinTools:
    """Test built-in tool registration."""

    def test_register_all_tools_with_defaults(self):
        """Test registration of all tools with default config."""
        registry = ToolRegistry()

        with patch.dict(os.environ, {}, clear=True):
            stats = register_builtin_tools(registry)

            # Should register utility tools, Wikipedia, HTTP tools, file operations
            # Should skip Google Search (no API key) and code execution (disabled by default)
            assert stats["total_registered"] > 0
            assert stats["total_skipped"] > 0
            assert "calculator" in stats["registered_tools"]
            assert "echo" in stats["registered_tools"]
            assert "get_current_time" in stats["registered_tools"]
            assert "file_operations" in stats["registered_tools"]
            assert "wikipedia_search" in stats["registered_tools"]
            assert "http_request" in stats["registered_tools"]

    def test_register_with_google_credentials(self):
        """Test Google Search registered when credentials provided."""
        registry = ToolRegistry()
        env_vars = {
            "GOOGLE_API_KEY": "test_key",
            "GOOGLE_SEARCH_ENGINE_ID": "test_engine_id",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            assert "google_search" in stats["registered_tools"]

    def test_register_with_code_execution_enabled(self):
        """Test code execution tools registered when enabled."""
        registry = ToolRegistry()
        env_vars = {"ENABLE_CODE_EXECUTION": "true"}

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            assert "execute_python" in stats["registered_tools"]
            assert "evaluate_expression" in stats["registered_tools"]

    def test_skip_tools_when_disabled(self):
        """Test tools skipped when disabled via environment."""
        registry = ToolRegistry()
        env_vars = {
            "ENABLE_HTTP_TOOLS": "false",
            "ENABLE_FILE_OPERATIONS": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            # Check that HTTP tools and file operations were skipped
            skipped_ids = [item["tool_id"] for item in stats["skipped_tools"]]
            assert "http_tools" in skipped_ids
            assert "file_operations" in skipped_ids

    def test_register_nothing_when_all_disabled(self):
        """Test no tools registered when ENABLE_ALL_TOOLS=false."""
        registry = ToolRegistry()
        env_vars = {"ENABLE_ALL_TOOLS": "false"}

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            assert stats["total_registered"] == 0
            assert stats["total_skipped"] > 0

    def test_register_with_custom_file_directories(self):
        """Test file operations tool with custom allowed directories."""
        registry = ToolRegistry()
        env_vars = {"FILE_ALLOWED_DIRECTORIES": "/custom/path1,/custom/path2"}

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            assert "file_operations" in stats["registered_tools"]

            # Verify the tool was registered
            tool = registry.get("file_operations")
            assert tool is not None
            assert tool.allowed_directories == ["/custom/path1", "/custom/path2"]

    def test_registration_stats_structure(self):
        """Test registration statistics have correct structure."""
        registry = ToolRegistry()

        with patch.dict(os.environ, {}, clear=True):
            stats = register_builtin_tools(registry)

            # Verify stats structure
            assert "total_registered" in stats
            assert "total_skipped" in stats
            assert "total_failed" in stats
            assert "registered_tools" in stats
            assert "skipped_tools" in stats
            assert "failed_tools" in stats

            assert isinstance(stats["total_registered"], int)
            assert isinstance(stats["registered_tools"], list)
            assert isinstance(stats["skipped_tools"], list)

    def test_registration_with_custom_config(self):
        """Test registration with custom config object."""
        registry = ToolRegistry()
        config = ToolStartupConfig()
        config.enable_all_tools = True
        config.enable_code_execution = True

        with patch.dict(os.environ, {}, clear=True):
            stats = register_builtin_tools(registry, config=config)

            assert "execute_python" in stats["registered_tools"]
            assert "evaluate_expression" in stats["registered_tools"]


class TestInitializeToolSystem:
    """Test tool system initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_new_registry(self):
        """Test initialization creates new registry if none provided."""
        with patch.dict(os.environ, {}, clear=True):
            registry = await initialize_tool_system()

            assert registry is not None
            assert isinstance(registry, ToolRegistry)
            assert len(registry.list_all()) > 0

    @pytest.mark.asyncio
    async def test_initialize_uses_provided_registry(self):
        """Test initialization uses provided registry."""
        existing_registry = ToolRegistry()

        with patch.dict(os.environ, {}, clear=True):
            registry = await initialize_tool_system(registry=existing_registry)

            assert registry is existing_registry
            assert len(registry.list_all()) > 0

    @pytest.mark.asyncio
    async def test_initialize_returns_stats(self):
        """Test initialization completes successfully."""
        with patch.dict(os.environ, {}, clear=True):
            registry = await initialize_tool_system()

            # Verify tools were registered
            tools = registry.list_all()
            assert len(tools) > 0

            # Check for expected tools
            tool_ids = [t.metadata.tool_id for t in tools]
            assert "calculator" in tool_ids
            assert "echo" in tool_ids


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_production_configuration(self):
        """Test production-like configuration."""
        registry = ToolRegistry()
        env_vars = {
            "ENABLE_ALL_TOOLS": "true",
            "ENABLE_CODE_EXECUTION": "false",  # Disabled in production
            "ENABLE_HTTP_TOOLS": "true",
            "ENABLE_FILE_OPERATIONS": "true",
            "FILE_ALLOWED_DIRECTORIES": "/app/data,/app/uploads",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            # Should have utility, HTTP, file ops, search tools
            assert stats["total_registered"] > 5
            assert "file_operations" in stats["registered_tools"]
            assert "http_request" in stats["registered_tools"]

            # Code execution should be skipped
            skipped_ids = [item["tool_id"] for item in stats["skipped_tools"]]
            assert "code_execution_tools" in skipped_ids

    def test_development_configuration(self):
        """Test development configuration with all tools."""
        registry = ToolRegistry()
        env_vars = {
            "ENABLE_ALL_TOOLS": "true",
            "ENABLE_CODE_EXECUTION": "true",
            "GOOGLE_API_KEY": "dev_key",
            "GOOGLE_SEARCH_ENGINE_ID": "dev_engine",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            # Should register everything
            assert "execute_python" in stats["registered_tools"]
            assert "google_search" in stats["registered_tools"]
            assert stats["total_failed"] == 0

    def test_minimal_configuration(self):
        """Test minimal configuration with only essential tools."""
        registry = ToolRegistry()
        env_vars = {
            "ENABLE_ALL_TOOLS": "true",
            "ENABLE_HTTP_TOOLS": "false",
            "ENABLE_CODE_EXECUTION": "false",
            "ENABLE_FILE_OPERATIONS": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            stats = register_builtin_tools(registry)

            # Should only have utility and search tools
            assert "calculator" in stats["registered_tools"]
            assert "wikipedia_search" in stats["registered_tools"]

            # Others should be skipped
            skipped_ids = [item["tool_id"] for item in stats["skipped_tools"]]
            assert "http_tools" in skipped_ids
            assert "code_execution_tools" in skipped_ids
            assert "file_operations" in skipped_ids
