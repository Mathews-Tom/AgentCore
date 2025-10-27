"""Tests for agent configuration models."""

import pytest
from pydantic import ValidationError

from agentcore.agent_runtime.models.agent_config import (
    AgentConfig,
    AgentPhilosophy,
    ResourceLimits,
    SecurityProfile)


def test_agent_philosophy_enum() -> None:
    """Test AgentPhilosophy enum values."""
    assert AgentPhilosophy.REACT == "react"
    assert AgentPhilosophy.CHAIN_OF_THOUGHT == "cot"
    assert AgentPhilosophy.MULTI_AGENT == "multi_agent"
    assert AgentPhilosophy.AUTONOMOUS == "autonomous"


def test_resource_limits_defaults() -> None:
    """Test ResourceLimits default values."""
    limits = ResourceLimits()
    assert limits.max_memory_mb == 512
    assert limits.max_cpu_cores == 1.0
    assert limits.max_execution_time_seconds == 300
    assert limits.max_file_descriptors == 100
    assert limits.network_access == "restricted"
    assert limits.storage_quota_mb == 1024


def test_resource_limits_validation() -> None:
    """Test ResourceLimits validation."""
    # Test minimum values
    with pytest.raises(ValidationError):
        ResourceLimits(max_memory_mb=50)  # Below minimum 128

    with pytest.raises(ValidationError):
        ResourceLimits(max_cpu_cores=0.05)  # Below minimum 0.1

    # Test maximum values
    with pytest.raises(ValidationError):
        ResourceLimits(max_memory_mb=10000)  # Above maximum 8192


def test_security_profile_defaults() -> None:
    """Test SecurityProfile default values."""
    profile = SecurityProfile()
    assert profile.profile_name == "standard"
    assert profile.allowed_syscalls == []
    assert "mount" in profile.blocked_syscalls
    assert "chroot" in profile.blocked_syscalls
    assert profile.user_namespace is True
    assert profile.read_only_filesystem is True
    assert profile.no_new_privileges is True


def test_agent_config_valid() -> None:
    """Test valid AgentConfig creation."""
    config = AgentConfig(
        agent_id="test-agent-001",
        philosophy=AgentPhilosophy.REACT)
    assert config.agent_id == "test-agent-001"
    assert config.philosophy == AgentPhilosophy.REACT
    assert config.tools == []
    assert config.environment_variables == {}


def test_agent_config_invalid_agent_id() -> None:
    """Test AgentConfig with invalid agent_id."""
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="invalid agent id!",  # Contains spaces and special chars
            philosophy=AgentPhilosophy.REACT)


def test_agent_config_too_many_tools() -> None:
    """Test AgentConfig with too many tools."""
    tools = [f"tool_{i}" for i in range(101)]  # 101 tools (max is 100)
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            tools=tools)


def test_agent_config_restricted_env_vars() -> None:
    """Test AgentConfig with restricted environment variables."""
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            environment_variables={"PATH": "/usr/bin"},  # PATH is restricted
        )


def test_agent_config_with_custom_resources() -> None:
    """Test AgentConfig with custom resource limits."""
    config = AgentConfig(
        agent_id="high-performance-agent",
        philosophy=AgentPhilosophy.AUTONOMOUS,
        resource_limits=ResourceLimits(
            max_memory_mb=2048,
            max_cpu_cores=4.0,
            max_execution_time_seconds=600))
    assert config.resource_limits.max_memory_mb == 2048
    assert config.resource_limits.max_cpu_cores == 4.0
    assert config.resource_limits.max_execution_time_seconds == 600


def test_agent_config_with_security_profile() -> None:
    """Test AgentConfig with custom security profile."""
    config = AgentConfig(
        agent_id="secure-agent",
        philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT,
        security_profile=SecurityProfile(
            profile_name="minimal",
            user_namespace=True,
            read_only_filesystem=True))
    assert config.security_profile.profile_name == "minimal"
    assert config.security_profile.user_namespace is True
