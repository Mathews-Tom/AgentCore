"""Tests for configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentcore_cli.config import (
    AgentDefaults,
    ApiConfig,
    AuthConfig,
    Config,
    Defaults,
    OutputConfig,
    TaskDefaults,
    WorkflowDefaults,
)


class TestApiConfig:
    """Tests for API configuration model."""

    def test_default_values(self) -> None:
        """Test default API configuration values."""
        config = ApiConfig()
        assert config.url == "http://localhost:8001"
        assert config.timeout == 30
        assert config.retries == 3
        assert config.verify_ssl is True

    def test_url_validation(self) -> None:
        """Test URL validation."""
        # Valid URLs
        ApiConfig(url="http://localhost:8001")
        ApiConfig(url="https://api.example.com")

        # Invalid URL
        with pytest.raises(ValueError, match="must start with http"):
            ApiConfig(url="localhost:8001")

    def test_url_normalization(self) -> None:
        """Test URL trailing slash removal."""
        config = ApiConfig(url="http://localhost:8001/")
        assert config.url == "http://localhost:8001"

    def test_timeout_validation(self) -> None:
        """Test timeout value constraints."""
        ApiConfig(timeout=1)  # Minimum
        ApiConfig(timeout=300)  # Maximum

        with pytest.raises(ValueError):
            ApiConfig(timeout=0)

        with pytest.raises(ValueError):
            ApiConfig(timeout=301)

    def test_retries_validation(self) -> None:
        """Test retries value constraints."""
        ApiConfig(retries=0)  # Minimum
        ApiConfig(retries=10)  # Maximum

        with pytest.raises(ValueError):
            ApiConfig(retries=-1)

        with pytest.raises(ValueError):
            ApiConfig(retries=11)


class TestAuthConfig:
    """Tests for authentication configuration model."""

    def test_default_values(self) -> None:
        """Test default auth configuration values."""
        config = AuthConfig()
        assert config.type == "none"
        assert config.token is None
        assert config.api_key is None

    def test_jwt_auth(self) -> None:
        """Test JWT authentication configuration."""
        config = AuthConfig(type="jwt", token="test-token")
        assert config.type == "jwt"
        assert config.token == "test-token"

    def test_api_key_auth(self) -> None:
        """Test API key authentication configuration."""
        config = AuthConfig(type="api_key", api_key="test-key")
        assert config.type == "api_key"
        assert config.api_key == "test-key"


class TestOutputConfig:
    """Tests for output configuration model."""

    def test_default_values(self) -> None:
        """Test default output configuration values."""
        config = OutputConfig()
        assert config.format == "table"
        assert config.color is True
        assert config.timestamps is False
        assert config.verbose is False

    def test_format_options(self) -> None:
        """Test valid output format options."""
        OutputConfig(format="json")
        OutputConfig(format="table")
        OutputConfig(format="tree")

        with pytest.raises(ValueError):
            OutputConfig(format="invalid")  # type: ignore[arg-type]


class TestDefaults:
    """Tests for command defaults configuration."""

    def test_task_defaults(self) -> None:
        """Test task default values."""
        defaults = TaskDefaults()
        assert defaults.priority == "medium"
        assert defaults.timeout == 3600
        assert defaults.requirements == {}

    def test_agent_defaults(self) -> None:
        """Test agent default values."""
        defaults = AgentDefaults()
        assert defaults.cost_per_request == 0.01
        assert defaults.requirements == {}

    def test_workflow_defaults(self) -> None:
        """Test workflow default values."""
        defaults = WorkflowDefaults()
        assert defaults.max_retries == 3
        assert defaults.timeout == 7200


class TestConfig:
    """Tests for complete configuration model."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()
        assert isinstance(config.api, ApiConfig)
        assert isinstance(config.auth, AuthConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.defaults, Defaults)

    def test_load_empty_config(self, tmp_path: Path) -> None:
        """Test loading with no config files."""
        os.chdir(tmp_path)
        config = Config.load(skip_global=True, skip_project=True)
        assert config.api.url == "http://localhost:8001"

    def test_load_global_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading global config file."""
        # Create global config
        global_config_dir = tmp_path / ".agentcore"
        global_config_dir.mkdir()
        global_config_file = global_config_dir / "config.yaml"

        config_data = {
            "api": {
                "url": "https://global.example.com",
                "timeout": 60,
            }
        }
        global_config_file.write_text(yaml.dump(config_data))

        # Mock home directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Load config
        config = Config.load(skip_project=True)
        assert config.api.url == "https://global.example.com"
        assert config.api.timeout == 60
        assert config.api.retries == 3  # Default value

    def test_load_project_config(self, tmp_path: Path) -> None:
        """Test loading project config file."""
        os.chdir(tmp_path)

        # Create project config
        project_config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "url": "https://project.example.com",
            }
        }
        project_config_file.write_text(yaml.dump(config_data))

        # Load config
        config = Config.load(skip_global=True)
        assert config.api.url == "https://project.example.com"

    def test_config_precedence(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test configuration precedence: project > global."""
        # Create global config
        global_config_dir = tmp_path / ".agentcore"
        global_config_dir.mkdir()
        global_config_file = global_config_dir / "config.yaml"

        global_data = {
            "api": {
                "url": "https://global.example.com",
                "timeout": 60,
            }
        }
        global_config_file.write_text(yaml.dump(global_data))

        # Create project config
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        os.chdir(project_dir)

        project_config_file = project_dir / ".agentcore.yaml"
        project_data = {
            "api": {
                "url": "https://project.example.com",
            }
        }
        project_config_file.write_text(yaml.dump(project_data))

        # Mock home directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Load config
        config = Config.load()

        # Project overrides global for url, but timeout comes from global
        assert config.api.url == "https://project.example.com"
        assert config.api.timeout == 60

    def test_env_var_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test environment variable override."""
        os.chdir(tmp_path)

        # Set environment variables
        monkeypatch.setenv("AGENTCORE_API_URL", "https://env.example.com")
        monkeypatch.setenv("AGENTCORE_API_TIMEOUT", "120")
        monkeypatch.setenv("AGENTCORE_VERIFY_SSL", "false")

        # Load config
        config = Config.load(skip_global=True, skip_project=True)

        assert config.api.url == "https://env.example.com"
        assert config.api.timeout == 120
        assert config.api.verify_ssl is False

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable substitution in config values."""
        os.chdir(tmp_path)

        # Set environment variable
        monkeypatch.setenv("MY_TOKEN", "secret-token-123")

        # Create config with variable reference
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "auth": {
                "type": "jwt",
                "token": "${MY_TOKEN}",
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Load config
        config = Config.load(skip_global=True)

        assert config.auth.token == "secret-token-123"

    def test_env_var_substitution_missing(self, tmp_path: Path) -> None:
        """Test environment variable substitution with missing variable."""
        os.chdir(tmp_path)

        # Create config with missing variable reference
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "auth": {
                "type": "jwt",
                "token": "${MISSING_VAR}",
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Load config - should keep original string
        config = Config.load(skip_global=True)
        assert config.auth.token == "${MISSING_VAR}"

    def test_explicit_config_file(self, tmp_path: Path) -> None:
        """Test loading explicit config file."""
        os.chdir(tmp_path)

        # Create custom config file
        custom_config = tmp_path / "custom.yaml"
        config_data = {
            "api": {
                "url": "https://custom.example.com",
            }
        }
        custom_config.write_text(yaml.dump(config_data))

        # Load config
        config = Config.load(config_path=custom_config, skip_global=True, skip_project=True)
        assert config.api.url == "https://custom.example.com"

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error handling for invalid YAML."""
        os.chdir(tmp_path)

        # Create invalid YAML file
        config_file = tmp_path / ".agentcore.yaml"
        config_file.write_text("invalid: yaml: syntax:")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid YAML"):
            Config.load(skip_global=True)

    def test_invalid_config_values(self, tmp_path: Path) -> None:
        """Test error handling for invalid config values."""
        os.chdir(tmp_path)

        # Create config with invalid values
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "timeout": -1,  # Invalid: must be >= 1
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid configuration"):
            Config.load(skip_global=True)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "api" in config_dict
        assert "auth" in config_dict
        assert "output" in config_dict
        assert "defaults" in config_dict

    def test_to_yaml(self) -> None:
        """Test conversion to YAML."""
        config = Config()
        yaml_str = config.to_yaml()

        assert isinstance(yaml_str, str)
        assert "api:" in yaml_str
        assert "url:" in yaml_str

    def test_get_template(self) -> None:
        """Test config template generation."""
        template = Config.get_template()

        assert isinstance(template, str)
        assert "api:" in template
        assert "auth:" in template
        assert "output:" in template
        assert "defaults:" in template
        assert "${AGENTCORE_TOKEN}" in template

    def test_validate_config_warnings(self) -> None:
        """Test configuration validation warnings."""
        # Test SSL warning
        config = Config(
            api=ApiConfig(
                url="https://example.com",
                verify_ssl=False,
            )
        )
        warnings = config.validate_config()
        assert any("SSL verification is disabled" in w for w in warnings)

        # Test hardcoded token warning
        config = Config(
            auth=AuthConfig(
                type="jwt",
                token="hardcoded-token",
            )
        )
        warnings = config.validate_config()
        assert any("hardcoded" in w for w in warnings)

        # Test low timeout warning
        config = Config(
            api=ApiConfig(timeout=3)
        )
        warnings = config.validate_config()
        assert any("timeout is very low" in w for w in warnings)

    def test_validate_config_no_warnings(self) -> None:
        """Test configuration with no warnings."""
        config = Config(
            api=ApiConfig(
                url="https://example.com",
                verify_ssl=True,
            ),
            auth=AuthConfig(
                type="jwt",
                token="${AGENTCORE_TOKEN}",
            ),
        )
        warnings = config.validate_config()
        assert len(warnings) == 0

    def test_deep_merge(self) -> None:
        """Test deep merge of nested dictionaries."""
        base = {
            "api": {
                "url": "http://localhost:8001",
                "timeout": 30,
            },
            "auth": {
                "type": "none",
            },
        }

        override = {
            "api": {
                "timeout": 60,
            },
            "output": {
                "format": "json",
            },
        }

        result = Config._deep_merge(base, override)

        assert result["api"]["url"] == "http://localhost:8001"
        assert result["api"]["timeout"] == 60
        assert result["auth"]["type"] == "none"
        assert result["output"]["format"] == "json"

    def test_set_nested(self) -> None:
        """Test setting nested dictionary values."""
        data: dict[str, Any] = {}

        Config._set_nested(data, ["api", "url"], "http://example.com")
        assert data["api"]["url"] == "http://example.com"

        Config._set_nested(data, ["api", "timeout"], 60)
        assert data["api"]["timeout"] == 60
        assert data["api"]["url"] == "http://example.com"

    def test_convert_env_value_boolean(self) -> None:
        """Test environment value conversion for booleans."""
        assert Config._convert_env_value("true", ["verify_ssl"]) is True
        assert Config._convert_env_value("1", ["color"]) is True
        assert Config._convert_env_value("yes", ["timestamps"]) is True
        assert Config._convert_env_value("false", ["verbose"]) is False
        assert Config._convert_env_value("0", ["verify_ssl"]) is False

    def test_convert_env_value_integer(self) -> None:
        """Test environment value conversion for integers."""
        assert Config._convert_env_value("30", ["timeout"]) == 30
        assert Config._convert_env_value("5", ["retries"]) == 5
        assert Config._convert_env_value("invalid", ["timeout"]) == "invalid"

    def test_convert_env_value_float(self) -> None:
        """Test environment value conversion for floats."""
        assert Config._convert_env_value("0.01", ["cost_per_request"]) == 0.01
        assert Config._convert_env_value("invalid", ["cost_per_request"]) == "invalid"

    def test_load_from_env_all_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading all supported environment variables."""
        monkeypatch.setenv("AGENTCORE_API_URL", "https://env.example.com")
        monkeypatch.setenv("AGENTCORE_API_TIMEOUT", "120")
        monkeypatch.setenv("AGENTCORE_API_RETRIES", "5")
        monkeypatch.setenv("AGENTCORE_VERIFY_SSL", "false")
        monkeypatch.setenv("AGENTCORE_TOKEN", "env-token")
        monkeypatch.setenv("AGENTCORE_API_KEY", "env-key")
        monkeypatch.setenv("AGENTCORE_AUTH_TYPE", "jwt")
        monkeypatch.setenv("AGENTCORE_OUTPUT_FORMAT", "json")
        monkeypatch.setenv("AGENTCORE_COLOR", "false")
        monkeypatch.setenv("AGENTCORE_TIMESTAMPS", "true")
        monkeypatch.setenv("AGENTCORE_VERBOSE", "true")

        env_config = Config._load_from_env()

        assert env_config["api"]["url"] == "https://env.example.com"
        assert env_config["api"]["timeout"] == 120
        assert env_config["api"]["retries"] == 5
        assert env_config["api"]["verify_ssl"] is False
        assert env_config["auth"]["token"] == "env-token"
        assert env_config["auth"]["api_key"] == "env-key"
        assert env_config["auth"]["type"] == "jwt"
        assert env_config["output"]["format"] == "json"
        assert env_config["output"]["color"] is False
        assert env_config["output"]["timestamps"] is True
        assert env_config["output"]["verbose"] is True
