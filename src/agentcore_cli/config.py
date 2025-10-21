"""Configuration management for AgentCore CLI.

Implements multi-level configuration with precedence:
1. CLI arguments (highest priority)
2. Environment variables (AGENTCORE_* prefix)
3. Project config (./.agentcore.yaml)
4. Global config (~/.agentcore/config.yaml)
5. Built-in defaults (lowest priority)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, HttpUrl, field_validator


class ApiConfig(BaseModel):
    """API connection configuration."""

    url: str = Field(default="http://localhost:8001")
    timeout: int = Field(default=30, ge=1, le=300)
    retries: int = Field(default=3, ge=0, le=10)
    verify_ssl: bool = True

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate and normalize URL."""
        # Allow both HTTP and HTTPS URLs
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.rstrip("/")


class AuthConfig(BaseModel):
    """Authentication configuration."""

    type: Literal["jwt", "api_key", "none"] = "none"
    token: str | None = None
    api_key: str | None = None


class OutputConfig(BaseModel):
    """Output formatting configuration."""

    format: Literal["json", "table", "tree"] = "table"
    color: bool = True
    timestamps: bool = False
    verbose: bool = False


class TaskDefaults(BaseModel):
    """Default values for task commands."""

    priority: Literal["low", "medium", "high", "critical"] = "medium"
    timeout: int = Field(default=3600, ge=1)
    requirements: dict[str, Any] = Field(default_factory=dict)


class AgentDefaults(BaseModel):
    """Default values for agent commands."""

    cost_per_request: float = Field(default=0.01, ge=0.0)
    requirements: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefaults(BaseModel):
    """Default values for workflow commands."""

    max_retries: int = Field(default=3, ge=0)
    timeout: int = Field(default=7200, ge=1)


class Defaults(BaseModel):
    """Default values for various commands."""

    task: TaskDefaults = Field(default_factory=TaskDefaults)
    agent: AgentDefaults = Field(default_factory=AgentDefaults)
    workflow: WorkflowDefaults = Field(default_factory=WorkflowDefaults)


class Config(BaseModel):
    """Complete CLI configuration."""

    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    defaults: Defaults = Field(default_factory=Defaults)

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        skip_global: bool = False,
        skip_project: bool = False,
    ) -> Config:
        """Load configuration with precedence: CLI > env > project > global > defaults.

        Args:
            config_path: Optional explicit config file path
            skip_global: Skip loading global config
            skip_project: Skip loading project config

        Returns:
            Loaded and merged configuration

        Raises:
            ValueError: If config file is invalid
        """
        config_data: dict[str, Any] = {}

        # 1. Load global config (~/.agentcore/config.yaml)
        if not skip_global:
            global_config_path = Path.home() / ".agentcore" / "config.yaml"
            if global_config_path.exists():
                config_data = cls._load_yaml_file(global_config_path)

        # 2. Load project config (./.agentcore.yaml)
        if not skip_project and not config_path:
            project_config_path = Path.cwd() / ".agentcore.yaml"
            if project_config_path.exists():
                project_data = cls._load_yaml_file(project_config_path)
                config_data = cls._deep_merge(config_data, project_data)

        # 3. Load explicit config file if provided
        if config_path:
            if not config_path.exists():
                raise ValueError(f"Config file not found: {config_path}")
            explicit_data = cls._load_yaml_file(config_path)
            config_data = cls._deep_merge(config_data, explicit_data)

        # 4. Load environment variables (override files)
        env_overrides = cls._load_from_env()
        config_data = cls._deep_merge(config_data, env_overrides)

        # 5. Substitute environment variables in values
        config_data = cls._substitute_env_vars(config_data)

        # 6. Create Config instance with validation
        try:
            return cls(**config_data)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") from e

    @staticmethod
    def _load_yaml_file(path: Path) -> dict[str, Any]:
        """Load and parse YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML data

        Raises:
            ValueError: If file is invalid YAML
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to read {path}: {e}") from e

    @staticmethod
    def _load_from_env() -> dict[str, Any]:
        """Load configuration from environment variables.

        Environment variables are prefixed with AGENTCORE_ and map to config keys:
        - AGENTCORE_API_URL -> api.url
        - AGENTCORE_API_TIMEOUT -> api.timeout
        - AGENTCORE_TOKEN -> auth.token
        - AGENTCORE_OUTPUT_FORMAT -> output.format
        - etc.

        Returns:
            Configuration dictionary from environment variables
        """
        env_mapping = {
            # API configuration
            "AGENTCORE_API_URL": ["api", "url"],
            "AGENTCORE_API_TIMEOUT": ["api", "timeout"],
            "AGENTCORE_API_RETRIES": ["api", "retries"],
            "AGENTCORE_VERIFY_SSL": ["api", "verify_ssl"],
            # Authentication
            "AGENTCORE_TOKEN": ["auth", "token"],
            "AGENTCORE_API_KEY": ["auth", "api_key"],
            "AGENTCORE_AUTH_TYPE": ["auth", "type"],
            # Output
            "AGENTCORE_OUTPUT_FORMAT": ["output", "format"],
            "AGENTCORE_COLOR": ["output", "color"],
            "AGENTCORE_TIMESTAMPS": ["output", "timestamps"],
            "AGENTCORE_VERBOSE": ["output", "verbose"],
        }

        result: dict[str, Any] = {}
        for env_var, path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = Config._convert_env_value(value, path)
                Config._set_nested(result, path, converted_value)

        return result

    @staticmethod
    def _convert_env_value(value: str, path: list[str]) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: Environment variable value
            path: Configuration path (used to infer type)

        Returns:
            Converted value
        """
        # Boolean conversion
        if any(
            key in ["verify_ssl", "color", "timestamps", "verbose"]
            for key in path
        ):
            return value.lower() in ("true", "1", "yes", "on")

        # Integer conversion
        if any(key in ["timeout", "retries", "max_retries"] for key in path):
            try:
                return int(value)
            except ValueError:
                return value

        # Float conversion
        if "cost_per_request" in path:
            try:
                return float(value)
            except ValueError:
                return value

        return value

    @staticmethod
    def _substitute_env_vars(data: dict[str, Any]) -> dict[str, Any]:
        """Substitute environment variable references in config values.

        Supports ${VAR_NAME} syntax in string values.

        Args:
            data: Configuration dictionary

        Returns:
            Dictionary with environment variables substituted
        """
        if isinstance(data, dict):
            return {
                k: Config._substitute_env_vars(v) for k, v in data.items()
            }
        if isinstance(data, list):
            return [Config._substitute_env_vars(item) for item in data]
        if isinstance(data, str):
            # Simple substitution for ${VAR_NAME}
            import re
            def replace_env(match: re.Match[str]) -> str:
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            return re.sub(r'\$\{([A-Z_][A-Z0-9_]*)\}', replace_env, data)
        return data

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
        """Set a value in a nested dictionary.

        Args:
            data: Dictionary to modify
            path: List of keys representing the path
            value: Value to set
        """
        for key in path[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        data[path[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def to_yaml(self) -> str:
        """Convert config to YAML string.

        Returns:
            Configuration as YAML
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def get_template(cls) -> str:
        """Get configuration file template.

        Returns:
            YAML template with comments
        """
        return """# AgentCore CLI Configuration

# API Connection
api:
  url: http://localhost:8001
  timeout: 30
  retries: 3
  verify_ssl: true

# Authentication (optional)
auth:
  type: jwt  # jwt | api_key | none
  token: ${AGENTCORE_TOKEN}
  # api_key: ${AGENTCORE_API_KEY}

# Output Preferences
output:
  format: table  # json | table | tree
  color: true
  timestamps: false
  verbose: false

# Defaults for commands
defaults:
  task:
    priority: medium  # low | medium | high | critical
    timeout: 3600
    requirements: {}
  agent:
    cost_per_request: 0.01
    requirements: {}
  workflow:
    max_retries: 3
    timeout: 7200
"""

    def validate_config(self) -> list[str]:
        """Validate configuration and return any warnings.

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings: list[str] = []

        # Check for insecure settings
        if not self.api.verify_ssl and self.api.url.startswith("https://"):
            warnings.append(
                "SSL verification is disabled for HTTPS URL. "
                "This is insecure and not recommended for production."
            )

        # Check for plaintext secrets
        if self.auth.token and not self.auth.token.startswith("${"):
            warnings.append(
                "Auth token appears to be hardcoded. "
                "Use environment variable reference: ${AGENTCORE_TOKEN}"
            )

        if self.auth.api_key and not self.auth.api_key.startswith("${"):
            warnings.append(
                "API key appears to be hardcoded. "
                "Use environment variable reference: ${AGENTCORE_API_KEY}"
            )

        # Check for reasonable timeout values
        if self.api.timeout < 5:
            warnings.append(
                f"API timeout is very low ({self.api.timeout}s). "
                "This may cause frequent timeouts."
            )

        if self.defaults.task.timeout < 60:
            warnings.append(
                f"Task timeout is very low ({self.defaults.task.timeout}s). "
                "Most tasks may timeout."
            )

        return warnings
