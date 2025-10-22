"""Integration tests for config commands.

These tests verify that the config commands properly interact with the
configuration system and display/manage configuration values.
"""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.container import Config, ApiConfig, AuthConfig


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration."""
    config = Config()
    config.api = ApiConfig(
        url="http://localhost:8001",
        timeout=30,
        retries=3,
        verify_ssl=True,
    )
    config.auth = AuthConfig(
        type="jwt",
        token="test-token-12345",
    )
    return config


class TestConfigShowCommand:
    """Test suite for config show command."""

    def test_show_success_table_format(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test successful config show with table format."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Current Configuration" in result.output
        assert "API Settings" in result.output
        # Check for "Authentication" (the title may have extra formatting/spacing)
        assert "Authentication" in result.output
        assert "http://localhost:8001" in result.output
        assert "30s" in result.output
        assert "3" in result.output
        assert "True" in result.output
        assert "jwt" in result.output
        assert "***" in result.output  # Token should be masked

    def test_show_success_json_format(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test successful config show with JSON format."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "show", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert '"api"' in result.output
        assert '"auth"' in result.output
        assert '"url": "http://localhost:8001"' in result.output
        assert '"timeout": 30' in result.output
        assert '"retries": 3' in result.output
        assert '"verify_ssl": true' in result.output
        assert '"type": "jwt"' in result.output
        assert '"token": "***"' in result.output  # Token should be masked

    def test_show_no_token(self, runner: CliRunner, mock_config: Config) -> None:
        """Test config show when no token is set."""
        mock_config.auth.token = None

        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "show", "--json"])

        assert result.exit_code == 0
        assert '"token": null' in result.output

    def test_show_error(self, runner: CliRunner) -> None:
        """Test config show with error."""
        with patch(
            "agentcore_cli.commands.config.get_config",
            side_effect=Exception("Config error"),
        ):
            result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 1
        assert "Error reading configuration" in result.output
        assert "Config error" in result.output


class TestConfigGetCommand:
    """Test suite for config get command."""

    def test_get_api_url(self, runner: CliRunner, mock_config: Config) -> None:
        """Test get API URL."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "api.url"])

        assert result.exit_code == 0
        assert "http://localhost:8001" in result.output

    def test_get_api_timeout(self, runner: CliRunner, mock_config: Config) -> None:
        """Test get API timeout."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "api.timeout"])

        assert result.exit_code == 0
        assert "30" in result.output

    def test_get_api_retries(self, runner: CliRunner, mock_config: Config) -> None:
        """Test get API retries."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "api.retries"])

        assert result.exit_code == 0
        assert "3" in result.output

    def test_get_api_verify_ssl(self, runner: CliRunner, mock_config: Config) -> None:
        """Test get API verify_ssl."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "api.verify_ssl"])

        assert result.exit_code == 0
        assert "True" in result.output

    def test_get_auth_type(self, runner: CliRunner, mock_config: Config) -> None:
        """Test get auth type."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "auth.type"])

        assert result.exit_code == 0
        assert "jwt" in result.output

    def test_get_auth_token(self, runner: CliRunner, mock_config: Config) -> None:
        """Test get auth token (should be masked)."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "auth.token"])

        assert result.exit_code == 0
        assert "***" in result.output

    def test_get_auth_token_not_set(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test get auth token when not set."""
        mock_config.auth.token = None

        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "auth.token"])

        assert result.exit_code == 0
        assert "None" in result.output

    def test_get_invalid_key_format(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test get with invalid key format."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "invalid"])

        assert result.exit_code == 2
        assert "Invalid key format" in result.output
        assert "Use dot notation" in result.output

    def test_get_invalid_section(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test get with invalid section."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "invalid.setting"])

        assert result.exit_code == 2
        assert "Unknown configuration section" in result.output

    def test_get_invalid_api_setting(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test get with invalid API setting."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "api.invalid"])

        assert result.exit_code == 2
        assert "Unknown API setting" in result.output

    def test_get_invalid_auth_setting(
        self, runner: CliRunner, mock_config: Config
    ) -> None:
        """Test get with invalid auth setting."""
        with patch("agentcore_cli.commands.config.get_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "get", "auth.invalid"])

        assert result.exit_code == 2
        assert "Unknown auth setting" in result.output

    def test_get_error(self, runner: CliRunner) -> None:
        """Test get with error."""
        with patch(
            "agentcore_cli.commands.config.get_config",
            side_effect=Exception("Config error"),
        ):
            result = runner.invoke(app, ["config", "get", "api.url"])

        assert result.exit_code == 1
        assert "Error reading configuration" in result.output


class TestConfigSetCommand:
    """Test suite for config set command."""

    def test_set_api_url(self, runner: CliRunner) -> None:
        """Test set API URL."""
        result = runner.invoke(
            app, ["config", "set", "api.url", "http://prod.example.com"]
        )

        assert result.exit_code == 0
        assert "export AGENTCORE_API_URL='http://prod.example.com'" in result.output
        assert "AGENTCORE_API_URL" in result.output

    def test_set_api_timeout_valid(self, runner: CliRunner) -> None:
        """Test set API timeout with valid value."""
        result = runner.invoke(app, ["config", "set", "api.timeout", "60"])

        assert result.exit_code == 0
        assert "export AGENTCORE_API_TIMEOUT='60'" in result.output

    def test_set_api_timeout_invalid_range(self, runner: CliRunner) -> None:
        """Test set API timeout with invalid range."""
        result = runner.invoke(app, ["config", "set", "api.timeout", "500"])

        assert result.exit_code == 2
        assert "Invalid timeout" in result.output
        assert "Must be between 1 and 300" in result.output

    def test_set_api_timeout_invalid_type(self, runner: CliRunner) -> None:
        """Test set API timeout with invalid type."""
        result = runner.invoke(app, ["config", "set", "api.timeout", "invalid"])

        assert result.exit_code == 2
        assert "Invalid timeout" in result.output
        assert "Must be an integer" in result.output

    def test_set_api_retries_valid(self, runner: CliRunner) -> None:
        """Test set API retries with valid value."""
        result = runner.invoke(app, ["config", "set", "api.retries", "5"])

        assert result.exit_code == 0
        assert "export AGENTCORE_API_RETRIES='5'" in result.output

    def test_set_api_retries_invalid_range(self, runner: CliRunner) -> None:
        """Test set API retries with invalid range."""
        result = runner.invoke(app, ["config", "set", "api.retries", "20"])

        assert result.exit_code == 2
        assert "Invalid retries" in result.output
        assert "Must be between 0 and 10" in result.output

    def test_set_api_retries_invalid_type(self, runner: CliRunner) -> None:
        """Test set API retries with invalid type."""
        result = runner.invoke(app, ["config", "set", "api.retries", "invalid"])

        assert result.exit_code == 2
        assert "Invalid retries" in result.output
        assert "Must be an integer" in result.output

    def test_set_api_verify_ssl_true(self, runner: CliRunner) -> None:
        """Test set API verify_ssl to true."""
        result = runner.invoke(app, ["config", "set", "api.verify_ssl", "true"])

        assert result.exit_code == 0
        assert "export AGENTCORE_API_VERIFY_SSL='true'" in result.output

    def test_set_api_verify_ssl_false(self, runner: CliRunner) -> None:
        """Test set API verify_ssl to false."""
        result = runner.invoke(app, ["config", "set", "api.verify_ssl", "false"])

        assert result.exit_code == 0
        assert "export AGENTCORE_API_VERIFY_SSL='false'" in result.output

    def test_set_api_verify_ssl_invalid(self, runner: CliRunner) -> None:
        """Test set API verify_ssl with invalid value."""
        result = runner.invoke(app, ["config", "set", "api.verify_ssl", "invalid"])

        assert result.exit_code == 2
        assert "Invalid verify_ssl" in result.output
        assert "Must be true/false" in result.output

    def test_set_auth_type_jwt(self, runner: CliRunner) -> None:
        """Test set auth type to jwt."""
        result = runner.invoke(app, ["config", "set", "auth.type", "jwt"])

        assert result.exit_code == 0
        assert "export AGENTCORE_AUTH_TYPE='jwt'" in result.output

    def test_set_auth_type_api_key(self, runner: CliRunner) -> None:
        """Test set auth type to api_key."""
        result = runner.invoke(app, ["config", "set", "auth.type", "api_key"])

        assert result.exit_code == 0
        assert "export AGENTCORE_AUTH_TYPE='api_key'" in result.output

    def test_set_auth_type_none(self, runner: CliRunner) -> None:
        """Test set auth type to none."""
        result = runner.invoke(app, ["config", "set", "auth.type", "none"])

        assert result.exit_code == 0
        assert "export AGENTCORE_AUTH_TYPE='none'" in result.output

    def test_set_auth_type_invalid(self, runner: CliRunner) -> None:
        """Test set auth type with invalid value."""
        result = runner.invoke(app, ["config", "set", "auth.type", "invalid"])

        assert result.exit_code == 2
        assert "Invalid auth type" in result.output
        assert "Must be one of: none, jwt, api_key" in result.output

    def test_set_auth_token(self, runner: CliRunner) -> None:
        """Test set auth token."""
        result = runner.invoke(
            app, ["config", "set", "auth.token", "my-secret-token"]
        )

        assert result.exit_code == 0
        assert "export AGENTCORE_AUTH_TOKEN='my-secret-token'" in result.output

    def test_set_invalid_key_format(self, runner: CliRunner) -> None:
        """Test set with invalid key format."""
        result = runner.invoke(app, ["config", "set", "invalid", "value"])

        assert result.exit_code == 2
        assert "Invalid key format" in result.output

    def test_set_invalid_key(self, runner: CliRunner) -> None:
        """Test set with invalid key."""
        result = runner.invoke(app, ["config", "set", "invalid.key", "value"])

        assert result.exit_code == 2
        assert "Unknown configuration key" in result.output


class TestConfigInitCommand:
    """Test suite for config init command."""

    def test_init_success(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test successful config init."""
        config_file = tmp_path / "config.toml"

        with patch(
            "agentcore_cli.commands.config.DEFAULT_CONFIG_FILE", config_file
        ), patch("agentcore_cli.commands.config.DEFAULT_CONFIG_DIR", tmp_path):
            result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        assert "Configuration file created" in result.output
        assert config_file.exists()

        # Verify file contents
        content = config_file.read_text()
        assert "[api]" in content
        assert "[auth]" in content
        assert "url = " in content
        assert "timeout = " in content

    def test_init_file_exists_no_force(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test config init when file exists without force."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("existing content")

        with patch(
            "agentcore_cli.commands.config.DEFAULT_CONFIG_FILE", config_file
        ), patch("agentcore_cli.commands.config.DEFAULT_CONFIG_DIR", tmp_path):
            result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        assert "already exists" in result.output
        assert "Use --force to overwrite" in result.output

        # Verify file was not overwritten
        content = config_file.read_text()
        assert content == "existing content"

    def test_init_file_exists_with_force(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test config init when file exists with force."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("existing content")

        with patch(
            "agentcore_cli.commands.config.DEFAULT_CONFIG_FILE", config_file
        ), patch("agentcore_cli.commands.config.DEFAULT_CONFIG_DIR", tmp_path):
            result = runner.invoke(app, ["config", "init", "--force"])

        assert result.exit_code == 0
        assert "Configuration file created" in result.output

        # Verify file was overwritten
        content = config_file.read_text()
        assert "[api]" in content
        assert "existing content" not in content

    def test_init_creates_directory(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test config init creates directory if it doesn't exist."""
        config_dir = tmp_path / "subdir" / "config"
        config_file = config_dir / "config.toml"

        with patch(
            "agentcore_cli.commands.config.DEFAULT_CONFIG_FILE", config_file
        ), patch("agentcore_cli.commands.config.DEFAULT_CONFIG_DIR", config_dir):
            result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        assert config_dir.exists()
        assert config_file.exists()

    def test_init_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test config init with error."""
        config_file = tmp_path / "config.toml"

        with patch(
            "agentcore_cli.commands.config.DEFAULT_CONFIG_FILE", config_file
        ), patch("agentcore_cli.commands.config.DEFAULT_CONFIG_DIR", tmp_path), patch(
            "pathlib.Path.mkdir", side_effect=Exception("Permission denied")
        ):
            result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 1
        assert "Error creating configuration file" in result.output
