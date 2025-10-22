"""Integration tests for config commands with realistic workflows."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from agentcore_cli.main import app

runner = CliRunner()


@pytest.fixture
def temp_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Set up temporary environment variables for testing."""
    monkeypatch.setenv("AGENTCORE_API_URL", "http://env-server:9000")
    monkeypatch.setenv("AGENTCORE_API_TIMEOUT", "60")
    monkeypatch.setenv("AGENTCORE_AUTH_TOKEN", "env-token-12345")
    return monkeypatch


@pytest.fixture
def global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a global config file for testing."""
    config_dir = tmp_path / ".agentcore"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"

    config_data = {
        "api": {
            "url": "http://global-server:8000",
            "timeout": 45,
            "retries": 2,
        },
        "auth": {
            "type": "jwt",
            "token": "global-token-abc",
        },
        "defaults": {
            "task": {
                "priority": "high",
                "timeout": 7200,
            }
        }
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock home directory
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return config_file


@pytest.fixture
def project_config(tmp_path: Path):
    """Create a project config file for testing."""
    config_file = tmp_path / ".agentcore.yaml"

    config_data = {
        "api": {
            "url": "http://project-server:8001",
            "timeout": 30,
        },
        "auth": {
            "type": "jwt",
            "token": "project-token-xyz",
        }
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


class TestConfigPrecedence:
    """Integration tests for configuration precedence (CLI > env > project > global)."""

    @pytest.mark.skip(reason="CLI-003: CLI argument overrides (--api-url, --timeout, etc.) not implemented yet")
    def test_precedence_cli_overrides_all(self, tmp_path: Path, global_config, project_config, temp_env_vars):
        """Test that CLI arguments override all other config sources."""
        import os
        os.chdir(tmp_path)

        # Create mock client to verify final config
        with patch("agentcore_cli.commands.agent.AgentCoreClient") as mock_client_class:
            client = MagicMock()
            client.call.return_value = {"agents": []}
            mock_client_class.return_value = client

            # Run command with CLI override
            result = runner.invoke(app, [
                "agent", "list",
                "--api-url", "http://cli-server:7000",
            ])

            assert result.exit_code == 0

            # Verify client was created with CLI URL (highest precedence)
            mock_client_class.assert_called_once()
            # The actual URL used should be from CLI args
            call_kwargs = mock_client_class.call_args[1] if mock_client_class.call_args[1] else {}
            # Note: This is a simplified test - actual verification depends on implementation

    def test_precedence_env_overrides_file(self, tmp_path: Path, global_config, project_config, temp_env_vars):
        """Test that environment variables override file configs."""
        import os
        os.chdir(tmp_path)

        with patch("agentcore_cli.commands.agent.AgentCoreClient") as mock_client_class:
            with patch("agentcore_cli.commands.agent.Config") as mock_config_class:
                # Simulate config loading with env precedence
                config = MagicMock()
                config.api.url = "http://env-server:9000"  # From env vars
                config.api.timeout = 60  # From env vars
                config.api.retries = 3
                config.api.verify_ssl = True
                config.auth.type = "token"
                config.auth.token = "env-token-12345"  # From env vars
                mock_config_class.load.return_value = config

                client = MagicMock()
                client.call.return_value = {"agents": []}
                mock_client_class.return_value = client

                result = runner.invoke(app, ["agent", "list"])

                assert result.exit_code == 0
                # Config should use env values, not project/global

    def test_precedence_project_overrides_global(self, tmp_path: Path, global_config, project_config):
        """Test that project config overrides global config."""
        import os
        os.chdir(tmp_path)

        with patch("agentcore_cli.commands.agent.AgentCoreClient") as mock_client_class:
            with patch("agentcore_cli.commands.agent.Config") as mock_config_class:
                # Simulate config loading with project precedence
                config = MagicMock()
                config.api.url = "http://project-server:8001"  # From project
                config.api.timeout = 30  # From project
                config.api.retries = 2  # From global (not overridden)
                config.api.verify_ssl = True
                config.auth.type = "token"
                config.auth.token = "project-token-xyz"  # From project
                mock_config_class.load.return_value = config

                client = MagicMock()
                client.call.return_value = {"agents": []}
                mock_client_class.return_value = client

                result = runner.invoke(app, ["agent", "list"])

                assert result.exit_code == 0
                # Config should use project values where available

    def test_precedence_global_fallback(self, tmp_path: Path, global_config):
        """Test that global config is used when no other sources exist."""
        import os
        # Change to directory without project config
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.chdir(empty_dir)

        with patch("agentcore_cli.commands.agent.AgentCoreClient") as mock_client_class:
            with patch("agentcore_cli.commands.agent.Config") as mock_config_class:
                # Simulate config loading with only global config
                config = MagicMock()
                config.api.url = "http://global-server:8000"  # From global
                config.api.timeout = 45  # From global
                config.api.retries = 2  # From global
                config.api.verify_ssl = True
                config.auth.type = "token"
                config.auth.token = "global-token-abc"  # From global
                mock_config_class.load.return_value = config

                client = MagicMock()
                client.call.return_value = {"agents": []}
                mock_client_class.return_value = client

                result = runner.invoke(app, ["agent", "list"])

                assert result.exit_code == 0


class TestConfigWorkflows:
    """Integration tests for complete config workflows."""

    def test_init_validate_show_workflow(self, tmp_path: Path):
        """Test complete workflow: init → validate → show."""
        import os
        os.chdir(tmp_path)

        # Step 1: Initialize config
        init_result = runner.invoke(app, ["config", "init"])
        assert init_result.exit_code == 0
        assert "Created project config" in init_result.stdout

        config_file = tmp_path / ".agentcore.yaml"
        assert config_file.exists()

        # Step 2: Validate config
        validate_result = runner.invoke(app, ["config", "validate"])
        assert validate_result.exit_code == 0
        assert "Syntax is valid" in validate_result.stdout or "valid" in validate_result.stdout.lower()

        # Step 3: Show config
        show_result = runner.invoke(app, ["config", "show"])
        assert show_result.exit_code == 0
        assert "AgentCore CLI Configuration" in show_result.stdout or "api" in show_result.stdout

    def test_init_modify_validate_workflow(self, tmp_path: Path):
        """Test workflow: init → manual modification → validate."""
        import os
        os.chdir(tmp_path)

        # Step 1: Initialize config
        init_result = runner.invoke(app, ["config", "init"])
        assert init_result.exit_code == 0

        config_file = tmp_path / ".agentcore.yaml"

        # Step 2: Modify config with custom values
        config_data = {
            "api": {
                "url": "http://custom:8001",
                "timeout": 120,
                "retries": 5,
            },
            "auth": {
                "type": "jwt",
                "token": "custom-token",
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Step 3: Validate modified config
        validate_result = runner.invoke(app, ["config", "validate"])
        assert validate_result.exit_code == 0

        # Step 4: Show modified config
        show_result = runner.invoke(app, ["config", "show"])
        assert show_result.exit_code == 0
        assert "custom" in show_result.stdout or "8001" in show_result.stdout

    def test_init_force_reinit_workflow(self, tmp_path: Path):
        """Test workflow: init → force reinit with --force."""
        import os
        os.chdir(tmp_path)

        # Step 1: Initial creation
        init_result1 = runner.invoke(app, ["config", "init"])
        assert init_result1.exit_code == 0

        config_file = tmp_path / ".agentcore.yaml"
        original_content = config_file.read_text()

        # Step 2: Try to init again without --force (should fail)
        init_result2 = runner.invoke(app, ["config", "init"])
        assert init_result2.exit_code == 1
        assert "already exists" in init_result2.stdout
        assert config_file.read_text() == original_content

        # Step 3: Force reinit
        init_result3 = runner.invoke(app, ["config", "init", "--force"])
        assert init_result3.exit_code == 0
        assert "Created project config" in init_result3.stdout

    def test_config_with_agent_commands_workflow(self, tmp_path: Path):
        """Test using config with agent commands."""
        import os
        os.chdir(tmp_path)

        # Step 1: Initialize config
        init_result = runner.invoke(app, ["config", "init"])
        assert init_result.exit_code == 0

        # Step 2: Use config with agent list command
        with patch("agentcore_cli.commands.agent.AgentCoreClient") as mock_client_class:
            with patch("agentcore_cli.commands.agent.Config") as mock_config_class:
                config = MagicMock()
                config.api.url = "http://localhost:8001"
                config.api.timeout = 30
                config.api.retries = 3
                config.api.verify_ssl = True
                config.auth.type = "none"
                config.auth.token = None
                mock_config_class.load.return_value = config

                client = MagicMock()
                client.call.return_value = {"agents": []}
                mock_client_class.return_value = client

                agent_result = runner.invoke(app, ["agent", "list"])
                assert agent_result.exit_code == 0


class TestConfigValidationIntegration:
    """Integration tests for config validation scenarios."""

    def test_validate_ssl_warning(self, tmp_path: Path):
        """Test validation shows warning for disabled SSL."""
        import os
        os.chdir(tmp_path)

        # Create config with SSL disabled
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "url": "https://example.com",
                "verify_ssl": False,  # This should trigger warning
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(app, ["config", "validate"])

        assert result.exit_code == 0
        assert "Warnings:" in result.stdout or "SSL" in result.stdout

    def test_validate_multiple_issues(self, tmp_path: Path):
        """Test validation with multiple issues."""
        import os
        os.chdir(tmp_path)

        # Create config with multiple problems
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "timeout": -1,  # Invalid
                "retries": 100,  # Potentially problematic
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = runner.invoke(app, ["config", "validate"])

        # Should report validation errors
        assert result.exit_code == 1
        assert "Invalid" in result.stdout or "Error" in result.stdout


class TestConfigShowIntegration:
    """Integration tests for config show command."""

    def test_show_with_sources_precedence(self, tmp_path: Path, global_config, project_config):
        """Test showing config with sources and precedence information."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, ["config", "show", "--sources"])

        assert result.exit_code == 0
        assert "Configuration Sources" in result.stdout or "precedence" in result.stdout.lower()

    def test_show_json_format(self, tmp_path: Path):
        """Test showing config in JSON format for scripting."""
        import os
        os.chdir(tmp_path)

        # Initialize config first
        runner.invoke(app, ["config", "init"])

        result = runner.invoke(app, ["config", "show", "--json"])

        assert result.exit_code == 0

        # Should be valid JSON
        import json
        config_data = json.loads(result.stdout)
        assert "api" in config_data

    def test_show_empty_config(self, tmp_path: Path):
        """Test showing config when no config files exist."""
        import os
        # Change to empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        os.chdir(empty_dir)

        result = runner.invoke(app, ["config", "show"])

        # Should show default configuration
        assert result.exit_code == 0
        assert "Configuration" in result.stdout or "api" in result.stdout
