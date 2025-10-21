"""Tests for config commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agentcore_cli.commands.config import app

runner = CliRunner()


class TestConfigInitCommand:
    """Tests for config init command."""

    def test_init_project_config(self, tmp_path: Path) -> None:
        """Test creating project config."""
        # Change to temp directory
        import os
        os.chdir(tmp_path)

        # Run command
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Created project config" in result.stdout

        # Check file was created
        config_file = tmp_path / ".agentcore.yaml"
        assert config_file.exists()

        # Verify content
        content = config_file.read_text()
        assert "api:" in content
        assert "auth:" in content

    def test_init_global_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating global config."""
        # Mock home directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Run command
        result = runner.invoke(app, ["init", "--global"])

        assert result.exit_code == 0
        assert "Created global config" in result.stdout

        # Check file was created
        config_file = tmp_path / ".agentcore" / "config.yaml"
        assert config_file.exists()

    def test_init_existing_config_no_force(self, tmp_path: Path) -> None:
        """Test init with existing config without --force."""
        import os
        os.chdir(tmp_path)

        # Create existing config
        config_file = tmp_path / ".agentcore.yaml"
        config_file.write_text("existing: config")

        # Run command
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 1
        assert "already exists" in result.stdout

        # Original content should be preserved
        assert config_file.read_text() == "existing: config"

    def test_init_existing_config_with_force(self, tmp_path: Path) -> None:
        """Test init with existing config with --force."""
        import os
        os.chdir(tmp_path)

        # Create existing config
        config_file = tmp_path / ".agentcore.yaml"
        config_file.write_text("existing: config")

        # Run command with --force
        result = runner.invoke(app, ["init", "--force"])

        assert result.exit_code == 0
        assert "Created project config" in result.stdout

        # Content should be replaced
        content = config_file.read_text()
        assert "existing: config" not in content
        assert "api:" in content


class TestConfigShowCommand:
    """Tests for config show command."""

    def test_show_default_config(self, tmp_path: Path) -> None:
        """Test showing default configuration."""
        import os
        os.chdir(tmp_path)

        # Run command
        result = runner.invoke(app, ["show"])

        assert result.exit_code == 0
        assert "AgentCore CLI Configuration" in result.stdout

    def test_show_config_json(self, tmp_path: Path) -> None:
        """Test showing config as JSON."""
        import os
        os.chdir(tmp_path)

        # Run command
        result = runner.invoke(app, ["show", "--json"])

        assert result.exit_code == 0

        # Should be valid JSON
        import json
        config_data = json.loads(result.stdout)
        assert "api" in config_data
        assert "auth" in config_data

    def test_show_config_with_sources(self, tmp_path: Path) -> None:
        """Test showing config with sources."""
        import os
        os.chdir(tmp_path)

        # Run command
        result = runner.invoke(app, ["show", "--sources"])

        assert result.exit_code == 0
        assert "Configuration Sources" in result.stdout
        assert "precedence" in result.stdout.lower()

    def test_show_config_with_warnings(self, tmp_path: Path) -> None:
        """Test showing config with validation warnings."""
        import os
        os.chdir(tmp_path)

        # Create config with warnings
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "url": "https://example.com",
                "verify_ssl": False,
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Run command
        result = runner.invoke(app, ["show"])

        assert result.exit_code == 0
        assert "Warnings:" in result.stdout
        assert "SSL verification" in result.stdout


class TestConfigValidateCommand:
    """Tests for config validate command."""

    def test_validate_valid_config(self, tmp_path: Path) -> None:
        """Test validating valid config."""
        import os
        os.chdir(tmp_path)

        # Create valid config
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "url": "http://localhost:8001",
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Run command
        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "Syntax is valid" in result.stdout
        assert "valid" in result.stdout.lower()

    def test_validate_invalid_yaml(self, tmp_path: Path) -> None:
        """Test validating invalid YAML."""
        import os
        os.chdir(tmp_path)

        # Create invalid YAML
        config_file = tmp_path / ".agentcore.yaml"
        config_file.write_text("invalid: yaml: syntax:")

        # Run command
        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 1
        assert "Invalid" in result.stdout

    def test_validate_invalid_values(self, tmp_path: Path) -> None:
        """Test validating invalid config values."""
        import os
        os.chdir(tmp_path)

        # Create config with invalid values
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "timeout": -1,  # Invalid
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Run command
        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 1
        assert "Invalid" in result.stdout

    def test_validate_no_config_files(self, tmp_path: Path) -> None:
        """Test validate with no config files."""
        import os
        os.chdir(tmp_path)

        # Run command
        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "No config files found" in result.stdout

    def test_validate_with_warnings(self, tmp_path: Path) -> None:
        """Test validate with warnings."""
        import os
        os.chdir(tmp_path)

        # Create config with warnings
        config_file = tmp_path / ".agentcore.yaml"
        config_data = {
            "api": {
                "url": "https://example.com",
                "verify_ssl": False,
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Run command
        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "Syntax is valid" in result.stdout
        assert "Warnings:" in result.stdout

    def test_validate_specific_file(self, tmp_path: Path) -> None:
        """Test validating specific config file."""
        import os
        os.chdir(tmp_path)

        # Create custom config
        custom_config = tmp_path / "custom.yaml"
        config_data = {
            "api": {
                "url": "http://localhost:8001",
            }
        }
        custom_config.write_text(yaml.dump(config_data))

        # Run command
        result = runner.invoke(app, ["validate", "--config", str(custom_config)])

        assert result.exit_code == 0
        assert "Syntax is valid" in result.stdout

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test validating nonexistent file."""
        import os
        os.chdir(tmp_path)

        # Run command with nonexistent file
        result = runner.invoke(app, ["validate", "--config", "nonexistent.yaml"])

        assert result.exit_code == 1
        assert "not found" in result.stdout
