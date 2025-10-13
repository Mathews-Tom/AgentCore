"""Tests for CLI main entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from agentcore_cli import __version__
from agentcore_cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    """Test that --help flag works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AgentCore CLI" in result.stdout
    assert "command-line interface" in result.stdout.lower()


def test_cli_version() -> None:
    """Test that --version flag works."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
    assert "AgentCore CLI version" in result.stdout


def test_cli_version_short() -> None:
    """Test that -v flag works for version."""
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_no_args() -> None:
    """Test that CLI shows help when no arguments provided."""
    result = runner.invoke(app, [])
    # Typer exits with code 2 when no_args_is_help=True
    assert result.exit_code in (0, 2)
    assert "Usage:" in result.stdout or "AgentCore CLI" in result.stdout
