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


def test_cli_main_keyboard_interrupt() -> None:
    """Test that KeyboardInterrupt is handled gracefully in cli_main()."""
    from unittest.mock import patch

    from agentcore_cli.main import cli_main

    with patch("agentcore_cli.main.app") as mock_app:
        mock_app.side_effect = KeyboardInterrupt()
        # cli_main should catch KeyboardInterrupt and exit with code 130
        import sys
        from unittest.mock import patch as patch_exit

        with patch_exit.object(sys, "exit") as mock_exit:
            cli_main()
            mock_exit.assert_called_once_with(130)


def test_cli_main_generic_exception() -> None:
    """Test that generic exceptions are handled in cli_main()."""
    from unittest.mock import patch

    from agentcore_cli.main import cli_main

    with patch("agentcore_cli.main.app") as mock_app:
        mock_app.side_effect = RuntimeError("Test error")
        # cli_main should catch Exception and exit with code 1
        import sys
        from unittest.mock import patch as patch_exit

        with patch_exit.object(sys, "exit") as mock_exit:
            cli_main()
            mock_exit.assert_called_once_with(1)


def test_cli_main_entry_point() -> None:
    """Test the __main__ entry point executes cli_main."""
    import subprocess
    import sys

    # Execute the main module to test the __name__ == "__main__" path
    result = subprocess.run(
        [sys.executable, "-m", "agentcore_cli.main", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Should execute successfully and show help
    assert result.returncode == 0
    assert "AgentCore CLI" in result.stdout
