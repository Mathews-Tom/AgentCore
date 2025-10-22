"""Configuration management commands.

This module provides CLI commands for configuration management:
- show: Display current configuration
- set: Set a configuration value
- get: Get a specific configuration value
- init: Initialize configuration file

NOTE: Unlike other command groups, config commands interact directly with
the configuration system (container.py) rather than using a service layer,
since configuration is infrastructure-level.
"""

from __future__ import annotations

from typing import Annotated
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentcore_cli.container import get_config, get_config as _get_config_cached

app = typer.Typer(
    name="config",
    help="Manage CLI configuration",
    no_args_is_help=True,
)

console = Console()

# Default config file location
DEFAULT_CONFIG_DIR = Path.home() / ".agentcore"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"


@app.command()
def show(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output in JSON format",
        ),
    ] = False,
) -> None:
    """Display current configuration.

    Shows all configuration values from environment variables and config file.
    Configuration precedence:
    1. Environment variables (AGENTCORE_*)
    2. Configuration file (~/.agentcore/config.toml)
    3. Defaults

    Examples:
        # Show configuration as table
        agentcore config show

        # Show configuration as JSON
        agentcore config show --json
    """
    try:
        # Get current configuration
        config = get_config()

        if json_output:
            # Output as JSON
            config_dict = {
                "api": {
                    "url": config.api.url,
                    "timeout": config.api.timeout,
                    "retries": config.api.retries,
                    "verify_ssl": config.api.verify_ssl,
                },
                "auth": {
                    "type": config.auth.type,
                    "token": "***" if config.auth.token else None,
                },
            }
            console.print(json.dumps(config_dict, indent=2))
        else:
            # Output as table
            console.print("[bold]Current Configuration[/bold]\n")

            # API Configuration
            api_table = Table(title="API Settings", show_header=True)
            api_table.add_column("Setting", style="cyan")
            api_table.add_column("Value", style="green")
            api_table.add_row("URL", config.api.url)
            api_table.add_row("Timeout", f"{config.api.timeout}s")
            api_table.add_row("Retries", str(config.api.retries))
            api_table.add_row("Verify SSL", str(config.api.verify_ssl))
            console.print(api_table)
            console.print()

            # Auth Configuration
            auth_table = Table(title="Authentication Settings", show_header=True)
            auth_table.add_column("Setting", style="cyan")
            auth_table.add_column("Value", style="green")
            auth_table.add_row("Type", config.auth.type)
            auth_table.add_row("Token", "***" if config.auth.token else "Not set")
            console.print(auth_table)

            # Show environment variable hints
            console.print("\n[dim]Set via environment variables:[/dim]")
            console.print("[dim]  AGENTCORE_API_URL, AGENTCORE_API_TIMEOUT,[/dim]")
            console.print("[dim]  AGENTCORE_AUTH_TYPE, AGENTCORE_AUTH_TOKEN[/dim]")

    except Exception as e:
        console.print(f"[red]Error reading configuration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def get(
    key: Annotated[
        str,
        typer.Argument(
            help="Configuration key to retrieve (e.g., 'api.url', 'auth.type')",
        ),
    ],
) -> None:
    """Get a specific configuration value.

    Retrieves a single configuration value by key. Use dot notation
    for nested values (e.g., 'api.url', 'auth.type').

    Examples:
        # Get API URL
        agentcore config get api.url

        # Get authentication type
        agentcore config get auth.type

        # Get timeout value
        agentcore config get api.timeout
    """
    try:
        # Get current configuration
        config = get_config()

        # Parse the key (e.g., "api.url" -> ["api", "url"])
        parts = key.split(".")
        if len(parts) != 2:
            console.print(
                f"[red]Invalid key format:[/red] '{key}'. "
                "Use dot notation (e.g., 'api.url', 'auth.type')"
            )
            raise typer.Exit(2)

        section, setting = parts

        # Get the value
        if section == "api":
            if setting == "url":
                value = config.api.url
            elif setting == "timeout":
                value = config.api.timeout
            elif setting == "retries":
                value = config.api.retries
            elif setting == "verify_ssl":
                value = config.api.verify_ssl
            else:
                console.print(
                    f"[red]Unknown API setting:[/red] '{setting}'. "
                    "Valid options: url, timeout, retries, verify_ssl"
                )
                raise typer.Exit(2)
        elif section == "auth":
            if setting == "type":
                value = config.auth.type
            elif setting == "token":
                value = "***" if config.auth.token else None
            else:
                console.print(
                    f"[red]Unknown auth setting:[/red] '{setting}'. "
                    "Valid options: type, token"
                )
                raise typer.Exit(2)
        else:
            console.print(
                f"[red]Unknown configuration section:[/red] '{section}'. "
                "Valid sections: api, auth"
            )
            raise typer.Exit(2)

        # Print the value
        console.print(f"{value}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error reading configuration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def set(
    key: Annotated[
        str,
        typer.Argument(
            help="Configuration key to set (e.g., 'api.url', 'auth.type')",
        ),
    ],
    value: Annotated[
        str,
        typer.Argument(
            help="Value to set",
        ),
    ],
) -> None:
    """Set a configuration value.

    Sets a configuration value by updating the environment variable.
    This command guides you to set the appropriate environment variable.

    NOTE: This command does NOT persist configuration to a file.
    Use environment variables or create ~/.agentcore/config.toml manually.

    Examples:
        # Set API URL (shows environment variable to set)
        agentcore config set api.url http://localhost:8001

        # Set authentication token (shows environment variable to set)
        agentcore config set auth.token your-token-here
    """
    try:
        # Parse the key
        parts = key.split(".")
        if len(parts) != 2:
            console.print(
                f"[red]Invalid key format:[/red] '{key}'. "
                "Use dot notation (e.g., 'api.url', 'auth.type')"
            )
            raise typer.Exit(2)

        section, setting = parts

        # Map to environment variable
        env_var_map = {
            "api.url": "AGENTCORE_API_URL",
            "api.timeout": "AGENTCORE_API_TIMEOUT",
            "api.retries": "AGENTCORE_API_RETRIES",
            "api.verify_ssl": "AGENTCORE_API_VERIFY_SSL",
            "auth.type": "AGENTCORE_AUTH_TYPE",
            "auth.token": "AGENTCORE_AUTH_TOKEN",
        }

        env_var = env_var_map.get(key)
        if not env_var:
            console.print(
                f"[red]Unknown configuration key:[/red] '{key}'. "
                "Valid keys: api.url, api.timeout, api.retries, api.verify_ssl, "
                "auth.type, auth.token"
            )
            raise typer.Exit(2)

        # Validate value based on setting
        if setting == "timeout":
            try:
                timeout_val = int(value)
                if timeout_val < 1 or timeout_val > 300:
                    console.print(
                        f"[red]Invalid timeout:[/red] {value}. Must be between 1 and 300 seconds."
                    )
                    raise typer.Exit(2)
            except ValueError:
                console.print(f"[red]Invalid timeout:[/red] {value}. Must be an integer.")
                raise typer.Exit(2)
        elif setting == "retries":
            try:
                retries_val = int(value)
                if retries_val < 0 or retries_val > 10:
                    console.print(
                        f"[red]Invalid retries:[/red] {value}. Must be between 0 and 10."
                    )
                    raise typer.Exit(2)
            except ValueError:
                console.print(f"[red]Invalid retries:[/red] {value}. Must be an integer.")
                raise typer.Exit(2)
        elif setting == "verify_ssl":
            if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                console.print(
                    f"[red]Invalid verify_ssl:[/red] {value}. Must be true/false."
                )
                raise typer.Exit(2)
        elif setting == "type":
            if value not in ("none", "jwt", "api_key"):
                console.print(
                    f"[red]Invalid auth type:[/red] {value}. "
                    "Must be one of: none, jwt, api_key"
                )
                raise typer.Exit(2)

        # Show instructions to set environment variable
        console.print(f"[green]✓[/green] To set '{key}' to '{value}', use:\n")
        console.print(f"[bold]export {env_var}='{value}'[/bold]\n")
        console.print("Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
        console.print(f"[dim]echo \"export {env_var}='{value}'\" >> ~/.bashrc[/dim]\n")
        console.print("[yellow]Note:[/yellow] This change will take effect in new shell sessions.")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error setting configuration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def init(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing configuration file",
        ),
    ] = False,
) -> None:
    """Initialize configuration file.

    Creates a default configuration file at ~/.agentcore/config.toml
    with template values.

    NOTE: Currently, configuration is loaded from environment variables.
    This command creates a template file for reference.

    Examples:
        # Create config file
        agentcore config init

        # Overwrite existing config file
        agentcore config init --force
    """
    try:
        # Create config directory if it doesn't exist
        config_dir = DEFAULT_CONFIG_DIR
        config_file = DEFAULT_CONFIG_FILE

        # Check if file exists
        if config_file.exists() and not force:
            console.print(
                f"[yellow]Configuration file already exists:[/yellow] {config_file}\n"
                "Use --force to overwrite."
            )
            raise typer.Exit(0)

        # Create directory
        config_dir.mkdir(parents=True, exist_ok=True)

        # Create template config file
        template_content = """# AgentCore CLI Configuration
#
# Configuration precedence (highest to lowest):
# 1. Environment variables (AGENTCORE_*)
# 2. This configuration file
# 3. Defaults
#
# Note: Currently, configuration is loaded from environment variables only.
# This file serves as a reference template.

[api]
# API server base URL
url = "http://localhost:8001"

# Request timeout in seconds (1-300)
timeout = 30

# Number of retry attempts (0-10)
retries = 3

# Whether to verify SSL certificates
verify_ssl = true

[auth]
# Authentication type: none, jwt, or api_key
type = "none"

# Authentication token (JWT or API key)
# token = "your-token-here"
"""

        # Write config file
        config_file.write_text(template_content)

        console.print(f"[green]✓[/green] Configuration file created: {config_file}\n")
        console.print("[bold]To use environment variables (recommended):[/bold]")
        console.print("[dim]export AGENTCORE_API_URL=http://localhost:8001[/dim]")
        console.print("[dim]export AGENTCORE_AUTH_TOKEN=your-token-here[/dim]\n")
        console.print("[bold]Edit the configuration file:[/bold]")
        console.print(f"[dim]vim {config_file}[/dim]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error creating configuration file:[/red] {str(e)}")
        raise typer.Exit(1)
