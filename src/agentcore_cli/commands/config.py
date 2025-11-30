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
            print(json.dumps(config_dict, indent=2))
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


@app.command()
def validate(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file to validate",
        ),
    ] = None,
) -> None:
    """Validate a configuration file.

    Checks the configuration file for syntax errors and validates
    all settings against expected types and value ranges.

    If no config file is specified, validates the default config file
    at ~/.agentcore/config.toml if it exists.

    Examples:
        # Validate default config
        agentcore config validate

        # Validate specific config file
        agentcore config validate --config ./my-config.toml
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    errors: list[str] = []
    warnings: list[str] = []

    # Determine config file to validate
    if config_path:
        config_file = config_path
    else:
        # Check for default config file
        config_file = DEFAULT_CONFIG_FILE
        if not config_file.exists():
            # Also check current directory
            local_config = Path(".agentcore.toml")
            if local_config.exists():
                config_file = local_config
            else:
                console.print(
                    "[yellow]No configuration file found.[/yellow]\n"
                    f"Expected at: {DEFAULT_CONFIG_FILE}\n"
                    "Run 'agentcore config init' to create one."
                )
                raise typer.Exit(0)

    if not config_file.exists():
        console.print(f"[red]Configuration file not found:[/red] {config_file}")
        raise typer.Exit(1)

    console.print(f"[bold]Validating:[/bold] {config_file}\n")

    # Parse the config file
    try:
        content = config_file.read_text()
        config_data = tomllib.loads(content)
    except Exception as e:
        console.print(f"[red]Failed to parse configuration file:[/red] {str(e)}")
        raise typer.Exit(1)

    # Validate [api] section
    if "api" in config_data:
        api = config_data["api"]

        # Validate URL
        if "url" in api:
            url = api["url"]
            if not isinstance(url, str):
                errors.append("api.url must be a string")
            elif not (url.startswith("http://") or url.startswith("https://")):
                errors.append("api.url must start with http:// or https://")

        # Validate timeout
        if "timeout" in api:
            timeout = api["timeout"]
            if not isinstance(timeout, int):
                errors.append("api.timeout must be an integer")
            elif timeout < 1 or timeout > 300:
                errors.append("api.timeout must be between 1 and 300 seconds")

        # Validate retries
        if "retries" in api:
            retries = api["retries"]
            if not isinstance(retries, int):
                errors.append("api.retries must be an integer")
            elif retries < 0 or retries > 10:
                errors.append("api.retries must be between 0 and 10")

        # Validate verify_ssl
        if "verify_ssl" in api:
            verify_ssl = api["verify_ssl"]
            if not isinstance(verify_ssl, bool):
                errors.append("api.verify_ssl must be a boolean (true/false)")
    else:
        warnings.append("Missing [api] section - defaults will be used")

    # Validate [auth] section
    if "auth" in config_data:
        auth = config_data["auth"]

        # Validate type
        if "type" in auth:
            auth_type = auth["type"]
            valid_types = ["none", "jwt", "api_key"]
            if not isinstance(auth_type, str):
                errors.append("auth.type must be a string")
            elif auth_type not in valid_types:
                errors.append(f"auth.type must be one of: {', '.join(valid_types)}")

        # Validate token
        if "token" in auth:
            token = auth["token"]
            if not isinstance(token, str):
                errors.append("auth.token must be a string")
            elif len(token) < 8:
                warnings.append("auth.token seems too short (less than 8 characters)")
    else:
        warnings.append("Missing [auth] section - defaults will be used")

    # Check for unknown sections
    known_sections = {"api", "auth"}
    for section in config_data:
        if section not in known_sections:
            warnings.append(f"Unknown section [{section}] - will be ignored")

    # Report results
    if errors:
        console.print("[red]✗ Validation failed[/red]\n")
        console.print("[bold]Errors:[/bold]")
        for error in errors:
            console.print(f"  [red]•[/red] {error}")
        if warnings:
            console.print("\n[bold]Warnings:[/bold]")
            for warning in warnings:
                console.print(f"  [yellow]•[/yellow] {warning}")
        raise typer.Exit(1)

    if warnings:
        console.print("[yellow]✓ Validation passed with warnings[/yellow]\n")
        console.print("[bold]Warnings:[/bold]")
        for warning in warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")
    else:
        console.print("[green]✓ Configuration is valid[/green]")

    # Show parsed values
    console.print("\n[bold]Parsed configuration:[/bold]")
    if "api" in config_data:
        api = config_data["api"]
        console.print(f"  api.url: {api.get('url', '(default)')}")
        console.print(f"  api.timeout: {api.get('timeout', '(default)')}")
        console.print(f"  api.retries: {api.get('retries', '(default)')}")
        console.print(f"  api.verify_ssl: {api.get('verify_ssl', '(default)')}")
    if "auth" in config_data:
        auth = config_data["auth"]
        console.print(f"  auth.type: {auth.get('type', '(default)')}")
        console.print(f"  auth.token: {'***' if auth.get('token') else '(not set)'}")
