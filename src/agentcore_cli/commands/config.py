"""Configuration management commands for AgentCore CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agentcore_cli.config import Config

app = typer.Typer(help="Manage CLI configuration", no_args_is_help=True)
console = Console()


@app.command()
def init(
    global_config: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Initialize global config (~/.agentcore/config.yaml)",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing config file",
        ),
    ] = False,
) -> None:
    """Initialize configuration file with default template.

    By default creates project config (.agentcore.yaml) in current directory.
    Use --global to create global config in home directory.

    Examples:
        agentcore config init
        agentcore config init --global
        agentcore config init --force
    """
    if global_config:
        config_path = Path.home() / ".agentcore" / "config.yaml"
        config_type = "global"
    else:
        config_path = Path.cwd() / ".agentcore.yaml"
        config_type = "project"

    # Check if file already exists
    if config_path.exists() and not force:
        console.print(
            f"[yellow]Config file already exists:[/yellow] {config_path}",
            highlight=False,
        )
        console.print(
            "\nUse --force to overwrite, or edit the file manually.",
            highlight=False,
        )
        raise typer.Exit(1)

    # Create parent directory if needed (for global config)
    if global_config:
        config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write template
    template = Config.get_template()
    config_path.write_text(template)

    console.print(
        f"[green]✓[/green] Created {config_type} config: {config_path}",
        highlight=False,
    )
    console.print(
        "\nNext steps:",
        highlight=False,
    )
    console.print(
        f"  1. Edit {config_path} with your settings",
        highlight=False,
    )
    console.print(
        "  2. Validate config: agentcore config validate",
        highlight=False,
    )
    console.print(
        "  3. View merged config: agentcore config show",
        highlight=False,
    )


@app.command()
def show(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to config file",
        ),
    ] = None,
    show_defaults: Annotated[
        bool,
        typer.Option(
            "--defaults",
            help="Show default values",
        ),
    ] = False,
    show_sources: Annotated[
        bool,
        typer.Option(
            "--sources",
            help="Show configuration sources",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
) -> None:
    """Display current configuration with merged values.

    Shows the effective configuration after merging all sources:
    - CLI arguments (highest priority)
    - Environment variables
    - Project config (.agentcore.yaml)
    - Global config (~/.agentcore/config.yaml)
    - Built-in defaults (lowest priority)

    Examples:
        agentcore config show
        agentcore config show --sources
        agentcore config show --json
    """
    try:
        # Load configuration
        config = Config.load(config_path=config_path)

        if json_output:
            # Output as JSON
            import json
            console.print(json.dumps(config.to_dict(), indent=2))
        else:
            # Pretty-print configuration
            console.print(
                Panel(
                    "[bold]AgentCore CLI Configuration[/bold]",
                    style="blue",
                )
            )

            if show_sources:
                _show_config_sources()

            # Display configuration as YAML with syntax highlighting
            yaml_str = config.to_yaml()
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
            console.print(syntax)

            # Show warnings if any
            warnings = config.validate_config()
            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}", highlight=False)

    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}", highlight=False)
        raise typer.Exit(1)


@app.command()
def validate(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to config file to validate",
        ),
    ] = None,
) -> None:
    """Validate configuration file syntax and values.

    Checks:
    - YAML syntax is valid
    - Required fields are present
    - Field types are correct
    - Values are within valid ranges
    - Security best practices (no hardcoded secrets, etc.)

    Examples:
        agentcore config validate
        agentcore config validate --config ~/.agentcore/config.yaml
    """
    # Determine which file to validate
    if config_path:
        if not config_path.exists():
            console.print(
                f"[red]Error:[/red] Config file not found: {config_path}",
                highlight=False,
            )
            raise typer.Exit(1)
        files_to_check = [config_path]
    else:
        # Check both global and project configs
        files_to_check = []
        global_config = Path.home() / ".agentcore" / "config.yaml"
        project_config = Path.cwd() / ".agentcore.yaml"

        if global_config.exists():
            files_to_check.append(global_config)
        if project_config.exists():
            files_to_check.append(project_config)

        if not files_to_check:
            console.print(
                "[yellow]No config files found.[/yellow]",
                highlight=False,
            )
            console.print(
                "\nRun 'agentcore config init' to create a config file.",
                highlight=False,
            )
            raise typer.Exit(0)

    all_valid = True
    for file_path in files_to_check:
        console.print(f"\nValidating: {file_path}", highlight=False)

        try:
            # Try to load the config
            Config.load(config_path=file_path, skip_global=True, skip_project=True)

            console.print("[green]✓[/green] Syntax is valid", highlight=False)

            # Load again to get validation warnings
            config = Config.load(config_path=file_path, skip_global=True, skip_project=True)
            warnings = config.validate_config()

            if warnings:
                console.print("[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}", highlight=False)
            else:
                console.print("[green]✓[/green] No warnings", highlight=False)

        except Exception as e:
            console.print(f"[red]✗ Invalid:[/red] {e}", highlight=False)
            all_valid = False

    if all_valid:
        console.print("\n[green]All configuration files are valid![/green]")
    else:
        console.print("\n[red]Some configuration files have errors.[/red]")
        raise typer.Exit(1)


def _show_config_sources() -> None:
    """Display configuration sources and precedence."""
    table = Table(title="Configuration Sources (in precedence order)", show_header=True)
    table.add_column("Priority", style="cyan", width=8)
    table.add_column("Source", style="magenta")
    table.add_column("Location", style="white")
    table.add_column("Status", style="green")

    sources = [
        ("1", "CLI arguments", "Command-line flags", "Active when provided"),
        ("2", "Environment vars", "AGENTCORE_* variables", "Active when set"),
        (
            "3",
            "Project config",
            str(Path.cwd() / ".agentcore.yaml"),
            "✓ Found" if (Path.cwd() / ".agentcore.yaml").exists() else "Not found",
        ),
        (
            "4",
            "Global config",
            str(Path.home() / ".agentcore/config.yaml"),
            "✓ Found" if (Path.home() / ".agentcore/config.yaml").exists() else "Not found",
        ),
        ("5", "Built-in defaults", "Hardcoded fallbacks", "Always active"),
    ]

    for priority, source, location, status in sources:
        table.add_row(priority, source, location, status)

    console.print(table)
    console.print()
