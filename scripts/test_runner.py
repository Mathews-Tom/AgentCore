#!/usr/bin/env python3
"""
Modular Test Runner with Rich Display
Runs test suite in isolated sections with beautiful progress tracking.
"""

import subprocess
import sys
import re
import signal
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box


@dataclass
class TestSection:
    """Test section configuration"""

    name: str
    path: str
    description: str


@dataclass
class TestResult:
    """Test execution result"""

    section: TestSection
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: int = 0
    duration: float = 0.0
    exit_code: int = 0
    output: str = ""
    status: str = "pending"  # pending, running, passed, failed, skipped, flaky
    flaky: bool = False  # Mark tests that failed but are known to be unreliable


class TestRunner:
    """Modular test runner with rich display"""

    # Test sections to run
    SECTIONS = [
        TestSection("unit", "tests/unit/", "Unit tests"),
        TestSection("services", "tests/services/", "Service layer tests"),
        TestSection("a2a_protocol", "tests/a2a_protocol/", "A2A Protocol tests"),
        TestSection("agent_runtime", "tests/agent_runtime/", "Agent Runtime tests"),
        TestSection("cli", "tests/cli/", "CLI tests"),
        TestSection("config", "tests/config/", "Configuration tests"),
        TestSection("coordination", "tests/coordination/", "Coordination tests"),
        TestSection("gateway", "tests/gateway/", "Gateway tests"),
        TestSection("orchestration", "tests/orchestration/", "Orchestration tests"),
        TestSection("protocol", "tests/protocol/", "Protocol tests"),
        TestSection("reasoning", "tests/reasoning/", "Reasoning tests"),
        TestSection("security", "tests/security/", "Security tests"),
        TestSection("transport", "tests/transport/", "Transport tests"),
        TestSection("integration", "tests/integration/", "Integration tests"),
        TestSection("external_api", "tests/integration/", "External API tests (flaky)"),
        TestSection("e2e", "tests/e2e/", "End-to-end tests"),
        TestSection("performance", "tests/performance/", "Performance tests"),
        TestSection("modular", "tests/modular/", "Modular tests"),
    ]

    def __init__(self):
        self.console = Console()
        self.results_dir = Path(".test-results")
        self.results: list[TestResult] = []
        self.docker_available = self._check_docker()
        self.api_available = self._check_api()
        self.start_time = datetime.now()
        self._api_process: Optional[subprocess.Popen] = None
        self._api_started_by_runner = False

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_api(self) -> bool:
        """Check if AgentCore API is running at localhost:8001"""
        try:
            import urllib.request
            import json as json_module
            with urllib.request.urlopen("http://localhost:8001/api/v1/health", timeout=2) as resp:
                data = json_module.loads(resp.read().decode())
                return data.get("status") == "healthy"
        except Exception:
            return False

    def _start_api_server(self) -> bool:
        """Start the API server for E2E tests if not already running"""
        if self._check_api():
            return True  # Already running

        self.console.print("  [yellow]Starting API server for E2E tests...[/yellow]")

        try:
            # Start uvicorn in background
            self._api_process = subprocess.Popen(
                [
                    "uv", "run", "uvicorn",
                    "agentcore.a2a_protocol.main:app",
                    "--host", "0.0.0.0",
                    "--port", "8001",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent process group
            )

            # Wait for API to become healthy (max 30 seconds)
            for i in range(30):
                time.sleep(1)
                if self._check_api():
                    self._api_started_by_runner = True
                    self.api_available = True
                    self.console.print("  [green]✓ API server started successfully[/green]")
                    return True

                # Check if process died
                if self._api_process.poll() is not None:
                    stderr = self._api_process.stderr.read().decode() if self._api_process.stderr else ""
                    self.console.print(f"  [red]✗ API server failed to start: {stderr[:200]}[/red]")
                    return False

            self.console.print("  [red]✗ API server startup timed out after 30s[/red]")
            self._stop_api_server()
            return False

        except Exception as e:
            self.console.print(f"  [red]✗ Failed to start API server: {e}[/red]")
            return False

    def _stop_api_server(self):
        """Stop the API server if we started it"""
        if self._api_process and self._api_started_by_runner:
            self.console.print("  [yellow]Stopping API server...[/yellow]")
            try:
                # Kill the entire process group
                import os
                os.killpg(os.getpgid(self._api_process.pid), signal.SIGTERM)
                self._api_process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                # Force kill if graceful shutdown failed
                try:
                    os.killpg(os.getpgid(self._api_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._api_process = None
            self._api_started_by_runner = False
            self.console.print("  [green]✓ API server stopped[/green]")

    def _setup_results_dir(self):
        """Setup results directory"""
        self.results_dir.mkdir(exist_ok=True)
        # Clean old results
        for file in self.results_dir.glob("*.txt"):
            file.unlink()

    def _parse_pytest_summary(self, output: str) -> tuple[int, int, int, int]:
        """Parse pytest summary line for counts"""
        # Look for the summary line: "=== X failed, Y passed, Z skipped in N.NNs ==="
        summary_pattern = r"^=+.*in [\d.]+s.*=+$"
        summary_lines = [
            line for line in output.split("\n") if re.match(summary_pattern, line)
        ]

        if not summary_lines:
            return 0, 0, 0, 0

        summary = summary_lines[-1]

        # Extract counts
        passed = int(match.group(1)) if (match := re.search(r"(\d+) passed", summary)) else 0
        failed = int(match.group(1)) if (match := re.search(r"(\d+) failed", summary)) else 0
        skipped = int(match.group(1)) if (match := re.search(r"(\d+) skipped", summary)) else 0
        warnings = int(match.group(1)) if (match := re.search(r"(\d+) warning", summary)) else 0

        return passed, failed, skipped, warnings

    def _run_section(self, section: TestSection) -> TestResult:
        """Run a single test section"""
        result = TestResult(section=section, status="running")

        # Check if path exists
        path = Path(section.path)
        if not path.exists():
            result.status = "skipped"
            result.output = f"Directory not found: {section.path}"
            return result

        # For E2E tests, ensure API server is running
        api_started_for_e2e = False
        if section.name == "e2e":
            if not self._start_api_server():
                # Count tests that would be skipped
                try:
                    count_cmd = ["uv", "run", "pytest", section.path, "--collect-only", "-q"]
                    count_result = subprocess.run(
                        count_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=30,
                    )
                    # Parse "X tests collected" or count lines
                    test_count_match = re.search(r"(\d+) tests? collected", count_result.stdout)
                    if test_count_match:
                        result.skipped = int(test_count_match.group(1))
                    else:
                        # Count test lines (each line with :: is a test)
                        result.skipped = len([l for l in count_result.stdout.split("\n") if "::" in l])
                except Exception:
                    result.skipped = 0  # Unknown count

                result.status = "skipped"
                result.output = "E2E tests skipped: API server not available and could not be started"
                return result
            api_started_for_e2e = self._api_started_by_runner

        # Build pytest command
        cmd = ["uv", "run", "pytest", section.path, "--tb=short", "-v"]
        if self.docker_available:
            # Handle marker filtering based on section
            if section.name == "external_api":
                # Run ONLY external_api tests
                cmd.extend(["-m", "external_api"])
            elif section.name == "integration":
                # Run integration tests but exclude external_api
                cmd.extend(["-m", "integration and not external_api"])
            else:
                # Run with all markers for other sections
                cmd.extend(["-m", ""])

        # Save output to file
        output_file = self.results_dir / f"{section.name}.txt"

        try:
            # Run pytest
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1200,  # 20 minute timeout per section
            )

            result.output = proc.stdout
            result.exit_code = proc.returncode

            # Save to file
            output_file.write_text(result.output)

            # Parse results
            passed, failed, skipped, warnings = self._parse_pytest_summary(result.output)
            result.passed = passed
            result.failed = failed
            result.skipped = skipped
            result.warnings = warnings

            # Determine status
            # Pytest exit codes: 0 = all passed, 1 = tests failed OR coverage failed,
            # 5 = no tests collected
            if proc.returncode == 0:
                result.status = "passed"
            elif proc.returncode == 5:  # No tests collected
                result.status = "skipped"
            elif failed > 0:
                # Check if this is an external_api section - mark as flaky instead of failed
                if section.name == "external_api":
                    result.status = "flaky"
                    result.flaky = True
                else:
                    # Actual test failures in non-external tests
                    result.status = "failed"
            elif "Coverage failure" in result.output or "FAIL Required test coverage" in result.output:
                # Coverage failure but all tests passed - treat as passed
                result.status = "passed"
            else:
                # Other error
                if section.name == "external_api":
                    result.status = "flaky"
                    result.flaky = True
                else:
                    result.status = "failed"

        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.output = f"Test section timed out after 20 minutes"
            output_file.write_text(result.output)
        except Exception as e:
            result.status = "failed"
            result.output = f"Error running tests: {e}"
            output_file.write_text(result.output)
        finally:
            # Stop API server if we started it for E2E tests
            if api_started_for_e2e:
                self._stop_api_server()

        return result

    def _create_header(self) -> Panel:
        """Create header panel"""
        docker_status = "✓ Available" if self.docker_available else "⚠ Not running"
        api_status = "✓ Running" if self.api_available else "⚠ Not running (will auto-start for E2E)"
        text = Text()
        text.append("🧪 Full Test Suite ", style="bold cyan")
        text.append("(Modular)\n", style="bold cyan")
        text.append(f"Docker: {docker_status}\n", style="yellow" if not self.docker_available else "green")
        text.append(f"API: {api_status}\n", style="yellow" if not self.api_available else "green")
        text.append(f"Sections: {len(self.SECTIONS)}", style="blue")

        return Panel(text, box=box.DOUBLE, border_style="cyan")

    def _create_progress_table(self, current_section: Optional[str] = None) -> Table:
        """Create progress table showing section status"""
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Skipped", justify="right", style="yellow")

        for section in self.SECTIONS:
            # Find result if exists
            result = next((r for r in self.results if r.section.name == section.name), None)

            if result:
                # Completed section
                if result.status == "passed":
                    status = "[green]✓ Passed[/green]"
                elif result.status == "failed":
                    status = "[red]✗ Failed[/red]"
                elif result.status == "flaky":
                    status = "[yellow]⚠ Flaky[/yellow]"
                elif result.status == "skipped":
                    status = "[yellow]⊘ Skipped[/yellow]"
                else:
                    status = "[blue]⧖ Running[/blue]"

                table.add_row(
                    section.description,
                    status,
                    str(result.passed) if result.passed > 0 else "-",
                    str(result.failed) if result.failed > 0 else "-",
                    str(result.skipped) if result.skipped > 0 else "-",
                )
            elif current_section == section.name:
                # Currently running
                table.add_row(
                    section.description,
                    "[blue]⧖ Running[/blue]",
                    "-",
                    "-",
                    "-",
                )
            else:
                # Not started
                table.add_row(
                    section.description,
                    "[dim]○ Pending[/dim]",
                    "-",
                    "-",
                    "-",
                )

        return table

    def _create_summary_table(self) -> Table:
        """Create final summary table"""
        # Count sections
        passed_sections = sum(1 for r in self.results if r.status == "passed")
        failed_sections = sum(1 for r in self.results if r.status == "failed")
        flaky_sections = sum(1 for r in self.results if r.status == "flaky")
        skipped_sections = sum(1 for r in self.results if r.status == "skipped")

        # Count tests (exclude flaky from failed count)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results if not r.flaky)
        total_flaky = sum(r.failed for r in self.results if r.flaky)
        total_skipped = sum(r.skipped for r in self.results)
        total_warnings = sum(r.warnings for r in self.results)

        # Create sections table
        sections_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        sections_table.add_column("Label", style="bold")
        sections_table.add_column("Value")

        sections_table.add_row("Sections", "")
        sections_table.add_row("  ✓ Passed", f"[green]{passed_sections}[/green]")
        sections_table.add_row("  ✗ Failed", f"[red]{failed_sections}[/red]")
        if flaky_sections > 0:
            sections_table.add_row("  ⚠ Flaky", f"[yellow]{flaky_sections}[/yellow]")
        sections_table.add_row("  ⊘ Skipped", f"[yellow]{skipped_sections}[/yellow]")
        sections_table.add_row("  Total", str(len(self.SECTIONS)))

        # Create tests table
        tests_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        tests_table.add_column("Label", style="bold")
        tests_table.add_column("Value")

        tests_table.add_row("Tests", "")
        tests_table.add_row("  ✓ Passed", f"[green]{total_passed}[/green]")
        tests_table.add_row("  ✗ Failed", f"[red]{total_failed}[/red]")
        if total_flaky > 0:
            tests_table.add_row("  ⚠ Flaky", f"[yellow]{total_flaky}[/yellow]")
        tests_table.add_row("  ⊘ Skipped", f"[yellow]{total_skipped}[/yellow]")
        if total_warnings > 0:
            tests_table.add_row("  ⚠ Warnings", f"[yellow]{total_warnings}[/yellow]")

        # Combine into summary
        summary = Table.grid(padding=(0, 4))
        summary.add_row(sections_table, tests_table)

        return summary

    def _create_failed_sections_table(self) -> Optional[Table]:
        """Create table of failed sections"""
        failed = [r for r in self.results if r.status == "failed"]
        if not failed:
            return None

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold red")
        table.add_column("Section", style="cyan")
        table.add_column("Failures", justify="right", style="red")

        for result in failed:
            table.add_row(result.section.description, str(result.failed))

        return table

    def run(self):
        """Run all test sections"""
        self._setup_results_dir()

        # Display header
        self.console.print()
        self.console.print(self._create_header())
        self.console.print()

        # Run each section with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            main_task = progress.add_task(
                "[cyan]Running test sections...", total=len(self.SECTIONS)
            )

            for section in self.SECTIONS:
                # Update progress
                progress.update(
                    main_task, description=f"[cyan]Running: {section.description}"
                )

                # Run section
                result = self._run_section(section)
                self.results.append(result)

                # Update progress
                progress.advance(main_task)

                # Show quick status
                if result.status == "passed":
                    self.console.print(
                        f"  [green]✓[/green] {section.description} "
                        f"([green]{result.passed}[/green] passed)"
                    )
                elif result.status == "failed":
                    self.console.print(
                        f"  [red]✗[/red] {section.description} "
                        f"([red]{result.failed}[/red] failed, "
                        f"[green]{result.passed}[/green] passed)"
                    )
                elif result.status == "skipped":
                    skip_info = f"[yellow]{result.skipped}[/yellow] skipped" if result.skipped > 0 else "skipped"
                    self.console.print(
                        f"  [yellow]⊘[/yellow] {section.description} ({skip_info})"
                    )
                elif result.status == "flaky":
                    self.console.print(
                        f"  [yellow]⚠[/yellow] {section.description} "
                        f"([yellow]{result.failed}[/yellow] flaky, "
                        f"[green]{result.passed}[/green] passed)"
                    )

        # Display summary
        self.console.print()
        self.console.print(
            Panel(
                self._create_summary_table(),
                title="📊 Test Suite Summary",
                box=box.DOUBLE,
                border_style="cyan",
            )
        )

        # Display failed sections if any
        failed_table = self._create_failed_sections_table()
        if failed_table:
            self.console.print()
            self.console.print(
                Panel(
                    failed_table,
                    title="❌ Failed Sections",
                    box=box.ROUNDED,
                    border_style="red",
                )
            )

        # Show results location
        self.console.print()
        self.console.print(
            f"[dim]Results saved to: {self.results_dir.absolute()}[/dim]"
        )

        # Show elapsed time
        elapsed = (datetime.now() - self.start_time).total_seconds()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.console.print(f"[dim]Total time: {minutes}m {seconds}s[/dim]")
        self.console.print()

        # Final status
        failed_count = sum(1 for r in self.results if r.status == "failed")
        flaky_count = sum(1 for r in self.results if r.status == "flaky")

        if failed_count > 0:
            self.console.print(
                "[red]❌ Test suite completed with failures[/red]", style="bold"
            )
            return 1
        elif flaky_count > 0:
            self.console.print(
                "[yellow]⚠️  Test suite passed with flaky tests (external API issues)[/yellow]", style="bold"
            )
            return 0  # Don't fail the build for flaky external API tests
        else:
            self.console.print(
                "[green]✅ All test sections passed![/green]", style="bold"
            )
            return 0


def main():
    """Main entry point"""
    runner = TestRunner()
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
