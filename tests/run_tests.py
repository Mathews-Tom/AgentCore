#!/usr/bin/env python3
"""
Comprehensive Test Runner with Detailed Metrics

Runs all tests in the background and provides detailed, readable reports
on test status, coverage, and performance metrics.

Usage:
    uv run python tests/run_tests.py [options]

Options:
    --components <name>  Run specific component(s) (comma-separated)
    --parallel           Run test components in parallel
    --no-coverage        Skip coverage reporting
    --verbose            Show verbose output
    --json <file>        Output results as JSON to file

Component Mapping:
    Test Directory          Source Component        Ticket Prefix
    ---------------         ----------------        -------------
    a2a_protocol            a2a_protocol            A2A
    agent_runtime           agent_runtime           ART
    reasoning               reasoning               BCR
    training                training                FLOW
    orchestration           orchestration           ORCH
    integration             integration             INT
    cli                     (not yet implemented)   CLI
    gateway                 (not yet implemented)   GATE
    config                  (infrastructure tests)  N/A
    load                    (load tests)            N/A

Examples:
    # Run all tests
    uv run python tests/run_tests.py

    # Run specific components
    uv run python tests/run_tests.py --components a2a_protocol,training

    # Run in parallel without coverage
    uv run python tests/run_tests.py --parallel --no-coverage

    # Save results to JSON
    uv run python tests/run_tests.py --json test-results.json
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ComponentResult:
    """Test results for a single component."""

    name: str
    passed: int
    failed: int
    skipped: int
    errors: int
    warnings: int
    duration: float
    coverage: float | None = None
    coverage_lines: str | None = None
    status: str = "unknown"

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100


@dataclass
class TestReport:
    """Complete test run report."""

    start_time: datetime
    end_time: datetime
    total_duration: float
    components: list[ComponentResult]
    overall_coverage: float | None = None

    @property
    def total_tests(self) -> int:
        return sum(c.total for c in self.components)

    @property
    def total_passed(self) -> int:
        return sum(c.passed for c in self.components)

    @property
    def total_failed(self) -> int:
        return sum(c.failed for c in self.components)

    @property
    def total_skipped(self) -> int:
        return sum(c.skipped for c in self.components)

    @property
    def overall_success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.total_passed / self.total_tests) * 100


class TestRunner:
    """Orchestrates test execution and reporting."""

    # Mapping between test directories and source components
    # Keys are actual test directory names, values are source component paths
    # None means no source component (infrastructure/not implemented)
    COMPONENT_MAPPING = {
        "a2a_protocol": "a2a_protocol",  # A2A - Agent-to-Agent Protocol
        "agent_runtime": "agent_runtime",  # ART - Agent Runtime Layer
        "reasoning": "reasoning",  # BCR - Bounded Context Reasoning
        "training": "training",  # FLOW - Training System
        "orchestration": "orchestration",  # ORCH - Orchestration Engine
        "integration": "integration",  # INT - Integration Tests
        "cli": None,  # CLI - Command Line Interface (not yet implemented)
        "gateway": None,  # GATE - API Gateway (not yet implemented)
        "config": None,  # Config tests (infrastructure, no source component)
        "load": None,  # Load tests (infrastructure, no source component)
    }

    # Human-readable display names for components
    DISPLAY_NAMES = {
        "a2a_protocol": "A2A Protocol",
        "agent_runtime": "Agent Runtime",
        "reasoning": "Context Reasoning",
        "training": "Training System",
        "orchestration": "Orchestration Engine",
        "integration": "Integration Tests",
        "cli": "CLI",
        "gateway": "API Gateway",
        "config": "Configuration Tests",
        "load": "Load Tests",
    }

    @classmethod
    def get_display_name(cls, component: str) -> str:
        """Get human-readable display name for a component."""
        return cls.DISPLAY_NAMES.get(component, component)

    def __init__(
        self,
        components: list[str] | None = None,
        parallel: bool = False,
        coverage: bool = True,
        verbose: bool = False,
    ):
        self.components = components
        self.parallel = parallel
        self.coverage = coverage
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"

    def discover_components(self) -> list[str]:
        """Discover all test components."""
        if self.components:
            return self.components

        # Find all directories in tests/ that contain test files
        components = []
        for item in self.test_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                # Check if directory contains test files
                test_files = list(item.rglob("test_*.py"))
                if test_files:
                    components.append(item.name)

        return sorted(components)

    def run_component_tests(self, component: str) -> ComponentResult:
        """Run tests for a single component."""
        start_time = time.time()
        component_path = self.test_dir / component

        # Build pytest command
        cmd = [
            "uv",
            "run",
            "pytest",
            str(component_path),
            "-v",
            "--tb=short",
            "-q",  # Quiet mode for cleaner output
        ]

        if self.coverage:
            # Measure full codebase coverage for all test runs to capture
            # cross-component coverage (e.g., integration tests exercising a2a_protocol)
            cmd.extend(
                [
                    "--cov=src/agentcore/a2a_protocol",
                    "--cov=src/agentcore/agent_runtime",
                    "--cov=src/agentcore/reasoning",
                    "--cov=src/agentcore/training",
                    "--cov=src/agentcore/orchestration",
                    "--cov=src/agentcore/integration",
                    "--cov-report=term-missing",
                    "--cov-report=json:.coverage.json",
                    "--cov-fail-under=0",  # Don't fail on coverage threshold for per-component runs
                ]
            )

        if self.verbose:
            cmd.append("-vv")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per component
            )

            duration = time.time() - start_time

            # Parse output
            parsed = self._parse_pytest_output(result.stdout + result.stderr)

            # Extract coverage if available
            coverage_pct = None
            coverage_lines = None
            if self.coverage:
                coverage_pct, coverage_lines = self._extract_coverage(result.stdout)

            status = "passed" if result.returncode == 0 else "failed"

            return ComponentResult(
                name=self.get_display_name(component),
                passed=parsed["passed"],
                failed=parsed["failed"],
                skipped=parsed["skipped"],
                errors=parsed["errors"],
                warnings=parsed["warnings"],
                duration=duration,
                coverage=coverage_pct,
                coverage_lines=coverage_lines,
                status=status,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ComponentResult(
                name=self.get_display_name(component),
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                warnings=0,
                duration=duration,
                status="timeout",
            )
        except Exception as e:
            duration = time.time() - start_time
            return ComponentResult(
                name=self.get_display_name(component),
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                warnings=0,
                duration=duration,
                status=f"error: {str(e)}",
            )

    def _parse_pytest_output(self, output: str) -> dict[str, int]:
        """Parse pytest output for test counts."""
        result = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "warnings": 0}

        # Look for summary line like: "10 passed, 2 warnings in 2.5s"
        for line in output.split("\n"):
            if "passed" in line or "failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if i > 0:
                        prev = parts[i - 1]
                        if prev.isdigit():
                            count = int(prev)
                            if "passed" in part:
                                result["passed"] = count
                            elif "failed" in part:
                                result["failed"] = count
                            elif "skipped" in part:
                                result["skipped"] = count
                            elif "error" in part:
                                result["errors"] = count
                            elif "warning" in part:
                                result["warnings"] = count

        return result

    def _extract_coverage(self, output: str) -> tuple[float | None, str | None]:
        """Extract coverage percentage from pytest output."""
        coverage_pct = None
        coverage_lines = None

        for line in output.split("\n"):
            # Look for TOTAL line with coverage percentage
            if line.startswith("TOTAL"):
                parts = line.split()
                for i, part in enumerate(parts):
                    if "%" in part:
                        try:
                            coverage_pct = float(part.rstrip("%"))
                            if i > 0:
                                # Try to get lines info (Stmts, Miss)
                                coverage_lines = f"{parts[i - 2]}/{parts[i - 1]}"
                        except (ValueError, IndexError):
                            pass
                break

        return coverage_pct, coverage_lines

    async def run_all_async(self) -> TestReport:
        """Run all tests asynchronously."""
        start_time = datetime.now(UTC)
        components = self.discover_components()

        print(f"\n{'=' * 80}")
        print(f"🧪 AgentCore Test Suite")
        print(f"{'=' * 80}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Components: {len(components)}")
        print(f"Mode: {'Parallel' if self.parallel else 'Sequential'}")
        print(f"Coverage: {'Enabled' if self.coverage else 'Disabled'}")
        print(f"{'=' * 80}")

        # Show component mapping
        print(f"\n📋 Component Coverage Map:")
        print(f"{'-' * 80}")
        with_source = []
        without_source = []
        for comp in sorted(components):
            display_name = self.get_display_name(comp)
            src_comp = self.COMPONENT_MAPPING.get(comp, comp)
            if src_comp:
                with_source.append(f"  ✅ {display_name:<30} → src/agentcore/{src_comp}")
            else:
                without_source.append(f"  ⚪ {display_name:<30} (no source component)")

        for line in with_source + without_source:
            print(line)
        print(f"{'-' * 80}\n")

        results = []

        if self.parallel:
            # Run components in parallel
            tasks = [
                asyncio.to_thread(self.run_component_tests, comp) for comp in components
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Run sequentially with progress
            for i, component in enumerate(components, 1):
                display_name = self.get_display_name(component)
                print(
                    f"[{i}/{len(components)}] Running {display_name}...",
                    end=" ",
                    flush=True,
                )
                result = self.run_component_tests(component)
                results.append(result)

                # Quick status
                status_icon = "✅" if result.status == "passed" else "❌"
                print(f"{status_icon} ({result.duration:.1f}s)")

        end_time = datetime.now(UTC)
        total_duration = (end_time - start_time).total_seconds()

        # Calculate overall coverage if available
        overall_coverage = None
        if self.coverage and results:
            total_coverage = sum(r.coverage for r in results if r.coverage is not None)
            count = sum(1 for r in results if r.coverage is not None)
            if count > 0:
                overall_coverage = total_coverage / count

        return TestReport(
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            components=results,
            overall_coverage=overall_coverage,
        )

    def run_all(self) -> TestReport:
        """Run all tests (synchronous wrapper)."""
        return asyncio.run(self.run_all_async())


class ReportFormatter:
    """Formats test reports for display."""

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format percentage with color coding."""
        if value >= 90:
            return f"🟢 {value:.1f}%"
        elif value >= 70:
            return f"🟡 {value:.1f}%"
        else:
            return f"🔴 {value:.1f}%"

    @staticmethod
    def print_detailed_report(report: TestReport):
        """Print detailed human-readable report."""
        print(f"\n{'=' * 80}")
        print(f"📊 Test Results Summary")
        print(f"{'=' * 80}\n")

        # Component Results Table
        print(
            f"{'Component':<25} {'Tests':>8} {'Pass':>8} {'Fail':>8} {'Skip':>8} {'Time':>10} {'Coverage':>10}"
        )
        print(
            f"{'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}"
        )

        for comp in sorted(report.components, key=lambda x: x.name):
            status_icon = "✅" if comp.status == "passed" else "❌"
            coverage_str = f"{comp.coverage:.1f}%" if comp.coverage else "N/A"

            print(
                f"{status_icon} {comp.name:<22} "
                f"{comp.total:>8} "
                f"{comp.passed:>8} "
                f"{comp.failed:>8} "
                f"{comp.skipped:>8} "
                f"{ReportFormatter.format_duration(comp.duration):>10} "
                f"{coverage_str:>10}"
            )

        print(
            f"{'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}"
        )

        # Totals
        coverage_str = (
            f"{report.overall_coverage:.1f}%" if report.overall_coverage else "N/A"
        )
        print(
            f"{'TOTAL':<25} "
            f"{report.total_tests:>8} "
            f"{report.total_passed:>8} "
            f"{report.total_failed:>8} "
            f"{report.total_skipped:>8} "
            f"{ReportFormatter.format_duration(report.total_duration):>10} "
            f"{coverage_str:>10}"
        )

        # Overall Statistics
        print(f"\n{'=' * 80}")
        print(f"📈 Overall Statistics")
        print(f"{'=' * 80}\n")

        print(
            f"Total Duration:    {ReportFormatter.format_duration(report.total_duration)}"
        )
        print(
            f"Success Rate:      {ReportFormatter.format_percentage(report.overall_success_rate)}"
        )
        if report.overall_coverage:
            print(
                f"Average Coverage:  {ReportFormatter.format_percentage(report.overall_coverage)}"
            )

        # Performance Metrics
        print(f"\n{'=' * 80}")
        print(f"⚡ Performance Metrics")
        print(f"{'=' * 80}\n")

        fastest = min(report.components, key=lambda x: x.duration)
        slowest = max(report.components, key=lambda x: x.duration)

        print(
            f"Fastest Component: {fastest.name} ({ReportFormatter.format_duration(fastest.duration)})"
        )
        print(
            f"Slowest Component: {slowest.name} ({ReportFormatter.format_duration(slowest.duration)})"
        )

        avg_duration = (
            report.total_duration / len(report.components) if report.components else 0
        )
        print(f"Average Duration:  {ReportFormatter.format_duration(avg_duration)}")

        # Coverage Breakdown (if available)
        if any(c.coverage for c in report.components):
            print(f"\n{'=' * 80}")
            print(f"📋 Coverage Breakdown")
            print(f"{'=' * 80}\n")

            coverage_components = [
                (c.name, c.coverage)
                for c in report.components
                if c.coverage is not None
            ]
            coverage_components.sort(key=lambda x: x[1], reverse=True)

            for name, coverage in coverage_components:
                bar_length = int(coverage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"{name:<25} {bar} {coverage:>6.1f}%")

        # Final Status
        print(f"\n{'=' * 80}")
        if report.total_failed == 0:
            print(f"✅ All tests passed!")
        else:
            print(f"❌ {report.total_failed} test(s) failed")
        print(f"{'=' * 80}\n")

    @staticmethod
    def save_json_report(report: TestReport, filepath: Path):
        """Save report as JSON."""
        data = {
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat(),
            "total_duration": report.total_duration,
            "total_tests": report.total_tests,
            "total_passed": report.total_passed,
            "total_failed": report.total_failed,
            "total_skipped": report.total_skipped,
            "overall_success_rate": report.overall_success_rate,
            "overall_coverage": report.overall_coverage,
            "components": [asdict(c) for c in report.components],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n📄 JSON report saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner with detailed metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--components",
        type=str,
        help='Comma-separated list of components to test (e.g., "a2a_protocol,agent_runtime")',
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run test components in parallel"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage reporting"
    )
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument(
        "--json", type=str, metavar="FILE", help="Save results as JSON to FILE"
    )

    args = parser.parse_args()

    # Parse components
    components = None
    if args.components:
        components = [c.strip() for c in args.components.split(",")]

    # Create runner
    runner = TestRunner(
        components=components,
        parallel=args.parallel,
        coverage=not args.no_coverage,
        verbose=args.verbose,
    )

    # Run tests
    try:
        report = runner.run_all()

        # Print report
        ReportFormatter.print_detailed_report(report)

        # Save JSON if requested
        if args.json:
            ReportFormatter.save_json_report(report, Path(args.json))

        # Exit with appropriate code
        sys.exit(0 if report.total_failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
