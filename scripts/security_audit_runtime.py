#!/usr/bin/env python3
"""
Comprehensive security audit script for AgentCore Runtime.

This script performs security checks on:
- Docker container security configuration
- Seccomp profiles
- AppArmor profiles
- Capability restrictions
- File system permissions
- Network isolation
- Resource limits

Run with: uv run python scripts/security_audit_runtime.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class SecurityAuditor:
    """Perform comprehensive security audit of AgentCore Runtime."""

    def __init__(self) -> None:
        """Initialize security auditor."""
        self.findings: list[dict[str, Any]] = []
        self.passed_checks = 0
        self.failed_checks = 0
        self.warning_checks = 0

    def audit(self) -> bool:
        """
        Run complete security audit.

        Returns:
            True if all critical checks pass, False otherwise
        """
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║   AgentCore Runtime Security Audit                            ║")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        # Run all audit checks
        self._check_docker_security()
        self._check_apparmor()
        self._check_seccomp()
        self._check_capabilities()
        self._check_file_permissions()
        self._check_security_profiles_directory()
        self._check_container_defaults()

        # Print summary
        self._print_summary()

        # Return True if no critical failures
        return self.failed_checks == 0

    def _check_docker_security(self) -> None:
        """Audit Docker daemon security configuration."""
        print("\n[1/7] Docker Security Configuration")
        print("─" * 65)

        # Check if Docker is running
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                self._add_finding(
                    "FAIL",
                    "Docker",
                    "Docker daemon is not running or not accessible",
                    "critical",
                )
                return

            docker_info = json.loads(result.stdout)

            # Check user namespace remapping
            userns_remap = docker_info.get("SecurityOptions", [])
            has_userns = any("userns" in opt for opt in userns_remap)
            if not has_userns:
                self._add_finding(
                    "WARN",
                    "Docker",
                    "User namespace remapping not enabled",
                    "warning",
                    "Consider enabling with: dockerd --userns-remap=default",
                )

            # Check if running in swarm mode (less secure for agent isolation)
            if docker_info.get("Swarm", {}).get("LocalNodeState") == "active":
                self._add_finding(
                    "WARN",
                    "Docker",
                    "Swarm mode is active - may reduce container isolation",
                    "warning",
                )

            # Check live restore (should be enabled for production)
            if not docker_info.get("LiveRestoreEnabled", False):
                self._add_finding(
                    "WARN",
                    "Docker",
                    "Live restore not enabled",
                    "info",
                    "Enable with: dockerd --live-restore",
                )

            self._add_finding(
                "PASS",
                "Docker",
                "Docker daemon is accessible and configured",
                "info",
            )

        except FileNotFoundError:
            self._add_finding(
                "FAIL",
                "Docker",
                "Docker is not installed",
                "critical",
            )
        except subprocess.TimeoutExpired:
            self._add_finding(
                "FAIL",
                "Docker",
                "Docker command timed out - daemon may be unresponsive",
                "critical",
            )
        except Exception as e:
            self._add_finding(
                "FAIL",
                "Docker",
                f"Failed to check Docker configuration: {e}",
                "critical",
            )

    def _check_apparmor(self) -> None:
        """Audit AppArmor configuration."""
        print("\n[2/7] AppArmor Security Module")
        print("─" * 65)

        # Check if AppArmor is enabled
        apparmor_status_path = Path("/sys/module/apparmor/parameters/enabled")
        if not apparmor_status_path.exists():
            self._add_finding(
                "WARN",
                "AppArmor",
                "AppArmor kernel module not found - MAC not available",
                "warning",
                "AppArmor is Linux-specific. On macOS/Windows, use alternative security",
            )
            return

        try:
            with open(apparmor_status_path) as f:
                enabled = f.read().strip() == "Y"

            if not enabled:
                self._add_finding(
                    "FAIL",
                    "AppArmor",
                    "AppArmor is installed but not enabled",
                    "critical",
                    "Enable with: sudo systemctl enable apparmor && sudo systemctl start apparmor",
                )
                return

            # Check for loaded profiles
            result = subprocess.run(
                ["sudo", "apparmor_status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout
                if "profiles are loaded" in output:
                    # Extract number of loaded profiles
                    line = [l for l in output.split("\n") if "profiles are loaded" in l][0]
                    num_profiles = int(line.split()[0])
                    self._add_finding(
                        "PASS",
                        "AppArmor",
                        f"AppArmor is enabled with {num_profiles} profiles loaded",
                        "info",
                    )
                else:
                    self._add_finding(
                        "WARN",
                        "AppArmor",
                        "AppArmor is enabled but no profiles loaded",
                        "warning",
                    )
            else:
                self._add_finding(
                    "WARN",
                    "AppArmor",
                    "Could not check AppArmor status (requires sudo)",
                    "info",
                )

        except Exception as e:
            self._add_finding(
                "WARN",
                "AppArmor",
                f"Could not verify AppArmor status: {e}",
                "info",
            )

    def _check_seccomp(self) -> None:
        """Audit Seccomp configuration."""
        print("\n[3/7] Seccomp System Call Filtering")
        print("─" * 65)

        # Check if seccomp is supported
        seccomp_path = Path("/proc/self/status")
        if not seccomp_path.exists():
            self._add_finding(
                "FAIL",
                "Seccomp",
                "Cannot access /proc/self/status",
                "critical",
            )
            return

        try:
            with open(seccomp_path) as f:
                for line in f:
                    if line.startswith("Seccomp:"):
                        seccomp_status = line.split(":")[1].strip()
                        if seccomp_status == "0":
                            self._add_finding(
                                "WARN",
                                "Seccomp",
                                "Seccomp is not enabled for this process",
                                "warning",
                                "Seccomp will be enabled for agent containers",
                            )
                        elif seccomp_status in ["1", "2"]:
                            self._add_finding(
                                "PASS",
                                "Seccomp",
                                f"Seccomp is supported (mode: {seccomp_status})",
                                "info",
                            )
                        break
            else:
                self._add_finding(
                    "WARN",
                    "Seccomp",
                    "Could not determine Seccomp status",
                    "warning",
                )

            # Check Docker's seccomp support
            result = subprocess.run(
                ["docker", "info", "--format", "{{.SecurityOptions}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "seccomp" in result.stdout.lower():
                self._add_finding(
                    "PASS",
                    "Seccomp",
                    "Docker has Seccomp support enabled",
                    "info",
                )
            else:
                self._add_finding(
                    "WARN",
                    "Seccomp",
                    "Docker seccomp support not detected",
                    "warning",
                )

        except Exception as e:
            self._add_finding(
                "WARN",
                "Seccomp",
                f"Could not verify Seccomp status: {e}",
                "info",
            )

    def _check_capabilities(self) -> None:
        """Audit Linux capabilities configuration."""
        print("\n[4/7] Linux Capabilities")
        print("─" * 65)

        # Check if we can query capabilities
        try:
            # Check current process capabilities
            result = subprocess.run(
                ["getpcaps", str(subprocess.os.getpid())],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                caps = result.stdout
                dangerous_caps = ["cap_sys_admin", "cap_sys_module", "cap_sys_boot"]
                has_dangerous = any(cap in caps.lower() for cap in dangerous_caps)

                if has_dangerous:
                    self._add_finding(
                        "WARN",
                        "Capabilities",
                        "Process has dangerous capabilities - containers will drop these",
                        "warning",
                    )
                else:
                    self._add_finding(
                        "PASS",
                        "Capabilities",
                        "Process capabilities look safe",
                        "info",
                    )
            else:
                self._add_finding(
                    "INFO",
                    "Capabilities",
                    "Could not check capabilities (getpcaps not available)",
                    "info",
                )

            # Verify capability management in code
            security_profiles_path = Path("src/agentcore/agent_runtime/services/security_profiles.py")
            if security_profiles_path.exists():
                with open(security_profiles_path) as f:
                    content = f.read()
                    if "CAP_DROP" in content or "CapDrop" in content:
                        self._add_finding(
                            "PASS",
                            "Capabilities",
                            "Capability management implemented in code",
                            "info",
                        )
                    else:
                        self._add_finding(
                            "FAIL",
                            "Capabilities",
                            "Capability management not found in code",
                            "critical",
                        )

        except FileNotFoundError:
            self._add_finding(
                "INFO",
                "Capabilities",
                "getpcaps tool not available - skipping runtime check",
                "info",
            )
        except Exception as e:
            self._add_finding(
                "WARN",
                "Capabilities",
                f"Could not verify capabilities: {e}",
                "warning",
            )

    def _check_file_permissions(self) -> None:
        """Audit file system permissions for security-sensitive paths."""
        print("\n[5/7] File System Permissions")
        print("─" * 65)

        sensitive_paths = [
            ("src/agentcore/agent_runtime/services", "0o755", "Service code directory"),
            ("src/agentcore/agent_runtime/models", "0o755", "Model definitions"),
        ]

        for path_str, expected_mode, description in sensitive_paths:
            path = Path(path_str)
            if not path.exists():
                self._add_finding(
                    "INFO",
                    "Permissions",
                    f"{description} not found: {path}",
                    "info",
                )
                continue

            try:
                mode = oct(path.stat().st_mode)[-3:]
                # Check if readable (6 or 7 in any position)
                if any(digit in ["6", "7"] for digit in mode):
                    self._add_finding(
                        "PASS",
                        "Permissions",
                        f"{description}: {mode} (readable)",
                        "info",
                    )
                else:
                    self._add_finding(
                        "WARN",
                        "Permissions",
                        f"{description}: {mode} (restricted)",
                        "warning",
                    )
            except Exception as e:
                self._add_finding(
                    "WARN",
                    "Permissions",
                    f"Could not check {description}: {e}",
                    "warning",
                )

    def _check_security_profiles_directory(self) -> None:
        """Check security profiles directory configuration."""
        print("\n[6/7] Security Profiles Directory")
        print("─" * 65)

        profiles_dir = Path("/tmp/agentcore-profiles")
        if profiles_dir.exists():
            # Check seccomp profiles
            seccomp_dir = profiles_dir / "seccomp"
            if seccomp_dir.exists():
                seccomp_count = len(list(seccomp_dir.glob("*.json")))
                self._add_finding(
                    "PASS",
                    "Profiles",
                    f"Seccomp profiles directory exists ({seccomp_count} profiles)",
                    "info",
                )
            else:
                self._add_finding(
                    "INFO",
                    "Profiles",
                    "Seccomp profiles directory will be created on first use",
                    "info",
                )

            # Check AppArmor profiles
            apparmor_dir = profiles_dir / "apparmor"
            if apparmor_dir.exists():
                apparmor_count = len(list(apparmor_dir.glob("*.profile")))
                self._add_finding(
                    "PASS",
                    "Profiles",
                    f"AppArmor profiles directory exists ({apparmor_count} profiles)",
                    "info",
                )
            else:
                self._add_finding(
                    "INFO",
                    "Profiles",
                    "AppArmor profiles directory will be created on first use",
                    "info",
                )
        else:
            self._add_finding(
                "INFO",
                "Profiles",
                "Security profiles directory will be created on first container creation",
                "info",
            )

    def _check_container_defaults(self) -> None:
        """Check default container security configuration."""
        print("\n[7/7] Container Security Defaults")
        print("─" * 65)

        # Check container manager code
        container_manager_path = Path("src/agentcore/agent_runtime/services/container_manager.py")
        if not container_manager_path.exists():
            self._add_finding(
                "FAIL",
                "Defaults",
                "Container manager code not found",
                "critical",
            )
            return

        try:
            with open(container_manager_path) as f:
                content = f.read()

                # Check for security features
                checks = [
                    ("CapDrop", "Capability dropping"),
                    ("ReadonlyRootfs", "Read-only root filesystem"),
                    ("SecurityOpt", "Security options"),
                    ("no-new-privileges", "No new privileges"),
                    ("PidsLimit", "Process limits"),
                ]

                for feature, description in checks:
                    if feature in content:
                        self._add_finding(
                            "PASS",
                            "Defaults",
                            f"{description} configured in code",
                            "info",
                        )
                    else:
                        self._add_finding(
                            "WARN",
                            "Defaults",
                            f"{description} not found in code",
                            "warning",
                        )

        except Exception as e:
            self._add_finding(
                "FAIL",
                "Defaults",
                f"Could not verify container defaults: {e}",
                "critical",
            )

    def _add_finding(
        self,
        status: str,
        category: str,
        message: str,
        severity: str,
        recommendation: str = "",
    ) -> None:
        """Add a security finding."""
        finding = {
            "status": status,
            "category": category,
            "message": message,
            "severity": severity,
            "recommendation": recommendation,
        }
        self.findings.append(finding)

        # Update counters
        if status == "PASS":
            self.passed_checks += 1
            icon = "✓"
            color = "\033[32m"  # Green
        elif status == "FAIL":
            self.failed_checks += 1
            icon = "✗"
            color = "\033[31m"  # Red
        elif status == "WARN":
            self.warning_checks += 1
            icon = "⚠"
            color = "\033[33m"  # Yellow
        else:  # INFO
            icon = "ℹ"
            color = "\033[36m"  # Cyan

        reset = "\033[0m"

        # Print finding
        print(f"{color}[{icon}] {category}: {message}{reset}")
        if recommendation:
            print(f"    💡 {recommendation}")

    def _print_summary(self) -> None:
        """Print audit summary."""
        print("\n" + "═" * 65)
        print("AUDIT SUMMARY")
        print("═" * 65)

        total = self.passed_checks + self.failed_checks + self.warning_checks
        print(f"Total checks: {total}")
        print(f"✓ Passed:     {self.passed_checks}")
        print(f"⚠ Warnings:   {self.warning_checks}")
        print(f"✗ Failed:     {self.failed_checks}")

        if self.failed_checks == 0:
            print("\n\033[32m✓ Security audit PASSED - No critical issues found\033[0m")
        else:
            print(f"\n\033[31m✗ Security audit FAILED - {self.failed_checks} critical issues found\033[0m")
            print("\nPlease address critical findings before deploying to production.")

        # Print recommendations
        critical_findings = [f for f in self.findings if f["severity"] == "critical" and f["recommendation"]]
        if critical_findings:
            print("\n" + "─" * 65)
            print("CRITICAL RECOMMENDATIONS:")
            for finding in critical_findings:
                print(f"\n• {finding['category']}: {finding['message']}")
                print(f"  → {finding['recommendation']}")


def main() -> int:
    """Run security audit."""
    auditor = SecurityAuditor()
    success = auditor.audit()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
