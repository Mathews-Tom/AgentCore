"""Comprehensive tests for security profile generation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agentcore.agent_runtime.models.agent_config import SecurityProfile
from agentcore.agent_runtime.services.security_profiles import (
    BLOCKED_SYSCALLS,
    MINIMAL_CAPABILITIES,
    MINIMAL_SYSCALLS,
    STANDARD_CAPABILITIES,
    SecurityProfileGenerator,
)


class TestSecurityProfileGenerator:
    """Test SecurityProfileGenerator service."""

    @pytest.fixture
    def temp_profiles_dir(self) -> Path:
        """Create temporary directory for profiles."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def generator(self, temp_profiles_dir: Path) -> SecurityProfileGenerator:
        """Create SecurityProfileGenerator instance."""
        return SecurityProfileGenerator(temp_profiles_dir)

    def test_initialization_creates_directories(self, temp_profiles_dir: Path) -> None:
        """Test that generator creates necessary directories."""
        generator = SecurityProfileGenerator(temp_profiles_dir)

        assert (temp_profiles_dir / "seccomp").exists()
        assert (temp_profiles_dir / "apparmor").exists()

    def test_generate_minimal_seccomp_profile(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test generation of minimal seccomp profile."""
        profile = SecurityProfile(
            profile_name="minimal",
            seccomp_profile="agentcore-minimal",
        )

        profile_path = generator.generate_seccomp_profile("test-minimal", profile)

        assert profile_path.exists()
        assert profile_path.suffix == ".json"

        # Verify profile content
        with open(profile_path) as f:
            data = json.load(f)

        assert data["defaultAction"] == "SCMP_ACT_ERRNO"
        assert "syscalls" in data
        assert len(data["syscalls"]) > 0

        # Verify allowed syscalls include minimal set
        allowed_syscalls = data["syscalls"][0]["names"]
        assert "read" in allowed_syscalls
        assert "write" in allowed_syscalls
        assert "open" in allowed_syscalls

        # Verify dangerous syscalls are NOT allowed
        for dangerous in ["mount", "umount", "kexec_load"]:
            assert dangerous not in allowed_syscalls

    def test_generate_standard_seccomp_profile(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test generation of standard seccomp profile."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="agentcore-standard",
        )

        profile_path = generator.generate_seccomp_profile("test-standard", profile)

        assert profile_path.exists()

        with open(profile_path) as f:
            data = json.load(f)

        # Standard should have more syscalls than minimal
        allowed_syscalls = set(data["syscalls"][0]["names"])
        assert len(allowed_syscalls) >= len(MINIMAL_SYSCALLS)

    def test_generate_unconfined_seccomp_profile(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test generation of unconfined seccomp profile."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="unconfined",
        )

        profile_path = generator.generate_seccomp_profile("test-unconfined", profile)

        assert profile_path.exists()

        with open(profile_path) as f:
            data = json.load(f)

        # Unconfined allows all syscalls
        assert data["defaultAction"] == "SCMP_ACT_ALLOW"

    def test_generate_runtime_default_seccomp(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test runtime/default seccomp profile reference."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="runtime/default",
        )

        profile_path = generator.generate_seccomp_profile("test-runtime", profile)

        # Should return special path indicating Docker's default
        assert profile_path == Path("runtime/default")

    def test_seccomp_custom_allowed_syscalls(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test seccomp profile with custom allowed syscalls."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="agentcore-standard",
            allowed_syscalls=["custom_syscall_1", "custom_syscall_2"],
        )

        profile_path = generator.generate_seccomp_profile("test-custom", profile)

        with open(profile_path) as f:
            data = json.load(f)

        allowed_syscalls = data["syscalls"][0]["names"]
        assert "custom_syscall_1" in allowed_syscalls
        assert "custom_syscall_2" in allowed_syscalls

    def test_seccomp_custom_blocked_syscalls(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test seccomp profile with custom blocked syscalls."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="agentcore-standard",
            blocked_syscalls=["read", "write"],  # Block essential syscalls
        )

        profile_path = generator.generate_seccomp_profile("test-blocked", profile)

        with open(profile_path) as f:
            data = json.load(f)

        allowed_syscalls = data["syscalls"][0]["names"]
        # Custom blocked syscalls should be removed
        assert "read" not in allowed_syscalls
        assert "write" not in allowed_syscalls

    def test_seccomp_arch_map_included(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test that seccomp profile includes architecture mappings."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="agentcore-standard",
        )

        profile_path = generator.generate_seccomp_profile("test-arch", profile)

        with open(profile_path) as f:
            data = json.load(f)

        # Verify architecture mappings exist
        assert "archMap" in data
        assert len(data["archMap"]) > 0

        # Check for common architectures
        architectures = [arch["architecture"] for arch in data["archMap"]]
        assert "SCMP_ARCH_X86_64" in architectures
        assert "SCMP_ARCH_AARCH64" in architectures

    def test_generate_minimal_apparmor_profile(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test generation of minimal AppArmor profile."""
        profile = SecurityProfile(
            profile_name="minimal",
            apparmor_profile="agentcore-minimal",
        )

        profile_name = generator.generate_apparmor_profile("test-minimal", profile)

        # Should return formatted profile name
        assert profile_name == "agentcore-test-minimal"

        # Verify profile file exists
        profile_path = temp_profiles_dir / "apparmor" / "test-minimal.profile"
        assert profile_path.exists()

        # Verify profile content
        with open(profile_path) as f:
            content = f.read()

        assert "profile agentcore-test-minimal" in content
        assert "deny capability sys_admin" in content
        assert "/tmp/** rw" in content
        assert "/workspace/** rw" in content

    def test_generate_standard_apparmor_profile(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test generation of standard AppArmor profile."""
        profile = SecurityProfile(
            profile_name="standard",
            apparmor_profile="agentcore-standard",
        )

        profile_name = generator.generate_apparmor_profile("test-standard", profile)

        profile_path = temp_profiles_dir / "apparmor" / "test-standard.profile"
        with open(profile_path) as f:
            content = f.read()

        # Standard should have more file access than minimal
        assert "/proc/** r" in content
        assert "/sys/** r" in content

    def test_apparmor_unconfined(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test unconfined AppArmor profile."""
        profile = SecurityProfile(
            profile_name="standard",
            apparmor_profile="unconfined",
        )

        profile_name = generator.generate_apparmor_profile("test-unconfined", profile)

        # Should return built-in profile name
        assert profile_name == "unconfined"

    def test_apparmor_docker_default(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test docker-default AppArmor profile."""
        profile = SecurityProfile(
            profile_name="standard",
            apparmor_profile="docker-default",
        )

        profile_name = generator.generate_apparmor_profile("test-docker-default", profile)

        # Should return built-in profile name
        assert profile_name == "docker-default"

    def test_apparmor_with_capabilities(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test AppArmor profile with custom capabilities."""
        profile = SecurityProfile(
            profile_name="standard",
            apparmor_profile="agentcore-standard",
            allowed_capabilities=["CAP_NET_BIND_SERVICE", "CAP_CHOWN"],
        )

        profile_name = generator.generate_apparmor_profile("test-caps", profile)

        profile_path = temp_profiles_dir / "apparmor" / "test-caps.profile"
        with open(profile_path) as f:
            content = f.read()

        # Capabilities should be added (without CAP_ prefix)
        assert "capability net_bind_service" in content
        assert "capability chown" in content

    def test_apparmor_dangerous_operations_denied(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test that AppArmor profile denies dangerous operations."""
        profile = SecurityProfile(
            profile_name="standard",
            apparmor_profile="agentcore-standard",
        )

        profile_name = generator.generate_apparmor_profile("test-deny", profile)

        profile_path = temp_profiles_dir / "apparmor" / "test-deny.profile"
        with open(profile_path) as f:
            content = f.read()

        # Verify dangerous operations are denied
        assert "deny mount" in content
        assert "deny umount" in content
        assert "deny pivot_root" in content
        assert "deny chroot" in content

    def test_get_capabilities_minimal_profile(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test capability extraction for minimal profile."""
        profile = SecurityProfile(
            profile_name="minimal",
            allowed_capabilities=[],
        )

        drop_caps, add_caps = generator.get_capabilities_for_profile(profile)

        assert drop_caps == ["ALL"]
        assert add_caps == []  # Minimal with no custom caps

    def test_get_capabilities_standard_profile(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test capability extraction for standard profile."""
        profile = SecurityProfile(
            profile_name="standard",
            allowed_capabilities=[],
        )

        drop_caps, add_caps = generator.get_capabilities_for_profile(profile)

        assert drop_caps == ["ALL"]
        assert set(add_caps) == set(MINIMAL_CAPABILITIES)

    def test_get_capabilities_privileged_profile(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test capability extraction for privileged profile."""
        profile = SecurityProfile(
            profile_name="privileged",
            allowed_capabilities=[],
        )

        drop_caps, add_caps = generator.get_capabilities_for_profile(profile)

        assert drop_caps == ["ALL"]
        # Privileged should have standard capabilities
        assert set(STANDARD_CAPABILITIES).issubset(set(add_caps))

    def test_get_capabilities_with_custom(
        self,
        generator: SecurityProfileGenerator,
    ) -> None:
        """Test capability extraction with custom capabilities."""
        profile = SecurityProfile(
            profile_name="standard",
            allowed_capabilities=["CAP_SYS_PTRACE", "CAP_SYS_ADMIN"],
        )

        drop_caps, add_caps = generator.get_capabilities_for_profile(profile)

        assert drop_caps == ["ALL"]
        # Should include both minimal and custom
        assert "CAP_SYS_PTRACE" in add_caps
        assert "CAP_SYS_ADMIN" in add_caps
        assert "CAP_NET_BIND_SERVICE" in add_caps  # From minimal

    def test_blocked_syscalls_comprehensive(self) -> None:
        """Test that blocked syscalls list is comprehensive."""
        # Verify critical dangerous syscalls are blocked
        critical_dangerous = [
            "mount", "umount", "umount2",
            "pivot_root", "chroot",
            "kexec_load", "kexec_file_load",
            "bpf", "perf_event_open",
            "reboot", "swapon", "swapoff",
            "userfaultfd",
        ]

        for syscall in critical_dangerous:
            assert syscall in BLOCKED_SYSCALLS, f"{syscall} should be in BLOCKED_SYSCALLS"

    def test_minimal_syscalls_essential(self) -> None:
        """Test that minimal syscalls include essential operations."""
        # Verify essential syscalls for Python execution
        essential_syscalls = [
            "read", "write", "open", "close",
            "brk", "mmap", "munmap",
            "execve", "exit", "exit_group",
            "fork", "clone",
            "socket", "connect", "accept",
        ]

        for syscall in essential_syscalls:
            assert syscall in MINIMAL_SYSCALLS, f"{syscall} should be in MINIMAL_SYSCALLS"

    def test_profile_generation_idempotent(
        self,
        generator: SecurityProfileGenerator,
        temp_profiles_dir: Path,
    ) -> None:
        """Test that generating same profile twice produces same result."""
        profile = SecurityProfile(
            profile_name="standard",
            seccomp_profile="agentcore-standard",
        )

        # Generate profile twice
        path1 = generator.generate_seccomp_profile("test-idempotent", profile)
        path2 = generator.generate_seccomp_profile("test-idempotent", profile)

        # Should overwrite with same content
        assert path1 == path2

        with open(path1) as f1, open(path2) as f2:
            assert f1.read() == f2.read()


class TestSecurityProfileConstants:
    """Test security profile constants."""

    def test_minimal_capabilities_secure(self) -> None:
        """Test that minimal capabilities don't include dangerous ones."""
        dangerous_caps = [
            "CAP_SYS_ADMIN",
            "CAP_SYS_MODULE",
            "CAP_SYS_BOOT",
            "CAP_SYS_RAWIO",
            "CAP_SYS_PTRACE",
        ]

        for cap in dangerous_caps:
            assert cap not in MINIMAL_CAPABILITIES

    def test_standard_capabilities_include_minimal(self) -> None:
        """Test that standard capabilities are superset of minimal."""
        for cap in MINIMAL_CAPABILITIES:
            assert cap in STANDARD_CAPABILITIES

    def test_blocked_syscalls_not_in_minimal(self) -> None:
        """Test that blocked syscalls are not in minimal allowed list."""
        for syscall in BLOCKED_SYSCALLS:
            assert syscall not in MINIMAL_SYSCALLS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
