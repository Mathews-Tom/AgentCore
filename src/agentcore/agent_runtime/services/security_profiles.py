"""Security profile generation for containers (Seccomp, AppArmor, Capabilities)."""

import json
from pathlib import Path
from typing import Any

import structlog

from ..models.agent_config import SecurityProfile

logger = structlog.get_logger()


# Minimal set of capabilities required for basic agent operation
MINIMAL_CAPABILITIES = [
    "CAP_NET_BIND_SERVICE",  # Bind to ports < 1024 if needed for HTTP servers
    "CAP_CHOWN",  # Change file ownership in workspace
    "CAP_SETGID",  # Set group ID for file operations
    "CAP_SETUID",  # Set user ID for file operations
]

# Standard set includes minimal + additional common capabilities
STANDARD_CAPABILITIES = [
    *MINIMAL_CAPABILITIES,
    "CAP_DAC_OVERRIDE",  # Bypass file read/write/execute permission checks
    "CAP_FOWNER",  # Bypass permission checks on operations that require file ownership match
]

# Comprehensive syscall lists for different security profiles
# Based on Docker's default seccomp profile and CRI-O recommendations
# https://github.com/moby/moby/blob/master/profiles/seccomp/default.json

MINIMAL_SYSCALLS = [
    # Essential syscalls for basic Python execution
    "accept", "accept4", "access", "arch_prctl", "bind", "brk",
    "capget", "capset", "chdir", "chmod", "chown",
    "clock_getres", "clock_gettime", "clock_nanosleep", "clone", "clone3",
    "close", "close_range", "connect", "dup", "dup2", "dup3",
    "epoll_create", "epoll_create1", "epoll_ctl", "epoll_pwait", "epoll_wait",
    "eventfd", "eventfd2", "execve", "execveat", "exit", "exit_group",
    "faccessat", "faccessat2", "fadvise64", "fallocate", "fchdir", "fchmod",
    "fchmodat", "fchown", "fchownat", "fcntl", "fdatasync", "fgetxattr",
    "flistxattr", "flock", "fork", "fstat", "fstatfs", "fsync", "ftruncate",
    "futex", "futex_time64", "getcpu", "getcwd", "getdents", "getdents64",
    "getegid", "geteuid", "getgid", "getgroups", "getitimer", "getpeername",
    "getpgid", "getpgrp", "getpid", "getppid", "getpriority", "getrandom",
    "getresgid", "getresuid", "getrlimit", "getrusage", "getsid", "getsockname",
    "getsockopt", "gettid", "gettimeofday", "getuid", "getxattr",
    "io_uring_enter", "io_uring_register", "io_uring_setup", "ioctl",
    "kill", "lgetxattr", "link", "linkat", "listen", "listxattr", "llistxattr",
    "lseek", "lstat", "madvise", "memfd_create", "mkdir", "mkdirat",
    "mlock", "mlock2", "mlockall", "mmap", "mprotect", "mremap", "munlock",
    "munlockall", "munmap", "nanosleep", "newfstatat", "open", "openat",
    "openat2", "pause", "pipe", "pipe2", "poll", "ppoll", "ppoll_time64",
    "prctl", "pread64", "preadv", "preadv2", "prlimit64", "pselect6",
    "pselect6_time64", "pwrite64", "pwritev", "pwritev2",
    "read", "readahead", "readlink", "readlinkat", "readv", "recv",
    "recvfrom", "recvmmsg", "recvmmsg_time64", "recvmsg", "remap_file_pages",
    "removexattr", "rename", "renameat", "renameat2", "restart_syscall",
    "rmdir", "rt_sigaction", "rt_sigpending", "rt_sigprocmask", "rt_sigqueueinfo",
    "rt_sigreturn", "rt_sigsuspend", "rt_sigtimedwait", "rt_sigtimedwait_time64",
    "rt_tgsigqueueinfo", "sched_getaffinity", "sched_getattr", "sched_getparam",
    "sched_get_priority_max", "sched_get_priority_min", "sched_getscheduler",
    "sched_rr_get_interval", "sched_rr_get_interval_time64", "sched_setaffinity",
    "sched_setattr", "sched_setparam", "sched_setscheduler", "sched_yield",
    "seccomp", "select", "semctl", "semget", "semop", "semtimedop",
    "semtimedop_time64", "send", "sendfile", "sendfile64", "sendmmsg",
    "sendmsg", "sendto", "set_robust_list", "set_thread_area", "set_tid_address",
    "setfsgid", "setfsuid", "setgid", "setgroups", "setitimer", "setpgid",
    "setpriority", "setregid", "setresgid", "setresuid", "setreuid", "setrlimit",
    "setsid", "setsockopt", "setuid", "setxattr", "shutdown", "sigaltstack",
    "signalfd", "signalfd4", "socket", "socketcall", "socketpair", "splice",
    "stat", "statfs", "statx", "symlink", "symlinkat", "sync", "sync_file_range",
    "syncfs", "sysinfo", "tee", "tgkill", "time", "timer_create", "timer_delete",
    "timer_getoverrun", "timer_gettime", "timer_gettime64", "timer_settime",
    "timer_settime64", "timerfd_create", "timerfd_gettime", "timerfd_gettime64",
    "timerfd_settime", "timerfd_settime64", "times", "tkill", "truncate",
    "ugetrlimit", "umask", "uname", "unlink", "unlinkat", "utime", "utimensat",
    "utimensat_time64", "utimes", "vfork", "vmsplice", "wait4", "waitid",
    "waitpid", "write", "writev",
]

# Dangerous syscalls that should NEVER be allowed
BLOCKED_SYSCALLS = [
    # Container escape risks
    "mount", "umount", "umount2", "pivot_root", "chroot",
    # Kernel module manipulation
    "create_module", "delete_module", "init_module", "finit_module",
    # System configuration changes
    "acct", "add_key", "bpf", "clock_adjtime", "clock_settime",
    # Process tracing/debugging (can be used for container escape)
    "kcmp", "kexec_file_load", "kexec_load", "keyctl", "lookup_dcookie",
    # Performance events (can leak information)
    "perf_event_open",
    # Quota management
    "quotactl",
    # Reboot and system control
    "reboot", "sethostname", "setdomainname",
    # Swap operations
    "swapoff", "swapon",
    # User namespace manipulation (if not explicitly needed)
    "unshare", "setns",
    # USERFAULTFD (can be used for privilege escalation)
    "userfaultfd",
]


class SecurityProfileGenerator:
    """Generate Seccomp and AppArmor profiles for container security."""

    def __init__(self, profiles_dir: Path) -> None:
        """
        Initialize profile generator.

        Args:
            profiles_dir: Directory to store generated profiles
        """
        self._profiles_dir = profiles_dir
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

        # Create profile subdirectories
        (self._profiles_dir / "seccomp").mkdir(exist_ok=True)
        (self._profiles_dir / "apparmor").mkdir(exist_ok=True)

    def generate_seccomp_profile(
        self,
        profile_name: str,
        security_profile: SecurityProfile,
    ) -> Path:
        """
        Generate Seccomp profile JSON for container.

        Args:
            profile_name: Unique profile name
            security_profile: Security profile configuration

        Returns:
            Path to generated seccomp profile
        """
        # Determine syscall list based on profile type
        if security_profile.seccomp_profile == "unconfined":
            # No seccomp restrictions
            return self._create_unconfined_seccomp_profile(profile_name)
        elif security_profile.seccomp_profile == "runtime/default":
            # Use Docker's default runtime profile (no file needed)
            return Path("runtime/default")

        # Build allowed syscalls list
        allowed_syscalls = set(MINIMAL_SYSCALLS)

        # Merge with explicitly allowed syscalls
        if security_profile.allowed_syscalls:
            allowed_syscalls.update(security_profile.allowed_syscalls)

        # Remove explicitly blocked syscalls
        blocked_syscalls = set(BLOCKED_SYSCALLS)
        if security_profile.blocked_syscalls:
            blocked_syscalls.update(security_profile.blocked_syscalls)

        allowed_syscalls -= blocked_syscalls

        # Build seccomp profile JSON
        profile = {
            "defaultAction": "SCMP_ACT_ERRNO",  # Deny by default
            "defaultErrnoRet": 1,  # Return EPERM
            "archMap": [
                {"architecture": "SCMP_ARCH_X86_64", "subArchitectures": ["SCMP_ARCH_X86", "SCMP_ARCH_X32"]},
                {"architecture": "SCMP_ARCH_AARCH64", "subArchitectures": ["SCMP_ARCH_ARM"]},
                {"architecture": "SCMP_ARCH_MIPS64", "subArchitectures": ["SCMP_ARCH_MIPS", "SCMP_ARCH_MIPS64N32"]},
                {"architecture": "SCMP_ARCH_MIPS64N32", "subArchitectures": ["SCMP_ARCH_MIPS", "SCMP_ARCH_MIPS64"]},
                {"architecture": "SCMP_ARCH_MIPSEL64", "subArchitectures": ["SCMP_ARCH_MIPSEL", "SCMP_ARCH_MIPSEL64N32"]},
                {"architecture": "SCMP_ARCH_MIPSEL64N32", "subArchitectures": ["SCMP_ARCH_MIPSEL", "SCMP_ARCH_MIPSEL64"]},
                {"architecture": "SCMP_ARCH_S390X", "subArchitectures": ["SCMP_ARCH_S390"]},
            ],
            "syscalls": [
                {
                    "names": sorted(allowed_syscalls),
                    "action": "SCMP_ACT_ALLOW",
                }
            ],
        }

        # Write profile to file
        profile_path = self._profiles_dir / "seccomp" / f"{profile_name}.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info(
            "seccomp_profile_generated",
            profile_name=profile_name,
            allowed_syscalls_count=len(allowed_syscalls),
            path=str(profile_path),
        )

        return profile_path

    def generate_apparmor_profile(
        self,
        profile_name: str,
        security_profile: SecurityProfile,
    ) -> str:
        """
        Generate AppArmor profile for container.

        Args:
            profile_name: Unique profile name
            security_profile: Security profile configuration

        Returns:
            AppArmor profile name (for SecurityOpt)
        """
        if security_profile.apparmor_profile in ["unconfined", "docker-default"]:
            # Use built-in profiles
            return security_profile.apparmor_profile

        # Generate custom AppArmor profile content
        profile_content = self._build_apparmor_profile_content(
            profile_name,
            security_profile,
        )

        # Write profile to file
        profile_path = self._profiles_dir / "apparmor" / f"{profile_name}.profile"
        with open(profile_path, "w") as f:
            f.write(profile_content)

        logger.info(
            "apparmor_profile_generated",
            profile_name=profile_name,
            path=str(profile_path),
        )

        # Return profile name for Docker SecurityOpt
        # Note: AppArmor profiles need to be loaded with: apparmor_parser -r profile_path
        return f"agentcore-{profile_name}"

    def _create_unconfined_seccomp_profile(self, profile_name: str) -> Path:
        """Create unconfined seccomp profile (allows all syscalls)."""
        profile = {
            "defaultAction": "SCMP_ACT_ALLOW",
        }

        profile_path = self._profiles_dir / "seccomp" / f"{profile_name}_unconfined.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        return profile_path

    def _build_apparmor_profile_content(
        self,
        profile_name: str,
        security_profile: SecurityProfile,
    ) -> str:
        """Build AppArmor profile content based on security profile."""
        # Start with base profile
        profile_lines = [
            f"#include <tunables/global>",
            f"",
            f"profile agentcore-{profile_name} flags=(attach_disconnected,mediate_deleted) {{",
            f"  #include <abstractions/base>",
            f"",
            f"  # Deny potentially dangerous capabilities",
            f"  deny capability sys_admin,",
            f"  deny capability sys_boot,",
            f"  deny capability sys_module,",
            f"  deny capability sys_rawio,",
            f"",
        ]

        # Add capability rules based on allowed capabilities
        if security_profile.allowed_capabilities:
            profile_lines.append("  # Allowed capabilities")
            for cap in security_profile.allowed_capabilities:
                # Remove CAP_ prefix for AppArmor
                cap_name = cap.replace("CAP_", "").lower()
                profile_lines.append(f"  capability {cap_name},")
            profile_lines.append("")

        # Add file access rules
        if security_profile.profile_name == "minimal":
            profile_lines.extend([
                "  # Minimal file access",
                "  /tmp/** rw,",
                "  /workspace/** rw,",
                "  /app/** r,",
                "  /usr/** r,",
                "  /lib/** r,",
                "  /etc/** r,",
                "",
            ])
        else:
            profile_lines.extend([
                "  # Standard file access",
                "  /tmp/** rw,",
                "  /workspace/** rw,",
                "  /app/** r,",
                "  /usr/** r,",
                "  /lib/** r,",
                "  /lib64/** r,",
                "  /etc/** r,",
                "  /proc/** r,",
                "  /sys/** r,",
                "",
            ])

        # Network access
        profile_lines.extend([
            "  # Network access",
            "  network inet stream,",
            "  network inet6 stream,",
            "  network inet dgram,",
            "  network inet6 dgram,",
            "",
        ])

        # Deny dangerous operations
        profile_lines.extend([
            "  # Deny dangerous operations",
            "  deny /proc/sys/** w,",
            "  deny /sys/** w,",
            "  deny /dev/** w,",
            "  deny mount,",
            "  deny remount,",
            "  deny umount,",
            "  deny pivot_root,",
            "  deny chroot,",
            "",
        ])

        # Close profile
        profile_lines.append("}")

        return "\n".join(profile_lines)

    def get_capabilities_for_profile(
        self,
        security_profile: SecurityProfile,
    ) -> tuple[list[str], list[str]]:
        """
        Get Docker capabilities to drop and add based on security profile.

        Args:
            security_profile: Security profile configuration

        Returns:
            Tuple of (capabilities_to_drop, capabilities_to_add)
        """
        # Always drop ALL first for security
        drop_caps = ["ALL"]

        # Determine which capabilities to add back
        if security_profile.profile_name == "privileged":
            # Privileged mode: add back all standard + custom
            add_caps = list(set(STANDARD_CAPABILITIES) | set(security_profile.allowed_capabilities))
        elif security_profile.profile_name == "standard":
            # Standard mode: minimal + custom
            add_caps = list(set(MINIMAL_CAPABILITIES) | set(security_profile.allowed_capabilities))
        else:  # minimal
            # Minimal mode: only custom or nothing
            add_caps = security_profile.allowed_capabilities if security_profile.allowed_capabilities else []

        logger.debug(
            "capabilities_determined",
            profile=security_profile.profile_name,
            add_count=len(add_caps),
            add_caps=add_caps,
        )

        return drop_caps, add_caps
