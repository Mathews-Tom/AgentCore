# Container Security Guide

This guide covers the comprehensive security features implemented in AgentCore Runtime for container isolation and protection.

## Table of Contents

1. [Overview](#overview)
2. [Security Layers](#security-layers)
3. [Security Profiles](#security-profiles)
4. [Seccomp Filtering](#seccomp-filtering)
5. [AppArmor Mandatory Access Control](#apparmor-mandatory-access-control)
6. [Docker Capabilities](#docker-capabilities)
7. [Resource Limits](#resource-limits)
8. [Network Isolation](#network-isolation)
9. [Best Practices](#best-practices)
10. [Security Audit](#security-audit)

## Overview

AgentCore Runtime implements multiple layers of security for agent container isolation:

- **Seccomp**: System call filtering (200+ allowed, 20+ blocked)
- **AppArmor**: Mandatory access control with custom profiles
- **Capabilities**: Docker capability restrictions (drop ALL, add minimal)
- **Namespaces**: User namespace remapping for privilege isolation
- **Resource Limits**: CPU, memory, process, and file descriptor limits
- **Network Isolation**: Configurable network access restrictions

## Security Layers

AgentCore uses defense-in-depth with multiple security layers:

```
┌─────────────────────────────────────────────────┐
│  Application Layer                              │
│  ├─ RestrictedPython (safe code execution)      │
│  └─ Permission checking                         │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│  Container Layer                                │
│  ├─ Seccomp (syscall filtering)                 │
│  ├─ AppArmor (MAC)                             │
│  ├─ Capabilities (privilege restriction)        │
│  └─ Namespaces (isolation)                     │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│  Kernel Layer                                   │
│  ├─ cgroups (resource limits)                  │
│  └─ SELinux/AppArmor (kernel MAC)             │
└─────────────────────────────────────────────────┘
```

## Security Profiles

AgentCore provides three predefined security profiles:

### Minimal Profile

**Use Case**: Maximum security for untrusted code execution

**Features**:
- Only custom capabilities (if specified)
- Minimal syscall allowlist
- Strict AppArmor profile
- Read-only filesystem
- No network access by default
- User namespace isolation

**Configuration**:
```python
from agentcore.agent_runtime.models.agent_config import SecurityProfile

security_profile = SecurityProfile(
    profile_name="minimal",
    seccomp_profile="agentcore-minimal",
    apparmor_profile="agentcore-minimal",
    allowed_capabilities=[],  # No capabilities
    read_only_filesystem=True,
    no_new_privileges=True,
)
```

### Standard Profile (Default)

**Use Case**: Balanced security and functionality for most agents

**Features**:
- Standard capabilities (NET_BIND_SERVICE, CHOWN, SETGID, SETUID, DAC_OVERRIDE, FOWNER)
- Comprehensive syscall allowlist
- Standard AppArmor profile with file/network access
- Configurable filesystem access
- Optional network access
- User namespace isolation

**Configuration**:
```python
security_profile = SecurityProfile(
    profile_name="standard",
    seccomp_profile="agentcore-standard",
    apparmor_profile="agentcore-standard",
    read_only_filesystem=True,
    no_new_privileges=True,
)
```

### Privileged Profile

**Use Case**: Trusted agents requiring elevated permissions (use with caution)

**Features**:
- All standard capabilities + custom
- Extended syscall allowlist
- Relaxed AppArmor profile
- Read/write filesystem access
- Full network access
- Optional user namespace

**Configuration**:
```python
security_profile = SecurityProfile(
    profile_name="privileged",
    seccomp_profile="agentcore-standard",
    apparmor_profile="agentcore-standard",
    allowed_capabilities=["CAP_NET_ADMIN", "CAP_SYS_PTRACE"],
    read_only_filesystem=False,
    no_new_privileges=False,
)
```

## Seccomp Filtering

Seccomp (Secure Computing Mode) filters system calls at the kernel level.

### Allowed Syscalls

AgentCore allows 200+ syscalls essential for Python/agent execution:

**File Operations**: `read`, `write`, `open`, `openat`, `close`, `stat`, `fstat`, `lseek`, `mmap`, `munmap`

**Process Management**: `fork`, `clone`, `clone3`, `execve`, `exit`, `exit_group`, `wait4`, `kill`, `getpid`, `getppid`

**Network**: `socket`, `connect`, `bind`, `listen`, `accept`, `send`, `recv`, `sendto`, `recvfrom`, `shutdown`

**Time**: `clock_gettime`, `gettimeofday`, `nanosleep`, `clock_nanosleep`

**Signals**: `rt_sigaction`, `rt_sigprocmask`, `rt_sigreturn`, `sigaltstack`, `signalfd`

### Blocked Syscalls

AgentCore blocks dangerous syscalls that could enable container escape or system compromise:

**Container Escape**: `mount`, `umount`, `umount2`, `pivot_root`, `chroot`, `unshare`, `setns`

**Kernel Manipulation**: `create_module`, `delete_module`, `init_module`, `finit_module`, `kexec_load`

**System Control**: `reboot`, `sethostname`, `setdomainname`, `acct`

**Performance Events**: `perf_event_open` (can leak information)

**Advanced Features**: `bpf`, `userfaultfd`, `kcmp`

### Custom Syscalls

Add custom syscalls to security profile:

```python
security_profile = SecurityProfile(
    profile_name="standard",
    allowed_syscalls=["custom_syscall_1", "custom_syscall_2"],
    blocked_syscalls=["fork", "clone"],  # Block forking
)
```

### Seccomp Profile Location

Generated profiles are stored in `/tmp/agentcore-profiles/seccomp/{agent_id}-{profile_name}.json`

## AppArmor Mandatory Access Control

AppArmor provides mandatory access control (MAC) at the kernel level.

### Profile Structure

```apparmor
profile agentcore-{agent_id}-{profile_name} flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_boot,
  deny capability sys_module,

  # Allow specific capabilities
  capability net_bind_service,
  capability chown,

  # File access rules
  /tmp/** rw,
  /workspace/** rw,
  /app/** r,
  /usr/** r,

  # Network access
  network inet stream,
  network inet6 stream,

  # Deny dangerous operations
  deny /proc/sys/** w,
  deny /sys/** w,
  deny mount,
  deny umount,
  deny pivot_root,
  deny chroot,
}
```

### AppArmor Profiles

**agentcore-minimal**: Minimal file/network access for untrusted code
**agentcore-standard**: Standard access for most agents
**docker-default**: Docker's default profile
**unconfined**: No AppArmor restrictions (not recommended)

### Loading AppArmor Profiles

Profiles must be loaded before use:

```bash
# Load all AgentCore profiles
find /tmp/agentcore-profiles/apparmor -name "*.profile" | while read profile; do
    sudo apparmor_parser -r "$profile"
done

# Verify loaded
sudo apparmor_status | grep agentcore
```

## Docker Capabilities

Linux capabilities provide fine-grained privilege control.

### Capability Strategy

AgentCore uses **drop ALL, add minimal** approach:

1. Drop ALL capabilities first
2. Add back only required capabilities based on profile

### Capability Sets

**Minimal** (empty by default):
- Only custom capabilities specified by user

**Standard**:
- `CAP_NET_BIND_SERVICE`: Bind to ports < 1024
- `CAP_CHOWN`: Change file ownership
- `CAP_SETGID`: Set group ID
- `CAP_SETUID`: Set user ID
- `CAP_DAC_OVERRIDE`: Bypass file permission checks
- `CAP_FOWNER`: Bypass ownership checks

**Privileged**:
- All standard capabilities
- Plus custom capabilities

### Custom Capabilities

```python
security_profile = SecurityProfile(
    profile_name="standard",
    allowed_capabilities=[
        "CAP_NET_ADMIN",      # Network administration
        "CAP_SYS_PTRACE",     # Process tracing (debugging)
        "CAP_IPC_LOCK",       # Lock memory
    ],
)
```

### Dangerous Capabilities to Avoid

**Never add these unless absolutely necessary**:

- `CAP_SYS_ADMIN`: Almost root-equivalent (mount, quotas, etc.)
- `CAP_SYS_MODULE`: Load kernel modules
- `CAP_SYS_BOOT`: Reboot system
- `CAP_SYS_RAWIO`: Direct device access
- `CAP_DAC_READ_SEARCH`: Bypass all file read restrictions

## Resource Limits

Resource limits prevent denial-of-service and resource exhaustion.

### Container Limits

```python
from agentcore.agent_runtime.models.agent_config import ResourceLimits

resource_limits = ResourceLimits(
    max_memory_mb=512,                    # Memory limit
    max_cpu_cores=1.0,                    # CPU cores (fractional)
    max_execution_time_seconds=300,       # Timeout
    max_file_descriptors=100,             # Open files
    storage_quota_mb=1024,                # Disk space
)
```

### Sandbox Limits

Additional execution limits for code execution:

```python
from agentcore.agent_runtime.models.sandbox import ExecutionLimits

execution_limits = ExecutionLimits(
    max_execution_time_seconds=30,        # Per-execution timeout
    max_memory_mb=256,                    # Execution memory
    max_cpu_percent=50.0,                 # CPU percentage
    max_processes=10,                     # Process count
    max_file_descriptors=50,              # File descriptors
    max_network_requests=100,             # Network requests
)
```

### cgroups Integration

Container limits are enforced via Docker's cgroup integration:

- **Memory**: `MemoryLimit`, `MemorySwap` (no swap)
- **CPU**: `CpuQuota`, `CpuPeriod`
- **Processes**: `PidsLimit`
- **Storage**: `StorageOpt`

## Network Isolation

Network access can be completely disabled or restricted.

### Network Modes

**None** (Most Secure):
```python
sandbox_config = SandboxConfig(
    allow_network=False,  # No network access
)
```
Container network mode: `none`

**Restricted** (Default):
```python
resource_limits = ResourceLimits(
    network_access="restricted",  # Limited network
)
```
Container network mode: `bridge` with host filtering

**Full**:
```python
resource_limits = ResourceLimits(
    network_access="full",  # Full network access
)
```
Container network mode: `bridge`

### Host Allowlist

Restrict network access to specific hosts:

```python
sandbox_config = SandboxConfig(
    allow_network=True,
    allowed_hosts=[
        "api.example.com",
        "data.example.com",
        "10.0.0.0/8",  # Internal network
    ],
)
```

## Best Practices

### 1. Use Minimal Profile for Untrusted Code

```python
security_profile = SecurityProfile(
    profile_name="minimal",
    seccomp_profile="agentcore-minimal",
    apparmor_profile="agentcore-minimal",
    allowed_capabilities=[],
)
```

### 2. Enable All Security Layers

```python
security_profile = SecurityProfile(
    user_namespace=True,           # User namespace isolation
    read_only_filesystem=True,      # Immutable root filesystem
    no_new_privileges=True,         # Prevent privilege escalation
)
```

### 3. Restrict Network Access

```python
sandbox_config = SandboxConfig(
    allow_network=False,  # Disable network unless required
)
```

### 4. Set Aggressive Resource Limits

```python
execution_limits = ExecutionLimits(
    max_execution_time_seconds=30,   # Short timeout
    max_memory_mb=256,               # Limited memory
    max_processes=10,                # Few processes
    max_network_requests=0,          # No network
)
```

### 5. Regular Security Audits

Run security audit before production:

```bash
uv run python scripts/security_audit_runtime.py
```

### 6. Monitor and Log Security Events

Enable audit logging:

```python
from agentcore.agent_runtime.services.audit_logger import AuditLogger

audit_logger = AuditLogger(log_dir="/var/log/agentcore/audit")
```

### 7. Keep Profiles Updated

Regularly review and update security profiles based on:
- New CVEs
- Agent requirements
- Security best practices

### 8. Test with Real Workloads

Verify security doesn't break functionality:

```bash
# Test with minimal profile
uv run pytest tests/integration/test_sandbox_security.py

# Test with standard profile
uv run pytest tests/integration/test_agent_lifecycle.py
```

## Security Audit

### Running Security Audit

```bash
uv run python scripts/security_audit_runtime.py
```

### Audit Checks

1. **Docker Security**: Daemon configuration, user namespaces, live restore
2. **AppArmor**: Kernel module status, loaded profiles
3. **Seccomp**: Kernel support, Docker integration
4. **Capabilities**: Current process capabilities, code implementation
5. **File Permissions**: Security-sensitive paths
6. **Security Profiles**: Profile directory setup
7. **Container Defaults**: Security features in code

### Sample Output

```
╔═══════════════════════════════════════════════════════════════╗
║   AgentCore Runtime Security Audit                            ║
╚═══════════════════════════════════════════════════════════════╝

[1/7] Docker Security Configuration
─────────────────────────────────────────────────────────────────
[✓] Docker: Docker daemon is accessible and configured
[⚠] Docker: User namespace remapping not enabled

[2/7] AppArmor Security Module
─────────────────────────────────────────────────────────────────
[✓] AppArmor: AppArmor is enabled with 42 profiles loaded

[3/7] Seccomp System Call Filtering
─────────────────────────────────────────────────────────────────
[✓] Seccomp: Seccomp is supported (mode: 2)
[✓] Seccomp: Docker has Seccomp support enabled

[4/7] Linux Capabilities
─────────────────────────────────────────────────────────────────
[✓] Capabilities: Process capabilities look safe
[✓] Capabilities: Capability management implemented in code

[5/7] File System Permissions
─────────────────────────────────────────────────────────────────
[✓] Permissions: Service code directory: 755 (readable)
[✓] Permissions: Model definitions: 755 (readable)

[6/7] Security Profiles Directory
─────────────────────────────────────────────────────────────────
[✓] Profiles: Seccomp profiles directory exists (5 profiles)
[✓] Profiles: AppArmor profiles directory exists (3 profiles)

[7/7] Container Security Defaults
─────────────────────────────────────────────────────────────────
[✓] Defaults: Capability dropping configured in code
[✓] Defaults: Read-only root filesystem configured in code
[✓] Defaults: Security options configured in code
[✓] Defaults: No new privileges configured in code
[✓] Defaults: Process limits configured in code

═══════════════════════════════════════════════════════════════════
AUDIT SUMMARY
═══════════════════════════════════════════════════════════════════
Total checks: 17
✓ Passed:     15
⚠ Warnings:   2
✗ Failed:     0

✓ Security audit PASSED - No critical issues found
```

## Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Seccomp in Docker](https://docs.docker.com/engine/security/seccomp/)
- [AppArmor Documentation](https://gitlab.com/apparmor/apparmor/-/wikis/Documentation)
- [Linux Capabilities](https://man7.org/linux/man-pages/man7/capabilities.7.html)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Application Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)

## Support

For security questions or concerns:
- Review existing security documentation
- Run security audit script
- Check test coverage in `tests/agent_runtime/services/test_security_profiles.py`
- Consult with security team before weakening security profiles
